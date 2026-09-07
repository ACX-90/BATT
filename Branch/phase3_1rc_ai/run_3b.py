"""Phase-3 3B1–3B4: PyBaMM-equivalent 1RC LUT + δU GRU (approach Δ).

Does NOT read Data/ai_mlp. LUT teacher lives under Branch/phase3_1rc_ai/out/.

Usage (repo root, venv with torch/scipy/pybamm if regenerating grid):

    python Branch/phase3_1rc_ai/run_3b.py
    python Branch/phase3_1rc_ai/run_3b.py --skip-train-mlp   # reuse out/mlp_pybamm
    python Branch/phase3_1rc_ai/run_3b.py --epochs-mlp 8 --epochs-gru 25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "Src" / "AI"))
sys.path.insert(0, str(HERE))

from MLP.config import TrainConfig  # noqa: E402
from MLP.dataset import (  # noqa: E402
    FeatureScaler,
    TrajectoryDataset,
    collate_traj,
    fit_scaler,
    load_grid_sequences,
    split_sequences,
)
from MLP.ecm import ecm_forward  # noqa: E402
# voltage_loss already respects cfg.tbptt
from MLP.infer import load_bundle  # noqa: E402
from MLP.model import ParamMLP  # noqa: E402
from MLP.train import set_seed, voltage_loss  # noqa: E402

from rc_sim import (  # noqa: E402
    C1_STAR,
    DT_S,
    edge_r0,
    fit_ltis_1rc_c1star,
    fit_ltis_2rc,
    load_traj,
    nlinear,
    rmse_mv,
    segment_masks,
    segment_rmse,
    sim_1rc,
    sim_2rc,
)
from residual_gru import CausalDeltaUGRU, build_features, huber  # noqa: E402

OUT = HERE / "out"
MLP_DIR = OUT / "mlp_pybamm"
LUT_DIR = OUT / "lut_pybamm"
GRID = REPO / "Data" / "grid_pybamm"
AI_MLP = REPO / "Data" / "ai_mlp"


def assert_no_ai_mlp(paths: list[Path]) -> None:
    forbidden = AI_MLP.resolve()
    for p in paths:
        rp = p.resolve()
        if rp == forbidden or forbidden in rp.parents or str(rp).startswith(str(forbidden)):
            raise RuntimeError(f"Phase-3 PyBaMM track must not touch Data/ai_mlp: {p}")


def list_grid_files() -> list[Path]:
    files = sorted(GRID.glob("nmc100ah_pybamm_s*_t*.csv"))
    if not files:
        raise FileNotFoundError(f"Missing {GRID}; run nmc100ah_gen_grid.py --pybamm first")
    return files


def train_mlp_pybamm(*, epochs: int, pretrain: int, device: torch.device, seed: int) -> Path:
    """Train scheme-B MLP on PyBaMM grid → Branch/phase3_1rc_ai/out/mlp_pybamm."""
    assert_no_ai_mlp([GRID, MLP_DIR])
    MLP_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    cfg = TrainConfig(
        scheme="B",
        data_dir="Data/grid_pybamm",
        out_dir=str(MLP_DIR.relative_to(REPO)),
        use_true_inputs=True,
        voltage_target="true",
        epochs=int(epochs),
        pretrain_epochs=int(pretrain),
        batch_size=4,
        lr=2.0e-3,
        device=str(device),
        seed=seed,
        tbptt=100,
        hidden=(64, 64),
        c1_star=C1_STAR,
    )
    # Prove data path
    assert cfg.data_path().resolve() == GRID.resolve()
    assert "ai_mlp" not in str(cfg.data_path())
    assert "ai_mlp" not in str(cfg.output_path())

    seqs = load_grid_sequences(cfg)
    train_s, val_s = split_sequences(seqs, cfg.val_ratio, cfg.seed)
    scaler = fit_scaler(train_s)
    train_loader = DataLoader(
        TrajectoryDataset(train_s, scaler),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_traj,
    )
    val_loader = (
        DataLoader(
            TrajectoryDataset(val_s, scaler),
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_traj,
        )
        if val_s
        else None
    )
    model = ParamMLP(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float("inf")
    hist = []

    # optional teacher pretrain on CSV ECM columns (continuity only; voltage is main)
    if pretrain > 0:
        model.train()
        for ep in range(1, pretrain + 1):
            tot = 0.0
            n = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                r0, r1, _ = model(batch["x"])
                loss = 0.5 * (
                    (torch.log(r0.clamp_min(1e-12)) - torch.log(batch["r0"].clamp_min(1e-12))).pow(2).mean()
                    + (torch.log(r1.clamp_min(1e-12)) - torch.log(batch["r1"].clamp_min(1e-12))).pow(2).mean()
                )
                loss.backward()
                opt.step()
                tot += float(loss.detach())
                n += 1
            print(f"  mlp pre [{ep:02d}/{pretrain}] logR {tot/max(n,1):.5f}", flush=True)

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        n = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss, stats = voltage_loss(model, batch, cfg)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += stats["rmse_v"]
            n += 1
        train_rmse = tot / max(n, 1)
        val_rmse = train_rmse
        if val_loader is not None:
            model.eval()
            vt = 0.0
            vn = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    _, stats = voltage_loss(model, batch, cfg)
                    vt += stats["rmse_v"]
                    vn += 1
            val_rmse = vt / max(vn, 1)
        hist.append({"epoch": ep, "train_rmse_v": train_rmse, "val_rmse_v": val_rmse})
        print(f"  mlp [{ep:02d}/{epochs}] train {train_rmse*1e3:.2f} mV  val {val_rmse*1e3:.2f} mV", flush=True)
        payload = {
            "model": model.state_dict(),
            "scheme": "B",
            "best_rmse": val_rmse,
            "epoch_done": ep,
            "source": "grid_pybamm",
            "not_ai_mlp": True,
        }
        torch.save(payload, MLP_DIR / "last.pt")
        if val_rmse <= best:
            best = val_rmse
            torch.save(payload, MLP_DIR / "best.pt")
    scaler.save(MLP_DIR / "scaler.json")
    cfg.to_json(MLP_DIR / "config.json")
    (MLP_DIR / "history.json").write_text(json.dumps(hist, indent=2) + "\n", encoding="utf-8")
    assert_no_ai_mlp([MLP_DIR / "best.pt", MLP_DIR / "scaler.json"])
    print(f"  mlp best val {best*1e3:.2f} mV → {MLP_DIR}", flush=True)
    return MLP_DIR


@torch.no_grad()
def bake_lut(mlp_dir: Path, *, n_i: int = 9, n_s: int = 9, n_t: int = 7) -> dict:
    assert_no_ai_mlp([mlp_dir])
    LUT_DIR.mkdir(parents=True, exist_ok=True)
    model, scaler, cfg = load_bundle(mlp_dir / "best.pt", mlp_dir / "config.json", mlp_dir / "scaler.json")
    model.eval()
    i_ax = np.linspace(-200.0, 200.0, n_i)
    s_ax = np.linspace(0.05, 0.95, n_s)
    t_ax = np.linspace(-10.0, 50.0, n_t)
    gi, gs, gt = np.meshgrid(i_ax, s_ax, t_ax, indexing="ij")
    feat = np.stack([gi.ravel(), gs.ravel(), gt.ravel()], axis=-1)
    xn = torch.from_numpy(scaler.transform(feat).astype(np.float32))
    r0, r1, _ = model(xn)
    table = np.stack([r0.cpu().numpy(), r1.cpu().numpy()], axis=-1).reshape(n_i, n_s, n_t, 2)
    meta = {
        "kind": "3d",
        "log_r": True,
        "c1_star": C1_STAR,
        "scheme": "B",
        "teacher_dir": str(mlp_dir.relative_to(REPO)),
        "source_grid": "Data/grid_pybamm",
        "not_ai_mlp": True,
        "axes": {"i": i_ax.tolist(), "s": s_ax.tolist(), "t": t_ax.tolist()},
        "n_i": n_i,
        "n_s": n_s,
        "n_t": n_t,
        "rom_bytes": int(n_i * n_s * n_t * 2 * 4),
    }
    np.savez_compressed(LUT_DIR / "lut_3d.npz", table=table, i_ax=i_ax, s_ax=s_ax, t_ax=t_ax)
    (LUT_DIR / "lut_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    assert_no_ai_mlp([LUT_DIR / "lut_3d.npz"])
    print(f"  baked LUT 3d {n_i}x{n_s}x{n_t}  ROM={meta['rom_bytes']/1024:.1f} KB → {LUT_DIR}", flush=True)
    return {"axes": [i_ax, s_ax, t_ax], "table": table, "meta": meta}


def load_lut() -> dict:
    z = np.load(LUT_DIR / "lut_3d.npz")
    meta = json.loads((LUT_DIR / "lut_meta.json").read_text(encoding="utf-8"))
    assert meta.get("not_ai_mlp") is True
    assert "ai_mlp" not in meta.get("teacher_dir", "")
    return {"axes": [z["i_ax"], z["s_ax"], z["t_ax"]], "table": z["table"], "meta": meta}


def query_r(lut: dict, i: np.ndarray, soc: np.ndarray, t_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.stack([i.ravel(), soc.ravel(), t_c.ravel()], axis=-1)
    rr = nlinear(lut["axes"], lut["table"], q, log_r=True)
    return rr[:, 0].reshape(i.shape), rr[:, 1].reshape(i.shape)


def openloop_1rc_lut(traj: dict, lut: dict) -> np.ndarray:
    r0, r1 = query_r(lut, traj["i"], traj["soc"], traj["t_c"])
    return sim_1rc(traj["i"], traj["ocv"], r0, r1, C1_STAR)


def identify_physical(traj: dict) -> dict:
    """Per-file 1RC (C1*) and 2RC LTI projections for col2 / teachers."""
    i, ocv, u = traj["i"], traj["ocv"], traj["u"]
    r0e = edge_r0(i, u)
    # seed R1 from CSV teacher mean
    r1_h = float(np.mean(traj["r1_csv"]))
    f1 = fit_ltis_1rc_c1star(i, ocv, u, r0_hint=r0e, r1_hint=r1_h)
    # rest double-exp seeds for 2RC
    rest = traj["cmd"] == 2
    tau1_h, tau2_h, r2_h = 8.0, 120.0, 3e-4
    if rest.sum() > 50:
        from scipy.optimize import least_squares

        t0 = traj["t"][rest][0]
        tr = traj["t"][rest] - t0
        eta = traj["ocv"][rest] - traj["u"][rest]

        def fun(z):
            t1 = float(np.clip(np.exp(z[0]), 0.4, 40))
            t2 = float(np.clip(max(np.exp(z[1]), 4 * t1), 12, 400))
            phi = np.column_stack([np.exp(-tr / t1), np.exp(-tr / t2)])
            coef, *_ = np.linalg.lstsq(phi, eta, rcond=None)
            return phi @ coef - eta

        try:
            res = least_squares(fun, np.array([np.log(8.0), np.log(120.0)]), method="trf", max_nfev=80)
            tau1_h = float(np.clip(np.exp(res.x[0]), 0.4, 40))
            tau2_h = float(np.clip(max(np.exp(res.x[1]), 4 * tau1_h), 12, 400))
            phi = np.column_stack([np.exp(-tr / tau1_h), np.exp(-tr / tau2_h)])
            coef, *_ = np.linalg.lstsq(phi, eta, rcond=None)
            # rough R from amplitude / I_pre
            pulse = traj["cmd"] == 1
            i_pre = float(np.mean(traj["i"][pulse])) if pulse.any() else 100.0
            tp = float(pulse.sum() * DT_S)
            for amp, tau, key in ((coef[0], tau1_h, "r1"), (coef[1], tau2_h, "r2")):
                gain = 1.0 - np.exp(-tp / tau)
                if abs(i_pre) > 1 and gain > 0.05:
                    if key == "r2":
                        r2_h = float(amp / (i_pre * gain))
        except Exception:  # noqa: BLE001
            pass

    f2 = fit_ltis_2rc(
        i,
        ocv,
        u,
        r0_hint=r0e,
        r1_hint=f1["r1_ohm"],
        tau1_hint=tau1_h,
        r2_hint=r2_h,
        tau2_hint=tau2_h,
    )
    return {"f1": f1, "f2": f2, "r0_edge": r0e}


def mean_seg(rows: list[dict], key: str) -> dict[str, float]:
    keys = rows[0][key].keys()
    return {k: float(np.nanmean([r[key][k] for r in rows])) for k in keys}


def train_gru(
    cases: list[dict],
    *,
    teacher: str,
    clip_mv: float,
    epochs: int,
    device: torch.device,
    seed: int,
    tag: str,
) -> dict:
    """Train δU GRU. teacher: 'proper' | 'diagnostic'."""
    set_seed(seed)
    model = CausalDeltaUGRU(d=4, n_in=4, clip_v=clip_mv * 1e-3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    # pack tensors
    packs = []
    for c in cases:
        i = torch.from_numpy(c["i"].astype(np.float32))
        u1 = torch.from_numpy(c["u1"].astype(np.float32))
        u2 = torch.from_numpy(c["u2"].astype(np.float32))
        upy = torch.from_numpy(c["u"].astype(np.float32))
        e_ol = u1 - upy
        if teacher == "proper":
            tgt = torch.clamp(upy - u2, -clip_mv * 1e-3, clip_mv * 1e-3)
        elif teacher == "diagnostic":
            tgt = torch.clamp(upy - u1, -clip_mv * 1e-3, clip_mv * 1e-3)
        else:
            raise ValueError(teacher)
        packs.append({"i": i, "e_ol": e_ol, "tgt": tgt, "u1": u1, "upy": upy, "name": c["name"]})

    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for p in packs:
            opt.zero_grad(set_to_none=True)
            x = build_features(p["i"].unsqueeze(0).to(device), p["e_ol"].unsqueeze(0).to(device))
            du, _ = model(x)
            loss = huber(du.squeeze(0), p["tgt"].to(device))
            # also mild voltage consistency on 1RC+δU vs 1RC+tgt (same as matching tgt)
            loss = loss + 0.25 * (du.squeeze(0) - p["tgt"].to(device)).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
        hist.append({"epoch": ep, "loss": tot / max(len(packs), 1)})
        if ep == 1 or ep == epochs or ep % 5 == 0:
            print(f"  gru/{tag} [{ep:02d}/{epochs}] loss={hist[-1]['loss']:.6f}  n={model.n_params()}", flush=True)

    ckpt = OUT / f"gru_{tag}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "teacher": teacher,
            "clip_mv": clip_mv,
            "d": 4,
            "n_params": model.n_params(),
            "history": hist,
            "not_ai_mlp": True,
        },
        ckpt,
    )
    model.eval()
    return {"model": model, "ckpt": str(ckpt.relative_to(REPO)), "n_params": model.n_params(), "history": hist}


@torch.no_grad()
def eval_gru(model: CausalDeltaUGRU, cases: list[dict], device: torch.device) -> list[dict]:
    rows = []
    model.eval()
    for c in cases:
        i = torch.from_numpy(c["i"].astype(np.float32)).unsqueeze(0).to(device)
        e_ol = torch.from_numpy((c["u1"] - c["u"]).astype(np.float32)).unsqueeze(0).to(device)
        x = build_features(i, e_ol)
        du, _ = model(x)
        du_np = du.squeeze(0).cpu().numpy()
        u4 = c["u1"] + du_np
        masks = segment_masks(c["cmd"], c["i"])
        rows.append(
            {
                "name": c["name"],
                "soc0": float(c["soc"][0]),
                "t_c": float(np.mean(c["t_c"])),
                "rmse_col4": segment_rmse(u4, c["u"], masks),
                "du_rms_mv": float(np.sqrt(np.mean(du_np**2)) * 1e3),
                "du_max_mv": float(np.max(np.abs(du_np)) * 1e3),
                "rest_resid": (c["u"] - u4)[masks["rest_after_1c"]] if masks["rest_after_1c"].any() else np.array([]),
                "rest_t": (c["t"][masks["rest_after_1c"]] - c["t"][masks["rest_after_1c"]][0])
                if masks["rest_after_1c"].any()
                else np.array([]),
            }
        )
    return rows


def rest_shape_score(resid_v: np.ndarray, t_s: np.ndarray) -> dict:
    """Heuristic: does rest residual look like a 2nd exponential (slow tail)?"""
    if resid_v.size < 50:
        return {"ok": False}
    eta = np.asarray(resid_v, dtype=float)
    # work on signed overpotential-like residual; use magnitude for shape
    # prefer late window where fast pole has died
    m = t_s >= 20.0
    if np.sum(m) < 40:
        tt, yy = t_s.copy(), np.abs(eta)
    else:
        tt, yy = t_s[m], np.abs(eta[m])
    yy = np.clip(yy, 1e-6, None)
    # 1-exp
    best = None
    for tau in np.logspace(np.log10(20), np.log10(300), 24):
        phi = np.column_stack([np.exp(-tt / tau), np.ones_like(tt)])
        coef, *_ = np.linalg.lstsq(phi, yy, rcond=None)
        yhat = phi @ coef
        sse = float(np.dot(yy - yhat, yy - yhat))
        if best is None or sse < best[0]:
            ss_tot = float(np.dot(yy - yy.mean(), yy - yy.mean())) + 1e-18
            r2 = 1.0 - sse / ss_tot
            best = (sse, float(tau), float(r2), float(coef[0]))
    # 2-exp vs 1-exp gain
    best2 = None
    for t1 in np.logspace(np.log10(2), np.log10(25), 10):
        for t2 in np.logspace(np.log10(40), np.log10(280), 12):
            if t2 < 4 * t1:
                continue
            phi = np.column_stack([np.exp(-tt / t1), np.exp(-tt / t2)])
            coef, *_ = np.linalg.lstsq(phi, yy, rcond=None)
            yhat = phi @ coef
            sse = float(np.dot(yy - yhat, yy - yhat))
            if best2 is None or sse < best2[0]:
                ss_tot = float(np.dot(yy - yy.mean(), yy - yy.mean())) + 1e-18
                r2 = 1.0 - sse / ss_tot
                best2 = (sse, float(t1), float(t2), float(r2))
    looks = bool(best is not None and best[2] > 0.85 and 40 < best[1] < 350)
    if best2 is not None and best is not None and best2[0] < 0.5 * best[0] and best2[3] > 0.9:
        looks = True
    return {
        "ok": True,
        "tau1_s": None if best2 is None else best2[1],
        "tau2_s": None if best is None else best[1],
        "tau2_s_1exp": None if best is None else best[1],
        "r2_1exp": None if best is None else best[2],
        "r2_2exp": None if best2 is None else best2[3],
        "looks_like_exp": looks,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-3 3B1–3B4 runner")
    p.add_argument("--epochs-mlp", type=int, default=12)
    p.add_argument("--pretrain-mlp", type=int, default=3)
    p.add_argument("--epochs-gru", type=int, default=30)
    p.add_argument("--clip-mv", type=float, default=8.0, help="δU clip in mV (try 8 first)")
    p.add_argument("--skip-train-mlp", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-files", type=int, default=0, help="0=all grid files")
    args = p.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    files = list_grid_files()
    if args.max_files > 0:
        files = files[: args.max_files]
    print(f"Phase-3 3B  grid={GRID}  n={len(files)}  clip={args.clip_mv} mV", flush=True)
    print(f"  assert not using {AI_MLP} (exists={AI_MLP.exists()})", flush=True)

    # --- train / load MLP + LUT (3B1 floor) ---
    if args.skip_train_mlp and (MLP_DIR / "best.pt").exists():
        print("  reuse mlp; rebake LUT", flush=True)
        print("=== bake 1RC LUT ===", flush=True)
        lut = bake_lut(MLP_DIR)
    else:
        print("=== train PyBaMM-equivalent MLP (scheme B) ===", flush=True)
        train_mlp_pybamm(epochs=args.epochs_mlp, pretrain=args.pretrain_mlp, device=device, seed=args.seed)
        print("=== bake 1RC LUT ===", flush=True)
        lut = bake_lut(MLP_DIR)

    # --- per-file identify + open-loop columns ---
    print("=== identify 1RC/2RC + open-loop eval ===", flush=True)
    cases = []
    rows_b1 = []
    for fp in files:
        traj = load_traj(fp)
        u1 = openloop_1rc_lut(traj, lut)
        phys = identify_physical(traj)
        u2 = phys["f2"]["uh"]
        masks = segment_masks(traj["cmd"], traj["i"])
        row = {
            "name": traj["name"],
            "soc0": float(traj["soc"][0]),
            "t_c": float(np.mean(traj["t_c"])),
            "rmse_col1": segment_rmse(u1, traj["u"], masks),
            "rmse_col2": segment_rmse(u2, traj["u"], masks),
            "ltis1_const_mv": phys["f1"]["rmse_mv"],
            "ltis2_mv": phys["f2"]["rmse_mv"],
            "r0_edge_mohm": float(phys["r0_edge"] * 1e3) if np.isfinite(phys["r0_edge"]) else None,
            "f2": {k: phys["f2"][k] for k in ("r0_ohm", "r1_ohm", "r2_ohm", "tau1_s", "tau2_s", "rmse_mv")},
        }
        rows_b1.append(row)
        cases.append(
            {
                **traj,
                "u1": u1,
                "u2": u2,
                "phys": phys,
            }
        )
        print(
            f"  {traj['name']}: col1={row['rmse_col1']['all']:.2f}  "
            f"col2={row['rmse_col2']['all']:.2f}  "
            f"rest60={row['rmse_col1']['rest_60_120s']:.2f}/{row['rmse_col2']['rest_60_120s']:.2f} mV",
            flush=True,
        )

    summary_b1 = {
        "n_files": len(files),
        "files": [f.name for f in files],
        "lut_meta": lut["meta"],
        "mlp_dir": str(MLP_DIR.relative_to(REPO)),
        "used_ai_mlp": False,
        "mean_rmse_col1_mv": mean_seg(rows_b1, "rmse_col1"),
        "mean_rmse_col2_mv": mean_seg(rows_b1, "rmse_col2"),
        "per_file": [{k: v for k, v in r.items() if k != "f2"} | {"f2": r["f2"]} for r in rows_b1],
    }
    (OUT / "3b1_floor.json").write_text(json.dumps(summary_b1, indent=2) + "\n", encoding="utf-8")

    # train/val split by file for GRU (hold out ~20%)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(cases))
    rng.shuffle(idx)
    n_val = max(1, int(round(0.2 * len(cases)))) if len(cases) > 3 else 0
    val_i = set(idx[:n_val].tolist()) if n_val else set()
    train_cases = [cases[i] for i in range(len(cases)) if i not in val_i]
    val_cases = [cases[i] for i in range(len(cases)) if i in val_i] or train_cases
    # also evaluate on all for reporting
    eval_cases = cases

    print("=== 3B2 train δU on proper teacher (U_py - U_2RC) ===", flush=True)
    proper = train_gru(
        train_cases,
        teacher="proper",
        clip_mv=args.clip_mv,
        epochs=args.epochs_gru,
        device=device,
        seed=args.seed,
        tag="proper",
    )
    rows_p = eval_gru(proper["model"], eval_cases, device)

    print("=== 3B4 train δU on diagnostic teacher (U_py - U_1RC) ===", flush=True)
    diag = train_gru(
        train_cases,
        teacher="diagnostic",
        clip_mv=args.clip_mv,
        epochs=args.epochs_gru,
        device=device,
        seed=args.seed + 1,
        tag="diagnostic",
    )
    rows_d = eval_gru(diag["model"], eval_cases, device)

    # shape analysis on rest for mid-SOC ~20C file if present
    def pick_mid(rows_src, cases_src):
        best = None
        best_score = 1e9
        for r, c in zip(rows_src, cases_src):
            score = abs(r["t_c"] - 20.0) + abs(r["soc0"] - 0.5) * 20
            if score < best_score:
                best_score = score
                best = (r, c)
        return best

    mid_p = pick_mid(rows_p, eval_cases)
    mid_d = pick_mid(rows_d, eval_cases)
    shape_p = rest_shape_score(mid_p[0]["rest_resid"], mid_p[0]["rest_t"]) if mid_p else {"ok": False}
    shape_d = rest_shape_score(mid_d[0]["rest_resid"], mid_d[0]["rest_t"]) if mid_d else {"ok": False}
    # Also score diagnostic residual that was "eaten" vs 1RC on rest: |U_py-U4| small + rest looks double-exp in (U_py-U1)
    mid_c = mid_d[1] if mid_d else None
    diag_rest_raw = None
    if mid_c is not None:
        m = segment_masks(mid_c["cmd"], mid_c["i"])["rest_after_1c"]
        diag_rest_raw = rest_shape_score((mid_c["u"] - mid_c["u1"])[m], mid_c["t"][m] - mid_c["t"][m][0])

    def strip_resid(rows):
        out = []
        for r in rows:
            d = {k: v for k, v in r.items() if k not in ("rest_resid", "rest_t")}
            out.append(d)
        return out

    mean_c1 = mean_seg(rows_b1, "rmse_col1")
    mean_c2 = mean_seg(rows_b1, "rmse_col2")
    mean_c4p = {k: float(np.nanmean([r["rmse_col4"][k] for r in rows_p])) for k in rows_p[0]["rmse_col4"]}
    mean_c4d = {k: float(np.nanmean([r["rmse_col4"][k] for r in rows_d])) for k in rows_d[0]["rmse_col4"]}

    # 3B4 fail criterion: diagnostic "wins" on rest vs col1, and raw residual is double-exp shaped
    rest_gain_diag = mean_c1["rest_60_120s"] - mean_c4d["rest_60_120s"]
    rest_gain_proper = mean_c1["rest_60_120s"] - mean_c4p["rest_60_120s"]
    pulse_gain_diag = mean_c1["pulse_first_30s"] - mean_c4d["pulse_first_30s"]
    # FAIL对照 if diagnostic mainly harvests rest double-exp that physical 2RC already owns
    rest_gap_1_vs_2 = mean_c1["rest_60_120s"] - mean_c2["rest_60_120s"]
    fail_diag = bool(
        rest_gain_diag > 2.0
        and rest_gain_diag > rest_gain_proper + 1.0
        and rest_gap_1_vs_2 > 5.0
        and (
            (diag_rest_raw or {}).get("looks_like_exp", False)
            or rest_gain_diag >= pulse_gain_diag
        )
    )

    # beat 2RC? not a pass
    beats_2rc_proper = mean_c4p["all"] < mean_c2["all"]
    beats_2rc_note = "NOT default pass" if beats_2rc_proper else "does not beat 2RC (expected)"

    result = {
        "phase": "3B",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "grid": str(GRID.relative_to(REPO)),
        "n_files": len(files),
        "files": [f.name for f in files],
        "clip_mv": args.clip_mv,
        "used_ai_mlp": False,
        "mlp_dir": str(MLP_DIR.relative_to(REPO)),
        "lut_dir": str(LUT_DIR.relative_to(REPO)),
        "bv_ocp_labels": False,
        "caveats": [
            "Subset: regenerated Data/grid_pybamm 5×5 (25 SEQUENCE cases), no measurement noise.",
            "Proper teacher = U_py - U_2RC (LTI 2RC per file); BV/OCP labels not applied.",
            "δU clip tried at 8 mV; GRU d=4; freeze θ for increment not exercised here (open-loop only).",
            "Rest SOC / NIS not computed (open-loop voltage study).",
        ],
        "3B1": {
            "mean_rmse_col1_mv": mean_c1,
            "lut_rom_kb": lut["meta"]["rom_bytes"] / 1024,
            "not_ai_mlp": True,
        },
        "3B2": {
            "teacher": "proper",
            "n_params": proper["n_params"],
            "mean_rmse_col4_mv": mean_c4p,
            "mean_rmse_col1_mv": mean_c1,
            "gain_vs_col1_mv": {k: mean_c1[k] - mean_c4p[k] for k in mean_c1},
            "mid_case": mid_p[0]["name"] if mid_p else None,
            "rest_shape": shape_p,
        },
        "3B3": {
            "mean_rmse_col2_mv": mean_c2,
            "mean_rmse_col4_proper_mv": mean_c4p,
            "col4_minus_col2_mv": {k: mean_c4p[k] - mean_c2[k] for k in mean_c2},
            "beats_2rc_all": bool(beats_2rc_proper),
            "beats_2rc_note": beats_2rc_note,
            "rest_shape_proper": shape_p,
            "mid_case": mid_p[0]["name"] if mid_p else None,
            "upgrade_priority": bool(mean_c2["rest_60_120s"] + 0.5 < mean_c4p["rest_60_120s"]),
        },
        "3B4": {
            "teacher": "diagnostic",
            "n_params": diag["n_params"],
            "mean_rmse_col4_mv": mean_c4d,
            "gain_vs_col1_mv": {k: mean_c1[k] - mean_c4d[k] for k in mean_c1},
            "rest_gain_mv": rest_gain_diag,
            "pulse_gain_mv": pulse_gain_diag,
            "raw_rest_looks_like_2nd_exp": (diag_rest_raw or {}).get("looks_like_exp"),
            "raw_rest_shape": diag_rest_raw,
            "fail_control": fail_diag,
            "fail_note": (
                "FAIL对照: diagnostic head wins mainly on rest double-exp"
                if fail_diag
                else "diagnostic did not clearly win only via rest double-exp (see numbers)"
            ),
            "mid_case": mid_d[0]["name"] if mid_d else None,
            "rest_shape_after": shape_d,
        },
        "per_file_col1": strip_resid([{**r, "rmse_col4": None} for r in rows_b1]),
        "per_file_proper": strip_resid(rows_p),
        "per_file_diagnostic": strip_resid(rows_d),
    }
    (OUT / "summary_3b.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # compact table for docs
    def fmt(d, k):
        return f"{d[k]:.2f}"

    print("\n===== Phase-3 open-loop RMSE / mV (mean over grid) =====")
    hdr = f"{'seg':<18} {'col1 1RC':>10} {'col2 2RC':>10} {'col4 prop':>10} {'col4 diag':>10}"
    print(hdr)
    for seg in ("all", "edge", "pulse_first_30s", "rest_60_120s", "rest_after_1c"):
        print(
            f"{seg:<18} {fmt(mean_c1,seg):>10} {fmt(mean_c2,seg):>10} "
            f"{fmt(mean_c4p,seg):>10} {fmt(mean_c4d,seg):>10}"
        )
    print(f"\n3B4 fail_control={fail_diag}  beats_2rc_proper={beats_2rc_proper} ({beats_2rc_note})")
    print(f"wrote {OUT / 'summary_3b.json'}  elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
