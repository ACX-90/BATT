"""车上滑窗：(SOC,T) 上 5×4 k 网格。无 Replay，SGD，只动被点到的节点。

仓库根目录：

    python Src/AI/EV_Local/kgrid.py --exp a
    python Src/AI/EV_Local/kgrid.py --exp cold --make-cold
    python Src/AI/EV_Local/kgrid.py --exp both --make-cold

不覆盖 Data/ai_mlp / Data/grid / Data/soh_k115。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

KF_DIR = Path(__file__).resolve().parent
AI_DIR = KF_DIR.parent
SIM_DIR = KF_DIR.parent.parent / "Sim"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from KF.adapter import KGridAdapter
from KF.config import REPO_ROOT
from KF.increment import load_incr_sequences
from window import window_gate
from MLP.dataset import FeatureScaler
from MLP.ecm import ecm_forward
from MLP.infer import load_bundle
from MLP.train import set_seed


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _seq_tensors(seq: dict, scaler: FeatureScaler, device: torch.device) -> dict[str, torch.Tensor]:
    feat = np.stack([seq["i"], seq["soc"], seq["t"]], axis=-1)
    xn = scaler.transform(feat).astype(np.float32)
    return {
        "x": torch.from_numpy(xn).to(device),
        "i": torch.from_numpy(seq["i"].astype(np.float32)).to(device),
        "soc": torch.from_numpy(seq["soc"].astype(np.float32)).to(device),
        "t": torch.from_numpy(seq["t"].astype(np.float32)).to(device),
        "u_ocv": torch.from_numpy(seq["u_ocv"].astype(np.float32)).to(device),
        "u_t": torch.from_numpy(seq["u_t"].astype(np.float32)).to(device),
    }


def _apply(model: KGridAdapter, sl: dict[str, torch.Tensor], center: int | None, n: int):
    if center is None:
        return model(sl["x"].unsqueeze(0), sl["soc"].unsqueeze(0), sl["t"].unsqueeze(0))
    r0, r1, c1 = model(
        sl["x"][center : center + 1].unsqueeze(0),
        sl["soc"][center : center + 1].unsqueeze(0),
        sl["t"][center : center + 1].unsqueeze(0),
    )
    return r0.expand(1, n), r1.expand(1, n), c1.expand(1, n)


@torch.no_grad()
def overall_rmse(model: KGridAdapter, seqs: list[dict], scaler: FeatureScaler, cfg, device) -> float:
    sse = 0.0
    n_tot = 0
    model.eval()
    for seq in seqs:
        t = _seq_tensors(seq, scaler, device)
        r0, r1, c1 = _apply(model, t, None, 0)
        u_hat, _ = ecm_forward(
            t["i"].unsqueeze(0),
            t["u_ocv"].unsqueeze(0),
            r0,
            r1,
            c1,
            dt_s=cfg.dt_s,
            u_p0=t["i"].new_tensor([float(seq.get("u_p0", 0.0))]),
        )
        err = u_hat.squeeze(0) - t["u_t"]
        sse += float(err.pow(2).sum().cpu())
        n_tot += int(err.numel())
    return (sse / max(n_tot, 1)) ** 0.5


def _roll_up(model, sl, u_p0, dt_s, center) -> torch.Tensor:
    with torch.no_grad():
        n = sl["i"].shape[0]
        r0, r1, c1 = _apply(model, sl, center, n)
        _, u_p = ecm_forward(
            sl["i"].unsqueeze(0), sl["u_ocv"].unsqueeze(0), r0, r1, c1, dt_s=dt_s, u_p0=u_p0
        )
    return u_p[:, -1].detach()


def _smooth_live(model: KGridAdapter, soc: torch.Tensor, t_c: torch.Tensor, lam: float) -> torch.Tensor:
    if lam <= 0:
        return soc.new_zeros(())
    is_, it_, ws, wt = model._corners(soc, t_c)
    i0, j0 = int(is_), int(it_)
    wts = [
        ((i0, j0), (1.0 - float(ws)) * (1.0 - float(wt))),
        ((i0 + 1, j0), float(ws) * (1.0 - float(wt))),
        ((i0, j0 + 1), (1.0 - float(ws)) * float(wt)),
        ((i0 + 1, j0 + 1), float(ws) * float(wt)),
    ]
    live0 = [model.log_k0[i, j] for (i, j), w in wts if w >= 0.05]
    live1 = [model.log_k1[i, j] for (i, j), w in wts if w >= 0.05]
    loss = soc.new_zeros(())
    if len(live0) >= 2:
        z = torch.stack(live0)
        loss = loss + 0.5 * ((z - z.mean()) ** 2).mean()
    if len(live1) >= 2:
        z = torch.stack(live1)
        loss = loss + 0.5 * ((z - z.mean()) ** 2).mean()
    return lam * loss


def step_window(model, opt, sl, u_p0, *, dt_s, i_np, args, i_prev, hit0, hit1) -> dict:
    n = int(sl["i"].shape[0])
    mid = n // 2
    gated, gstat = window_gate(
        i_np,
        dt_s=dt_s,
        i_edge_a=args.i_edge,
        rest_eps=args.rest_eps,
        rest_s=args.rest_s,
        i_prev=i_prev,
    )
    i_rms = float(np.sqrt(np.mean(np.square(i_np))))
    stats = {"gated": gated, "updated": False, "rmse": float("nan"), "i_rms": i_rms, **gstat}
    if not gated:
        stats["u_p0"] = _roll_up(model, sl, u_p0, dt_s, mid)
        return stats

    opt.zero_grad(set_to_none=True)
    r0, r1, c1 = _apply(model, sl, mid, n)
    u_hat, u_p = ecm_forward(
        sl["i"].unsqueeze(0), sl["u_ocv"].unsqueeze(0), r0, r1, c1, dt_s=dt_s, u_p0=u_p0
    )
    err = u_hat.squeeze(0) - sl["u_t"]
    rmse = float(err.pow(2).mean().sqrt().detach().cpu())
    stats["rmse"] = rmse
    if rmse < args.e_ol_min:
        stats["u_p0"] = u_p[:, -1].detach()
        return stats

    loss = 0.5 * err.pow(2).mean()
    loss = loss + _smooth_live(model, sl["soc"][mid], sl["t"][mid], args.smooth)
    loss.backward()
    if i_rms < 20.0 and model.log_k0.grad is not None:
        model.log_k0.grad.zero_()
    if gstat["rest_s"] < args.rest_s and model.log_k1.grad is not None:
        model.log_k1.grad.zero_()
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_([model.log_k0, model.log_k1], args.grad_clip)
    opt.step()
    with torch.no_grad():
        lo, hi = float(np.log(args.k_lo)), float(np.log(args.k_hi))
        model.log_k0.clamp_(lo, hi)
        model.log_k1.clamp_(lo, hi)
        is_, it_, ws, wt = model._corners(sl["soc"][mid], sl["t"][mid])
        i0, j0 = int(is_), int(it_)
        w00 = (1.0 - float(ws)) * (1.0 - float(wt))
        w10 = float(ws) * (1.0 - float(wt))
        w01 = (1.0 - float(ws)) * float(wt)
        w11 = float(ws) * float(wt)
        if i_rms >= 20.0:
            hit0[i0, j0] += w00
            hit0[i0 + 1, j0] += w10
            hit0[i0, j0 + 1] += w01
            hit0[i0 + 1, j0 + 1] += w11
        if gstat["rest_s"] >= args.rest_s:
            hit1[i0, j0] += w00
            hit1[i0 + 1, j0] += w10
            hit1[i0, j0 + 1] += w01
            hit1[i0 + 1, j0 + 1] += w11
    stats["updated"] = True
    stats["u_p0"] = u_p[:, -1].detach()
    return stats


def run_pass(model, opt, seqs, scaler, cfg, device, args) -> dict:
    n_win = n_upd = n_skip_gate = n_skip_small = 0
    ns, nt = model.log_k0.shape
    hit0 = np.zeros((ns, nt), dtype=float)
    hit1 = np.zeros((ns, nt), dtype=float)
    win = int(args.win)
    for si, seq in enumerate(seqs):
        t = _seq_tensors(seq, scaler, device)
        n = int(t["i"].shape[0])
        u_p0 = t["i"].new_tensor([float(seq.get("u_p0", 0.0))])
        for start in range(0, n, win):
            end = min(start + win, n)
            if end - start < max(win // 4, 20):
                break
            sl = {k: v[start:end] for k, v in t.items()}
            i_prev = float(seq["i"][start - 1]) if start > 0 else None
            st = step_window(
                model,
                opt,
                sl,
                u_p0,
                dt_s=cfg.dt_s,
                i_np=seq["i"][start:end],
                args=args,
                i_prev=i_prev,
                hit0=hit0,
                hit1=hit1,
            )
            u_p0 = st.pop("u_p0")
            n_win += 1
            if st["updated"]:
                n_upd += 1
            elif not st["gated"]:
                n_skip_gate += 1
            else:
                n_skip_small += 1
        k0, k1 = model.k_at(float(seq["soc"].mean()), float(seq["t"].mean()))
        print(
            f"  seq {si+1:02d}/{len(seqs)} {seq['name']}  "
            f"k@mean=({k0:.3f},{k1:.3f})  T={float(seq['t'].mean()):+.1f}",
            flush=True,
        )
    return {
        "n_win": n_win,
        "n_update": n_upd,
        "n_skip_gate": n_skip_gate,
        "n_skip_small": n_skip_small,
        "hit0": hit0.tolist(),
        "hit1": hit1.tolist(),
    }


def cc_probe(base, scaler, seqs, cfg, device, args) -> dict:
    model = KGridAdapter(base).to(device)
    opt = torch.optim.SGD([model.log_k0, model.log_k1], lr=args.lr)
    n_win = n_upd = 0
    dummy0 = np.zeros_like(model.log_k0.detach().cpu().numpy())
    dummy1 = dummy0.copy()
    for seq in seqs[: min(5, len(seqs))]:
        mag = np.abs(seq["i"])
        mask = (mag > 80.0) & (mag < 120.0)
        if mask.sum() < args.win:
            continue
        idx = np.flatnonzero(mask)
        cuts = np.where(np.diff(idx) > 1)[0]
        block = idx if len(cuts) == 0 else max(np.split(idx, cuts + 1), key=len)
        t = _seq_tensors(seq, scaler, device)
        sl_all = {k: v[int(block[0]) : int(block[-1]) + 1] for k, v in t.items()}
        i_np = seq["i"][int(block[0]) : int(block[-1]) + 1]
        u_p0 = t["i"].new_tensor([0.0])
        n = len(i_np)
        for start in range(0, n, args.win):
            end = min(start + args.win, n)
            if end - start < max(args.win // 4, 20):
                break
            sl = {k: v[start:end] for k, v in sl_all.items()}
            i_prev = float(i_np[start - 1]) if start > 0 else None
            st = step_window(
                model,
                opt,
                sl,
                u_p0,
                dt_s=cfg.dt_s,
                i_np=i_np[start:end],
                args=args,
                i_prev=i_prev,
                hit0=dummy0,
                hit1=dummy1,
            )
            u_p0 = st.pop("u_p0")
            n_win += 1
            n_upd += int(st["updated"])
    tab = model.k_tables()
    return {"n_win": n_win, "n_update": n_upd, "k0_max": float(np.max(tab["k0"])), "k1_max": float(np.max(tab["k1"]))}


def fmt_table(name: str, soc: list, t_c: list, k: list[list[float]]) -> str:
    head = f"{name:4s}" + "".join(f"  T{tc:+.0f}" for tc in t_c)
    lines = [head]
    for i, s in enumerate(soc):
        row = f"s{s:.2f}" + "".join(f"  {k[i][j]:5.3f}" for j in range(len(t_c)))
        lines.append(row)
    return "\n".join(lines)


def by_temp(seqs: list[dict], lo: float, hi: float) -> list[dict]:
    out = []
    for seq in seqs:
        tm = float(np.mean(seq["t"]))
        if lo <= tm <= hi:
            out.append(seq)
    return out


def summarize_k(tab: dict) -> dict:
    k0 = np.asarray(tab["k0"], dtype=float)
    k1 = np.asarray(tab["k1"], dtype=float)
    t_c = np.asarray(tab["t_node"], dtype=float)
    return {
        "k0_mean": float(k0.mean()),
        "k1_mean": float(k1.mean()),
        "k0_cold": float(k0[:, 0].mean()),
        "k1_cold": float(k1[:, 0].mean()),
        "k0_mid": float(k0[:, 1:-1].mean()) if k0.shape[1] > 2 else float(k0[:, 1].mean()),
        "k1_mid": float(k1[:, 1:-1].mean()) if k1.shape[1] > 2 else float(k1[:, 1].mean()),
        "k0_hot": float(k0[:, -1].mean()),
        "k1_hot": float(k1[:, -1].mean()),
        "t_cold": float(t_c[0]),
        "t_hot": float(t_c[-1]),
    }


def make_cold_grid(out_dir: Path, *, scale: float = 1.15, cold_t: float = -10.0) -> Path:
    from nmc100ah_ecm import make_ecm
    from nmc100ah_gen import (
        DT_S,
        ENABLE_CUTOFF,
        NOISE_ENABLE,
        NOISE_SEED,
        NOISE_STD,
        SEQUENCE,
        U_P0,
        simulate,
        write_csv,
    )
    from nmc100ah_gen_grid import _write_index, build_grid, case_name, clear_output_dir

    soc_axis, t_axis = build_grid(5, 5)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_old = clear_output_dir(out_dir)
    if n_old:
        print(f"已删除旧文件 {n_old} 份  {out_dir}")
    rows = []
    idx = 0
    print(f"冷端错配网格 {out_dir}  T<={cold_t} 时 R×{scale:g}，其余 ×1")
    for i, soc0 in enumerate(soc_axis):
        for j, t_c in enumerate(t_axis):
            sc = scale if float(t_c) <= cold_t + 0.5 else 1.0
            model = make_ecm(r0_scale=sc, r1_scale=sc, c1_scale=1.0)
            fname = case_name(i, j, float(soc0), float(t_c))
            seed = int(NOISE_SEED) + i * 100 + j
            data = simulate(
                model,
                SEQUENCE,
                dt_s=DT_S,
                soc0=float(soc0),
                t_ambient_c=float(t_c),
                u_p0=U_P0,
                noise_enable=NOISE_ENABLE,
                noise_seed=seed,
                noise_std=dict(NOISE_STD),
                enable_cutoff=ENABLE_CUTOFF,
            )
            csv_path = write_csv(
                out_dir / fname,
                data,
                dt_s=DT_S,
                soc0=float(soc0),
                noise_enable=NOISE_ENABLE,
                noise_seed=seed,
                noise_std=dict(NOISE_STD),
                sequence=SEQUENCE,
                extra_meta=[
                    "# source=kgrid_make_cold",
                    f"# r0_scale={sc}",
                    f"# r1_scale={sc}",
                    f"# cold_t={cold_t}",
                ],
            )
            rec = {
                "idx": idx,
                "i_soc": i,
                "j_temp": j,
                "soc0": float(soc0),
                "t_celsius": float(t_c),
                "noise_seed": seed,
                "n_steps": int(len(data["time_s"])),
                "duration_s": float(data["time_s"][-1] + DT_S),
                "soc_end": float(data["soc_true"][-1]),
                "ut_end": float(data["u_t_true_v"][-1]),
                "cutoff_steps": int(np.sum(data["cutoff"] > 0)),
                "file": fname,
                "path": str(csv_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "r_scale": sc,
            }
            print(
                f"  [{idx:02d}] SOC {soc0:.2f}  T={t_c:+6.1f}  R×{sc:g}  {fname}",
                flush=True,
            )
            rows.append(rec)
            idx += 1
    _write_index(out_dir / "index.csv", rows)
    (out_dir / "ecm_meta.json").write_text(
        json.dumps(
            {"r_scale_cold": scale, "cold_t": cold_t, "note": "only T<=cold_t raised"},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out_dir


def run_exp(args, *, new_dir: Path, out_dir: Path, tag: str) -> dict:
    mlp_dir = _resolve(args.mlp_dir)
    old_dir = _resolve(args.old_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if new_dir.resolve() == old_dir.resolve():
        raise RuntimeError("新年份不能和旧网格是同一目录")
    device = torch.device(args.device)
    ckpt = mlp_dir / "best.pt"
    base, scaler, cfg = load_bundle(ckpt, mlp_dir / "config.json", mlp_dir / "scaler.json")
    for par in base.parameters():
        par.requires_grad_(False)
    base = base.to(device).eval()

    new_seq = load_incr_sequences(
        new_dir, pattern=None, use_true_inputs=args.use_true_inputs, weight=1.0, style="grid"
    )
    old_true = bool(cfg.use_true_inputs)
    old_seq = load_incr_sequences(
        old_dir, pattern=None, use_true_inputs=old_true, weight=1.0, style="grid"
    )
    print(
        f"kgrid tag={tag} mlp={mlp_dir} new={len(new_seq)} old_eval={len(old_seq)}  "
        f"win={args.win} lr={args.lr} smooth={args.smooth} Replay=OFF"
    )

    frozen = KGridAdapter(base).to(device)
    new0 = overall_rmse(frozen, new_seq, scaler, cfg, device)
    old0 = overall_rmse(frozen, old_seq, scaler, cfg, device)
    cold_seq = by_temp(new_seq, -15, -5)
    mid_seq = by_temp(new_seq, 15, 60)
    new0_cold = overall_rmse(frozen, cold_seq, scaler, cfg, device) if cold_seq else None
    new0_mid = overall_rmse(frozen, mid_seq, scaler, cfg, device) if mid_seq else None
    print(f"冻结  新 {new0*1e3:.2f} mV  旧 {old0*1e3:.2f} mV", end="")
    if new0_cold is not None:
        print(f"  冷 {new0_cold*1e3:.2f}  中温 {new0_mid*1e3:.2f}", end="")
    print()

    cc = cc_probe(base, scaler, new_seq, cfg, device, args)
    print(f"恒流探针  win={cc['n_win']} update={cc['n_update']}  kmax={cc['k0_max']:.4f}/{cc['k1_max']:.4f}")

    model = KGridAdapter(base).to(device)
    opt = torch.optim.SGD([model.log_k0, model.log_k1], lr=args.lr)
    row = None
    for p_i in range(1, args.passes + 1):
        print(f"---- {tag} pass {p_i}/{args.passes} ----", flush=True)
        row = run_pass(model, opt, new_seq, scaler, cfg, device, args)
        new_a = overall_rmse(model, new_seq, scaler, cfg, device)
        old_a = overall_rmse(model, old_seq, scaler, cfg, device)
        row["pass"] = p_i
        row["new_rmse"] = new_a
        row["old_rmse"] = old_a
        print(
            f"  pass {p_i}: upd {row['n_update']}/{row['n_win']}  "
            f"新 {new_a*1e3:.2f} mV  旧 {old_a*1e3:.2f} mV",
            flush=True,
        )

    tab = model.k_tables()
    print(fmt_table("k0", tab["soc_node"], tab["t_node"], tab["k0"]))
    print(fmt_table("k1", tab["soc_node"], tab["t_node"], tab["k1"]))
    ksum = summarize_k(tab)
    print(
        f"  列均  冷 T{ksum['t_cold']:+.0f} k0/k1={ksum['k0_cold']:.3f}/{ksum['k1_cold']:.3f}  "
        f"中温 {ksum['k0_mid']:.3f}/{ksum['k1_mid']:.3f}  "
        f"热 T{ksum['t_hot']:+.0f} {ksum['k0_hot']:.3f}/{ksum['k1_hot']:.3f}"
    )
    new_a = row["new_rmse"]
    old_a = row["old_rmse"]
    new_a_cold = overall_rmse(model, cold_seq, scaler, cfg, device) if cold_seq else None
    new_a_mid = overall_rmse(model, mid_seq, scaler, cfg, device) if mid_seq else None
    k_ref = model.k_at(0.50, 25.0)
    meta = {
        "mode": "window_kgrid",
        "tag": tag,
        "mlp_dir": str(mlp_dir),
        "new_dir": str(new_dir),
        "old_dir": str(old_dir),
        "win": args.win,
        "lr": args.lr,
        "smooth": args.smooth,
        "n_passes": args.passes,
        "optimizer": "SGD",
        "replay": False,
        "center_R": True,
        "n_new": len(new_seq),
        "n_old_eval": len(old_seq),
        "new_rmse_before": new0,
        "old_rmse_before": old0,
        "new_rmse_after": new_a,
        "old_rmse_after": old_a,
        "new_rmse_cold_before": new0_cold,
        "new_rmse_cold_after": new_a_cold,
        "new_rmse_mid_before": new0_mid,
        "new_rmse_mid_after": new_a_mid,
        "k_at_ref": list(k_ref),
        "k_summary": ksum,
        "k_tables": tab,
        "n_win": row["n_win"],
        "n_update": row["n_update"],
        "n_skip_gate": row["n_skip_gate"],
        "n_skip_small": row["n_skip_small"],
        "hit0": row["hit0"],
        "hit1": row["hit1"],
        "cc_probe": cc,
    }
    (out_dir / "kgrid.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "log_k0": model.log_k0.detach().cpu(),
            "log_k1": model.log_k1.detach().cpu(),
            "incr_mode": "window_kgrid",
            "k_tables": tab,
        },
        out_dir / "last.pt",
    )
    scaler.save(out_dir / "scaler.json")
    cfg.to_json(out_dir / "config.json")
    print(f"写出 {out_dir / 'kgrid.json'}")
    if not args.no_plot:
        from plot import plot_from_out

        prefix = args.fig_prefix or f"local/{out_dir.name}"
        figs = plot_from_out(
            out_dir,
            new_dir=new_dir,
            mlp_dir=mlp_dir,
            fig_prefix=prefix,
            use_true_inputs=args.use_true_inputs,
        )
        for fig in figs:
            print(f"图    {fig}")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="车上滑窗 5×4 k 网格")
    p.add_argument("--exp", default="both", choices=["a", "cold", "both"])
    p.add_argument("--mlp-dir", default="Data/ai_mlp")
    p.add_argument("--new-dir", default="Data/soh_k115")
    p.add_argument("--cold-dir", default="Data/soh_cold_tm10")
    p.add_argument("--old-dir", default="Data/grid")
    p.add_argument("--out-a", default="Data/ai_local/kgrid_k115")
    p.add_argument("--out-cold", default="Data/ai_local/kgrid_cold")
    p.add_argument("--make-cold", action="store_true")
    p.add_argument("--win", type=int, default=100)
    p.add_argument("--lr", type=float, default=10.0)
    p.add_argument("--smooth", type=float, default=1.0e-3)
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--i-edge", type=float, default=20.0)
    p.add_argument("--rest-s", type=float, default=3.0)
    p.add_argument("--rest-eps", type=float, default=1.0)
    p.add_argument("--e-ol-min", type=float, default=0.002)
    p.add_argument("--k-lo", type=float, default=0.5)
    p.add_argument("--k-hi", type=float, default=2.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--use-true-inputs", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--fig-prefix", default=None)
    args = p.parse_args()
    set_seed(args.seed)

    if args.exp in {"cold", "both"} and args.make_cold:
        make_cold_grid(_resolve(args.cold_dir))

    if args.exp in {"a", "both"}:
        run_exp(args, new_dir=_resolve(args.new_dir), out_dir=_resolve(args.out_a), tag="A_x115")
    if args.exp in {"cold", "both"}:
        run_exp(args, new_dir=_resolve(args.cold_dir), out_dir=_resolve(args.out_cold), tag="cold_tm10")


if __name__ == "__main__":
    main()
