"""包级滑窗 k 网格 + 包级门（Doc/06-a §7）。

    python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2a1 --mode freeze
    python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2a1 --mode kgrid --out-dir Data/pack/2a1_kgrid
    python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2a1 --mode ifx_demo --out-dir Data/pack/2a1_ifx_demo

不覆盖 Data/grid / Data/ai_mlp / Data/ai_local。更新路径不读旧网格。
ifx_demo 是 10 mV / 0.1 s 失败对照，不要用 10 s 窗假装。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

KF_DIR = Path(__file__).resolve().parent
AI_DIR = KF_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from KF.adapter import KGRID_SOC, KGRID_T, KGridAdapter, MlpParamProvider, ResidualHeadAdapter  # noqa: E402
from KF.config import KfConfig, REPO_ROOT  # noqa: E402
from KF.filter import filter_metrics, run_filter  # noqa: E402
from KF.ocv import docv_ds, ocv_nmc  # noqa: E402
from KF.pack_gate import last_edge_age_s, pack_gate, window_policy  # noqa: E402
from MLP.ecm import ecm_forward  # noqa: E402
from MLP.infer import load_bundle  # noqa: E402
from MLP.train import set_seed  # noqa: E402
from window import window_gate  # noqa: E402
from kgrid import _roll_up, _seq_tensors, overall_rmse, step_window  # noqa: E402


FORBIDDEN = {
    "Data/grid",
    "Data/grid_pybamm",
    "Data/ai_mlp",
    "Data/ai_local",
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _guard_out(path: Path) -> None:
    rel = _rel(path)
    for bad in FORBIDDEN:
        if rel == bad or rel.startswith(bad + "/"):
            raise RuntimeError(f"禁止写到 {rel}")


def load_pack(pack_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    meta = json.loads((pack_dir / "pack.json").read_text(encoding="utf-8"))
    blob = np.load(pack_dir / "pack.npz")
    data = {k: blob[k] for k in blob.files}
    exp = str(meta.get("exp", ""))
    b_i = float(meta.get("b_I", 0.0))
    if b_i != 0.0 and exp.startswith("2a"):
        raise RuntimeError(f"{exp} 必须 b_I=0")
    if exp == "2b" and abs(b_i - 5.0) > 1e-6:
        raise RuntimeError("2b 必须 b_I=5")
    if exp == "2e" and abs(b_i) > 1e-6:
        raise RuntimeError("2e 必须 b_I=0")
    if exp.startswith("2d") and abs(b_i) > 1e-6:
        raise RuntimeError(f"{exp} 必须 b_I=0")
    return meta, data


def pack_ocv_fn(data: dict[str, np.ndarray]):
    """用生成器写出的 bulk OCV 当本包 OCV 表（真包会先标定，不要混仓库 ocv.py）。"""
    s = np.asarray(data["soc_true"], dtype=float).ravel()
    v = np.asarray(data["u_ocv"], dtype=float).ravel()
    order = np.argsort(s)
    s, v = s[order], v[order]
    _, idx = np.unique(np.round(s, 6), return_index=True)
    s_u, v_u = s[idx], v[idx]
    if s_u.size < 4:
        return ocv_nmc, docv_ds
    ds = np.diff(s_u)
    dv = np.diff(v_u)
    slopes = np.divide(dv, np.maximum(ds, 1e-8))

    def ocv(soc, t_c=25.0):
        del t_c
        x = np.clip(np.asarray(soc, dtype=float), float(s_u[0]), float(s_u[-1]))
        val = np.interp(x, s_u, v_u)
        return float(val) if np.ndim(soc) == 0 else val

    def docv(soc, t_c=25.0):
        del t_c
        x = np.clip(np.asarray(soc, dtype=float), float(s_u[0]), float(s_u[-1]))
        j = np.clip(np.searchsorted(s_u, x, side="right") - 1, 0, len(slopes) - 1)
        val = slopes[j]
        return float(val) if np.ndim(soc) == 0 else val

    return ocv, docv


def cell_seq(
    data: dict[str, np.ndarray],
    i: int,
    *,
    soc: np.ndarray,
    name: str,
    ocv=None,
) -> dict:
    t_c = data["t_meas"][:, i]
    ocv_fn = ocv_nmc if ocv is None else ocv
    return {
        "name": name,
        "i": np.asarray(data["i_meas"], dtype=float),
        "soc": np.asarray(soc, dtype=float),
        "t": np.asarray(t_c, dtype=float),
        "u_ocv": np.asarray(ocv_fn(soc, t_c), dtype=float),
        "u_t": np.asarray(data["u_t_meas"][:, i], dtype=float),
        "u_p0": 0.0,
    }


def run_cell_filter(
    provider: MlpParamProvider,
    data: dict,
    i: int,
    soc0: float,
    *,
    ocv=None,
    docv=None,
    cfg: KfConfig | None = None,
) -> dict:
    log = run_filter(
        provider,
        np.asarray(data["i_meas"], dtype=float),
        np.asarray(data["t_meas"][:, i], dtype=float),
        np.asarray(data["u_t_meas"][:, i], dtype=float),
        cfg=cfg,
        s0=float(soc0),
        soc_true=np.asarray(data["soc_true"][:, i], dtype=float),
        ocv=ocv,
        docv=docv,
    )
    return log


def _trip_index(start: int, trips: list[int]) -> int:
    idx = 0
    for i, t0 in enumerate(trips):
        if start >= int(t0):
            idx = i
        else:
            break
    return idx


def _kgrid_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        i_edge=args.i_edge,
        rest_eps=args.rest_eps,
        rest_s=args.rest_s,
        e_ol_min=args.e_ol_min,
        smooth=args.smooth,
        k_lo=args.k_lo,
        k_hi=args.k_hi,
        grad_clip=args.grad_clip,
        win=args.win,
    )


def run_cell_kgrid(
    model: KGridAdapter,
    seq: dict,
    scaler,
    cfg,
    device,
    args,
    *,
    pack_blocked: bool,
    nogate: bool,
    trips: list[int] | None = None,
    k_after_trips: int = 1,
) -> dict:
    win = int(args.win)
    dt_s = cfg.dt_s
    t = _seq_tensors(seq, scaler, device)
    n = int(t["i"].shape[0])
    u_p0 = t["i"].new_tensor([0.0])
    opt = torch.optim.SGD([model.log_k0, model.log_k1], lr=args.lr)
    ns, nt = model.log_k0.shape
    hit0 = np.zeros((ns, nt), dtype=float)
    hit1 = np.zeros((ns, nt), dtype=float)
    n_win = n_upd = n_skip_gate = n_skip_pack = n_skip_park = n_skip_trip = 0
    kg_args = _kgrid_args(args)
    i_np_all = seq["i"]
    trip_starts = [int(x) for x in (trips or [0])]
    hold_until = max(int(k_after_trips) - 1, 0)
    for start in range(0, n, win):
        end = min(start + win, n)
        if end - start < max(win // 4, 20):
            break
        n_win += 1
        sl = {k: v[start:end] for k, v in t.items()}
        i_prev = float(i_np_all[start - 1]) if start > 0 else None
        i_np = i_np_all[start:end]
        gated, gstat = window_gate(
            i_np,
            dt_s=dt_s,
            i_edge_a=args.i_edge,
            rest_eps=args.rest_eps,
            rest_s=args.rest_s,
            i_prev=i_prev,
        )
        age = last_edge_age_s(
            i_np_all, start, dt_s=dt_s, i_edge_a=args.i_edge, i_prev=None
        )
        pol = window_policy(has_edge=bool(gstat["has_edge"]), last_edge_age_s=age)
        if pack_blocked and not nogate:
            n_skip_pack += 1
            u_p0 = _roll_up(model, sl, u_p0, dt_s, len(i_np) // 2)
            continue
        if _trip_index(start, trip_starts) < hold_until:
            n_skip_trip += 1
            u_p0 = _roll_up(model, sl, u_p0, dt_s, len(i_np) // 2)
            continue
        if not pol["write_k"]:
            n_skip_park += 1
            u_p0 = _roll_up(model, sl, u_p0, dt_s, len(i_np) // 2)
            continue
        if not pol["allow_k1"]:
            kg_args.rest_s = 1e9
        else:
            kg_args.rest_s = args.rest_s
        st = step_window(
            model,
            opt,
            sl,
            u_p0,
            dt_s=dt_s,
            i_np=i_np,
            args=kg_args,
            i_prev=i_prev,
            hit0=hit0,
            hit1=hit1,
        )
        u_p0 = st.pop("u_p0")
        if st["updated"]:
            n_upd += 1
        elif not gated:
            n_skip_gate += 1
    k_ref = model.k_at(0.50, 25.0)
    k_mean = model.k_at(float(seq["soc"].mean()), float(seq["t"].mean()))
    return {
        "n_win": n_win,
        "n_update": n_upd,
        "n_skip_gate": n_skip_gate,
        "n_skip_pack": n_skip_pack,
        "n_skip_park": n_skip_park,
        "n_skip_trip": n_skip_trip,
        "k_at_ref": list(k_ref),
        "k_at_mean": list(k_mean),
        "k_tables": model.k_tables(),
        "hit0": hit0.tolist(),
        "hit1": hit1.tolist(),
    }


def _xn(scaler, i_a: float, soc: float, t_c: float, device) -> torch.Tensor:
    feat = np.array([[i_a, soc, t_c]], dtype=float)
    return torch.from_numpy(scaler.transform(feat).astype(np.float32)).to(device)


def _head_k_at(model: ResidualHeadAdapter, scaler, i_a: float, soc: float, t_c: float, device):
    x = _xn(scaler, i_a, soc, t_c, device)
    with torch.no_grad():
        r0, r1, _ = model(x)
        r0f, r1f, _ = model.fleet(x)
        d = model.delta(x).reshape(2)
    return (
        float(r0 / r0f.clamp_min(1e-12)),
        float(r1 / r1f.clamp_min(1e-12)),
        float(d[0]),
        float(d[1]),
    )


def _head_k_tables(model: ResidualHeadAdapter, scaler, device) -> dict:
    k0, k1, dr0, dr1 = [], [], [], []
    for s in KGRID_SOC:
        row_k0, row_k1, row_d0, row_d1 = [], [], [], []
        for t_c in KGRID_T:
            kk0, kk1, d0, d1 = _head_k_at(model, scaler, 100.0, float(s), float(t_c), device)
            row_k0.append(kk0)
            row_k1.append(kk1)
            row_d0.append(d0)
            row_d1.append(d1)
        k0.append(row_k0)
        k1.append(row_k1)
        dr0.append(row_d0)
        dr1.append(row_d1)
    return {
        "soc_node": list(KGRID_SOC),
        "t_node": list(KGRID_T),
        "k0": k0,
        "k1": k1,
        "dr0": dr0,
        "dr1": dr1,
    }


def run_cell_demo(
    model: ResidualHeadAdapter,
    seq: dict,
    scaler,
    cfg,
    device,
    *,
    e_thr: float,
    lr: float,
    grad_clip: float,
) -> dict:
    """|e_ol|>10 mV 就当前拍 SGD，无窗门 / 包门 / 停放门（06-a ifx_demo 原样）。

    舰队 / φ 按拍预计算；循环里只动 18 个数。
    """
    dt_s = float(cfg.dt_s)
    t = _seq_tensors(seq, scaler, device)
    n = int(t["i"].shape[0])
    with torch.no_grad():
        r0f, r1f, c1f = model.fleet(t["x"])
        h = model.phi(t["x"])
    r0f = r0f.detach()
    r1f = r1f.detach()
    c1f = c1f.detach()
    h = h.detach()
    i_all = t["i"]
    ocv_all = t["u_ocv"]
    ut_all = t["u_t"]
    u_p = t["i"].new_zeros(())
    opt = torch.optim.SGD(model.trainable_parameters(), lr=lr)
    n_upd = 0
    dt = t["i"].new_tensor(dt_s)
    dr_max = model.dr_max
    for k in range(n):
        i_k = i_all[k]
        c1 = c1f[k]
        with torch.no_grad():
            z = model.head(h[k])
            d = dr_max * torch.tanh(z)
            r0 = r0f[k] + d[0]
            r1 = r1f[k] + d[1]
            tau = (r1 * c1).clamp_min(1.0e-6)
            alpha = torch.exp(-dt / tau)
            u_p_new = alpha * u_p + r1 * (1.0 - alpha) * i_k
            e = (ocv_all[k] - i_k * r0 - u_p_new) - ut_all[k]
            e_abs = float(e.abs())
        if e_abs > e_thr:
            opt.zero_grad(set_to_none=True)
            z = model.head(h[k])
            d = dr_max * torch.tanh(z)
            r0 = r0f[k] + d[0]
            r1 = r1f[k] + d[1]
            tau = (r1 * c1).clamp_min(1.0e-6)
            alpha = torch.exp(-dt / tau)
            u_p_new = alpha * u_p + r1 * (1.0 - alpha) * i_k
            e = (ocv_all[k] - i_k * r0 - u_p_new) - ut_all[k]
            (0.5 * e.pow(2)).backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
            opt.step()
            n_upd += 1
            with torch.no_grad():
                z = model.head(h[k])
                d = dr_max * torch.tanh(z)
                r1 = r1f[k] + d[1]
                tau = (r1 * c1).clamp_min(1.0e-6)
                alpha = torch.exp(-dt / tau)
                u_p = (alpha * u_p.detach() + r1 * (1.0 - alpha) * i_k).detach()
        else:
            u_p = u_p_new.detach()

    soc_m = float(seq["soc"].mean())
    t_m = float(seq["t"].mean())
    k_ref = _head_k_at(model, scaler, 100.0, 0.50, 25.0, device)
    k_mean = _head_k_at(model, scaler, 100.0, soc_m, t_m, device)
    return {
        "n_win": n,
        "n_update": n_upd,
        "n_skip_gate": n - n_upd,
        "n_skip_pack": 0,
        "n_skip_park": 0,
        "k_at_ref": [k_ref[0], k_ref[1]],
        "k_at_mean": [k_mean[0], k_mean[1]],
        "dr_at_mean": [k_mean[2], k_mean[3]],
        "k_tables": _head_k_tables(model, scaler, device),
        "demo_e_thr_mV": e_thr * 1e3,
        "demo_lr": lr,
    }


@torch.no_grad()
def _rmse_head(model: ResidualHeadAdapter, seq: dict, scaler, cfg, device) -> float:
    t = _seq_tensors(seq, scaler, device)
    r0, r1, c1 = model(t["x"].unsqueeze(0))
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
    return float(err.pow(2).mean().sqrt().cpu())


def _load_phi(phi_dir: Path, fleet, scaler, cfg, device):
    from head import load_or_make_phi

    args = SimpleNamespace(make_phi=not (phi_dir / "best.pt").exists(), phi_epochs=40)
    if args.make_phi:
        print(f"缺少 {phi_dir}，实验室蒸馏 3×8 前层（读 Data/grid，不写回舰队）", flush=True)
    return load_or_make_phi(fleet, scaler, cfg, phi_dir, device, args)


def summarize_cells(cells: list[dict], rows: list[dict]) -> dict:
    aged = [r for r, c in zip(rows, cells) if c["aged"]]
    nom = [r for r, c in zip(rows, cells) if not c["aged"]]

    def _mean_k(group, idx):
        if not group:
            return float("nan")
        return float(np.mean([g["k_at_mean"][idx] for g in group]))

    def _mean(group, key):
        if not group or key not in group[0]:
            return float("nan")
        return float(np.mean([g[key] for g in group]))

    out = {
        "n_aged": len(aged),
        "n_nom": len(nom),
        "k0_aged": _mean_k(aged, 0),
        "k1_aged": _mean_k(aged, 1),
        "k0_nom": _mean_k(nom, 0),
        "k1_nom": _mean_k(nom, 1),
        "rmse_ol_aged_mV": _mean(aged, "e_ol_rmse_mV"),
        "rmse_ol_nom_mV": _mean(nom, "e_ol_rmse_mV"),
        "rmse_frozen_aged_mV": _mean(aged, "rmse_frozen_true_mV"),
        "rmse_frozen_nom_mV": _mean(nom, "rmse_frozen_true_mV"),
        "rmse_after_aged_mV": _mean(aged, "rmse_after_true_mV"),
        "rmse_after_nom_mV": _mean(nom, "rmse_after_true_mV"),
        "n_update_aged": _mean(aged, "n_update"),
        "n_update_nom": _mean(nom, "n_update"),
    }
    if (not aged and nom) or (aged and not nom):
        group = nom if nom else aged
        k0 = np.array([g["k_at_mean"][0] for g in group], dtype=float)
        k1 = np.array([g["k_at_mean"][1] for g in group], dtype=float)
        out.update(
            {
                "k0_p05": float(np.percentile(k0, 5)),
                "k0_p50": float(np.percentile(k0, 50)),
                "k0_p95": float(np.percentile(k0, 95)),
                "k1_p50": float(np.percentile(k1, 50)),
                "dk0_median": float(np.median(k0 - 1.0)),
                "dk1_median": float(np.median(k1 - 1.0)),
                "frac_k0_gt_1": float(np.mean(k0 > 1.0)),
                "n_k0_in_1pm03": int(np.sum(np.abs(k0 - 1.0) <= 0.03)),
                "n_k0_up": int(np.sum(k0 > 1.03)),
                "n_k0_dn": int(np.sum(k0 < 0.97)),
            }
        )
        if group and "d_r0_end_uOhm" in group[0]:
            dr = np.array([g["d_r0_end_uOhm"] for g in group], dtype=float)
            out["d_r0_p50_uOhm"] = float(np.percentile(dr, 50))
            out["d_r0_mean_uOhm"] = float(np.mean(dr))
        if group and "s_end_post_err_pp" in group[0]:
            se = np.array([g["s_end_post_err_pp"] for g in group], dtype=float)
            out["s_post_err_p50_pp"] = float(np.median(se))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="包级 k 网格 + 包级门")
    p.add_argument("--pack-dir", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--mlp-dir", default="Data/ai_mlp")
    p.add_argument("--mode", default="kgrid", choices=["freeze", "kgrid", "kgrid-nogate", "ifx_demo"])
    p.add_argument("--phi-dir", default="Data/ai_mlp_h8")
    p.add_argument("--demo-lr", type=float, default=0.02, help="ifx_demo 0.1 s SGD；不是 kgrid 的 lr=10")
    p.add_argument("--demo-e-thr", type=float, default=0.010, help="|e_ol| 死区 / V")
    p.add_argument("--dr-max", type=float, default=2.0e-3)
    p.add_argument("--win", type=int, default=100)
    p.add_argument("--lr", type=float, default=10.0)
    p.add_argument("--smooth", type=float, default=1.0e-3)
    p.add_argument("--i-edge", type=float, default=20.0)
    p.add_argument("--rest-s", type=float, default=3.0)
    p.add_argument("--rest-eps", type=float, default=1.0)
    p.add_argument("--e-ol-min", type=float, default=0.002)
    p.add_argument("--k-lo", type=float, default=0.5)
    p.add_argument("--k-hi", type=float, default=2.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dr0", action="store_true", help="EKF 慢变 δR0（2D）")
    p.add_argument("--r0-scale", type=float, default=1.0, help="放大 MLP 读出的 R0，2D 用 1.06")
    p.add_argument(
        "--k-after-trips",
        type=int,
        default=1,
        help="第 N 趟才写 k（2D 主档 3；1 是把这一趟升级成老化的失败对照）",
    )
    args = p.parse_args()
    set_seed(args.seed)

    pack_dir = _resolve(args.pack_dir)
    out_dir = _resolve(args.out_dir or (str(pack_dir) + "_" + args.mode))
    _guard_out(out_dir)
    mlp_dir = _resolve(args.mlp_dir)
    meta, data = load_pack(pack_dir)
    cells = meta["cells"]
    n = int(meta["n"])
    print(
        f"pack {meta['exp']} n={n} engine={meta.get('engine')} b_I={meta.get('b_I')}  "
        f"mode={args.mode} mlp={mlp_dir}  r0_scale={args.r0_scale:g} dr0={int(args.dr0)}  "
        f"k_after_trips={args.k_after_trips}",
        flush=True,
    )

    device = torch.device(args.device)
    ckpt = mlp_dir / "best.pt"
    base, scaler, cfg = load_bundle(ckpt, mlp_dir / "config.json", mlp_dir / "scaler.json")
    for par in base.parameters():
        par.requires_grad_(False)
    base = base.to(device).eval()
    provider = MlpParamProvider(base, scaler, device=device, r0_scale=args.r0_scale)
    kf_cfg = KfConfig(estimate_dr0=bool(args.dr0))

    ocv_fn, docv_fn = ocv_nmc, docv_ds
    if str(meta.get("engine", "ecm")).lower() == "pybamm":
        ocv_fn, docv_fn = pack_ocv_fn(data)
        print("OCV: PyBaMM bulk（本包标定，不混仓库 ocv.py）", flush=True)

    # 1) 冻结 EKF：Δs 给包级门，s_ah 给增量
    logs = []
    ds_last = np.empty(n)
    ds_traj = []
    for i, cell in enumerate(cells):
        log = run_cell_filter(
            provider, data, i, cell["soc0"], ocv=ocv_fn, docv=docv_fn, cfg=kf_cfg
        )
        logs.append(log)
        ds = log["soc_post"] - log["soc_ah"]
        ds_traj.append(ds)
        ds_last[i] = float(ds[-1])
        m = filter_metrics(log)
        s_err = m.get("s_end_post_err", float("nan"))
        print(
            f"  filt {i:03d} aged={int(cell['aged'])}  e_ol={m['e_ol_rmse_mV']:.2f} mV  "
            f"Δs={ds_last[i]*1e2:+.3f} pp  s_post_err={s_err*1e2:+.3f} pp  "
            f"δR0={m['d_r0_end_uOhm']:+.1f} µΩ",
            flush=True,
        )
    ds_mat = np.stack(ds_traj, axis=1)
    gate = pack_gate(ds_mat, dt_s=float(meta["dt_s"]))
    print(
        f"pack_gate blocked={gate['blocked']} reason={gate['reason']}  "
        f"m={gate['m']*1e2:.3f} pp  f_same={gate['f_same']:.2f}  "
        f"slope={gate.get('slope_pph', float('nan')):.3f} pp/h",
        flush=True,
    )

    rows = []
    nogate = args.mode == "kgrid-nogate"
    do_k = args.mode in {"kgrid", "kgrid-nogate"}
    do_ifx_demo = args.mode == "ifx_demo"
    frozen_model = KGridAdapter(base, r0_scale=args.r0_scale).to(device)
    trips = [int(x) for x in meta.get("trips", [0])]
    phi = None
    if do_ifx_demo:
        phi = _load_phi(_resolve(args.phi_dir), base, scaler, cfg, device)
    for i, cell in enumerate(cells):
        seq_true = cell_seq(
            data, i, soc=data["soc_true"][:, i], name=f"c{i:03d}_true", ocv=ocv_fn
        )
        seq_ah = cell_seq(
            data, i, soc=logs[i]["soc_ah"], name=f"c{i:03d}_ah", ocv=ocv_fn
        )
        rmse0 = overall_rmse(frozen_model, [seq_true], scaler, cfg, device)
        fm = filter_metrics(logs[i])
        row = {
            "id": i,
            "aged": cell["aged"],
            "k_true": cell["k"],
            "soc0": cell["soc0"],
            "t_c": cell["t_c"],
            "e_ol_rmse_mV": fm["e_ol_rmse_mV"],
            "e_pri_rmse_mV": fm["e_pri_rmse_mV"],
            "ds_end": float(ds_last[i]),
            "d_r0_end_uOhm": fm["d_r0_end_uOhm"],
            "d_r0_mean_uOhm": fm["d_r0_mean_uOhm"],
            "d_r0_i_uOhm": fm["d_r0_i_uOhm"],
            "s_end_post_err_pp": float(fm.get("s_end_post_err", float("nan")) * 1e2),
            "rmse_frozen_true_mV": rmse0 * 1e3,
            "k_at_ref": [1.0, 1.0],
            "k_at_mean": [1.0, 1.0],
        }
        if do_k:
            model = KGridAdapter(base, r0_scale=args.r0_scale).to(device)
            kg = run_cell_kgrid(
                model,
                seq_ah,
                scaler,
                cfg,
                device,
                args,
                pack_blocked=bool(gate["blocked"]),
                nogate=nogate,
                trips=trips,
                k_after_trips=int(args.k_after_trips),
            )
            row.update(kg)
            rmse1 = overall_rmse(model, [seq_true], scaler, cfg, device)
            row["rmse_after_true_mV"] = rmse1 * 1e3
            print(
                f"  kgrid {i:03d} aged={int(cell['aged'])}  "
                f"k_ref=({kg['k_at_ref'][0]:.3f},{kg['k_at_ref'][1]:.3f})  "
                f"upd={kg['n_update']}/{kg['n_win']}  skip_trip={kg['n_skip_trip']}  "
                f"rmse {rmse0*1e3:.2f}→{rmse1*1e3:.2f} mV",
                flush=True,
            )
        if do_ifx_demo:
            model = ResidualHeadAdapter(base, phi, dr_max=args.dr_max).to(device)
            dg = run_cell_demo(
                model,
                seq_ah,
                scaler,
                cfg,
                device,
                e_thr=args.demo_e_thr,
                lr=args.demo_lr,
                grad_clip=args.grad_clip,
            )
            row.update(dg)
            rmse1 = _rmse_head(model, seq_true, scaler, cfg, device)
            row["rmse_after_true_mV"] = rmse1 * 1e3
            print(
                f"  ifx_demo {i:03d} aged={int(cell['aged'])}  "
                f"k_mean=({dg['k_at_mean'][0]:.3f},{dg['k_at_mean'][1]:.3f})  "
                f"upd={dg['n_update']}/{dg['n_win']}  "
                f"rmse {rmse0*1e3:.2f}→{rmse1*1e3:.2f} mV",
                flush=True,
            )
        rows.append(row)

    summary = summarize_cells(cells, rows) if (do_k or do_ifx_demo) else {}
    if not summary:
        aged = [r for r, c in zip(rows, cells) if c["aged"]]
        nom = [r for r, c in zip(rows, cells) if not c["aged"]]
        summary = {
            "n_aged": len(aged),
            "n_nom": len(nom),
            "rmse_ol_aged_mV": float(np.mean([g["e_ol_rmse_mV"] for g in aged])) if aged else float("nan"),
            "rmse_ol_nom_mV": float(np.mean([g["e_ol_rmse_mV"] for g in nom])) if nom else float("nan"),
            "rmse_frozen_aged_mV": float(np.mean([g["rmse_frozen_true_mV"] for g in aged])) if aged else float("nan"),
            "rmse_frozen_nom_mV": float(np.mean([g["rmse_frozen_true_mV"] for g in nom])) if nom else float("nan"),
        }
    dr_end = np.array([r["d_r0_end_uOhm"] for r in rows], dtype=float)
    dr_i = np.array([r["d_r0_i_uOhm"] for r in rows], dtype=float)
    dr_m = np.array([r["d_r0_mean_uOhm"] for r in rows], dtype=float)
    se = np.array([r["s_end_post_err_pp"] for r in rows], dtype=float)
    summary["d_r0_p50_uOhm"] = float(np.percentile(dr_end, 50))
    summary["d_r0_mean_uOhm"] = float(np.median(dr_m))
    summary["d_r0_i_p50_uOhm"] = float(np.nanmedian(dr_i))
    summary["s_post_err_p50_pp"] = float(np.median(se))
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": args.mode,
        "pack_dir": _rel(pack_dir),
        "mlp_dir": _rel(mlp_dir),
        "exp": meta["exp"],
        "engine": meta.get("engine"),
        "n": n,
        "b_I": meta.get("b_I"),
        "r0_scale": args.r0_scale,
        "dr0": bool(args.dr0),
        "k_after_trips": int(args.k_after_trips),
        "optimizer": "SGD",
        "replay": False,
        "read_old_grid": False,
        "pack_gate": gate,
        "cells": [
            {k: v for k, v in r.items() if k not in {"k_tables", "hit0", "hit1"}}
            for r in rows
        ],
        "summary": summary,
        "win": args.win,
        "lr": args.lr,
        "phi_dir": _rel(_resolve(args.phi_dir)) if do_ifx_demo else None,
        "demo_lr": args.demo_lr if do_ifx_demo else None,
        "demo_e_thr_mV": args.demo_e_thr * 1e3 if do_ifx_demo else None,
    }
    (out_dir / "pack_run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if do_k or do_ifx_demo:
        torch.save(
            {f"cell_{r['id']}": {"k_at_ref": r["k_at_ref"], "k_tables": r.get("k_tables")} for r in rows},
            out_dir / "last.pt",
        )
        if "k0_p50" in summary:
            n_rep = summary["n_nom"] if summary.get("n_nom", 0) else summary.get("n_aged", n)
            print(
                f"summary  k0 p50={summary['k0_p50']:.3f}  Δk_med={summary['dk0_median']:+.3f}  "
                f"in_1±0.03={summary['n_k0_in_1pm03']}/{n_rep}  "
                f"up/dn={summary['n_k0_up']}/{summary['n_k0_dn']}  "
                f"δR0_I p50={summary['d_r0_i_p50_uOhm']:+.1f} µΩ  "
                f"s_err p50={summary['s_post_err_p50_pp']:+.3f} pp",
                flush=True,
            )
        else:
            print(
                f"summary  k0 aged/nom={summary['k0_aged']:.3f}/{summary['k0_nom']:.3f}  "
                f"k1 {summary['k1_aged']:.3f}/{summary['k1_nom']:.3f}",
                flush=True,
            )
    else:
        print(
            f"summary  δR0_I p50={summary['d_r0_i_p50_uOhm']:+.1f} µΩ  "
            f"δR0_end p50={summary['d_r0_p50_uOhm']:+.1f} µΩ  "
            f"s_err p50={summary['s_post_err_p50_pp']:+.3f} pp",
            flush=True,
        )
        if summary.get("n_aged", 0) == 0:
            print(
                f"summary  freeze RMSE mean="
                f"{summary['rmse_frozen_nom_mV']:.2f} mV",
                flush=True,
            )
        elif summary.get("n_nom", 0) == 0:
            print(
                f"summary  freeze RMSE mean="
                f"{summary['rmse_frozen_aged_mV']:.2f} mV",
                flush=True,
            )
        else:
            print(
                f"summary  freeze RMSE aged/nom="
                f"{summary['rmse_frozen_aged_mV']:.2f}/{summary['rmse_frozen_nom_mV']:.2f} mV",
                flush=True,
            )
    print(f"写出 {out_dir / 'pack_run.json'}", flush=True)

    # 2A1 烟测断言（数字作废，只拦脚本写错）
    if meta["exp"] == "2a1" and do_k and n <= 8:
        if gate["blocked"]:
            print("WARN 2A1 包级门不应触发")
        if summary["k0_nom"] > 1.08:
            print(f"WARN 未涨芯 k0={summary['k0_nom']:.3f} 偏高（期望 ~1）")
        if summary["k0_aged"] < summary["k0_nom"] + 0.015:
            print(
                f"WARN 涨阻芯 k0={summary['k0_aged']:.3f} 相对未涨 "
                f"{summary['k0_nom']:.3f} 没朝 1.15 走（看的是点到的节点，不是 0.5/25 参考点）"
            )
    if meta["exp"] in {"2a3", "2a4"} and do_k:
        if gate["blocked"]:
            print(f"WARN {meta['exp']} 包级门不应触发")
        if summary.get("n_k0_up", 0) == n or summary.get("n_k0_dn", 0) == n:
            print(f"WARN {meta['exp']} 全包同号，抽签或门可能写错")
    if meta["exp"] == "2b":
        if do_k and not nogate:
            if not gate["blocked"]:
                print("WARN 2B 包级门应触发")
            if summary.get("n_k0_in_1pm03", n) < n:
                print(
                    f"WARN 2B 主档 k 应保持 1（in_1±0.03="
                    f"{summary.get('n_k0_in_1pm03')}/{n}）"
                )
        if do_ifx_demo:
            n_side = max(summary.get("n_k0_up", 0), summary.get("n_k0_dn", 0))
            if n_side < max(1, int(0.8 * n)):
                print(
                    f"WARN 2B ifx_demo 应全包同号大改 k（up/dn="
                    f"{summary.get('n_k0_up')}/{summary.get('n_k0_dn')}）"
                )
    if meta["exp"] == "2e":
        if do_k and not nogate:
            if gate["blocked"]:
                print("WARN 2E 包级门不应触发")
            k50 = float(summary.get("k0_p50", summary.get("k0_aged", 1.0)))
            if k50 < 1.015:
                print(f"WARN 2E 主档 k 应朝 1.15（k0 p50={k50:.3f}）")
    if str(meta["exp"]).startswith("2d"):
        if gate["blocked"]:
            print(f"WARN {meta['exp']} 包级门不应触发")
        if abs(args.r0_scale - 1.06) > 1e-3:
            print(f"WARN {meta['exp']} 应用 --r0-scale 1.06（现在 {args.r0_scale:g}）")
        if args.dr0 and summary.get("d_r0_i_p50_uOhm", 0.0) > -20.0:
            print(
                f"WARN {meta['exp']} 开 δR0 大电流段应朝负（|I|≥20 A p50="
                f"{summary.get('d_r0_i_p50_uOhm'):+.1f} µΩ）"
            )
        if args.dr0 and int(args.k_after_trips) >= 3 and do_k:
            if summary.get("n_k0_in_1pm03", n) < n:
                print(
                    f"WARN {meta['exp']} 主档第 3 趟前 k 应保持 1（in_1±0.03="
                    f"{summary.get('n_k0_in_1pm03')}/{n}）"
                )


if __name__ == "__main__":
    main()
