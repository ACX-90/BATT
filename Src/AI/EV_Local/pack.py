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
    if exp == "2g" and abs(b_i) > 1e-6:
        raise RuntimeError("2g 必须 b_I=0（不要和零偏叠）")
    if exp == "2c" and abs(b_i - 5.0) > 1e-6:
        raise RuntimeError("2c 必须 b_I=5")
    if exp == "2e" and abs(b_i) > 1e-6:
        raise RuntimeError("2e 必须 b_I=0")
    if exp.startswith("2d") and abs(b_i) > 1e-6:
        raise RuntimeError(f"{exp} 必须 b_I=0")
    if exp in {"2i1", "2i2", "2i3"}:
        if abs(b_i) > 1e-6:
            raise RuntimeError(f"{exp} 必须 b_I=0（不要叠 2B）")
    if exp in {"2h1", "2h2", "2h3"}:
        if abs(b_i) > 1e-6:
            raise RuntimeError(f"{exp} 必须 b_I=0（不要叠 2B）")
        if "i_cell" not in data:
            raise RuntimeError(f"{exp} pack.npz 缺 i_cell（AFE 真电流）")
        i_meas = np.asarray(data["i_meas"], dtype=float)
        i_cell = np.asarray(data["i_cell"], dtype=float)
        if exp == "2h3":
            n_chg = int(meta.get("n_charge_steps", 0))
            if n_chg < 2:
                raise RuntimeError("2h3 缺 n_charge_steps（应先 1C 充再停）")
            # 充段 I_meas≈−100 A；停放段分流器≈0，I_cell≈12 mA
            i_m_chg = float(np.median(i_meas[:n_chg]))
            i_m_park = abs(float(np.median(i_meas[n_chg:])))
            i_c_park = float(np.median(i_cell[n_chg:]))
            if i_m_chg > -50.0:
                raise RuntimeError(f"2h3 充段 I_meas 中位 {i_m_chg:.2f} A，应≈−100 A")
            if i_m_park > 0.05:
                raise RuntimeError(f"2h3 停放 I_meas 中位 {i_m_park:.4f} A，应≈0")
            if i_c_park < 0.005:
                raise RuntimeError(f"2h3 停放 I_cell 中位 {i_c_park:.4f} A，应有 ~12 mA AFE")
        else:
            # 分流器测量不得偷带 AFE
            i_m = abs(float(np.median(i_meas)))
            i_c = float(np.median(i_cell))
            if i_m > 0.05:
                raise RuntimeError(f"{exp} I_meas 中位 {i_m:.4f} A，停放应≈0")
            if i_c < 0.005:
                raise RuntimeError(f"{exp} I_cell 中位 {i_c:.4f} A，应有 ~12 mA AFE")
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


def trip_pack_gates(
    ds_mat: np.ndarray,
    trips: list[int],
    n_steps: int,
    *,
    dt_s: float,
) -> list[dict]:
    """按趟估包级门。2C：放电趟拦、脉冲趟不把整段 |m| 闩死（06-a §3.4）。"""
    starts = [int(x) for x in (trips or [0])]
    out: list[dict] = []
    n = int(n_steps)
    for i, t0 in enumerate(starts):
        t1 = starts[i + 1] if i + 1 < len(starts) else n
        lo, hi = max(int(t0), 0), max(int(t1), int(t0) + 1)
        g = pack_gate(ds_mat[lo:hi], dt_s=dt_s)
        g["trip"] = i
        g["start"] = lo
        g["end"] = min(hi, n)
        out.append(g)
    return out


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
    trip_blocked: list[bool] | None = None,
    disable_park_gate: bool = False,
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
    # 2H 快路径：全程无边沿 + 主档停放门 → 全部 park skip，不必逐步 SGD
    min_len = max(win // 4, max(1, int(round(2.0 / float(dt_s)))))
    if (
        not disable_park_gate
        and not nogate
        and not pack_blocked
        and float(np.max(np.abs(i_np_all))) < float(args.rest_eps)
    ):
        n_win = max(1, (n + win - 1) // win)
        # 末窗太短时与主循环一致少计（≥2 s 墙钟，兼容 dt=5 s 停放）
        if n - (n_win - 1) * win < min_len and n_win > 1:
            n_win -= 1
        n_skip_park = n_win
        k_ref = model.k_at(0.50, 25.0)
        k_mean = model.k_at(float(seq["soc"].mean()), float(seq["t"].mean()))
        return {
            "n_win": n_win,
            "n_update": 0,
            "n_skip_gate": 0,
            "n_skip_pack": 0,
            "n_skip_park": n_skip_park,
            "n_skip_trip": 0,
            "k_at_ref": list(k_ref),
            "k_at_mean": list(k_mean),
            "k_tables": model.k_tables(),
            "hit0": hit0.tolist(),
            "hit1": hit1.tolist(),
        }
    # 2H3：前缀有大电流、其后长时间 |I|≈0 → 只逐步走充段，停放整段记 skip_park
    if (
        not disable_park_gate
        and not nogate
        and not pack_blocked
        and float(np.max(np.abs(i_np_all))) >= float(args.i_edge)
    ):
        abs_i = np.abs(i_np_all)
        # 最后一次 |I|≥边沿 之后的下标
        big = np.flatnonzero(abs_i >= float(args.i_edge))
        if big.size > 0:
            park0 = int(big[-1]) + 1
            # 停放占比够大才走快路径（避免误伤 2C 等）
            if park0 < n and (n - park0) >= int(0.8 * n) and (n - park0) * dt_s >= 1800.0:
                # 先逐步处理 [0, park0)
                n_win = n_upd = n_skip_gate = n_skip_pack = n_skip_park = n_skip_trip = 0
                kg_args = _kgrid_args(args)
                u_p0_local = u_p0
                hold_until = max(int(k_after_trips) - 1, 0)
                trip_starts = [int(x) for x in (trips or [0])]
                for start in range(0, park0, win):
                    end = min(start + win, park0)
                    if end - start < min_len:
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
                    ti = _trip_index(start, trip_starts)
                    if ti < hold_until:
                        n_skip_trip += 1
                        u_p0_local = _roll_up(model, sl, u_p0_local, dt_s, len(i_np) // 2)
                        continue
                    if (not disable_park_gate) and (not pol["write_k"]):
                        n_skip_park += 1
                        u_p0_local = _roll_up(model, sl, u_p0_local, dt_s, len(i_np) // 2)
                        continue
                    if not pol["allow_k1"]:
                        kg_args.rest_s = 1e9
                    else:
                        kg_args.rest_s = args.rest_s
                    st = step_window(
                        model,
                        opt,
                        sl,
                        u_p0_local,
                        dt_s=dt_s,
                        i_np=i_np,
                        args=kg_args,
                        i_prev=i_prev,
                        hit0=hit0,
                        hit1=hit1,
                    )
                    u_p0_local = st["u_p0"]
                    if st["updated"]:
                        n_upd += 1
                    else:
                        n_skip_gate += 1
                # 停放段整段 skip_park
                n_park_win = max(0, (n - park0 + win - 1) // win)
                if n - park0 - (n_park_win - 1) * win < min_len and n_park_win > 1:
                    n_park_win -= 1
                n_win += n_park_win
                n_skip_park += n_park_win
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
    min_len = max(win // 4, max(1, int(round(2.0 / float(dt_s)))))
    for start in range(0, n, win):
        end = min(start + win, n)
        if end - start < min_len:
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
        ti = _trip_index(start, trip_starts)
        blocked_now = bool(pack_blocked)
        if trip_blocked is not None and ti < len(trip_blocked):
            blocked_now = bool(trip_blocked[ti])
        if blocked_now and not nogate:
            n_skip_pack += 1
            u_p0 = _roll_up(model, sl, u_p0, dt_s, len(i_np) // 2)
            continue
        if ti < hold_until:
            n_skip_trip += 1
            u_p0 = _roll_up(model, sl, u_p0, dt_s, len(i_np) // 2)
            continue
        if (not disable_park_gate) and (not pol["write_k"]):
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
        "--capacity-scale",
        type=float,
        default=None,
        help="EKF/Ah 容量乘子（对齐 KF/run.py）。默认读 pack.json；2G=0.95",
    )
    p.add_argument(
        "--capacity-ah",
        type=float,
        default=None,
        help="规格书容量 Ah；默认 100 或 pack.json capacity_ah_true",
    )
    p.add_argument(
        "--k-after-trips",
        type=int,
        default=1,
        help="第 N 趟才写 k（2D 主档 3；1 是把这一趟升级成老化的失败对照）",
    )
    p.add_argument(
        "--disable-park-gate",
        action="store_true",
        help="关掉 §3.5 无边沿停放门（2H 失败对照：只留 ≥3 s 静置门）",
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
    q_true = float(
        args.capacity_ah
        if args.capacity_ah is not None
        else meta.get("capacity_ah_true", meta.get("capacity_ah", 100.0))
    )
    if args.capacity_scale is not None:
        cap_scale = float(args.capacity_scale)
    elif "capacity_scale" in meta:
        cap_scale = float(meta["capacity_scale"])
    elif "hat_q_ah" in meta:
        cap_scale = float(meta["hat_q_ah"]) / q_true
    else:
        cap_scale = 1.0
    hat_q = float(meta.get("hat_q_ah", q_true * cap_scale))
    if str(meta.get("exp")) == "2g" and abs(cap_scale - 0.95) > 1e-3:
        raise RuntimeError(f"2g 需要 capacity_scale=0.95（现在 {cap_scale:g}）")
    dt_pack = float(meta.get("dt_s", 0.1))
    if "suggested_win" in meta and int(args.win) == 100 and abs(dt_pack - 0.1) > 1e-9:
        args.win = int(meta["suggested_win"])
        print(f"win←pack.json suggested_win={args.win}（保持 ~10 s 墙钟）", flush=True)
    print(
        f"pack {meta['exp']} n={n} engine={meta.get('engine')} b_I={meta.get('b_I')}  "
        f"mode={args.mode} mlp={mlp_dir}  r0_scale={args.r0_scale:g} dr0={int(args.dr0)}  "
        f"k_after_trips={args.k_after_trips}  disable_park={int(args.disable_park_gate)}  "
        f"dt={dt_pack:g}s win={args.win}  "
        f"Q={q_true:g}Ah hatQ={hat_q:g}Ah scale={cap_scale:g}",
        flush=True,
    )

    device = torch.device(args.device)
    ckpt = mlp_dir / "best.pt"
    base, scaler, cfg = load_bundle(ckpt, mlp_dir / "config.json", mlp_dir / "scaler.json")
    for par in base.parameters():
        par.requires_grad_(False)
    base = base.to(device).eval()
    # 停放 1 s 采样：EKF / 滑窗离散必须跟 pack dt，不能死钉训练网格的 0.1 s
    cfg.dt_s = dt_pack
    provider = MlpParamProvider(base, scaler, device=device, r0_scale=args.r0_scale)
    # 对齐 KF/run.py：capacity_ah * capacity_scale 进 EKF/Ah 分母
    kf_cfg = KfConfig(
        estimate_dr0=bool(args.dr0),
        capacity_ah=q_true * cap_scale,
        dt_s=dt_pack,
    )

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
    trips = [int(x) for x in meta.get("trips", [0])]
    trip_gates = trip_pack_gates(
        ds_mat, trips, int(ds_mat.shape[0]), dt_s=float(meta["dt_s"])
    )
    trip_blocked = [bool(g["blocked"]) for g in trip_gates]
    if len(trip_gates) > 1:
        for g in trip_gates:
            print(
                f"  trip {g['trip']} [{g['start']}:{g['end']}]  "
                f"blocked={g['blocked']} reason={g['reason']}  "
                f"m={g['m']*1e2:.3f} pp  hours={g.get('hours', float('nan')):.2f}",
                flush=True,
            )

    rows = []
    nogate = args.mode == "kgrid-nogate"
    do_k = args.mode in {"kgrid", "kgrid-nogate"}
    do_ifx_demo = args.mode == "ifx_demo"
    frozen_model = KGridAdapter(base, r0_scale=args.r0_scale).to(device)
    phi = None
    if do_ifx_demo:
        phi = _load_phi(_resolve(args.phi_dir), base, scaler, cfg, device)
    for i, cell in enumerate(cells):
        seq_true = cell_seq(
            data, i, soc=data["soc_true"][:, i], name=f"c{i:03d}_true", ocv=ocv_fn
        )
        # 2C：放电段 s_ah 已被 5 A 零偏拉穿；脉冲写 k 用 s^-（休息电压已钉，06-a §3.4）。
        soc_k = logs[i]["soc_pred"] if str(meta["exp"]) == "2c" else logs[i]["soc_ah"]
        seq_ah = cell_seq(
            data, i, soc=soc_k, name=f"c{i:03d}_ah", ocv=ocv_fn
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
                trip_blocked=trip_blocked,
                disable_park_gate=bool(args.disable_park_gate),
            )
            row.update(kg)
            rmse1 = overall_rmse(model, [seq_true], scaler, cfg, device)
            row["rmse_after_true_mV"] = rmse1 * 1e3
            print(
                f"  kgrid {i:03d} aged={int(cell['aged'])}  "
                f"k_ref=({kg['k_at_ref'][0]:.3f},{kg['k_at_ref'][1]:.3f})  "
                f"upd={kg['n_update']}/{kg['n_win']}  "
                f"skip_pack={kg['n_skip_pack']} skip_park={kg['n_skip_park']}  "
                f"skip_trip={kg['n_skip_trip']}  "
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
    summary["d_r0_i_p50_uOhm"] = float(np.nanmedian(dr_i)) if np.any(np.isfinite(dr_i)) else float("nan")
    summary["s_post_err_p50_pp"] = float(np.median(se))
    if str(meta["exp"]) == "2c" and do_k:
        aged_rows = [r for r, c in zip(rows, cells) if c["aged"] and "k_tables" in r]
        nom_rows = [r for r, c in zip(rows, cells) if (not c["aged"]) and "k_tables" in r]

        def _node_k0(group: list[dict]) -> float:
            if not group:
                return float("nan")
            return float(np.mean([g["k_tables"]["k0"][0][2] for g in group]))

        summary["k0_s010_t30_aged"] = _node_k0(aged_rows)
        summary["k0_s010_t30_nom"] = _node_k0(nom_rows)
    # 2H：24 h 切片、s_ah 钉住、Up 终值
    slice_24h = None
    s_ah_pin = None
    up_end = None
    rebound = None
    if str(meta.get("exp", "")).startswith("2h"):
        idx24 = int(round(24.0 * 3600.0 / dt_pack))
        if ds_mat.shape[0] > idx24:
            g24 = pack_gate(ds_mat[: idx24 + 1], dt_s=dt_pack)
            slice_24h = {
                "index": idx24,
                "hours": 24.0,
                "m_pp": float(g24["m"] * 1e2),
                "f_same": float(g24["f_same"]),
                "slope_pph": float(g24.get("slope_pph", float("nan"))),
                "blocked": bool(g24["blocked"]),
                "reason": g24["reason"],
                "ds_p50_pp": float(np.median(ds_mat[idx24]) * 1e2),
                "ds_p05_pp": float(np.percentile(ds_mat[idx24], 5) * 1e2),
                "ds_p95_pp": float(np.percentile(ds_mat[idx24], 95) * 1e2),
            }
            print(
                f"slice_24h  m={slice_24h['m_pp']:+.3f} pp  blocked={slice_24h['blocked']}  "
                f"ds_p50={slice_24h['ds_p50_pp']:+.3f} pp",
                flush=True,
            )
        if str(meta.get("exp")) == "2h3":
            n_chg = int(meta.get("n_charge_steps", 0))
            s_ref = np.array([float(logs[i]["soc_ah"][n_chg]) for i in range(n)], dtype=float)
            s_ah_end = np.array([float(logs[i]["soc_ah"][-1]) for i in range(n)], dtype=float)
            s_ah_pin = {
                "scope": "park_only",
                "s_park0_p50": float(np.median(s_ref)),
                "s_ah_end_p50": float(np.median(s_ah_end)),
                "max_abs_drift_pp": float(np.max(np.abs(s_ah_end - s_ref)) * 1e2),
            }
            print(
                f"s_ah_pin(park)  max|Δ|={s_ah_pin['max_abs_drift_pp']:.4f} pp  "
                f"(停放段应≈0；充段安时会动)",
                flush=True,
            )
        else:
            s0 = np.array([float(c["soc0"]) for c in cells], dtype=float)
            s_ah_end = np.array([float(logs[i]["soc_ah"][-1]) for i in range(n)], dtype=float)
            s_ah_pin = {
                "s0_p50": float(np.median(s0)),
                "s_ah_end_p50": float(np.median(s_ah_end)),
                "max_abs_drift_pp": float(np.max(np.abs(s_ah_end - s0)) * 1e2),
            }
            print(
                f"s_ah_pin  max|Δ|={s_ah_pin['max_abs_drift_pp']:.4f} pp  "
                f"(应≈0，禁止把 12 mA 估进 hat b_I)",
                flush=True,
            )
        if "u_p" in data:
            up_last = np.asarray(data["u_p"][-1], dtype=float)
            up_end = {
                "p50_uV": float(np.median(up_last) * 1e6),
                "p05_uV": float(np.percentile(up_last, 5) * 1e6),
                "p95_uV": float(np.percentile(up_last, 95) * 1e6),
                "mean_uV": float(np.mean(up_last) * 1e6),
            }
            print(
                f"u_p_end  p50={up_end['p50_uV']:.2f} µV（久置稳态 ~I·R1，不是 0）",
                flush=True,
            )
        # 2H3：切停放后 Up 应从负（充电极化）往 0 回，不能像放电回弹（正→0）
        if str(meta.get("exp")) == "2h3" and "u_p" in data:
            n_chg = int(meta.get("n_charge_steps", 0))
            up = np.asarray(data["u_p"], dtype=float)
            if n_chg >= 2 and up.shape[0] > n_chg + 10:
                up_chg_end = up[n_chg - 1]
                # 停放后 30 min（或所能达到的长度）
                n_reb = min(up.shape[0] - n_chg, int(round(30 * 60 / dt_pack)))
                up_reb = up[n_chg : n_chg + n_reb]
                # 中位轨迹
                tr = np.median(up_reb, axis=1)
                up0 = float(np.median(up_chg_end))
                up_1min = float(tr[min(len(tr) - 1, max(1, int(round(60 / dt_pack))))])
                up_5min = float(tr[min(len(tr) - 1, max(1, int(round(300 / dt_pack))))])
                up_30min = float(tr[-1])
                # 充电回弹：Up0 < 0，随后单调往 0 抬（电压往下掉）
                charge_like = up0 < -0.01 and up_30min > up0 and up_30min < 0.005
                discharge_like = up0 > 0.01  # 错成放电支路建立正极化
                rebound = {
                    "n_reb_steps": int(n_reb),
                    "up_chg_end_p50_mV": up0 * 1e3,
                    "up_1min_p50_mV": up_1min * 1e3,
                    "up_5min_p50_mV": up_5min * 1e3,
                    "up_30min_p50_mV": up_30min * 1e3,
                    "charge_like": bool(charge_like),
                    "discharge_like_fail": bool(discharge_like),
                }
                print(
                    f"2h3_rebound  Up@切停={up0*1e3:+.2f} mV  "
                    f"1/5/30min={up_1min*1e3:+.2f}/{up_5min*1e3:+.2f}/{up_30min*1e3:+.2f} mV  "
                    f"charge_like={int(charge_like)} discharge_fail={int(discharge_like)}",
                    flush=True,
                )
                if discharge_like:
                    print("WARN 2H3 Up 切停后为正——像放电回弹，§3.6 可能没钉充电表", flush=True)
                elif not charge_like:
                    print("WARN 2H3 Up 回弹不像充电（应从负往 0）", flush=True)

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
        "capacity_ah_true": q_true,
        "hat_q_ah": hat_q,
        "capacity_scale": cap_scale,
        "dr0": bool(args.dr0),
        "k_after_trips": int(args.k_after_trips),
        "k_soc": "pred" if str(meta["exp"]) == "2c" else "ah",
        "optimizer": "SGD",
        "replay": False,
        "read_old_grid": False,
        "pack_gate": gate,
        "trip_gates": trip_gates,
        "cells": [
            {k: v for k, v in r.items() if k not in {"k_tables", "hit0", "hit1"}}
            for r in rows
        ],
        "summary": summary,
        "win": args.win,
        "lr": args.lr,
        "dt_s": dt_pack,
        "disable_park_gate": bool(args.disable_park_gate),
        "slice_24h": slice_24h,
        "s_ah_pin": s_ah_pin,
        "u_p_end": up_end,
        "rebound_2h3": rebound if str(meta.get("exp")) == "2h3" else None,
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
    if meta["exp"] == "2c" and do_k:
        print(
            f"summary  2C node s=0.10 T=30  k0 aged/nom="
            f"{summary.get('k0_s010_t30_aged', float('nan')):.3f}/"
            f"{summary.get('k0_s010_t30_nom', float('nan')):.3f}  "
            f"upd aged/nom={summary.get('n_update_aged', float('nan')):.1f}/"
            f"{summary.get('n_update_nom', float('nan')):.1f}",
            flush=True,
        )
        if not nogate:
            if not trip_blocked[0]:
                print("WARN 2C 放电段包级门应触发")
            if len(trip_blocked) > 1 and trip_blocked[1]:
                print("WARN 2C 脉冲段门不应再拦（第一列不估 b_I，按趟放行）")
            if summary.get("k0_s010_t30_nom", 1.0) > 1.08:
                print(
                    f"WARN 2C 未涨芯脉冲节点 k0={summary['k0_s010_t30_nom']:.3f} 偏高"
                )
            if summary.get("k0_s010_t30_aged", 1.0) < summary.get("k0_s010_t30_nom", 1.0) + 0.015:
                print(
                    f"WARN 2C 涨阻芯脉冲节点 k0={summary['k0_s010_t30_aged']:.3f} "
                    f"相对未涨 {summary.get('k0_s010_t30_nom', 1.0):.3f} 没朝 1.15 走"
                )
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
    if str(meta["exp"]).startswith("2h"):
        if do_k and not nogate and not args.disable_park_gate:
            if gate["blocked"]:
                print("WARN 2H 包级门不应触发（靠 §3.5，不是 1 pp 门）")
            n_park = int(np.mean([r.get("n_skip_park", 0) for r in rows]))
            n_upd = int(np.mean([r.get("n_update", 0) for r in rows]))
            if meta["exp"] in {"2h1", "2h2"} and n_upd > 0:
                print(f"WARN 2H 主档不应写 k（mean upd={n_upd}）")
            if meta["exp"] == "2h3" and n_upd > 30:
                print(
                    f"WARN 2H3 充段可写少量窗，但 mean upd={n_upd} 偏多（停放应 skip_park）"
                )
            if n_park < 1:
                print("WARN 2H 主档应走 n_skip_park（§3.5）")
            if summary.get("n_k0_in_1pm03", n) < n:
                print(
                    f"WARN 2H 主档 k 应保持 1（in_1±0.03="
                    f"{summary.get('n_k0_in_1pm03')}/{n}）"
                )
            # 2H1/2H2：全程钉住；2H3：安时在充段会动，只查停放段漂移在 gen 侧
            if meta["exp"] in {"2h1", "2h2"} and s_ah_pin and s_ah_pin["max_abs_drift_pp"] > 0.05:
                print(
                    f"WARN 2H s_ah 应钉在出发值（max|Δ|="
                    f"{s_ah_pin['max_abs_drift_pp']:.3f} pp）"
                )
        if do_k and args.disable_park_gate:
            print("NOTE 2H 无 §3.5：预期全包 k1 同号爬（失败对照）")
    if meta["exp"] == "2g":
        if abs(cap_scale - 0.95) > 1e-3:
            print(f"WARN 2G capacity_scale 应为 0.95（现在 {cap_scale:g}）")
        if args.dr0:
            print("WARN 2G 不要开 δR0 去跟容量斜坡")
        if do_k and not nogate:
            if not gate["blocked"]:
                print("WARN 2G 包级门应触发")
            if summary.get("n_k0_in_1pm03", n) < n:
                print(
                    f"WARN 2G 主档 k 应保持 1（in_1±0.03="
                    f"{summary.get('n_k0_in_1pm03')}/{n}）"
                )
    if meta["exp"] == "2e":
        if do_k and not nogate:
            if gate["blocked"]:
                print("WARN 2E 包级门不应触发")
            k50 = float(summary.get("k0_p50", summary.get("k0_aged", 1.0)))
            if k50 < 1.015:
                print(f"WARN 2E 主档 k 应朝 1.15（k0 p50={k50:.3f}）")
    if str(meta["exp"]).startswith("2i") and do_k and not nogate:
        # 06-a §5.9：b_I=0 → 写 k 的包门不拦；2I1 斜坡不写假 k0；2I2/2I3 仍要 2A 类隔离
        # 2I3 整段 ~24 min 可能踩中 ≥15 min 斜率备份（整轨报表）；写路径看按趟门
        if meta["exp"] == "2i3":
            if any(trip_blocked):
                print(
                    f"WARN 2i3 按趟包门不应触发（blocked trips="
                    f"{[i for i, b in enumerate(trip_blocked) if b]}）"
                )
            elif gate["blocked"]:
                print(
                    f"NOTE 2i3 整轨 gate blocked={gate['reason']} "
                    f"（≥15 min 斜率备份；写 k 已按趟放行）"
                )
        elif gate["blocked"]:
            print(f"WARN {meta['exp']} 包级门不应触发（b_I=0）")
        edge = meta.get("edge") or {}
        if edge:
            print(
                f"summary  2I edge_frac={edge.get('edge_frac', float('nan')):.6f} "
                f"({edge.get('n_edge', '?')}/{edge.get('n_di', '?')})  "
                f"max|ΔI|={edge.get('max_abs_di', float('nan')):.2f} A",
                flush=True,
            )
        if meta["exp"] == "2i1":
            # 斜坡爬升不开门：未涨芯不应被抖/假边沿抬走
            if summary.get("n_nom", 0) and float(summary.get("k0_nom", 1.0)) > 1.08:
                print(
                    f"WARN 2I1 未涨芯 k0={summary['k0_nom']:.3f} 偏高（斜坡不应假写 k0）"
                )
            n_edge = int(edge.get("n_edge", -1)) if edge else -1
            if n_edge > 0:
                print(f"NOTE 2I1 n_edge={n_edge}（预览期望 0；查 ramp 是否写错）")
        if meta["exp"] in {"2i2", "2i3"}:
            if summary.get("n_aged", 0) and summary.get("n_nom", 0):
                if float(summary.get("k0_aged", 1.0)) < float(summary.get("k0_nom", 1.0)) + 0.01:
                    print(
                        f"WARN {meta['exp']} 涨阻芯 k0={summary['k0_aged']:.3f} "
                        f"相对未涨 {summary['k0_nom']:.3f} 隔离偏弱"
                    )
                # 未涨芯 k0 中位应靠近 1（相对涨阻芯）
                if n >= 180 and float(summary.get("k0_nom", 1.0)) > 1.05:
                    print(
                        f"WARN {meta['exp']} 未涨芯 k0={summary['k0_nom']:.3f} 偏高"
                    )
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
