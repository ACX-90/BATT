"""NumPy 1RC/2RC open-loop helpers + segment RMSE (Phase-3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import lfilter

DT_S = 0.1
C1_STAR = 2.8e4


def load_traj(path: Path) -> dict:
    import csv

    with path.open("r", encoding="utf-8", newline="") as fh:
        body = (line for line in fh if line.strip() and not line.lstrip().startswith("#"))
        reader = csv.DictReader(body)
        cols: dict[str, list] = {n: [] for n in (reader.fieldnames or [])}
        for row in reader:
            for k in cols:
                cols[k].append(row[k])
    def f(name: str) -> np.ndarray:
        return np.asarray(cols[name], dtype=float)

    return {
        "name": path.name,
        "t": f("time_s"),
        "i": f("i_true_a"),
        "u": f("u_t_true_v"),
        "ocv": f("u_ocv_v"),
        "soc": f("soc_true"),
        "t_c": f("t_true_c"),
        "cmd": f("cmd_id").astype(int),
        "r0_csv": f("r0_ohm"),
        "r1_csv": f("r1_ohm"),
        "c1_csv": f("c1_f"),
    }


def rc_up(i: np.ndarray, r: float | np.ndarray, tau: float | np.ndarray, dt: float = DT_S) -> np.ndarray:
    i = np.asarray(i, dtype=float)
    if np.isscalar(r) and np.isscalar(tau):
        a = float(np.exp(-dt / max(float(tau), 1e-6)))
        return lfilter([float(r) * (1.0 - a)], [1.0, -a], i)
    r = np.asarray(r, dtype=float)
    tau = np.asarray(tau, dtype=float)
    u_p = 0.0
    out = np.empty_like(i)
    for k in range(i.size):
        a = float(np.exp(-dt / max(float(tau[k]), 1e-6)))
        u_p = a * u_p + float(r[k]) * (1.0 - a) * float(i[k])
        out[k] = u_p
    return out


def sim_1rc(
    i: np.ndarray,
    ocv: np.ndarray,
    r0: float | np.ndarray,
    r1: float | np.ndarray,
    c1: float | np.ndarray = C1_STAR,
    dt: float = DT_S,
) -> np.ndarray:
    r0 = np.asarray(r0, dtype=float)
    r1 = np.asarray(r1, dtype=float)
    c1 = np.asarray(c1, dtype=float)
    if r0.ndim == 0:
        r0 = np.full_like(i, float(r0), dtype=float)
    if r1.ndim == 0:
        r1 = np.full_like(i, float(r1), dtype=float)
    if c1.ndim == 0:
        c1 = np.full_like(i, float(c1), dtype=float)
    tau = np.maximum(r1 * c1, 1e-6)
    up = rc_up(i, r1, tau, dt)
    return ocv - i * r0 - up


def sim_2rc(
    i: np.ndarray,
    ocv: np.ndarray,
    r0: float,
    r1: float,
    tau1: float,
    r2: float,
    tau2: float,
    dt: float = DT_S,
) -> np.ndarray:
    up1 = rc_up(i, r1, tau1, dt)
    up2 = rc_up(i, r2, tau2, dt) if r2 > 0 else 0.0
    return ocv - i * float(r0) - up1 - up2


def rmse_mv(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(d * d)) * 1e3)


def segment_masks(cmd: np.ndarray, i: np.ndarray, dt: float = DT_S) -> dict[str, np.ndarray]:
    """Segment masks used in Phase-3 reports."""
    n = cmd.size
    edge = np.zeros(n, dtype=bool)
    di = np.diff(i, prepend=i[0])
    edge |= np.abs(di) > 20.0
    # expand edge ±0.5 s
    w = max(1, int(round(0.5 / dt)))
    edge_exp = edge.copy()
    idx = np.flatnonzero(edge)
    for k in idx:
        edge_exp[max(0, k - w) : min(n, k + w + 1)] = True

    rest = np.abs(i) < 1.0
    # rest 60–120 s into each rest block (cmd changes)
    rest_late = np.zeros(n, dtype=bool)
    k = 0
    while k < n:
        if not rest[k]:
            k += 1
            continue
        j = k
        while j < n and rest[j]:
            j += 1
        t_rel = (np.arange(j - k) * dt)
        late = (t_rel >= 60.0) & (t_rel <= 120.0)
        rest_late[k:j] = late
        k = j

    # first 30 s of each non-rest pulse
    pulse_early = np.zeros(n, dtype=bool)
    powered = ~rest
    k = 0
    while k < n:
        if not powered[k]:
            k += 1
            continue
        j = k
        while j < n and powered[j]:
            j += 1
        t_rel = np.arange(j - k) * dt
        pulse_early[k:j] = t_rel < 30.0
        k = j

    return {
        "all": np.ones(n, dtype=bool),
        "edge": edge_exp,
        "pulse_first_30s": pulse_early,
        "rest_60_120s": rest_late,
        "rest": rest,
        "dis1c": cmd == 1,
        "rest_after_1c": cmd == 2,
        "chg": cmd == 3,
        "dis2c": cmd == 5,
    }


def segment_rmse(u_hat: np.ndarray, u: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    out = {}
    for name, m in masks.items():
        if m.any():
            out[name] = rmse_mv(u_hat[m], u[m])
        else:
            out[name] = float("nan")
    return out


def nlinear(axes: list[np.ndarray], table: np.ndarray, query: np.ndarray, log_r: bool = True) -> np.ndarray:
    from itertools import product

    vals = np.log(np.clip(table, 1e-12, None)) if log_r else table
    ndim = len(axes)
    idx0: list[np.ndarray] = []
    w1: list[np.ndarray] = []
    for d, ax in enumerate(axes):
        step = float(ax[1] - ax[0])
        t = (query[:, d] - float(ax[0])) / step
        t = np.clip(t, 0.0, float(len(ax) - 1) - 1e-6)
        i0 = np.floor(t).astype(np.int32)
        idx0.append(i0)
        w1.append((t - i0).astype(np.float64))
    out = np.zeros((query.shape[0], vals.shape[-1]), dtype=np.float64)
    for bits in product((0, 1), repeat=ndim):
        w = np.ones(query.shape[0], dtype=np.float64)
        sl: list[np.ndarray] = []
        for d, bit in enumerate(bits):
            sl.append(idx0[d] + bit)
            wd = w1[d] if bit else (1.0 - w1[d])
            w = w * wd
        out += w[:, None] * vals[tuple(sl)]
    return np.exp(out) if log_r else out


def fit_ltis_1rc_c1star(
    i: np.ndarray,
    ocv: np.ndarray,
    u: np.ndarray,
    *,
    r0_hint: float,
    r1_hint: float,
    c1: float = C1_STAR,
) -> dict:
    """LTI 1RC with C1 pinned (scheme B)."""
    from scipy.optimize import least_squares

    r0_h = float(np.clip(r0_hint if np.isfinite(r0_hint) else 1e-3, 2e-4, 8e-3))
    r1_h = float(np.clip(abs(r1_hint) if np.isfinite(r1_hint) else 4e-4, 5e-5, 8e-3))
    z0 = np.array([np.log(r0_h), np.log(r1_h)], dtype=float)

    def unpack(z):
        r0 = float(np.clip(np.exp(z[0]), 1e-4, 1e-2))
        r1 = float(np.clip(np.exp(z[1]), 2e-5, 1e-2))
        return r0, r1

    def fun(z):
        r0, r1 = unpack(z)
        uh = sim_1rc(i, ocv, r0, r1, c1)
        return (uh - u) * 1e3

    res = least_squares(fun, z0, method="trf", max_nfev=200)
    r0, r1 = unpack(res.x)
    uh = sim_1rc(i, ocv, r0, r1, c1)
    return {
        "ok": bool(res.success),
        "r0_ohm": r0,
        "r1_ohm": r1,
        "c1_f": float(c1),
        "tau1_s": float(r1 * c1),
        "rmse_mv": rmse_mv(u, uh),
        "uh": uh,
    }


def fit_ltis_2rc(
    i: np.ndarray,
    ocv: np.ndarray,
    u: np.ndarray,
    *,
    r0_hint: float,
    r1_hint: float,
    tau1_hint: float,
    r2_hint: float,
    tau2_hint: float,
) -> dict:
    from scipy.optimize import least_squares

    r0_h = float(np.clip(r0_hint if np.isfinite(r0_hint) else 1e-3, 2e-4, 8e-3))
    r1_h = float(np.clip(abs(r1_hint) if np.isfinite(r1_hint) else 4e-4, 5e-5, 8e-3))
    t1_h = float(np.clip(tau1_hint if np.isfinite(tau1_hint) else 5.0, 0.5, 30.0))
    r2_h = float(np.clip(abs(r2_hint) if np.isfinite(r2_hint) else 3e-4, 5e-5, 8e-3))
    t2_h = float(np.clip(tau2_hint if np.isfinite(tau2_hint) else 90.0, 20.0, 350.0))
    z0 = np.array([np.log(r0_h), np.log(r1_h), np.log(t1_h), np.log(r2_h), np.log(t2_h)], dtype=float)

    def unpack(z):
        r0 = float(np.clip(np.exp(z[0]), 1e-4, 1e-2))
        r1 = float(np.clip(np.exp(z[1]), 2e-5, 1e-2))
        tau1 = float(np.clip(np.exp(z[2]), 0.3, 40.0))
        r2 = float(np.clip(np.exp(z[3]), 2e-5, 1e-2))
        tau2 = float(np.clip(max(np.exp(z[4]), 4.0 * tau1), 12.0, 400.0))
        return r0, r1, tau1, r2, tau2

    def fun(z):
        r0, r1, tau1, r2, tau2 = unpack(z)
        uh = sim_2rc(i, ocv, r0, r1, tau1, r2, tau2)
        return (uh - u) * 1e3

    res = least_squares(fun, z0, method="trf", max_nfev=250)
    r0, r1, tau1, r2, tau2 = unpack(res.x)
    uh = sim_2rc(i, ocv, r0, r1, tau1, r2, tau2)
    return {
        "ok": bool(res.success),
        "r0_ohm": r0,
        "r1_ohm": r1,
        "tau1_s": tau1,
        "r2_ohm": r2,
        "tau2_s": tau2,
        "rmse_mv": rmse_mv(u, uh),
        "uh": uh,
    }


def edge_r0(i: np.ndarray, u: np.ndarray, *, di_min: float = 20.0) -> float:
    di = np.diff(i)
    idx = np.where(np.abs(di) > di_min)[0] + 1
    ones = []
    for k in idx:
        d_i = float(i[k] - i[k - 1])
        if 70.0 <= abs(d_i) <= 130.0:
            ones.append(-(u[k] - u[k - 1]) / d_i)
    if not ones:
        for k in idx:
            d_i = float(i[k] - i[k - 1])
            if 40.0 <= abs(d_i) <= 250.0:
                ones.append(-(u[k] - u[k - 1]) / d_i)
    return float(np.mean(ones)) if ones else float("nan")
