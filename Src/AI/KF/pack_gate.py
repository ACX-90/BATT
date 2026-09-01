"""包级同向门（Doc/06-a §3.1 / §3.5）。纯函数，写 k 之前拦。"""
from __future__ import annotations

import numpy as np


def delta_s(s_post: np.ndarray, s_ah: np.ndarray) -> np.ndarray:
    return np.asarray(s_post, dtype=float) - np.asarray(s_ah, dtype=float)


def pack_stats(ds: np.ndarray) -> dict[str, float]:
    """ds: (n_steps, n_cells) 或 (n_cells,) 末拍。"""
    a = np.asarray(ds, dtype=float)
    if a.ndim == 2:
        last = a[-1]
        # 放电段斜率：用整段对时间（步）的中位差，调用方也可自己报 pp/h
        slope = np.median(a[-1] - a[0])
    else:
        last = a
        slope = float("nan")
    m = float(np.median(last))
    if abs(m) < 1e-18:
        f_same = 1.0
    else:
        f_same = float(np.mean(np.sign(last) == np.sign(m)))
    return {"m": m, "f_same": f_same, "slope_steps": float(slope), "n": int(last.size)}


def pack_gate(
    ds: np.ndarray,
    *,
    dt_s: float = 0.1,
    m_pp: float = 0.01,
    f_same_min: float = 0.80,
    slope_pph: float = 0.005,
) -> dict:
    """|m|≥1 pp 且 f_same≥0.80，或斜率 ≥0.5 pp/h 且同号 → 拦。"""
    st = pack_stats(ds)
    a = np.asarray(ds, dtype=float)
    hours = 0.0
    slope = float("nan")
    if a.ndim == 2 and a.shape[0] >= 2:
        hours = max((a.shape[0] - 1) * dt_s / 3600.0, 1e-12)
        slope = float(np.median(a[-1] - a[0]) / hours)
    st["slope_pph"] = slope
    st["hours"] = hours
    # 2A SEQUENCE ~10 min：SPM / 2RC 残差会被 EKF 啃成 ~1 pp 同号 Δs，不是分流器。
    # 中位门和斜率门都要够长（≥15 min），2B 短波 / 小时级才开。
    block_level = (
        hours >= 0.25
        and abs(st["m"]) >= m_pp
        and st["f_same"] >= f_same_min
    )
    # 斜率备份给 2B 短波（≥15 min）；2A SEQUENCE ~10 min 上 EKF 啃边沿会假同号
    block_slope = (
        hours >= 0.25
        and np.isfinite(slope)
        and abs(slope) >= slope_pph
        and st["f_same"] >= f_same_min
    )
    blocked = bool(block_level or block_slope)
    reason = "ok"
    if block_level:
        reason = "median_ds"
    elif block_slope:
        reason = "slope"
    return {**st, "blocked": blocked, "reason": reason}


def last_edge_age_s(
    i_a: np.ndarray,
    end: int,
    *,
    dt_s: float,
    i_edge_a: float = 20.0,
    i_prev: float | None = None,
) -> float | None:
    """窗末往前最近一次 |ΔI|≥边沿 的年龄（秒）。从未有过则 None。"""
    i_a = np.asarray(i_a, dtype=float)
    left = i_a[: max(end, 0)]
    if i_prev is not None:
        series = np.concatenate(([float(i_prev)], left))
    else:
        series = left
    if series.size < 2:
        return None
    di = np.abs(np.diff(series))
    hits = np.flatnonzero(di >= i_edge_a)
    if hits.size == 0:
        return None
    last = int(hits[-1]) + 1  # 边沿落在 series 的右端点
    age_steps = series.size - 1 - last
    return float(age_steps * dt_s)


def window_policy(
    *,
    has_edge: bool,
    last_edge_age_s: float | None,
    rest_k1_horizon_s: float = 40.0,
    park_s: float = 60.0,
) -> dict:
    """停放无边沿不写 k；回弹只在 ~2τ1 内允许 k1。"""
    if not has_edge and (last_edge_age_s is None or last_edge_age_s > park_s):
        return {"write_k": False, "allow_k1": False, "reason": "park"}
    allow_k1 = bool(has_edge) or (
        last_edge_age_s is not None and last_edge_age_s <= rest_k1_horizon_s
    )
    return {"write_k": True, "allow_k1": allow_k1, "reason": "ok"}


def _self_test() -> None:
    rng = np.random.default_rng(0)
    n_c = 8
    n_s = int(20 * 60 / 0.1)
    # 全包同向 2 pp，20 min
    ds = np.linspace(0.0, 0.02, n_s)[:, None] + rng.normal(0, 1e-4, (n_s, n_c))
    g = pack_gate(ds, dt_s=0.1)
    assert g["blocked"] and g["reason"] == "median_ds", g
    # 符号混、中位 ~0
    ds2 = rng.normal(0, 0.002, (n_s, n_c))
    ds2[:, :4] += 0.004
    ds2[:, 4:] -= 0.004
    g2 = pack_gate(ds2, dt_s=0.1)
    assert not g2["blocked"], g2
    # 短波斜率：0.6 pp/h × 20 min
    n_short = int(20 * 60 / 0.1)
    ds3 = np.linspace(0.0, 0.006 * (20 / 60), n_short)[:, None] + rng.normal(0, 1e-5, (n_short, n_c))
    g3 = pack_gate(ds3, dt_s=0.1)
    assert g3["blocked"] and g3["reason"] == "slope", g3
    n_10 = int(10 * 60 / 0.1)
    ds4 = np.linspace(0.0, 0.0016, n_10)[:, None] + rng.normal(0, 1e-5, (n_10, n_c))
    g4 = pack_gate(ds4, dt_s=0.1)
    assert not g4["blocked"], g4
    pol = window_policy(has_edge=False, last_edge_age_s=120.0)
    assert pol["reason"] == "park" and not pol["write_k"]
    pol2 = window_policy(has_edge=False, last_edge_age_s=20.0)
    assert pol2["write_k"] and pol2["allow_k1"]
    pol3 = window_policy(has_edge=False, last_edge_age_s=50.0)
    assert pol3["write_k"] and not pol3["allow_k1"]
    print("pack_gate self-test ok")


if __name__ == "__main__":
    _self_test()
