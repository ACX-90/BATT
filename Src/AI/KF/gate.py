"""增量门控：健康滤波 + 开环有内容 + 激励足够 + 温度未出旧 scaler（Doc/06 §4.1）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from MLP.dataset import FeatureScaler

from .config import KfConfig


@dataclass
class GateResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)


def longest_rest_s(i_a: np.ndarray, dt_s: float, rest_eps: float) -> float:
    best = 0
    cur = 0
    for val in np.abs(i_a) <= rest_eps:
        if val:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best * dt_s


def gate_log(
    log: dict[str, np.ndarray],
    scaler: FeatureScaler,
    cfg: KfConfig,
    *,
    nis_lo: float = 0.05,
    nis_hi: float = 8.0,
    diverge_nis: float = 30.0,
    diverge_run: int = 50,
    noise_k: float = 3.0,
    i_edge_a: float = 20.0,
    rest_s: float = 30.0,
    rest_eps: float = 1.0,
    t_sigma: float = 4.0,
) -> GateResult:
    nis = np.asarray(log["nis"], dtype=float)
    e_ol = np.asarray(log["e_ol"], dtype=float)
    i_a = np.asarray(log.get("i_used_a", log["i_meas_a"]), dtype=float)
    t_c = np.asarray(log["t_meas_c"], dtype=float)

    nis_med = float(np.median(nis))
    rms_ol = float(np.sqrt(np.mean(e_ol**2)))
    run = 0
    max_run = 0
    for v in nis:
        if v > diverge_nis:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    di = np.abs(np.diff(i_a)) if len(i_a) > 1 else np.array([0.0])
    has_edge = bool(np.any(di >= i_edge_a))
    rest_len = longest_rest_s(i_a, cfg.dt_s, rest_eps)
    t_mean = float(scaler.mean[2])
    t_std = float(max(scaler.std[2], 1e-6))
    t_ok = bool(np.all(np.abs(t_c - t_mean) <= t_sigma * t_std))

    stats = {
        "nis_median": nis_med,
        "nis_max_run": float(max_run),
        "e_ol_rms_mV": rms_ol * 1e3,
        "has_edge": float(has_edge),
        "rest_s": rest_len,
        "t_min": float(np.min(t_c)),
        "t_max": float(np.max(t_c)),
        "t_ok": float(t_ok),
    }
    reasons: list[str] = []
    if not (nis_lo <= nis_med <= nis_hi):
        reasons.append(f"NIS 中位数 {nis_med:.2f} 不在 [{nis_lo}, {nis_hi}]")
    if max_run >= diverge_run:
        reasons.append(f"NIS 连续发散 {max_run} 步")
    if rms_ol < noise_k * cfg.rv_std:
        reasons.append(f"开环 RMSE {rms_ol*1e3:.2f} mV 未明显高于测量噪声")
    if not (has_edge or rest_len >= rest_s):
        reasons.append("没有足够电流边沿或 ≥30 s 静置，不拆 R0/R1")
    if not t_ok:
        reasons.append(f"温度超出旧 scaler μ±{t_sigma:g}σ，先当填洞而不是老化")

    return GateResult(accepted=not reasons, reasons=reasons, stats=stats)
