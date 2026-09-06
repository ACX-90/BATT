"""Phase-2 2I shared I(t) builders (Doc/06-a §5.9).

Reused by nmc100ah_gen_pack (--exp 2i1/2i2/2i3) and preview scripts.
Does NOT touch nmc100ah_gen.SEQUENCE header. Jitter is profile shape, not NOISE_STD.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from nmc100ah_gen import DT_S, SEQUENCE, expand_sequence

QN_AH = 100.0
DT = float(DT_S)
EDGE_A = 20.0
RAMP_S = 2.0
# Preview-aligned jitter seeds (waveform shape); pack cell seeds are separate (213/214/215).
JITTER_SEED_2I2 = 20260906
JITTER_SEED_2I3 = 42
JITTER_RMS_A = 10.0


def steps_to_arrays(plan: list[tuple[int, str, float, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(plan)
    t = np.arange(n, dtype=float) * DT
    i = np.array([p[2] for p in plan], dtype=float)
    mode = np.array([p[1] for p in plan], dtype=object)
    return t, i, mode


def abs_di(i: np.ndarray) -> np.ndarray:
    d = np.empty_like(i)
    d[0] = 0.0
    d[1:] = np.abs(np.diff(i))
    return d


def edge_stats(i: np.ndarray) -> dict[str, float | int]:
    di = abs_di(i)
    n_di = int(max(len(di) - 1, 0))
    n_edge = int(np.sum(di[1:] >= EDGE_A)) if n_di else 0
    return {
        "edge_a": float(EDGE_A),
        "n_steps": int(len(i)),
        "n_di": n_di,
        "n_edge": n_edge,
        "edge_frac": float(n_edge / n_di) if n_di else 0.0,
        "max_abs_di": float(np.max(di[1:])) if n_di else 0.0,
        "peak_abs_i": float(np.max(np.abs(i))) if len(i) else 0.0,
        "duration_s": float(len(i) * DT),
    }


def bandlimited_jitter(n: int, rng: np.random.Generator, rms: float = JITTER_RMS_A) -> np.ndarray:
    """Band-limited jitter: typical |ΔI|/step < 20 A, RMS ≈ rms."""
    if n <= 0:
        return np.zeros(0, dtype=float)
    raw = rng.normal(0.0, 1.0, size=n)
    win = 11
    kernel = np.ones(win, dtype=float) / win
    smooth = np.convolve(raw, kernel, mode="same")
    smooth = np.convolve(smooth, kernel, mode="same")
    alpha = 0.25
    y = np.empty(n, dtype=float)
    y[0] = smooth[0]
    for k in range(1, n):
        y[k] = y[k - 1] + alpha * (smooth[k] - y[k - 1])
    cur = float(np.std(y))
    if cur < 1e-12:
        return np.zeros(n)
    y *= rms / cur
    di = np.diff(y, prepend=y[0])
    max_step = 18.0
    for _ in range(3):
        if not np.any(np.abs(di) > max_step):
            break
        di = np.clip(di, -max_step, max_step)
        y = np.cumsum(di)
        y -= np.mean(y)
        s = float(np.std(y))
        if s > 1e-12:
            y *= rms / s
        di = np.diff(y, prepend=y[0])
    return y


def build_2i1_ramp_sequence() -> list[dict]:
    """Default SEQUENCE shape; charge/discharge edges via dis_ramp/chg_ramp over 2 s."""
    ramp_steps_s = RAMP_S
    seq: list[dict] = []
    for cmd in SEQUENCE:
        mode = str(cmd["mode"]).strip().lower()
        dur = float(cmd["duration_s"])
        if mode in {"rest", "idle", "pause"}:
            seq.append(dict(cmd))
            continue
        if "c_rate" not in cmd:
            raise ValueError(f"expected c_rate on pulse: {cmd}")
        rate = float(cmd["c_rate"])
        if mode in {"discharge", "dch", "dis"}:
            ramp_up, ramp_dn, hold = "dis_ramp", "dis_ramp", "discharge"
        elif mode in {"charge", "chg", "cha"}:
            ramp_up, ramp_dn, hold = "chg_ramp", "chg_ramp", "charge"
        else:
            raise ValueError(mode)
        hold_s = max(dur - 2.0 * ramp_steps_s, DT)
        seq.append(
            {
                "mode": ramp_up,
                "duration_s": ramp_steps_s,
                "c_rate_start": 0.0,
                "c_rate_end": rate,
            }
        )
        seq.append({"mode": hold, "duration_s": hold_s, "c_rate": rate})
        seq.append(
            {
                "mode": ramp_dn,
                "duration_s": ramp_steps_s,
                "c_rate_start": rate,
                "c_rate_end": 0.0,
            }
        )
    return seq


def drive_2i3_pieces() -> list[dict]:
    """~24 min isothermal shared drive: ramps / regen / idle. Not WLTP."""
    return [
        {"mode": "rest", "duration_s": 20.0},
        {"mode": "dis_ramp", "duration_s": 8.0, "c_rate_start": 0.0, "c_rate_end": 0.4},
        {"mode": "discharge", "duration_s": 40.0, "c_rate": 0.4},
        {"mode": "dis_ramp", "duration_s": 5.0, "c_rate_start": 0.4, "c_rate_end": 0.15},
        {"mode": "discharge", "duration_s": 30.0, "c_rate": 0.15},
        {"mode": "dis_ramp", "duration_s": 3.0, "c_rate_start": 0.15, "c_rate_end": 0.0},
        {"mode": "rest", "duration_s": 15.0},
        {"mode": "dis_ramp", "duration_s": 4.0, "c_rate_start": 0.0, "c_rate_end": 1.2},
        {"mode": "discharge", "duration_s": 12.0, "c_rate": 1.2},
        {"mode": "dis_ramp", "duration_s": 6.0, "c_rate_start": 1.2, "c_rate_end": 0.5},
        {"mode": "discharge", "duration_s": 50.0, "c_rate": 0.5},
        {"mode": "dis_ramp", "duration_s": 2.0, "c_rate_start": 0.5, "c_rate_end": 0.0},
        {"mode": "chg_ramp", "duration_s": 3.0, "c_rate_start": 0.0, "c_rate_end": 0.6},
        {"mode": "charge", "duration_s": 8.0, "c_rate": 0.6},
        {"mode": "chg_ramp", "duration_s": 4.0, "c_rate_start": 0.6, "c_rate_end": 0.0},
        {"mode": "rest", "duration_s": 25.0},
        {"mode": "dis_ramp", "duration_s": 10.0, "c_rate_start": 0.0, "c_rate_end": 0.35},
        {"mode": "discharge", "duration_s": 80.0, "c_rate": 0.35},
        {"mode": "dis_ramp", "duration_s": 8.0, "c_rate_start": 0.35, "c_rate_end": 0.55},
        {"mode": "discharge", "duration_s": 60.0, "c_rate": 0.55},
        {"mode": "dis_ramp", "duration_s": 8.0, "c_rate_start": 0.55, "c_rate_end": 0.25},
        {"mode": "discharge", "duration_s": 70.0, "c_rate": 0.25},
        {"mode": "dis_ramp", "duration_s": 5.0, "c_rate_start": 0.25, "c_rate_end": 0.0},
        {"mode": "rest", "duration_s": 40.0},
        {"mode": "dis_ramp", "duration_s": 3.0, "c_rate_start": 0.0, "c_rate_end": 0.8},
        {"mode": "discharge", "duration_s": 6.0, "c_rate": 0.8},
        {"mode": "dis_ramp", "duration_s": 2.0, "c_rate_start": 0.8, "c_rate_end": 0.0},
        {"mode": "chg_ramp", "duration_s": 2.5, "c_rate_start": 0.0, "c_rate_end": 0.45},
        {"mode": "charge", "duration_s": 5.0, "c_rate": 0.45},
        {"mode": "chg_ramp", "duration_s": 2.0, "c_rate_start": 0.45, "c_rate_end": 0.0},
        {"mode": "rest", "duration_s": 12.0},
        {"mode": "dis_ramp", "duration_s": 3.0, "c_rate_start": 0.0, "c_rate_end": 0.7},
        {"mode": "discharge", "duration_s": 8.0, "c_rate": 0.7},
        {"mode": "dis_ramp", "duration_s": 2.0, "c_rate_start": 0.7, "c_rate_end": 0.0},
        {"mode": "chg_ramp", "duration_s": 2.0, "c_rate_start": 0.0, "c_rate_end": 0.5},
        {"mode": "charge", "duration_s": 4.0, "c_rate": 0.5},
        {"mode": "chg_ramp", "duration_s": 2.0, "c_rate_start": 0.5, "c_rate_end": 0.0},
        {"mode": "rest", "duration_s": 18.0},
        {"mode": "dis_ramp", "duration_s": 4.0, "c_rate_start": 0.0, "c_rate_end": 0.9},
        {"mode": "discharge", "duration_s": 10.0, "c_rate": 0.9},
        {"mode": "dis_ramp", "duration_s": 3.0, "c_rate_start": 0.9, "c_rate_end": 0.2},
        {"mode": "discharge", "duration_s": 25.0, "c_rate": 0.2},
        {"mode": "dis_ramp", "duration_s": 2.0, "c_rate_start": 0.2, "c_rate_end": 0.0},
        {"mode": "rest", "duration_s": 30.0},
        {"mode": "dis_ramp", "duration_s": 12.0, "c_rate_start": 0.0, "c_rate_end": 1.5},
        {"mode": "discharge", "duration_s": 90.0, "c_rate": 1.5},
        {"mode": "dis_ramp", "duration_s": 15.0, "c_rate_start": 1.5, "c_rate_end": 0.6},
        {"mode": "discharge", "duration_s": 120.0, "c_rate": 0.6},
        {"mode": "dis_ramp", "duration_s": 8.0, "c_rate_start": 0.6, "c_rate_end": 0.0},
        {"mode": "chg_ramp", "duration_s": 5.0, "c_rate_start": 0.0, "c_rate_end": 0.8},
        {"mode": "charge", "duration_s": 15.0, "c_rate": 0.8},
        {"mode": "chg_ramp", "duration_s": 6.0, "c_rate_start": 0.8, "c_rate_end": 0.0},
        {"mode": "rest", "duration_s": 45.0},
        {"mode": "dis_ramp", "duration_s": 6.0, "c_rate_start": 0.0, "c_rate_end": 0.3},
        {"mode": "discharge", "duration_s": 100.0, "c_rate": 0.3},
        {"mode": "dis_ramp", "duration_s": 10.0, "c_rate_start": 0.3, "c_rate_end": 0.45},
        {"mode": "discharge", "duration_s": 80.0, "c_rate": 0.45},
        {"mode": "dis_ramp", "duration_s": 8.0, "c_rate_start": 0.45, "c_rate_end": 0.1},
        {"mode": "discharge", "duration_s": 40.0, "c_rate": 0.1},
        {"mode": "dis_ramp", "duration_s": 4.0, "c_rate_start": 0.1, "c_rate_end": 0.0},
        {"mode": "rest", "duration_s": 60.0},
        {"mode": "dis_ramp", "duration_s": 5.0, "c_rate_start": 0.0, "c_rate_end": 1.8},
        {"mode": "discharge", "duration_s": 8.0, "c_rate": 1.8},
        {"mode": "dis_ramp", "duration_s": 4.0, "c_rate_start": 1.8, "c_rate_end": 0.0},
        {"mode": "chg_ramp", "duration_s": 3.0, "c_rate_start": 0.0, "c_rate_end": 0.7},
        {"mode": "charge", "duration_s": 10.0, "c_rate": 0.7},
        {"mode": "chg_ramp", "duration_s": 4.0, "c_rate_start": 0.7, "c_rate_end": 0.0},
        {"mode": "rest", "duration_s": 40.0},
        {"mode": "dis_ramp", "duration_s": 0.4, "c_rate_start": 0.0, "c_rate_end": 1.0},
        {"mode": "discharge", "duration_s": 6.0, "c_rate": 1.0},
        {"mode": "dis_ramp", "duration_s": 0.4, "c_rate_start": 1.0, "c_rate_end": 0.0},
        {"mode": "rest", "duration_s": 8.0},
        {"mode": "chg_ramp", "duration_s": 0.3, "c_rate_start": 0.0, "c_rate_end": 0.8},
        {"mode": "charge", "duration_s": 5.0, "c_rate": 0.8},
        {"mode": "chg_ramp", "duration_s": 0.3, "c_rate_start": 0.8, "c_rate_end": 0.0},
        {"mode": "rest", "duration_s": 60.0},
    ]


def _apply_jitter_on_mask(
    i: np.ndarray,
    mask: np.ndarray,
    *,
    rng: np.random.Generator,
    rms: float,
) -> np.ndarray:
    """Add band-limited jitter on contiguous True runs; skip first/last sample of each run."""
    jitter = np.zeros_like(i)
    n = len(i)
    k = 0
    while k < n:
        if not mask[k]:
            k += 1
            continue
        j = k
        while j < n and mask[j]:
            j += 1
        lo = k + 1
        hi = j - 1
        if hi > lo:
            jitter[lo:hi] = bandlimited_jitter(hi - lo, rng, rms=rms)
        k = j
    return i + jitter



def _trip_starts_after_rests(mode: np.ndarray, *, min_rest_s: float = 20.0) -> list[int]:
    """Trip starts at 0 and after each rest run lasting >= min_rest_s (if more samples follow)."""
    rest = np.isin(mode, ["rest", "idle", "pause"])
    trips = [0]
    n = len(mode)
    k = 0
    while k < n:
        if not rest[k]:
            k += 1
            continue
        j = k
        while j < n and rest[j]:
            j += 1
        dur = (j - k) * DT
        if dur >= float(min_rest_s) and j < n and j not in trips:
            trips.append(int(j))
        k = j
    return trips

def build_pack_wave(exp: str, *, dt_s: float = DT, capacity_ah: float = QN_AH) -> dict[str, Any]:
    """Return seq, i_true, modes, edge stats, and meta notes for gen_pack."""
    exp = str(exp).lower()
    if exp == "2i1":
        seq = build_2i1_ramp_sequence()
        plan = expand_sequence(seq, dt_s=dt_s, capacity_ah=capacity_ah, t_default=25.0)
        t, i, mode = steps_to_arrays(plan)
        stats = edge_stats(i)
        return {
            "seq": seq,
            "i_true": i,
            "mode": mode,
            "time_s": t,
            "wave": "sequence_ramp_2s",
            "jitter_seed": None,
            "jitter_rms_a": 0.0,
            "edge": stats,
            "note": (
                "2I1：默认 SEQUENCE 形状，充放沿用 dis_ramp/chg_ramp 0→target→0 共 2 s "
                f"（约 {EDGE_A/4:.0f} A/拍）；不改 SEQUENCE 头。边沿门仍 |ΔI|≥{EDGE_A:g} A/拍。"
            ),
        }
    if exp == "2i2":
        seq = list(SEQUENCE)
        plan = expand_sequence(seq, dt_s=dt_s, capacity_ah=capacity_ah, t_default=25.0)
        t, i_clean, mode = steps_to_arrays(plan)
        rng = np.random.default_rng(JITTER_SEED_2I2)
        dis = mode == "discharge"
        i = _apply_jitter_on_mask(i_clean, dis, rng=rng, rms=JITTER_RMS_A)
        stats = edge_stats(i)
        stats_clean = edge_stats(i_clean)
        return {
            "seq": seq,
            "i_true": i,
            "i_clean": i_clean,
            "mode": mode,
            "time_s": t,
            "wave": "sequence_discharge_jitter",
            "jitter_seed": JITTER_SEED_2I2,
            "jitter_rms_a": float(JITTER_RMS_A),
            "edge": stats,
            "edge_clean": stats_clean,
            "note": (
                "2I2：SEQUENCE 放电平台叠带宽有限抖动 RMS≈10 A（seed=20260906）；"
                "方波沿保留；抖动不是 NOISE_STD。"
            ),
        }
    if exp == "2i3":
        seq = drive_2i3_pieces()
        plan = expand_sequence(seq, dt_s=dt_s, capacity_ah=capacity_ah, t_default=25.0)
        t, i_clean, mode = steps_to_arrays(plan)
        rng = np.random.default_rng(JITTER_SEED_2I3)
        restish = np.isin(mode, ["rest", "idle", "pause"])
        active = ~restish
        i = _apply_jitter_on_mask(i_clean, active, rng=rng, rms=JITTER_RMS_A)
        stats = edge_stats(i)
        stats_clean = edge_stats(i_clean)
        jit = i - i_clean
        applied = np.abs(jit) > 1e-12
        rms_applied = float(np.std(jit[applied])) if np.any(applied) else 0.0
        # 按 ≥20 s 静置拆趟：整段 ~24 min 会踩中 ≥15 min 斜率备份门（EKF 啃 IR 残差），
        # 不是分流器。按趟估门后每段 <15 min，b_I=0 时包门不拦写 k（对齐 2C 拆趟）。
        trips = _trip_starts_after_rests(mode, min_rest_s=20.0)
        return {
            "seq": seq,
            "i_true": i,
            "i_clean": i_clean,
            "mode": mode,
            "time_s": t,
            "wave": "drive_jitter",
            "jitter_seed": JITTER_SEED_2I3,
            "jitter_rms_a": float(JITTER_RMS_A),
            "jitter_rms_active": rms_applied,
            "edge": stats,
            "edge_clean": stats_clean,
            "trips": trips,
            "note": (
                "2I3：~24 min 共享驾驶（斜坡/回收/怠速）+ 非怠速段同风格抖动 RMS≈10 A "
                "（seed=42，用户选定）；等温；不是认证 WLTP。抖动不是 NOISE_STD。"
                "按 ≥20 s 静置拆趟，避免整段 ≥15 min 斜率门误拦。"
            ),
        }
    raise ValueError(f"unknown 2I exp {exp}")
