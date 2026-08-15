"""NMC 100 Ah OCV 表，与 Src/Sim/nmc100ah_ecm_gen.py 同一套数。

KF 不学 OCV。dU_ocv/ds 只用于观测雅可比和平台区增益调度。
"""

from __future__ import annotations

import numpy as np

# 典型 100 Ah NMC/石墨 OCV 表（25 °C）
OCV_SOC = np.array(
    [0.00, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 1.00],
    dtype=float,
)
OCV_V = np.array(
    [3.280, 3.400, 3.480, 3.545, 3.595, 3.630, 3.658, 3.690, 3.735, 3.800, 3.890, 4.020, 4.100, 4.155, 4.185],
    dtype=float,
)
OCV_DUDT = -4.0e-4  # V/°C，相对 25 °C


def ocv_nmc(soc: float | np.ndarray, t_celsius: float | np.ndarray = 25.0) -> float | np.ndarray:
    s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
    t = np.asarray(t_celsius, dtype=float)
    val = np.interp(s, OCV_SOC, OCV_V) + OCV_DUDT * (t - 25.0)
    if np.ndim(soc) == 0 and np.ndim(t_celsius) == 0:
        return float(val)
    return val


def docv_ds(soc: float | np.ndarray, t_celsius: float | np.ndarray = 25.0) -> float | np.ndarray:
    """分段线性 OCV 对 SOC 的斜率。与温度无关。"""
    del t_celsius
    s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
    slopes = np.diff(OCV_V) / np.diff(OCV_SOC)
    idx = np.searchsorted(OCV_SOC, s, side="right") - 1
    idx = np.clip(idx, 0, len(slopes) - 1)
    val = slopes[idx]
    if np.ndim(soc) == 0:
        return float(val)
    return val


def inv_ocv(u_ocv: float | np.ndarray, t_celsius: float | np.ndarray = 25.0) -> float | np.ndarray:
    """用 25 °C 单调表反查 SOC。久置后 U_p≈0 时可用端电压当 OCV。"""
    t = np.asarray(t_celsius, dtype=float)
    u25 = np.asarray(u_ocv, dtype=float) - OCV_DUDT * (t - 25.0)
    val = np.clip(np.interp(u25, OCV_V, OCV_SOC), 0.0, 1.0)
    if np.ndim(u_ocv) == 0 and np.ndim(t_celsius) == 0:
        return float(val)
    return val
