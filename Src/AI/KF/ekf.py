"""最小 EKF：状态 [s, U_p]（可选慢变 δR0）。

预测用安时 + 一阶 ECM；测量是端电压。R0/R1 由调用方按预测 SOC 提供，
本拍不回代后验再算电阻，避免代数环（Doc/06 §2.2）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import KfConfig
from .ocv import docv_ds, ocv_nmc


@dataclass
class EkfStep:
    s_pred: float
    s_post: float
    u_p_pred: float
    u_p_post: float
    d_r0: float
    r0_used: float
    r1: float
    c1: float
    u_ocv: float
    u_t_pri: float
    u_t_post: float
    e_pri: float
    e_post: float
    nis: float
    docv_ds: float
    k_s: float
    k_up: float
    alpha: float
    s: float  # HPH^T + Rv


class SocUpEKF:
    def __init__(self, cfg: KfConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else KfConfig()
        self.n = 3 if self.cfg.estimate_dr0 else 2
        self.s = 0.5
        self.u_p = 0.0
        self.d_r0 = 0.0
        self.P = np.eye(self.n)
        self._i = 0.0
        self._s_pred = 0.5
        self.reset(0.5, 0.0)

    def reset(self, s0: float, u_p0: float = 0.0, d_r0: float = 0.0) -> None:
        cfg = self.cfg
        self.s = float(np.clip(s0, cfg.soc_min, cfg.soc_max))
        self.u_p = float(u_p0)
        self.d_r0 = float(d_r0)
        diag = [cfg.p0_s, cfg.p0_up]
        if self.n == 3:
            diag.append(cfg.p0_dr0)
        self.P = np.diag(diag).astype(float)

    def predict_soc(self, i_a: float) -> float:
        """安时预测 s_{k|k-1}。MLP 必须用这个 SOC，不要用后验。"""
        cfg = self.cfg
        self._i = float(i_a)
        self._s_pred = float(
            np.clip(self.s - self._i * cfg.dt_s / cfg.q_coulomb, cfg.soc_min, cfg.soc_max)
        )
        return self._s_pred

    def update(self, t_celsius: float, u_meas: float, r0: float, r1: float, c1: float) -> EkfStep:
        cfg = self.cfg
        i_a = self._i
        s_pred = self._s_pred
        dt = cfg.dt_s
        tau = max(float(r1) * float(c1), 1.0e-6)
        alpha = float(np.exp(-dt / tau))
        r0_used = float(r0) + (self.d_r0 if cfg.estimate_dr0 else 0.0)
        u_p_pred = alpha * self.u_p + float(r1) * (1.0 - alpha) * i_a
        u_ocv = float(ocv_nmc(s_pred, t_celsius))
        slope = float(docv_ds(s_pred, t_celsius))
        u_t_pri = u_ocv - i_a * r0_used - u_p_pred
        e_pri = float(u_meas) - u_t_pri

        F = np.eye(self.n)
        F[1, 1] = alpha
        Q = np.diag([cfg.q_s, cfg.q_up] + ([cfg.q_dr0] if self.n == 3 else []))
        P_pri = F @ self.P @ F.T + Q

        rv = cfg.rv
        if cfg.schedule_rv:
            scale = (cfg.slope_min / max(abs(slope), 1.0e-6)) ** 2
            rv = rv * min(max(scale, 1.0), cfg.rv_max_scale)

        if self.n == 3:
            H = np.array([[slope, -1.0, -i_a]], dtype=float)
        else:
            H = np.array([[slope, -1.0]], dtype=float)

        s_innov = float(np.asarray(H @ P_pri @ H.T).reshape(-1)[0] + rv)
        s_innov = max(s_innov, 1.0e-18)
        K = (P_pri @ H.T) / s_innov
        if cfg.ks_max > 0.0:
            K[0, 0] = float(np.clip(K[0, 0], -cfg.ks_max, cfg.ks_max))

        dx = (K * e_pri).reshape(-1)
        s_post = float(np.clip(s_pred + dx[0], cfg.soc_min, cfg.soc_max))
        u_p_post = float(u_p_pred + dx[1])
        d_r0_post = float(self.d_r0 + dx[2]) if self.n == 3 else 0.0

        i_kh = np.eye(self.n) - K @ H
        self.P = i_kh @ P_pri @ i_kh.T + (K * rv) @ K.T
        self.s = s_post
        self.u_p = u_p_post
        self.d_r0 = d_r0_post

        r0_post = float(r0) + (d_r0_post if cfg.estimate_dr0 else 0.0)
        u_t_post = float(ocv_nmc(s_post, t_celsius)) - i_a * r0_post - u_p_post
        e_post = float(u_meas) - u_t_post
        nis = (e_pri * e_pri) / s_innov

        return EkfStep(
            s_pred=s_pred,
            s_post=s_post,
            u_p_pred=u_p_pred,
            u_p_post=u_p_post,
            d_r0=d_r0_post,
            r0_used=r0_used,
            r1=float(r1),
            c1=float(c1),
            u_ocv=u_ocv,
            u_t_pri=u_t_pri,
            u_t_post=u_t_post,
            e_pri=e_pri,
            e_post=e_post,
            nis=float(nis),
            docv_ds=slope,
            k_s=float(K[0, 0]),
            k_up=float(K[1, 0]),
            alpha=alpha,
            s=s_innov,
        )


def selftest() -> dict[str, float]:
    """静置时用电压把错误的 SOC 初值拉回来。"""
    cfg = KfConfig(rv_std=0.5e-3, q_s=1.0e-8, schedule_rv=False)
    ekf = SocUpEKF(cfg)
    s_true = 0.80
    t_c = 25.0
    r0, r1, c1 = 8.0e-4, 6.5e-4, 2.8e4
    u_meas = float(ocv_nmc(s_true, t_c))
    ekf.reset(0.70, 0.0)
    for _ in range(80):
        ekf.predict_soc(0.0)
        step = ekf.update(t_c, u_meas, r0, r1, c1)
    err0 = abs(0.70 - s_true)
    err1 = abs(step.s_post - s_true)
    if err1 >= 0.4 * err0:
        raise RuntimeError(f"EKF 静置纠偏失败: s={step.s_post:.4f}  期望靠近 {s_true:.2f}")
    return {"s_post": step.s_post, "e_pri": step.e_pri, "err0": err0, "err1": err1}
