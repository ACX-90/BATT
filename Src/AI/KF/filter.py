"""闭环 EKF + 开环对照。

每拍顺序（Doc/03-c §2.2）：
  安时预测 s^- → MLP(I, s^-, T) → 极化预测 → 先验电压 → 新息 → 更新 s, Up
开环 e_ol 用同一套 MLP 和安时 SOC，不经过 KF，供增量损失使用。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from MLP.ecm import ecm_forward

from .adapter import MlpParamProvider
from .config import KfConfig
from .ekf import SocUpEKF
from .ocv import inv_ocv, ocv_nmc


@dataclass
class FilterInit:
    s0: float
    u_p0: float = 0.0
    source: str = "given"


def pick_soc0(
    i_a: np.ndarray,
    t_c: np.ndarray,
    u_meas: np.ndarray,
    *,
    soc0: float | None,
    rest_eps: float = 1.0,
    rest_steps: int = 20,
) -> FilterInit:
    """久置开头用 OCV 反查；否则必须给 soc0。"""
    if soc0 is not None:
        return FilterInit(s0=float(soc0), u_p0=0.0, source="given")
    n = min(int(rest_steps), len(i_a))
    if n > 0 and np.all(np.abs(i_a[:n]) <= rest_eps):
        s0 = float(inv_ocv(float(np.mean(u_meas[:n])), float(np.mean(t_c[:n]))))
        return FilterInit(s0=s0, u_p0=0.0, source="ocv")
    raise ValueError("无法从电压反查 SOC0：开头不是静置，请显式传入 --soc0")


def _open_loop(
    provider: MlpParamProvider,
    i_a: np.ndarray,
    s_ah: np.ndarray,
    t_c: np.ndarray,
    u_meas: np.ndarray,
    *,
    dt_s: float,
    u_p0: float,
    ocv=None,
) -> dict[str, np.ndarray]:
    r0, r1, c1 = provider.params_seq(i_a, s_ah, t_c)
    ocv_fn = ocv_nmc if ocv is None else ocv
    u_ocv = np.asarray(ocv_fn(s_ah, t_c), dtype=float)
    i_t = torch.from_numpy(i_a.astype(np.float32)).unsqueeze(0)
    ocv_t = torch.from_numpy(u_ocv.astype(np.float32)).unsqueeze(0)
    r0_t = torch.from_numpy(r0.astype(np.float32)).unsqueeze(0)
    r1_t = torch.from_numpy(r1.astype(np.float32)).unsqueeze(0)
    c1_t = torch.from_numpy(c1.astype(np.float32)).unsqueeze(0)
    u_p0_t = torch.tensor([u_p0], dtype=torch.float32)
    u_hat, u_p = ecm_forward(i_t, ocv_t, r0_t, r1_t, c1_t, dt_s=dt_s, u_p0=u_p0_t)
    u_ol = u_hat.squeeze(0).numpy().astype(float)
    return {
        "r0_ol": r0,
        "r1_ol": r1,
        "c1_ol": c1,
        "u_ocv_ol": u_ocv,
        "u_t_ol": u_ol,
        "u_p_ol": u_p.squeeze(0).numpy().astype(float),
        "e_ol": u_meas.astype(float) - u_ol,
    }


def run_filter(
    provider: MlpParamProvider,
    i_a: np.ndarray,
    t_c: np.ndarray,
    u_meas: np.ndarray,
    *,
    cfg: KfConfig | None = None,
    s0: float | None = None,
    u_p0: float = 0.0,
    soc_error: float = 0.0,
    current_bias: float = 0.0,
    time_s: np.ndarray | None = None,
    soc_true: np.ndarray | None = None,
    ocv=None,
    docv=None,
) -> dict[str, np.ndarray]:
    """只吃测量电流 / 温度 / 电压。current_bias 加在所用电流上（演示传感器零偏）。"""
    cfg = cfg if cfg is not None else KfConfig()
    i_a = np.asarray(i_a, dtype=float)
    t_c = np.asarray(t_c, dtype=float)
    u_meas = np.asarray(u_meas, dtype=float)
    n = len(i_a)
    if time_s is None:
        time_s = np.arange(n, dtype=float) * cfg.dt_s
    i_used = i_a + float(current_bias)

    init = pick_soc0(i_used, t_c, u_meas, soc0=s0)
    s_init = float(np.clip(init.s0 + soc_error, cfg.soc_min, cfg.soc_max))
    ekf = SocUpEKF(cfg, ocv=ocv, docv=docv)
    ekf.reset(s_init, u_p0 if u_p0 != 0.0 else init.u_p0)

    s_ah = np.empty(n)
    s_pred = np.empty(n)
    s_post = np.empty(n)
    u_p_pred = np.empty(n)
    u_p_post = np.empty(n)
    d_r0 = np.empty(n)
    r0_k = np.empty(n)
    r1_k = np.empty(n)
    c1_k = np.empty(n)
    u_ocv = np.empty(n)
    u_t_pri = np.empty(n)
    u_t_post = np.empty(n)
    e_pri = np.empty(n)
    e_post = np.empty(n)
    nis = np.empty(n)
    slope = np.empty(n)
    k_s = np.empty(n)
    k_up = np.empty(n)

    soc_ah = s_init
    q = cfg.q_coulomb
    for k in range(n):
        s_hat = ekf.predict_soc(i_used[k])
        r0, r1, c1 = provider.params(i_used[k], s_hat, t_c[k])
        step = ekf.update(t_c[k], u_meas[k], r0, r1, c1)
        soc_ah = float(np.clip(soc_ah - i_used[k] * cfg.dt_s / q, cfg.soc_min, cfg.soc_max))
        s_ah[k] = soc_ah
        s_pred[k] = step.s_pred
        s_post[k] = step.s_post
        u_p_pred[k] = step.u_p_pred
        u_p_post[k] = step.u_p_post
        d_r0[k] = step.d_r0
        r0_k[k] = step.r0_used
        r1_k[k] = step.r1
        c1_k[k] = step.c1
        u_ocv[k] = step.u_ocv
        u_t_pri[k] = step.u_t_pri
        u_t_post[k] = step.u_t_post
        e_pri[k] = step.e_pri
        e_post[k] = step.e_post
        nis[k] = step.nis
        slope[k] = step.docv_ds
        k_s[k] = step.k_s
        k_up[k] = step.k_up

    ol = _open_loop(
        provider, i_used, s_ah, t_c, u_meas, dt_s=cfg.dt_s, u_p0=init.u_p0, ocv=ocv
    )
    out: dict[str, np.ndarray] = {
        "time_s": np.asarray(time_s, dtype=float),
        "i_meas_a": i_a,
        "i_used_a": i_used,
        "t_meas_c": t_c,
        "u_t_meas_v": u_meas,
        "soc_ah": s_ah,
        "soc_pred": s_pred,
        "soc_post": s_post,
        "u_p_pred_v": u_p_pred,
        "u_p_post_v": u_p_post,
        "d_r0_ohm": d_r0,
        "r0_ohm": r0_k,
        "r1_ohm": r1_k,
        "c1_f": c1_k,
        "u_ocv_v": u_ocv,
        "u_t_pri_v": u_t_pri,
        "u_t_post_v": u_t_post,
        "e_pri": e_pri,
        "e_post": e_post,
        "nis": nis,
        "docv_ds": slope,
        "k_s": k_s,
        "k_up": k_up,
        **ol,
    }
    if soc_true is not None:
        out["soc_true"] = np.asarray(soc_true, dtype=float)
    return out


def filter_metrics(log: dict[str, np.ndarray]) -> dict[str, float]:
    def rmse(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x))))

    out = {
        "e_ol_rmse_mV": rmse(log["e_ol"]) * 1e3,
        "e_pri_rmse_mV": rmse(log["e_pri"]) * 1e3,
        "e_post_rmse_mV": rmse(log["e_post"]) * 1e3,
        "nis_median": float(np.median(log["nis"])),
        "nis_mean": float(np.mean(log["nis"])),
        "s_end_ah": float(log["soc_ah"][-1]),
        "s_end_post": float(log["soc_post"][-1]),
    }
    if "soc_true" in log:
        out["s_ah_rmse"] = rmse(log["soc_ah"] - log["soc_true"])
        out["s_post_rmse"] = rmse(log["soc_post"] - log["soc_true"])
        out["s_end_true"] = float(log["soc_true"][-1])
        out["s_end_ah_err"] = float(log["soc_ah"][-1] - log["soc_true"][-1])
        out["s_end_post_err"] = float(log["soc_post"][-1] - log["soc_true"][-1])
    return out
