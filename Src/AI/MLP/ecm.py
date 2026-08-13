"""可微一阶 Thevenin 前向，离散化与 Src/Sim/nmc100ah_ecm_gen.py 一致。"""

from __future__ import annotations

import torch


def ecm_forward(
    current: torch.Tensor,
    u_ocv: torch.Tensor,
    r0: torch.Tensor,
    r1: torch.Tensor,
    c1: torch.Tensor,
    *,
    dt_s: float,
    u_p0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """逐步推进极化状态。

    参数
    ----
    current, u_ocv, r0, r1, c1
        形状 (B, T)，放电电流为正。
    u_p0
        可选 (B,)，缺省为 0。

    返回
    ----
    u_t, u_p : (B, T)
    """
    if current.ndim != 2:
        raise ValueError("ECM 输入应为 (B, T)")
    batch, n_step = current.shape
    u_p = current.new_zeros(batch) if u_p0 is None else u_p0
    hist: list[torch.Tensor] = []
    dt = current.new_tensor(dt_s)

    for k in range(n_step):
        tau = (r1[:, k] * c1[:, k]).clamp_min(1.0e-6)
        alpha = torch.exp(-dt / tau)
        u_p = alpha * u_p + r1[:, k] * (1.0 - alpha) * current[:, k]
        hist.append(u_p)

    u_p_seq = torch.stack(hist, dim=1)
    u_t = u_ocv - current * r0 - u_p_seq
    return u_t, u_p_seq


def ecm_forward_tbptt(
    current: torch.Tensor,
    u_ocv: torch.Tensor,
    r0: torch.Tensor,
    r1: torch.Tensor,
    c1: torch.Tensor,
    *,
    dt_s: float,
    window: int,
) -> torch.Tensor:
    """按窗口截断 BPTT：窗口之间 detach 极化状态。"""
    if window <= 0:
        u_t, _ = ecm_forward(current, u_ocv, r0, r1, c1, dt_s=dt_s)
        return u_t

    batch, n_step = current.shape
    u_p = current.new_zeros(batch)
    chunks: list[torch.Tensor] = []
    for start in range(0, n_step, window):
        end = min(start + window, n_step)
        u_t, u_p_seq = ecm_forward(
            current[:, start:end],
            u_ocv[:, start:end],
            r0[:, start:end],
            r1[:, start:end],
            c1[:, start:end],
            dt_s=dt_s,
            u_p0=u_p,
        )
        chunks.append(u_t)
        u_p = u_p_seq[:, -1].detach()
    return torch.cat(chunks, dim=1)
