"""Causal small GRU residual head: scalar δU only (approach Δ)."""
from __future__ import annotations

import math

import torch
from torch import nn


class CausalDeltaUGRU(nn.Module):
    """d=4 GRU → scalar δU, soft-clipped to ±clip_v (default 8 mV)."""

    def __init__(self, *, d: int = 4, n_in: int = 4, clip_v: float = 0.008) -> None:
        super().__init__()
        if d > 4:
            raise ValueError(f"Phase-3 default forbids d>4, got d={d}")
        self.d = int(d)
        self.n_in = int(n_in)
        self.clip_v = float(clip_v)
        self.gru = nn.GRU(self.n_in, self.d, batch_first=True)
        self.head = nn.Linear(self.d, 1)
        self._near_zero_init()

    def _near_zero_init(self) -> None:
        for name, p in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p, gain=0.1)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, n_in) → δU (B, T), h_n (1, B, d)."""
        out, h_n = self.gru(x, h0)
        raw = self.head(out).squeeze(-1)
        # soft clip: clip * tanh(raw / clip) so |δU| ≤ clip
        du = self.clip_v * torch.tanh(raw / max(self.clip_v, 1e-6))
        return du, h_n

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_features(
    i_a: torch.Tensor,
    e_ol: torch.Tensor,
    *,
    dt_s: float = 0.1,
    rms_win_s: float = 1.0,
) -> torch.Tensor:
    """Causal features (B,T,4): I/100, e_ol_prev/10mV, sign(I), |I|_RMS/100.

    e_ol is open-loop voltage error of 1RC vs truth (U_1RC - U_py), already (B,T).
    Uses e_ol shifted by 1 (causal); step 0 uses 0.
    """
    if i_a.ndim != 2:
        raise ValueError("i_a must be (B,T)")
    b, t = i_a.shape
    i_n = i_a / 100.0
    e_prev = torch.zeros_like(e_ol)
    e_prev[:, 1:] = e_ol[:, :-1]
    e_n = e_prev / 0.01
    sgn = torch.sign(i_a)
    win = max(1, int(round(rms_win_s / dt_s)))
    # causal rolling RMS of |I|
    abs_i = i_a.abs()
    csum = torch.cumsum(abs_i.pow(2), dim=1)
    pad = torch.zeros(b, 1, device=i_a.device, dtype=i_a.dtype)
    csum_pad = torch.cat([pad, csum], dim=1)
    idx = torch.arange(t, device=i_a.device)
    start = (idx + 1 - win).clamp_min(0)
    # gather end=csum[:,k], start=csum_pad[:,start]
    end_v = csum
    start_v = csum_pad.gather(1, start.unsqueeze(0).expand(b, -1))
    n = (idx + 1 - start).to(i_a.dtype).unsqueeze(0)
    rms = torch.sqrt(((end_v - start_v) / n).clamp_min(0.0)) / 100.0
    return torch.stack([i_n, e_n, sgn, rms], dim=-1)


def huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.005) -> torch.Tensor:
    err = pred - target
    abs_e = err.abs()
    quad = torch.clamp(abs_e, max=delta)
    lin = abs_e - quad
    return (0.5 * quad.pow(2) / delta + lin).mean()
