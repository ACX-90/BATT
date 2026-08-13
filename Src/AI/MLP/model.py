"""参数 MLP：输入 (I, SOC, T)，输出正的 R0/R1（及可选 C1）。"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import TrainConfig


def softplus_inv(y: float) -> float:
    """softplus(z) = y 的反函数，用于把最后一层偏置设到参考值附近。"""
    y = max(float(y), 1e-12)
    return math.log(math.expm1(y))


class ParamMLP(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        scheme = cfg.scheme.upper()
        if scheme not in {"A", "B", "B+"}:
            raise ValueError(f"未知 scheme={cfg.scheme}，应为 A / B / B+")
        self.scheme = scheme
        self.r0_min = cfg.r0_min
        self.r1_min = cfg.r1_min
        self.c1_min = cfg.c1_min
        self.c1_star = cfg.c1_star

        out_dim = 3 if scheme == "A" else 2
        dims = [3, *cfg.hidden, out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        self._init_head(cfg)

        if scheme == "B+":
            self.phi = nn.Parameter(
                torch.tensor(softplus_inv(cfg.c1_star - cfg.c1_min), dtype=torch.float32)
            )
        else:
            self.register_parameter("phi", None)

    def _init_head(self, cfg: TrainConfig) -> None:
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        bias = [
            softplus_inv(cfg.r0_ref - cfg.r0_min),
            softplus_inv(cfg.r1_ref - cfg.r1_min),
        ]
        if self.scheme == "A":
            bias.append(softplus_inv(cfg.c1_star - cfg.c1_min))
        last.bias.data.copy_(torch.tensor(bias, dtype=last.bias.dtype))

    def forward(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x_norm: (..., 3) → R0, R1, C1 同形状 (...,)。"""
        z = self.net(x_norm)
        r0 = self.r0_min + F.softplus(z[..., 0])
        r1 = self.r1_min + F.softplus(z[..., 1])
        if self.scheme == "A":
            c1 = self.c1_min + F.softplus(z[..., 2])
        elif self.scheme == "B+":
            assert self.phi is not None
            c1_val = self.c1_min + F.softplus(self.phi)
            c1 = c1_val + torch.zeros_like(r0)
        else:
            c1 = r0.new_full(r0.shape, self.c1_star)
        return r0, r1, c1
