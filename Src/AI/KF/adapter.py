"""MLP → (R0, R1, C1)。滤波每拍用预测 SOC，不回代后验。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from MLP.config import TrainConfig
from MLP.dataset import FeatureScaler
from MLP.infer import load_bundle
from MLP.model import ParamMLP


class MlpParamProvider:
    """逐步查询。开环整段可走 params_seq。"""

    def __init__(
        self,
        model: nn.Module,
        scaler: FeatureScaler,
        *,
        device: str | torch.device = "cpu",
        r0_scale: float = 1.0,
        r1_scale: float = 1.0,
    ) -> None:
        self.model = model.to(device).eval()
        self.scaler = scaler
        self.device = torch.device(device)
        self.r0_scale = float(r0_scale)
        self.r1_scale = float(r1_scale)

    @classmethod
    def from_dir(
        cls,
        out_dir: Path,
        *,
        ckpt: Path,
        device: str = "cpu",
        r0_scale: float = 1.0,
        r1_scale: float = 1.0,
    ) -> tuple["MlpParamProvider", TrainConfig]:
        model, scaler, cfg = load_bundle(ckpt, out_dir / "config.json", out_dir / "scaler.json")
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "log_k0" in blob or "k0" in blob:
            model = ScaleAdapter.from_blob(model, blob)
            model.eval()
        return cls(model, scaler, device=device, r0_scale=r0_scale, r1_scale=r1_scale), cfg

    @torch.no_grad()
    def params(self, i_a: float, soc: float, t_c: float) -> tuple[float, float, float]:
        feat = np.array([[i_a, soc, t_c]], dtype=float)
        xn = self.scaler.transform(feat).astype(np.float32)
        x = torch.from_numpy(xn).to(self.device)
        r0, r1, c1 = self.model(x)
        return (
            float(r0) * self.r0_scale,
            float(r1) * self.r1_scale,
            float(c1),
        )

    @torch.no_grad()
    def params_seq(
        self,
        i_a: np.ndarray,
        soc: np.ndarray,
        t_c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feat = np.stack([i_a, soc, t_c], axis=-1).astype(float)
        xn = self.scaler.transform(feat).astype(np.float32)
        x = torch.from_numpy(xn).to(self.device)
        r0, r1, c1 = self.model(x)
        return (
            r0.detach().cpu().numpy() * self.r0_scale,
            r1.detach().cpu().numpy() * self.r1_scale,
            c1.detach().cpu().numpy(),
        )


class ScaleAdapter(nn.Module):
    """冻住 MLP，只学 R0'=k0 R0、R1'=k1 R1（Doc/03-a §3.5）。"""

    def __init__(self, base: ParamMLP) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.log_k0 = nn.Parameter(torch.zeros(()))
        self.log_k1 = nn.Parameter(torch.zeros(()))

    def forward(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r0, r1, c1 = self.base(x_norm)
        return r0 * torch.exp(self.log_k0), r1 * torch.exp(self.log_k1), c1

    @property
    def k0(self) -> float:
        return float(self.log_k0.detach().exp())

    @property
    def k1(self) -> float:
        return float(self.log_k1.detach().exp())

    @classmethod
    def from_blob(cls, base: ParamMLP, blob: dict) -> "ScaleAdapter":
        wrap = cls(base)
        if "log_k0" in blob:
            wrap.log_k0.data.copy_(torch.as_tensor(blob["log_k0"], dtype=wrap.log_k0.dtype))
            wrap.log_k1.data.copy_(torch.as_tensor(blob["log_k1"], dtype=wrap.log_k1.dtype))
        else:
            wrap.log_k0.data.fill_(float(np.log(max(float(blob.get("k0", 1.0)), 1e-12))))
            wrap.log_k1.data.fill_(float(np.log(max(float(blob.get("k1", 1.0)), 1e-12))))
        return wrap
