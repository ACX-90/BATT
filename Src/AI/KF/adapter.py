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


# 与 Src/MCU_Eval/eval_k_grid.c 同一套出厂节点
KGRID_SOC = (0.10, 0.30, 0.50, 0.70, 0.90)
KGRID_T = (-10.0, 10.0, 30.0, 50.0)


class KGridAdapter(nn.Module):
    """冻 MLP，(SOC,T) 上双线性 k0/k1。没点到的节点保持 1。"""

    def __init__(
        self,
        base: ParamMLP,
        *,
        soc_node: tuple[float, ...] = KGRID_SOC,
        t_node: tuple[float, ...] = KGRID_T,
    ) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.register_buffer("soc_node", torch.tensor(soc_node, dtype=torch.float32))
        self.register_buffer("t_node", torch.tensor(t_node, dtype=torch.float32))
        ns, nt = len(soc_node), len(t_node)
        self.log_k0 = nn.Parameter(torch.zeros(ns, nt))
        self.log_k1 = nn.Parameter(torch.zeros(ns, nt))

    def _corners(self, soc: torch.Tensor, t_c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        s_ax = self.soc_node
        t_ax = self.t_node
        soc = soc.clamp(s_ax[0], s_ax[-1])
        t_c = t_c.clamp(t_ax[0], t_ax[-1])
        is_ = torch.searchsorted(s_ax, soc, right=True) - 1
        it_ = torch.searchsorted(t_ax, t_c, right=True) - 1
        is_ = is_.clamp(0, s_ax.numel() - 2)
        it_ = it_.clamp(0, t_ax.numel() - 2)
        s0, s1 = s_ax[is_], s_ax[is_ + 1]
        t0, t1 = t_ax[it_], t_ax[it_ + 1]
        ws = (soc - s0) / (s1 - s0).clamp_min(1e-6)
        wt = (t_c - t0) / (t1 - t0).clamp_min(1e-6)
        return is_, it_, ws, wt

    def interp_k(self, log_k: torch.Tensor, soc: torch.Tensor, t_c: torch.Tensor) -> torch.Tensor:
        is_, it_, ws, wt = self._corners(soc, t_c)
        k = torch.exp(log_k)
        k00 = k[is_, it_]
        k10 = k[is_ + 1, it_]
        k01 = k[is_, it_ + 1]
        k11 = k[is_ + 1, it_ + 1]
        return (1.0 - ws) * (1.0 - wt) * k00 + ws * (1.0 - wt) * k10 + (1.0 - ws) * wt * k01 + ws * wt * k11

    def forward(
        self,
        x_norm: torch.Tensor,
        soc: torch.Tensor,
        t_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r0, r1, c1 = self.base(x_norm)
        k0 = self.interp_k(self.log_k0, soc, t_c)
        k1 = self.interp_k(self.log_k1, soc, t_c)
        return r0 * k0, r1 * k1, c1

    def k_tables(self) -> dict[str, list]:
        with torch.no_grad():
            return {
                "soc_node": self.soc_node.cpu().tolist(),
                "t_node": self.t_node.cpu().tolist(),
                "k0": torch.exp(self.log_k0).cpu().tolist(),
                "k1": torch.exp(self.log_k1).cpu().tolist(),
            }

    def k_at(self, soc: float, t_c: float) -> tuple[float, float]:
        s = torch.tensor(soc, dtype=self.log_k0.dtype, device=self.log_k0.device)
        t = torch.tensor(t_c, dtype=self.log_k0.dtype, device=self.log_k0.device)
        with torch.no_grad():
            k0 = float(self.interp_k(self.log_k0, s, t))
            k1 = float(self.interp_k(self.log_k1, s, t))
        return k0, k1


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


def phi_from_mlp(mlp: ParamMLP) -> nn.Sequential:
    """3×8×2 的前层（Linear+GELU），输出 8 维特征。"""
    kids = list(mlp.net.children())
    if len(kids) < 2 or not isinstance(kids[-1], nn.Linear):
        raise ValueError("ParamMLP 末层不是 Linear，抽不出 φ")
    last_h = [m for m in kids[:-1] if isinstance(m, nn.Linear)]
    if not last_h or int(last_h[-1].out_features) != 8:
        raise ValueError(f"残差头 φ 必须是 8 维隐层，得到 {[int(m.out_features) for m in last_h]}")
    phi = nn.Sequential(*kids[:-1])
    for p in phi.parameters():
        p.requires_grad = False
    return phi


class ResidualHeadAdapter(nn.Module):
    """冻舰队 MLP 与 3×8 前层，只训 8→2（18 个数）。

    R = f_fleet + dr_max * tanh(W φ + b)，头清零时 ΔR≡0（Doc/03-d 式 (6)）。
    """

    def __init__(self, fleet: ParamMLP, phi: nn.Module, *, dr_max: float = 2.0e-3) -> None:
        super().__init__()
        self.fleet = fleet
        for p in self.fleet.parameters():
            p.requires_grad = False
        self.phi = phi
        for p in self.phi.parameters():
            p.requires_grad = False
        last_h = [m for m in phi.modules() if isinstance(m, nn.Linear)]
        if not last_h or int(last_h[-1].out_features) != 8:
            raise ValueError("φ 末维必须是 8")
        self.head = nn.Linear(8, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.dr_max = float(dr_max)
        self.r0_min = float(fleet.r0_min)
        self.r1_min = float(fleet.r1_min)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def n_trainable(self) -> int:
        return int(sum(p.numel() for p in self.trainable_parameters()))

    def delta(self, x_norm: torch.Tensor) -> torch.Tensor:
        h = self.phi(x_norm)
        z = self.head(h)
        return self.dr_max * torch.tanh(z)

    def forward(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r0, r1, c1 = self.fleet(x_norm)
        d = self.delta(x_norm)
        r0 = (r0 + d[..., 0]).clamp_min(self.r0_min)
        r1 = (r1 + d[..., 1]).clamp_min(self.r1_min)
        return r0, r1, c1

    def delta_at(self, x_norm: torch.Tensor) -> tuple[float, float]:
        with torch.no_grad():
            d = self.delta(x_norm.reshape(1, 3)).reshape(2)
        return float(d[0]), float(d[1])
