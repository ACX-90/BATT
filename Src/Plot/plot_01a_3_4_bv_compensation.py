"""§3.4 BV 补偿因子 h(I) 绘图：asinh vs 软化近似，标注 0.5C/1C/2C 工作点。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Plot"))
from _common import apply_style, save_figure

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = REPO_ROOT / "Doc" / "Fig"

# ── 参数 ──────────────────────────────────────────────
I_1C = 50.0          # 1C 特征电流 / A（模板值，§3.4 / §4.4）
k0 = 0.06            # 软化系数（§3.4 仓库模板）
T_C = 25.0           # 参考温度 / °C
T_K = T_C + 273.15   # 参考温度 / K
R_GAS = 8.314        # 通用气体常数 / J·mol⁻¹·K⁻¹
F = 96485.0          # Faraday 常数 / C·mol⁻¹
nRT_F = 2 * R_GAS * T_K / F   # 2RT/F ≈ 51 mV（对称 α=1/2，一电子）

# ── 函数 ──────────────────────────────────────────────
def h_BV(I: np.ndarray) -> np.ndarray:
    x = np.abs(I) / I_1C
    out = np.ones_like(x)
    nz = x > 0
    out[nz] = np.arcsinh(x[nz]) / x[nz]
    return out

def h_soft(I: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + k0 * np.abs(I) / I_1C)

def eta_BV(I: np.ndarray) -> np.ndarray:
    x = I / I_1C
    out = np.zeros_like(x)
    nz = np.abs(x) > 1e-12
    out[nz] = nRT_F * np.arcsinh(x[nz])
    return out

# ── 绘图 ──────────────────────────────────────────────
apply_style()

I_range = np.linspace(0, 3 * I_1C, 600)
I_abs = np.abs(I_range)

fig, ax1 = plt.subplots(figsize=(7.2, 4.6))

# 主轴：h(I) 因子
ax1.plot(I_range, h_BV(I_range), color="#2563eb", lw=2.2, label=r"$h_\mathrm{BV}(I) = \frac{\mathrm{asinh}(|I|/I_s)}{|I|/I_s}$")
ax1.plot(I_range, h_soft(I_range), color="#dc2626", lw=2.0, ls="--", label=r"$h_\mathrm{soft}(I) = \frac{1}{1 + k_0\,|I|/I_{1\mathrm{C}}}$，$k_0=" + f"{k0}$")
ax1.axhline(1.0, color="gray", ls=":", lw=1.0, alpha=0.6)

# 次轴：η_BV
ax2 = ax1.twinx()
ax2.plot(I_range, eta_BV(I_range) * 1000, color="#16a34a", lw=1.6, ls="-.", alpha=0.75,
         label=r"$\eta_\mathrm{BV}(I) = \frac{2RT}{F}\,\mathrm{asinh}\!\left(\frac{I}{I_s}\right)$")

# ── 标注工作点 ────────────────────────────────────────
work_points = [
    (0.5 * I_1C, "0.5C", "#f59e0b"),
    (1.0 * I_1C, "1C",   "#7c3aed"),
    (2.0 * I_1C, "2C",   "#0891b2"),
]

for I_wp, label, color in work_points:
    h_val = h_BV(np.array([I_wp]))[0]
    eta_val = eta_BV(np.array([I_wp]))[0] * 1000
    # 主轴标注
    ax1.plot(I_wp, h_val, "o", color=color, markersize=8, zorder=5)
    ax1.annotate(
        f"{label}\nh={h_val:.3f}",
        xy=(I_wp, h_val),
        xytext=(I_wp + 8, h_val + 0.012),
        fontsize=8, color=color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.85),
    )
    # 次轴标注
    ax2.plot(I_wp, eta_val, "s", color=color, markersize=6, zorder=5, alpha=0.7)

# ── 低 η 极限标注 ─────────────────────────────────────
ax1.axhline(1.0, color="gray", ls=":", lw=1.0, alpha=0.5)
ax1.annotate(
    r"$I \to 0$：$h \to 1$，$R_0 \to R_{0,\mathrm{ref}}$",
    xy=(0, 1.0), xytext=(60, 1.03),
    fontsize=8, color="gray",
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.7),
)

# ── 轴标签 / 标题 ─────────────────────────────────────
ax1.set_xlabel(r"电流 $I$ / A", fontsize=11)
ax1.set_ylabel(r"补偿因子 $h(I)$", fontsize=11, color="#2563eb")
ax2.set_ylabel(r"BV 过电位 $\eta_\mathrm{BV}$ / mV", fontsize=11, color="#16a34a")
ax1.set_title(r"§3.4  BV 补偿因子 $h(I)$：$\mathrm{asinh}$ 全量 vs 软化近似", fontsize=12, fontweight="bold")

# ── 图例 ──────────────────────────────────────────────
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8, framealpha=0.9)

ax1.set_xlim(0, 3 * I_1C)
ax1.set_ylim(0.85, 1.08)
ax2.set_ylim(0, eta_BV(np.array([3 * I_1C]))[0] * 1000 * 1.15)

# ── 参数文本框 ────────────────────────────────────────
param_text = (
    f"$I_s = {I_1C:.0f}$ A  "
    f"$2RT/F = {nRT_F*1000:.1f}$ mV  "
    f"$k_0 = {k0}$  "
    f"$T = {T_C:.0f}\\,^\\circ\\mathrm{{C}}$"
)
ax1.text(0.02, 0.02, param_text, transform=ax1.transAxes,
         fontsize=8, verticalalignment="bottom",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="gray", alpha=0.85))

fig.tight_layout()
out_path = FIG_DIR / "01-a-3_4-BV-compensation.png"
save_figure(fig, out_path, show=False)
print(f"Saved: {out_path}")
