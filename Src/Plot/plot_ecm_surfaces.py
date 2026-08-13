"""绘制 NMC 100Ah ECM 参数 R0 / R1 / C1 的 3D 曲面。

上排：固定 1C 放电，SOC–温度
下排：固定 25 °C，SOC–电流

用法（仓库根目录）：

    python Src/Plot/plot_ecm_surfaces.py
    python Src/Plot/plot_ecm_surfaces.py --show
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from _common import FIG_DIR, apply_style, save_figure

from nmc100ah_ecm import NMC100AhECM  # noqa: E402


def _surface(ax, x, y, z, *, cmap, zlabel: str, title: str) -> None:
    surf = ax.plot_surface(
        x,
        y,
        z,
        cmap=cmap,
        linewidth=0.0,
        antialiased=True,
        rstride=1,
        cstride=1,
        alpha=0.95,
    )
    ax.set_xlabel("SOC")
    ax.set_title(title)
    ax.set_zlabel(zlabel)
    ax.view_init(elev=24, azim=-140)
    ax.tick_params(labelsize=8)
    plt.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, fraction=0.046)


def build_surfaces(
    model: NMC100AhECM,
    *,
    n_soc: int = 41,
    n_t: int = 31,
    n_i: int = 31,
    i_fixed_a: float = 100.0,
    t_fixed_c: float = 25.0,
) -> dict[str, np.ndarray]:
    lim = model.params.validity
    soc = np.linspace(lim.soc_min, lim.soc_max, n_soc)
    t_c = np.linspace(lim.t_min_c, lim.t_max_c, n_t)
    i_a = np.linspace(-2.0 * model.params.cell.capacity_ah, 2.0 * model.params.cell.capacity_ah, n_i)

    soc_t, t_g = np.meshgrid(soc, t_c, indexing="ij")
    soc_i, i_g = np.meshgrid(soc, i_a, indexing="ij")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        st = model.evaluate_full(
            i_a=np.full_like(soc_t, i_fixed_a),
            t_celsius=t_g,
            soc=soc_t,
            return_factors=False,
        )
        si = model.evaluate_full(
            i_a=i_g,
            t_celsius=np.full_like(soc_i, t_fixed_c),
            soc=soc_i,
            return_factors=False,
        )
    return {
        "soc": soc,
        "t_celsius": t_c,
        "i_a": i_a,
        "soc_t": soc_t,
        "t_g": t_g,
        "r0_st": np.asarray(st.R0) * 1e3,
        "r1_st": np.asarray(st.R1) * 1e3,
        "c1_st": np.asarray(st.C1) * 1e-3,
        "soc_i": soc_i,
        "i_g": i_g,
        "r0_si": np.asarray(si.R0) * 1e3,
        "r1_si": np.asarray(si.R1) * 1e3,
        "c1_si": np.asarray(si.C1) * 1e-3,
        "i_fixed_a": np.array(i_fixed_a),
        "t_fixed_c": np.array(t_fixed_c),
    }


def plot_surfaces(data: dict[str, np.ndarray], *, show: bool = False) -> Path:
    apply_style()
    i_fixed = float(data["i_fixed_a"])
    t_fixed = float(data["t_fixed_c"])

    fig = plt.figure(figsize=(14.5, 8.8))
    fig.suptitle("NMC 100Ah 一阶 ECM 参数曲面  $R_0,\\,R_1,\\,C_1$", fontsize=13, y=0.98)

    specs = [
        (1, data["soc_t"], data["t_g"], data["r0_st"], "温度 / °C", "$R_0$ / mΩ", f"$R_0$(SOC, $T$), $I$={i_fixed:.0f} A", cm.viridis),
        (2, data["soc_t"], data["t_g"], data["r1_st"], "温度 / °C", "$R_1$ / mΩ", f"$R_1$(SOC, $T$), $I$={i_fixed:.0f} A", cm.inferno),
        (3, data["soc_t"], data["t_g"], data["c1_st"], "温度 / °C", "$C_1$ / kF", f"$C_1$(SOC, $T$), $I$={i_fixed:.0f} A", cm.cividis),
        (4, data["soc_i"], data["i_g"], data["r0_si"], "电流 / A", "$R_0$ / mΩ", f"$R_0$(SOC, $I$), $T$={t_fixed:.0f} °C", cm.viridis),
        (5, data["soc_i"], data["i_g"], data["r1_si"], "电流 / A", "$R_1$ / mΩ", f"$R_1$(SOC, $I$), $T$={t_fixed:.0f} °C", cm.inferno),
        (6, data["soc_i"], data["i_g"], data["c1_si"], "电流 / A", "$C_1$ / kF", f"$C_1$(SOC, $I$), $T$={t_fixed:.0f} °C", cm.cividis),
    ]
    for idx, x, y, z, ylabel, zlabel, title, cmap in specs:
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        _surface(ax, x, y, z, cmap=cmap, zlabel=zlabel, title=title)
        ax.set_ylabel(ylabel)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, FIG_DIR / "nmc100ah_ecm_surfaces.png", show=show)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 R0/R1/C1 三维曲面")
    parser.add_argument("--show", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    model = NMC100AhECM()
    data = build_surfaces(model)
    path = plot_surfaces(data, show=args.show)
    print(f"已写出 {path}")


if __name__ == "__main__":
    main()
