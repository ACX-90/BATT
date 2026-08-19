"""绘制 ECM 时域仿真波形。

默认读取 Data/nmc100ah_ecm_sim.csv。

用法（仓库根目录）：

    python Src/Plot/plot_sim_waveforms.py
    python Src/Plot/plot_sim_waveforms.py --csv Data/nmc100ah_ecm_sim.csv --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from _common import DEFAULT_SIM_CSV, FIG_DIR, apply_style, load_sim_csv, mode_spans, plot_overlay, save_figure

MODE_FACE = {
    "rest": (0.72, 0.72, 0.72, 0.18),
    "discharge": (0.86, 0.32, 0.24, 0.16),
    "charge": (0.24, 0.48, 0.82, 0.16),
}
MODE_LABEL = {"rest": "静置", "discharge": "放电", "charge": "充电"}


def _shade_modes(axes, time_s: np.ndarray, modes: np.ndarray) -> None:
    spans = mode_spans(time_s, modes)
    for ax in axes:
        for mode, t0, t1 in spans:
            ax.axvspan(t0, t1, color=MODE_FACE.get(mode, (0.8, 0.8, 0.8, 0.12)), lw=0)


def plot_waveforms(data: dict[str, np.ndarray], *, show: bool = False) -> Path:
    apply_style()
    t = data["time_s"]
    modes = data["mode"]

    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(12.5, 13.5), constrained_layout=True)
    fig.suptitle("NMC 100Ah ECM 仿真波形", fontsize=13)
    _shade_modes(axes, t, modes)

    plot_overlay(axes[0], t, data["i_true_a"], "truth", color="#1f4e79", label="电流真值")
    plot_overlay(axes[0], t, data["i_meas_a"], "meas", color="#7aa6d6", label="电流测量")
    axes[0].set_ylabel("电流 / A")

    plot_overlay(axes[1], t, data["u_ocv_v"], "ol", color="#6b6b6b", label="OCV")
    plot_overlay(axes[1], t, data["u_t_true_v"], "truth", color="#9c2a2a", label="端电压真值")
    plot_overlay(axes[1], t, data["u_t_meas_v"], "meas", color="#e08a7a", label="端电压测量")
    axes[1].set_ylabel("电压 / V")
    axes[1].legend(loc="upper right", ncol=3)

    plot_overlay(axes[2], t, data["soc_true"] * 100.0, "truth", color="#2e7d32", label="真值")
    plot_overlay(axes[2], t, data["soc_meas"] * 100.0, "meas", color="#81c784", label="测量")
    axes[2].set_ylabel("SOC / %")
    axes[2].legend(loc="upper right", ncol=2)

    plot_overlay(axes[3], t, data["u_p_v"] * 1e3, "truth", color="#6a1b9a", label="$U_{p1}$")
    if "u_p2_v" in data:
        plot_overlay(axes[3], t, data["u_p2_v"] * 1e3, "ol", color="#8e24aa", label="$U_{p2}$")
        axes[3].legend(loc="upper right", ncol=2)
    axes[3].set_ylabel("$U_p$ / mV")

    plot_overlay(axes[4], t, data["r0_ohm"] * 1e3, "truth", color="#1565c0", label="$R_0$")
    plot_overlay(axes[4], t, data["r1_ohm"] * 1e3, "ol", color="#ef6c00", label="$R_1$")
    if "r2_ohm" in data:
        plot_overlay(axes[4], t, data["r2_ohm"] * 1e3, "pri", color="#6a1b9a", label="$R_2$")
    axes[4].set_ylabel("电阻 / mΩ")
    axes[4].legend(loc="upper right", ncol=3)

    axes[5].plot(t, data["c1_f"] * 1e-3, color="#00838f", lw=1.1)
    axes[5].set_ylabel("$C_1$ / kF")

    axes[6].plot(t, data["tau1_s"], color="#4e342e", lw=1.1)
    axes[6].set_ylabel("$\\tau_1$ / s")
    axes[6].set_xlabel("时间 / s")

    mode_handles = [
        Patch(facecolor=MODE_FACE[k], edgecolor="none", label=MODE_LABEL[k])
        for k in ("rest", "discharge", "charge")
    ]
    line_handles, line_labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        line_handles + mode_handles,
        line_labels + [MODE_LABEL[k] for k in ("rest", "discharge", "charge")],
        loc="upper right",
        ncol=5,
    )

    for ax in axes:
        ax.margins(x=0.01)

    return save_figure(fig, FIG_DIR / "nmc100ah_ecm_waveforms.png", show=show)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 ECM 仿真波形")
    parser.add_argument("--csv", default=str(DEFAULT_SIM_CSV), help="仿真 CSV 路径")
    parser.add_argument("--show", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    data = load_sim_csv(args.csv)
    path = plot_waveforms(data, show=args.show)
    print(f"已写出 {path}")
    print(
        f"  时长 {data['time_s'][-1]:.1f} s  "
        f"SOC {data['soc_true'][0]:.3f}->{data['soc_true'][-1]:.3f}  "
        f"Ut {data['u_t_true_v'][0]:.3f}->{data['u_t_true_v'][-1]:.3f} V"
    )


if __name__ == "__main__":
    main()
