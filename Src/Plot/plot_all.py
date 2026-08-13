"""一次生成参数曲面图和仿真波形图。

用法（仓库根目录）：

    python Src/Plot/plot_all.py
    python Src/Plot/plot_all.py --show
"""

from __future__ import annotations

import argparse

from plot_ecm_surfaces import build_surfaces, plot_surfaces
from plot_sim_waveforms import plot_waveforms
from _common import DEFAULT_SIM_CSV, load_sim_csv
from nmc100ah_ecm import NMC100AhECM


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 ECM 曲面与仿真波形")
    parser.add_argument("--csv", default=str(DEFAULT_SIM_CSV), help="仿真 CSV 路径")
    parser.add_argument("--show", action="store_true", help="弹窗显示")
    args = parser.parse_args()

    surf = plot_surfaces(build_surfaces(NMC100AhECM()), show=args.show)
    print(f"曲面  {surf}")

    wave = plot_waveforms(load_sim_csv(args.csv), show=args.show)
    print(f"波形  {wave}")


if __name__ == "__main__":
    main()
