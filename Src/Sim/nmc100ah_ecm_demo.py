"""100 Ah NMC ECM 参数模型示例。

用法（仓库根目录）：

    python Src/Sim/nmc100ah_ecm_demo.py
    python Src/Sim/nmc100ah_ecm_demo.py --csv Doc/NMC100Ah_ECM_lookup.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from nmc100ah_ecm import NMC100AhECM  # noqa: E402


def _print_point(model: NMC100AhECM, i_a: float, t_c: float, soc: float) -> None:
    r = model.evaluate_full(i_a=i_a, t_celsius=t_c, soc=soc)
    print(
        f"  I={i_a:+7.1f} A  T={t_c:+5.1f} °C  SOC={soc:5.2f}  "
        f"R0={r.R0_mohm:6.3f} mΩ  R1={r.R1_mohm:6.3f} mΩ  "
        f"C1={r.C1:8.1f} F  τ1={r.tau1:6.2f} s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NMC 100Ah ECM 参数映射示例")
    parser.add_argument("--csv", type=str, default="", help="导出查找表 CSV 路径")
    parser.add_argument("--json", type=str, default="", help="把默认参数写出为 JSON")
    args = parser.parse_args()

    model = NMC100AhECM()
    p = model.params

    print(f"电芯: {p.cell.name}  {p.cell.capacity_ah:.0f} Ah  {p.cell.chemistry}")
    print(
        "参考点: "
        f"SOC={p.reference.soc:.0%}  T={p.reference.t_celsius:.0f}°C  "
        f"I={p.reference.i_a:.0f} A 放电"
    )
    print(
        "参考值: "
        f"R0={p.r0.ref_value*1e3:.3f} mΩ  "
        f"R1={p.r1.ref_value*1e3:.3f} mΩ  "
        f"C1={p.c1.ref_value:.1f} F"
    )
    print()

    print("参考点复核（因子应全为 1，输出应等于参考值）：")
    ref = model.evaluate_full(i_a=100.0, t_celsius=25.0, soc=0.5)
    print(
        f"  R0={ref.R0_mohm:.6f} mΩ  R1={ref.R1_mohm:.6f} mΩ  "
        f"C1={ref.C1:.4f} F  τ1={ref.tau1:.4f} s"
    )
    assert ref.f_r0 is not None
    print("  R0 因子:", {k: f"{v:.6f}" for k, v in ref.f_r0.items()})
    print()

    print("典型工况：")
    cases = [
        (100.0, 25.0, 0.50),
        (100.0, 25.0, 0.10),
        (100.0, 25.0, 0.90),
        (100.0, 0.0, 0.50),
        (100.0, -10.0, 0.20),
        (200.0, 25.0, 0.50),
        (-100.0, 25.0, 0.90),
        (50.0, 40.0, 0.80),
    ]
    for i_a, t_c, soc in cases:
        _print_point(model, i_a, t_c, soc)

    print()
    print("支持百分数 SOC 与 C 率输入：")
    r0, r1, c1 = model.evaluate(i_c=1.0, t_celsius=25.0, soc=50, soc_in_percent=True)
    print(f"  evaluate(i_c=1, T=25, SOC=50%) -> R0={r0*1e3:.3f} mΩ, R1={r1*1e3:.3f} mΩ, C1={c1:.1f} F")

    if args.json:
        model.params.to_json(args.json)
        print(f"\n已写出参数 JSON: {args.json}")

    if args.csv:
        soc = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
        t_celsius = [-20.0, -10.0, 0.0, 15.0, 25.0, 40.0, 55.0]
        i_a = [-200.0, -100.0, 50.0, 100.0, 200.0]
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        model.export_csv(args.csv, soc=soc, t_celsius=t_celsius, i_a=i_a)
        print(f"\n已写出查找表: {args.csv}  ({len(soc)*len(t_celsius)*len(i_a)} 行)")


if __name__ == "__main__":
    main()
