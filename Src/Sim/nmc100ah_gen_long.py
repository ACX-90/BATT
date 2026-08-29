"""小时级负例（任务 F）。默认 SEQUENCE 不动，工况从这里传入 run_sim。

    python Src/Sim/nmc100ah_gen_long.py
    python Src/Sim/nmc100ah_gen_long.py --only cc_rest
    python Src/Sim/nmc100ah_gen_long.py --only schg_park --pybamm
    --only 是指定生成序列，可选：
        cc_rest     放电静置
        chg_park    充电静置
        fchg_park   快充静置
        schg_park   超充静置（含 chg_ramp 坡度电流）
        loop        循环
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from nmc100ah_gen import run_sim

# 负例 A：0.3C 放电 2 h + 静置 30 min。SOC0=0.70，约放到 0.10。
SEQ_CC_REST = [
    {"mode": "rest", "duration_s": 30.0},
    {"mode": "discharge", "duration_s": 7200.0, "c_rate": 0.3},
    {"mode": "rest", "duration_s": 1800.0},
]

# 负例 B：1C 充电 40 min + 停车 2 h。
SEQ_CHG_PARK = [
    {"mode": "rest", "duration_s": 30.0},
    {"mode": "charge", "duration_s": 2400.0, "c_rate": 1.0},
    {"mode": "rest", "duration_s": 7200.0},
]

# 负例 B：2C 充电 20 min + 停车 1 h。
SEQ_FCHG_PARK = [
    {"mode": "rest", "duration_s": 30.0},
    {"mode": "charge", "duration_s": 1200.0, "c_rate": 2.0},
    {"mode": "rest", "duration_s": 3600.0},
]

# 负例 B：2.9C 自定义超充序列（注意可能触发保护）
SEQ_SCHG_PARK = [
    {"mode": "rest", "duration_s": 30.0}, # 0.1
    {"mode": "chg_ramp", "duration_s": 60, "c_rate_start": 0.0, "c_rate_end": 2.9}, # 0.1-0.124
    {"mode": "charge", "duration_s": 590, "c_rate": 2.9}, # 0.124-0.6
    {"mode": "chg_ramp", "duration_s": 370, "c_rate_start": 2.9, "c_rate_end": 1}, # 0.6-0.8
    {"mode": "chg_ramp", "duration_s": 600, "c_rate_start": 1, "c_rate_end": 0.3}, # 0.9+
    {"mode": "rest", "duration_s": 600.0},
]

# 自定义放电序列
# Reserved

# 自定义WLTC
# Reserved


SEQ_CHG_DIS_LOOP = [ {"mode": "rest", "duration_s": 30.0}, ] + \
[
    {"mode": "charge", "duration_s": 300, "c_rate": 1.0},
    {"mode": "rest", "duration_s": 200},
] * 10 + \
[
    {"mode": "rest", "duration_s": 1000.0},
] + \
[
    {"mode": "discharge", "duration_s": 300, "c_rate": 1.0},
    {"mode": "rest", "duration_s": 200},
] * 10
SEQ_CHG_DIS_LOOP = SEQ_CHG_DIS_LOOP * 3

CASES = {
    "cc_rest": {
        "out": "Data/long/cc_rest.csv",
        "sequence": SEQ_CC_REST,
        "soc0": 0.70,
        "seed": 20260831,
        "tag": "F-cc_rest",
    },
    "chg_park": {
        "out": "Data/long/chg_park.csv",
        "sequence": SEQ_CHG_PARK,
        "soc0": 0.30,
        "seed": 20260832,
        "tag": "F-chg_park",
    },
    "fchg_park": {
        "out": "Data/long/fchg_park.csv",
        "sequence": SEQ_FCHG_PARK,
        "soc0": 0.20,
        "seed": 20260833,
        "tag": "F-Fchg_park",
    },
    "schg_park": {
        "out": "Data/long/schg_park.csv",
        "sequence": SEQ_SCHG_PARK,
        "soc0": 0.10,
        "seed": 20260834,
        "tag": "F-Schg_park",
    },
    "loop": {
        "out": "Data/long/loop.csv",
        "sequence": SEQ_CHG_DIS_LOOP,
        "soc0": 0.10,
        "seed": 20260833,
        "tag": "F-loop",
    }
}


def main() -> None:
    p = argparse.ArgumentParser(description="任务 F：写出小时级负例 CSV，不改默认 SEQUENCE")
    p.add_argument("--only", choices=tuple(CASES), default=None)
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--rc2", action="store_true", help="叠加慢支路 R2C2，BMS 仍 1RC")
    p.add_argument(
        "--pybamm",
        action="store_true",
        help="用 PyBaMM SPM 出电压；chg_ramp / dis_ramp 走 drive cycle，默认写 Data/long_pybamm/",
    )
    p.add_argument(
        "--thermal",
        action="store_true",
        help="仅 --pybamm：打开 lumped 热模型（默认等温）",
    )
    p.add_argument("--out-dir", default=None, help="覆盖输出目录")
    args = p.parse_args()
    if args.thermal and not args.pybamm:
        p.error("--thermal 只能与 --pybamm 一起用")
    if args.pybamm and args.rc2:
        p.error("--pybamm 不能与 --rc2 一起用")
    out_dir = args.out_dir
    if out_dir is None and args.pybamm:
        out_dir = "Data/long_pybamm"
    names = [args.only] if args.only else list(CASES)
    for name in names:
        spec = CASES[name]
        out = spec["out"]
        if out_dir:
            out = str(Path(out_dir) / Path(spec["out"]).name)
        extra = [
            f"# source=nmc100ah_gen_long",
            f"# case={name}",
            f"# tag={spec['tag']}",
        ]
        print(f"======== {name}  {spec['tag']} ========")
        if args.pybamm:
            from nmc100ah_pybamm import run_sim as run_pybamm

            run_pybamm(
                out=out,
                sequence=spec["sequence"],
                soc0=spec["soc0"],
                noise_seed=spec["seed"],
                noise_enable=False if args.no_noise else None,
                thermal=args.thermal,
                extra_meta=extra,
            )
        else:
            run_sim(
                out=out,
                sequence=spec["sequence"],
                soc0=spec["soc0"],
                noise_seed=spec["seed"],
                noise_enable=False if args.no_noise else None,
                rc2=args.rc2,
                extra_meta=extra,
            )


if __name__ == "__main__":
    main()
