"""小时级负例（任务 F）。默认 SEQUENCE 不动，工况从这里传入 run_sim。

    python Src/Sim/nmc100ah_ecm_gen_long.py
    python Src/Sim/nmc100ah_ecm_gen_long.py --only cc_rest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from nmc100ah_ecm_gen import run_sim

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
    args = p.parse_args()
    names = [args.only] if args.only else list(CASES)
    for name in names:
        spec = CASES[name]
        print(f"======== {name}  {spec['tag']} ========")
        run_sim(
            out=spec["out"],
            sequence=spec["sequence"],
            soc0=spec["soc0"],
            noise_seed=spec["seed"],
            noise_enable=False if args.no_noise else None,
            extra_meta=[f"# source=nmc100ah_ecm_gen_long", f"# case={name}", f"# tag={spec['tag']}"],
        )


if __name__ == "__main__":
    main()
