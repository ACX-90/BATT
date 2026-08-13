"""按起始 SOC × 温度网格批量生成 ECM 仿真波形。

默认 5×5：SOC 从低到高 5 档，温度从低到高 5 档，共 25 份 CSV。
每份沿用 nmc100ah_ecm_gen.py 的充/放/静置指令序列，仅改初值。

用法（仓库根目录）：

    python Src/Sim/nmc100ah_ecm_gen_grid.py
    python Src/Sim/nmc100ah_ecm_gen_grid.py --n-soc 5 --n-temp 5
    python Src/Sim/nmc100ah_ecm_gen_grid.py --dry-run
"""

from __future__ import annotations

# =============================================================================
# 头部配置
# =============================================================================

# 网格密度。5×5 → 25 份波形
N_SOC = 5
N_TEMP = 5

# 起始 SOC 扫描区间（含端点；n=1 时取中点）
SOC_MIN = 0.10
SOC_MAX = 0.90

# 温度扫描区间 / °C（含端点；n=1 时取中点）
T_MIN_C = -10.0
T_MAX_C = 50.0

# 若给出列表则不再按 MIN/MAX 均分，档数以列表长度为准
# 例如：SOC_VALUES = [0.15, 0.40, 0.70] ; T_VALUES_C = [-10, 25, 45]
SOC_VALUES: list[float] | None = None
T_VALUES_C: list[float] | None = None

# 输出目录（相对仓库根目录）
OUTPUT_DIR = "Data/grid"

# 文件名：s 为 SOC 档位，t 为温度档位
# 例：nmc100ah_ecm_s00_t02_soc010_T+20.csv
FILE_NAME = "nmc100ah_ecm_s{i:02d}_t{j:02d}_soc{soc_pct:03.0f}_T{t_c:+03.0f}.csv"

# =============================================================================
# 以下为扫描实现
# =============================================================================

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from nmc100ah_ecm import NMC100AhECM  # noqa: E402
from nmc100ah_ecm_gen import (  # noqa: E402
    DT_S,
    ENABLE_CUTOFF,
    NOISE_ENABLE,
    NOISE_SEED,
    NOISE_STD,
    SEQUENCE,
    U_P0,
    REPO_ROOT,
    simulate,
    write_csv,
)


def linspace_axis(lo: float, hi: float, n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("档位数必须 >= 1")
    if n == 1:
        return np.array([(lo + hi) / 2.0], dtype=float)
    return np.linspace(lo, hi, n, dtype=float)


def case_name(i: int, j: int, soc: float, t_c: float) -> str:
    return FILE_NAME.format(i=i, j=j, soc_pct=soc * 100.0, t_c=t_c)


def build_grid(n_soc: int, n_temp: int) -> tuple[np.ndarray, np.ndarray]:
    soc = np.asarray(SOC_VALUES, dtype=float) if SOC_VALUES is not None else linspace_axis(SOC_MIN, SOC_MAX, n_soc)
    temp = np.asarray(T_VALUES_C, dtype=float) if T_VALUES_C is not None else linspace_axis(T_MIN_C, T_MAX_C, n_temp)
    if soc.size < 1 or temp.size < 1:
        raise ValueError("SOC / 温度网格不能为空")
    return soc, temp


def run_grid(
    *,
    n_soc: int = N_SOC,
    n_temp: int = N_TEMP,
    output_dir: str | Path = OUTPUT_DIR,
    noise_enable: bool = NOISE_ENABLE,
    noise_seed: int = NOISE_SEED,
    dry_run: bool = False,
) -> list[dict]:
    soc_axis, t_axis = build_grid(n_soc, n_temp)
    n_soc, n_temp = int(soc_axis.size), int(t_axis.size)
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    model = None if dry_run else NMC100AhECM()
    rows: list[dict] = []

    print(f"网格 {n_soc}×{n_temp} = {n_soc * n_temp} 份")
    print("SOC : " + ", ".join(f"{s:.3f}" for s in soc_axis))
    print("T   : " + ", ".join(f"{t:+.1f} °C" for t in t_axis))
    if dry_run:
        print("dry-run，不写文件")

    idx = 0
    for i, soc0 in enumerate(soc_axis):
        for j, t_c in enumerate(t_axis):
            fname = case_name(i, j, float(soc0), float(t_c))
            rel = Path(OUTPUT_DIR if not Path(output_dir).is_absolute() else out_dir) / fname
            seed = int(noise_seed) + i * 100 + j
            rec = {
                "idx": idx,
                "i_soc": i,
                "j_temp": j,
                "soc0": float(soc0),
                "t_celsius": float(t_c),
                "noise_seed": seed,
                "file": fname,
                "path": str(rel).replace("\\", "/"),
            }

            if dry_run:
                print(f"  [{idx:02d}] SOC={soc0:.3f}  T={t_c:+6.1f} °C  -> {fname}")
                rec.update(n_steps=0, duration_s=0.0, soc_end=float(soc0), ut_end=float("nan"), cutoff_steps=0)
                rows.append(rec)
                idx += 1
                continue

            assert model is not None
            data = simulate(
                model,
                SEQUENCE,
                dt_s=DT_S,
                soc0=float(soc0),
                t_ambient_c=float(t_c),
                u_p0=U_P0,
                noise_enable=noise_enable,
                noise_seed=seed,
                noise_std=NOISE_STD,
                enable_cutoff=ENABLE_CUTOFF,
            )
            csv_path = write_csv(
                out_dir / fname,
                data,
                dt_s=DT_S,
                soc0=float(soc0),
                noise_enable=noise_enable,
                noise_seed=seed,
                noise_std=NOISE_STD,
                sequence=SEQUENCE,
                extra_meta=[
                    "# source=nmc100ah_ecm_gen_grid",
                    f"# t0_c={t_c}",
                    f"# grid_i_soc={i}",
                    f"# grid_j_temp={j}",
                    f"# grid_n_soc={n_soc}",
                    f"# grid_n_temp={n_temp}",
                ],
            )
            rec.update(
                n_steps=int(len(data["time_s"])),
                duration_s=float(data["time_s"][-1] + DT_S),
                soc_end=float(data["soc_true"][-1]),
                ut_end=float(data["u_t_true_v"][-1]),
                cutoff_steps=int(np.sum(data["cutoff"] > 0)),
                path=str(csv_path.relative_to(REPO_ROOT)).replace("\\", "/")
                if csv_path.is_relative_to(REPO_ROOT)
                else str(csv_path).replace("\\", "/"),
            )
            print(
                f"  [{idx:02d}] SOC {soc0:.3f}->{rec['soc_end']:.3f}  "
                f"T={t_c:+6.1f} °C  Ut={rec['ut_end']:.3f} V  {fname}"
            )
            rows.append(rec)
            idx += 1

    if not dry_run:
        _write_index(out_dir / "index.csv", rows)
        print(f"索引  {out_dir / 'index.csv'}")
    return rows


def _write_index(path: Path, rows: list[dict]) -> None:
    fields = [
        "idx",
        "i_soc",
        "j_temp",
        "soc0",
        "t_celsius",
        "noise_seed",
        "n_steps",
        "duration_s",
        "soc_end",
        "ut_end",
        "cutoff_steps",
        "file",
        "path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="按 SOC×温度网格批量生成 ECM 波形 CSV")
    parser.add_argument("--n-soc", type=int, default=N_SOC, help="起始 SOC 档数")
    parser.add_argument("--n-temp", type=int, default=N_TEMP, help="温度档数")
    parser.add_argument("--out-dir", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--seed", type=int, default=None, help="覆盖噪声种子")
    parser.add_argument("--no-noise", action="store_true", help="关闭噪声")
    parser.add_argument("--dry-run", action="store_true", help="只打印网格，不仿真")
    args = parser.parse_args()

    run_grid(
        n_soc=args.n_soc,
        n_temp=args.n_temp,
        output_dir=args.out_dir,
        noise_enable=NOISE_ENABLE and not args.no_noise,
        noise_seed=NOISE_SEED if args.seed is None else args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
