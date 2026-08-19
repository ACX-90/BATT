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
import json
import sys
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from nmc100ah_ecm import make_ecm  # noqa: E402
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


def clear_output_dir(out_dir: Path) -> int:
    """删掉输出目录里已有的网格 CSV，避免换档数后旧文件残留。"""
    if not out_dir.is_dir():
        return 0
    removed = 0
    for path in out_dir.glob("*.csv"):
        path.unlink()
        removed += 1
    return removed


def run_grid(
    *,
    n_soc: int = N_SOC,
    n_temp: int = N_TEMP,
    output_dir: str | Path = OUTPUT_DIR,
    noise_enable: bool = NOISE_ENABLE,
    noise_seed: int = NOISE_SEED,
    dry_run: bool = False,
    r0_scale: float = 1.0,
    r1_scale: float = 1.0,
    c1_scale: float = 1.0,
    sequence: list[dict] | None = None,
    soh: float = 1.0,
    soh_a0: float = 1.0,
    soh_a1: float = 1.5,
    soh_ac: float = -0.3,
    soh_capacity: float = 1.0,
    rc2: bool = False,
) -> list[dict]:
    seq = SEQUENCE if sequence is None else sequence
    soc_axis, t_axis = build_grid(n_soc, n_temp)
    n_soc, n_temp = int(soc_axis.size), int(t_axis.size)
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    if not dry_run:
        n_old = clear_output_dir(out_dir)
        if n_old:
            print(f"已删除旧文件 {n_old} 份  {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        model = None
    else:
        model = make_ecm(
            r0_scale=r0_scale,
            r1_scale=r1_scale,
            c1_scale=c1_scale,
            soh=soh,
            soh_a0=soh_a0,
            soh_a1=soh_a1,
            soh_ac=soh_ac,
            soh_capacity=soh_capacity,
            rc2=rc2,
        )
    rows: list[dict] = []

    print(f"网格 {n_soc}×{n_temp} = {n_soc * n_temp} 份")
    print("SOC : " + ", ".join(f"{s:.3f}" for s in soc_axis))
    print("T   : " + ", ".join(f"{t:+.1f} °C" for t in t_axis))
    if r0_scale != 1.0 or r1_scale != 1.0 or c1_scale != 1.0:
        print(f"电阻缩放  R0×{r0_scale:g}  R1×{r1_scale:g}  C1×{c1_scale:g}")
    if abs(soh - 1.0) > 1e-12 or abs(soh_capacity - 1.0) > 1e-12:
        print(f"SOH  q={soh:g}  q_Q={soh_capacity:g}  a0={soh_a0:g} a1={soh_a1:g} aC={soh_ac:g}")
    if rc2:
        print("2RC  叠加慢支路（CSV 多 r2/c2/up2；BMS 不读）")
    if dry_run:
        print("dry-run，不写文件")

    idx = 0
    for i, soc0 in enumerate(soc_axis):
        for j, t_c in enumerate(t_axis):
            fname = case_name(i, j, float(soc0), float(t_c))
            try:
                rel = (out_dir.relative_to(REPO_ROOT) / fname)
            except ValueError:
                rel = out_dir / fname
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
                seq,
                dt_s=DT_S,
                soc0=float(soc0),
                t_ambient_c=float(t_c),
                u_p0=U_P0,
                noise_enable=noise_enable,
                noise_seed=seed,
                noise_std=NOISE_STD,
                enable_cutoff=ENABLE_CUTOFF,
                rc2=rc2,
            )
            csv_path = write_csv(
                out_dir / fname,
                data,
                dt_s=DT_S,
                soc0=float(soc0),
                noise_enable=noise_enable,
                noise_seed=seed,
                noise_std=NOISE_STD,
                sequence=seq,
                extra_meta=[
                    "# source=nmc100ah_ecm_gen_grid",
                    f"# t0_c={t_c}",
                    f"# grid_i_soc={i}",
                    f"# grid_j_temp={j}",
                    f"# grid_n_soc={n_soc}",
                    f"# grid_n_temp={n_temp}",
                    f"# r0_scale={r0_scale}",
                    f"# r1_scale={r1_scale}",
                    f"# c1_scale={c1_scale}",
                    "# schema_version=1.1.0",
                    f"# soh={soh}",
                    f"# soh_capacity={soh_capacity}",
                    f"# rc2={int(rc2)}",
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
        meta = {
            "schema_version": "1.1.0",
            "soh": soh,
            "soh_capacity": soh_capacity,
            "a_r0": soh_a0,
            "a_r1": soh_a1,
            "a_c1": soh_ac,
            "rc2": rc2,
            "r0_scale": r0_scale,
            "r1_scale": r1_scale,
            "c1_scale": c1_scale,
            "ocv_aging": False,
        }
        if hasattr(model, "meta"):
            meta.update(model.meta())
        (out_dir / "ecm_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
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
    parser.add_argument("--r0-scale", type=float, default=1.0, help="整张 R0 乘子（假老化 / 换对象）")
    parser.add_argument("--r1-scale", type=float, default=1.0, help="整张 R1 乘子")
    parser.add_argument("--c1-scale", type=float, default=1.0, help="整张 C1 乘子")
    parser.add_argument("--soh", type=float, default=1.0, help="电阻/电容寿命因子 q，1=BOL")
    parser.add_argument("--soh-a0", type=float, default=1.0, help="R0 的 a0，q=0.9 且 a0=1 → ×1.10")
    parser.add_argument("--soh-a1", type=float, default=1.5, help="R1 的 a1，略大于 a0")
    parser.add_argument("--soh-ac", type=float, default=-0.3, help="C1 的 aC，应 <0")
    parser.add_argument("--soh-capacity", type=float, default=1.0, help="容量因子 q_Q，第一次不要和 q 叠")
    parser.add_argument("--rc2", action="store_true", help="叠加慢支路，BMS 仍 1RC")
    args = parser.parse_args()

    run_grid(
        n_soc=args.n_soc,
        n_temp=args.n_temp,
        output_dir=args.out_dir,
        noise_enable=NOISE_ENABLE and not args.no_noise,
        noise_seed=NOISE_SEED if args.seed is None else args.seed,
        dry_run=args.dry_run,
        r0_scale=args.r0_scale,
        r1_scale=args.r1_scale,
        c1_scale=args.c1_scale,
        soh=args.soh,
        soh_a0=args.soh_a0,
        soh_a1=args.soh_a1,
        soh_ac=args.soh_ac,
        soh_capacity=args.soh_capacity,
        rc2=args.rc2,
    )


if __name__ == "__main__":
    main()
