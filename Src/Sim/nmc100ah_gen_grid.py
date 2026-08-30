"""按起始 SOC × 温度网格批量生成仿真波形。默认 ECM，`--pybamm` 换 SPM。

默认 5×5：SOC 从低到高 5 档，温度从低到高 5 档，共 25 份 CSV。
每份沿用 nmc100ah_gen.py 的充/放/静置指令序列，仅改初值。

用法（仓库根目录）：

    python Src/Sim/nmc100ah_gen_grid.py
    python Src/Sim/nmc100ah_gen_grid.py --n-soc 5 --n-temp 5
    python Src/Sim/nmc100ah_gen_grid.py --dry-run
    python Src/Sim/nmc100ah_gen_grid.py --pybamm --out-dir Data/grid_pybamm
"""

from __future__ import annotations

# =============================================================================
# 头部配置
# =============================================================================

# 网格密度。5×5 → 25 份波形
N_SOC = 5
N_TEMP = 5

# 起始 SOC 扫描区间（含端点；n=1 时取中点）
# SOC_MIN = 0.10
# SOC_MAX = 0.90
SOC_MIN = 0.9
SOC_MAX = 0.1

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
FILE_NAME_PYBAMM = "nmc100ah_pybamm_s{i:02d}_t{j:02d}_soc{soc_pct:03.0f}_T{t_c:+03.0f}.csv"

# --pybamm：本格 1x 起步。充电序列触发保护 → 本格充电电流 ×0.7 重跑；
# 还触发就 ×0.7²、×0.7³… 直到通过。放电同理、独立计数。下一格仍从 1x 开始。
# PROTECT_RETRY_MAX 只防步初已超压时缩电流也过不了。
PROTECT_RETRY_SCALE = 0.7
PROTECT_RETRY_MAX = 12

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
from nmc100ah_gen import (  # noqa: E402
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


def case_name(i: int, j: int, soc: float, t_c: float, *, file_name: str | None = None) -> str:
    pattern = FILE_NAME if file_name is None else file_name
    return pattern.format(i=i, j=j, soc_pct=soc * 100.0, t_c=t_c)


_CHG_MODES = {"charge", "chg", "cha", "chg_ramp"}
_DIS_MODES = {"discharge", "dch", "dis", "dis_ramp"}
_CURRENT_KEYS = ("c_rate", "current_a", "c_rate_start", "c_rate_end")


def _is_unity(scale: float) -> bool:
    return abs(float(scale) - 1.0) < 1e-12


def scale_sequence_current(
    sequence: list[dict],
    chg_s: float,
    dis_s: float,
) -> list[dict]:
    """复制工况，按充/放乘子缩放电流；rest 不动。乘子相对原始 SEQUENCE（累计 0.7^n）。"""
    chg_s = float(chg_s)
    dis_s = float(dis_s)
    if _is_unity(chg_s) and _is_unity(dis_s):
        return list(sequence)
    out: list[dict] = []
    for cmd in sequence:
        mode = str(cmd.get("mode", "")).strip().lower()
        if mode in _CHG_MODES:
            scale = chg_s
        elif mode in _DIS_MODES:
            scale = dis_s
        else:
            scale = 1.0
        if _is_unity(scale):
            out.append(cmd)
            continue
        item = dict(cmd)
        for key in _CURRENT_KEYS:
            if key in item:
                item[key] = float(item[key]) * scale
        out.append(item)
    return out


def polarity_cutoff(data: dict, sequence: list[dict]) -> tuple[bool, bool]:
    """本轮仿真里充电 / 放电序列是否触发了保护。"""
    cutoff = np.asarray(data["cutoff"], dtype=float)
    cmd_ids = np.rint(np.asarray(data["cmd_id"], dtype=float)).astype(int)
    chg_hit = False
    dis_hit = False
    for idx, cmd in enumerate(sequence):
        mode = str(cmd.get("mode", "")).strip().lower()
        if mode in _CHG_MODES:
            charge = True
        elif mode in _DIS_MODES:
            charge = False
        else:
            continue
        if not np.any((cmd_ids == idx) & (cutoff > 0)):
            continue
        if charge:
            chg_hit = True
        else:
            dis_hit = True
    return chg_hit, dis_hit


def simulate_pybamm_with_protect_retry(
    simulate_fn,
    sequence: list[dict],
    *,
    idx: int,
    dt_s: float,
    soc0: float,
    t_ambient_c: float,
    noise_enable: bool,
    noise_seed: int,
    noise_std: dict[str, float],
    enable_cutoff: bool,
    thermal: bool,
    verbose: bool,
) -> tuple[dict[str, np.ndarray], list[dict], float, float]:
    """本格 1x 起步；充/放各自触发保护则 ×0.7、×0.7²、×0.7³… 直到通过。"""
    chg_s, dis_s = 1.0, 1.0
    n_chg = n_dis = 0
    first = True
    data: dict[str, np.ndarray] | None = None
    seq_k = list(sequence)
    while True:
        seq_k = scale_sequence_current(sequence, chg_s, dis_s)
        data = simulate_fn(
            seq_k,
            dt_s=dt_s,
            soc0=soc0,
            t_ambient_c=t_ambient_c,
            noise_enable=noise_enable,
            noise_seed=noise_seed,
            noise_std=noise_std,
            enable_cutoff=enable_cutoff,
            thermal=thermal,
            verbose=verbose and first,
        )
        first = False
        hit_chg, hit_dis = polarity_cutoff(data, sequence)
        retry_chg = hit_chg and n_chg < PROTECT_RETRY_MAX
        retry_dis = hit_dis and n_dis < PROTECT_RETRY_MAX
        if not (retry_chg or retry_dis):
            if hit_chg or hit_dis:
                stuck = []
                if hit_chg:
                    stuck.append(f"充电仍保护×{chg_s:g}")
                if hit_dis:
                    stuck.append(f"放电仍保护×{dis_s:g}")
                print(f"  [{idx:02d}] 保护重跑达上限  {' '.join(stuck)}")
            return data, seq_k, chg_s, dis_s
        bits = []
        if retry_chg:
            n_chg += 1
            chg_s *= PROTECT_RETRY_SCALE
            bits.append(f"充电×{chg_s:g}")
        if retry_dis:
            n_dis += 1
            dis_s *= PROTECT_RETRY_SCALE
            bits.append(f"放电×{dis_s:g}")
        print(f"  [{idx:02d}] 保护重跑  {' '.join(bits)}")


def _scale_tag(chg_s: float, dis_s: float) -> str:
    bits = []
    if chg_s != 1.0:
        bits.append(f"chg×{chg_s:g}")
    if dis_s != 1.0:
        bits.append(f"dis×{dis_s:g}")
    return ("  " + " ".join(bits)) if bits else ""


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
    noise_std: dict[str, float] | None = None,
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
    pybamm: bool = False,
    thermal: bool = False,
) -> list[dict]:
    seq = SEQUENCE if sequence is None else sequence
    std = dict(NOISE_STD if noise_std is None else noise_std)
    file_name = FILE_NAME_PYBAMM if pybamm else FILE_NAME
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

    simulate_pybamm = None
    if pybamm and not dry_run:
        from nmc100ah_pybamm import simulate as simulate_pybamm
    if dry_run or pybamm:
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
    if pybamm:
        print("backend  PyBaMM SPM  Chen2020×100Ah" + ("  lumped thermal" if thermal else "  等温"))
    if noise_enable:
        print(
            "噪声  "
            f"U={std.get('voltage_v', 0.0)*1e3:.2f} mV  "
            f"I={std.get('current_a', 0.0)*1e3:.1f} mA  "
            f"T={std.get('temp_c', 0.0):.3f} °C  "
            f"SOC={std.get('soc', 0.0)*1e2:.3f} pp"
        )
    else:
        print("噪声  关闭")
    if r0_scale != 1.0 or r1_scale != 1.0 or c1_scale != 1.0:
        print(f"电阻缩放  R0×{r0_scale:g}  R1×{r1_scale:g}  C1×{c1_scale:g}")
    if abs(soh - 1.0) > 1e-12 or abs(soh_capacity - 1.0) > 1e-12:
        print(f"SOH  q={soh:g}  q_Q={soh_capacity:g}  a0={soh_a0:g} a1={soh_a1:g} aC={soh_ac:g}")
    if rc2:
        print("2RC  叠加慢支路（CSV 多 r2/c2/up2；BMS 不读）")
    if pybamm:
        print(
            f"保护重跑  充电/放电触发保护则本格对应电流×{PROTECT_RETRY_SCALE:g}、"
            f"×{PROTECT_RETRY_SCALE:g}^2… 直到通过；下一格仍 1x"
        )
    if dry_run:
        print("dry-run，不写文件")

    idx = 0
    for i, soc0 in enumerate(soc_axis):
        for j, t_c in enumerate(t_axis):
            fname = case_name(i, j, float(soc0), float(t_c), file_name=file_name)
            try:
                rel = (out_dir.relative_to(REPO_ROOT) / fname)
            except ValueError:
                rel = out_dir / fname
            seed = int(noise_seed) + i * 100 + j
            chg_s, dis_s = 1.0, 1.0
            seq_k = list(seq)
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
                print(
                    f"  [{idx:02d}] SOC={soc0:.3f}  T={t_c:+6.1f} °C"
                    f"{_scale_tag(chg_s, dis_s)}  -> {fname}"
                )
                rec.update(
                    n_steps=0,
                    duration_s=0.0,
                    soc_end=float(soc0),
                    ut_end=float("nan"),
                    cutoff_steps=0,
                    charge_c_scale=float(chg_s),
                    discharge_c_scale=float(dis_s),
                )
                rows.append(rec)
                idx += 1
                continue

            if pybamm:
                assert simulate_pybamm is not None
                data, seq_k, chg_s, dis_s = simulate_pybamm_with_protect_retry(
                    simulate_pybamm,
                    seq,
                    idx=idx,
                    dt_s=DT_S,
                    soc0=float(soc0),
                    t_ambient_c=float(t_c),
                    noise_enable=noise_enable,
                    noise_seed=seed,
                    noise_std=std,
                    enable_cutoff=ENABLE_CUTOFF,
                    thermal=thermal,
                    verbose=idx == 0,
                )
                extra_meta = [
                    "# source=nmc100ah_gen_grid",
                    "# backend=pybamm",
                    "# pybamm_model=SPM",
                    "# pybamm_params=Chen2020_100Ah",
                    "# teacher_rc=ecm_eval",
                    f"# pybamm_thermal={int(thermal)}",
                    f"# t0_c={t_c}",
                    f"# grid_i_soc={i}",
                    f"# grid_j_temp={j}",
                    f"# grid_n_soc={n_soc}",
                    f"# grid_n_temp={n_temp}",
                    f"# protect_retry=1",
                    f"# protect_retry_scale={PROTECT_RETRY_SCALE:g}",
                    f"# protect_retry_max={PROTECT_RETRY_MAX}",
                    f"# charge_c_scale={chg_s:g}",
                    f"# discharge_c_scale={dis_s:g}",
                    "# schema_version=1.1.0",
                ]
                csv_source = "nmc100ah_pybamm"
            else:
                assert model is not None
                data = simulate(
                    model,
                    seq_k,
                    dt_s=DT_S,
                    soc0=float(soc0),
                    t_ambient_c=float(t_c),
                    u_p0=U_P0,
                    noise_enable=noise_enable,
                    noise_seed=seed,
                    noise_std=std,
                    enable_cutoff=ENABLE_CUTOFF,
                    rc2=rc2,
                )
                extra_meta = [
                    "# source=nmc100ah_gen_grid",
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
                ]
                csv_source = "nmc100ah_gen"
            csv_path = write_csv(
                out_dir / fname,
                data,
                dt_s=DT_S,
                soc0=float(soc0),
                noise_enable=noise_enable,
                noise_seed=seed,
                noise_std=std,
                sequence=seq_k,
                extra_meta=extra_meta,
                source=csv_source,
            )
            rec.update(
                n_steps=int(len(data["time_s"])),
                duration_s=float(data["time_s"][-1] + DT_S),
                soc_end=float(data["soc_true"][-1]),
                ut_end=float(data["u_t_true_v"][-1]),
                cutoff_steps=int(np.sum(data["cutoff"] > 0)),
                charge_c_scale=float(chg_s),
                discharge_c_scale=float(dis_s),
                path=str(csv_path.relative_to(REPO_ROOT)).replace("\\", "/")
                if csv_path.is_relative_to(REPO_ROOT)
                else str(csv_path).replace("\\", "/"),
            )
            print(
                f"  [{idx:02d}] SOC {soc0:.3f}->{rec['soc_end']:.3f}  "
                f"T={t_c:+6.1f} °C  Ut={rec['ut_end']:.3f} V"
                f"{_scale_tag(chg_s, dis_s)}  {fname}"
            )
            rows.append(rec)
            idx += 1

    if not dry_run:
        _write_index(out_dir / "index.csv", rows)
        print(f"索引  {out_dir / 'index.csv'}")
        meta = {
            "schema_version": "1.1.0",
            "backend": "pybamm" if pybamm else "ecm",
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
            "noise_enable": bool(noise_enable),
            "noise_std": std,
            "protect_retry": bool(pybamm),
            "protect_retry_scale": PROTECT_RETRY_SCALE,
            "protect_retry_max": PROTECT_RETRY_MAX,
        }
        if pybamm:
            meta.update(
                {
                    "pybamm_model": "SPM",
                    "pybamm_params": "Chen2020_100Ah",
                    "pybamm_thermal": bool(thermal),
                    "teacher_rc": "ecm_eval",
                }
            )
        if model is not None and hasattr(model, "meta"):
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
        "charge_c_scale",
        "discharge_c_scale",
        "file",
        "path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="按 SOC×温度网格批量生成波形 CSV（默认 ECM，--pybamm 换 SPM）")
    parser.add_argument("--n-soc", type=int, default=N_SOC, help="起始 SOC 档数")
    parser.add_argument("--n-temp", type=int, default=N_TEMP, help="温度档数")
    parser.add_argument("--out-dir", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--seed", type=int, default=None, help="覆盖噪声种子")
    parser.add_argument("--no-noise", action="store_true", help="关闭噪声")
    parser.add_argument("--noise-voltage", type=float, default=None, help="端电压测量噪声标准差 / V")
    parser.add_argument("--noise-current", type=float, default=None, help="电流测量噪声标准差 / A")
    parser.add_argument("--noise-temp", type=float, default=None, help="温度测量噪声标准差 / °C")
    parser.add_argument("--noise-soc", type=float, default=None, help="SOC 测量噪声标准差，1 表示 100 个百分点")
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
    parser.add_argument(
        "--pybamm",
        action="store_true",
        help="用 PyBaMM SPM（Chen2020 放大到 100Ah）出电压，CSV 列与 ECM 相同；建议另指定 --out-dir",
    )
    parser.add_argument(
        "--thermal",
        action="store_true",
        help="仅 --pybamm：打开 lumped 热模型（默认等温，与网格温度轴一致）",
    )
    args = parser.parse_args()

    if args.thermal and not args.pybamm:
        parser.error("--thermal 只能与 --pybamm 一起用")
    if args.pybamm:
        from nmc100ah_pybamm import _ecm_flags_incompatible

        msg = _ecm_flags_incompatible(
            rc2=args.rc2,
            soh=args.soh,
            soh_capacity=args.soh_capacity,
            r0_scale=args.r0_scale,
            r1_scale=args.r1_scale,
            c1_scale=args.c1_scale,
        )
        if msg:
            parser.error(msg)

    std = dict(NOISE_STD)
    if args.noise_voltage is not None:
        std["voltage_v"] = args.noise_voltage
    if args.noise_current is not None:
        std["current_a"] = args.noise_current
    if args.noise_temp is not None:
        std["temp_c"] = args.noise_temp
    if args.noise_soc is not None:
        std["soc"] = args.noise_soc

    run_grid(
        n_soc=args.n_soc,
        n_temp=args.n_temp,
        output_dir=args.out_dir,
        noise_enable=NOISE_ENABLE and not args.no_noise,
        noise_seed=NOISE_SEED if args.seed is None else args.seed,
        noise_std=std,
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
        pybamm=args.pybamm,
        thermal=args.thermal,
    )


if __name__ == "__main__":
    main()
