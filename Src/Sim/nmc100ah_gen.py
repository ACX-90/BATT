"""100 Ah NMC 时域仿真。默认 ECM，`--pybamm` 换 SPM。

按头部配置的充电 / 放电 / 静置指令序列推进电芯状态，
步长固定为 DT_S（默认 0.1 s），测量通道叠加高斯噪声，
结果写入 Data/*.csv。

用法（仓库根目录）：

    python Src/Sim/nmc100ah_gen.py
    python Src/Sim/nmc100ah_gen.py --out Data/my_run.csv
    python Src/Sim/nmc100ah_gen.py --out Data/long/cc_rest.csv --soc0 0.70
    python Src/Sim/nmc100ah_gen.py --pybamm --out Data/nmc100ah_pybamm_sim.csv

其它模块不要改本文件头部的 SEQUENCE。传入可选工况：

    from nmc100ah_gen import run_sim
    run_sim(out="Data/long/cc_rest.csv", sequence=my_seq, soc0=0.70)
"""

from __future__ import annotations

# =============================================================================
# 头部配置（改这里即可，不必改下面的仿真逻辑）
# =============================================================================

# 仿真步长。时间单位 = 0.1 秒
DT_S = 0.1

# 初始状态
SOC0 = 0.70
T_AMBIENT_C = 25.0
U_P0 = 0.0  # 极化电压初值 / V

# 电压 / SOC 保护：触发后本条指令剩余时间改为静置
ENABLE_CUTOFF = True

# ---------------------------------------------------------------------------
# 噪声：高斯 N(0, std)。只加在 *_meas 列，不参与安时积分与极化递推。
# 某一项设为 0 即关闭该通道噪声。
# ---------------------------------------------------------------------------
NOISE_ENABLE = True
NOISE_SEED = 20260813
NOISE_STD = {
    "voltage_v": 1e-3,    # 端电压 1 mV
    "current_a": 2.0e-2,  # 电流 20 mA
    "temp_c": 5.0e-2,     # 温度 0.05 °C
    "soc": 5.0e-4,        # SOC 0.05 个百分点
}

# 输出路径（相对仓库根目录）。目录不存在时自动创建。
OUTPUT_CSV = "Data/nmc100ah_ecm_sim.csv"

# ---------------------------------------------------------------------------
# 指令序列
#   mode         : "charge" | "discharge" | "rest"
#   duration_s   : 持续时间，秒（会按 DT_S 四舍五入到整步）
#   duration_steps: 可选，直接给步数；与 duration_s 二选一
#   c_rate       : 倍率，1.0 = 100 A；rest 可省略
#   current_a    : 电流绝对值，A；与 c_rate 二选一
#   t_celsius    : 可选，覆盖本段温度，默认 T_AMBIENT_C
# 电流符号由 mode 决定：放电为正，充电为负，静置为 0。
# ---------------------------------------------------------------------------
SEQUENCE: list[dict] = [
    {"mode": "rest", "duration_s": 30.0},
    {"mode": "discharge", "duration_s": 180.0, "c_rate": 1.0},
    {"mode": "rest", "duration_s": 120.0},
    {"mode": "charge", "duration_s": 90.0, "c_rate": 0.5},
    {"mode": "rest", "duration_s": 60.0},
    {"mode": "discharge", "duration_s": 10.0, "c_rate": 2.0},
    {"mode": "rest", "duration_s": 60.0},
    {"mode": "charge", "duration_s": 10.0, "c_rate": 1.0},
    {"mode": "rest", "duration_s": 60.0},
]

# =============================================================================
# 以下为仿真实现
# =============================================================================

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parent
REPO_ROOT = SIM_DIR.parent.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from nmc100ah_ecm import (  # noqa: E402
    NMC100AhECM,
    make_ecm,
    R2_OVER_R1,
    TAU2_S,
    SOH_A0,
    SOH_A1,
    SOH_AC,
)

# 典型 100 Ah NMC/石墨 OCV 表（25 °C），用于电压仿真
_OCV_SOC = np.array(
    [0.00, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 1.00],
    dtype=float,
)
_OCV_V = np.array(
    [3.280, 3.400, 3.480, 3.545, 3.595, 3.630, 3.658, 3.690, 3.735, 3.800, 3.890, 4.020, 4.100, 4.155, 4.185],
    dtype=float,
)
_OCV_DUDT = -4.0e-4  # V/°C，相对 25 °C


CSV_COLUMNS = [
    "time_s",
    "step",
    "cmd_id",
    "mode",
    "cutoff",
    "i_true_a",
    "i_meas_a",
    "t_true_c",
    "t_meas_c",
    "soc_true",
    "soc_meas",
    "u_ocv_v",
    "r0_ohm",
    "r1_ohm",
    "c1_f",
    "tau1_s",
    "u_p_v",
    "u_t_true_v",
    "u_t_meas_v",
]


def ocv_nmc(soc: float, t_celsius: float) -> float:
    s = float(np.clip(soc, 0.0, 1.0))
    return float(np.interp(s, _OCV_SOC, _OCV_V) + _OCV_DUDT * (t_celsius - 25.0))


def _cmd_current_a(cmd: dict, capacity_ah: float) -> float:
    mode = str(cmd["mode"]).strip().lower()
    if mode in {"rest", "idle", "pause"}:
        return 0.0
    if "current_a" in cmd and "c_rate" in cmd:
        raise ValueError(f"指令不能同时给 current_a 和 c_rate: {cmd}")
    if "current_a" in cmd:
        mag = abs(float(cmd["current_a"]))
    elif "c_rate" in cmd:
        mag = abs(float(cmd["c_rate"])) * capacity_ah
    else:
        raise ValueError(f"charge/discharge 必须提供 current_a 或 c_rate: {cmd}")
    if mode in {"discharge", "dch", "dis"}:
        return mag
    if mode in {"charge", "chg", "cha"}:
        return -mag
    raise ValueError(f"未知 mode={cmd.get('mode')!r}，应为 charge / discharge / rest")


def _cmd_steps(cmd: dict, dt_s: float) -> int:
    has_s = "duration_s" in cmd
    has_n = "duration_steps" in cmd
    if has_s == has_n:
        raise ValueError(f"duration_s 与 duration_steps 必须二选一: {cmd}")
    if has_n:
        n = int(cmd["duration_steps"])
    else:
        n = int(round(float(cmd["duration_s"]) / dt_s))
    if n <= 0:
        raise ValueError(f"指令时长必须 > 0: {cmd}")
    return n


def _normalize_mode(mode: str) -> str:
    key = mode.strip().lower()
    if key in {"rest", "idle", "pause"}:
        return "rest"
    if key in {"discharge", "dch", "dis"}:
        return "discharge"
    if key in {"charge", "chg", "cha"}:
        return "charge"
    if key in {"dis_ramp", "chg_ramp"}:
        return key
    raise ValueError(f"未知 mode={mode!r}")


def _ramp_current_a(cmd: dict, capacity_ah: float) -> tuple[float, float]:
    """返回 (i_start_a, i_end_a)。c_rate 方向由 mode 决定。"""
    rate_start = float(cmd["c_rate_start"]) * capacity_ah
    rate_end = float(cmd["c_rate_end"]) * capacity_ah
    if "chg_ramp" == str(cmd["mode"]).strip().lower():
        return -rate_start, -rate_end
    return rate_start, rate_end


def expand_sequence(
    sequence: list[dict],
    *,
    dt_s: float,
    capacity_ah: float,
    t_default: float,
) -> list[tuple[int, str, float, float]]:
    """展开为逐步指令：(cmd_id, mode, i_a, t_celsius)。"""
    steps: list[tuple[int, str, float, float]] = []
    for idx, cmd in enumerate(sequence):
        mode = _normalize_mode(str(cmd["mode"]))
        t_c = float(cmd.get("t_celsius", t_default))
        if "ramp" in mode:
            i_start, i_end = _ramp_current_a(cmd, capacity_ah)
            n = _cmd_steps(cmd, dt_s)
            for k in range(n):
                i_a = i_start + (i_end - i_start) * k / max(n - 1, 1)
                steps.append((idx, mode, i_a, t_c))
        else:
            i_a = _cmd_current_a(cmd, capacity_ah)
            for _ in range(_cmd_steps(cmd, dt_s)):
                steps.append((idx, mode, i_a, t_c))
    if not steps:
        raise ValueError("SEQUENCE 为空")
    return steps


def _gauss(rng: np.random.Generator, std: float) -> float:
    if std <= 0.0:
        return 0.0
    return float(rng.normal(0.0, std))


def simulate(
    model: NMC100AhECM,
    sequence: list[dict],
    *,
    dt_s: float = DT_S,
    soc0: float = SOC0,
    t_ambient_c: float = T_AMBIENT_C,
    u_p0: float = U_P0,
    noise_enable: bool = NOISE_ENABLE,
    noise_seed: int = NOISE_SEED,
    noise_std: dict[str, float] | None = None,
    enable_cutoff: bool = ENABLE_CUTOFF,
    rc2: bool | None = None,
    u_p2_0: float = 0.0,
    soc_capacity_ah: float | None = None,
) -> dict[str, np.ndarray]:
    cell = model.params.cell
    lim = model.params.validity
    noise_std = dict(NOISE_STD if noise_std is None else noise_std)
    rng = np.random.default_rng(noise_seed)
    use_rc2 = bool(getattr(model, "rc2", False) if rc2 is None else rc2)
    soh = float(getattr(model, "q", 1.0))
    write_soh = abs(soh - 1.0) > 1e-12 or use_rc2

    plan = expand_sequence(
        sequence, dt_s=dt_s, capacity_ah=cell.capacity_ah, t_default=t_ambient_c
    )
    n = len(plan)
    # 电流按铭牌 C 率（包共享 I）；真 SOC 可用每芯 Q_i（2A4）。
    q_as = float(cell.capacity_ah if soc_capacity_ah is None else soc_capacity_ah) * 3600.0

    extra = []
    if use_rc2:
        extra.extend(["r2_ohm", "c2_f", "tau2_s", "u_p2_v"])
    if write_soh:
        extra.append("soh")
    out = {name: np.empty(n, dtype=float) for name in CSV_COLUMNS + extra if name != "mode"}
    modes = np.empty(n, dtype=object)

    soc = float(soc0)
    u_p = float(u_p0)
    u_p2 = float(u_p2_0)
    cutoff_cmd: int | None = None

    for k, (cmd_id, mode, i_cmd, t_cmd) in enumerate(plan):
        i_a = 0.0 if (cutoff_cmd is not None and cmd_id == cutoff_cmd) else float(i_cmd)
        mode_k = "rest" if i_a == 0.0 and mode != "rest" and cutoff_cmd == cmd_id else mode

        soc_param = float(np.clip(soc, lim.soc_min, lim.soc_max))
        r0, r1, c1 = model.evaluate(i_a=i_a, t_celsius=t_cmd, soc=soc_param)
        r0, r1, c1 = float(r0), float(r1), float(c1)
        tau1 = r1 * c1
        if tau1 <= 1e-12:
            raise RuntimeError("tau1 非正，检查 C1/R1")

        alpha = math.exp(-dt_s / tau1)
        u_p_next = alpha * u_p + r1 * (1.0 - alpha) * i_a
        r2 = c2 = tau2 = 0.0
        u_p2_next = 0.0
        if use_rc2:
            if hasattr(model, "rc2_from_r1"):
                r2, c2, tau2 = model.rc2_from_r1(r1)
            else:
                r2 = R2_OVER_R1 * r1
                tau2 = TAU2_S
                c2 = tau2 / max(r2, 1e-12)
            alpha2 = math.exp(-dt_s / tau2)
            u_p2_next = alpha2 * u_p2 + r2 * (1.0 - alpha2) * i_a
        u_ocv = ocv_nmc(soc, t_cmd)
        u_t = u_ocv - i_a * r0 - u_p_next - (u_p2_next if use_rc2 else 0.0)

        hit = False
        if enable_cutoff and i_a != 0.0:
            if i_a > 0.0 and (u_t <= cell.v_min or soc <= lim.soc_min + 1e-6):
                hit = True
            elif i_a < 0.0 and (u_t >= cell.v_max or soc >= lim.soc_max - 1e-6):
                hit = True
        if hit:
            cutoff_cmd = cmd_id
            i_a = 0.0
            mode_k = "rest"
            r0, r1, c1 = model.evaluate(i_a=0.0, t_celsius=t_cmd, soc=soc_param)
            r0, r1, c1 = float(r0), float(r1), float(c1)
            tau1 = r1 * c1
            alpha = math.exp(-dt_s / tau1)
            u_p_next = alpha * u_p
            if use_rc2:
                if hasattr(model, "rc2_from_r1"):
                    r2, c2, tau2 = model.rc2_from_r1(r1)
                else:
                    r2 = R2_OVER_R1 * r1
                    tau2 = TAU2_S
                    c2 = tau2 / max(r2, 1e-12)
                alpha2 = math.exp(-dt_s / tau2)
                u_p2_next = alpha2 * u_p2
            u_t = u_ocv - u_p_next - (u_p2_next if use_rc2 else 0.0)

        u_p = u_p_next
        if use_rc2:
            u_p2 = u_p2_next

        soc = float(np.clip(soc - i_a * dt_s / q_as, 0.0, 1.0))

        if noise_enable:
            i_meas = i_a + _gauss(rng, noise_std.get("current_a", 0.0))
            t_meas = t_cmd + _gauss(rng, noise_std.get("temp_c", 0.0))
            soc_meas = float(np.clip(soc + _gauss(rng, noise_std.get("soc", 0.0)), 0.0, 1.0))
            u_meas = u_t + _gauss(rng, noise_std.get("voltage_v", 0.0))
        else:
            i_meas, t_meas, soc_meas, u_meas = i_a, t_cmd, soc, u_t

        out["time_s"][k] = k * dt_s
        out["step"][k] = k
        out["cmd_id"][k] = cmd_id
        out["cutoff"][k] = 1.0 if hit or (cutoff_cmd == cmd_id) else 0.0
        out["i_true_a"][k] = i_a
        out["i_meas_a"][k] = i_meas
        out["t_true_c"][k] = t_cmd
        out["t_meas_c"][k] = t_meas
        out["soc_true"][k] = soc
        out["soc_meas"][k] = soc_meas
        out["u_ocv_v"][k] = u_ocv
        out["r0_ohm"][k] = r0
        out["r1_ohm"][k] = r1
        out["c1_f"][k] = c1
        out["tau1_s"][k] = tau1
        out["u_p_v"][k] = u_p
        out["u_t_true_v"][k] = u_t
        out["u_t_meas_v"][k] = u_meas
        if use_rc2:
            out["r2_ohm"][k] = r2
            out["c2_f"][k] = c2
            out["tau2_s"][k] = tau2
            out["u_p2_v"][k] = u_p2
        if write_soh:
            out["soh"][k] = soh
        modes[k] = mode_k

    out["mode"] = modes
    return out


def write_csv(
    path: str | Path,
    data: dict[str, np.ndarray],
    *,
    dt_s: float,
    soc0: float,
    noise_enable: bool,
    noise_seed: int,
    noise_std: dict[str, float],
    sequence: list[dict],
    extra_meta: list[str] | None = None,
    source: str = "nmc100ah_gen",
) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(data["time_s"])
    meta = [
        f"# {source}",
        f"# dt_s={dt_s}",
        f"# n_steps={n}",
        f"# duration_s={n * dt_s:.1f}",
        f"# soc0={soc0}",
        f"# noise_enable={int(noise_enable)}",
        f"# noise_seed={noise_seed}",
        f"# noise_std_voltage_v={noise_std.get('voltage_v', 0.0)}",
        f"# noise_std_current_a={noise_std.get('current_a', 0.0)}",
        f"# noise_std_temp_c={noise_std.get('temp_c', 0.0)}",
        f"# noise_std_soc={noise_std.get('soc', 0.0)}",
        f"# n_commands={len(sequence)}",
        f"# current_sign=discharge_positive",
    ]
    if extra_meta:
        meta.extend(extra_meta)

    with path.open("w", encoding="utf-8", newline="") as fh:
        for line in meta:
            fh.write(line + "\n")
        extra = [k for k in ("r2_ohm", "c2_f", "tau2_s", "u_p2_v", "soh") if k in data]
        cols = CSV_COLUMNS + extra
        writer = csv.writer(fh)
        writer.writerow(cols)
        for k in range(n):
            row = []
            for name in cols:
                val = data[name][k]
                if name == "mode":
                    row.append(val)
                elif name in {"step", "cmd_id", "cutoff"}:
                    row.append(int(val))
                else:
                    row.append(f"{float(val):.8g}")
            writer.writerow(row)
    return path


def run_sim(
    *,
    out: str | Path = OUTPUT_CSV,
    sequence: list[dict] | None = None,
    soc0: float | None = None,
    t_ambient_c: float | None = None,
    u_p0: float | None = None,
    dt_s: float | None = None,
    noise_enable: bool | None = None,
    noise_seed: int | None = None,
    noise_std: dict[str, float] | None = None,
    enable_cutoff: bool | None = None,
    extra_meta: list[str] | None = None,
    model: NMC100AhECM | None = None,
    summarize: bool = True,
    soh: float = 1.0,
    soh_a0: float = SOH_A0,
    soh_a1: float = SOH_A1,
    soh_ac: float = SOH_AC,
    soh_capacity: float = 1.0,
    rc2: bool = False,
    r0_scale: float = 1.0,
    r1_scale: float = 1.0,
    c1_scale: float = 1.0,
) -> Path:
    """跑一条轨迹并写 CSV。sequence / soc0 等为 None 时用本文件头部默认，不改默认网格。"""
    seq = SEQUENCE if sequence is None else list(sequence)
    s0 = SOC0 if soc0 is None else float(soc0)
    t_c = T_AMBIENT_C if t_ambient_c is None else float(t_ambient_c)
    up0 = U_P0 if u_p0 is None else float(u_p0)
    dt = DT_S if dt_s is None else float(dt_s)
    if dt <= 0:
        raise ValueError("dt_s 必须 > 0")
    n_en = NOISE_ENABLE if noise_enable is None else bool(noise_enable)
    seed = NOISE_SEED if noise_seed is None else int(noise_seed)
    std = dict(NOISE_STD if noise_std is None else noise_std)
    cut = ENABLE_CUTOFF if enable_cutoff is None else bool(enable_cutoff)
    if model is None:
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
    meta = list(extra_meta or [])
    if hasattr(model, "meta"):
        blob = model.meta()
        meta.extend(
            [
                "# schema_version=1.1.0",
                f"# soh={blob.get('soh', 1.0)}",
                f"# soh_capacity={blob.get('soh_capacity', 1.0)}",
                f"# rc2={int(bool(blob.get('rc2', False)))}",
            ]
        )
    elif rc2 or abs(soh - 1.0) > 1e-12:
        meta.extend([f"# schema_version=1.1.0", f"# soh={soh}", f"# rc2={int(rc2)}"])

    data = simulate(
        model,
        seq,
        dt_s=dt,
        soc0=s0,
        t_ambient_c=t_c,
        u_p0=up0,
        noise_enable=n_en,
        noise_seed=seed,
        noise_std=std,
        enable_cutoff=cut,
        rc2=rc2,
    )
    path = write_csv(
        out,
        data,
        dt_s=dt,
        soc0=s0,
        noise_enable=n_en,
        noise_seed=seed,
        noise_std=std,
        sequence=seq,
        extra_meta=meta,
    )
    print(f"已写出 {path}")
    if summarize:
        _summarize(data, dt_s=dt)
    return path


def _summarize(data: dict[str, np.ndarray], *, dt_s: float = DT_S) -> None:
    t = data["time_s"]
    print(f"步数 {len(t)}  时长 {t[-1] + dt_s:.1f} s  步长 {dt_s} s")
    print(
        f"SOC  {data['soc_true'][0]:.4f} -> {data['soc_true'][-1]:.4f}    "
        f"Ut   {data['u_t_true_v'][0]:.4f} -> {data['u_t_true_v'][-1]:.4f} V"
    )
    print(
        f"Ut_meas  std(残差)={np.std(data['u_t_meas_v'] - data['u_t_true_v'])*1e3:.2f} mV  "
        f"I_meas std(残差)={np.std(data['i_meas_a'] - data['i_true_a'])*1e3:.1f} mA"
    )
    modes, counts = np.unique(data["mode"], return_counts=True)
    parts = ", ".join(f"{m}={c}" for m, c in zip(modes, counts))
    print(f"模式步数  {parts}")
    if np.any(data["cutoff"] > 0):
        print(f"触发保护步数  {int(np.sum(data['cutoff'] > 0))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NMC 100Ah 时域仿真（默认 ECM，--pybamm 换 SPM），输出 Data/*.csv")
    parser.add_argument("--out", default=OUTPUT_CSV, help="输出 CSV 路径")
    parser.add_argument("--soc0", type=float, default=None, help="覆盖头部 SOC0")
    parser.add_argument("--t-ambient", type=float, default=None, help="覆盖头部环境温度 / °C")
    parser.add_argument("--seed", type=int, default=None, help="覆盖头部 NOISE_SEED")
    parser.add_argument("--no-noise", action="store_true", help="关闭噪声")
    parser.add_argument("--soh", type=float, default=1.0, help="电阻/电容寿命因子 q，1=BOL")
    parser.add_argument("--rc2", action="store_true", help="叠加慢支路 R2C2，BMS 仍读 1RC")
    parser.add_argument(
        "--pybamm",
        action="store_true",
        help="用 PyBaMM SPM 出电压，CSV 列与 ECM 相同",
    )
    parser.add_argument(
        "--thermal",
        action="store_true",
        help="仅 --pybamm：打开 lumped 热模型（默认等温）",
    )
    args = parser.parse_args()

    if args.thermal and not args.pybamm:
        parser.error("--thermal 只能与 --pybamm 一起用")
    if args.pybamm:
        if args.rc2 or abs(args.soh - 1.0) > 1e-12:
            parser.error("--pybamm 不能与 --rc2 / --soh 一起用")
        from nmc100ah_pybamm import run_sim as run_pybamm

        run_pybamm(
            out=args.out,
            soc0=args.soc0,
            t_ambient_c=args.t_ambient,
            noise_enable=False if args.no_noise else None,
            noise_seed=args.seed,
            thermal=args.thermal,
        )
        return

    run_sim(
        out=args.out,
        soc0=args.soc0,
        t_ambient_c=args.t_ambient,
        noise_enable=False if args.no_noise else None,
        noise_seed=args.seed,
        soh=args.soh,
        rc2=args.rc2,
    )


if __name__ == "__main__":
    main()
