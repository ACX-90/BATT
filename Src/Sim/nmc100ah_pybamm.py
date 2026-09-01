"""PyBaMM SPM 100Ah NMC 时域仿真，CSV 列与 nmc100ah_gen.py 对齐。

Chen2020 参数按容量比例放大电极面积到 100Ah（厚度不变）。
工况沿用 nmc100ah_gen.SEQUENCE：放电电流为正、步长 DT_S。
端电压 / OCV / SOC / 温度来自 SPM；r0/r1/c1/tau1 在同一 (I,T,SOC) 上
用仓库 ECM 求值，作为教师列（主损失仍是电压）。

用法（仓库根目录）：

    python Src/Sim/nmc100ah_pybamm.py
    python Src/Sim/nmc100ah_pybamm.py --out Data/nmc100ah_pybamm_sim.csv
    python Src/Sim/nmc100ah_gen_grid.py --pybamm --out-dir Data/grid_pybamm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from nmc100ah_ecm import make_ecm  # noqa: E402
from nmc100ah_gen import (  # noqa: E402
    CSV_COLUMNS,
    DT_S,
    ENABLE_CUTOFF,
    NOISE_ENABLE,
    NOISE_SEED,
    NOISE_STD,
    SEQUENCE,
    SOC0,
    T_AMBIENT_C,
    _gauss,
    _summarize,
    expand_sequence,
    write_csv,
)
from nmc100ah_ecm_params import default_param_set  # noqa: E402

CAPACITY_AH = 100.0
OUTPUT_CSV = "Data/nmc100ah_pybamm_sim.csv"
_PARAMS_CACHE: object | None = None


def _import_pybamm():
    try:
        import pybamm
    except ImportError as exc:
        raise ImportError("需要安装 PyBaMM：pip install pybamm") from exc
    return pybamm


def build_100ah_params(*, verbose: bool = True):
    """Chen2020 面积放大到 100Ah，电压窗与仓库 CellSpec 对齐。"""
    global _PARAMS_CACHE
    pybamm = _import_pybamm()
    if _PARAMS_CACHE is not None:
        params = _PARAMS_CACHE.copy()
    else:
        params = pybamm.ParameterValues("Chen2020")
        orig_cap = float(params["Nominal cell capacity [A.h]"])
        scale = CAPACITY_AH / orig_cap
        width = float(params["Electrode width [m]"])
        height = float(params["Electrode height [m]"])
        area = width * height
        new_dim = float(np.sqrt(area * scale))
        params.update(
            {
                "Nominal cell capacity [A.h]": CAPACITY_AH,
                "Electrode width [m]": new_dim,
                "Electrode height [m]": new_dim,
                "Current collector width [m]": new_dim,
                "Current collector height [m]": new_dim,
            }
        )
        cell = default_param_set().cell
        params.update(
            {
                "Lower voltage cut-off [V]": float(cell.v_min),
                "Upper voltage cut-off [V]": float(cell.v_max),
            }
        )
        _PARAMS_CACHE = params.copy()
        if verbose:
            neg_t = float(params["Negative electrode thickness [m]"])
            pos_t = float(params["Positive electrode thickness [m]"])
            sep_t = float(params["Separator thickness [m]"])
            print(
                f"Chen2020 ({orig_cap:g} Ah) -> {CAPACITY_AH:.0f} Ah | "
                f"面积放大 {scale:.0f}x（厚度不变）"
            )
            print(
                f"电极厚度: 负极 {neg_t*1e3:.3f} mm / 正极 {pos_t*1e3:.3f} mm / "
                f"隔膜 {sep_t*1e6:.1f} um"
            )
            print(f"极片面积: {area*1e4:.1f} cm2 -> {area*scale*1e4:.1f} cm2")
            print(f"电压窗: {cell.v_min:.2f} - {cell.v_max:.2f} V")

        params = _PARAMS_CACHE.copy()

    cell = default_param_set().cell
    params.update(
        {
            "Lower voltage cut-off [V]": float(cell.v_min),
            "Upper voltage cut-off [V]": float(cell.v_max),
        }
    )
    return params


def _sol_array(sol, *names: str) -> np.ndarray:
    last_err: Exception | None = None
    for name in names:
        try:
            raw = sol[name].data
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
        y = np.asarray(raw, dtype=float)
        if y.ndim > 1:
            y = np.reshape(y, (y.shape[-1],))
        return np.ravel(y)
    raise KeyError(f"PyBaMM 解中找不到变量 {names}") from last_err


def _unique_time(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if t.size <= 1:
        return np.array([True], dtype=bool)
    return np.concatenate(([True], np.diff(t) > 1e-12))


def _unpack_step(sol, pybamm, param) -> dict[str, np.ndarray]:
    t = np.asarray(sol["Time [s]"].data, dtype=float)
    keep = _unique_time(t)
    t = t[keep]
    i_a = _sol_array(sol, "Current [A]")[keep]
    u_t = _sol_array(sol, "Terminal voltage [V]", "Voltage [V]")[keep]
    u_ocv = _sol_array(
        sol, "Battery open-circuit voltage [V]", "Bulk open-circuit voltage [V]"
    )[keep]
    try:
        t_c = _sol_array(
            sol,
            "Volume-averaged cell temperature [C]",
            "Cell temperature [C]",
            "Ambient temperature [C]",
        )[keep]
    except KeyError:
        t_c = np.full_like(t, float("nan"))
    x_avg = _sol_array(sol, "Average negative particle stoichiometry")[keep]
    x0, x100, _, _ = pybamm.lithium_ion.get_min_max_stoichiometries(param)
    soc = (x_avg - float(x0)) / (float(x100) - float(x0))
    return {
        "t": t,
        "i_a": i_a,
        "u_t": u_t,
        "u_ocv": u_ocv,
        "t_c": t_c,
        "soc": soc,
    }


def _interp_piece(piece: dict[str, np.ndarray], t_query: np.ndarray) -> dict[str, np.ndarray]:
    t = piece["t"] - piece["t"][0]
    t = np.maximum.accumulate(t)
    t_query = np.clip(t_query, t[0], t[-1])
    out = {}
    for key, y in piece.items():
        if key == "t":
            continue
        out[key] = np.interp(t_query, t, np.asarray(y, dtype=float))
    return out


def _concat_pieces(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    t_b = b["t"]
    y_b = {k: v for k, v in b.items() if k != "t"}
    if t_b[0] <= a["t"][-1] + 1e-12:
        t_b = t_b[1:]
        y_b = {k: v[1:] for k, v in y_b.items()}
    if t_b.size == 0:
        return a
    return {
        "t": np.concatenate([a["t"], t_b]),
        **{k: np.concatenate([a[k], y_b[k]]) for k in y_b},
    }


def _step_direction(mode: str, i_arr: np.ndarray) -> str:
    """充/放方向看 mode（chg_ramp 可能从 0 A 起步，不能看 i[0]）。"""
    key = str(mode).strip().lower()
    if "chg" in key:
        return "charge"
    if "dis" in key:
        return "discharge"
    if i_arr.size == 0 or not np.isfinite(i_arr).any():
        return "rest"
    i_ref = float(i_arr[np.nanargmax(np.abs(i_arr))])
    if i_ref < 0:
        return "charge"
    if i_ref > 0:
        return "discharge"
    return "rest"


def _drive_cycle(i_arr: np.ndarray, duration: float, dt_s: float) -> np.ndarray:
    """逐步电流 → PyBaMM 两列 drive cycle（t=0 起，含终点）。"""
    i_arr = np.asarray(i_arr, dtype=float)
    t = np.arange(i_arr.size, dtype=float) * dt_s
    if t[-1] < duration:
        t = np.append(t, duration)
        i_arr = np.append(i_arr, i_arr[-1])
    return np.column_stack([t, i_arr])


def _make_step(
    pybamm,
    *,
    mode: str,
    i_arr: np.ndarray,
    duration: float,
    dt_s: float,
    v_min: float,
    v_max: float,
    t_k: float,
    rest: bool,
):
    period = dt_s
    i_arr = np.asarray(i_arr, dtype=float)
    if rest or mode == "rest" or np.all(np.abs(i_arr) < 1e-12):
        return pybamm.step.rest(
            duration=duration, period=period, temperature=t_k, skip_ok=False
        )
    direction = _step_direction(mode, i_arr)
    if direction == "charge":
        termination = [f">{v_max} V"]
    else:
        termination = [f"<{v_min} V"]
    # skip_ok：步初就超压时跳过本步，不抛 SolverError。调用方必须用
    # 周期数判断是否跳过，否则会把上一段静置误当成充电数据。
    if np.ptp(i_arr) < 1e-9:
        step = pybamm.step.current(
            float(i_arr[0]),
            duration=duration,
            period=period,
            temperature=t_k,
            termination=termination,
            skip_ok=True,
        )
    else:
        step = pybamm.step.current(
            _drive_cycle(i_arr, duration, dt_s),
            duration=duration,
            period=period,
            temperature=t_k,
            termination=termination,
            skip_ok=True,
        )
    # chg_ramp 从 0 A 起步时 PyBaMM 会把 direction 判成 rest，盖掉
    step.direction = direction
    return step


def _group_plan(
    plan: list[tuple[int, str, float, float]],
) -> list[tuple[int, str, np.ndarray, float]]:
    groups: list[tuple[int, str, list[float], float]] = []
    for cmd_id, mode, i_a, t_c in plan:
        if not groups or groups[-1][0] != cmd_id:
            groups.append((cmd_id, mode, [float(i_a)], float(t_c)))
        else:
            groups[-1][2].append(float(i_a))
    return [(c, m, np.asarray(ii, dtype=float), t) for c, m, ii, t in groups]


def _solve_step(pybamm, model, param, step, t_k, *, sol, soc0: float):
    p = param.copy()
    p["Ambient temperature [K]"] = t_k
    experiment = pybamm.Experiment([step], temperature=t_k)
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=p)
    if sol is None:
        return sim.solve(initial_soc=soc0, calc_esoh=False)
    return sim.solve(starting_solution=sol, calc_esoh=False)


def _n_cycles(sol) -> int:
    if sol is None:
        return 0
    cycles = getattr(sol, "cycles", None)
    return 0 if cycles is None else len(cycles)


def _infeasible_at_ic(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "infeasible" in msg and "initial" in msg


def apply_k_r(params, k_r: float, *, r_dc_ohm: float = 1.45e-3) -> None:
    """2A1 的 k 乘到 SPM：串联接触电阻 (k-1)*(R0+R1)_ref。

    不改 Ds / 交换电流——那些是 SPM 真物理；k 只模拟同一点涨阻。
    1C 上 (k-1)*1.45 mΩ ≈ 22 mV，和 ECM ×1.15 的墙同量级。
    """
    k_r = float(k_r)
    if abs(k_r - 1.0) < 1e-12:
        return
    if k_r <= 0:
        raise ValueError(f"k_r 必须为正，得到 {k_r}")
    extra = (k_r - 1.0) * float(r_dc_ohm)
    params["Contact resistance [Ohm]"] = float(params["Contact resistance [Ohm]"]) + extra


def simulate(
    sequence: list[dict] | None = None,
    *,
    dt_s: float = DT_S,
    soc0: float = SOC0,
    t_ambient_c: float = T_AMBIENT_C,
    noise_enable: bool = NOISE_ENABLE,
    noise_seed: int = NOISE_SEED,
    noise_std: dict[str, float] | None = None,
    enable_cutoff: bool = ENABLE_CUTOFF,
    thermal: bool = False,
    verbose: bool = False,
    k_r: float = 1.0,
) -> dict[str, np.ndarray]:
    """按 SEQUENCE 跑 SPM，返回与 nmc100ah_gen.simulate 相同的列。"""
    pybamm = _import_pybamm()
    pybamm.set_logging_level("ERROR")
    seq = SEQUENCE if sequence is None else list(sequence)
    std = dict(NOISE_STD if noise_std is None else noise_std)
    cell = default_param_set().cell
    plan = expand_sequence(
        seq, dt_s=dt_s, capacity_ah=cell.capacity_ah, t_default=t_ambient_c
    )
    n = len(plan)
    groups = _group_plan(plan)

    param = build_100ah_params(verbose=verbose)
    apply_k_r(param, k_r)
    t0_k = float(t_ambient_c) + 273.15
    param["Initial temperature [K]"] = t0_k
    param["Ambient temperature [K]"] = t0_k

    options = {"thermal": "lumped"} if thermal else None
    model = pybamm.lithium_ion.SPM(options) if options else pybamm.lithium_ion.SPM()
    v_min = float(param["Lower voltage cut-off [V]"])
    v_max = float(param["Upper voltage cut-off [V]"])

    ecm = make_ecm()
    rng = np.random.default_rng(noise_seed)

    i_true = np.empty(n, dtype=float)
    t_true = np.empty(n, dtype=float)
    soc_true = np.empty(n, dtype=float)
    u_ocv = np.empty(n, dtype=float)
    u_t = np.empty(n, dtype=float)
    cutoff = np.zeros(n, dtype=float)
    cmd_ids = np.empty(n, dtype=float)
    modes = np.empty(n, dtype=object)

    sol = None
    offset = 0
    for cmd_id, mode, i_arr, t_c in groups:
        n_cmd = int(i_arr.size)
        duration = n_cmd * dt_s
        t_k = float(t_c) + 273.15
        param["Ambient temperature [K]"] = t_k
        step = _make_step(
            pybamm,
            mode=mode,
            i_arr=i_arr,
            duration=duration,
            dt_s=dt_s,
            v_min=v_min,
            v_max=v_max,
            t_k=t_k,
            rest=False,
        )
        cmd_live = bool(np.max(np.abs(i_arr)) > 1e-12)
        n_before = _n_cycles(sol)
        piece = None
        t_span = 0.0
        skipped = False
        try:
            new_sol = _solve_step(
                pybamm, model, param, step, t_k, sol=sol, soc0=float(soc0)
            )
        except Exception as exc:  # noqa: BLE001
            solver_err = getattr(pybamm, "SolverError", ())
            if not (
                enable_cutoff
                and cmd_live
                and solver_err
                and isinstance(exc, solver_err)
                and _infeasible_at_ic(exc)
            ):
                raise
            skipped = True
        else:
            skipped = cmd_live and _n_cycles(new_sol) == n_before
            if skipped:
                if not enable_cutoff:
                    raise RuntimeError(
                        f"PyBaMM 跳过 {mode} 步（步初超压），但 enable_cutoff=False"
                    )
            else:
                sol = new_sol
                piece = _unpack_step(sol.cycles[-1].steps[0], pybamm, param)
                t_span = float(piece["t"][-1] - piece["t"][0])
        hit = enable_cutoff and cmd_live and (skipped or t_span < duration - 0.5 * dt_s)
        if hit:
            remaining = max(duration - t_span, dt_s)
            rest_step = _make_step(
                pybamm,
                mode="rest",
                i_arr=np.zeros(1),
                duration=remaining,
                dt_s=dt_s,
                v_min=v_min,
                v_max=v_max,
                t_k=t_k,
                rest=True,
            )
            sol = _solve_step(
                pybamm, model, param, rest_step, t_k, sol=sol, soc0=float(soc0)
            )
            rest_piece = _unpack_step(sol.cycles[-1].steps[0], pybamm, param)
            piece = rest_piece if piece is None else _concat_pieces(piece, rest_piece)
        if piece is None:
            raise RuntimeError(f"PyBaMM 步解为空 mode={mode} cmd_id={cmd_id}")

        t_query = np.minimum((np.arange(n_cmd, dtype=float) + 1.0) * dt_s, duration)
        sampled = _interp_piece(piece, t_query)
        sl = slice(offset, offset + n_cmd)
        i_true[sl] = sampled["i_a"]
        t_true[sl] = sampled["t_c"]
        soc_true[sl] = np.clip(sampled["soc"], 0.0, 1.0)
        u_ocv[sl] = sampled["u_ocv"]
        u_t[sl] = sampled["u_t"]
        cmd_ids[sl] = cmd_id
        if hit:
            local = t_query
            rest_from = t_span
            cutoff[sl] = (local >= rest_from - 0.5 * dt_s).astype(float)
            modes[sl] = np.where(cutoff[sl] > 0, "rest", mode)
        else:
            modes[sl] = mode
        if not np.isfinite(t_true[sl]).all():
            t_true[sl] = t_c
        offset += n_cmd

    soc_param = np.clip(soc_true, 0.02, 0.98)
    r0, r1, c1 = ecm.evaluate(i_a=i_true, t_celsius=t_true, soc=soc_param)
    r0 = np.asarray(r0, dtype=float)
    r1 = np.asarray(r1, dtype=float)
    c1 = np.asarray(c1, dtype=float)
    tau1 = r1 * c1
    u_p = u_ocv - u_t - i_true * r0

    out = {name: np.empty(n, dtype=float) for name in CSV_COLUMNS if name != "mode"}
    for k in range(n):
        if noise_enable:
            i_meas = i_true[k] + _gauss(rng, std.get("current_a", 0.0))
            t_meas = t_true[k] + _gauss(rng, std.get("temp_c", 0.0))
            soc_meas = float(np.clip(soc_true[k] + _gauss(rng, std.get("soc", 0.0)), 0.0, 1.0))
            u_meas = u_t[k] + _gauss(rng, std.get("voltage_v", 0.0))
        else:
            i_meas, t_meas, soc_meas, u_meas = i_true[k], t_true[k], soc_true[k], u_t[k]
        out["time_s"][k] = k * dt_s
        out["step"][k] = k
        out["cmd_id"][k] = cmd_ids[k]
        out["cutoff"][k] = cutoff[k]
        out["i_true_a"][k] = i_true[k]
        out["i_meas_a"][k] = i_meas
        out["t_true_c"][k] = t_true[k]
        out["t_meas_c"][k] = t_meas
        out["soc_true"][k] = soc_true[k]
        out["soc_meas"][k] = soc_meas
        out["u_ocv_v"][k] = u_ocv[k]
        out["r0_ohm"][k] = r0[k]
        out["r1_ohm"][k] = r1[k]
        out["c1_f"][k] = c1[k]
        out["tau1_s"][k] = tau1[k]
        out["u_p_v"][k] = u_p[k]
        out["u_t_true_v"][k] = u_t[k]
        out["u_t_meas_v"][k] = u_meas
    out["mode"] = modes
    return out


def run_sim(
    *,
    out: str | Path = OUTPUT_CSV,
    sequence: list[dict] | None = None,
    soc0: float | None = None,
    t_ambient_c: float | None = None,
    dt_s: float | None = None,
    noise_enable: bool | None = None,
    noise_seed: int | None = None,
    noise_std: dict[str, float] | None = None,
    enable_cutoff: bool | None = None,
    extra_meta: list[str] | None = None,
    summarize: bool = True,
    thermal: bool = False,
    verbose: bool = True,
) -> Path:
    seq = SEQUENCE if sequence is None else list(sequence)
    s0 = SOC0 if soc0 is None else float(soc0)
    t_c = T_AMBIENT_C if t_ambient_c is None else float(t_ambient_c)
    dt = DT_S if dt_s is None else float(dt_s)
    n_en = NOISE_ENABLE if noise_enable is None else bool(noise_enable)
    seed = NOISE_SEED if noise_seed is None else int(noise_seed)
    std = dict(NOISE_STD if noise_std is None else noise_std)
    cut = ENABLE_CUTOFF if enable_cutoff is None else bool(enable_cutoff)
    if verbose:
        print("PyBaMM 100Ah NMC  SPM" + (" + lumped thermal" if thermal else " 等温"))
        print(f"初始 SOC: {s0:.3f} | 初始温度: {t_c:.1f} C | dt={dt} s")
        print("Solving... ", flush=True)
    data = simulate(
        seq,
        dt_s=dt,
        soc0=s0,
        t_ambient_c=t_c,
        noise_enable=n_en,
        noise_seed=seed,
        noise_std=std,
        enable_cutoff=cut,
        thermal=thermal,
        verbose=verbose,
    )
    if verbose:
        print("done")
    meta = list(extra_meta or [])
    meta.extend(
        [
            "# backend=pybamm",
            "# pybamm_model=SPM",
            "# pybamm_params=Chen2020_100Ah",
            "# teacher_rc=ecm_eval",
            f"# pybamm_thermal={int(thermal)}",
            "# schema_version=1.1.0",
        ]
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
        source="nmc100ah_pybamm",
    )
    print(f"已写出 {path}")
    if summarize:
        _summarize(data, dt_s=dt)
    return path


def _ecm_flags_incompatible(**kwargs) -> str | None:
    bad = []
    if kwargs.get("rc2"):
        bad.append("--rc2")
    if abs(float(kwargs.get("soh", 1.0)) - 1.0) > 1e-12:
        bad.append("--soh")
    if abs(float(kwargs.get("soh_capacity", 1.0)) - 1.0) > 1e-12:
        bad.append("--soh-capacity")
    for key, flag in (("r0_scale", "--r0-scale"), ("r1_scale", "--r1-scale"), ("c1_scale", "--c1-scale")):
        if abs(float(kwargs.get(key, 1.0)) - 1.0) > 1e-12:
            bad.append(flag)
    if not bad:
        return None
    return "--pybamm 不能与 " + " / ".join(bad) + " 一起用（那些是 ECM 开关）"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyBaMM SPM 100Ah NMC 时域仿真，CSV 格式与 ECM 生成器相同"
    )
    parser.add_argument("--out", default=OUTPUT_CSV, help="输出 CSV 路径")
    parser.add_argument("--soc0", type=float, default=None, help="覆盖头部 SOC0")
    parser.add_argument("--t-ambient", type=float, default=None, help="覆盖头部环境温度 / °C")
    parser.add_argument("--seed", type=int, default=None, help="覆盖头部 NOISE_SEED")
    parser.add_argument("--no-noise", action="store_true", help="关闭噪声")
    parser.add_argument("--thermal", action="store_true", help="打开 lumped 热模型（默认等温，与网格 T 轴一致）")
    args = parser.parse_args()
    run_sim(
        out=args.out,
        soc0=args.soc0,
        t_ambient_c=args.t_ambient,
        noise_enable=False if args.no_noise else None,
        noise_seed=args.seed,
        thermal=args.thermal,
    )


if __name__ == "__main__":
    main()
