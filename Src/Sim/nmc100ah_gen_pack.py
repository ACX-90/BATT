"""包级生成器：一条真电流 → N_s 只电压（Doc/06-a §2 / §7）。

默认 ECM；`--engine pybamm` 用 SPM 顶真包。2A3 / 2A4 真值是每芯通道，强制 ecm。不覆盖 Data/grid。

    python Src/Sim/nmc100ah_gen_pack.py --exp 2a1 --n 8 --engine pybamm --out-dir Data/pack/2a1_smoke
    python Src/Sim/nmc100ah_gen_pack.py --exp 2a1 --n 180 --engine ecm --out-dir Data/pack/2a1
    python Src/Sim/nmc100ah_gen_pack.py --exp 2a3 --n 8 --seed 203 --out-dir Data/pack/2a3_n8
    python Src/Sim/nmc100ah_gen_pack.py --exp 2a4 --n 8 --seed 204 --out-dir Data/pack/2a4_n8
    python Src/Sim/nmc100ah_gen_pack.py --exp 2b --n 8 --seed 205 --out-dir Data/pack/2b_n8
    python Src/Sim/nmc100ah_gen_pack.py --exp 2c --n 8 --seed 208 --out-dir Data/pack/2c_n8
    python Src/Sim/nmc100ah_gen_pack.py --exp 2e --n 8 --seed 206 --out-dir Data/pack/2e_n8
    python Src/Sim/nmc100ah_gen_pack.py --exp 2d1 --n 8 --seed 207 --out-dir Data/pack/2d1_n8
    python Src/Sim/nmc100ah_gen_pack.py --exp 2d2 --n 8 --seed 207 --out-dir Data/pack/2d2_n8
    python Src/Sim/nmc100ah_gen_pack.py --exp 2g --n 8 --seed 209 --out-dir Data/pack/2g_n8
    python Src/Sim/nmc100ah_gen_pack.py --exp 2h1 --n 8 --seed 210 --park-h 6 --out-dir Data/pack/2h1_n8
    python Src/Sim/nmc100ah_gen_pack.py --exp 2h1 --n 180 --seed 210 --out-dir Data/pack/2h1
    python Src/Sim/nmc100ah_gen_pack.py --exp 2h3 --n 8 --seed 212 --park-h 1 --charge-min 5 --out-dir Data/pack/2h3_n8
    python Src/Sim/nmc100ah_gen_pack.py --exp 2h3 --n 180 --seed 212 --out-dir Data/pack/2h3
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parent
REPO_ROOT = SIM_DIR.parent.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from nmc100ah_ecm import NMC100AhECM, make_ecm  # noqa: E402
from nmc100ah_ecm_params import default_param_set  # noqa: E402
from nmc100ah_gen import (  # noqa: E402
    DT_S,
    ENABLE_CUTOFF,
    NOISE_STD,
    SEQUENCE,
    U_P0,
    _gauss,
    expand_sequence,
    simulate as ecm_simulate,
)
from nmc100ah_gen_long import SEQ_CC_REST  # noqa: E402

FORBIDDEN_OUT = {
    "Data/grid",
    "Data/grid_pybamm",
    "Data/ai_mlp",
    "Data/ai_local",
    "Data/soh_k115",
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def n_aged_of(n: int) -> int:
    if n <= 8:
        return min(2, n)
    return int(round(n * 20 / 180)) or 1


# 2H：BQ79718 模组切法（06-a §5.7）。正式 10×18；烟测可用 4×2。
I_AFE_A = 0.012
I_AFE_TOP_EXTRA_A = 0.002
PARK_H_DEFAULT = 48.0
PARK_DT_S = 5.0  # PC 停放默认 5 s（可 --park-dt 1）；门控仍按秒验
CHARGE_C_RATE = 1.0  # 2H3：1C 充再停（06-a §3.6 / §5.7）
CHARGE_MIN_DEFAULT = 5.0
I_EDGE_PIN_A = 20.0  # 钉上一档大电流：|I|≥20 A


def module_layout(n: int) -> tuple[int, int]:
    if n >= 180:
        return 10, 18
    if n % 2 == 0 and n >= 4:
        return n // 2, 2
    return 1, n


def afe_current_a(exp: str, cell_id: int, *, n: int) -> float:
    i = float(I_AFE_A)
    if exp == "2h2":
        n_mod, cpm = module_layout(n)
        # 每模组顶芯（模组内最后一只）再 +2 mA
        if cpm > 0 and (cell_id % cpm) == (cpm - 1) and cell_id // cpm < n_mod:
            i += float(I_AFE_TOP_EXTRA_A)
    return i


def _ocv_vec(soc: np.ndarray, t_celsius: float) -> np.ndarray:
    # 与 nmc100ah_gen.ocv_nmc / KF.ocv 同一张表；向量化给停放批仿真用
    from nmc100ah_gen import _OCV_SOC, _OCV_V, _OCV_DUDT

    s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
    return np.interp(s, _OCV_SOC, _OCV_V) + _OCV_DUDT * (float(t_celsius) - 25.0)


def simulate_afe_park(
    cells: list[dict],
    *,
    i_afe: np.ndarray,
    dt_s: float,
    park_s: float,
    t_c: float = 25.0,
    r_refresh_s: float = 60.0,
) -> dict[str, np.ndarray]:
    """I_cell=I_AFE 常数停放；ECM 吃 I_cell。R 按分钟刷新（SOC 只掉 <1 pp）。"""
    n = len(cells)
    n_steps = int(round(float(park_s) / float(dt_s)))
    if n_steps < 2:
        raise ValueError("park 太短")
    model = make_ecm(r0_scale=1.0, r1_scale=1.0, c1_scale=1.0)
    soc = np.array([float(c["soc0"]) for c in cells], dtype=float)
    u_p = np.zeros(n, dtype=float)
    q_as = np.array([float(c.get("q_ah", 100.0)) * 3600.0 for c in cells], dtype=float)
    i_afe = np.asarray(i_afe, dtype=float).reshape(n)
    refresh = max(1, int(round(float(r_refresh_s) / float(dt_s))))

    u_t = np.empty((n_steps, n), dtype=np.float32)
    soc_out = np.empty((n_steps, n), dtype=np.float32)
    u_ocv = np.empty((n_steps, n), dtype=np.float32)
    u_p_out = np.empty((n_steps, n), dtype=np.float32)
    t_true = np.full((n_steps, n), float(t_c), dtype=np.float32)
    cutoff = np.zeros((n_steps, n), dtype=np.float32)
    r0 = np.empty(n)
    r1 = np.empty(n)
    c1 = np.empty(n)
    alpha = np.empty(n)

    for k in range(n_steps):
        if k % refresh == 0:
            for i in range(n):
                a, b, c = model.evaluate(i_a=float(i_afe[i]), t_celsius=t_c, soc=float(soc[i]))
                r0[i], r1[i], c1[i] = float(a), float(b), float(c)
            alpha = np.exp(-float(dt_s) / np.maximum(r1 * c1, 1e-12))
        u_p = alpha * u_p + r1 * (1.0 - alpha) * i_afe
        uocv = _ocv_vec(soc, t_c)
        ut = uocv - i_afe * r0 - u_p
        soc = np.clip(soc - i_afe * float(dt_s) / q_as, 0.0, 1.0)
        u_t[k] = ut
        soc_out[k] = soc
        u_ocv[k] = uocv
        u_p_out[k] = u_p
    return {
        "u_t_true": u_t,
        "soc_true": soc_out,
        "t_true": t_true,
        "u_ocv": u_ocv,
        "u_p": u_p_out.astype(np.float32),
        "cutoff": cutoff,
    }


def simulate_charge_afe_park(
    cells: list[dict],
    *,
    i_afe: np.ndarray,
    dt_s: float,
    charge_s: float,
    park_s: float,
    t_c: float = 25.0,
    capacity_ah: float = 100.0,
    r_refresh_s: float = 60.0,
) -> dict[str, np.ndarray]:
    """1C 充数分钟 → AFE 停放。滚已有 Up 的 R/τ 钉上一档大电流（充电 1C），驱动项用 I_cell。

    06-a §3.6：不要因现在 +12 mA 就切放电表。错误实现会把充电回弹拧成放电形状。
    """
    n = len(cells)
    n_chg = max(2, int(round(float(charge_s) / float(dt_s))))
    n_park = max(2, int(round(float(park_s) / float(dt_s))))
    n_steps = n_chg + n_park
    model = make_ecm(r0_scale=1.0, r1_scale=1.0, c1_scale=1.0)
    soc = np.array([float(c["soc0"]) for c in cells], dtype=float)
    u_p = np.zeros(n, dtype=float)
    q_as = np.array([float(c.get("q_ah", capacity_ah)) * 3600.0 for c in cells], dtype=float)
    i_afe = np.asarray(i_afe, dtype=float).reshape(n)
    i_chg_pack = -float(CHARGE_C_RATE) * float(capacity_ah)  # 放电为正 → 充电为负
    refresh = max(1, int(round(float(r_refresh_s) / float(dt_s))))

    u_t = np.empty((n_steps, n), dtype=np.float32)
    soc_out = np.empty((n_steps, n), dtype=np.float32)
    u_ocv = np.empty((n_steps, n), dtype=np.float32)
    u_p_out = np.empty((n_steps, n), dtype=np.float32)
    t_true = np.full((n_steps, n), float(t_c), dtype=np.float32)
    cutoff = np.zeros((n_steps, n), dtype=np.float32)
    i_pack = np.empty(n_steps, dtype=np.float64)
    i_cell = np.empty((n_steps, n), dtype=np.float32)
    i_lookup = np.empty((n_steps, n), dtype=np.float32)

    r0 = np.empty(n)
    r1 = np.empty(n)
    c1 = np.empty(n)
    alpha = np.empty(n)
    # 上一档 |I|≥20 A 的查表电流（符号保留）；很久没有大电流才退回 I_cell
    i_pin = np.zeros(n, dtype=float)

    for k in range(n_steps):
        if k < n_chg:
            i_p = i_chg_pack
            i_c = i_p + i_afe  # I_cell = I_pack + I_AFE
        else:
            i_p = 0.0
            i_c = i_afe.copy()
        i_pack[k] = i_p
        i_cell[k] = i_c

        # 钉上一档大电流（§3.6）
        big = np.abs(i_c) >= float(I_EDGE_PIN_A)
        i_pin = np.where(big, i_c, i_pin)
        # 若尚未有过大电流（不应发生在充后停），退回当前 I_cell
        no_hist = np.abs(i_pin) < 1e-12
        i_look = np.where(no_hist, i_c, i_pin)
        i_lookup[k] = i_look

        if k % refresh == 0 or k == n_chg:
            for i in range(n):
                a, b, c = model.evaluate(i_a=float(i_look[i]), t_celsius=t_c, soc=float(soc[i]))
                r0[i], r1[i], c1[i] = float(a), float(b), float(c)
            alpha = np.exp(-float(dt_s) / np.maximum(r1 * c1, 1e-12))

        u_p = alpha * u_p + r1 * (1.0 - alpha) * i_c
        # 欧姆仍用现电流；极化 R0 查表也钉 pin（与 τ 一致）
        uocv = _ocv_vec(soc, t_c)
        ut = uocv - i_c * r0 - u_p
        soc = np.clip(soc - i_c * float(dt_s) / q_as, 0.0, 1.0)
        u_t[k] = ut
        soc_out[k] = soc
        u_ocv[k] = uocv
        u_p_out[k] = u_p

    return {
        "u_t_true": u_t,
        "soc_true": soc_out,
        "t_true": t_true,
        "u_ocv": u_ocv,
        "u_p": u_p_out.astype(np.float32),
        "cutoff": cutoff,
        "i_pack": i_pack.astype(np.float64),
        "i_cell": i_cell,
        "i_lookup": i_lookup.astype(np.float32),
        "n_charge_steps": n_chg,
        "n_park_steps": n_park,
    }


def assign_cells(
    exp: str,
    n: int,
    *,
    seed: int,
    k_aged: float = 1.15,
    t_nom: float = 25.0,
    t_cold: float = -10.0,
    b_i: float = 0.0,
) -> tuple[list[dict], float]:
    # 2A4 的 Q 用 rng(204) 另抽；s0 不能和 Q_i 共用同一串 z（06-a §5.1.2）。
    if exp == "2a4":
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 40]))
    else:
        rng = np.random.default_rng(seed)
    n_aged = n_aged_of(n)
    cells = []
    for i in range(n):
        aged = i < n_aged
        soc0 = float(np.clip(rng.normal(0.70, 0.01), 0.65, 0.75))
        t_c = t_nom
        k = 1.0
        if exp == "2a1":
            k = k_aged if aged else 1.0
        elif exp == "2a2":
            if aged:
                k = k_aged
                t_c = t_cold
        elif exp in {"2a3", "2a4"}:
            k = 1.0
            aged = False
        elif exp == "2b":
            k = 1.0
            aged = False
        elif exp == "2g":
            # 06-a §5.6：全包 hatQ 错；k=1、b_I=0、Qi≡100（不开 --tol）
            k = 1.0
            aged = False
        elif exp == "2c":
            k = k_aged if aged else 1.0
        elif exp == "2e":
            k = k_aged
            aged = True
        elif exp in {"2d1", "2d2"}:
            k = 1.0
            aged = False
        elif exp in {"2h1", "2h2", "2h3"}:
            # 06-a §5.7：k=1、b_I=0、化学 I_sd=0；AFE 另挂 I_cell
            k = 1.0
            aged = False
        else:
            raise ValueError(f"未实现 --exp {exp}")
        cells.append(
            {
                "id": i,
                "aged": bool(aged),
                "soc0": soc0,
                "t_c": float(t_c),
                "k": float(k),
                "q_ah": 100.0,
            }
        )
    bias = 5.0 if exp in {"2b", "2c"} else float(b_i)
    return cells, bias


def _lognormal_clip(rng: np.random.Generator, star: float, sigma: float) -> tuple[float, float]:
    z = float(rng.normal())
    val = float(star * np.exp(sigma * z))
    lo, hi = 0.70 * float(star), 1.30 * float(star)
    if lo > hi:
        lo, hi = hi, lo
    return float(np.clip(val, lo, hi)), z


def _draw_shape(rng: np.random.Generator, ch, *, sig_shape: float, sig_phase: float, sig_ea: float) -> dict:
    a_low, z_al = _lognormal_clip(rng, ch.soc.a_low, sig_shape)
    a_high, z_ah = _lognormal_clip(rng, ch.soc.a_high, sig_shape)
    quad, z_q = _lognormal_clip(rng, ch.soc.quad, sig_shape)
    amp, z_a = _lognormal_clip(rng, ch.phase.amplitude, sig_phase)
    ea, z_e = _lognormal_clip(rng, ch.temperature.ea_j_per_mol, sig_ea)
    return {
        "a_low": a_low,
        "a_high": a_high,
        "quad": quad,
        "phase_A": amp,
        "ea": ea,
        "z": {
            "a_low": z_al,
            "a_high": z_ah,
            "quad": z_q,
            "phase_A": z_a,
            "ea": z_e,
        },
    }


def _draw_2a3_channel(rng: np.random.Generator, ch) -> dict:
    ref, z_ref = _lognormal_clip(rng, ch.ref_value, 0.05)
    shape = _draw_shape(rng, ch, sig_shape=0.15, sig_phase=0.20, sig_ea=0.10)
    shape["ref_value"] = ref
    shape["ref_ratio"] = float(ref / ch.ref_value)
    shape["z"]["ref_value"] = z_ref
    return shape


def sample_2a3_channels(n: int, seed: int) -> list[dict]:
    """06-a §5.1.1：只抽幅度，冻位置 / 电流维 / C1。种子 203，z 一次抽齐。"""
    star = default_param_set()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        draws.append({"r0": _draw_2a3_channel(rng, star.r0), "r1": _draw_2a3_channel(rng, star.r1)})
    return draws


def sample_2a4(n: int, seed: int) -> list[dict]:
    """06-a §5.1.2：Q_i 抽签 + R_ref ∝ 1/Q + 一半形状 σ。种子 204，不叠 σ_ref=0.05。"""
    star = default_param_set()
    qn = float(star.cell.capacity_ah)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        z_q = float(rng.normal())
        q = float(np.clip(qn * np.exp(0.01 * z_q), 0.97 * qn, 1.03 * qn))
        z_r0 = float(rng.normal())
        z_r1 = float(rng.normal())
        r0_ref = float(star.r0.ref_value * (qn / q) * np.exp(0.03 * z_r0))
        r1_ref = float(star.r1.ref_value * (qn / q) * np.exp(0.03 * z_r1))
        r0 = _draw_shape(rng, star.r0, sig_shape=0.075, sig_phase=0.10, sig_ea=0.05)
        r1 = _draw_shape(rng, star.r1, sig_shape=0.075, sig_phase=0.10, sig_ea=0.05)
        r0["ref_value"] = r0_ref
        r0["ref_ratio"] = float(r0_ref / star.r0.ref_value)
        r0["z"]["ref_value"] = z_r0
        r1["ref_value"] = r1_ref
        r1["ref_ratio"] = float(r1_ref / star.r1.ref_value)
        r1["z"]["ref_value"] = z_r1
        draws.append(
            {
                "q_ah": q,
                "q_ratio": float(q / qn),
                "z_q": z_q,
                "r0": r0,
                "r1": r1,
            }
        )
    return draws


def ecm_from_2a3(draw: dict) -> NMC100AhECM:
    star = default_param_set()
    r0d, r1d = draw["r0"], draw["r1"]
    r0 = replace(
        star.r0,
        ref_value=float(r0d["ref_value"]),
        soc=replace(star.r0.soc, a_low=float(r0d["a_low"]), a_high=float(r0d["a_high"]), quad=float(r0d["quad"])),
        phase=replace(star.r0.phase, amplitude=float(r0d["phase_A"])),
        temperature=replace(star.r0.temperature, ea_j_per_mol=float(r0d["ea"])),
    )
    r1 = replace(
        star.r1,
        ref_value=float(r1d["ref_value"]),
        soc=replace(star.r1.soc, a_low=float(r1d["a_low"]), a_high=float(r1d["a_high"]), quad=float(r1d["quad"])),
        phase=replace(star.r1.phase, amplitude=float(r1d["phase_A"])),
        temperature=replace(star.r1.temperature, ea_j_per_mol=float(r1d["ea"])),
    )
    return NMC100AhECM(replace(star, r0=r0, r1=r1))


def _simulate_cell(
    engine: str,
    cell: dict,
    sequence: list[dict],
    *,
    dt_s: float,
    noise_seed: int,
    verbose: bool,
) -> dict[str, np.ndarray]:
    if engine == "ecm":
        if cell.get("channels"):
            model = ecm_from_2a3(cell["channels"])
        else:
            model = make_ecm(r0_scale=cell["k"], r1_scale=cell["k"], c1_scale=1.0)
        q_ah = float(cell.get("q_ah", 100.0))
        soc_q = None if abs(q_ah - 100.0) < 1e-9 else q_ah
        return ecm_simulate(
            model,
            sequence,
            dt_s=dt_s,
            soc0=cell["soc0"],
            t_ambient_c=cell["t_c"],
            u_p0=U_P0,
            noise_enable=False,
            noise_seed=noise_seed,
            enable_cutoff=ENABLE_CUTOFF,
            soc_capacity_ah=soc_q,
        )
    from nmc100ah_pybamm import simulate as pybamm_simulate

    return pybamm_simulate(
        sequence,
        dt_s=dt_s,
        soc0=cell["soc0"],
        t_ambient_c=cell["t_c"],
        noise_enable=False,
        noise_seed=noise_seed,
        enable_cutoff=ENABLE_CUTOFF,
        verbose=verbose,
        k_r=1.0,
    )


def _repeat_trips(base: list[dict], n_trips: int, rest_s: float, *, dt_s: float) -> tuple[list[dict], list[int]]:
    """同一条 SEQUENCE 串 n 次，中间休息 rest_s（06-a §5.4 2D2，~3τ1）。"""
    seq: list[dict] = []
    starts: list[int] = []
    for i in range(int(n_trips)):
        if i:
            seq.append({"mode": "rest", "duration_s": float(rest_s)})
        n_so_far = len(expand_sequence(seq, dt_s=dt_s, capacity_ah=100.0, t_default=25.0)) if seq else 0
        starts.append(int(n_so_far))
        seq.extend(list(base))
    return seq, starts


def generate_pack(
    *,
    exp: str,
    n: int,
    engine: str,
    out_dir: Path,
    seed: int,
    k_aged: float = 1.15,
    write_csv_samples: bool = True,
    park_h: float | None = None,
    park_dt_s: float | None = None,
    charge_min: float | None = None,
) -> Path:
    rel = str(out_dir.relative_to(REPO_ROOT)).replace("\\", "/") if out_dir.is_relative_to(REPO_ROOT) else str(out_dir)
    if rel.rstrip("/") in FORBIDDEN_OUT or any(rel.startswith(x + "/") for x in FORBIDDEN_OUT):
        raise RuntimeError(f"禁止写到 {rel}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cells, b_i = assign_cells(exp, n, seed=seed, k_aged=k_aged)
    if exp == "2a3":
        draws = sample_2a3_channels(n, seed)
        for cell, draw in zip(cells, draws):
            cell["channels"] = draw
    elif exp == "2a4":
        draws = sample_2a4(n, seed)
        for cell, draw in zip(cells, draws):
            cell["q_ah"] = draw["q_ah"]
            cell["q_ratio"] = draw["q_ratio"]
            cell["z_q"] = draw["z_q"]
            cell["channels"] = {"r0": draw["r0"], "r1": draw["r1"]}

    if exp in {"2h1", "2h2", "2h3"}:
        if abs(b_i) > 1e-12:
            raise RuntimeError("2H 禁止叠 2B 的 5 A 零偏")
        if engine != "ecm":
            raise RuntimeError("2H 强制 --engine ecm")
        dt_s = float(PARK_DT_S if park_dt_s is None else park_dt_s)
        hours = float(PARK_H_DEFAULT if park_h is None else park_h)
        park_s = hours * 3600.0
        chg_min = float(CHARGE_MIN_DEFAULT if charge_min is None else charge_min)
        chg_s = chg_min * 60.0 if exp == "2h3" else 0.0
        n_mod, cpm = module_layout(n)
        # 2H3 与 2H1 同均匀 AFE（顶芯不加）
        afe_exp = "2h1" if exp == "2h3" else exp
        i_afe = np.array([afe_current_a(afe_exp, i, n=n) for i in range(n)], dtype=float)
        for i, cell in enumerate(cells):
            cell["i_afe_a"] = float(i_afe[i])
            cell["module"] = int(i // cpm) if cpm else 0
            cell["module_top"] = bool(cpm and (i % cpm) == (cpm - 1))
        print(
            f"gen_pack exp={exp} n={n} engine={engine} seed={seed} b_I=0  "
            f"park={hours:g} h dt={dt_s:g}s  I_AFE={I_AFE_A*1e3:.0f} mA  "
            f"modules={n_mod}x{cpm}"
            + (f"  charge={chg_min:g} min@1C" if exp == "2h3" else ""),
            flush=True,
        )
        for i, cell in enumerate(cells):
            print(
                f"  cell {i:03d}/{n}  mod={cell['module']} top={int(cell['module_top'])}  "
                f"I_afe={cell['i_afe_a']*1e3:.1f} mA  soc0={cell['soc0']:.3f}",
                flush=True,
            )
        if exp == "2h3":
            sim = simulate_charge_afe_park(
                cells,
                i_afe=i_afe,
                dt_s=dt_s,
                charge_s=chg_s,
                park_s=park_s,
                t_c=25.0,
            )
            n_steps = int(sim["u_t_true"].shape[0])
            time_s = np.arange(n_steps, dtype=float) * dt_s
            i_true = np.asarray(sim["i_pack"], dtype=float)
            i_cell = np.asarray(sim["i_cell"], dtype=np.float32)
        else:
            sim = simulate_afe_park(cells, i_afe=i_afe, dt_s=dt_s, park_s=park_s, t_c=25.0)
            n_steps = int(sim["u_t_true"].shape[0])
            time_s = np.arange(n_steps, dtype=float) * dt_s
            i_true = np.zeros(n_steps, dtype=float)  # 分流器 / 包电流
            i_cell = np.repeat(i_afe.reshape(1, -1), n_steps, axis=0).astype(np.float32)
        cmd_id = np.zeros(n_steps, dtype=float)
        u_t_true = sim["u_t_true"]
        soc_true = sim["soc_true"]
        t_true = sim["t_true"]
        u_ocv = sim["u_ocv"]
        u_p = sim["u_p"]
        cutoff = sim["cutoff"]
        rng = np.random.default_rng(seed + 1)
        # I_meas = I_pack + 噪声；禁止把 12 mA 加进 I_meas
        i_meas = i_true + np.array([_gauss(rng, NOISE_STD["current_a"]) for _ in range(n_steps)])
        u_t_meas = u_t_true + rng.normal(0.0, NOISE_STD["voltage_v"], size=u_t_true.shape).astype(np.float32)
        t_meas = t_true + rng.normal(0.0, NOISE_STD["temp_c"], size=t_true.shape).astype(np.float32)
        save_kw = dict(
            time_s=time_s,
            i_true=i_true,
            i_meas=i_meas,
            i_cell=i_cell,
            u_t_true=u_t_true,
            u_t_meas=u_t_meas,
            t_true=t_true,
            t_meas=t_meas,
            soc_true=soc_true,
            u_ocv=u_ocv,
            u_p=u_p,
            cutoff=cutoff,
            cmd_id=cmd_id,
        )
        if exp == "2h3" and "i_lookup" in sim:
            save_kw["i_lookup"] = np.asarray(sim["i_lookup"], dtype=np.float32)
        np.savez_compressed(out_dir / "pack.npz", **save_kw)
        if exp == "2h3":
            seq = [
                {"mode": "charge", "duration_s": float(chg_s), "c_rate": float(CHARGE_C_RATE)},
                {"mode": "rest", "duration_s": float(park_s)},
            ]
            n_chg_steps = int(sim["n_charge_steps"])
            wave = "charge_afe_park"
            note = (
                "2H3：1C 充数分钟再停；I_meas 充段≈−100 A、停放≈0；"
                "I_cell=I_pack+I_AFE；滚 Up 的 R/τ 钉上一档充电大电流（§3.6）；"
                "禁止把 12 mA 估进 hat b_I。"
            )
        else:
            seq = [{"mode": "rest", "duration_s": float(park_s)}]
            n_chg_steps = 0
            wave = "afe_park"
            note = (
                "2H：I_meas≈0（分流器）；I_cell=I_AFE 串过模组进 ECM/SOC 真值；"
                "禁止把 12 mA 估进 hat b_I。§3.5 无边沿不写 k。"
            )
        meta = {
            "exp": exp,
            "n": n,
            "n_aged": 0,
            "engine": engine,
            "dt_s": dt_s,
            "n_steps": n_steps,
            "b_I": 0.0,
            "seed": seed,
            "k_aged": k_aged,
            "channel_sigma": None,
            "wave": wave,
            "park_h": hours,
            "charge_min": float(chg_min) if exp == "2h3" else 0.0,
            "n_charge_steps": int(n_chg_steps),
            "i_afe_a": float(I_AFE_A),
            "i_afe_top_extra_a": float(I_AFE_TOP_EXTRA_A) if exp == "2h2" else 0.0,
            "n_modules": n_mod,
            "cells_per_module": cpm,
            # 2H3 不按趟拆门：整段一趟，停放靠 §3.5 skip_park（拆趟会让短停放被 slope 误闩）
            "trips": [0],
            "n_trips": 1,
            "sequence": seq,
            "cells": cells,
            "noise_std": dict(NOISE_STD),
            "hat_q_ah": 100.0,
            "capacity_ah_true": 100.0,
            "capacity_scale": 1.0,
            "suggested_win": max(2, int(round(10.0 / dt_s))),
            "note": note,
        }
        (out_dir / "pack.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if write_csv_samples:
            from nmc100ah_gen import write_csv

            sample_ids = [0]
            if exp == "2h2":
                tops = [c["id"] for c in cells if c.get("module_top")]
                if tops:
                    sample_ids.append(tops[0])
            mode_arr = np.array(["afe_park"] * n_steps, dtype=object)
            if exp == "2h3":
                mode_arr[: int(n_chg_steps)] = "charge"
                mode_arr[int(n_chg_steps) :] = "afe_park"
            for cid in dict.fromkeys(sample_ids):
                data = {
                    "time_s": time_s,
                    "step": np.arange(n_steps, dtype=float),
                    "cmd_id": cmd_id,
                    "mode": mode_arr,
                    "cutoff": cutoff[:, cid],
                    "i_true_a": np.asarray(i_cell[:, cid], dtype=float),
                    "i_meas_a": i_meas,
                    "t_true_c": t_true[:, cid],
                    "t_meas_c": t_meas[:, cid],
                    "soc_true": soc_true[:, cid],
                    "soc_meas": soc_true[:, cid],
                    "u_ocv_v": u_ocv[:, cid],
                    "r0_ohm": np.full(n_steps, np.nan),
                    "r1_ohm": np.full(n_steps, np.nan),
                    "c1_f": np.full(n_steps, np.nan),
                    "tau1_s": np.full(n_steps, np.nan),
                    "u_p_v": u_p[:, cid],
                    "u_t_true_v": u_t_true[:, cid],
                    "u_t_meas_v": u_t_meas[:, cid],
                }
                write_csv(
                    out_dir / f"cell_{cid:03d}.csv",
                    data,
                    dt_s=dt_s,
                    soc0=cells[cid]["soc0"],
                    noise_enable=True,
                    noise_seed=seed,
                    noise_std=dict(NOISE_STD),
                    sequence=seq,
                    extra_meta=[
                        f"# pack_exp={exp}",
                        f"# engine={engine}",
                        f"# cell={cid}",
                        f"# i_afe_a={cells[cid]['i_afe_a']}",
                        f"# i_meas_is_pack=1",
                        f"# b_I=0",
                    ],
                    source="nmc100ah_gen_pack",
                )
        print(f"已写出 {out_dir / 'pack.npz'}", flush=True)
        return out_dir

    dt_s = DT_S
    trip_starts = [0]
    if exp in {"2b", "2g"}:
        seq = list(SEQ_CC_REST)
    elif exp == "2c":
        # 06-a §2.4 / §5.3：cc_rest 后再接 SEQUENCE 的 1C 放电 + 回弹。
        pulse = [
            {"mode": "discharge", "duration_s": 180.0, "c_rate": 1.0},
            {"mode": "rest", "duration_s": 120.0},
        ]
        seq = list(SEQ_CC_REST) + pulse
        n_cc = len(expand_sequence(list(SEQ_CC_REST), dt_s=dt_s, capacity_ah=100.0, t_default=25.0))
        trip_starts = [0, int(n_cc)]
    elif exp == "2d2":
        seq, trip_starts = _repeat_trips(SEQUENCE, 3, 60.0, dt_s=dt_s)
    else:
        seq = list(SEQUENCE)
    plan = expand_sequence(seq, dt_s=dt_s, capacity_ah=100.0, t_default=25.0)
    n_steps = len(plan)
    i_true = np.array([p[2] for p in plan], dtype=float)
    time_s = np.arange(n_steps, dtype=float) * dt_s
    cmd_id = np.array([p[0] for p in plan], dtype=float)

    u_t_true = np.empty((n_steps, n), dtype=np.float32)
    soc_true = np.empty((n_steps, n), dtype=np.float32)
    t_true = np.empty((n_steps, n), dtype=np.float32)
    u_ocv = np.empty((n_steps, n), dtype=np.float32)
    cutoff = np.zeros((n_steps, n), dtype=np.float32)

    print(f"gen_pack exp={exp} n={n} engine={engine} seed={seed} b_I={b_i:g} A", flush=True)
    for i, cell in enumerate(cells):
        extra = ""
        if cell.get("channels"):
            extra = (
                f"  R0ref×{cell['channels']['r0']['ref_ratio']:.3f}"
                f" R1ref×{cell['channels']['r1']['ref_ratio']:.3f}"
            )
        if abs(float(cell.get("q_ah", 100.0)) - 100.0) > 1e-9:
            extra += f"  Q={cell['q_ah']:.2f}Ah"
        print(
            f"  cell {i:03d}/{n}  aged={int(cell['aged'])} k={cell['k']:.3f}  "
            f"soc0={cell['soc0']:.3f} T={cell['t_c']:+.1f}{extra}",
            flush=True,
        )
        data = _simulate_cell(
            engine,
            cell,
            seq,
            dt_s=dt_s,
            noise_seed=seed + 17 * i,
            verbose=(i == 0 and engine == "pybamm"),
        )
        u_cell = np.asarray(data["u_t_true_v"], dtype=float)
        if engine == "pybamm" and abs(float(cell["k"]) - 1.0) > 1e-12:
            # SPM 的 Contact resistance 在本模型里不进端电压；2A1 的 k 用串联 IR 叠上去。
            r_extra = (float(cell["k"]) - 1.0) * 1.45e-3
            u_cell = u_cell - np.asarray(data["i_true_a"], dtype=float) * r_extra
        u_t_true[:, i] = u_cell
        soc_true[:, i] = data["soc_true"]
        t_true[:, i] = data["t_true_c"]
        u_ocv[:, i] = data["u_ocv_v"]
        cutoff[:, i] = data["cutoff"]

    rng = np.random.default_rng(seed + 1)
    i_meas = i_true + b_i + np.array([_gauss(rng, NOISE_STD["current_a"]) for _ in range(n_steps)])
    u_t_meas = u_t_true + rng.normal(0.0, NOISE_STD["voltage_v"], size=u_t_true.shape).astype(np.float32)
    t_meas = t_true + rng.normal(0.0, NOISE_STD["temp_c"], size=t_true.shape).astype(np.float32)

    np.savez_compressed(
        out_dir / "pack.npz",
        time_s=time_s,
        i_true=i_true,
        i_meas=i_meas,
        u_t_true=u_t_true,
        u_t_meas=u_t_meas,
        t_true=t_true,
        t_meas=t_meas,
        soc_true=soc_true,
        u_ocv=u_ocv,
        cutoff=cutoff,
        cmd_id=cmd_id,
    )
    meta = {
        "exp": exp,
        "n": n,
        "n_aged": int(sum(c["aged"] for c in cells)),
        "engine": engine,
        "dt_s": dt_s,
        "n_steps": n_steps,
        "b_I": b_i,
        "seed": seed,
        "k_aged": k_aged,
        "channel_sigma": (
            {"ref": 0.05, "shape": 0.15, "phase_A": 0.20, "ea": 0.10}
            if exp == "2a3"
            else (
                {"q": 0.01, "r_ref": 0.03, "shape": 0.075, "phase_A": 0.10, "ea": 0.05}
                if exp == "2a4"
                else None
            )
        ),
        "wave": (
            "cc_rest"
            if exp in {"2b", "2g"}
            else ("cc_rest_pulse" if exp == "2c" else ("sequence_x3" if exp == "2d2" else "sequence"))
        ),
        "trips": trip_starts,
        "n_trips": len(trip_starts),
        "sequence": seq,
        "cells": cells,
        "noise_std": dict(NOISE_STD),
        "note": "共享 I_true / I_meas（零偏一个数）；电压按芯。估计器只看见 I_meas。",
    }
    if exp == "2g":
        # BMS 规格书分母错：CSV / 真值 Qi 仍 100 Ah；EKF/Ah 用 hatQ=95 Ah。
        meta["capacity_ah_true"] = 100.0
        meta["hat_q_ah"] = 95.0
        meta["capacity_scale"] = 0.95
        meta["note"] = (
            "共享 I_true / I_meas；电压按芯。2G：真值 Qi≡100 Ah，BMS hatQ=95 Ah "
            "(capacity_scale=0.95)；b_I=0，不开 --tol。"
        )
    (out_dir / "pack.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if write_csv_samples:
        from nmc100ah_gen import write_csv

        sample_ids = [0]
        for c in cells:
            if c["aged"]:
                sample_ids.append(c["id"])
                break
        for cid in dict.fromkeys(sample_ids):
            data = {
                "time_s": time_s,
                "step": np.arange(n_steps, dtype=float),
                "cmd_id": cmd_id,
                "mode": np.array(["pack"] * n_steps, dtype=object),
                "cutoff": cutoff[:, cid],
                "i_true_a": i_true,
                "i_meas_a": i_meas,
                "t_true_c": t_true[:, cid],
                "t_meas_c": t_meas[:, cid],
                "soc_true": soc_true[:, cid],
                "soc_meas": soc_true[:, cid],
                "u_ocv_v": u_ocv[:, cid],
                "r0_ohm": np.full(n_steps, np.nan),
                "r1_ohm": np.full(n_steps, np.nan),
                "c1_f": np.full(n_steps, np.nan),
                "tau1_s": np.full(n_steps, np.nan),
                "u_p_v": np.full(n_steps, np.nan),
                "u_t_true_v": u_t_true[:, cid],
                "u_t_meas_v": u_t_meas[:, cid],
            }
            write_csv(
                out_dir / f"cell_{cid:03d}.csv",
                data,
                dt_s=dt_s,
                soc0=cells[cid]["soc0"],
                noise_enable=True,
                noise_seed=seed,
                noise_std=dict(NOISE_STD),
                sequence=seq,
                extra_meta=[
                    f"# pack_exp={exp}",
                    f"# engine={engine}",
                    f"# cell={cid}",
                    f"# k={cells[cid]['k']}",
                    f"# b_I={b_i}",
                ],
                source="nmc100ah_gen_pack",
            )
    print(f"已写出 {out_dir / 'pack.npz'}", flush=True)
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="包级生成器：共享电流，按芯电压")
    p.add_argument(
        "--exp",
        default="2a1",
        choices=["2a1", "2a2", "2a3", "2a4", "2b", "2c", "2e", "2d1", "2d2", "2g", "2h1", "2h2", "2h3"],
    )
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--engine", default="pybamm", choices=["ecm", "pybamm"])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--k-aged", type=float, default=1.15)
    p.add_argument("--park-h", type=float, default=None, help="2H 停放小时；默认 48，烟测 6")
    p.add_argument("--park-dt", type=float, default=None, help="2H 采样秒；默认 5（门控仍按秒）")
    p.add_argument("--charge-min", type=float, default=None, help="2H3 充电分钟；默认 5")
    args = p.parse_args()
    if args.exp in {"2a3", "2a4"} and args.engine == "pybamm":
        print(f"{args.exp} 真值是每芯通道 / Q_i，改用 --engine ecm", flush=True)
        args.engine = "ecm"
    if args.exp == "2b" and args.engine == "pybamm":
        print("2b 零偏硬标准对齐 04-a F，改用 --engine ecm（不叠 SPM 墙）", flush=True)
        args.engine = "ecm"
    if args.exp == "2g" and args.engine == "pybamm":
        print("2g 全包 hatQ 错对齐 04-a / 2B 波型，改用 --engine ecm（不叠 SPM 墙）", flush=True)
        args.engine = "ecm"
    if args.exp == "2c" and args.engine == "pybamm":
        print("2c 先拦后写对齐 2B 零偏 + SEQUENCE 1C 边沿，改用 --engine ecm", flush=True)
        args.engine = "ecm"
    if args.exp == "2e" and args.engine == "pybamm":
        print("2e 电压形态对齐 1a 任务 A，改用 --engine ecm（不叠 SPM 墙）", flush=True)
        args.engine = "ecm"
    if args.exp in {"2d1", "2d2"} and args.engine == "pybamm":
        print(f"{args.exp} 滤波层对照对齐 04-a E，改用 --engine ecm（BOL 真值，表偏在估计器）", flush=True)
        args.engine = "ecm"
    if args.exp in {"2h1", "2h2", "2h3"} and args.engine == "pybamm":
        print(f"{args.exp} AFE 停放对齐 06-a §5.7，改用 --engine ecm（不叠 SPM / 2B）", flush=True)
        args.engine = "ecm"
    seed = args.seed
    if seed is None:
        seed = {
            "2a1": 201,
            "2a2": 202,
            "2a3": 203,
            "2a4": 204,
            "2b": 205,
            "2g": 209,
            "2c": 208,
            "2e": 206,
            "2d1": 207,
            "2d2": 207,
            "2h1": 210,
            "2h2": 211,
            "2h3": 212,
        }.get(args.exp, 201)
    out = args.out_dir or f"Data/pack/{args.exp}" + ("" if args.n >= 180 else f"_n{args.n}")
    generate_pack(
        exp=args.exp,
        n=args.n,
        engine=args.engine,
        out_dir=_resolve(out),
        seed=int(seed),
        k_aged=args.k_aged,
        park_h=args.park_h,
        park_dt_s=args.park_dt,
        charge_min=args.charge_min,
    )


if __name__ == "__main__":
    main()
