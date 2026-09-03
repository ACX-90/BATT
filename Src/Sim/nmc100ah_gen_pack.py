"""包级生成器：一条真电流 → N_s 只电压（Doc/06-a §2 / §7）。

默认 ECM；`--engine pybamm` 用 SPM 顶真包。2A3 / 2A4 真值是每芯通道，强制 ecm。不覆盖 Data/grid。

    python Src/Sim/nmc100ah_gen_pack.py --exp 2a1 --n 8 --engine pybamm --out-dir Data/pack/2a1_smoke
    python Src/Sim/nmc100ah_gen_pack.py --exp 2a1 --n 180 --engine ecm --out-dir Data/pack/2a1
    python Src/Sim/nmc100ah_gen_pack.py --exp 2a3 --n 8 --seed 203 --out-dir Data/pack/2a3_n8
    python Src/Sim/nmc100ah_gen_pack.py --exp 2a4 --n 8 --seed 204 --out-dir Data/pack/2a4_n8
    python Src/Sim/nmc100ah_gen_pack.py --exp 2b --n 8 --seed 205 --out-dir Data/pack/2b_n8
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
    bias = 5.0 if exp == "2b" else float(b_i)
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


def generate_pack(
    *,
    exp: str,
    n: int,
    engine: str,
    out_dir: Path,
    seed: int,
    k_aged: float = 1.15,
    write_csv_samples: bool = True,
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
    seq = list(SEQ_CC_REST) if exp == "2b" else list(SEQUENCE)
    dt_s = DT_S
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
        "wave": "cc_rest" if exp == "2b" else "sequence",
        "sequence": seq,
        "cells": cells,
        "noise_std": dict(NOISE_STD),
        "note": "共享 I_true / I_meas（零偏一个数）；电压按芯。估计器只看见 I_meas。",
    }
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
    p.add_argument("--exp", default="2a1", choices=["2a1", "2a2", "2a3", "2a4", "2b"])
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--engine", default="pybamm", choices=["ecm", "pybamm"])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--k-aged", type=float, default=1.15)
    args = p.parse_args()
    if args.exp in {"2a3", "2a4"} and args.engine == "pybamm":
        print(f"{args.exp} 真值是每芯通道 / Q_i，改用 --engine ecm", flush=True)
        args.engine = "ecm"
    if args.exp == "2b" and args.engine == "pybamm":
        print("2b 零偏硬标准对齐 04-a F，改用 --engine ecm（不叠 SPM 墙）", flush=True)
        args.engine = "ecm"
    seed = args.seed
    if seed is None:
        seed = {"2a1": 201, "2a2": 202, "2a3": 203, "2a4": 204, "2b": 205}.get(args.exp, 201)
    out = args.out_dir or f"Data/pack/{args.exp}" + ("" if args.n >= 180 else f"_n{args.n}")
    generate_pack(
        exp=args.exp,
        n=args.n,
        engine=args.engine,
        out_dir=_resolve(out),
        seed=int(seed),
        k_aged=args.k_aged,
    )


if __name__ == "__main__":
    main()
