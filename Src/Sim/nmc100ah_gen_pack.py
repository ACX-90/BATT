"""包级生成器：一条真电流 → N_s 只电压（Doc/06-a §2 / §7）。

默认 ECM；`--engine pybamm` 用 SPM 顶真包。不覆盖 Data/grid。

    python Src/Sim/nmc100ah_gen_pack.py --exp 2a1 --n 8 --engine pybamm --out-dir Data/pack/2a1_smoke
    python Src/Sim/nmc100ah_gen_pack.py --exp 2a1 --n 180 --engine ecm --out-dir Data/pack/2a1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parent
REPO_ROOT = SIM_DIR.parent.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from nmc100ah_ecm import make_ecm  # noqa: E402
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
        model = make_ecm(r0_scale=cell["k"], r1_scale=cell["k"], c1_scale=1.0)
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
    seq = list(SEQUENCE)
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
        print(
            f"  cell {i:03d}/{n}  aged={int(cell['aged'])} k={cell['k']:.3f}  "
            f"soc0={cell['soc0']:.3f} T={cell['t_c']:+.1f}",
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
    if args.exp in {"2a3", "2a4"}:
        p.error(f"{args.exp} 通道系数 / Q_i 抽签尚未接进本入口（06-a §5.1）；先跑 2a1")
    if args.exp == "2b":
        p.error("2b 要 cc_rest 小时级，尚未接；先跑 2a1")
    seed = args.seed
    if seed is None:
        seed = {"2a1": 201, "2a2": 202}.get(args.exp, 201)
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
