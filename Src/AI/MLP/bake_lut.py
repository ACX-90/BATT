"""把已训 64×64 烤成查找表，按与蒸馏相同的网格口径报电压。

仓库根目录：

    python Src/AI/MLP/bake_lut.py --teacher-dir Data/ai_mlp --out-dir Data/ai_mlp_lut

不覆盖 Data/ai_mlp。表存 float32 的 R0/R1；车上再乘每芯 k 网格。
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch

MLP_DIR = Path(__file__).resolve().parent
AI_DIR = MLP_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from MLP.config import REPO_ROOT, TrainConfig
from MLP.dataset import load_grid_sequences, split_sequences
from MLP.infer import load_bundle

I_LO, I_HI = -200.0, 200.0
S_LO, S_HI = 0.05, 0.95
T_LO, T_HI = -10.0, 50.0
C1_STAR = 2.8e4
DT_S = 0.1


def ecm_np(
    current: np.ndarray,
    u_ocv: np.ndarray,
    r0: np.ndarray,
    r1: np.ndarray,
    c1: np.ndarray,
    dt_s: float = DT_S,
) -> np.ndarray:
    batch, n_step = current.shape
    u_p = np.zeros(batch, dtype=np.float64)
    u_t = np.empty((batch, n_step), dtype=np.float64)
    dt = float(dt_s)
    for k in range(n_step):
        tau = np.maximum(r1[:, k] * c1[:, k], 1.0e-6)
        alpha = np.exp(-dt / tau)
        u_p = alpha * u_p + r1[:, k] * (1.0 - alpha) * current[:, k]
        u_t[:, k] = u_ocv[:, k] - current[:, k] * r0[:, k] - u_p
    return u_t


@torch.no_grad()
def teacher_r(
    model,
    scaler,
    i_a: np.ndarray,
    soc: np.ndarray,
    t_c: np.ndarray,
    batch: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    feat = np.stack([np.asarray(i_a, dtype=float), np.asarray(soc, dtype=float), np.asarray(t_c, dtype=float)], axis=-1)
    n = feat.shape[0]
    r0 = np.empty(n, dtype=np.float64)
    r1 = np.empty(n, dtype=np.float64)
    for start in range(0, n, batch):
        sl = slice(start, min(start + batch, n))
        x = torch.from_numpy(scaler.transform(feat[sl]).astype(np.float32))
        a, b, _ = model(x)
        r0[sl] = a.cpu().numpy()
        r1[sl] = b.cpu().numpy()
    return r0, r1


def nlinear(axes: list[np.ndarray], table: np.ndarray, query: np.ndarray, log_r: bool = False) -> np.ndarray:
    """均匀轴上的 n 线性插值。table 末维是通道（R0,R1）。query (N, ndim)。"""
    vals = np.log(np.clip(table, 1e-12, None)) if log_r else table
    ndim = len(axes)
    idx0: list[np.ndarray] = []
    w1: list[np.ndarray] = []
    for d, ax in enumerate(axes):
        step = float(ax[1] - ax[0])
        t = (query[:, d] - float(ax[0])) / step
        t = np.clip(t, 0.0, float(len(ax) - 1) - 1e-6)
        i0 = np.floor(t).astype(np.int32)
        idx0.append(i0)
        w1.append((t - i0).astype(np.float64))
    out = np.zeros((query.shape[0], vals.shape[-1]), dtype=np.float64)
    for bits in product((0, 1), repeat=ndim):
        w = np.ones(query.shape[0], dtype=np.float64)
        sl: list[np.ndarray] = []
        for d, bit in enumerate(bits):
            sl.append(idx0[d] + bit)
            wd = w1[d] if bit else (1.0 - w1[d])
            w = w * wd
        out += w[:, None] * vals[tuple(sl)]
    if log_r:
        return np.exp(out)
    return out


def rom_bytes(n_node: int, n_ch: int = 2, dtype_b: int = 4) -> int:
    return int(n_node * n_ch * dtype_b)


def pack_seqs(seqs: list[dict]) -> dict[str, np.ndarray]:
    return {
        "i": np.stack([s["i"] for s in seqs]),
        "soc": np.stack([s["soc"] for s in seqs]),
        "t": np.stack([s["t"] for s in seqs]),
        "u_ocv": np.stack([s["u_ocv"] for s in seqs]),
        "u_t": np.stack([s["u_t"] for s in seqs]),
        "r0": np.stack([s["r0"] for s in seqs]),
        "r1": np.stack([s["r1"] for s in seqs]),
    }


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def i_bins(i_a: np.ndarray) -> dict[str, np.ndarray]:
    mag = np.abs(i_a)
    return {
        "rest_|I|<5": mag < 5.0,
        "0.5C_~50A": (mag >= 5.0) & (mag < 75.0),
        "1C_~100A": (mag >= 75.0) & (mag < 150.0),
        "2C_~200A": mag >= 150.0,
    }


def eval_r_u(
    r0: np.ndarray,
    r1: np.ndarray,
    pack: dict[str, np.ndarray],
    r0_t: np.ndarray,
    r1_t: np.ndarray,
    u_tchr: np.ndarray,
) -> dict[str, float]:
    c1 = np.full_like(r0, C1_STAR)
    u_hat = ecm_np(pack["i"], pack["u_ocv"], r0, r1, c1)
    out = {
        "rmse_v_meas": rmse(u_hat, pack["u_t"]),
        "rmse_v_teacher": rmse(u_hat, u_tchr),
        "r0_vs_teacher": rmse(r0, r0_t),
        "r1_vs_teacher": rmse(r1, r1_t),
        "r0_vs_csv": rmse(r0, pack["r0"]),
        "r1_vs_csv": rmse(r1, pack["r1"]),
    }
    flat_i = pack["i"].reshape(-1)
    flat_r1 = r1.reshape(-1)
    flat_r1t = r1_t.reshape(-1)
    for name, mask in i_bins(flat_i).items():
        if mask.any():
            out[f"r1_{name}"] = rmse(flat_r1[mask], flat_r1t[mask])
        else:
            out[f"r1_{name}"] = float("nan")
    return out


def bake_3d(model, scaler, n_i: int, n_s: int, n_t: int) -> tuple[list[np.ndarray], np.ndarray]:
    i_ax = np.linspace(I_LO, I_HI, n_i)
    s_ax = np.linspace(S_LO, S_HI, n_s)
    t_ax = np.linspace(T_LO, T_HI, n_t)
    gi, gs, gt = np.meshgrid(i_ax, s_ax, t_ax, indexing="ij")
    r0, r1 = teacher_r(model, scaler, gi.ravel(), gs.ravel(), gt.ravel())
    table = np.stack([r0, r1], axis=-1).reshape(n_i, n_s, n_t, 2)
    return [i_ax, s_ax, t_ax], table


def bake_2d(model, scaler, n_s: int, n_t: int, i_fixed: float) -> tuple[list[np.ndarray], np.ndarray]:
    s_ax = np.linspace(S_LO, S_HI, n_s)
    t_ax = np.linspace(T_LO, T_HI, n_t)
    gs, gt = np.meshgrid(s_ax, t_ax, indexing="ij")
    r0, r1 = teacher_r(model, scaler, np.full(gs.size, i_fixed), gs.ravel(), gt.ravel())
    table = np.stack([r0, r1], axis=-1).reshape(n_s, n_t, 2)
    return [s_ax, t_ax], table


def query_lut(kind: str, axes, table, i_a, soc, t_c, log_r: bool, extra=None) -> tuple[np.ndarray, np.ndarray]:
    qst = np.stack([soc.ravel(), t_c.ravel()], axis=-1)
    if kind == "3d":
        q = np.stack([i_a.ravel(), soc.ravel(), t_c.ravel()], axis=-1)
        rr = nlinear(axes, table, q, log_r=log_r)
    elif kind == "2d_fixed":
        rr = nlinear(axes, table, qst, log_r=log_r)
    elif kind == "2d_signed":
        pos = extra["pos"]
        neg = extra["neg"]
        rp = nlinear(axes, pos, qst, log_r=log_r)
        rn = nlinear(axes, neg, qst, log_r=log_r)
        # 两张表烤在 ±1C：只过渡充/放方向，|I| 仍钉在 1C。
        w = 0.5 * (np.clip(i_a.ravel(), -100.0, 100.0) / 100.0 + 1.0)
        rr = (1.0 - w)[:, None] * rn + w[:, None] * rp
    elif kind == "sep":
        r_st = nlinear(axes, table, qst, log_r=log_r)
        f = nlinear([extra["i_ax"]], extra["f"], i_a.ravel()[:, None], log_r=False)
        rr = r_st * f
    else:
        raise ValueError(kind)
    r0 = rr[:, 0].reshape(i_a.shape)
    r1 = rr[:, 1].reshape(i_a.shape)
    return r0, r1


def fmt_row(row: dict) -> str:
    return (
        f"{row['name']:<20} ROM {row['rom_kb']:.2f} KB  "
        f"val {row['val']['rmse_v_meas']*1e3:5.2f} mV  "
        f"grid {row['all']['rmse_v_meas']*1e3:5.2f} mV  "
        f"+vsT {row['all']['rmse_v_teacher']*1e3:5.2f} mV  "
        f"R0 {row['all']['r0_vs_teacher']*1e6:6.1f} uOhm  "
        f"R1 {row['all']['r1_vs_teacher']*1e6:6.1f} uOhm"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="64×64 → 查找表")
    p.add_argument("--teacher-dir", default="Data/ai_mlp")
    p.add_argument("--out-dir", default="Data/ai_mlp_lut")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    teacher_dir = Path(args.teacher_dir)
    if not teacher_dir.is_absolute():
        teacher_dir = REPO_ROOT / teacher_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model, scaler, t_cfg = load_bundle(
        teacher_dir / "best.pt", teacher_dir / "config.json", teacher_dir / "scaler.json"
    )
    model.eval()
    cfg = TrainConfig()
    cfg.scheme = "B"
    cfg.data_dir = t_cfg.data_dir
    cfg.use_true_inputs = t_cfg.use_true_inputs
    cfg.voltage_target = t_cfg.voltage_target

    sequences = load_grid_sequences(cfg)
    train_seq, val_seq = split_sequences(sequences, cfg.val_ratio, cfg.seed)
    packs = {
        "train": pack_seqs(train_seq),
        "val": pack_seqs(val_seq),
        "all": pack_seqs(sequences),
    }
    print(
        f"bake teacher={teacher_dir} n_seq={len(sequences)} "
        f"train={len(train_seq)} val={len(val_seq)} T={packs['all']['i'].shape[1]}"
    )
    print(
        f"data I=[{packs['all']['i'].min():.1f},{packs['all']['i'].max():.1f}] "
        f"SOC=[{packs['all']['soc'].min():.3f},{packs['all']['soc'].max():.3f}] "
        f"T=[{packs['all']['t'].min():.1f},{packs['all']['t'].max():.1f}]"
    )

    teacher_r_u: dict[str, dict[str, np.ndarray]] = {}
    for split, pack in packs.items():
        print(f"teacher forward {split} ...", flush=True)
        r0, r1 = teacher_r(model, scaler, pack["i"].ravel(), pack["soc"].ravel(), pack["t"].ravel())
        r0 = r0.reshape(pack["i"].shape)
        r1 = r1.reshape(pack["i"].shape)
        c1 = np.full_like(r0, C1_STAR)
        u_hat = ecm_np(pack["i"], pack["u_ocv"], r0, r1, c1)
        teacher_r_u[split] = {"r0": r0, "r1": r1, "u": u_hat}
        print(
            f"  teacher {split}: meas {rmse(u_hat, pack['u_t'])*1e3:.2f} mV  "
            f"R0vsCSV {rmse(r0, pack['r0'])*1e6:.1f} uOhm  "
            f"R1vsCSV {rmse(r1, pack['r1'])*1e6:.1f} uOhm",
            flush=True,
        )

    rng = np.random.default_rng(0)
    n_probe = 20000
    probe = np.stack(
        [
            rng.uniform(I_LO, I_HI, n_probe),
            rng.uniform(S_LO, S_HI, n_probe),
            rng.uniform(T_LO, T_HI, n_probe),
        ],
        axis=1,
    )
    pr0, pr1 = teacher_r(model, scaler, probe[:, 0], probe[:, 1], probe[:, 2])
    probe_t = np.stack([pr0, pr1], axis=-1)

    configs = [
        dict(name="2d_I+100_11x9", kind="2d_fixed", n_s=11, n_t=9, i_fixed=100.0),
        dict(name="2d_I+100_21x13", kind="2d_fixed", n_s=21, n_t=13, i_fixed=100.0),
        dict(name="2d_signed_11x9", kind="2d_signed", n_s=11, n_t=9),
        dict(name="2d11x9_x_i9", kind="sep", n_s=11, n_t=9, n_i=9),
        dict(name="2d21x13_x_i9", kind="sep", n_s=21, n_t=13, n_i=9),
        dict(name="3d_3x11x9", kind="3d", n_i=3, n_s=11, n_t=9),
        dict(name="3d_5x11x9", kind="3d", n_i=5, n_s=11, n_t=9),
        dict(name="3d_9x11x9", kind="3d", n_i=9, n_s=11, n_t=9),
        dict(name="3d_5x21x13", kind="3d", n_i=5, n_s=21, n_t=13),
        dict(name="3d_9x21x13", kind="3d", n_i=9, n_s=21, n_t=13),
        dict(name="3d_5x11x9_log", kind="3d", n_i=5, n_s=11, n_t=9, log_r=True),
        dict(name="3d_9x11x9_log", kind="3d", n_i=9, n_s=11, n_t=9, log_r=True),
    ]

    results: list[dict] = []
    save_tables: dict[str, dict] = {}

    for spec in configs:
        kind = spec["kind"]
        log_r = bool(spec.get("log_r", False))
        extra = None
        n_s = int(spec["n_s"])
        n_t = int(spec["n_t"])
        if kind == "3d":
            n_i = int(spec["n_i"])
            axes, table = bake_3d(model, scaler, n_i, n_s, n_t)
            n_node = n_i * n_s * n_t
        elif kind == "2d_fixed":
            axes, table = bake_2d(model, scaler, n_s, n_t, float(spec["i_fixed"]))
            n_node = n_s * n_t
        elif kind == "2d_signed":
            axes, pos = bake_2d(model, scaler, n_s, n_t, 100.0)
            _, neg = bake_2d(model, scaler, n_s, n_t, -100.0)
            table = pos
            extra = {"pos": pos, "neg": neg}
            n_node = 2 * n_s * n_t
        else:
            n_i = int(spec["n_i"])
            axes, table = bake_2d(model, scaler, n_s, n_t, 100.0)
            i_ax = np.linspace(I_LO, I_HI, n_i)
            r0i, r1i = teacher_r(model, scaler, i_ax, np.full(n_i, 0.50), np.full(n_i, 25.0))
            r0r, r1r = teacher_r(model, scaler, np.array([100.0]), np.array([0.50]), np.array([25.0]))
            f = np.stack([r0i / r0r[0], r1i / r1r[0]], axis=-1)
            extra = {"i_ax": i_ax, "f": f}
            n_node = n_s * n_t + n_i

        split_stats = {}
        for split, pack in packs.items():
            r0, r1 = query_lut(kind, axes, table, pack["i"], pack["soc"], pack["t"], log_r, extra)
            split_stats[split] = eval_r_u(
                r0, r1, pack, teacher_r_u[split]["r0"], teacher_r_u[split]["r1"], teacher_r_u[split]["u"]
            )
        pq = np.stack([probe[:, 0], probe[:, 1], probe[:, 2]], axis=0)
        r0p, r1p = query_lut(kind, axes, table, pq[0], pq[1], pq[2], log_r, extra)
        probe_err = {
            "r0": rmse(r0p, probe_t[:, 0]),
            "r1": rmse(r1p, probe_t[:, 1]),
        }
        row = {
            "name": spec["name"],
            "kind": kind,
            "n_i": spec.get("n_i"),
            "n_s": n_s,
            "n_t": n_t,
            "log_r": log_r,
            "n_node": n_node,
            "rom_B": rom_bytes(n_node),
            "rom_kb": rom_bytes(n_node) / 1024.0,
            "train": split_stats["train"],
            "val": split_stats["val"],
            "all": split_stats["all"],
            "probe": probe_err,
        }
        results.append(row)
        print(fmt_row(row), flush=True)
        if spec["name"] in {"3d_5x11x9", "3d_9x11x9", "3d_5x11x9_log", "3d_9x11x9_log"}:
            save_tables[spec["name"]] = {
                "axes": [a.tolist() for a in axes],
                "table": table.astype(np.float32),
                "log_r": log_r,
            }

    payload = {
        "teacher_dir": str(teacher_dir),
        "axes_box": {"I": [I_LO, I_HI], "SOC": [S_LO, S_HI], "T": [T_LO, T_HI]},
        "teacher": {
            split: {
                "rmse_v_meas": rmse(teacher_r_u[split]["u"], packs[split]["u_t"]),
                "r0_vs_csv": rmse(teacher_r_u[split]["r0"], packs[split]["r0"]),
                "r1_vs_csv": rmse(teacher_r_u[split]["r1"], packs[split]["r1"]),
            }
            for split in packs
        },
        "luts": results,
    }
    (out_dir / "eval.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for name, blob in save_tables.items():
        np.savez(
            out_dir / f"{name}.npz",
            i_ax=np.asarray(blob["axes"][0]),
            s_ax=np.asarray(blob["axes"][1]),
            t_ax=np.asarray(blob["axes"][2]),
            table=blob["table"],
            log_r=np.array(blob["log_r"]),
        )
    print(f"写出 {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
