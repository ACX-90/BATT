"""包级滑窗 k 网格 + 包级门（Doc/06-a §7）。

    python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2a1_n8 --mode freeze
    python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2a1_n8 --mode kgrid --out-dir Data/pack/2a1_n8_kgrid

不覆盖 Data/grid / Data/ai_mlp / Data/ai_local。更新路径不读旧网格。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

KF_DIR = Path(__file__).resolve().parent
AI_DIR = KF_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from KF.adapter import KGridAdapter, MlpParamProvider  # noqa: E402
from KF.config import REPO_ROOT  # noqa: E402
from KF.filter import filter_metrics, run_filter  # noqa: E402
from KF.ocv import docv_ds, ocv_nmc  # noqa: E402
from KF.pack_gate import last_edge_age_s, pack_gate, window_policy  # noqa: E402
from MLP.infer import load_bundle  # noqa: E402
from MLP.train import set_seed  # noqa: E402
from window import window_gate  # noqa: E402
from kgrid import _roll_up, _seq_tensors, overall_rmse, step_window  # noqa: E402


FORBIDDEN = {
    "Data/grid",
    "Data/grid_pybamm",
    "Data/ai_mlp",
    "Data/ai_local",
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _guard_out(path: Path) -> None:
    rel = _rel(path)
    for bad in FORBIDDEN:
        if rel == bad or rel.startswith(bad + "/"):
            raise RuntimeError(f"禁止写到 {rel}")


def load_pack(pack_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    meta = json.loads((pack_dir / "pack.json").read_text(encoding="utf-8"))
    blob = np.load(pack_dir / "pack.npz")
    data = {k: blob[k] for k in blob.files}
    if float(meta.get("b_I", 0.0)) != 0.0 and meta.get("exp") == "2a1":
        raise RuntimeError("2A1 必须 b_I=0")
    return meta, data


def pack_ocv_fn(data: dict[str, np.ndarray]):
    """用生成器写出的 bulk OCV 当本包 OCV 表（真包会先标定，不要混仓库 ocv.py）。"""
    s = np.asarray(data["soc_true"], dtype=float).ravel()
    v = np.asarray(data["u_ocv"], dtype=float).ravel()
    order = np.argsort(s)
    s, v = s[order], v[order]
    _, idx = np.unique(np.round(s, 6), return_index=True)
    s_u, v_u = s[idx], v[idx]
    if s_u.size < 4:
        return ocv_nmc, docv_ds
    ds = np.diff(s_u)
    dv = np.diff(v_u)
    slopes = np.divide(dv, np.maximum(ds, 1e-8))

    def ocv(soc, t_c=25.0):
        del t_c
        x = np.clip(np.asarray(soc, dtype=float), float(s_u[0]), float(s_u[-1]))
        val = np.interp(x, s_u, v_u)
        return float(val) if np.ndim(soc) == 0 else val

    def docv(soc, t_c=25.0):
        del t_c
        x = np.clip(np.asarray(soc, dtype=float), float(s_u[0]), float(s_u[-1]))
        j = np.clip(np.searchsorted(s_u, x, side="right") - 1, 0, len(slopes) - 1)
        val = slopes[j]
        return float(val) if np.ndim(soc) == 0 else val

    return ocv, docv


def cell_seq(
    data: dict[str, np.ndarray],
    i: int,
    *,
    soc: np.ndarray,
    name: str,
    ocv=None,
) -> dict:
    t_c = data["t_meas"][:, i]
    ocv_fn = ocv_nmc if ocv is None else ocv
    return {
        "name": name,
        "i": np.asarray(data["i_meas"], dtype=float),
        "soc": np.asarray(soc, dtype=float),
        "t": np.asarray(t_c, dtype=float),
        "u_ocv": np.asarray(ocv_fn(soc, t_c), dtype=float),
        "u_t": np.asarray(data["u_t_meas"][:, i], dtype=float),
        "u_p0": 0.0,
    }


def run_cell_filter(
    provider: MlpParamProvider,
    data: dict,
    i: int,
    soc0: float,
    *,
    ocv=None,
    docv=None,
) -> dict:
    log = run_filter(
        provider,
        np.asarray(data["i_meas"], dtype=float),
        np.asarray(data["t_meas"][:, i], dtype=float),
        np.asarray(data["u_t_meas"][:, i], dtype=float),
        s0=float(soc0),
        soc_true=np.asarray(data["soc_true"][:, i], dtype=float),
        ocv=ocv,
        docv=docv,
    )
    return log


def _kgrid_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        i_edge=args.i_edge,
        rest_eps=args.rest_eps,
        rest_s=args.rest_s,
        e_ol_min=args.e_ol_min,
        smooth=args.smooth,
        k_lo=args.k_lo,
        k_hi=args.k_hi,
        grad_clip=args.grad_clip,
        win=args.win,
    )


def run_cell_kgrid(
    model: KGridAdapter,
    seq: dict,
    scaler,
    cfg,
    device,
    args,
    *,
    pack_blocked: bool,
    nogate: bool,
) -> dict:
    win = int(args.win)
    dt_s = cfg.dt_s
    t = _seq_tensors(seq, scaler, device)
    n = int(t["i"].shape[0])
    u_p0 = t["i"].new_tensor([0.0])
    opt = torch.optim.SGD([model.log_k0, model.log_k1], lr=args.lr)
    ns, nt = model.log_k0.shape
    hit0 = np.zeros((ns, nt), dtype=float)
    hit1 = np.zeros((ns, nt), dtype=float)
    n_win = n_upd = n_skip_gate = n_skip_pack = n_skip_park = 0
    kg_args = _kgrid_args(args)
    i_np_all = seq["i"]
    for start in range(0, n, win):
        end = min(start + win, n)
        if end - start < max(win // 4, 20):
            break
        n_win += 1
        sl = {k: v[start:end] for k, v in t.items()}
        i_prev = float(i_np_all[start - 1]) if start > 0 else None
        i_np = i_np_all[start:end]
        gated, gstat = window_gate(
            i_np,
            dt_s=dt_s,
            i_edge_a=args.i_edge,
            rest_eps=args.rest_eps,
            rest_s=args.rest_s,
            i_prev=i_prev,
        )
        age = last_edge_age_s(
            i_np_all, start, dt_s=dt_s, i_edge_a=args.i_edge, i_prev=None
        )
        pol = window_policy(has_edge=bool(gstat["has_edge"]), last_edge_age_s=age)
        if pack_blocked and not nogate:
            n_skip_pack += 1
            u_p0 = _roll_up(model, sl, u_p0, dt_s, len(i_np) // 2)
            continue
        if not pol["write_k"]:
            n_skip_park += 1
            u_p0 = _roll_up(model, sl, u_p0, dt_s, len(i_np) // 2)
            continue
        if not pol["allow_k1"]:
            kg_args.rest_s = 1e9
        else:
            kg_args.rest_s = args.rest_s
        st = step_window(
            model,
            opt,
            sl,
            u_p0,
            dt_s=dt_s,
            i_np=i_np,
            args=kg_args,
            i_prev=i_prev,
            hit0=hit0,
            hit1=hit1,
        )
        u_p0 = st.pop("u_p0")
        if st["updated"]:
            n_upd += 1
        elif not gated:
            n_skip_gate += 1
    k_ref = model.k_at(0.50, 25.0)
    k_mean = model.k_at(float(seq["soc"].mean()), float(seq["t"].mean()))
    return {
        "n_win": n_win,
        "n_update": n_upd,
        "n_skip_gate": n_skip_gate,
        "n_skip_pack": n_skip_pack,
        "n_skip_park": n_skip_park,
        "k_at_ref": list(k_ref),
        "k_at_mean": list(k_mean),
        "k_tables": model.k_tables(),
        "hit0": hit0.tolist(),
        "hit1": hit1.tolist(),
    }


def summarize_cells(cells: list[dict], rows: list[dict]) -> dict:
    aged = [r for r, c in zip(rows, cells) if c["aged"]]
    nom = [r for r, c in zip(rows, cells) if not c["aged"]]

    def _mean_k(group, idx):
        if not group:
            return float("nan")
        return float(np.mean([g["k_at_mean"][idx] for g in group]))

    return {
        "n_aged": len(aged),
        "n_nom": len(nom),
        "k0_aged": _mean_k(aged, 0),
        "k1_aged": _mean_k(aged, 1),
        "k0_nom": _mean_k(nom, 0),
        "k1_nom": _mean_k(nom, 1),
        "rmse_ol_aged_mV": float(np.mean([g["e_ol_rmse_mV"] for g in aged])) if aged else float("nan"),
        "rmse_ol_nom_mV": float(np.mean([g["e_ol_rmse_mV"] for g in nom])) if nom else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="包级 k 网格 + 包级门")
    p.add_argument("--pack-dir", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--mlp-dir", default="Data/ai_mlp")
    p.add_argument("--mode", default="kgrid", choices=["freeze", "kgrid", "kgrid-nogate"])
    p.add_argument("--win", type=int, default=100)
    p.add_argument("--lr", type=float, default=10.0)
    p.add_argument("--smooth", type=float, default=1.0e-3)
    p.add_argument("--i-edge", type=float, default=20.0)
    p.add_argument("--rest-s", type=float, default=3.0)
    p.add_argument("--rest-eps", type=float, default=1.0)
    p.add_argument("--e-ol-min", type=float, default=0.002)
    p.add_argument("--k-lo", type=float, default=0.5)
    p.add_argument("--k-hi", type=float, default=2.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)

    pack_dir = _resolve(args.pack_dir)
    out_dir = _resolve(args.out_dir or (str(pack_dir) + "_" + args.mode))
    _guard_out(out_dir)
    mlp_dir = _resolve(args.mlp_dir)
    meta, data = load_pack(pack_dir)
    cells = meta["cells"]
    n = int(meta["n"])
    print(
        f"pack {meta['exp']} n={n} engine={meta.get('engine')} b_I={meta.get('b_I')}  "
        f"mode={args.mode} mlp={mlp_dir}",
        flush=True,
    )

    device = torch.device(args.device)
    ckpt = mlp_dir / "best.pt"
    base, scaler, cfg = load_bundle(ckpt, mlp_dir / "config.json", mlp_dir / "scaler.json")
    for par in base.parameters():
        par.requires_grad_(False)
    base = base.to(device).eval()
    provider = MlpParamProvider(base, scaler, device=device)

    ocv_fn, docv_fn = ocv_nmc, docv_ds
    if str(meta.get("engine", "ecm")).lower() == "pybamm":
        ocv_fn, docv_fn = pack_ocv_fn(data)
        print("OCV: PyBaMM bulk（本包标定，不混仓库 ocv.py）", flush=True)

    # 1) 冻结 EKF：Δs 给包级门，s_ah 给增量
    logs = []
    ds_last = np.empty(n)
    ds_traj = []
    for i, cell in enumerate(cells):
        log = run_cell_filter(
            provider, data, i, cell["soc0"], ocv=ocv_fn, docv=docv_fn
        )
        logs.append(log)
        ds = log["soc_post"] - log["soc_ah"]
        ds_traj.append(ds)
        ds_last[i] = float(ds[-1])
        m = filter_metrics(log)
        print(
            f"  filt {i:03d} aged={int(cell['aged'])}  e_ol={m['e_ol_rmse_mV']:.2f} mV  "
            f"Δs={ds_last[i]*1e2:+.3f} pp  s_ah={m['s_end_ah']:.4f}",
            flush=True,
        )
    ds_mat = np.stack(ds_traj, axis=1)
    gate = pack_gate(ds_mat, dt_s=float(meta["dt_s"]))
    print(
        f"pack_gate blocked={gate['blocked']} reason={gate['reason']}  "
        f"m={gate['m']*1e2:.3f} pp  f_same={gate['f_same']:.2f}  "
        f"slope={gate.get('slope_pph', float('nan')):.3f} pp/h",
        flush=True,
    )

    rows = []
    nogate = args.mode == "kgrid-nogate"
    do_k = args.mode in {"kgrid", "kgrid-nogate"}
    frozen_model = KGridAdapter(base).to(device)
    for i, cell in enumerate(cells):
        seq_true = cell_seq(
            data, i, soc=data["soc_true"][:, i], name=f"c{i:03d}_true", ocv=ocv_fn
        )
        seq_ah = cell_seq(
            data, i, soc=logs[i]["soc_ah"], name=f"c{i:03d}_ah", ocv=ocv_fn
        )
        rmse0 = overall_rmse(frozen_model, [seq_true], scaler, cfg, device)
        fm = filter_metrics(logs[i])
        row = {
            "id": i,
            "aged": cell["aged"],
            "k_true": cell["k"],
            "soc0": cell["soc0"],
            "t_c": cell["t_c"],
            "e_ol_rmse_mV": fm["e_ol_rmse_mV"],
            "e_pri_rmse_mV": fm["e_pri_rmse_mV"],
            "ds_end": float(ds_last[i]),
            "rmse_frozen_true_mV": rmse0 * 1e3,
            "k_at_ref": [1.0, 1.0],
            "k_at_mean": [1.0, 1.0],
        }
        if do_k:
            model = KGridAdapter(base).to(device)
            kg = run_cell_kgrid(
                model,
                seq_ah,
                scaler,
                cfg,
                device,
                args,
                pack_blocked=bool(gate["blocked"]),
                nogate=nogate,
            )
            row.update(kg)
            rmse1 = overall_rmse(model, [seq_true], scaler, cfg, device)
            row["rmse_after_true_mV"] = rmse1 * 1e3
            print(
                f"  kgrid {i:03d} aged={int(cell['aged'])}  "
                f"k_ref=({kg['k_at_ref'][0]:.3f},{kg['k_at_ref'][1]:.3f})  "
                f"upd={kg['n_update']}/{kg['n_win']}  "
                f"rmse {rmse0*1e3:.2f}→{rmse1*1e3:.2f} mV",
                flush=True,
            )
        rows.append(row)

    summary = summarize_cells(cells, rows) if do_k else {}
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": args.mode,
        "pack_dir": _rel(pack_dir),
        "mlp_dir": _rel(mlp_dir),
        "exp": meta["exp"],
        "engine": meta.get("engine"),
        "n": n,
        "b_I": meta.get("b_I"),
        "optimizer": "SGD",
        "replay": False,
        "read_old_grid": False,
        "pack_gate": gate,
        "cells": [
            {k: v for k, v in r.items() if k not in {"k_tables", "hit0", "hit1"}}
            for r in rows
        ],
        "summary": summary,
        "win": args.win,
        "lr": args.lr,
    }
    (out_dir / "pack_run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if do_k:
        torch.save(
            {f"cell_{r['id']}": {"k_at_ref": r["k_at_ref"], "k_tables": r.get("k_tables")} for r in rows},
            out_dir / "last.pt",
        )
        print(
            f"summary  k0 aged/nom={summary['k0_aged']:.3f}/{summary['k0_nom']:.3f}  "
            f"k1 {summary['k1_aged']:.3f}/{summary['k1_nom']:.3f}",
            flush=True,
        )
    print(f"写出 {out_dir / 'pack_run.json'}", flush=True)

    # 2A1 烟测断言（数字作废，只拦脚本写错）
    if meta["exp"] == "2a1" and do_k and n <= 8:
        if gate["blocked"]:
            print("WARN 2A1 包级门不应触发")
        if summary["k0_nom"] > 1.08:
            print(f"WARN 未涨芯 k0={summary['k0_nom']:.3f} 偏高（期望 ~1）")
        if summary["k0_aged"] < summary["k0_nom"] + 0.015:
            print(
                f"WARN 涨阻芯 k0={summary['k0_aged']:.3f} 相对未涨 "
                f"{summary['k0_nom']:.3f} 没朝 1.15 走（看的是点到的节点，不是 0.5/25 参考点）"
            )


if __name__ == "__main__":
    main()
