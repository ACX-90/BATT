"""车上滑窗：冻 MLP，只动全局 k0/k1。无 Replay，SGD，不存旧波形。

仓库根目录：

    python Src/AI/EV_Local/window.py --mlp-dir Data/ai_mlp --new-dir Data/soh_k115 --old-dir Data/grid --out-dir Data/ai_local/window_k115

不覆盖 Data/ai_mlp / Data/grid。更新路径不读旧网格。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

KF_DIR = Path(__file__).resolve().parent
AI_DIR = KF_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from KF.adapter import ScaleAdapter
from KF.config import REPO_ROOT
from KF.gate import longest_rest_s
from KF.increment import load_incr_sequences, ref_params
from MLP.dataset import FeatureScaler
from MLP.ecm import ecm_forward
from MLP.infer import load_bundle
from MLP.train import set_seed


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def window_gate(
    i_a: np.ndarray,
    *,
    dt_s: float,
    i_edge_a: float,
    rest_eps: float,
    rest_s: float,
    i_prev: float | None = None,
) -> tuple[bool, dict[str, float]]:
    # 拼上窗前一拍，避免边沿落在窗缝被 diff 漏掉
    if i_prev is None:
        series = i_a
    else:
        series = np.concatenate(([float(i_prev)], i_a))
    di = np.abs(np.diff(series)) if len(series) > 1 else np.array([0.0])
    has_edge = bool(np.any(di >= i_edge_a))
    rest_len = longest_rest_s(i_a, dt_s, rest_eps)
    ok = bool(has_edge or rest_len >= rest_s)
    return ok, {"has_edge": float(has_edge), "rest_s": float(rest_len)}


def _seq_tensors(seq: dict, scaler: FeatureScaler, device: torch.device) -> dict[str, torch.Tensor]:
    feat = np.stack([seq["i"], seq["soc"], seq["t"]], axis=-1)
    xn = scaler.transform(feat).astype(np.float32)
    return {
        "x": torch.from_numpy(xn).to(device),
        "i": torch.from_numpy(seq["i"].astype(np.float32)).to(device),
        "u_ocv": torch.from_numpy(seq["u_ocv"].astype(np.float32)).to(device),
        "u_t": torch.from_numpy(seq["u_t"].astype(np.float32)).to(device),
    }


@torch.no_grad()
def overall_rmse(model, seqs: list[dict], scaler: FeatureScaler, cfg, device: torch.device) -> float:
    sse = 0.0
    n = 0
    model.eval()
    for seq in seqs:
        t = _seq_tensors(seq, scaler, device)
        r0, r1, c1 = model(t["x"].unsqueeze(0))
        u_hat, _ = ecm_forward(
            t["i"].unsqueeze(0),
            t["u_ocv"].unsqueeze(0),
            r0,
            r1,
            c1,
            dt_s=cfg.dt_s,
            u_p0=t["i"].new_tensor([float(seq.get("u_p0", 0.0))]),
        )
        err = u_hat.squeeze(0) - t["u_t"]
        sse += float(err.pow(2).sum().cpu())
        n += int(err.numel())
    return (sse / max(n, 1)) ** 0.5


def _roll_up(
    model: ScaleAdapter,
    sl: dict[str, torch.Tensor],
    u_p0: torch.Tensor,
    dt_s: float,
    center: int,
) -> torch.Tensor:
    """窗内电阻钉在中心点，只为把极化状态推到窗末。"""
    with torch.no_grad():
        r0, r1, c1 = model(sl["x"][center : center + 1].unsqueeze(0))
        n = sl["i"].shape[0]
        r0 = r0.expand(1, n)
        r1 = r1.expand(1, n)
        c1 = c1.expand(1, n)
        _, u_p = ecm_forward(
            sl["i"].unsqueeze(0),
            sl["u_ocv"].unsqueeze(0),
            r0,
            r1,
            c1,
            dt_s=dt_s,
            u_p0=u_p0,
        )
    return u_p[:, -1].detach()


def step_window(
    model: ScaleAdapter,
    opt: torch.optim.Optimizer,
    sl: dict[str, torch.Tensor],
    u_p0: torch.Tensor,
    *,
    dt_s: float,
    i_np: np.ndarray,
    i_edge_a: float,
    rest_eps: float,
    rest_s: float,
    e_ol_min: float,
    k_lo: float,
    k_hi: float,
    grad_clip: float,
    i_prev: float | None = None,
) -> dict:
    n = int(sl["i"].shape[0])
    mid = n // 2
    gated, gstat = window_gate(
        i_np,
        dt_s=dt_s,
        i_edge_a=i_edge_a,
        rest_eps=rest_eps,
        rest_s=rest_s,
        i_prev=i_prev,
    )
    stats = {
        "n": n,
        "gated": gated,
        "updated": False,
        "rmse": float("nan"),
        "g0": 0.0,
        "g1": 0.0,
        "i_rms": float(np.sqrt(np.mean(np.square(i_np)))),
        "k0": model.k0,
        "k1": model.k1,
        **gstat,
    }
    if not gated:
        stats["u_p0"] = _roll_up(model, sl, u_p0, dt_s, mid)
        return stats

    opt.zero_grad(set_to_none=True)
    r0c, r1c, c1c = model(sl["x"][mid : mid + 1].unsqueeze(0))
    r0 = r0c.expand(1, n)
    r1 = r1c.expand(1, n)
    c1 = c1c.expand(1, n)
    u_hat, u_p = ecm_forward(
        sl["i"].unsqueeze(0),
        sl["u_ocv"].unsqueeze(0),
        r0,
        r1,
        c1,
        dt_s=dt_s,
        u_p0=u_p0,
    )
    err = u_hat.squeeze(0) - sl["u_t"]
    rmse = float(err.pow(2).mean().sqrt().detach().cpu())
    stats["rmse"] = rmse
    if rmse < e_ol_min:
        stats["u_p0"] = u_p[:, -1].detach()
        return stats

    loss = 0.5 * err.pow(2).mean()
    loss.backward()
    g0 = float(model.log_k0.grad.detach()) if model.log_k0.grad is not None else 0.0
    g1 = float(model.log_k1.grad.detach()) if model.log_k1.grad is not None else 0.0
    # 大电流窗只动 k0，回弹静置只动 k1，避免恒流把两通道拆乱
    i_rms = float(np.sqrt(np.mean(np.square(i_np))))
    if i_rms < 20.0 and model.log_k0.grad is not None:
        model.log_k0.grad.zero_()
        g0 = 0.0
    if gstat["rest_s"] < rest_s and model.log_k1.grad is not None:
        model.log_k1.grad.zero_()
        g1 = 0.0
    stats["g0"] = g0
    stats["g1"] = g1
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_([model.log_k0, model.log_k1], grad_clip)
    opt.step()
    with torch.no_grad():
        model.log_k0.clamp_(float(np.log(k_lo)), float(np.log(k_hi)))
        model.log_k1.clamp_(float(np.log(k_lo)), float(np.log(k_hi)))
    stats["updated"] = True
    stats["k0"] = model.k0
    stats["k1"] = model.k1
    stats["u_p0"] = u_p[:, -1].detach()
    return stats


def run_pass(
    model: ScaleAdapter,
    opt: torch.optim.Optimizer,
    seqs: list[dict],
    scaler: FeatureScaler,
    cfg,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    n_win = n_upd = n_skip_gate = n_skip_small = 0
    n_hi_i = 0
    g0s: list[float] = []
    g1s: list[float] = []
    hist: list[dict] = []
    win = int(args.win)
    for si, seq in enumerate(seqs):
        t = _seq_tensors(seq, scaler, device)
        n = int(t["i"].shape[0])
        u_p0 = t["i"].new_tensor([float(seq.get("u_p0", 0.0))])
        for start in range(0, n, win):
            end = min(start + win, n)
            if end - start < max(win // 4, 20):
                break
            sl = {k: v[start:end] for k, v in t.items()}
            i_prev = float(seq["i"][start - 1]) if start > 0 else None
            st = step_window(
                model,
                opt,
                sl,
                u_p0,
                dt_s=cfg.dt_s,
                i_np=seq["i"][start:end],
                i_edge_a=args.i_edge,
                rest_eps=args.rest_eps,
                rest_s=args.rest_s,
                e_ol_min=args.e_ol_min,
                k_lo=args.k_lo,
                k_hi=args.k_hi,
                grad_clip=args.grad_clip,
                i_prev=i_prev,
            )
            u_p0 = st.pop("u_p0")
            n_win += 1
            if st["updated"]:
                n_upd += 1
                g0s.append(float(st["g0"]))
                g1s.append(float(st["g1"]))
                if float(st["i_rms"]) >= 20.0:
                    n_hi_i += 1
            elif not st["gated"]:
                n_skip_gate += 1
            else:
                n_skip_small += 1
        hist.append({"seq": seq["name"], "k0": model.k0, "k1": model.k1, "idx": si})
        print(
            f"  seq {si+1:02d}/{len(seqs)} {seq['name']}  k0={model.k0:.4f} k1={model.k1:.4f}",
            flush=True,
        )
    return {
        "n_win": n_win,
        "n_update": n_upd,
        "n_skip_gate": n_skip_gate,
        "n_skip_small": n_skip_small,
        "n_update_hi_i": n_hi_i,
        "g0_mean": float(np.mean(g0s)) if g0s else 0.0,
        "g1_mean": float(np.mean(g1s)) if g1s else 0.0,
        "k0": model.k0,
        "k1": model.k1,
        "per_seq": hist,
    }


def cc_probe(
    base,
    scaler: FeatureScaler,
    seqs: list[dict],
    cfg,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    """恒流段应被门控挡住：k 停在 1。"""
    model = ScaleAdapter(base).to(device)
    opt = torch.optim.SGD([model.log_k0, model.log_k1], lr=args.lr)
    n_win = n_upd = 0
    for seq in seqs[: min(5, len(seqs))]:
        i_a = seq["i"]
        mag = np.abs(i_a)
        # 1C 附近、边沿之外的恒流
        mask = (mag > 80.0) & (mag < 120.0)
        if mask.sum() < args.win:
            continue
        idx = np.flatnonzero(mask)
        # 取最长连续段
        cuts = np.where(np.diff(idx) > 1)[0]
        if len(cuts) == 0:
            block = idx
        else:
            spans = np.split(idx, cuts + 1)
            block = max(spans, key=len)
        t = _seq_tensors(seq, scaler, device)
        u_p0 = t["i"].new_tensor([0.0])
        sl_all = {k: v[int(block[0]) : int(block[-1]) + 1] for k, v in t.items()}
        i_np = i_a[int(block[0]) : int(block[-1]) + 1]
        n = len(i_np)
        for start in range(0, n, args.win):
            end = min(start + args.win, n)
            if end - start < max(args.win // 4, 20):
                break
            sl = {k: v[start:end] for k, v in sl_all.items()}
            i_prev = float(i_np[start - 1]) if start > 0 else None
            st = step_window(
                model,
                opt,
                sl,
                u_p0,
                dt_s=cfg.dt_s,
                i_np=i_np[start:end],
                i_edge_a=args.i_edge,
                rest_eps=args.rest_eps,
                rest_s=args.rest_s,
                e_ol_min=args.e_ol_min,
                k_lo=args.k_lo,
                k_hi=args.k_hi,
                grad_clip=args.grad_clip,
                i_prev=i_prev,
            )
            u_p0 = st.pop("u_p0")
            n_win += 1
            n_upd += int(st["updated"])
    return {"n_win": n_win, "n_update": n_upd, "k0": model.k0, "k1": model.k1}


def main() -> None:
    p = argparse.ArgumentParser(description="车上滑窗全局 k0/k1")
    p.add_argument("--mlp-dir", default="Data/ai_mlp")
    p.add_argument("--new-dir", default="Data/soh_k115")
    p.add_argument("--old-dir", default="Data/grid")
    p.add_argument("--out-dir", default="Data/ai_local/window_k115")
    p.add_argument("--win", type=int, default=100, help="窗长，步；100=10 s")
    p.add_argument("--lr", type=float, default=10.0)
    p.add_argument("--passes", type=int, default=1, help="新年份扫几遍；车上默认 1")
    p.add_argument("--i-edge", type=float, default=20.0)
    p.add_argument("--rest-s", type=float, default=3.0, help="窗内静置阈值 / s（整段门控 30 s 放不进 10 s 窗）")
    p.add_argument("--rest-eps", type=float, default=1.0)
    p.add_argument("--e-ol-min", type=float, default=0.002, help="窗 RMSE 低于此不更新 / V")
    p.add_argument("--k-lo", type=float, default=0.5)
    p.add_argument("--k-hi", type=float, default=2.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--use-true-inputs", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--best", action="store_true", default=True)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--fig-prefix", default=None, help="默认 Fig/local/<out-dir 名>")
    args = p.parse_args()
    set_seed(args.seed)

    mlp_dir = _resolve(args.mlp_dir)
    new_dir = _resolve(args.new_dir)
    old_dir = _resolve(args.old_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if new_dir.resolve() == old_dir.resolve():
        raise RuntimeError("新年份不能和旧网格是同一目录")

    device = torch.device(args.device)
    ckpt = mlp_dir / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    base, scaler, cfg = load_bundle(ckpt, mlp_dir / "config.json", mlp_dir / "scaler.json")
    for par in base.parameters():
        par.requires_grad_(False)
    base = base.to(device).eval()

    new_seq = load_incr_sequences(
        new_dir, pattern=None, use_true_inputs=args.use_true_inputs, weight=1.0, style="grid"
    )
    old_true = bool(cfg.use_true_inputs)
    old_seq = load_incr_sequences(
        old_dir, pattern=None, use_true_inputs=old_true, weight=1.0, style="grid"
    )
    print(
        f"window mlp={mlp_dir} new={len(new_seq)} old_eval={len(old_seq)}  "
        f"win={args.win} ({args.win * cfg.dt_s:.1f}s) lr={args.lr} passes={args.passes}  "
        f"new_in={'true' if args.use_true_inputs else 'meas'}  "
        f"old_in={'true' if old_true else 'meas'}  Replay=OFF"
    )

    frozen = ScaleAdapter(base).to(device)
    r0_b, r1_b = ref_params(frozen, scaler)
    new0 = overall_rmse(frozen, new_seq, scaler, cfg, device)
    old0 = overall_rmse(frozen, old_seq, scaler, cfg, device)
    print(f"冻结  新 {new0*1e3:.2f} mV  旧 {old0*1e3:.2f} mV  R0={r0_b*1e3:.3f} mΩ  R1={r1_b*1e3:.3f} mΩ")

    cc = cc_probe(base, scaler, new_seq, cfg, device, args)
    print(
        f"恒流探针  win={cc['n_win']} update={cc['n_update']}  "
        f"k0={cc['k0']:.4f} k1={cc['k1']:.4f}  （应 ≈1、更新数 ≈0）"
    )

    model = ScaleAdapter(base).to(device)
    opt = torch.optim.SGD([model.log_k0, model.log_k1], lr=args.lr)
    pass_rows: list[dict] = []
    for p_i in range(1, args.passes + 1):
        print(f"---- pass {p_i}/{args.passes} ----", flush=True)
        row = run_pass(model, opt, new_seq, scaler, cfg, device, args)
        new_a = overall_rmse(model, new_seq, scaler, cfg, device)
        old_a = overall_rmse(model, old_seq, scaler, cfg, device)
        row["pass"] = p_i
        row["new_rmse"] = new_a
        row["old_rmse"] = old_a
        pass_rows.append(row)
        print(
            f"  pass {p_i}: upd {row['n_update']}/{row['n_win']}  "
            f"skip_gate {row['n_skip_gate']} skip_small {row['n_skip_small']}  "
            f"k0={model.k0:.4f} k1={model.k1:.4f}  "
            f"新 {new_a*1e3:.2f} mV  旧 {old_a*1e3:.2f} mV",
            flush=True,
        )

    r0_a, r1_a = ref_params(model, scaler)
    last = pass_rows[-1]
    meta = {
        "mode": "window_k",
        "mlp_dir": str(mlp_dir),
        "new_dir": str(new_dir),
        "old_dir": str(old_dir),
        "win": args.win,
        "dt_s": cfg.dt_s,
        "lr": args.lr,
        "n_passes": args.passes,
        "optimizer": "SGD",
        "replay": False,
        "center_R": True,
        "i_edge_a": args.i_edge,
        "rest_s": args.rest_s,
        "e_ol_min": args.e_ol_min,
        "new_in": "true" if args.use_true_inputs else "meas",
        "old_in": "true" if old_true else "meas",
        "n_new": len(new_seq),
        "n_old_eval": len(old_seq),
        "ref_before_mohm": [r0_b * 1e3, r1_b * 1e3],
        "ref_after_mohm": [r0_a * 1e3, r1_a * 1e3],
        "new_rmse_before": new0,
        "old_rmse_before": old0,
        "new_rmse_after": last["new_rmse"],
        "old_rmse_after": last["old_rmse"],
        "k0": model.k0,
        "k1": model.k1,
        "n_win": last["n_win"],
        "n_update": last["n_update"],
        "n_skip_gate": last["n_skip_gate"],
        "n_skip_small": last["n_skip_small"],
        "cc_probe": cc,
        "pass_log": [
            {k: v for k, v in row.items() if k != "per_seq"} | {"n_seq": len(row.get("per_seq") or [])}
            for row in pass_rows
        ],
        "k_per_seq": last.get("per_seq"),
    }
    (out_dir / "window.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    blob = {
        "model": model.base.state_dict(),
        "log_k0": float(model.log_k0.detach()),
        "log_k1": float(model.log_k1.detach()),
        "k0": model.k0,
        "k1": model.k1,
        "incr_mode": "window_k",
        "scheme": cfg.scheme,
    }
    torch.save(blob, out_dir / "last.pt")
    scaler.save(out_dir / "scaler.json")
    cfg.to_json(out_dir / "config.json")
    print(
        f"完成  k0={model.k0:.4f} k1={model.k1:.4f}  "
        f"新 {new0*1e3:.2f}→{last['new_rmse']*1e3:.2f} mV  "
        f"旧 {old0*1e3:.2f}→{last['old_rmse']*1e3:.2f} mV"
    )
    print(f"写出 {out_dir / 'window.json'}  （未覆盖 {mlp_dir / 'best.pt'}）")
    if not args.no_plot:
        from plot import plot_from_out

        prefix = args.fig_prefix or f"local/{out_dir.name}"
        figs = plot_from_out(
            out_dir,
            new_dir=new_dir,
            mlp_dir=mlp_dir,
            fig_prefix=prefix,
            use_true_inputs=args.use_true_inputs,
        )
        for fig in figs:
            print(f"图    {fig}")


if __name__ == "__main__":
    main()
