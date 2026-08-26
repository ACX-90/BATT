"""车上滑窗：3×8×2 残差头只对照。冻舰队与 φ，SGD 只动 18 个数。

仓库根目录：

    python Src/AI/EV_Local/head.py --exp both --make-phi
    python Src/AI/EV_Local/head.py --exp a --out-a Data/ai_local/head_k115_p4 --passes 4

无 Replay。不覆盖 Data/ai_mlp / Data/grid。φ 预训练是实验室点式蒸馏，不是增量。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

EV_DIR = Path(__file__).resolve().parent
AI_DIR = EV_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from KF.adapter import KGRID_SOC, KGRID_T, ResidualHeadAdapter, phi_from_mlp
from KF.config import REPO_ROOT
from KF.increment import load_incr_sequences, ref_params
from MLP.config import TrainConfig
from MLP.dataset import FeatureScaler, load_grid_sequences
from MLP.ecm import ecm_forward
from MLP.infer import load_bundle
from MLP.model import ParamMLP
from MLP.train import set_seed


def _load_window():
    spec = importlib.util.spec_from_file_location("ev_local_window", EV_DIR / "window.py")
    if spec is None or spec.loader is None:
        raise ImportError(EV_DIR / "window.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_win = _load_window()
window_gate = _win.window_gate
overall_rmse = _win.overall_rmse
_seq_tensors = _win._seq_tensors


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _roll_up(model, sl, u_p0, dt_s, center) -> torch.Tensor:
    with torch.no_grad():
        n = sl["i"].shape[0]
        r0, r1, c1 = model(sl["x"][center : center + 1].unsqueeze(0))
        r0, r1, c1 = r0.expand(1, n), r1.expand(1, n), c1.expand(1, n)
        _, u_p = ecm_forward(
            sl["i"].unsqueeze(0), sl["u_ocv"].unsqueeze(0), r0, r1, c1, dt_s=dt_s, u_p0=u_p0
        )
    return u_p[:, -1].detach()


def _zero_row(grad: torch.Tensor | None, row: int) -> None:
    if grad is None:
        return
    grad[row].zero_()


def step_window(model, opt, sl, u_p0, *, dt_s, i_np, args, i_prev) -> dict:
    n = int(sl["i"].shape[0])
    mid = n // 2
    gated, gstat = window_gate(
        i_np,
        dt_s=dt_s,
        i_edge_a=args.i_edge,
        rest_eps=args.rest_eps,
        rest_s=args.rest_s,
        i_prev=i_prev,
    )
    i_rms = float(np.sqrt(np.mean(np.square(i_np))))
    stats = {"gated": gated, "updated": False, "rmse": float("nan"), "i_rms": i_rms, **gstat}
    if not gated:
        stats["u_p0"] = _roll_up(model, sl, u_p0, dt_s, mid)
        return stats

    opt.zero_grad(set_to_none=True)
    r0c, r1c, c1c = model(sl["x"][mid : mid + 1].unsqueeze(0))
    r0, r1, c1 = r0c.expand(1, n), r1c.expand(1, n), c1c.expand(1, n)
    u_hat, u_p = ecm_forward(
        sl["i"].unsqueeze(0), sl["u_ocv"].unsqueeze(0), r0, r1, c1, dt_s=dt_s, u_p0=u_p0
    )
    err = u_hat.squeeze(0) - sl["u_t"]
    rmse = float(err.pow(2).mean().sqrt().detach().cpu())
    stats["rmse"] = rmse
    if rmse < args.e_ol_min:
        stats["u_p0"] = u_p[:, -1].detach()
        return stats

    loss = 0.5 * err.pow(2).mean()
    loss.backward()
    if i_rms < 20.0:
        _zero_row(model.head.weight.grad, 0)
        _zero_row(model.head.bias.grad, 0)
    if gstat["rest_s"] < args.rest_s:
        _zero_row(model.head.weight.grad, 1)
        _zero_row(model.head.bias.grad, 1)
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.grad_clip)
    opt.step()
    stats["updated"] = True
    stats["u_p0"] = u_p[:, -1].detach()
    return stats


def run_pass(model, opt, seqs, scaler, cfg, device, args) -> dict:
    n_win = n_upd = n_skip_gate = n_skip_small = n_hi_i = 0
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
                args=args,
                i_prev=i_prev,
            )
            u_p0 = st.pop("u_p0")
            n_win += 1
            if st["updated"]:
                n_upd += 1
                if float(st["i_rms"]) >= 20.0:
                    n_hi_i += 1
            elif not st["gated"]:
                n_skip_gate += 1
            else:
                n_skip_small += 1
        d0, d1 = _delta_at(model, scaler, 100.0, float(seq["soc"].mean()), float(seq["t"].mean()))
        print(
            f"  seq {si+1:02d}/{len(seqs)} {seq['name']}  "
            f"dR={d0*1e6:+.0f}/{d1*1e6:+.0f} uΩ  T={float(seq['t'].mean()):+.1f}",
            flush=True,
        )
    return {
        "n_win": n_win,
        "n_update": n_upd,
        "n_skip_gate": n_skip_gate,
        "n_skip_small": n_skip_small,
        "n_update_hi_i": n_hi_i,
    }


def cc_probe(model_factory, seqs, cfg, device, args) -> dict:
    model = model_factory()
    opt = torch.optim.SGD(model.trainable_parameters(), lr=args.lr)
    n_win = n_upd = 0
    for seq in seqs[: min(5, len(seqs))]:
        mag = np.abs(seq["i"])
        mask = (mag > 80.0) & (mag < 120.0)
        if mask.sum() < args.win:
            continue
        idx = np.flatnonzero(mask)
        cuts = np.where(np.diff(idx) > 1)[0]
        block = idx if len(cuts) == 0 else max(np.split(idx, cuts + 1), key=len)
        t = _seq_tensors(seq, model_factory.scaler, device)
        sl_all = {k: v[int(block[0]) : int(block[-1]) + 1] for k, v in t.items()}
        i_np = seq["i"][int(block[0]) : int(block[-1]) + 1]
        u_p0 = t["i"].new_tensor([0.0])
        n = len(i_np)
        for start in range(0, n, args.win):
            end = min(start + win_len(args), n)
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
                args=args,
                i_prev=i_prev,
            )
            u_p0 = st.pop("u_p0")
            n_win += 1
            n_upd += int(st["updated"])
    d0, d1 = _delta_at(model, model_factory.scaler, 100.0, 0.50, 25.0)
    return {
        "n_win": n_win,
        "n_update": n_upd,
        "dr0_uohm": d0 * 1e6,
        "dr1_uohm": d1 * 1e6,
    }


def win_len(args) -> int:
    return int(args.win)


def _xn(scaler: FeatureScaler, i_a: float, soc: float, t_c: float, device) -> torch.Tensor:
    feat = np.array([[i_a, soc, t_c]], dtype=float)
    return torch.from_numpy(scaler.transform(feat).astype(np.float32)).to(device)


def _delta_at(model: ResidualHeadAdapter, scaler: FeatureScaler, i_a: float, soc: float, t_c: float):
    return model.delta_at(_xn(scaler, i_a, soc, t_c, model.head.weight.device))


def snapshot(model: ResidualHeadAdapter, scaler: FeatureScaler, device) -> dict:
    k0, k1, dr0, dr1 = [], [], [], []
    for s in KGRID_SOC:
        row_k0, row_k1, row_d0, row_d1 = [], [], [], []
        for t_c in KGRID_T:
            x = _xn(scaler, 100.0, float(s), float(t_c), device)
            with torch.no_grad():
                r0, r1, _ = model(x)
                r0f, r1f, _ = model.fleet(x)
            row_d0.append(float(r0 - r0f))
            row_d1.append(float(r1 - r1f))
            row_k0.append(float(r0 / r0f.clamp_min(1e-12)))
            row_k1.append(float(r1 / r1f.clamp_min(1e-12)))
        k0.append(row_k0)
        k1.append(row_k1)
        dr0.append(row_d0)
        dr1.append(row_d1)
    a0, a1 = np.asarray(k0), np.asarray(k1)
    return {
        "soc_node": list(KGRID_SOC),
        "t_node": list(KGRID_T),
        "k0": k0,
        "k1": k1,
        "dr0": dr0,
        "dr1": dr1,
        "k0_cold": float(a0[:, 0].mean()),
        "k1_cold": float(a1[:, 0].mean()),
        "k0_mid": float(a0[:, 1:-1].mean()),
        "k1_mid": float(a1[:, 1:-1].mean()),
        "k0_hot": float(a0[:, -1].mean()),
        "k1_hot": float(a1[:, -1].mean()),
    }


def fmt_table(name: str, soc: list, t_c: list, k: list[list[float]]) -> str:
    head = f"{name:4s}" + "".join(f"  T{tc:+.0f}" for tc in t_c)
    lines = [head]
    for i, s in enumerate(soc):
        row = f"s{s:.2f}" + "".join(f"  {k[i][j]:5.3f}" for j in range(len(t_c)))
        lines.append(row)
    return "\n".join(lines)


def by_temp(seqs: list[dict], lo: float, hi: float) -> list[dict]:
    return [seq for seq in seqs if lo <= float(np.mean(seq["t"])) <= hi]


def pretrain_phi(
    fleet: ParamMLP,
    scaler: FeatureScaler,
    fleet_cfg: TrainConfig,
    phi_dir: Path,
    device: torch.device,
    *,
    epochs: int,
) -> ParamMLP:
    """点式对数电阻蒸馏到 3×8×2。实验室一次，Adam，不进车上增量。"""
    phi_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig.from_dict(
        {
            **{k: getattr(fleet_cfg, k) for k in TrainConfig.__dataclass_fields__},
            "hidden": (8,),
            "out_dir": str(phi_dir),
            "pretrain_epochs": 0,
            "epochs": int(epochs),
        }
    )
    student = ParamMLP(cfg).to(device)
    seqs = load_grid_sequences(fleet_cfg)
    xs, y0, y1 = [], [], []
    fleet.eval()
    with torch.no_grad():
        for seq in seqs:
            feat = np.stack([seq["i"], seq["soc"], seq["t"]], axis=-1)
            xn = torch.from_numpy(scaler.transform(feat).astype(np.float32)).to(device)
            r0, r1, _ = fleet(xn)
            xs.append(xn.cpu())
            y0.append(r0.cpu())
            y1.append(r1.cpu())
    x = torch.cat(xs, dim=0)
    r0t = torch.cat(y0, dim=0)
    r1t = torch.cat(y1, dim=0)
    n = int(x.shape[0])
    rng = np.random.default_rng(42)
    take = min(n, 80_000)
    idx = rng.choice(n, size=take, replace=False)
    x = x[idx].to(device)
    r0t = r0t[idx].to(device)
    r1t = r1t[idx].to(device)
    opt = torch.optim.Adam(student.parameters(), lr=2.0e-3, weight_decay=1.0e-6)
    bs = 4096
    print(f"蒸馏 φ  3×8×2  n={take} epochs={epochs}  → {phi_dir}", flush=True)
    student.train()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(take, device=device)
        tot = 0.0
        n_b = 0
        for start in range(0, take, bs):
            sl = perm[start : start + bs]
            opt.zero_grad(set_to_none=True)
            r0, r1, _ = student(x[sl])
            loss = 0.5 * (
                (torch.log(r0.clamp_min(1e-12)) - torch.log(r0t[sl].clamp_min(1e-12))).pow(2).mean()
                + (torch.log(r1.clamp_min(1e-12)) - torch.log(r1t[sl].clamp_min(1e-12))).pow(2).mean()
            )
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            n_b += 1
        if ep == 1 or ep == epochs or ep % 5 == 0:
            print(f"  phi [{ep:03d}/{epochs}] logR {tot / max(n_b, 1):.5f}", flush=True)
    student.eval()
    with torch.no_grad():
        r0, r1, _ = student(x[:4096])
        e0 = float((r0 - r0t[:4096]).pow(2).mean().sqrt())
        e1 = float((r1 - r1t[:4096]).pow(2).mean().sqrt())
    print(f"  φ vs 舰队  R0 {e0*1e6:.1f} µΩ  R1 {e1*1e6:.1f} µΩ", flush=True)
    payload = {
        "model": student.state_dict(),
        "scheme": "B",
        "best_rmse": float("nan"),
        "epoch_done": int(epochs),
        "phi_vs_fleet_r0": e0,
        "phi_vs_fleet_r1": e1,
    }
    torch.save(payload, phi_dir / "best.pt")
    torch.save(payload, phi_dir / "last.pt")
    scaler.save(phi_dir / "scaler.json")
    cfg.to_json(phi_dir / "config.json")
    return student


def load_or_make_phi(
    fleet: ParamMLP,
    scaler: FeatureScaler,
    fleet_cfg: TrainConfig,
    phi_dir: Path,
    device: torch.device,
    args,
) -> nn.Module:
    ckpt = phi_dir / "best.pt"
    if ckpt.exists() and not args.make_phi:
        h8, _, cfg_h = load_bundle(ckpt, phi_dir / "config.json", phi_dir / "scaler.json")
        if tuple(cfg_h.hidden) != (8,):
            raise ValueError(f"{phi_dir} hidden={cfg_h.hidden}，1c 必须是 3×8×2")
        h8 = h8.to(device).eval()
        print(f"加载 φ  {phi_dir}  hidden={cfg_h.hidden}")
        return phi_from_mlp(h8)
    h8 = pretrain_phi(fleet, scaler, fleet_cfg, phi_dir, device, epochs=args.phi_epochs)
    return phi_from_mlp(h8.to(device).eval())


class _Factory:
    def __init__(self, fleet, phi, scaler, dr_max, device):
        self.fleet = fleet
        self.phi = phi
        self.scaler = scaler
        self.dr_max = dr_max
        self.device = device

    def __call__(self) -> ResidualHeadAdapter:
        return ResidualHeadAdapter(self.fleet, self.phi, dr_max=self.dr_max).to(self.device)


def run_exp(args, *, new_dir: Path, out_dir: Path, tag: str, fleet, phi, scaler, cfg, device) -> dict:
    old_dir = _resolve(args.old_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if new_dir.resolve() == old_dir.resolve():
        raise RuntimeError("新年份不能和旧网格是同一目录")

    new_seq = load_incr_sequences(
        new_dir, pattern=None, use_true_inputs=args.use_true_inputs, weight=1.0, style="grid"
    )
    old_true = bool(cfg.use_true_inputs)
    old_seq = load_incr_sequences(
        old_dir, pattern=None, use_true_inputs=old_true, weight=1.0, style="grid"
    )
    factory = _Factory(fleet, phi, scaler, args.dr_max, device)
    frozen = factory()
    if frozen.n_trainable() != 18:
        raise RuntimeError(f"可训参数应是 18，得到 {frozen.n_trainable()}")

    print(
        f"head tag={tag} mlp={_resolve(args.mlp_dir)} new={len(new_seq)} old_eval={len(old_seq)}  "
        f"win={args.win} lr={args.lr} dr_max={args.dr_max} n_train={frozen.n_trainable()} Replay=OFF"
    )

    new0 = overall_rmse(frozen, new_seq, scaler, cfg, device)
    old0 = overall_rmse(frozen, old_seq, scaler, cfg, device)
    cold_seq = by_temp(new_seq, -15, -5)
    mid_seq = by_temp(new_seq, 15, 60)
    new0_cold = overall_rmse(frozen, cold_seq, scaler, cfg, device) if cold_seq else None
    new0_mid = overall_rmse(frozen, mid_seq, scaler, cfg, device) if mid_seq else None
    r0_b, r1_b = ref_params(frozen, scaler)
    print(f"冻结  新 {new0*1e3:.2f} mV  旧 {old0*1e3:.2f} mV  R0={r0_b*1e3:.3f} mΩ  R1={r1_b*1e3:.3f} mΩ", end="")
    if new0_cold is not None:
        print(f"  冷 {new0_cold*1e3:.2f}  中温 {new0_mid*1e3:.2f}", end="")
    print()

    cc = cc_probe(factory, new_seq, cfg, device, args)
    print(
        f"恒流探针  win={cc['n_win']} update={cc['n_update']}  "
        f"dR={cc['dr0_uohm']:+.1f}/{cc['dr1_uohm']:+.1f} uΩ"
    )

    model = factory()
    opt = torch.optim.SGD(model.trainable_parameters(), lr=args.lr)
    row = None
    pass_rows: list[dict] = []
    for p_i in range(1, args.passes + 1):
        print(f"---- {tag} pass {p_i}/{args.passes} ----", flush=True)
        row = run_pass(model, opt, new_seq, scaler, cfg, device, args)
        new_a = overall_rmse(model, new_seq, scaler, cfg, device)
        old_a = overall_rmse(model, old_seq, scaler, cfg, device)
        row["pass"] = p_i
        row["new_rmse"] = new_a
        row["old_rmse"] = old_a
        pass_rows.append({k: v for k, v in row.items()})
        print(
            f"  pass {p_i}: upd {row['n_update']}/{row['n_win']}  "
            f"skip_gate {row['n_skip_gate']} skip_small {row['n_skip_small']}  "
            f"新 {new_a*1e3:.2f} mV  旧 {old_a*1e3:.2f} mV",
            flush=True,
        )

    tab = snapshot(model, scaler, device)
    print(fmt_table("k0~", tab["soc_node"], tab["t_node"], tab["k0"]))
    print(fmt_table("k1~", tab["soc_node"], tab["t_node"], tab["k1"]))
    print(
        f"  列均  冷 k0/k1={tab['k0_cold']:.3f}/{tab['k1_cold']:.3f}  "
        f"中温 {tab['k0_mid']:.3f}/{tab['k1_mid']:.3f}  "
        f"热 {tab['k0_hot']:.3f}/{tab['k1_hot']:.3f}"
    )
    r0_a, r1_a = ref_params(model, scaler)
    d_ref = _delta_at(model, scaler, 100.0, 0.50, 25.0)
    new_a = row["new_rmse"]
    old_a = row["old_rmse"]
    new_a_cold = overall_rmse(model, cold_seq, scaler, cfg, device) if cold_seq else None
    new_a_mid = overall_rmse(model, mid_seq, scaler, cfg, device) if mid_seq else None
    meta = {
        "mode": "window_head382",
        "tag": tag,
        "mlp_dir": str(_resolve(args.mlp_dir)),
        "phi_dir": str(_resolve(args.phi_dir)),
        "new_dir": str(new_dir),
        "old_dir": str(old_dir),
        "win": args.win,
        "lr": args.lr,
        "dr_max": args.dr_max,
        "n_trainable": 18,
        "n_passes": args.passes,
        "optimizer": "SGD",
        "replay": False,
        "center_R": True,
        "n_new": len(new_seq),
        "n_old_eval": len(old_seq),
        "ref_before_mohm": [r0_b * 1e3, r1_b * 1e3],
        "ref_after_mohm": [r0_a * 1e3, r1_a * 1e3],
        "dr_ref_uohm": [d_ref[0] * 1e6, d_ref[1] * 1e6],
        "k_eq_ref": [r0_a / max(r0_b, 1e-12), r1_a / max(r1_b, 1e-12)],
        "new_rmse_before": new0,
        "old_rmse_before": old0,
        "new_rmse_after": new_a,
        "old_rmse_after": old_a,
        "new_rmse_cold_before": new0_cold,
        "new_rmse_cold_after": new_a_cold,
        "new_rmse_mid_before": new0_mid,
        "new_rmse_mid_after": new_a_mid,
        "k_summary": {
            "k0_cold": tab["k0_cold"],
            "k1_cold": tab["k1_cold"],
            "k0_mid": tab["k0_mid"],
            "k1_mid": tab["k1_mid"],
            "k0_hot": tab["k0_hot"],
            "k1_hot": tab["k1_hot"],
        },
        "k_tables": tab,
        "n_win": row["n_win"],
        "n_update": row["n_update"],
        "n_skip_gate": row["n_skip_gate"],
        "n_skip_small": row["n_skip_small"],
        "n_update_hi_i": row["n_update_hi_i"],
        "cc_probe": cc,
        "pass_log": pass_rows,
    }
    (out_dir / "head.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "head_weight": model.head.weight.detach().cpu(),
            "head_bias": model.head.bias.detach().cpu(),
            "dr_max": args.dr_max,
            "incr_mode": "window_head382",
            "k_tables": tab,
        },
        out_dir / "last.pt",
    )
    scaler.save(out_dir / "scaler.json")
    cfg.to_json(out_dir / "config.json")
    print(
        f"完成  新 {new0*1e3:.2f}→{new_a*1e3:.2f} mV  旧 {old0*1e3:.2f}→{old_a*1e3:.2f} mV  "
        f"k~ {r0_a / max(r0_b, 1e-12):.3f}/{r1_a / max(r1_b, 1e-12):.3f}"
    )
    print(f"写出 {out_dir / 'head.json'}  （未覆盖 {_resolve(args.mlp_dir) / 'best.pt'}）")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="车上滑窗 3×8×2 残差头（只对照）")
    p.add_argument("--exp", default="both", choices=["a", "cold", "both"])
    p.add_argument("--mlp-dir", default="Data/ai_mlp")
    p.add_argument("--phi-dir", default="Data/ai_mlp_h8")
    p.add_argument("--new-dir", default="Data/soh_k115")
    p.add_argument("--cold-dir", default="Data/soh_cold_tm10")
    p.add_argument("--old-dir", default="Data/grid")
    p.add_argument("--out-a", default="Data/ai_local/head_k115")
    p.add_argument("--out-cold", default="Data/ai_local/head_cold")
    p.add_argument("--make-phi", action="store_true", help="强制重蒸 3×8 前层")
    p.add_argument("--phi-epochs", type=int, default=40)
    p.add_argument("--win", type=int, default=100)
    p.add_argument("--lr", type=float, default=2.0, help="18 个数的 SGD；不是 log k 的 lr=10")
    p.add_argument("--dr-max", type=float, default=2.0e-3, help="tanh 残差幅度 / Ω")
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--i-edge", type=float, default=20.0)
    p.add_argument("--rest-s", type=float, default=3.0)
    p.add_argument("--rest-eps", type=float, default=1.0)
    p.add_argument("--e-ol-min", type=float, default=0.002)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--use-true-inputs", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    mlp_dir = _resolve(args.mlp_dir)
    ckpt = mlp_dir / "best.pt"
    fleet, scaler, cfg = load_bundle(ckpt, mlp_dir / "config.json", mlp_dir / "scaler.json")
    for par in fleet.parameters():
        par.requires_grad_(False)
    fleet = fleet.to(device).eval()
    phi = load_or_make_phi(fleet, scaler, cfg, _resolve(args.phi_dir), device, args)

    if args.exp in {"a", "both"}:
        run_exp(
            args,
            new_dir=_resolve(args.new_dir),
            out_dir=_resolve(args.out_a),
            tag="A_x115",
            fleet=fleet,
            phi=phi,
            scaler=scaler,
            cfg=cfg,
            device=device,
        )
    if args.exp in {"cold", "both"}:
        run_exp(
            args,
            new_dir=_resolve(args.cold_dir),
            out_dir=_resolve(args.out_cold),
            tag="cold_tm10",
            fleet=fleet,
            phi=phi,
            scaler=scaler,
            cfg=cfg,
            device=device,
        )


if __name__ == "__main__":
    main()
