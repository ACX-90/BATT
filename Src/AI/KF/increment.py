"""离线增量：冻 scaler，开环电压损失，Replay 或缩放适配。

不要用 e_post，不要逐步 backward，不要重拟合 scaler（Doc/03-a、Doc/03-c）。
用法（仓库根目录）：

    python Src/AI/KF/increment.py --mode replay --new-dir Data/ai_kf/logs --replay-dir Data/grid
    python Src/AI/KF/increment.py --mode scale --new-dir Data/ai_kf/logs
    python Src/AI/KF/increment.py --mode retrain --new-dir Data/soh_k115 --replay-dir Data/grid
    python Src/AI/KF/compare.py --make-new --r0-scale 1.15 --r1-scale 1.15
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

KF_DIR = Path(__file__).resolve().parent
AI_DIR = KF_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from KF.adapter import ScaleAdapter
from KF.config import REPO_ROOT
from KF.ocv import ocv_nmc
from MLP.ckpt import epoch_path, resolve_ckpt_path, write_latest_pointer
from MLP.config import TrainConfig
from MLP.dataset import FeatureScaler, _load_csv, split_sequences
from MLP.ecm import ecm_forward
from MLP.infer import load_bundle
from MLP.model import ParamMLP
from MLP.train import set_seed


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _list_csvs(data_dir: Path, pattern: str | None) -> list[Path]:
    index = data_dir / "index.csv"
    paths: list[Path] = []
    if pattern:
        paths = sorted(data_dir.glob(pattern))
    elif index.exists():
        with index.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rel = row.get("path") or row.get("file")
                if not rel:
                    continue
                p = Path(rel)
                paths.append(p if p.is_absolute() else REPO_ROOT / p)
    else:
        paths = sorted(p for p in data_dir.glob("*.csv") if p.name != "index.csv")
    out: list[Path] = []
    for path in paths:
        if path.exists():
            out.append(path)
            continue
        alt = data_dir / path.name
        if alt.exists():
            out.append(alt)
    if not out:
        raise FileNotFoundError(f"未找到轨迹 CSV：{data_dir} pattern={pattern}")
    return out


def _detect_style(raw: dict[str, np.ndarray]) -> str:
    if "soc_ah" in raw:
        return "field"
    return "grid"


def _col(raw: dict[str, np.ndarray], *names: str) -> np.ndarray:
    for name in names:
        if name in raw:
            return raw[name]
    raise KeyError(f"缺列 {names}，现有 {sorted(raw)}")


def load_incr_sequences(
    data_dir: Path,
    *,
    pattern: str | None,
    use_true_inputs: bool,
    weight: float,
    style: str = "auto",
) -> list[dict[str, np.ndarray]]:
    seqs: list[dict[str, np.ndarray]] = []
    for path in _list_csvs(data_dir, pattern):
        raw = _load_csv(path)
        kind = style if style != "auto" else _detect_style(raw)
        try:
            if kind == "field":
                i_a = _col(raw, "i_used_a", "i_meas_a", "i_true_a")
                soc = _col(raw, "soc_ah")
                t_c = _col(raw, "t_meas_c", "t_true_c")
                u_t = _col(raw, "u_t_meas_v", "u_t_true_v")
            else:
                if use_true_inputs:
                    i_a = _col(raw, "i_true_a", "i_meas_a")
                    soc = _col(raw, "soc_true", "soc_meas", "soc_ah")
                    t_c = _col(raw, "t_true_c", "t_meas_c")
                else:
                    i_a = _col(raw, "i_meas_a", "i_true_a")
                    soc = _col(raw, "soc_meas", "soc_true", "soc_ah")
                    t_c = _col(raw, "t_meas_c", "t_true_c")
                u_t = _col(raw, "u_t_meas_v", "u_t_true_v")
        except KeyError as exc:
            raise KeyError(f"{path}: {exc}") from exc
        u_ocv = np.asarray(ocv_nmc(soc, t_c), dtype=float)
        n = len(i_a)
        nan = np.full(n, np.nan)
        seqs.append(
            {
                "name": path.name,
                "i": np.asarray(i_a, dtype=float),
                "soc": np.asarray(soc, dtype=float),
                "t": np.asarray(t_c, dtype=float),
                "u_ocv": u_ocv,
                "u_t": np.asarray(u_t, dtype=float),
                "r0": raw.get("r0_ohm", nan),
                "r1": raw.get("r1_ohm", nan),
                "c1": raw.get("c1_f", nan),
                "u_p0": float(raw["u_p_ol"][0]) if "u_p_ol" in raw else 0.0,
                "weight": float(weight),
                "kind": kind,
            }
        )
    return seqs


def subsample(seqs: list[dict], n: int | None, seed: int) -> list[dict]:
    if n is None or n >= len(seqs) or n <= 0:
        return seqs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seqs), size=n, replace=False)
    return [seqs[i] for i in np.sort(idx)]


class IncrDataset(Dataset):
    def __init__(self, sequences: list[dict], scaler: FeatureScaler) -> None:
        self.sequences = sequences
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        feat = np.stack([seq["i"], seq["soc"], seq["t"]], axis=-1)
        feat_n = self.scaler.transform(feat).astype(np.float32)
        n = len(seq["i"])
        return {
            "x": torch.from_numpy(feat_n),
            "i": torch.from_numpy(seq["i"].astype(np.float32)),
            "u_ocv": torch.from_numpy(seq["u_ocv"].astype(np.float32)),
            "u_t": torch.from_numpy(seq["u_t"].astype(np.float32)),
            "u_p0": torch.tensor(float(seq.get("u_p0", 0.0)), dtype=torch.float32),
            "weight": torch.tensor(float(seq.get("weight", 1.0)), dtype=torch.float32),
            "mask": torch.ones(n, dtype=torch.float32),
        }


def collate_pad(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    t_max = max(item["i"].shape[0] for item in batch)
    out: dict[str, list[torch.Tensor]] = {k: [] for k in batch[0]}
    for item in batch:
        n = item["i"].shape[0]
        for key, val in item.items():
            if val.ndim == 0:
                out[key].append(val)
                continue
            if n == t_max:
                out[key].append(val)
                continue
            pad = val.new_zeros((t_max, *val.shape[1:]))
            pad[:n] = val
            out[key].append(pad)
    return {k: torch.stack(v, dim=0) for k, v in out.items()}


def voltage_loss_ol(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    cfg: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    r0, r1, c1 = model(batch["x"])
    u_hat, _ = ecm_forward(
        batch["i"],
        batch["u_ocv"],
        r0,
        r1,
        c1,
        dt_s=cfg.dt_s,
        u_p0=batch["u_p0"],
    )
    mask = batch["mask"]
    err = (u_hat - batch["u_t"]) * mask
    w = batch["weight"]
    denom = mask.sum(dim=1).clamp_min(1.0)
    per = 0.5 * err.pow(2).sum(dim=1) / denom
    l_v = (per * w).mean()
    log_r0 = torch.log(r0.clamp_min(1e-12))
    log_r1 = torch.log(r1.clamp_min(1e-12))
    m2 = mask[:, 1:] * mask[:, :-1]
    l_s = 0.5 * (
        ((log_r0[:, 1:] - log_r0[:, :-1]).pow(2) * m2).sum() / m2.sum().clamp_min(1.0)
        + ((log_r1[:, 1:] - log_r1[:, :-1]).pow(2) * m2).sum() / m2.sum().clamp_min(1.0)
    )
    loss = l_v + cfg.lambda_smooth * l_s
    rmse = (err.pow(2).sum() / mask.sum().clamp_min(1.0)).sqrt()
    return loss, {
        "loss": float(loss.detach()),
        "rmse_v": float(rmse.detach()),
        "l_v": float(l_v.detach()),
        "l_s": float(l_s.detach()),
    }


@torch.no_grad()
def evaluate(model, loader, cfg, device) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "rmse_v": 0.0, "l_v": 0.0, "l_s": 0.0}
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        _, stats = voltage_loss_ol(model, batch, cfg)
        for key, val in stats.items():
            totals[key] += val
        n += 1
    model.train()
    if n == 0:
        return totals
    return {k: v / n for k, v in totals.items()}


def ref_params(model: torch.nn.Module, scaler: FeatureScaler) -> tuple[float, float]:
    feat = np.array([[100.0, 0.50, 25.0]], dtype=float)
    xn = torch.from_numpy(scaler.transform(feat).astype(np.float32))
    with torch.no_grad():
        r0, r1, _ = model(xn)
    return float(r0), float(r1)


def _make_loader(seqs: list[dict], scaler: FeatureScaler, batch_size: int, shuffle: bool) -> DataLoader | None:
    if not seqs:
        return None
    ds = IncrDataset(seqs, scaler)
    return DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=shuffle,
        collate_fn=collate_pad,
    )


def run_increment(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mlp_dir = _resolve(args.mlp_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = mlp_dir / "scaler.json"
    if not scaler_path.exists():
        raise FileNotFoundError(f"增量必须沿用旧 scaler：{scaler_path}")
    scaler = FeatureScaler.load(scaler_path)
    ckpt = resolve_ckpt_path(
        mlp_dir,
        epoch=args.epoch,
        ckpt=args.ckpt,
        fallback_last=mlp_dir / "last.pt",
        fallback_best=mlp_dir / "best.pt",
    )
    if args.best:
        ckpt = mlp_dir / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
    model, _, cfg = load_bundle(ckpt, mlp_dir / "config.json", scaler_path)
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.pretrain_epochs = 0
    if args.batch_size:
        cfg.batch_size = args.batch_size
    cfg.device = str(device)

    if getattr(args, "from_scratch", False):
        model = ParamMLP(cfg)
        print("合集重训：新建网络，冻旧 scaler，不加载权重")

    new_seq = load_incr_sequences(
        _resolve(args.new_dir),
        pattern=args.new_glob,
        use_true_inputs=args.use_true_inputs,
        weight=1.0,
        style=args.new_style,
    )
    # 旧集跟舰队训练时的输入列：真值舰队仍吃真值（A/B/C 不变）；测量列舰队吃测量列
    old_true = bool(cfg.use_true_inputs)
    train_old_seq: list[dict] = []
    if args.mode in {"replay", "retrain"} and args.replay_dir:
        train_old_seq = load_incr_sequences(
            _resolve(args.replay_dir),
            pattern=args.replay_glob,
            use_true_inputs=old_true,
            weight=args.beta,
            style="grid",
        )
        if args.mode == "replay":
            train_old_seq = subsample(train_old_seq, args.replay_n, args.seed)

    eval_old_dir = getattr(args, "eval_old_dir", None) or (
        args.replay_dir if args.replay_dir else ""
    )
    eval_old_seq: list[dict] = []
    if eval_old_dir:
        eval_old_seq = load_incr_sequences(
            _resolve(eval_old_dir),
            pattern=args.replay_glob,
            use_true_inputs=old_true,
            weight=1.0,
            style="grid",
        )

    new_train, new_val = split_sequences(new_seq, args.val_ratio, args.seed)
    old_train, _old_val_unused = (
        split_sequences(train_old_seq, args.val_ratio, args.seed + 1) if train_old_seq else ([], [])
    )
    train_seq = new_train + old_train

    print(
        f"增量 mode={args.mode}  ckpt={ckpt.name}  "
        f"new={len(new_seq)} (train {len(new_train)} / val {len(new_val)})  "
        f"train_old={len(train_old_seq)}  eval_old={len(eval_old_seq)}  "
        f"new_in={'true' if args.use_true_inputs else 'meas'}  "
        f"old_in={'true' if old_true else 'meas'}  scaler 已冻结"
    )
    r0_b, r1_b = ref_params(model, scaler)
    print(f"旧参考点 (50%, 25°C, 1C)  R0={r0_b*1e3:.4f} mΩ  R1={r1_b*1e3:.4f} mΩ")

    new_val_loader = _make_loader(new_val or new_seq, scaler, cfg.batch_size, False)
    old_val_loader = _make_loader(eval_old_seq, scaler, cfg.batch_size, False)
    train_loader = _make_loader(train_seq, scaler, cfg.batch_size, True)
    if train_loader is None and not args.eval_only:
        raise RuntimeError("没有可训轨迹")

    if args.mode == "scale":
        model = ScaleAdapter(model).to(device)
        opt = torch.optim.Adam([model.log_k0, model.log_k1], lr=args.lr)
    else:
        model = model.to(device)
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=cfg.weight_decay,
        )

    def _eval_pair() -> tuple[dict[str, float], dict[str, float]]:
        new_s = evaluate(model, new_val_loader, cfg, device) if new_val_loader else {}
        old_s = evaluate(model, old_val_loader, cfg, device) if old_val_loader else {}
        return new_s, old_s

    new0, old0 = _eval_pair()
    if new0:
        print(f"增量前  新轨迹 RMSE={new0['rmse_v']*1e3:.2f} mV")
    if old0:
        print(f"增量前  旧回放 RMSE={old0['rmse_v']*1e3:.2f} mV")

    def _pack_meta(
        r0_a: float,
        r1_a: float,
        new_s: dict[str, float],
        old_s: dict[str, float],
    ) -> dict:
        meta = {
            "mode": args.mode,
            "source_ckpt": str(ckpt),
            "new_dir": args.new_dir,
            "replay_dir": args.replay_dir,
            "eval_old_dir": eval_old_dir,
            "beta": args.beta,
            "epochs": 0 if args.eval_only else args.epochs,
            "ref_before_mohm": [r0_b * 1e3, r1_b * 1e3],
            "ref_after_mohm": [r0_a * 1e3, r1_a * 1e3],
            "new_rmse_before": new0.get("rmse_v"),
            "old_rmse_before": old0.get("rmse_v"),
            "new_rmse_after": new_s.get("rmse_v"),
            "old_rmse_after": old_s.get("rmse_v"),
        }
        if isinstance(model, ScaleAdapter):
            meta["k0"] = model.k0
            meta["k1"] = model.k1
        return meta

    if args.eval_only:
        meta = _pack_meta(r0_b, r1_b, new0, old0)
        (out_dir / "incr.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return out_dir

    history = [{"phase": "before", "new": new0, "old": old0}]
    best_score = float("inf")
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    def _payload(epoch_done: int, score: float) -> dict:
        blob = {
            "model": (model.base if isinstance(model, ScaleAdapter) else model).state_dict(),
            "scheme": cfg.scheme,
            "best_rmse": float(score),
            "epoch_done": int(epoch_done),
            "incr_mode": args.mode,
        }
        if isinstance(model, ScaleAdapter):
            blob["log_k0"] = float(model.log_k0.detach())
            blob["log_k1"] = float(model.log_k1.detach())
            blob["k0"] = model.k0
            blob["k1"] = model.k1
        return blob

    for epoch in range(1, args.epochs + 1):
        model.train()
        acc = {"loss": 0.0, "rmse_v": 0.0, "l_v": 0.0, "l_s": 0.0}
        n = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss, stats = voltage_loss_ol(model, batch, cfg)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
                )
            opt.step()
            for key, val in stats.items():
                acc[key] += val
            n += 1
        train_stats = {k: v / max(n, 1) for k, v in acc.items()}
        new_s, old_s = _eval_pair()
        score = new_s.get("rmse_v", train_stats["rmse_v"])
        if old_s:
            score = score + args.beta_eval * old_s.get("rmse_v", 0.0)
        row = {"phase": "voltage", "epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}}
        row.update({f"new_{k}": v for k, v in new_s.items()})
        row.update({f"old_{k}": v for k, v in old_s.items()})
        history.append(row)
        payload = _payload(epoch, score)
        ep = epoch_path(out_dir, epoch)
        ep.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, ep)
        torch.save(payload, last_path)
        write_latest_pointer(out_dir, epoch)
        mark = ""
        if score < best_score:
            best_score = score
            torch.save(payload, best_path)
            mark = "  *"
        extra = ""
        if isinstance(model, ScaleAdapter):
            extra = f"  k0={model.k0:.4f} k1={model.k1:.4f}"
        print(
            f"[{epoch:03d}/{args.epochs}] train={train_stats['rmse_v']*1e3:.2f} mV  "
            f"new={new_s.get('rmse_v', float('nan'))*1e3:.2f} mV  "
            f"old={old_s.get('rmse_v', float('nan'))*1e3:.2f} mV{extra}{mark}"
        )

    r0_a, r1_a = ref_params(model, scaler)
    print(
        f"新参考点  R0={r0_a*1e3:.4f} mΩ ({(r0_a/r0_b-1)*100:+.2f}%)  "
        f"R1={r1_a*1e3:.4f} mΩ ({(r1_a/r1_b-1)*100:+.2f}%)"
    )
    scaler.save(out_dir / "scaler.json")
    cfg.to_json(out_dir / "config.json")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    last_new, last_old = _eval_pair()
    meta = _pack_meta(r0_a, r1_a, last_new or new0, last_old or old0)
    if isinstance(model, ScaleAdapter):
        meta["k0"] = model.k0
        meta["k1"] = model.k1
    (out_dir / "incr.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"已保存 {best_path}  (未覆盖 {mlp_dir / 'best.pt'})")
    return best_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLP-ECM 离线增量（开环电压，冻 scaler）")
    p.add_argument("--mode", default="replay", choices=["replay", "scale", "finetune", "retrain"])
    p.add_argument("--mlp-dir", default="Data/ai_mlp")
    p.add_argument("--out-dir", default="Data/ai_kf/incr")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--best", action="store_true", help="用 mlp-dir/best.pt")
    p.add_argument("--new-dir", default="Data/ai_kf/logs")
    p.add_argument("--new-glob", default=None)
    p.add_argument("--new-style", default="auto", choices=["auto", "grid", "field"])
    p.add_argument("--replay-dir", default="Data/grid")
    p.add_argument("--replay-glob", default=None)
    p.add_argument("--replay-n", type=int, default=None, help="回放条数上限")
    p.add_argument("--beta", type=float, default=1.0, help="回放样本权重")
    p.add_argument("--beta-eval", type=float, default=1.0, help="best 分数里旧集 RMSE 权重")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5.0e-4)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--use-true-inputs", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument(
        "--eval-old-dir",
        default=None,
        help="验收旧区域目录；默认用 --replay-dir（finetune/scale 也建议显式传入）",
    )
    p.add_argument(
        "--from-scratch",
        action="store_true",
        help="合集重训时新建网络（仍冻旧 scaler），不加载 mlp-dir 权重",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "finetune":
        args.replay_n = 0
        if not args.eval_old_dir:
            args.eval_old_dir = args.replay_dir or "Data/grid"
        args.replay_dir = ""
    if args.mode == "scale":
        if not args.eval_old_dir:
            args.eval_old_dir = args.replay_dir or "Data/grid"
        args.replay_dir = ""
    if args.mode == "retrain" and not args.replay_dir:
        args.replay_dir = "Data/grid"
    run_increment(args)


if __name__ == "__main__":
    main()
