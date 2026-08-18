"""训练 MLP-ECM。默认方案 B。

用法（仓库根目录）：

    python Src/AI/MLP/train.py
    python Src/AI/MLP/train.py --scheme B --epochs 40
    python Src/AI/MLP/train.py --resume
    python Src/AI/MLP/train.py --epoch 12 --epochs 10
    python Src/AI/MLP/train.py --list-ckpts
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

MLP_DIR = Path(__file__).resolve().parent
AI_DIR = MLP_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from MLP.ckpt import (
    epoch_path,
    format_epoch_list,
    latest_epoch,
    resolve_ckpt_path,
    write_latest_pointer,
)
from MLP.config import TrainConfig
from MLP.dataset import (
    FeatureScaler,
    TrajectoryDataset,
    collate_traj,
    fit_scaler,
    load_grid_sequences,
    split_sequences,
)
from MLP.ecm import ecm_forward, ecm_forward_tbptt
from MLP.model import ParamMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def voltage_loss(
    model: ParamMLP,
    batch: dict[str, torch.Tensor],
    cfg: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    r0, r1, c1 = model(batch["x"])
    if cfg.tbptt > 0:
        u_hat = ecm_forward_tbptt(
            batch["i"], batch["u_ocv"], r0, r1, c1, dt_s=cfg.dt_s, window=cfg.tbptt
        )
    else:
        u_hat, _ = ecm_forward(batch["i"], batch["u_ocv"], r0, r1, c1, dt_s=cfg.dt_s)
    err = u_hat - batch["u_t"]
    l_v = 0.5 * err.pow(2).mean()
    log_r0 = torch.log(r0.clamp_min(1e-12))
    log_r1 = torch.log(r1.clamp_min(1e-12))
    l_s = 0.5 * (
        (log_r0[:, 1:] - log_r0[:, :-1]).pow(2).mean()
        + (log_r1[:, 1:] - log_r1[:, :-1]).pow(2).mean()
    )
    loss = l_v + cfg.lambda_smooth * l_s
    rmse = err.detach().pow(2).mean().sqrt()
    return loss, {"loss": float(loss.detach()), "rmse_v": float(rmse), "l_v": float(l_v.detach()), "l_s": float(l_s.detach())}


def teacher_loss(model: ParamMLP, batch: dict[str, torch.Tensor], cfg: TrainConfig) -> torch.Tensor:
    r0, r1, c1 = model(batch["x"])
    l_r = (torch.log(r0) - torch.log(batch["r0"].clamp_min(1e-12))).pow(2).mean()
    l_r = l_r + (torch.log(r1) - torch.log(batch["r1"].clamp_min(1e-12))).pow(2).mean()
    if cfg.scheme.upper() == "A":
        l_r = l_r + (torch.log(c1) - torch.log(batch["c1"].clamp_min(1.0))).pow(2).mean()
    return 0.5 * l_r


@torch.no_grad()
def evaluate(model: ParamMLP, loader: DataLoader, cfg: TrainConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "rmse_v": 0.0, "l_v": 0.0, "l_s": 0.0}
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        _, stats = voltage_loss(model, batch, cfg)
        for key, val in stats.items():
            totals[key] += val
        n += 1
    model.train()
    if n == 0:
        return totals
    return {k: v / n for k, v in totals.items()}


def _ckpt_payload(model: ParamMLP, opt: torch.optim.Optimizer, cfg: TrainConfig, best_rmse: float, epoch_done: int) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheme": cfg.scheme,
        "best_rmse": float(best_rmse),
        "epoch_done": int(epoch_done),
    }


def _trim_history(history: list[dict], epoch_done: int) -> list[dict]:
    kept: list[dict] = []
    for row in history:
        if row.get("phase") == "pretrain":
            kept.append(row)
        elif row.get("phase") == "voltage" and int(row.get("epoch", 0)) <= epoch_done:
            kept.append(row)
    return kept


def run_training(
    cfg: TrainConfig,
    *,
    resume: bool = False,
    ckpt: str | None = None,
    epoch: int | None = None,
    list_only: bool = False,
) -> Path | None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    out_dir = cfg.output_path()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    hist_path = out_dir / "history.json"
    scaler_path = out_dir / "scaler.json"

    if list_only:
        print(format_epoch_list(out_dir))
        return None

    want_resume = resume or ckpt is not None or epoch is not None
    resume_path = None
    if want_resume:
        resume_path = resolve_ckpt_path(
            out_dir, epoch=epoch, ckpt=ckpt, fallback_last=last_path, fallback_best=best_path
        )
    sequences = load_grid_sequences(cfg)
    train_seq, val_seq = split_sequences(sequences, cfg.val_ratio, cfg.seed)

    if resume_path is not None:
        if not scaler_path.exists():
            raise FileNotFoundError(f"续训需要 {scaler_path}，请不要删归一化文件")
        scaler = FeatureScaler.load(scaler_path)
        blob = torch.load(resume_path, map_location=device, weights_only=False)
        saved_scheme = str(blob.get("scheme", cfg.scheme)).upper()
        if saved_scheme != cfg.scheme.upper():
            raise ValueError(f"权重方案 {saved_scheme} 与当前 --scheme {cfg.scheme} 不一致")
        start_epoch = int(blob.get("epoch_done", 0))
        best_rmse = float(blob.get("best_rmse", float("inf")))
        history = _trim_history(
            json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else [],
            start_epoch,
        )
        print(
            f"从 {resume_path.name} 续训  (已完成电压 epoch={start_epoch}  "
            f"best RMSE={best_rmse*1e3:.2f} mV)"
        )
        print(format_epoch_list(out_dir))
    else:
        existing = latest_epoch(out_dir)
        if existing is not None:
            print(
                f"发现已有权重到 epoch {existing[0]}，本次仍从头训。"
                f"接着最新一轮：--resume    指定轮次：--epoch N"
            )
        elif last_path.exists() or best_path.exists():
            print(
                f"发现已有 {last_path.name if last_path.exists() else best_path.name}，"
                "本次仍从头训。接着训请加 --resume"
            )
        scaler = fit_scaler(train_seq)
        scaler.save(scaler_path)
        cfg.to_json(out_dir / "config.json")
        start_epoch = 0
        best_rmse = float("inf")
        history = []
        blob = None

    train_ds = TrajectoryDataset(train_seq, scaler)
    val_ds = TrajectoryDataset(val_seq, scaler) if val_seq else None

    train_loader = DataLoader(
        train_ds,
        batch_size=min(cfg.batch_size, len(train_ds)),
        shuffle=True,
        collate_fn=collate_traj,
    )
    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=min(cfg.batch_size, len(val_ds)),
            shuffle=False,
            collate_fn=collate_traj,
        )

    model = ParamMLP(cfg).to(device)
    params = [{"params": [p for n, p in model.named_parameters() if n != "phi"], "lr": cfg.lr}]
    if model.phi is not None:
        params.append({"params": [model.phi], "lr": cfg.lr_c1})
    opt = torch.optim.Adam(params, weight_decay=cfg.weight_decay)
    if blob is not None:
        model.load_state_dict(blob["model"])
        if "optimizer" in blob:
            opt.load_state_dict(blob["optimizer"])
            for group in opt.param_groups:
                if group.get("params") and group["params"][0] is model.phi:
                    group["lr"] = cfg.lr_c1
                else:
                    group["lr"] = cfg.lr

    print(
        f"scheme={cfg.scheme}  inputs={'true' if cfg.use_true_inputs else 'meas'}  "
        f"train={len(train_ds)}  val={len(val_ds) if val_ds else 0}  "
        f"device={device}  steps/traj={len(train_seq[0]['i'])}"
    )

    def _epoch(kind: str, epoch: int) -> dict[str, float]:
        model.train()
        acc = {"loss": 0.0, "rmse_v": 0.0, "l_v": 0.0, "l_s": 0.0}
        n = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            if kind == "pretrain":
                loss = teacher_loss(model, batch, cfg)
                stats = {"loss": float(loss.detach()), "rmse_v": float("nan"), "l_v": 0.0, "l_s": 0.0}
            else:
                loss, stats = voltage_loss(model, batch, cfg)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            for key, val in stats.items():
                if val == val:
                    acc[key] = acc.get(key, 0.0) + val
            n += 1
        train_stats = {k: v / max(n, 1) for k, v in acc.items()}
        val_stats = evaluate(model, val_loader, cfg, device) if val_loader else {}
        row = {"phase": kind, "epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}}
        row.update({f"val_{k}": v for k, v in val_stats.items()})
        return row

    if resume_path is None:
        for epoch in range(1, cfg.pretrain_epochs + 1):
            row = _epoch("pretrain", epoch)
            history.append(row)
            print(f"[pre {epoch:03d}/{cfg.pretrain_epochs}] loss={row['train_loss']:.4e}")
    elif cfg.pretrain_epochs > 0:
        print("续训跳过预热")

    epoch_done = start_epoch
    for local_epoch in range(1, cfg.epochs + 1):
        epoch_done = start_epoch + local_epoch
        row = _epoch("voltage", epoch_done)
        history.append(row)
        val_rmse = row.get("val_rmse_v", row["train_rmse_v"])
        mark = ""
        payload = _ckpt_payload(model, opt, cfg, best_rmse, epoch_done)
        ep_file = epoch_path(out_dir, epoch_done)
        ep_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, ep_file)
        torch.save(payload, last_path)
        write_latest_pointer(out_dir, epoch_done)
        if val_rmse == val_rmse and val_rmse < best_rmse:
            best_rmse = val_rmse
            payload["best_rmse"] = best_rmse
            torch.save(payload, best_path)
            mark = "  *"
        print(
            f"[{epoch_done:03d} +{local_epoch}/{cfg.epochs}] "
            f"train RMSE={row['train_rmse_v']*1e3:.2f} mV  "
            f"val RMSE={row.get('val_rmse_v', float('nan'))*1e3:.2f} mV{mark}"
        )

    torch.save(_ckpt_payload(model, opt, cfg, best_rmse, epoch_done), last_path)
    if not best_path.exists():
        torch.save(_ckpt_payload(model, opt, cfg, best_rmse, epoch_done), best_path)

    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(
        f"已保存 {best_path}  (best val RMSE={best_rmse*1e3:.2f} mV  累计电压 epoch={epoch_done})"
    )
    print(format_epoch_list(out_dir))
    return best_path


def parse_args() -> tuple[TrainConfig, argparse.Namespace]:
    p = argparse.ArgumentParser(description="训练 MLP-ECM 灰箱模型")
    p.add_argument("--scheme", default="B", choices=["A", "B", "B+", "a", "b", "b+"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--pretrain-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tbptt", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-pretrain", action="store_true")
    p.add_argument("--resume", action="store_true", help="从已有权重接着训，默认最新 epoch")
    p.add_argument("--epoch", type=int, default=None, help="从该电压 epoch 接着训，例如 --epoch 12")
    p.add_argument("--ckpt", default=None, help="直接指定权重文件")
    p.add_argument("--fresh", action="store_true", help="忽略已有权重，强制从头训")
    p.add_argument("--list-ckpts", action="store_true", help="列出已保存的 epoch 后退出")
    p.add_argument(
        "--use-meas-inputs",
        action="store_true",
        help="网络输入用 i_meas/soc_meas/t_meas（任务 D）。默认仍是真值列",
    )
    args = p.parse_args()

    cfg = TrainConfig()
    cfg.scheme = args.scheme.upper()
    if args.use_meas_inputs:
        cfg.use_true_inputs = False
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.pretrain_epochs is not None:
        cfg.pretrain_epochs = args.pretrain_epochs
    if args.no_pretrain:
        cfg.pretrain_epochs = 0
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.tbptt is not None:
        cfg.tbptt = args.tbptt
    if args.device is not None:
        cfg.device = args.device
    elif torch.cuda.is_available():
        cfg.device = "cuda"
    return cfg, args


def main() -> None:
    cfg, args = parse_args()
    if args.list_ckpts:
        run_training(cfg, list_only=True)
        return
    resume = bool(args.resume or args.ckpt or args.epoch) and not args.fresh
    run_training(cfg, resume=resume, ckpt=args.ckpt, epoch=args.epoch)


if __name__ == "__main__":
    main()
