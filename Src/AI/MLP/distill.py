"""用已训 64×64 当教师，蒸馏到更小的学生网（默认 16×16）。

仓库根目录：

    python Src/AI/MLP/distill.py --teacher-dir Data/ai_mlp --hidden 16 16 --out-dir Data/ai_mlp_h16_kd --epochs 100

不覆盖 Data/ai_mlp。学生沿用教师的 scaler.json。
主损失：学生电阻对教师电阻（对数）。电压损失作辅助，避免只拟合 R 却不管 ECM。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

MLP_DIR = Path(__file__).resolve().parent
AI_DIR = MLP_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from MLP.ckpt import epoch_path, format_epoch_list, write_latest_pointer
from MLP.config import TrainConfig
from MLP.dataset import TrajectoryDataset, collate_traj, load_grid_sequences, split_sequences
from MLP.ecm import ecm_forward
from MLP.infer import load_bundle
from MLP.model import ParamMLP
from MLP.train import _ckpt_payload, set_seed, voltage_loss


def r_distill_loss(student: ParamMLP, teacher: ParamMLP, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    r0s, r1s, _ = student(batch["x"])
    with torch.no_grad():
        r0t, r1t, _ = teacher(batch["x"])
    l0 = (torch.log(r0s.clamp_min(1e-12)) - torch.log(r0t.clamp_min(1e-12))).pow(2).mean()
    l1 = (torch.log(r1s.clamp_min(1e-12)) - torch.log(r1t.clamp_min(1e-12))).pow(2).mean()
    return 0.5 * (l0 + l1)


@torch.no_grad()
def eval_pack(
    student: ParamMLP,
    teacher: ParamMLP,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    student.eval()
    tot = {"u": 0.0, "r0_t": 0.0, "r1_t": 0.0, "r0_csv": 0.0, "r1_csv": 0.0, "n": 0}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        r0s, r1s, c1 = student(batch["x"])
        r0t, r1t, _ = teacher(batch["x"])
        u_hat, _ = ecm_forward(batch["i"], batch["u_ocv"], r0s, r1s, c1, dt_s=cfg.dt_s)
        n = int(u_hat.numel())
        tot["u"] += float((u_hat - batch["u_t"]).pow(2).sum())
        tot["r0_t"] += float((r0s - r0t).pow(2).sum())
        tot["r1_t"] += float((r1s - r1t).pow(2).sum())
        tot["r0_csv"] += float((r0s - batch["r0"]).pow(2).sum())
        tot["r1_csv"] += float((r1s - batch["r1"]).pow(2).sum())
        tot["n"] += n
    n = max(tot["n"], 1)
    return {
        "val_rmse_v": (tot["u"] / n) ** 0.5,
        "val_r0_vs_teacher": (tot["r0_t"] / n) ** 0.5,
        "val_r1_vs_teacher": (tot["r1_t"] / n) ** 0.5,
        "val_r0_vs_csv": (tot["r0_csv"] / n) ** 0.5,
        "val_r1_vs_csv": (tot["r1_csv"] / n) ** 0.5,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="64×64 → 小网蒸馏")
    p.add_argument("--teacher-dir", default="Data/ai_mlp")
    p.add_argument("--out-dir", default="Data/ai_mlp_h16_kd")
    p.add_argument("--hidden", type=int, nargs="+", default=[16, 16])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lambda-v", type=float, default=1.0, help="电压损失系数；电阻蒸馏系数为 1")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    teacher_dir = Path(args.teacher_dir)
    if not teacher_dir.is_absolute():
        from MLP.config import REPO_ROOT
        teacher_dir = REPO_ROOT / teacher_dir

    t_model, scaler, t_cfg = load_bundle(
        teacher_dir / "best.pt", teacher_dir / "config.json", teacher_dir / "scaler.json"
    )
    t_model.eval()
    for par in t_model.parameters():
        par.requires_grad_(False)

    cfg = TrainConfig()
    cfg.scheme = "B"
    cfg.data_dir = t_cfg.data_dir
    cfg.out_dir = args.out_dir
    cfg.hidden = tuple(args.hidden)
    cfg.epochs = args.epochs
    cfg.pretrain_epochs = 0
    cfg.use_true_inputs = t_cfg.use_true_inputs
    cfg.voltage_target = t_cfg.voltage_target
    cfg.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    out_dir = cfg.output_path()
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler.save(out_dir / "scaler.json")
    cfg.to_json(out_dir / "config.json")
    (out_dir / "teacher.json").write_text(
        json.dumps({"teacher_dir": str(teacher_dir), "lambda_v": args.lambda_v}, indent=2) + "\n",
        encoding="utf-8",
    )

    sequences = load_grid_sequences(cfg)
    train_seq, val_seq = split_sequences(sequences, cfg.val_ratio, cfg.seed)
    train_ds = TrajectoryDataset(train_seq, scaler)
    val_ds = TrajectoryDataset(val_seq, scaler)
    train_loader = DataLoader(
        train_ds, batch_size=min(cfg.batch_size, len(train_ds)), shuffle=True, collate_fn=collate_traj
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(cfg.batch_size, len(val_ds)), shuffle=False, collate_fn=collate_traj
    )

    student = ParamMLP(cfg).to(device)
    t_model = t_model.to(device)
    opt = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print(
        f"distill teacher={teacher_dir} hidden={cfg.hidden} "
        f"train={len(train_ds)} val={len(val_ds)} device={device} lambda_v={args.lambda_v}"
    )

    history: list[dict] = []
    best_rmse = float("inf")
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    for epoch in range(1, cfg.epochs + 1):
        student.train()
        acc = {"loss": 0.0, "rmse_v": 0.0, "l_r": 0.0, "l_v": 0.0}
        n_b = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            l_r = r_distill_loss(student, t_model, batch)
            l_v, stats = voltage_loss(student, batch, cfg)
            loss = l_r + args.lambda_v * l_v
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
            opt.step()
            acc["loss"] += float(loss.detach())
            acc["rmse_v"] += stats["rmse_v"]
            acc["l_r"] += float(l_r.detach())
            acc["l_v"] += float(stats["l_v"])
            n_b += 1
        train_rmse = acc["rmse_v"] / max(n_b, 1)
        val = eval_pack(student, t_model, val_loader, cfg, device)
        row = {
            "phase": "distill",
            "epoch": epoch,
            "train_rmse_v": train_rmse,
            "train_l_r": acc["l_r"] / max(n_b, 1),
            **val,
        }
        history.append(row)
        payload = _ckpt_payload(student, opt, cfg, val["val_rmse_v"], epoch)
        ep_file = epoch_path(out_dir, epoch)
        ep_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, ep_file)
        torch.save(payload, last_path)
        write_latest_pointer(out_dir, epoch)
        mark = ""
        if val["val_rmse_v"] < best_rmse:
            best_rmse = val["val_rmse_v"]
            torch.save(payload, best_path)
            mark = "  *"
        print(
            f"[{epoch:03d}/{cfg.epochs}] train {train_rmse*1e3:.2f} mV  "
            f"val {val['val_rmse_v']*1e3:.2f} mV  "
            f"R0vsT {val['val_r0_vs_teacher']*1e6:.1f} uOhm  "
            f"R1vsT {val['val_r1_vs_teacher']*1e6:.1f} uOhm{mark}"
        )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"已保存 {best_path}  (best val RMSE={best_rmse*1e3:.2f} mV)")
    print(format_epoch_list(out_dir))


if __name__ == "__main__":
    main()
