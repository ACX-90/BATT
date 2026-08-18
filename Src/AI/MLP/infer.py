"""用训练好的 MLP-ECM 对单条轨迹做电压与参数推理。

用法（仓库根目录）：

    python Src/AI/MLP/infer.py
    python Src/AI/MLP/infer.py --ckpt Data/ai_mlp/best.pt --csv Data/nmc100ah_ecm_sim.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

MLP_DIR = Path(__file__).resolve().parent
AI_DIR = MLP_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from MLP.ckpt import format_epoch_list, resolve_ckpt_path
from MLP.config import REPO_ROOT, TrainConfig
from MLP.dataset import FeatureScaler, _load_csv
from MLP.ecm import ecm_forward
from MLP.model import ParamMLP


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_bundle(ckpt: Path, cfg_path: Path, scaler_path: Path) -> tuple[ParamMLP, FeatureScaler, TrainConfig]:
    cfg = TrainConfig.from_dict(json_load(cfg_path))
    scaler = FeatureScaler.load(scaler_path)
    model = ParamMLP(cfg)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, scaler, cfg


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


@torch.no_grad()
def infer_csv(model: ParamMLP, scaler: FeatureScaler, cfg: TrainConfig, csv_path: Path) -> dict[str, np.ndarray]:
    raw = _load_csv(csv_path)
    i_key = "i_true_a" if cfg.use_true_inputs else "i_meas_a"
    soc_key = "soc_true" if cfg.use_true_inputs else "soc_meas"
    t_key = "t_true_c" if cfg.use_true_inputs else "t_meas_c"
    feat = np.stack([raw[i_key], raw[soc_key], raw[t_key]], axis=-1)
    x = torch.from_numpy(scaler.transform(feat).astype(np.float32)).unsqueeze(0)
    i = torch.from_numpy(raw[i_key].astype(np.float32)).unsqueeze(0)
    u_ocv = torch.from_numpy(raw["u_ocv_v"].astype(np.float32)).unsqueeze(0)
    # 调用MLP，其中ParamMLP类型继承nn.Module，此类型无__call__时自动调forward方法
    r0, r1, c1 = model(x)
    u_hat, u_p = ecm_forward(i, u_ocv, r0, r1, c1, dt_s=cfg.dt_s)
    return {
        "time_s": raw["time_s"],
        "i_a": raw[i_key],
        "soc": raw[soc_key],
        "t_c": raw[t_key],
        "u_ocv_v": raw["u_ocv_v"],
        "u_t_meas_v": raw.get("u_t_meas_v", raw["u_t_true_v"]),
        "u_t_true_v": raw.get("u_t_true_v", raw["u_t_meas_v"]),
        "u_t_hat_v": u_hat.squeeze(0).numpy(),
        "u_p_hat_v": u_p.squeeze(0).numpy(),
        "r0_hat_ohm": r0.squeeze(0).numpy(),
        "r1_hat_ohm": r1.squeeze(0).numpy(),
        "c1_hat_f": c1.squeeze(0).numpy(),
        "r0_ohm": raw.get("r0_ohm", np.full_like(raw["time_s"], np.nan)),
        "r1_ohm": raw.get("r1_ohm", np.full_like(raw["time_s"], np.nan)),
        "c1_f": raw.get("c1_f", np.full_like(raw["time_s"], np.nan)),
    }


def write_infer_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(data.keys())
    n = len(data["time_s"])
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(keys)
        for k in range(n):
            writer.writerow([f"{float(data[name][k]):.8g}" for name in keys])


def maybe_plot(data: dict[str, np.ndarray], fig_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    t = data["time_s"]
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 8), constrained_layout=True)
    fig.suptitle("MLP-ECM 推理")
    axes[0].plot(t, data["u_t_meas_v"], color="#e08a7a", lw=0.7, label="测量")
    axes[0].plot(t, data["u_t_hat_v"], color="#9c2a2a", lw=1.1, label="MLP-ECM")
    axes[0].set_ylabel("电压 / V")
    axes[0].legend(loc="upper right")
    axes[1].plot(t, data["r0_ohm"] * 1e3, color="#90caf9", lw=0.8, label="教师 $R_0$")
    axes[1].plot(t, data["r0_hat_ohm"] * 1e3, color="#1565c0", lw=1.1, label="估计 $R_0$")
    axes[1].plot(t, data["r1_ohm"] * 1e3, color="#ffcc80", lw=0.8, label="教师 $R_1$")
    axes[1].plot(t, data["r1_hat_ohm"] * 1e3, color="#ef6c00", lw=1.1, label="估计 $R_1$")
    axes[1].set_ylabel("电阻 / mΩ")
    axes[1].legend(loc="upper right", ncol=2)
    axes[2].plot(t, data["c1_f"] * 1e-3, color="#80cbc4", lw=0.8, label="教师 $C_1$")
    axes[2].plot(t, data["c1_hat_f"] * 1e-3, color="#00838f", lw=1.1, label="估计 $C_1$")
    axes[2].set_ylabel("$C_1$ / kF")
    axes[2].legend(loc="upper right")
    err_mV = (data["u_t_hat_v"] - data["u_t_meas_v"]) * 1e3
    axes[3].plot(t, err_mV, color="#4e342e", lw=0.8)
    axes[3].set_ylabel("电压误差 / mV")
    axes[3].set_xlabel("时间 / s")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLP-ECM 单条轨迹推理")
    parser.add_argument("--ckpt", default=None, help="权重文件；与 --epoch 二选一")
    parser.add_argument("--epoch", type=int, default=None, help="用该 epoch 的权重，默认最新")
    parser.add_argument("--out-dir", default="Data/ai_mlp")
    parser.add_argument("--best", action="store_true", help="改用 best.pt，而不是最新 epoch")
    parser.add_argument("--csv", default="Data/nmc100ah_ecm_sim.csv")
    parser.add_argument("--out", default="Data/ai_mlp/infer.csv")
    parser.add_argument("--fig", default="Fig/mlp_ecm_infer.png")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--list-ckpts", action="store_true")
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir)
    if args.list_ckpts:
        print(format_epoch_list(out_dir))
        return
    if args.best:
        ckpt = out_dir / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"找不到 {ckpt}")
    else:
        ckpt = resolve_ckpt_path(
            out_dir,
            epoch=args.epoch,
            ckpt=args.ckpt,
            fallback_last=out_dir / "last.pt",
            fallback_best=out_dir / "best.pt",
        )
    cfg_path = out_dir / "config.json"
    scaler_path = out_dir / "scaler.json"
    print(f"使用权重 {ckpt}")

    model, scaler, cfg = load_bundle(ckpt, cfg_path, scaler_path)
    data = infer_csv(model, scaler, cfg, _resolve(args.csv))
    err = data["u_t_hat_v"] - data["u_t_meas_v"]
    write_infer_csv(_resolve(args.out), data)
    print(f"写出 {_resolve(args.out)}")
    print(f"电压 RMSE={np.sqrt(np.mean(err**2))*1e3:.2f} mV  MAX={np.max(np.abs(err))*1e3:.2f} mV")
    if not args.no_plot:
        fig = _resolve(args.fig)
        maybe_plot(data, fig)
        print(f"图    {fig}")


if __name__ == "__main__":
    main()
