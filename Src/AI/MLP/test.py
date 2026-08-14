"""在单条仿真轨迹上测试已训 MLP-ECM。

默认用最新权重和 Data/nmc100ah_ecm_sim.csv，画出：
  - 预测电压 vs 测量电压
  - 电压误差
  - R0 / R1 预测 vs 真值

用法（仓库根目录）：

    python Src/AI/MLP/test.py
    python Src/AI/MLP/test.py --epoch 400
    python Src/AI/MLP/test.py --ckpt Data/ai_mlp/best.pt --show
    python Src/AI/MLP/test.py --list-ckpts
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

MLP_DIR = Path(__file__).resolve().parent
AI_DIR = MLP_DIR.parent
PLOT_DIR = MLP_DIR.parent.parent / "Plot"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

from MLP.ckpt import format_epoch_list, resolve_ckpt_path
from MLP.config import REPO_ROOT
from MLP.dataset import _load_csv
from MLP.infer import infer_csv, load_bundle

from _common import apply_style, mode_spans, save_figure

MODE_FACE = {
    "rest": (0.72, 0.72, 0.72, 0.18),
    "discharge": (0.86, 0.32, 0.24, 0.16),
    "charge": (0.24, 0.48, 0.82, 0.16),
}
MODE_LABEL = {"rest": "静置", "discharge": "放电", "charge": "充电"}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _pick_ckpt(out_dir: Path, *, epoch: int | None, ckpt: str | None, best: bool) -> Path:
    if best:
        path = out_dir / "best.pt"
        if not path.exists():
            raise FileNotFoundError(f"找不到 {path}")
        return path
    return resolve_ckpt_path(
        out_dir,
        epoch=epoch,
        ckpt=ckpt,
        fallback_last=out_dir / "last.pt",
        fallback_best=out_dir / "best.pt",
    )


def _finite(arr: np.ndarray) -> bool:
    return np.isfinite(arr).any()


def attach_extras(data: dict[str, np.ndarray], csv_path: Path) -> dict[str, np.ndarray]:
    raw = _load_csv(csv_path)
    if "mode" in raw:
        data["mode"] = raw["mode"]
    data["u_err_v"] = data["u_t_hat_v"] - data["u_t_meas_v"]
    data["r0_err_ohm"] = data["r0_hat_ohm"] - data["r0_ohm"]
    data["r1_err_ohm"] = data["r1_hat_ohm"] - data["r1_ohm"]
    return data


def metrics(data: dict[str, np.ndarray]) -> dict[str, float]:
    err = data["u_err_v"]
    out = {
        "u_rmse_mV": float(np.sqrt(np.mean(err**2)) * 1e3),
        "u_mae_mV": float(np.mean(np.abs(err)) * 1e3),
        "u_max_mV": float(np.max(np.abs(err)) * 1e3),
    }
    if _finite(data["r0_ohm"]):
        r0e = data["r0_err_ohm"] * 1e3
        out["r0_rmse_mohm"] = float(np.sqrt(np.nanmean(r0e**2)))
        out["r0_max_mohm"] = float(np.nanmax(np.abs(r0e)))
    if _finite(data["r1_ohm"]):
        r1e = data["r1_err_ohm"] * 1e3
        out["r1_rmse_mohm"] = float(np.sqrt(np.nanmean(r1e**2)))
        out["r1_max_mohm"] = float(np.nanmax(np.abs(r1e)))
    return out


def write_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "time_s",
        "i_a",
        "u_t_meas_v",
        "u_t_hat_v",
        "u_err_v",
        "r0_ohm",
        "r0_hat_ohm",
        "r0_err_ohm",
        "r1_ohm",
        "r1_hat_ohm",
        "r1_err_ohm",
    ]
    n = len(data["time_s"])
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(keys)
        for k in range(n):
            writer.writerow([f"{float(data[name][k]):.8g}" for name in keys])


def _shade_modes(axes, time_s: np.ndarray, modes: np.ndarray | None) -> None:
    if modes is None:
        return
    for ax in axes:
        for mode, t0, t1 in mode_spans(time_s, modes):
            ax.axvspan(t0, t1, color=MODE_FACE.get(mode, (0.8, 0.8, 0.8, 0.12)), lw=0)


def plot_test(
    data: dict[str, np.ndarray],
    fig_path: Path,
    *,
    title: str,
    stats: dict[str, float],
    show: bool,
) -> Path:
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch

    apply_style()
    t = data["time_s"]
    modes = data.get("mode")

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12.5, 10.0), constrained_layout=True)
    fig.suptitle(title, fontsize=13)
    _shade_modes(axes, t, modes)

    axes[0].plot(t, data["u_t_meas_v"], color="#e08a7a", lw=0.7, alpha=0.85, label="测量 $U_t$")
    axes[0].plot(t, data["u_t_hat_v"], color="#9c2a2a", lw=1.15, label="预测 $U_t$")
    axes[0].set_ylabel("电压 / V")
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].set_title(
        f"电压 RMSE {stats['u_rmse_mV']:.2f} mV   "
        f"MAE {stats['u_mae_mV']:.2f} mV   "
        f"MAX {stats['u_max_mV']:.2f} mV",
        loc="left",
        fontsize=9,
    )

    axes[1].axhline(0.0, color="#9e9e9e", lw=0.8)
    axes[1].plot(t, data["u_err_v"] * 1e3, color="#4e342e", lw=0.85)
    axes[1].set_ylabel("电压误差 / mV")

    axes[2].plot(t, data["r0_ohm"] * 1e3, color="#90caf9", lw=0.9, label="真值 $R_0$")
    axes[2].plot(t, data["r0_hat_ohm"] * 1e3, color="#1565c0", lw=1.15, label="预测 $R_0$")
    axes[2].set_ylabel("$R_0$ / mΩ")
    axes[2].legend(loc="upper right", ncol=2)
    if "r0_rmse_mohm" in stats:
        axes[2].set_title(
            f"$R_0$ RMSE {stats['r0_rmse_mohm']:.3f} mΩ   MAX {stats['r0_max_mohm']:.3f} mΩ",
            loc="left",
            fontsize=9,
        )

    axes[3].plot(t, data["r1_ohm"] * 1e3, color="#ffcc80", lw=0.9, label="真值 $R_1$")
    axes[3].plot(t, data["r1_hat_ohm"] * 1e3, color="#ef6c00", lw=1.15, label="预测 $R_1$")
    axes[3].set_ylabel("$R_1$ / mΩ")
    axes[3].set_xlabel("时间 / s")
    axes[3].legend(loc="upper right", ncol=2)
    if "r1_rmse_mohm" in stats:
        axes[3].set_title(
            f"$R_1$ RMSE {stats['r1_rmse_mohm']:.3f} mΩ   MAX {stats['r1_max_mohm']:.3f} mΩ",
            loc="left",
            fontsize=9,
        )

    if modes is not None:
        mode_handles = [
            Patch(facecolor=MODE_FACE[k], edgecolor="none", label=MODE_LABEL[k])
            for k in ("rest", "discharge", "charge")
        ]
        line_handles, line_labels = axes[0].get_legend_handles_labels()
        axes[0].legend(
            line_handles + mode_handles,
            line_labels + [MODE_LABEL[k] for k in ("rest", "discharge", "charge")],
            loc="upper right",
            ncol=5,
        )

    for ax in axes:
        ax.margins(x=0.01)

    return save_figure(fig, fig_path, show=show)


def print_metrics(ckpt: Path, csv_path: Path, stats: dict[str, float]) -> None:
    print(f"权重  {ckpt}")
    print(f"输入  {csv_path}")
    print(
        f"电压  RMSE={stats['u_rmse_mV']:.2f} mV  "
        f"MAE={stats['u_mae_mV']:.2f} mV  "
        f"MAX={stats['u_max_mV']:.2f} mV"
    )
    if "r0_rmse_mohm" in stats:
        print(f"R0    RMSE={stats['r0_rmse_mohm']:.4f} mΩ  MAX={stats['r0_max_mohm']:.4f} mΩ")
    if "r1_rmse_mohm" in stats:
        print(f"R1    RMSE={stats['r1_rmse_mohm']:.4f} mΩ  MAX={stats['r1_max_mohm']:.4f} mΩ")


def main() -> None:
    parser = argparse.ArgumentParser(description="用仿真 CSV 测试 MLP-ECM 权重")
    parser.add_argument("--ckpt", default=None, help="权重文件；与 --epoch 二选一")
    parser.add_argument("--epoch", type=int, default=None, help="用该 epoch 的权重，默认最新")
    parser.add_argument("--best", action="store_true", help="改用 best.pt")
    parser.add_argument("--out-dir", default="Data/ai_mlp")
    parser.add_argument("--csv", default="Data/nmc100ah_ecm_sim.csv")
    parser.add_argument("--out", default="Data/ai_mlp/test.csv")
    parser.add_argument("--fig", default="Fig/mlp_ecm_test.png")
    parser.add_argument("--show", action="store_true", help="弹窗显示")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--list-ckpts", action="store_true")
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir)
    if args.list_ckpts:
        print(format_epoch_list(out_dir))
        return

    csv_path = _resolve(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到仿真数据: {csv_path}，请先运行 python Src/Sim/nmc100ah_ecm_gen.py")

    ckpt = _pick_ckpt(out_dir, epoch=args.epoch, ckpt=args.ckpt, best=args.best)
    model, scaler, cfg = load_bundle(ckpt, out_dir / "config.json", out_dir / "scaler.json")
    data = attach_extras(infer_csv(model, scaler, cfg, csv_path), csv_path)
    stats = metrics(data)

    write_csv(_resolve(args.out), data)
    print_metrics(ckpt, csv_path, stats)
    print(f"表    {_resolve(args.out)}")

    if not args.no_plot:
        fig_path = plot_test(
            data,
            _resolve(args.fig),
            title=f"MLP-ECM 测试  {ckpt.name}  scheme={cfg.scheme}",
            stats=stats,
            show=args.show,
        )
        print(f"图    {fig_path}")


if __name__ == "__main__":
    main()
