"""闭环：EKF 估 SOC，MLP–ECM 预测端电压。

用法（仓库根目录）：

    python Src/AI/KF/run.py
    python Src/AI/KF/run.py --soc-error 0.05 --current-bias 5
    python Src/AI/KF/run.py --selftest
    python Src/AI/KF/run.py --csv Data/nmc100ah_ecm_sim.csv --export Data/ai_kf/logs/sim.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

KF_DIR = Path(__file__).resolve().parent
AI_DIR = KF_DIR.parent
PLOT_DIR = KF_DIR.parent.parent / "Plot"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

from KF.adapter import MlpParamProvider
from KF.config import KfConfig, REPO_ROOT
from KF.ekf import selftest as ekf_selftest
from KF.filter import filter_metrics, run_filter
from KF.gate import gate_log
from MLP.ckpt import format_epoch_list, resolve_ckpt_path
from MLP.dataset import FeatureScaler, _load_csv


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


def write_log_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    skip = set()
    keys = [k for k, v in data.items() if k not in skip and hasattr(v, "shape") and v.ndim == 1]
    n = len(data["time_s"])
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(keys)
        for i in range(n):
            row = []
            for name in keys:
                val = data[name][i]
                if isinstance(val, (np.floating, float)):
                    row.append(f"{float(val):.8g}")
                else:
                    row.append(val)
            writer.writerow(row)


def plot_filter(data: dict[str, np.ndarray], fig_path: Path, *, title: str, stats: dict[str, float], show: bool) -> Path:
    from matplotlib import pyplot as plt

    from _common import apply_style, plot_overlay, save_figure

    apply_style()
    t = data["time_s"]
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(12.5, 11.5), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    ax = axes[0]
    if "soc_true" in data:
        plot_overlay(ax, t, data["soc_true"], "truth", color="#616161", label="真值")
    plot_overlay(ax, t, data["soc_ah"], "ah", color="#64b5f6", label="安时")
    plot_overlay(ax, t, data["soc_post"], "ekf", color="#0d47a1", label="EKF $s^+$")
    ax.set_ylabel("SOC")
    ax.legend(loc="upper right", ncol=3)
    if "s_post_rmse" in stats:
        ax.set_title(
            f"SOC RMSE  安时 {stats['s_ah_rmse']*100:.3f} pp   "
            f"EKF {stats['s_post_rmse']*100:.3f} pp   "
            f"终点误差 安时 {stats['s_end_ah_err']*100:+.3f} / EKF {stats['s_end_post_err']*100:+.3f} pp",
            loc="left",
            fontsize=9,
        )

    ax = axes[1]
    plot_overlay(ax, t, data["u_t_meas_v"], "meas", color="#e57373", label="测量")
    plot_overlay(ax, t, data["u_t_ol"], "ol", color="#5d4037", label="开环 ECM")
    plot_overlay(ax, t, data["u_t_pri_v"], "pri", color="#b71c1c", label="先验 $\\hat U_t^-$")
    ax.set_ylabel("电压 / V")
    ax.legend(loc="upper right", ncol=3)
    ax.set_title(
        f"电压 RMSE  开环 {stats['e_ol_rmse_mV']:.2f}  先验 {stats['e_pri_rmse_mV']:.2f}  "
        f"后验 {stats['e_post_rmse_mV']:.2f} mV   （增量只用开环）",
        loc="left",
        fontsize=9,
    )

    ax = axes[2]
    ax.axhline(0.0, color="#9e9e9e", lw=0.8, zorder=1)
    plot_overlay(ax, t, data["e_ol"] * 1e3, "ol", color="#5d4037", label="$e^{ol}$")
    plot_overlay(ax, t, data["e_pri"] * 1e3, "pri", color="#b71c1c", label="$e^{pri}$")
    plot_overlay(ax, t, data["e_post"] * 1e3, "post", color="#00695c", label="$e^{post}$")
    ax.set_ylabel("电压误差 / mV")
    ax.legend(loc="upper right", ncol=3)

    ax = axes[3]
    plot_overlay(ax, t, data["r0_ohm"] * 1e3, "truth", color="#1565c0", label="$R_0$")
    plot_overlay(ax, t, data["r1_ohm"] * 1e3, "ol", color="#ef6c00", label="$R_1$")
    ax.set_ylabel("电阻 / mΩ")
    ax.legend(loc="upper right", ncol=2)

    ax = axes[4]
    ax.axhline(1.0, color="#9e9e9e", ls="--", lw=0.9, zorder=1)
    plot_overlay(ax, t, data["nis"], "truth", color="#4527a0", label="NIS")
    ax.set_ylabel("NIS")
    ax.set_xlabel("时间 / s")
    ax.set_title(f"NIS 中位 {stats['nis_median']:.2f}  均值 {stats['nis_mean']:.2f}", loc="left", fontsize=9)
    ax.set_ylim(0.0, max(8.0, float(np.quantile(data["nis"], 0.98)) * 1.2))

    for a in axes:
        a.margins(x=0.01)
    return save_figure(fig, fig_path, show=show)


def print_metrics(ckpt: Path, csv_path: Path, stats: dict[str, float], gate_txt: str) -> None:
    print(f"权重  {ckpt}")
    print(f"输入  {csv_path}")
    print(
        f"电压  开环 RMSE={stats['e_ol_rmse_mV']:.2f} mV  "
        f"先验={stats['e_pri_rmse_mV']:.2f} mV  "
        f"后验={stats['e_post_rmse_mV']:.2f} mV"
    )
    if "s_post_rmse" in stats:
        print(
            f"SOC   安时 RMSE={stats['s_ah_rmse']*100:.3f} pp  "
            f"EKF RMSE={stats['s_post_rmse']*100:.3f} pp  "
            f"终点 安时 {stats['s_end_ah_err']*100:+.3f} / EKF {stats['s_end_post_err']*100:+.3f} pp"
        )
    print(f"NIS   中位={stats['nis_median']:.2f}  均值={stats['nis_mean']:.2f}")
    print(gate_txt)


def run_one(args: argparse.Namespace) -> dict[str, np.ndarray]:
    mlp_dir = _resolve(args.mlp_dir)
    ckpt = _pick_ckpt(mlp_dir, epoch=args.epoch, ckpt=args.ckpt, best=args.best)
    csv_path = _resolve(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 {csv_path}，请先运行 python Src/Sim/nmc100ah_ecm_gen.py")

    kf_cfg = KfConfig(
        dt_s=args.dt,
        capacity_ah=args.capacity_ah * args.capacity_scale,
        rv_std=args.rv_std,
        estimate_dr0=args.dr0,
        schedule_rv=not args.no_schedule,
    )
    provider, _mlp_cfg = MlpParamProvider.from_dir(
        mlp_dir,
        ckpt=ckpt,
        device=args.device or "cpu",
        r0_scale=args.r0_scale,
        r1_scale=args.r1_scale,
    )
    raw = _load_csv(csv_path)
    i_key = "i_true_a" if args.use_true_current else ("i_meas_a" if "i_meas_a" in raw else "i_true_a")
    t_key = "t_true_c" if args.use_true_temp else ("t_meas_c" if "t_meas_c" in raw else "t_true_c")
    u_key = "u_t_meas_v" if "u_t_meas_v" in raw else "u_t_true_v"
    soc_true = raw["soc_true"] if "soc_true" in raw else None
    soc0 = args.soc0
    if soc0 is None and soc_true is not None:
        # 仿真表 soc_true[0] 已是第一步安时之后，用它减回一步更接近开机值
        soc0 = float(soc_true[0])
        if i_key in raw:
            q = kf_cfg.q_coulomb
            soc0 = float(np.clip(soc0 + float(raw[i_key][0]) * kf_cfg.dt_s / q, 0.0, 1.0))
    log = run_filter(
        provider,
        raw[i_key],
        raw[t_key],
        raw[u_key],
        cfg=kf_cfg,
        s0=soc0,
        u_p0=args.up0,
        soc_error=args.soc_error,
        current_bias=args.current_bias,
        time_s=raw.get("time_s"),
        soc_true=soc_true,
    )
    stats = filter_metrics(log)
    scaler = FeatureScaler.load(mlp_dir / "scaler.json")
    gate = gate_log(log, scaler, kf_cfg)
    gate_txt = "门控  通过" if gate.accepted else "门控  拒绝：" + "；".join(gate.reasons)
    print_metrics(ckpt, csv_path, stats, gate_txt)

    out_csv = _resolve(args.out)
    write_log_csv(out_csv, log)
    print(f"表    {out_csv}")
    if args.export:
        exp = _resolve(args.export)
        write_log_csv(exp, log)
        print(f"增量日志  {exp}")

    if not args.no_plot:
        fig = plot_filter(
            log,
            _resolve(args.fig),
            title=f"EKF+MLP-ECM  {ckpt.name}  bias={args.current_bias:g}A  ds0={args.soc_error:g}",
            stats=stats,
            show=args.show,
        )
        print(f"图    {fig}")

    meta = {"ckpt": str(ckpt), "csv": str(csv_path), "stats": stats, "gate": gate.stats, "accepted": gate.accepted}
    meta_path = _resolve(args.out).with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EKF 估 SOC + MLP-ECM 预测端电压")
    p.add_argument("--selftest", action="store_true", help="只跑 EKF 静置纠偏自检")
    p.add_argument("--mlp-dir", default="Data/ai_mlp")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--best", action="store_true")
    p.add_argument("--list-ckpts", action="store_true")
    p.add_argument("--csv", default="Data/nmc100ah_ecm_sim.csv")
    p.add_argument("--out", default="Data/ai_kf/filter.csv")
    p.add_argument("--export", default=None, help="再写一份增量用日志")
    p.add_argument("--fig", default="Fig/kf_ecm_filter.png")
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--soc0", type=float, default=None)
    p.add_argument("--soc-error", type=float, default=0.0, help="加在 SOC0 上的偏差，演示用")
    p.add_argument("--up0", type=float, default=0.0)
    p.add_argument("--current-bias", type=float, default=0.0, help="电流零偏 / A，放电为正")
    p.add_argument("--capacity-ah", type=float, default=100.0)
    p.add_argument("--capacity-scale", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--rv-std", type=float, default=0.5e-3)
    p.add_argument("--dr0", action="store_true", help="附加慢变 δR0")
    p.add_argument("--no-schedule", action="store_true")
    p.add_argument("--r0-scale", type=float, default=1.0, help="故意把 MLP 的 R0 放大，验收用")
    p.add_argument("--r1-scale", type=float, default=1.0)
    p.add_argument("--use-true-current", action="store_true")
    p.add_argument("--use-true-temp", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.selftest:
        out = ekf_selftest()
        print(f"EKF 自检通过  s_post={out['s_post']:.4f}  e_pri={out['e_pri']*1e3:.2f} mV")
        return
    mlp_dir = _resolve(args.mlp_dir)
    if args.list_ckpts:
        print(format_epoch_list(mlp_dir))
        return
    run_one(args)


if __name__ == "__main__":
    main()
