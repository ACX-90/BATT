"""第 1 期开环电压 + EKF SOC：不只报 RMSE，画出边沿 / 回弹 / 稳态残差。

仓库根目录：

    python Src/AI/EV_Local/plot.py
    python Src/AI/EV_Local/plot.py --exp a
    python Src/AI/EV_Local/plot.py --exp kgrid --ckpt Data/ai_local/kgrid_k115_p4/last.pt
    python Src/AI/EV_Local/plot.py --csv Data/soh_k115/nmc100ah_ecm_s02_t02_soc050_T+20.csv --show
    python Src/AI/EV_Local/plot.py --no-kf

默认叠冻结（k=1）、1a 全局 k、1b 网格（若盘上有 last.pt）。新年份走测量列，与 window.py 一致。
滤波口径与 KF/run.py 相同：测量 I/T/U，电阻锁在预测 SOC。增量仍只用开环。
不覆盖 Data/ai_mlp，不读旧网格训练。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

EV_DIR = Path(__file__).resolve().parent
AI_DIR = EV_DIR.parent
PLOT_DIR = EV_DIR.parent.parent / "Plot"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

from KF.adapter import KGridAdapter, MlpParamProvider, ScaleAdapter
from KF.config import KfConfig, REPO_ROOT
from KF.filter import filter_metrics, run_filter
from MLP.dataset import FeatureScaler, _load_csv
from MLP.ecm import ecm_forward
from MLP.infer import load_bundle
from MLP.model import ParamMLP

from _common import apply_style, mode_spans, plot_overlay, save_figure

MODE_FACE = {
    "rest": (0.72, 0.72, 0.72, 0.18),
    "discharge": (0.86, 0.32, 0.24, 0.16),
    "charge": (0.24, 0.48, 0.82, 0.16),
}
MODE_LABEL = {"rest": "静置", "discharge": "放电", "charge": "充电"}

PREFERRED = (
    ("cold", "soc050_T-10"),
    ("mid", "soc050_T+20"),
    ("hot", "soc050_T+50"),
)
# 05-d §1 任务 A 一遍；盘上没有 window_k115/last.pt 时用这对
K1A_FALLBACK = (1.193, 1.170)
WINDOW_DEFAULT = "Data/ai_local/window_k115/last.pt"
KGRID_DEFAULT = "Data/ai_local/kgrid_k115_p4/last.pt"

STYLE = {
    "freeze": {"kind": "ol", "color": "#5d4037", "label": "冻结 $k=1$"},
    "a": {"kind": "pri", "color": "#0d47a1", "label": "1a 全局 $k$"},
    "b": {"kind": "post", "color": "#00695c", "label": "1b $k$ 网格"},
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _rmse_mV(err_v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err_v))) * 1e3)


def _max_mV(err_v: np.ndarray) -> float:
    return float(np.max(np.abs(err_v)) * 1e3)


def _shade_modes(axes, time_s: np.ndarray, modes: np.ndarray | None) -> None:
    if modes is None:
        return
    for ax in axes:
        for mode, t0, t1 in mode_spans(time_s, modes):
            ax.axvspan(t0, t1, color=MODE_FACE.get(mode, (0.8, 0.8, 0.8, 0.12)), lw=0)


def _legend_with_modes(ax) -> None:
    from matplotlib.patches import Patch

    line_handles, line_labels = ax.get_legend_handles_labels()
    mode_handles = [
        Patch(facecolor=MODE_FACE[k], edgecolor="none", label=MODE_LABEL[k])
        for k in ("rest", "discharge", "charge")
    ]
    ax.legend(
        line_handles + mode_handles,
        line_labels + [MODE_LABEL[k] for k in ("rest", "discharge", "charge")],
        loc="upper right",
        ncol=min(6, len(line_handles) + 3),
    )


def pick_csvs(new_dir: Path, csv: str | None) -> list[tuple[str, Path]]:
    if csv:
        path = _resolve(csv)
        if not path.exists():
            raise FileNotFoundError(path)
        return [("trace", path)]
    files = sorted(p for p in new_dir.glob("*.csv") if p.name != "index.csv")
    out: list[tuple[str, Path]] = []
    for tag, key in PREFERRED:
        hits = [p for p in files if key in p.name]
        if hits:
            out.append((tag, hits[0]))
    if not out and files:
        out.append(("trace", files[0]))
    if not out:
        raise FileNotFoundError(f"{new_dir} 里没有仿真 CSV")
    return out


def load_adapted(
    base: ParamMLP,
    *,
    ckpt: Path | None,
    k0: float | None,
    k1: float | None,
) -> torch.nn.Module:
    if ckpt is not None and ckpt.exists():
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        mode = str(blob.get("incr_mode", ""))
        log0 = blob.get("log_k0")
        if mode == "window_kgrid" or (
            log0 is not None and torch.as_tensor(log0).ndim == 2
        ):
            wrap = KGridAdapter(base)
            wrap.log_k0.data.copy_(torch.as_tensor(blob["log_k0"], dtype=wrap.log_k0.dtype))
            wrap.log_k1.data.copy_(torch.as_tensor(blob["log_k1"], dtype=wrap.log_k1.dtype))
            return wrap.eval()
        return ScaleAdapter.from_blob(base, blob).eval()
    wrap = ScaleAdapter(base)
    if k0 is not None:
        wrap.log_k0.data.fill_(float(math.log(max(float(k0), 1e-12))))
        wrap.log_k1.data.fill_(float(math.log(max(float(k1 if k1 is not None else 1.0), 1e-12))))
    return wrap.eval()


@torch.no_grad()
def infer_csv(
    model: torch.nn.Module,
    scaler: FeatureScaler,
    cfg,
    csv_path: Path,
    *,
    use_true_inputs: bool,
) -> dict[str, np.ndarray]:
    raw = _load_csv(csv_path)
    i_key = "i_true_a" if use_true_inputs else "i_meas_a"
    soc_key = "soc_true" if use_true_inputs else "soc_meas"
    t_key = "t_true_c" if use_true_inputs else "t_meas_c"
    if i_key not in raw:
        i_key = "i_true_a" if "i_true_a" in raw else "i_meas_a"
    if soc_key not in raw:
        soc_key = "soc_true" if "soc_true" in raw else "soc_meas"
    if t_key not in raw:
        t_key = "t_true_c" if "t_true_c" in raw else "t_meas_c"
    feat = np.stack([raw[i_key], raw[soc_key], raw[t_key]], axis=-1)
    x = torch.from_numpy(scaler.transform(feat).astype(np.float32))
    i = torch.from_numpy(raw[i_key].astype(np.float32))
    soc = torch.from_numpy(raw[soc_key].astype(np.float32))
    t_c = torch.from_numpy(raw[t_key].astype(np.float32))
    u_ocv = torch.from_numpy(raw["u_ocv_v"].astype(np.float32))
    if isinstance(model, KGridAdapter):
        r0, r1, c1 = model(x.unsqueeze(0), soc.unsqueeze(0), t_c.unsqueeze(0))
    else:
        r0, r1, c1 = model(x.unsqueeze(0))
    u_p0 = i.new_tensor([float(raw["u_p_v"][0])]) if "u_p_v" in raw else None
    u_hat, _ = ecm_forward(
        i.unsqueeze(0),
        u_ocv.unsqueeze(0),
        r0,
        r1,
        c1,
        dt_s=cfg.dt_s,
        u_p0=u_p0,
    )
    u_meas = raw.get("u_t_meas_v", raw["u_t_true_v"])
    return {
        "time_s": raw["time_s"],
        "i_a": raw[i_key],
        "soc": raw[soc_key],
        "t_c": raw[t_key],
        "u_t_meas_v": u_meas,
        "u_t_hat_v": u_hat.squeeze(0).cpu().numpy(),
        "r0_hat_ohm": r0.squeeze(0).cpu().numpy(),
        "r1_hat_ohm": r1.squeeze(0).cpu().numpy(),
        "r0_ohm": raw.get("r0_ohm", np.full_like(raw["time_s"], np.nan)),
        "r1_ohm": raw.get("r1_ohm", np.full_like(raw["time_s"], np.nan)),
        "mode": raw["mode"] if "mode" in raw else None,
    }


def _err(data: dict[str, np.ndarray]) -> np.ndarray:
    return data["u_t_hat_v"] - data["u_t_meas_v"]


def _soc_err_pp(log: dict[str, np.ndarray]) -> np.ndarray:
    return (log["soc_post"] - log["soc_true"]) * 100.0


def _ah_err_pp(log: dict[str, np.ndarray]) -> np.ndarray:
    return (log["soc_ah"] - log["soc_true"]) * 100.0


def run_ekf_csv(
    model: torch.nn.Module,
    scaler: FeatureScaler,
    csv_path: Path,
    *,
    use_true_inputs: bool,
) -> dict[str, np.ndarray]:
    raw = _load_csv(csv_path)
    i_key = "i_true_a" if use_true_inputs else "i_meas_a"
    t_key = "t_true_c" if use_true_inputs else "t_meas_c"
    if i_key not in raw:
        i_key = "i_true_a" if "i_true_a" in raw else "i_meas_a"
    if t_key not in raw:
        t_key = "t_true_c" if "t_true_c" in raw else "t_meas_c"
    u_key = "u_t_meas_v" if "u_t_meas_v" in raw else "u_t_true_v"
    kf_cfg = KfConfig()
    soc_true = raw["soc_true"] if "soc_true" in raw else None
    soc0 = None
    if soc_true is not None:
        soc0 = float(soc_true[0])
        soc0 = float(np.clip(soc0 + float(raw[i_key][0]) * kf_cfg.dt_s / kf_cfg.q_coulomb, 0.0, 1.0))
    log = run_filter(
        MlpParamProvider(model, scaler),
        raw[i_key],
        raw[t_key],
        raw[u_key],
        cfg=kf_cfg,
        s0=soc0,
        time_s=raw["time_s"],
        soc_true=soc_true,
    )
    if "mode" in raw:
        log["mode"] = raw["mode"]
    return log


def plot_soc(
    logs: dict[str, dict[str, np.ndarray]],
    fig_path: Path,
    *,
    title: str,
    show: bool,
) -> Path:
    from matplotlib import pyplot as plt

    apply_style()
    first = next(iter(logs.values()))
    t = first["time_s"]
    modes = first.get("mode")
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(12.5, 11.5), constrained_layout=True)
    fig.suptitle(title, fontsize=13)
    _shade_modes(axes, t, modes)

    plot_overlay(axes[0], t, first["i_used_a"], "truth", color="#1f4e79", label="电流")
    axes[0].set_ylabel("电流 / A")
    axes[0].legend(loc="upper right")

    if "soc_true" in first:
        plot_overlay(axes[1], t, first["soc_true"] * 100.0, "truth", color="#616161", label="真值")
    plot_overlay(axes[1], t, first["soc_ah"] * 100.0, "ah", color="#64b5f6", label="安时")
    bits = []
    if "soc_true" in first:
        bits.append(f"安时 {_rmse_pp(_ah_err_pp(first)):.3f} pp")
    for key, log in logs.items():
        st = STYLE[key]
        plot_overlay(axes[1], t, log["soc_post"] * 100.0, st["kind"], color=st["color"], label=f"EKF {st['label']}")
        if "soc_true" in log:
            bits.append(f"{st['label']} {_rmse_pp(_soc_err_pp(log)):.3f} pp")
    axes[1].set_ylabel("SOC / %")
    _legend_with_modes(axes[1])
    if bits:
        axes[1].set_title("SOC RMSE  " + "   ".join(bits), loc="left", fontsize=9)

    axes[2].axhline(0.0, color="#9e9e9e", lw=0.8, zorder=1)
    if "soc_true" in first:
        plot_overlay(axes[2], t, _ah_err_pp(first), "ah", color="#64b5f6", label="安时")
        end_bits = [f"安时终点 {_ah_err_pp(first)[-1]:+.3f} pp"]
        for key, log in logs.items():
            st = STYLE[key]
            err = _soc_err_pp(log)
            plot_overlay(axes[2], t, err, st["kind"], color=st["color"], label=st["label"])
            end_bits.append(f"{st['label']} {err[-1]:+.3f} pp")
        axes[2].set_title("终点  " + "   ".join(end_bits), loc="left", fontsize=9)
    axes[2].set_ylabel("SOC 误差 / pp")
    axes[2].legend(loc="upper right", ncol=len(logs) + 1)

    axes[3].axhline(0.0, color="#9e9e9e", lw=0.8, zorder=1)
    v_bits = []
    for key, log in logs.items():
        st = STYLE[key]
        plot_overlay(axes[3], t, log["e_pri"] * 1e3, st["kind"], color=st["color"], label=st["label"])
        v_bits.append(f"{st['label']} {float(np.sqrt(np.mean(log['e_pri']**2))*1e3):.2f} mV")
    axes[3].set_ylabel("先验创新 / mV")
    axes[3].legend(loc="upper right", ncol=len(logs))
    axes[3].set_title("电压 $e^{pri}$ RMSE  " + "   ".join(v_bits) + "   （增量仍只用开环）", loc="left", fontsize=9)

    nis_hi = 8.0
    for key, log in logs.items():
        st = STYLE[key]
        plot_overlay(axes[4], t, log["nis"], st["kind"], color=st["color"], label=st["label"])
        nis_hi = max(nis_hi, float(np.quantile(log["nis"], 0.98)) * 1.2)
    axes[4].axhline(1.0, color="#9e9e9e", ls="--", lw=0.9, zorder=1)
    axes[4].set_ylabel("NIS")
    axes[4].set_xlabel("时间 / s")
    axes[4].set_ylim(0.0, nis_hi)
    axes[4].legend(loc="upper right", ncol=len(logs))
    axes[4].set_title(
        "NIS 中位  " + "   ".join(
            f"{STYLE[k]['label']} {float(np.median(logs[k]['nis'])):.2f}" for k in logs
        ),
        loc="left",
        fontsize=9,
    )

    for ax in axes:
        ax.margins(x=0.01)
    return save_figure(fig, fig_path, show=show)


def _rmse_pp(err_pp: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err_pp))))


def plot_soc_board(
    by_tag: dict[str, dict[str, dict[str, np.ndarray]]],
    fig_path: Path,
    *,
    title: str,
    show: bool,
) -> Path:
    from matplotlib import pyplot as plt

    apply_style()
    tags = [k for k in ("cold", "mid", "hot", "trace") if k in by_tag]
    n = 1 + len(tags)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(12.5, 2.2 + 2.0 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13)
    first_logs = by_tag[tags[0]]
    first = next(iter(first_logs.values()))
    t = first["time_s"]
    modes = first.get("mode")
    _shade_modes(axes, t, modes)

    plot_overlay(axes[0], t, first["i_used_a"], "truth", color="#1f4e79", label="电流")
    axes[0].set_ylabel("电流 / A")
    _legend_with_modes(axes[0])

    label_map = {"cold": "冷端 −10 °C", "mid": "中温 +20 °C", "hot": "热端 +50 °C", "trace": "本条"}
    for ax, tag in zip(axes[1:], tags):
        ax.axhline(0.0, color="#9e9e9e", lw=0.8, zorder=1)
        logs = by_tag[tag]
        head = next(iter(logs.values()))
        bits = []
        if "soc_true" in head:
            plot_overlay(ax, t, _ah_err_pp(head), "ah", color="#64b5f6", label="安时")
            bits.append(f"安时 {_rmse_pp(_ah_err_pp(head)):.3f} (终 {_ah_err_pp(head)[-1]:+.3f})")
        for key, log in logs.items():
            st = STYLE[key]
            err = _soc_err_pp(log)
            plot_overlay(ax, t, err, st["kind"], color=st["color"], label=st["label"])
            bits.append(f"{st['label']} {_rmse_pp(err):.3f} (终 {err[-1]:+.3f})")
        ax.set_ylabel("SOC 误差 / pp")
        ax.set_title(f"{label_map.get(tag, tag)}    " + "   ".join(bits), loc="left", fontsize=9)
        ax.legend(loc="upper right", ncol=len(logs) + 1)
        ax.margins(x=0.01)
    axes[-1].set_xlabel("时间 / s")
    return save_figure(fig, fig_path, show=show)


def plot_wave(
    traces: dict[str, dict[str, np.ndarray]],
    fig_path: Path,
    *,
    title: str,
    show: bool,
) -> Path:
    from matplotlib import pyplot as plt

    apply_style()
    first = next(iter(traces.values()))
    t = first["time_s"]
    modes = first.get("mode")
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(12.5, 11.2), constrained_layout=True)
    fig.suptitle(title, fontsize=13)
    _shade_modes(axes, t, modes)

    plot_overlay(axes[0], t, first["i_a"], "truth", color="#1f4e79", label="电流")
    axes[0].set_ylabel("电流 / A")
    axes[0].legend(loc="upper right")

    plot_overlay(axes[1], t, first["u_t_meas_v"], "meas", color="#e57373", label="测量 $U_t$")
    bits = []
    for key, data in traces.items():
        st = STYLE[key]
        plot_overlay(axes[1], t, data["u_t_hat_v"], st["kind"], color=st["color"], label=st["label"])
        err = _err(data)
        bits.append(f"{st['label']} {_rmse_mV(err):.2f} mV")
    axes[1].set_ylabel("电压 / V")
    _legend_with_modes(axes[1])
    axes[1].set_title("   ".join(bits), loc="left", fontsize=9)

    axes[2].axhline(0.0, color="#9e9e9e", lw=0.8, zorder=1)
    for key, data in traces.items():
        st = STYLE[key]
        plot_overlay(axes[2], t, _err(data) * 1e3, st["kind"], color=st["color"], label=st["label"])
    axes[2].set_ylabel("开环误差 / mV")
    axes[2].legend(loc="upper right", ncol=len(traces))

    if np.isfinite(first["r0_ohm"]).any():
        plot_overlay(axes[3], t, first["r0_ohm"] * 1e3, "truth", color="#90caf9", label="教师 $R_0$")
    for key, data in traces.items():
        st = STYLE[key]
        plot_overlay(axes[3], t, data["r0_hat_ohm"] * 1e3, st["kind"], color=st["color"], label=st["label"])
    axes[3].set_ylabel("$R_0$ / mΩ")
    axes[3].legend(loc="upper right", ncol=len(traces) + 1)

    if np.isfinite(first["r1_ohm"]).any():
        plot_overlay(axes[4], t, first["r1_ohm"] * 1e3, "truth", color="#ffcc80", label="教师 $R_1$")
    for key, data in traces.items():
        st = STYLE[key]
        plot_overlay(axes[4], t, data["r1_hat_ohm"] * 1e3, st["kind"], color=st["color"], label=st["label"])
    axes[4].set_ylabel("$R_1$ / mΩ")
    axes[4].set_xlabel("时间 / s")
    axes[4].legend(loc="upper right", ncol=len(traces) + 1)

    for ax in axes:
        ax.margins(x=0.01)
    return save_figure(fig, fig_path, show=show)


def plot_resid_board(
    by_tag: dict[str, dict[str, dict[str, np.ndarray]]],
    fig_path: Path,
    *,
    title: str,
    show: bool,
) -> Path:
    from matplotlib import pyplot as plt

    apply_style()
    tags = [k for k in ("cold", "mid", "hot", "trace") if k in by_tag]
    n = 1 + len(tags)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(12.5, 2.2 + 2.0 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13)
    first_traces = by_tag[tags[0]]
    first = next(iter(first_traces.values()))
    t = first["time_s"]
    modes = first.get("mode")
    _shade_modes(axes, t, modes)

    plot_overlay(axes[0], t, first["i_a"], "truth", color="#1f4e79", label="电流")
    axes[0].set_ylabel("电流 / A")
    _legend_with_modes(axes[0])

    label_map = {"cold": "冷端 −10 °C", "mid": "中温 +20 °C", "hot": "热端 +50 °C", "trace": "本条"}
    for ax, tag in zip(axes[1:], tags):
        ax.axhline(0.0, color="#9e9e9e", lw=0.8, zorder=1)
        bits = []
        for key, data in by_tag[tag].items():
            st = STYLE[key]
            err = _err(data)
            plot_overlay(ax, t, err * 1e3, st["kind"], color=st["color"], label=st["label"])
            bits.append(f"{st['label']} {_rmse_mV(err):.2f} (max {_max_mV(err):.1f})")
        ax.set_ylabel("开环误差 / mV")
        ax.set_title(f"{label_map.get(tag, tag)}    " + "   ".join(bits), loc="left", fontsize=9)
        ax.legend(loc="upper right", ncol=len(by_tag[tag]))
        ax.margins(x=0.01)
    axes[-1].set_xlabel("时间 / s")
    return save_figure(fig, fig_path, show=show)


def plot_zoom(
    traces: dict[str, dict[str, np.ndarray]],
    fig_path: Path,
    *,
    title: str,
    show: bool,
    windows: tuple[tuple[float, float, str], ...] = (
        (20.0, 90.0, "1C 起跳"),
        (200.0, 280.0, "1C 切断 + 回弹"),
    ),
) -> Path:
    from matplotlib import pyplot as plt

    apply_style()
    first = next(iter(traces.values()))
    t_all = first["time_s"]
    modes = first.get("mode")
    ncols = len(windows)
    fig, axes = plt.subplots(3, ncols, sharey="row", figsize=(6.2 * ncols, 8.4), constrained_layout=True)
    if ncols == 1:
        axes = np.expand_dims(axes, 1)
    fig.suptitle(title, fontsize=13)
    for j, (t0, t1, name) in enumerate(windows):
        m = (t_all >= t0) & (t_all <= t1)
        t = t_all[m]
        sl_modes = modes[m] if modes is not None else None
        col = [axes[0, j], axes[1, j], axes[2, j]]
        _shade_modes(col, t, sl_modes)
        plot_overlay(col[0], t, first["i_a"][m], "truth", color="#1f4e79", label="电流")
        col[0].set_title(name, loc="left", fontsize=11)
        col[0].set_ylabel("电流 / A" if j == 0 else "")
        plot_overlay(col[1], t, first["u_t_meas_v"][m], "meas", color="#e57373", label="测量 $U_t$")
        for key, data in traces.items():
            st = STYLE[key]
            plot_overlay(col[1], t, data["u_t_hat_v"][m], st["kind"], color=st["color"], label=st["label"])
        col[1].set_ylabel("电压 / V" if j == 0 else "")
        col[2].axhline(0.0, color="#9e9e9e", lw=0.8, zorder=1)
        for key, data in traces.items():
            st = STYLE[key]
            plot_overlay(col[2], t, _err(data)[m] * 1e3, st["kind"], color=st["color"], label=st["label"])
        col[2].set_ylabel("开环误差 / mV" if j == 0 else "")
        col[2].set_xlabel("时间 / s")
        for ax in col:
            ax.margins(x=0.01)
        if j == ncols - 1:
            _legend_with_modes(col[1])
            col[2].legend(loc="upper right", ncol=1)
    return save_figure(fig, fig_path, show=show)


def plot_kgrid(
    tables: dict,
    fig_path: Path,
    *,
    title: str,
    show: bool,
    hits: tuple[list, list] | None = None,
) -> Path:
    from matplotlib import pyplot as plt

    apply_style()
    soc = np.asarray(tables["soc_node"], dtype=float)
    t_c = np.asarray(tables["t_node"], dtype=float)
    k0 = np.asarray(tables["k0"], dtype=float)
    k1 = np.asarray(tables["k1"], dtype=float)
    panels = [("k0", k0, 1.00, 1.20), ("k1", k1, 1.00, 1.20)]
    if hits is not None:
        panels.append(("hit0 大电流窗", np.asarray(hits[0], dtype=float), None, None))
        panels.append(("hit1 回弹窗", np.asarray(hits[1], dtype=float), None, None))
    nrows = 1 if len(panels) <= 2 else 2
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.2, 4.2 * nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    fig.suptitle(title, fontsize=13)
    for idx, (name, mat, vmin, vmax) in enumerate(panels):
        ax = axes[idx // ncols, idx % ncols]
        cmap = "YlOrRd" if name.startswith("k") else "Blues"
        im = ax.imshow(mat, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(t_c)))
        ax.set_xticklabels([f"{v:+.0f}" for v in t_c])
        ax.set_yticks(range(len(soc)))
        ax.set_yticklabels([f"{v * 100:.0f}" for v in soc])
        ax.set_xlabel("温度 / °C")
        ax.set_ylabel("SOC / %")
        ax.set_title(name)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.3f}" if name.startswith("k") else f"{mat[i, j]:.1f}",
                        ha="center", va="center", fontsize=8, color="#212121")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if len(panels) < nrows * ncols:
        axes[-1, -1].axis("off")
    return save_figure(fig, fig_path, show=show)


def _k_caption(model: torch.nn.Module) -> str:
    if isinstance(model, ScaleAdapter):
        return f"$k_0$={model.k0:.3f} $k_1$={model.k1:.3f}"
    if isinstance(model, KGridAdapter):
        k0, k1 = model.k_at(0.50, 25.0)
        return f"网格 $k(0.5,25^\\circ$C)={k0:.3f}/{k1:.3f}"
    return ""


def run_plots(args: argparse.Namespace) -> list[Path]:
    mlp_dir = _resolve(args.mlp_dir)
    new_dir = _resolve(args.new_dir)
    ckpt_mlp = mlp_dir / "best.pt"
    if not ckpt_mlp.exists():
        raise FileNotFoundError(ckpt_mlp)
    base, scaler, cfg = load_bundle(ckpt_mlp, mlp_dir / "config.json", mlp_dir / "scaler.json")
    base.eval()
    use_true = bool(args.use_true_inputs)
    frozen = ScaleAdapter(base).eval()

    models: dict[str, torch.nn.Module] = {"freeze": frozen}
    notes: list[str] = []

    want_a = args.exp in {"a", "both"}
    want_b = args.exp in {"kgrid", "both"}

    if want_a:
        win_ckpt = _resolve(args.window_ckpt) if args.window_ckpt else _resolve(WINDOW_DEFAULT)
        if args.k0 is not None:
            models["a"] = load_adapted(base, ckpt=None, k0=args.k0, k1=args.k1 if args.k1 is not None else args.k0)
            notes.append(f"1a 手动 k0={args.k0:.3f} k1={(args.k1 if args.k1 is not None else args.k0):.3f}")
        elif win_ckpt.exists():
            models["a"] = load_adapted(base, ckpt=win_ckpt, k0=None, k1=None)
            notes.append(f"1a {win_ckpt}  {_k_caption(models['a'])}")
        else:
            models["a"] = load_adapted(base, ckpt=None, k0=K1A_FALLBACK[0], k1=K1A_FALLBACK[1])
            notes.append(
                f"1a 盘上无 {WINDOW_DEFAULT}，用 05-d 任务 A 一遍 "
                f"k0/k1={K1A_FALLBACK[0]}/{K1A_FALLBACK[1]}"
            )

    kgrid_ckpt = _resolve(args.ckpt) if args.ckpt else _resolve(KGRID_DEFAULT)
    if want_b:
        if kgrid_ckpt.exists():
            models["b"] = load_adapted(base, ckpt=kgrid_ckpt, k0=None, k1=None)
            notes.append(f"1b {kgrid_ckpt}  {_k_caption(models['b'])}")
        else:
            notes.append(f"1b 找不到 {kgrid_ckpt}，跳过网格")

    csvs = pick_csvs(new_dir, args.csv)
    by_tag: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for tag, path in csvs:
        bundle = {}
        for key, model in models.items():
            bundle[key] = infer_csv(model, scaler, cfg, path, use_true_inputs=use_true)
        by_tag[tag] = bundle
        print(f"{tag}  {path.name}")
        for key, data in bundle.items():
            err = _err(data)
            print(f"  {STYLE[key]['label']:16s}  RMSE {_rmse_mV(err):6.2f} mV  MAX {_max_mV(err):6.1f} mV")

    written: list[Path] = []
    prefix = args.fig_prefix
    mid_key = "mid" if "mid" in by_tag else next(iter(by_tag))
    mid = by_tag[mid_key]
    t_mean = float(np.mean(next(iter(mid.values()))["t_c"]))
    written.append(
        plot_wave(
            mid,
            Path(f"{prefix}_mid_wave.png"),
            title=f"第 1 期开环  SOC 50%  T{t_mean:+.0f} °C  {new_dir.name}",
            show=args.show,
        )
    )
    written.append(
        plot_resid_board(
            by_tag,
            Path(f"{prefix}_resid_T.png"),
            title=f"第 1 期开环残差  边沿 / 恒流 / 回弹  {new_dir.name}",
            show=args.show,
        )
    )
    written.append(
        plot_zoom(
            mid,
            Path(f"{prefix}_zoom.png"),
            title=f"第 1 期局部  1C 边沿与回弹  SOC 50%  T{t_mean:+.0f} °C",
            show=args.show,
        )
    )

    tables = None
    hits = None
    meta_path = kgrid_ckpt.with_name("kgrid.json") if want_b and kgrid_ckpt.exists() else None
    if want_b and "b" in models and isinstance(models["b"], KGridAdapter):
        tables = models["b"].k_tables()
        if meta_path and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "hit0" in meta and "hit1" in meta:
                hits = (meta["hit0"], meta["hit1"])
    elif want_b and meta_path and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tables = meta.get("k_tables")
        if "hit0" in meta:
            hits = (meta["hit0"], meta["hit1"])
    if tables is not None:
        written.append(
            plot_kgrid(
                tables,
                Path(f"{prefix}_kgrid.png"),
                title="第 1 期 1b  $k$ 网格（4 遍）  节点 SOC×T",
                show=args.show,
                hits=hits,
            )
        )

    if not args.no_kf:
        kf_by_tag: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for tag, path in csvs:
            bundle = {}
            print(f"EKF {tag}  {path.name}")
            for key, model in models.items():
                log = run_ekf_csv(model, scaler, path, use_true_inputs=use_true)
                bundle[key] = log
                stats = filter_metrics(log)
                line = (
                    f"  {STYLE[key]['label']:16s}  "
                    f"e_pri {stats['e_pri_rmse_mV']:6.2f} mV  "
                    f"NIS {stats['nis_median']:.2f}"
                )
                if "s_post_rmse" in stats:
                    line += (
                        f"  SOC 安时 {stats['s_ah_rmse']*100:.3f} / "
                        f"EKF {stats['s_post_rmse']*100:.3f} pp  "
                        f"终点 {stats['s_end_post_err']*100:+.3f} pp"
                    )
                print(line)
            kf_by_tag[tag] = bundle
        kf_mid = kf_by_tag[mid_key]
        written.append(
            plot_soc(
                kf_mid,
                Path(f"{prefix}_soc_mid.png"),
                title=f"第 1 期 EKF SOC  SOC 50%  T{t_mean:+.0f} °C  {new_dir.name}",
                show=args.show,
            )
        )
        written.append(
            plot_soc_board(
                kf_by_tag,
                Path(f"{prefix}_soc_T.png"),
                title=f"第 1 期 EKF SOC 误差  {new_dir.name}",
                show=args.show,
            )
        )

    for line in notes:
        print(line)
    for path in written:
        print(f"图    {path}")
    return written


def plot_from_out(
    out_dir: Path,
    *,
    new_dir: Path,
    mlp_dir: Path,
    fig_prefix: str,
    use_true_inputs: bool = False,
) -> list[Path]:
    """window.py / kgrid.py 跑完后调用。"""
    ns = argparse.Namespace(
        mlp_dir=str(mlp_dir),
        new_dir=str(new_dir),
        csv=None,
        exp="kgrid",
        ckpt=str(out_dir / "last.pt"),
        window_ckpt=str(out_dir / "last.pt"),
        k0=None,
        k1=None,
        use_true_inputs=use_true_inputs,
        fig_prefix=fig_prefix,
        show=False,
        no_kf=False,
    )
    blob_path = out_dir / "last.pt"
    if blob_path.exists():
        blob = torch.load(blob_path, map_location="cpu", weights_only=False)
        mode = str(blob.get("incr_mode", ""))
        if mode == "window_k":
            ns.exp = "a"
            ns.window_ckpt = str(blob_path)
            ns.ckpt = ""
        elif mode == "window_kgrid":
            ns.exp = "kgrid"
            ns.ckpt = str(blob_path)
    return run_plots(ns)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="第 1 期开环残差与 EKF SOC 图")
    p.add_argument("--mlp-dir", default="Data/ai_mlp")
    p.add_argument("--new-dir", default="Data/soh_k115")
    p.add_argument("--csv", default=None, help="只画这一条；默认 SOC 50 冷/中/热各一条")
    p.add_argument("--exp", default="both", choices=["a", "kgrid", "both"])
    p.add_argument("--ckpt", default=None, help="1b last.pt，默认 Data/ai_local/kgrid_k115_p4/last.pt")
    p.add_argument("--window-ckpt", default=None, help="1a last.pt；没有则用 05-d 的 k0/k1")
    p.add_argument("--k0", type=float, default=None)
    p.add_argument("--k1", type=float, default=None)
    p.add_argument("--use-true-inputs", action="store_true")
    p.add_argument("--fig-prefix", default="local/phase1")
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-kf", action="store_true", help="只画开环，不跑 EKF")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_plots(args)


if __name__ == "__main__":
    main()
