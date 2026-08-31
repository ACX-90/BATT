"""从 Data/grid_pybamm 已有序列估算每档 (R0,R1,R2,tau1,tau2)。

方案（离线、每条 CSV 一组常参数）：
  1. 边沿 R0：0.1 s 的 -dU/dI，1C 升降沿为主，并报 0.5C / 2C（BV）
  2. 120 s 静置双指数（cmd 2，1C 180 s 之后）：1-exp / 1-exp+偏置 / 2-exp
  3. 整段 LTI 1RC / 2RC 最小二乘（OCV 用 CSV 的 u_ocv_v）

教师列 r0_ohm/r1_ohm 是仓库 ECM 求值，不是 PyBaMM 真值，只作对照不参与拟合。

用法（仓库根目录）：

    python Branch/eval_pybamm_rc/identify_rc.py
    python Branch/eval_pybamm_rc/identify_rc.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "Data" / "grid_pybamm"
OUT = HERE / "out"
DT_S = 0.1


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def rc_up(i: np.ndarray, r: float, tau: float, dt: float) -> np.ndarray:
    a = float(np.exp(-dt / max(tau, 1e-6)))
    return lfilter([r * (1.0 - a)], [1.0, -a], i)


def sim_ut(
    i: np.ndarray,
    ocv: np.ndarray,
    r0: float,
    r1: float,
    tau1: float,
    r2: float = 0.0,
    tau2: float = 90.0,
    dt: float = DT_S,
) -> np.ndarray:
    up1 = rc_up(i, r1, tau1, dt)
    up2 = rc_up(i, r2, tau2, dt) if r2 > 0.0 else 0.0
    return ocv - i * r0 - up1 - up2


def rmse_mv(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)) * 1e3)


def edge_r0(i: np.ndarray, u: np.ndarray, *, di_min: float = 20.0) -> list[dict]:
    di = np.diff(i)
    idx = np.where(np.abs(di) > di_min)[0] + 1
    rows = []
    for k in idx:
        d_i = float(i[k] - i[k - 1])
        if abs(d_i) < di_min:
            continue
        r01 = -(u[k] - u[k - 1]) / d_i
        n1 = min(k + 9, len(u) - 1)
        r10 = -(u[n1] - u[k - 1]) / d_i
        rows.append(
            {
                "k": int(k),
                "di_a": d_i,
                "i_after_a": float(i[k]),
                "r0_0p1_ohm": float(r01),
                "r_1s_ohm": float(r10),
            }
        )
    return rows


def pick_r0_1c(edges: list[dict]) -> tuple[float, float]:
    """1C 附近 (|di|~100 A) 的 0.1 s 电阻，升降沿平均。"""
    ones = [e for e in edges if 70.0 <= abs(e["di_a"]) <= 130.0]
    if not ones:
        ones = [e for e in edges if 40.0 <= abs(e["di_a"]) <= 250.0]
    if not ones:
        return float("nan"), float("nan")
    r0 = float(np.mean([e["r0_0p1_ohm"] for e in ones]))
    r1s = float(np.mean([e["r_1s_ohm"] for e in ones]))
    return r0, r1s


def fit_exp_grid(
    t: np.ndarray,
    y: np.ndarray,
    *,
    n_pole: int,
    with_offset: bool,
) -> dict:
    """对数网格 + 线性最小二乘。n_pole=1 或 2。"""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    tau1_grid = np.logspace(np.log10(0.4), np.log10(25.0), 18)
    tau2_grid = np.logspace(np.log10(18.0), np.log10(280.0), 20)
    best = None
    best_sse = np.inf
    if n_pole == 1:
        for tau1 in tau1_grid:
            cols = [np.exp(-t / tau1)]
            if with_offset:
                cols.append(np.ones_like(t))
            phi = np.column_stack(cols)
            coef, *_ = np.linalg.lstsq(phi, y, rcond=None)
            yhat = phi @ coef
            sse = float(np.dot(y - yhat, y - yhat))
            if sse < best_sse:
                best_sse = sse
                best = {"tau1": float(tau1), "A1": float(coef[0]), "C": float(coef[1]) if with_offset else 0.0, "yhat": yhat}
    else:
        for tau1 in tau1_grid:
            for tau2 in tau2_grid:
                if tau2 < 4.0 * tau1:
                    continue
                phi = np.column_stack([np.exp(-t / tau1), np.exp(-t / tau2)])
                coef, *_ = np.linalg.lstsq(phi, y, rcond=None)
                yhat = phi @ coef
                sse = float(np.dot(y - yhat, y - yhat))
                if sse < best_sse:
                    best_sse = sse
                    best = {
                        "tau1": float(tau1),
                        "tau2": float(tau2),
                        "A1": float(coef[0]),
                        "A2": float(coef[1]),
                        "yhat": yhat,
                    }
    if best is None:
        return {"ok": False, "rmse_mv": float("nan")}
    yhat = best.pop("yhat")
    out = {**best, "ok": True, "rmse_mv": rmse_mv(y, yhat), "sse": best_sse}
    if n_pole == 2:
        phi = np.column_stack([np.exp(-t / out["tau1"]), np.exp(-t / out["tau2"])])
        out["cond"] = float(np.linalg.cond(phi))
    return out


def refine_2exp(t: np.ndarray, y: np.ndarray, seed: dict) -> dict:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    def unpack(z):
        a1, a2 = z[0], z[1]
        tau1 = np.clip(np.exp(z[2]), 0.3, 40.0)
        tau2 = np.clip(max(np.exp(z[3]), 4.0 * tau1), 12.0, 400.0)
        return a1, a2, tau1, tau2

    def fun(z):
        a1, a2, tau1, tau2 = unpack(z)
        return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) - y

    z0 = np.array(
        [
            seed.get("A1", y[0] * 0.3),
            seed.get("A2", y[0] * 0.7),
            np.log(max(seed.get("tau1", 4.0), 0.5)),
            np.log(max(seed.get("tau2", 80.0), 20.0)),
        ],
        dtype=float,
    )
    res = least_squares(fun, z0, method="trf", max_nfev=400)
    a1, a2, tau1, tau2 = unpack(res.x)
    yhat = a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)
    phi = np.column_stack([np.exp(-t / tau1), np.exp(-t / tau2)])
    return {
        "ok": bool(res.success),
        "A1": float(a1),
        "A2": float(a2),
        "tau1": float(tau1),
        "tau2": float(tau2),
        "rmse_mv": rmse_mv(y, yhat),
        "cond": float(np.linalg.cond(phi)),
        "nfev": int(res.nfev),
    }


def amp_to_r(amp: float, i_pre: float, tp: float, tau: float) -> float:
    if abs(i_pre) < 1.0 or tau <= 0:
        return float("nan")
    gain = 1.0 - np.exp(-tp / tau)
    if gain < 0.05:
        return float("nan")
    return float(amp / (i_pre * gain))


def fit_ltis(
    i: np.ndarray,
    ocv: np.ndarray,
    u: np.ndarray,
    *,
    order: int,
    r0_hint: float,
    r1_hint: float,
    tau1_hint: float,
    r2_hint: float,
    tau2_hint: float,
) -> dict:
    i = np.asarray(i, dtype=float)
    ocv = np.asarray(ocv, dtype=float)
    u = np.asarray(u, dtype=float)
    r0_h = float(np.clip(r0_hint if np.isfinite(r0_hint) else 1e-3, 2e-4, 8e-3))
    r1_h = float(np.clip(abs(r1_hint) if np.isfinite(r1_hint) else 4e-4, 5e-5, 8e-3))
    t1_h = float(np.clip(tau1_hint if np.isfinite(tau1_hint) else 5.0, 0.5, 30.0))
    r2_h = float(np.clip(abs(r2_hint) if np.isfinite(r2_hint) else 3e-4, 5e-5, 8e-3))
    t2_h = float(np.clip(tau2_hint if np.isfinite(tau2_hint) else 90.0, 20.0, 350.0))

    if order == 1:
        z0 = np.array([np.log(r0_h), np.log(r1_h), np.log(t1_h)], dtype=float)

        def unpack(z):
            r0 = float(np.clip(np.exp(z[0]), 1e-4, 1e-2))
            r1 = float(np.clip(np.exp(z[1]), 2e-5, 1e-2))
            tau1 = float(np.clip(np.exp(z[2]), 0.3, 40.0))
            return r0, r1, tau1, 0.0, 90.0

    else:
        z0 = np.array(
            [np.log(r0_h), np.log(r1_h), np.log(t1_h), np.log(r2_h), np.log(t2_h)],
            dtype=float,
        )

        def unpack(z):
            r0 = float(np.clip(np.exp(z[0]), 1e-4, 1e-2))
            r1 = float(np.clip(np.exp(z[1]), 2e-5, 1e-2))
            tau1 = float(np.clip(np.exp(z[2]), 0.3, 40.0))
            r2 = float(np.clip(np.exp(z[3]), 2e-5, 1e-2))
            tau2 = float(np.clip(max(np.exp(z[4]), 4.0 * tau1), 12.0, 400.0))
            return r0, r1, tau1, r2, tau2

    def fun(z):
        r0, r1, tau1, r2, tau2 = unpack(z)
        uh = sim_ut(i, ocv, r0, r1, tau1, r2, tau2)
        return (uh - u) * 1e3

    res = least_squares(fun, z0, method="trf", max_nfev=250)
    r0, r1, tau1, r2, tau2 = unpack(res.x)
    uh = sim_ut(i, ocv, r0, r1, tau1, r2, tau2)
    return {
        "ok": bool(res.success),
        "r0_ohm": r0,
        "r1_ohm": r1,
        "tau1_s": tau1,
        "r2_ohm": r2,
        "tau2_s": tau2,
        "rmse_mv": rmse_mv(u, uh),
        "nfev": int(res.nfev),
        "uh": uh,
    }


def identify_one(path: Path) -> dict:
    df = load_csv(path)
    i = df.i_true_a.to_numpy(dtype=float)
    u = df.u_t_true_v.to_numpy(dtype=float)
    ocv = df.u_ocv_v.to_numpy(dtype=float)
    cmd = df.cmd_id.to_numpy(dtype=int)
    soc = df.soc_true.to_numpy(dtype=float)
    t_c = float(df.t_true_c.mean())
    soc0 = float(soc[0])
    cutoff = int(df.cutoff.to_numpy().sum())

    edges = edge_r0(i, u)
    r0_1c, r1s_1c = pick_r0_1c(edges)
    r0_by_i = {}
    for e in edges:
        mag = abs(e["di_a"])
        key = "2c" if mag > 150 else ("1c" if mag > 70 else ("0p5c" if mag > 35 else "other"))
        r0_by_i.setdefault(key, []).append(e["r0_0p1_ohm"])
    r0_i = {k: float(np.mean(v)) for k, v in r0_by_i.items()}

    rest = df[cmd == 2]
    rec = {
        "file": path.name,
        "soc0": soc0,
        "t_c": t_c,
        "soc_end": float(soc[-1]),
        "dsoc": float(soc[-1] - soc[0]),
        "cutoff_n": cutoff,
        "r0_1c_ohm": r0_1c,
        "r_1s_1c_ohm": r1s_1c,
        "r0_0p5c_ohm": r0_i.get("0p5c", float("nan")),
        "r0_2c_ohm": r0_i.get("2c", float("nan")),
        "teacher_r0_mohm": float(df.r0_ohm.mean() * 1e3),
        "teacher_r1_mohm": float(df.r1_ohm.mean() * 1e3),
        "teacher_tau1_s": float(df.tau1_s.mean()),
    }
    if len(rest) < 50:
        rec["rest_ok"] = False
        return rec

    t_rest = (rest.time_s.to_numpy() - rest.time_s.iloc[0]).astype(float)
    eta = (rest.u_ocv_v - rest.u_t_true_v).to_numpy(dtype=float)
    rec["rest_ok"] = True
    rec["eta0_mv"] = float(eta[0] * 1e3)
    rec["eta120_mv"] = float(eta[-1] * 1e3)
    rec["eta_frac"] = float(eta[-1] / eta[0]) if abs(eta[0]) > 1e-6 else float("nan")

    pulse = df[cmd == 1]
    i_pre = float(pulse.i_true_a.mean()) if len(pulse) else 100.0
    tp = float(len(pulse) * DT_S)

    f1 = fit_exp_grid(t_rest, eta, n_pole=1, with_offset=False)
    f1c = fit_exp_grid(t_rest, eta, n_pole=1, with_offset=True)
    f2g = fit_exp_grid(t_rest, eta, n_pole=2, with_offset=False)
    f2 = refine_2exp(t_rest, eta, f2g) if f2g.get("ok") else f2g

    rec["rest1_tau1_s"] = f1.get("tau1", float("nan"))
    rec["rest1_rmse_mv"] = f1.get("rmse_mv", float("nan"))
    rec["rest1c_tau1_s"] = f1c.get("tau1", float("nan"))
    rec["rest1c_c_mv"] = float(f1c.get("C", float("nan")) * 1e3) if f1c.get("ok") else float("nan")
    rec["rest1c_rmse_mv"] = f1c.get("rmse_mv", float("nan"))
    rec["rest2_tau1_s"] = f2.get("tau1", float("nan"))
    rec["rest2_tau2_s"] = f2.get("tau2", float("nan"))
    rec["rest2_rmse_mv"] = f2.get("rmse_mv", float("nan"))
    rec["rest2_cond"] = f2.get("cond", float("nan"))
    rec["rest2_r1_ohm"] = amp_to_r(f2.get("A1", float("nan")), i_pre, tp, f2.get("tau1", float("nan")))
    rec["rest2_r2_ohm"] = amp_to_r(f2.get("A2", float("nan")), i_pre, tp, f2.get("tau2", float("nan")))

    l1 = fit_ltis(
        i, ocv, u, order=1,
        r0_hint=r0_1c, r1_hint=rec["rest2_r1_ohm"], tau1_hint=f2.get("tau1", 8.0),
        r2_hint=0.0, tau2_hint=90.0,
    )
    l2 = fit_ltis(
        i, ocv, u, order=2,
        r0_hint=r0_1c,
        r1_hint=rec["rest2_r1_ohm"],
        tau1_hint=f2.get("tau1", 8.0),
        r2_hint=rec["rest2_r2_ohm"],
        tau2_hint=f2.get("tau2", 90.0),
    )
    rec["ltis1_r0_ohm"] = l1["r0_ohm"]
    rec["ltis1_r1_ohm"] = l1["r1_ohm"]
    rec["ltis1_tau1_s"] = l1["tau1_s"]
    rec["ltis1_rmse_mv"] = l1["rmse_mv"]
    rec["ltis2_r0_ohm"] = l2["r0_ohm"]
    rec["ltis2_r1_ohm"] = l2["r1_ohm"]
    rec["ltis2_r2_ohm"] = l2["r2_ohm"]
    rec["ltis2_tau1_s"] = l2["tau1_s"]
    rec["ltis2_tau2_s"] = l2["tau2_s"]
    rec["ltis2_rmse_mv"] = l2["rmse_mv"]
    rec["ltis_gain_mv"] = l1["rmse_mv"] - l2["rmse_mv"]
    rec["_uh1"] = l1["uh"]
    rec["_uh2"] = l2["uh"]
    rec["_u"] = u
    rec["_t"] = df.time_s.to_numpy(dtype=float)
    rec["_i"] = i
    rec["_eta_rest"] = eta
    rec["_t_rest"] = t_rest
    rec["_ocv"] = ocv
    rec["_cmd"] = cmd
    return rec


def plot_case(rec: dict, out: Path) -> None:
    t = rec["_t"]
    u = rec["_u"]
    uh1 = rec["_uh1"]
    uh2 = rec["_uh2"]
    cmd = rec["_cmd"]
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.5), sharex=False)
    ax = axes[0]
    ax.plot(t, rec["_i"], color="#1565c0", lw=1.0)
    ax.set_ylabel("I / A")
    ax.set_title(
        f"{rec['file']}\nSOC0={rec['soc0']:.2f}  T={rec['t_c']:.1f} C  "
        f"LTI 1RC {rec['ltis1_rmse_mv']:.1f} mV → 2RC {rec['ltis2_rmse_mv']:.1f} mV"
    )
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(t, u, color="k", lw=1.1, label="PyBaMM Ut")
    ax.plot(t, uh1, color="#e65100", lw=1.0, ls="--", label="LTI 1RC")
    ax.plot(t, uh2, color="#2e7d32", lw=1.0, ls="-.", label="LTI 2RC")
    ax.set_ylabel("Ut / V")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[2]
    m = cmd == 2
    tr = rec["_t_rest"]
    eta = rec["_eta_rest"] * 1e3
    ax.plot(tr, eta, color="k", lw=1.2, label="η = OCV−Ut")
    tau1, tau2 = rec["rest2_tau1_s"], rec["rest2_tau2_s"]
    r1, r2 = rec["rest2_r1_ohm"], rec["rest2_r2_ohm"]
    # redraw rest 2-exp from stored A via eta0 split is messy; plot LTI up at rest
    rest_uh2 = uh2[m]
    rest_ocv = rec["_ocv"][m]
    ax.plot(tr, (rest_ocv - rest_uh2) * 1e3, color="#2e7d32", ls="-.", label="LTI 2RC η")
    ax.plot(tr, (rest_ocv - uh1[m]) * 1e3, color="#e65100", ls="--", label="LTI 1RC η")
    ax.set_xlabel("rest t / s")
    ax.set_ylabel("η / mV")
    ax.set_title(
        f"120 s rest  η0={rec['eta0_mv']:.1f} mV  leftover={rec['eta120_mv']:.1f} mV  "
        f"rest 2-exp τ=({tau1:.1f},{tau2:.0f}) s  R=({r1*1e3:.2f},{r2*1e3:.2f}) mΩ"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_grid(tab: pd.DataFrame, out_dir: Path) -> None:
    def pivot(col):
        return tab.pivot_table(index="t_c", columns="soc0", values=col, aggfunc="mean")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    specs = [
        (0, 0, "r0_1c_ohm", "edge R0 1C / mOhm", 1e3),
        (0, 1, "rest2_r1_ohm", "rest 2-exp R1 / mOhm", 1e3),
        (0, 2, "rest2_r2_ohm", "rest 2-exp R2 / mOhm", 1e3),
        (1, 0, "rest2_tau1_s", "rest 2-exp tau1 / s", 1.0),
        (1, 1, "rest2_tau2_s", "rest 2-exp tau2 / s", 1.0),
        (1, 2, "rest2_rmse_mv", "rest 2-exp RMSE / mV", 1.0),
    ]
    for r, c, col, title, scale in specs:
        ax = axes[r][c]
        p = pivot(col) * scale
        im = ax.imshow(
            p.to_numpy(),
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=[
                float(p.columns.min()),
                float(p.columns.max()),
                float(p.index.min()),
                float(p.index.max()),
            ],
        )
        ax.set_title(title)
        ax.set_xlabel("SOC0")
        ax.set_ylabel("T / C")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "grid_ltis2.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(tab["t_c"], tab["r0_1c_ohm"] * 1e3, s=22, c=tab["soc0"], cmap="viridis", label="edge 1C 0.1s")
    ax.scatter(tab["t_c"], tab["r0_2c_ohm"] * 1e3, s=18, marker="x", c="#c62828", alpha=0.7, label="edge 2C 0.1s")
    ax.set_xlabel("T / C")
    ax.set_ylabel("R0 / mΩ")
    ax.set_title("0.1 s R0 vs T (color=SOC0); 2C < 1C is BV")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "r0_vs_T.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(tab["ltis1_rmse_mv"], tab["ltis2_rmse_mv"], s=22, c=tab["t_c"], cmap="coolwarm")
    lim = max(tab["ltis1_rmse_mv"].max(), tab["ltis2_rmse_mv"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="k", lw=0.8, ls="--")
    ax.set_xlabel("LTI 1RC RMSE / mV")
    ax.set_ylabel("LTI 2RC RMSE / mV")
    ax.set_title("Full-seq voltage RMSE: 2RC vs 1RC")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_1rc_2rc.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(tab["t_c"], tab["rest2_tau1_s"], s=20, label="rest τ1")
    ax.scatter(tab["t_c"], tab["rest2_tau2_s"], s=20, marker="s", label="rest τ2")
    ax.scatter(tab["t_c"], tab["ltis2_tau1_s"], s=14, marker="x", label="LTI τ1")
    ax.scatter(tab["t_c"], tab["ltis2_tau2_s"], s=14, marker="+", label="LTI τ2")
    ax.set_xlabel("T / C")
    ax.set_ylabel("τ / s")
    ax.set_title("tau vs T (120 s rest truncates slow tau)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tau_vs_T.png", dpi=140)
    plt.close(fig)


def summarize(tab: pd.DataFrame) -> dict:
    def q(col):
        x = tab[col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {}
        return {
            "n": int(x.size),
            "mean": float(np.mean(x)),
            "p10": float(np.percentile(x, 10)),
            "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)),
        }

    mid = tab[(tab.soc0 > 0.4) & (tab.soc0 < 0.7) & (tab.t_c > 15) & (tab.t_c < 35)]
    return {
        "n_files": int(len(tab)),
        "r0_1c_mohm": q("r0_1c_ohm") | ({"note": "ohm→mΩ"} if False else {}),
        "metrics": {
            "r0_1c_mohm": {k: (v * 1e3 if k != "n" else v) for k, v in q("r0_1c_ohm").items()},
            "r0_2c_mohm": {k: (v * 1e3 if k != "n" else v) for k, v in q("r0_2c_ohm").items()},
            "eta0_mv": q("eta0_mv"),
            "eta120_mv": q("eta120_mv"),
            "eta_frac": q("eta_frac"),
            "rest1_rmse_mv": q("rest1_rmse_mv"),
            "rest1c_rmse_mv": q("rest1c_rmse_mv"),
            "rest2_rmse_mv": q("rest2_rmse_mv"),
            "rest2_tau1_s": q("rest2_tau1_s"),
            "rest2_tau2_s": q("rest2_tau2_s"),
            "ltis1_rmse_mv": q("ltis1_rmse_mv"),
            "ltis2_rmse_mv": q("ltis2_rmse_mv"),
            "ltis_gain_mv": q("ltis_gain_mv"),
            "ltis2_r0_mohm": {k: (v * 1e3 if k != "n" else v) for k, v in q("ltis2_r0_ohm").items()},
            "ltis2_r1_mohm": {k: (v * 1e3 if k != "n" else v) for k, v in q("ltis2_r1_ohm").items()},
            "ltis2_r2_mohm": {k: (v * 1e3 if k != "n" else v) for k, v in q("ltis2_r2_ohm").items()},
            "ltis2_tau1_s": q("ltis2_tau1_s"),
            "ltis2_tau2_s": q("ltis2_tau2_s"),
        },
        "mid_n": int(len(mid)),
        "mid_mean": {
            "r0_1c_mohm": float(mid.r0_1c_ohm.mean() * 1e3) if len(mid) else None,
            "ltis2_r0_mohm": float(mid.ltis2_r0_ohm.mean() * 1e3) if len(mid) else None,
            "ltis2_r1_mohm": float(mid.ltis2_r1_ohm.mean() * 1e3) if len(mid) else None,
            "ltis2_r2_mohm": float(mid.ltis2_r2_ohm.mean() * 1e3) if len(mid) else None,
            "ltis2_tau1_s": float(mid.ltis2_tau1_s.mean()) if len(mid) else None,
            "ltis2_tau2_s": float(mid.ltis2_tau2_s.mean()) if len(mid) else None,
            "eta120_mv": float(mid.eta120_mv.mean()) if len(mid) else None,
            "ltis1_rmse_mv": float(mid.ltis1_rmse_mv.mean()) if len(mid) else None,
            "ltis2_rmse_mv": float(mid.ltis2_rmse_mv.mean()) if len(mid) else None,
        },
    }


def choose_files(quick: bool) -> list[Path]:
    files = sorted(DATA.glob("nmc100ah_pybamm_s*_t*.csv"))
    if not files:
        raise FileNotFoundError(f"没有序列：{DATA}")
    if not quick:
        return files
    want = {
        "nmc100ah_pybamm_s00_t00_soc090_T-10.csv",
        "nmc100ah_pybamm_s00_t05_soc090_T+23.csv",
        "nmc100ah_pybamm_s00_t09_soc090_T+50.csv",
        "nmc100ah_pybamm_s04_t00_soc054_T-10.csv",
        "nmc100ah_pybamm_s04_t05_soc054_T+23.csv",
        "nmc100ah_pybamm_s04_t09_soc054_T+50.csv",
        "nmc100ah_pybamm_s09_t00_soc010_T-10.csv",
        "nmc100ah_pybamm_s09_t05_soc010_T+23.csv",
        "nmc100ah_pybamm_s09_t09_soc010_T+50.csv",
    }
    picked = [p for p in files if p.name in want]
    return picked or files[:9]


def drop_plot_cols(tab: pd.DataFrame) -> pd.DataFrame:
    return tab.drop(columns=[c for c in tab.columns if c.startswith("_")], errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="只跑 9 档角点")
    parser.add_argument("--out-dir", default=str(OUT))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = choose_files(args.quick)
    print(f"identify {len(files)} files from {DATA}")
    rows = []
    plot_names = {
        "nmc100ah_pybamm_s04_t05_soc054_T+23.csv",
        "nmc100ah_pybamm_s04_t00_soc054_T-10.csv",
        "nmc100ah_pybamm_s09_t00_soc010_T-10.csv",
        "nmc100ah_pybamm_s00_t05_soc090_T+23.csv",
    }
    for k, path in enumerate(files, 1):
        rec = identify_one(path)
        print(
            f"[{k:3d}/{len(files)}] {path.name}  "
            f"R0_1C={rec.get('r0_1c_ohm', np.nan)*1e3:5.2f} mΩ  "
            f"η120={rec.get('eta120_mv', np.nan):6.1f} mV  "
            f"1RC={rec.get('ltis1_rmse_mv', np.nan):5.1f}  "
            f"2RC={rec.get('ltis2_rmse_mv', np.nan):5.1f} mV  "
            f"τ=({rec.get('ltis2_tau1_s', np.nan):4.1f},{rec.get('ltis2_tau2_s', np.nan):5.0f})"
        )
        if path.name in plot_names and "_uh2" in rec:
            plot_case(rec, out_dir / f"case_{path.stem}.png")
        rows.append(rec)
    tab = pd.DataFrame(drop_plot_cols(pd.DataFrame(rows)))
    csv_path = out_dir / "params.csv"
    tab.to_csv(csv_path, index=False, float_format="%.6g")
    summary = summarize(tab)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_grid(tab, out_dir)
    print(f"wrote {csv_path}")
    print(json.dumps(summary["mid_mean"], indent=2))
    print("grid metrics p50:")
    for key in ("r0_1c_mohm", "eta120_mv", "ltis1_rmse_mv", "ltis2_rmse_mv", "ltis2_tau1_s", "ltis2_tau2_s"):
        print(" ", key, summary["metrics"][key])


if __name__ == "__main__":
    sys.exit(main())
