"""NLS 能不能从 grid_pybamm 估 (R0,R1,R2,tau1,tau2)。

对比：
  - 静置可分离 NLS（VarPro：tau 非线性，A 线性）
  - 整段 VarPro（tau 非线性，R 线性）
  - 整段五参数联立 NLS + 多初值
  - 整段 VarPro + 一只 BV 电流修正（检验失败是优化器还是模型）

用法（仓库根目录）：

    python Branch/eval_pybamm_rc/eval_nls.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "Data" / "grid_pybamm"
OUT = HERE / "out_nls"
DT = 0.1

def load_all() -> list[Path]:
    files = sorted(DATA.glob("nmc100ah_pybamm_s*_t*.csv"))
    if not files:
        raise FileNotFoundError(f"没有波形文件：{DATA}")
    return files


def load(path: Path) -> dict:
    df = pd.read_csv(path, comment="#")
    return {
        "t": df.time_s.to_numpy(dtype=float),
        "i": df.i_true_a.to_numpy(dtype=float),
        "u": df.u_t_true_v.to_numpy(dtype=float),
        "ocv": df.u_ocv_v.to_numpy(dtype=float),
        "cmd": df.cmd_id.to_numpy(dtype=int),
        "soc0": float(df.soc_true.iloc[0]),
        "t_c": float(df.t_true_c.mean()),
        "name": path.name,
    }


def rmse_mv(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)) * 1e3)


def unit_rc(i: np.ndarray, tau: float, dt: float = DT) -> np.ndarray:
    a = float(np.exp(-dt / max(tau, 1e-6)))
    return lfilter([1.0 - a], [1.0, -a], i)


def lin_r(phi: np.ndarray, y: np.ndarray, *, positive: bool) -> tuple[np.ndarray, np.ndarray]:
    if positive:
        coef, _ = nnls(phi, y)
    else:
        coef, *_ = np.linalg.lstsq(phi, y, rcond=None)
    return np.asarray(coef, dtype=float), phi @ coef


def cov_from_jac(jac: np.ndarray, resid: np.ndarray, names: list[str]) -> dict:
    n, p = jac.shape
    dof = max(n - p, 1)
    sse = float(np.dot(resid, resid))
    sigma2 = sse / dof
    h = jac.T @ jac
    try:
        cov = sigma2 * np.linalg.inv(h)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(h)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = cov / np.outer(se + 1e-30, se + 1e-30)
    return {
        "sigma_mv": float(np.sqrt(sigma2) * 1e3) if np.max(np.abs(resid)) < 50 else float(np.sqrt(sigma2)),
        "cond_jtj": float(np.linalg.cond(h)),
        "se": {n: float(s) for n, s in zip(names, se)},
        "corr": {names[i]: {names[j]: float(corr[i, j]) for j in range(p)} for i in range(p)},
        "svd_s": [float(x) for x in np.linalg.svd(jac, compute_uv=False)],
    }


# ---------------------------------------------------------------------------
# Rest-window separable NLS
# ---------------------------------------------------------------------------
def rest_varpro(d: dict) -> dict:
    m = d["cmd"] == 2
    t0 = d["t"][m][0]
    t = d["t"][m] - t0
    eta = d["ocv"][m] - d["u"][m]
    pulse = d["cmd"] == 1
    i_pre = float(d["i"][pulse].mean()) if pulse.any() else 100.0
    tp = float(pulse.sum() * DT)

    def unpack(z):
        tau1 = float(np.clip(np.exp(z[0]), 0.4, 40.0))
        tau2 = float(np.clip(max(np.exp(z[1]), 4.0 * tau1), 12.0, 400.0))
        return tau1, tau2

    def pack(tau1, tau2):
        return np.array([np.log(tau1), np.log(tau2)], dtype=float)

    def phi_of(tau1, tau2):
        return np.column_stack([np.exp(-t / tau1), np.exp(-t / tau2)])

    def fun(z):
        tau1, tau2 = unpack(z)
        a, yhat = lin_r(phi_of(tau1, tau2), eta, positive=False)
        return yhat - eta

    # grid seed
    best = None
    best_sse = np.inf
    for tau1 in np.logspace(np.log10(0.5), np.log10(25), 12):
        for tau2 in np.logspace(np.log10(20), np.log10(280), 14):
            if tau2 < 4 * tau1:
                continue
            a, yhat = lin_r(phi_of(tau1, tau2), eta, positive=False)
            sse = float(np.dot(eta - yhat, eta - yhat))
            if sse < best_sse:
                best_sse = sse
                best = (tau1, tau2, a)
    z0 = pack(best[0], best[1])
    res = least_squares(fun, z0, method="trf", jac="2-point", max_nfev=200)
    tau1, tau2 = unpack(res.x)
    a, yhat = lin_r(phi_of(tau1, tau2), eta, positive=False)
    r1 = a[0] / (i_pre * (1.0 - np.exp(-tp / tau1)))
    r2 = a[1] / (i_pre * (1.0 - np.exp(-tp / tau2)))
    ident = cov_from_jac(res.jac, res.fun, ["ln_tau1", "ln_tau2"])
    # A-space jacobian for (A1,A2,ln tau1, ln tau2)
    e1 = np.exp(-t / tau1)
    e2 = np.exp(-t / tau2)
    jac4 = np.column_stack([e1, e2, a[0] * (t / tau1) * e1, a[1] * (t / tau2) * e2])
    ident4 = cov_from_jac(jac4, yhat - eta, ["A1", "A2", "ln_tau1", "ln_tau2"])
    return {
        "ok": bool(res.success),
        "tau1": tau1,
        "tau2": tau2,
        "A1": float(a[0]),
        "A2": float(a[1]),
        "r1_ohm": float(r1),
        "r2_ohm": float(r2),
        "rmse_mv": rmse_mv(eta, yhat),
        "nfev": int(res.nfev),
        "cond_phi": float(np.linalg.cond(phi_of(tau1, tau2))),
        "ident_tau": ident,
        "ident_full": ident4,
        "t": t,
        "eta": eta,
        "yhat": yhat,
        "i_pre": i_pre,
        "tp": tp,
    }


# ---------------------------------------------------------------------------
# Full-sequence VarPro
# ---------------------------------------------------------------------------
def full_varpro(d: dict, *, bv: bool = False, r0_fixed: float | None = None) -> dict:
    i, ocv, u = d["i"], d["ocv"], d["u"]
    y = ocv - u

    def unpack(z):
        tau1 = float(np.clip(np.exp(z[0]), 0.4, 40.0))
        tau2 = float(np.clip(max(np.exp(z[1]), 4.0 * tau1), 12.0, 400.0))
        return tau1, tau2

    def design(tau1, tau2):
        cols = []
        names = []
        if r0_fixed is None:
            cols.append(i)
            names.append("R0")
        cols.append(unit_rc(i, tau1))
        names.append("R1")
        cols.append(unit_rc(i, tau2))
        names.append("R2")
        if bv:
            cols.append(i / np.sqrt(np.abs(i) / 100.0 + 0.25))
            names.append("Rbv")
        return np.column_stack(cols), names

    def fun(z):
        tau1, tau2 = unpack(z)
        phi, _ = design(tau1, tau2)
        yy = y if r0_fixed is None else y - r0_fixed * i
        coef, yhat = lin_r(phi, yy, positive=True)
        return yhat - yy

    z0 = np.array([np.log(8.0), np.log(90.0)], dtype=float)
    res = least_squares(fun, z0, method="trf", jac="2-point", max_nfev=250)
    tau1, tau2 = unpack(res.x)
    phi, names = design(tau1, tau2)
    yy = y if r0_fixed is None else y - r0_fixed * i
    coef, yhat = lin_r(phi, yy, positive=True)
    uh = ocv - (yhat if r0_fixed is None else yhat + r0_fixed * i)
    out = {n: float(c) for n, c in zip(names, coef)}
    if r0_fixed is not None:
        out["R0"] = float(r0_fixed)
    ident = cov_from_jac(res.jac, res.fun, ["ln_tau1", "ln_tau2"])
    return {
        "ok": bool(res.success),
        "tau1": tau1,
        "tau2": tau2,
        "rmse_mv": rmse_mv(u, uh),
        "nfev": int(res.nfev),
        "cond_phi": float(np.linalg.cond(phi)),
        "ident_tau": ident,
        "uh": uh,
        "coef": out,
        "bv": bv,
        "r0_fixed": r0_fixed is not None,
    }


def joint_nls(d: dict, z0: np.ndarray) -> dict:
    i, ocv, u = d["i"], d["ocv"], d["u"]

    def unpack(z):
        r0 = float(np.clip(np.exp(z[0]), 1e-4, 1e-2))
        r1 = float(np.clip(np.exp(z[1]), 2e-5, 1e-2))
        tau1 = float(np.clip(np.exp(z[2]), 0.3, 40.0))
        r2 = float(np.clip(np.exp(z[3]), 2e-5, 1e-2))
        tau2 = float(np.clip(max(np.exp(z[4]), 4.0 * tau1), 12.0, 400.0))
        return r0, r1, tau1, r2, tau2

    def sim(z):
        r0, r1, tau1, r2, tau2 = unpack(z)
        up1 = unit_rc(i, tau1) * r1
        up2 = unit_rc(i, tau2) * r2
        return ocv - i * r0 - up1 - up2

    def fun(z):
        return (sim(z) - u) * 1e3

    res = least_squares(fun, z0, method="trf", jac="2-point", max_nfev=300)
    r0, r1, tau1, r2, tau2 = unpack(res.x)
    uh = sim(res.x)
    ident = cov_from_jac(res.jac, res.fun / 1e3, ["ln_R0", "ln_R1", "ln_tau1", "ln_R2", "ln_tau2"])
    # jac was in mV; rescale se: fun is mV so sigma already mV, se of z is ok
    ident = cov_from_jac(res.jac, res.fun, ["ln_R0", "ln_R1", "ln_tau1", "ln_R2", "ln_tau2"])
    return {
        "ok": bool(res.success),
        "r0": r0,
        "r1": r1,
        "tau1": tau1,
        "r2": r2,
        "tau2": tau2,
        "rmse_mv": rmse_mv(u, uh),
        "nfev": int(res.nfev),
        "ident": ident,
        "cost": float(res.cost),
        "z": res.x.copy(),
        "uh": uh,
    }


def multi_start(d: dict, n: int = 8, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        z0 = np.log(
            np.array(
                [
                    rng.uniform(4e-4, 2.5e-3),
                    rng.uniform(8e-5, 8e-4),
                    rng.uniform(2.0, 20.0),
                    rng.uniform(1e-4, 1.2e-3),
                    rng.uniform(40.0, 220.0),
                ]
            )
        )
        rec = joint_nls(d, z0)
        rec["start"] = k
        rec.pop("uh", None)
        rec.pop("z", None)
        rec.pop("ident", None)
        rows.append(rec)
    return rows


def landscape(d: dict, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau1s = np.logspace(np.log10(1.0), np.log10(30.0), 28)
    tau2s = np.logspace(np.log10(25.0), np.log10(320.0), 28)
    z = np.full((tau2s.size, tau1s.size), np.nan)
    i, ocv, u = d["i"], d["ocv"], d["u"]
    m = d["cmd"] == 2
    t = d["t"][m] - d["t"][m][0]
    eta = d["ocv"][m] - d["u"][m]
    y = ocv - u
    for a, tau1 in enumerate(tau1s):
        for b, tau2 in enumerate(tau2s):
            if tau2 < 4.0 * tau1:
                continue
            if kind == "rest":
                phi = np.column_stack([np.exp(-t / tau1), np.exp(-t / tau2)])
                _, yhat = lin_r(phi, eta, positive=False)
                z[b, a] = rmse_mv(eta, yhat)
            else:
                phi = np.column_stack([i, unit_rc(i, tau1), unit_rc(i, tau2)])
                coef, yhat = lin_r(phi, y, positive=True)
                z[b, a] = rmse_mv(y, yhat)
    return tau1s, tau2s, z


def plot_case(d, rest, full, full_bv, joint, land_rest, land_full, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6))
    ax = axes[0, 0]
    ax.plot(d["t"], d["u"], color="k", lw=1.1, label="PyBaMM")
    ax.plot(d["t"], full["uh"], color="#2e7d32", ls="-.", lw=1.0, label=f"VarPro 2RC {full['rmse_mv']:.1f} mV")
    ax.plot(d["t"], full_bv["uh"], color="#1565c0", ls="--", lw=1.0, label=f"VarPro+BV {full_bv['rmse_mv']:.1f} mV")
    ax.plot(d["t"], joint["uh"], color="#e65100", ls=":", lw=1.0, label=f"joint 5p {joint['rmse_mv']:.1f} mV")
    ax.set_title(f"{d['name']}\nSOC0={d['soc0']:.2f}  T={d['t_c']:.1f} C")
    ax.set_ylabel("Ut / V")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(rest["t"], rest["eta"] * 1e3, color="k", lw=1.2, label="eta")
    ax.plot(rest["t"], rest["yhat"] * 1e3, color="#2e7d32", ls="-.", label=f"rest VarPro {rest['rmse_mv']:.2f} mV")
    ax.set_xlabel("rest t / s")
    ax.set_ylabel("eta / mV")
    ax.set_title(
        f"rest  tau=({rest['tau1']:.1f},{rest['tau2']:.0f}) s  "
        f"R=({rest['r1_ohm']*1e3:.2f},{rest['r2_ohm']*1e3:.2f}) mOhm"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    def contour(ax, tau1s, tau2s, z, star, title):
        zz = np.ma.masked_invalid(z)
        cs = ax.contourf(tau1s, tau2s, zz, levels=18, cmap="magma")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(star[0], star[1], marker="*", color="w", ms=12, mec="k")
        ax.set_xlabel("tau1 / s")
        ax.set_ylabel("tau2 / s")
        ax.set_title(title)
        fig.colorbar(cs, ax=ax, fraction=0.046)

    contour(
        axes[1, 0],
        *land_rest,
        (rest["tau1"], rest["tau2"]),
        "rest VarPro RMSE / mV",
    )
    contour(
        axes[1, 1],
        *land_full,
        (full["tau1"], full["tau2"]),
        "full-seq VarPro RMSE / mV",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def slim_ident(ident: dict) -> dict:
    return {
        "cond_jtj": ident["cond_jtj"],
        "svd_s": ident["svd_s"],
        "se": ident["se"],
        "corr": ident["corr"],
    }


def run_one(path: Path) -> dict:
    d = load(path)
    rest = rest_varpro(d)
    full = full_varpro(d, bv=False)
    full_bv = full_varpro(d, bv=True)
    z0 = np.log(
        np.array(
            [
                max(full["coef"].get("R0", 1e-3), 2e-4),
                max(full["coef"].get("R1", 3e-4), 5e-5),
                rest["tau1"],
                max(full["coef"].get("R2", 4e-4), 5e-5),
                rest["tau2"],
            ]
        )
    )
    joint = joint_nls(d, z0)
    starts = multi_start(d, n=8, seed=7)
    rmses = np.array([s["rmse_mv"] for s in starts])
    tau2s = np.array([s["tau2"] for s in starts])
    land_rest = landscape(d, "rest")
    land_full = landscape(d, "full")
    plot_case(
        d,
        rest,
        full,
        full_bv,
        joint,
        land_rest,
        land_full,
        OUT / f"nls_{path.stem}.png",
    )
    # valley width: tau2 span with RMSE <= min+0.3 mV on rest landscape
    t1s, t2s, zr = land_rest
    zf = land_full[2]
    def valley_tau2(z, tau2s):
        m = np.nanmin(z)
        ok = np.where(np.nanmin(z, axis=1) <= m + 0.3)[0]
        if ok.size == 0:
            return [float("nan"), float("nan")]
        return [float(tau2s[ok[0]]), float(tau2s[ok[-1]])]

    return {
        "file": path.name,
        "soc0": d["soc0"],
        "t_c": d["t_c"],
        "rest": {
            "tau1": rest["tau1"],
            "tau2": rest["tau2"],
            "r1_mohm": rest["r1_ohm"] * 1e3,
            "r2_mohm": rest["r2_ohm"] * 1e3,
            "rmse_mv": rest["rmse_mv"],
            "cond_phi": rest["cond_phi"],
            "nfev": rest["nfev"],
            "ident_tau": slim_ident(rest["ident_tau"]),
            "ident_A_tau": slim_ident(rest["ident_full"]),
            "valley_tau2_0p3mV": valley_tau2(zr, t2s),
        },
        "full_varpro": {
            "tau1": full["tau1"],
            "tau2": full["tau2"],
            "coef_mohm": {k: v * 1e3 for k, v in full["coef"].items()},
            "rmse_mv": full["rmse_mv"],
            "cond_phi": full["cond_phi"],
            "nfev": full["nfev"],
            "ident_tau": slim_ident(full["ident_tau"]),
            "valley_tau2_0p3mV": valley_tau2(zf, land_full[1]),
        },
        "full_varpro_bv": {
            "tau1": full_bv["tau1"],
            "tau2": full_bv["tau2"],
            "coef_mohm": {k: v * 1e3 for k, v in full_bv["coef"].items()},
            "rmse_mv": full_bv["rmse_mv"],
            "nfev": full_bv["nfev"],
        },
        "joint5": {
            "r0_mohm": joint["r0"] * 1e3,
            "r1_mohm": joint["r1"] * 1e3,
            "r2_mohm": joint["r2"] * 1e3,
            "tau1": joint["tau1"],
            "tau2": joint["tau2"],
            "rmse_mv": joint["rmse_mv"],
            "nfev": joint["nfev"],
            "ident": slim_ident(joint["ident"]),
        },
        "multistart": {
            "n": len(starts),
            "rmse_min": float(rmses.min()),
            "rmse_max": float(rmses.max()),
            "tau2_min": float(tau2s.min()),
            "tau2_max": float(tau2s.max()),
            "tau2_p50": float(np.median(tau2s)),
            "n_near_best": int(np.sum(rmses <= rmses.min() + 0.2)),
            "rows": [
                {
                    "start": s["start"],
                    "rmse_mv": s["rmse_mv"],
                    "r0_mohm": s["r0"] * 1e3,
                    "r1_mohm": s["r1"] * 1e3,
                    "r2_mohm": s["r2"] * 1e3,
                    "tau1": s["tau1"],
                    "tau2": s["tau2"],
                }
                for s in starts
            ],
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = load_all()
    print(f"共 {len(files)} 条波形，开始 NLS …", flush=True)
    reports = []
    for path in files:
        print("NLS", path.name, flush=True)
        rec = run_one(path)
        reports.append(rec)
        print(
            f"  rest {rec['rest']['rmse_mv']:.3f} mV  tau=({rec['rest']['tau1']:.1f},{rec['rest']['tau2']:.0f})  "
            f"full {rec['full_varpro']['rmse_mv']:.2f}  bv {rec['full_varpro_bv']['rmse_mv']:.2f}  "
            f"joint {rec['joint5']['rmse_mv']:.2f}  "
            f"starts tau2 {rec['multistart']['tau2_min']:.0f}-{rec['multistart']['tau2_max']:.0f}"
        )
    (OUT / "nls_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    # compact csv
    rows = []
    for r in reports:
        rows.append(
            {
                "file": r["file"],
                "t_c": r["t_c"],
                "soc0": r["soc0"],
                "rest_rmse": r["rest"]["rmse_mv"],
                "rest_tau1": r["rest"]["tau1"],
                "rest_tau2": r["rest"]["tau2"],
                "rest_cond_phi": r["rest"]["cond_phi"],
                "rest_cond_jtj": r["rest"]["ident_tau"]["cond_jtj"],
                "full_rmse": r["full_varpro"]["rmse_mv"],
                "full_tau1": r["full_varpro"]["tau1"],
                "full_tau2": r["full_varpro"]["tau2"],
                "full_cond_phi": r["full_varpro"]["cond_phi"],
                "bv_rmse": r["full_varpro_bv"]["rmse_mv"],
                "joint_rmse": r["joint5"]["rmse_mv"],
                "joint_cond_jtj": r["joint5"]["ident"]["cond_jtj"],
                "ms_tau2_min": r["multistart"]["tau2_min"],
                "ms_tau2_max": r["multistart"]["tau2_max"],
                "ms_rmse_span": r["multistart"]["rmse_max"] - r["multistart"]["rmse_min"],
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "nls_summary.csv", index=False, float_format="%.6g")
    print("wrote", OUT / "nls_report.json")


if __name__ == "__main__":
    main()
