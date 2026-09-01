"""NLS 能不能从 grid_pybamm 估 (R0,R1,R2,tau1,tau2)。

对比：
  - 静置可分离 NLS（VarPro：tau 非线性，A 线性）
  - 整段 VarPro（tau 非线性，R 线性）
  - 整段五参数联立 NLS + 多初值
  - 整段 VarPro + 一只 BV 电流修正（检验失败是优化器还是模型）
  - 整段 VarPro + 表面 OCP（Chen2020 U(c_surf)，检验 §8.6 通电段直线）

用法（仓库根目录）：

    python Branch/eval_pybamm_rc/eval_nls.py
    python Branch/eval_pybamm_rc/eval_nls.py --only s00_t04
    python Branch/eval_pybamm_rc/eval_nls.py --quick
"""

from __future__ import annotations

import argparse
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
        "soc": df.soc_true.to_numpy(dtype=float),
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


# Chen2020 LGM50 electrode OCP. Stoich window from PyBaMM get_min_max_stoichiometries.
# α = (quasi-steady parabolic Δstoich) / I at 100 Ah 1C = 100 A; j is C-rate invariant.
_CHEN_X0 = 0.02634579027064577
_CHEN_X100 = 0.910618046652409
_CHEN_Y100 = 0.2638452245913301
_CHEN_Y0 = 0.853974674630047
_CHEN_AN = 1.653e-4  # Δx_ss / A
_CHEN_AP = 7.223e-4  # Δy_ss / A


def un_chen(sto: np.ndarray) -> np.ndarray:
    sto = np.asarray(sto, dtype=float)
    return (
        1.9793 * np.exp(-39.3631 * sto)
        + 0.2482
        - 0.0909 * np.tanh(29.8538 * (sto - 0.1234))
        - 0.04478 * np.tanh(14.9159 * (sto - 0.2769))
        - 0.0205 * np.tanh(30.4444 * (sto - 0.6103))
    )


def up_chen(sto: np.ndarray) -> np.ndarray:
    sto = np.asarray(sto, dtype=float)
    return (
        -0.8090 * sto
        + 4.4875
        - 0.0428 * np.tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * np.tanh(15.7890 * (sto - 0.3117))
        + 17.5842 * np.tanh(15.9308 * (sto - 0.3120))
    )


_CHEN_TAU_N = 5.86e-6**2 / (np.pi**2 * 3.3e-14)  # ~105 s
_CHEN_TAU_P = 5.22e-6**2 / (np.pi**2 * 4e-15)  # ~690 s


def surface_ocp(
    soc: np.ndarray,
    i: np.ndarray,
    tau_n: float,
    kn: float,
    tau_p: float | None = None,
    kp: float | None = None,
) -> np.ndarray:
    """Bulk SOC → average stoich; per-electrode diffusion state → surface OCP."""
    if tau_p is None:
        tau_p = tau_n
    if kp is None:
        kp = kn
    x = _CHEN_X0 + soc * (_CHEN_X100 - _CHEN_X0)
    y = _CHEN_Y0 + soc * (_CHEN_Y100 - _CHEN_Y0)
    qn = unit_rc(i, tau_n)
    qp = unit_rc(i, tau_p)
    xs = np.clip(x - kn * _CHEN_AN * qn, 1e-4, 0.999)
    ys = np.clip(y + kp * _CHEN_AP * qp, 1e-4, 0.999)
    return up_chen(ys) - un_chen(xs)


def segment_rmse(d: dict, uh: np.ndarray) -> dict:
    cmd, u = d["cmd"], d["u"]
    out = {}
    for c, name in ((1, "dis1c"), (2, "rest2"), (3, "chg0p5"), (5, "dis2c")):
        m = cmd == c
        out[name] = rmse_mv(u[m], uh[m]) if m.any() else float("nan")
    m = cmd == 1
    if m.any():
        out["dis1c_end_mv"] = float((uh[m][-1] - u[m][-1]) * 1e3)
    return out


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


def full_varpro_ocp(
    d: dict,
    *,
    tau_s: float,
    bv: bool = True,
    split: bool = False,
    split_tau: bool = False,
) -> dict:
    """Replace bulk OCV with Chen2020 surface OCP; rest τ2 is the default lag.

    split=False: one k on both electrodes, one τ.
    split=True: kn, kp independent.
    split_tau=True: also τn, τp independent (implies split).
    Linear part remains R0 + 1RC + optional BV.
    """
    i, u, soc = d["i"], d["u"], d["soc"]
    tau_s = float(np.clip(tau_s, 12.0, 400.0))
    if split_tau:
        split = True

    def design(tau1):
        cols = [i, unit_rc(i, tau1)]
        names = ["R0", "R1"]
        if bv:
            cols.append(i / np.sqrt(np.abs(i) / 100.0 + 0.25))
            names.append("Rbv")
        return np.column_stack(cols), names

    if not split:

        def unpack(z):
            tau1 = float(np.clip(np.exp(z[0]), 0.4, 40.0))
            k = float(np.clip(np.exp(z[1]), 0.05, 2.5))
            return tau1, tau_s, tau_s, k, k

        seeds = [
            np.array([np.log(t1), np.log(k)], dtype=float)
            for t1 in (6.0, 12.0, 20.0, 32.0)
            for k in np.linspace(0.25, 1.05, 9)
        ]
        names_nl = ["ln_tau1", "ln_k"]
        b_lo = np.log([0.4, 0.05])
        b_hi = np.log([40.0, 2.5])
    elif not split_tau:

        def unpack(z):
            tau1 = float(np.clip(np.exp(z[0]), 0.4, 40.0))
            kn = float(np.clip(np.exp(z[1]), 0.05, 4.0))
            kp = float(np.clip(np.exp(z[2]), 0.05, 2.5))
            return tau1, tau_s, tau_s, kn, kp

        seeds = [
            np.array([np.log(t1), np.log(kn), np.log(kp)], dtype=float)
            for t1 in (8.0, 16.0, 28.0)
            for kn in (0.25, 0.55, 0.90, 1.40, 2.20)
            for kp in (0.25, 0.55, 0.85, 1.20)
        ]
        names_nl = ["ln_tau1", "ln_kn", "ln_kp"]
        b_lo = np.log([0.4, 0.05, 0.05])
        b_hi = np.log([40.0, 4.0, 2.5])
    else:

        def unpack(z):
            tau1 = float(np.clip(np.exp(z[0]), 0.4, 40.0))
            kn = float(np.clip(np.exp(z[1]), 0.05, 4.0))
            kp = float(np.clip(np.exp(z[2]), 0.05, 2.5))
            tn = float(np.clip(np.exp(z[3]), 12.0, 250.0))
            tp = float(np.clip(np.exp(z[4]), 20.0, 400.0))
            return tau1, tn, tp, kn, kp

        seeds = [
            np.array([np.log(t1), np.log(kn), np.log(kp), np.log(tn), np.log(tp)], dtype=float)
            for t1 in (12.0, 24.0)
            for kn in (0.4, 0.9, 1.6)
            for kp in (0.4, 0.85)
            for tn in (40.0, tau_s, _CHEN_TAU_N)
            for tp in (tau_s, 170.0)
        ]
        names_nl = ["ln_tau1", "ln_kn", "ln_kp", "ln_tau_n", "ln_tau_p"]
        b_lo = np.log([0.4, 0.05, 0.05, 12.0, 20.0])
        b_hi = np.log([40.0, 4.0, 2.5, 250.0, 400.0])

    def fun(z):
        tau1, tn, tp, kn, kp = unpack(z)
        y = surface_ocp(soc, i, tn, kn, tp, kp) - u
        phi, _ = design(tau1)
        _, yhat = lin_r(phi, y, positive=True)
        return yhat - y

    best_z = None
    best_sse = np.inf
    for z in seeds:
        r = fun(z)
        sse = float(np.dot(r, r))
        if sse < best_sse:
            best_sse = sse
            best_z = z
    res = least_squares(
        fun, best_z, method="trf", jac="2-point", max_nfev=500, bounds=(b_lo, b_hi)
    )
    tau1, tn, tp, kn, kp = unpack(res.x)
    ocp = surface_ocp(soc, i, tn, kn, tp, kp)
    y = ocp - u
    phi, names = design(tau1)
    coef, yhat = lin_r(phi, y, positive=True)
    uh = ocp - yhat
    out = {n: float(c) for n, c in zip(names, coef)}
    ident = cov_from_jac(res.jac, res.fun, names_nl)
    rec = {
        "ok": bool(res.success),
        "tau1": tau1,
        "tau2": float("nan"),
        "tau_s": tau_s,
        "tau_n": tn,
        "tau_p": tp,
        "k": float(np.sqrt(kn * kp)),
        "kn": kn,
        "kp": kp,
        "split": split,
        "split_tau": split_tau,
        "rmse_mv": rmse_mv(u, uh),
        "nfev": int(res.nfev),
        "cond_phi": float(np.linalg.cond(phi)),
        "ident_tau": ident,
        "uh": uh,
        "ocp": ocp,
        "coef": out,
        "bv": bv,
    }
    rec.update(segment_rmse(d, uh))
    return rec


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


def plot_case(d, rest, full, full_bv, joint, land_rest, land_full, out: Path, full_ocp=None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6))
    ax = axes[0, 0]
    ax.plot(d["t"], d["u"], color="k", lw=1.1, label="PyBaMM")
    ax.plot(d["t"], full["uh"], color="#2e7d32", ls="-.", lw=1.0, label=f"VarPro 2RC {full['rmse_mv']:.1f} mV")
    ax.plot(d["t"], full_bv["uh"], color="#1565c0", ls="--", lw=1.0, label=f"VarPro+BV {full_bv['rmse_mv']:.1f} mV")
    ax.plot(d["t"], joint["uh"], color="#e65100", ls=":", lw=1.0, label=f"joint 5p {joint['rmse_mv']:.1f} mV")
    if full_ocp is not None:
        ax.plot(
            d["t"],
            full_ocp["uh"],
            color="#6a1b9a",
            ls="-",
            lw=1.05,
            label=f"VarPro+BV+OCP {full_ocp['rmse_mv']:.1f} mV",
        )
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


def _ocp_label(rec: dict) -> str:
    if rec.get("split_tau"):
        return (
            f"+OCP kn/kp {rec['rmse_mv']:.1f} mV  "
            f"kn={rec['kn']:.2f} kp={rec['kp']:.2f}  "
            f"τn/τp={rec['tau_n']:.0f}/{rec['tau_p']:.0f}"
        )
    if rec.get("split"):
        return f"+OCP kn/kp {rec['rmse_mv']:.1f} mV  kn={rec['kn']:.2f} kp={rec['kp']:.2f}"
    return f"+OCP k {rec['rmse_mv']:.1f} mV  k={rec['k']:.2f}"


def plot_ocp_case(d, rest, full, full_bv, full_ocp, out: Path, full_ocp_split=None, full_ocp_tau=None) -> None:
    """Ut + 1C zoom + residual + rest, for the §8.6 surface-OCP check."""
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6))
    t, u, cmd = d["t"], d["u"], d["cmd"]
    extra = []
    if full_ocp_split is not None:
        extra.append((full_ocp_split, "#c2185b", "-"))
    if full_ocp_tau is not None:
        extra.append((full_ocp_tau, "#ef6c00", "-"))

    ax = axes[0, 0]
    ax.plot(t, u, color="k", lw=1.15, label="PyBaMM")
    ax.plot(t, full["uh"], color="#2e7d32", ls="-.", lw=1.0, label=f"2RC {full['rmse_mv']:.1f} mV")
    ax.plot(t, full_bv["uh"], color="#1565c0", ls="--", lw=1.0, label=f"+BV {full_bv['rmse_mv']:.1f} mV")
    ax.plot(t, full_ocp["uh"], color="#6a1b9a", lw=1.05, label=_ocp_label(full_ocp))
    for rec, col, ls in extra:
        ax.plot(t, rec["uh"], color=col, ls=ls, lw=1.05, label=_ocp_label(rec))
    ax.set_title(f"{d['name']}\nSOC0={d['soc0']:.2f}  T={d['t_c']:.1f} C")
    ax.set_ylabel("Ut / V")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    m = cmd == 1
    ax.plot(t[m], u[m], color="k", lw=1.2, label="PyBaMM")
    ax.plot(t[m], full["uh"][m], color="#2e7d32", ls="-.", lw=1.0, label="2RC")
    ax.plot(t[m], full_bv["uh"][m], color="#1565c0", ls="--", lw=1.0, label="+BV")
    ax.plot(t[m], full_ocp["uh"][m], color="#6a1b9a", lw=1.1, label="OCP k")
    for rec, col, ls in extra:
        ax.plot(t[m], rec["uh"][m], color=col, ls=ls, lw=1.1, label="OCP kn/kp" if rec.get("split") and not rec.get("split_tau") else "OCP τn/τp")
    ends = [
        f"2RC {full.get('dis1c_end_mv', float('nan')):+.1f}",
        f"BV {full_bv.get('dis1c_end_mv', float('nan')):+.1f}",
        f"k {full_ocp['dis1c_end_mv']:+.1f}",
    ]
    if full_ocp_split is not None:
        ends.append(f"kn/kp {full_ocp_split['dis1c_end_mv']:+.1f}")
    if full_ocp_tau is not None:
        ends.append(f"τ {full_ocp_tau['dis1c_end_mv']:+.1f}")
    ax.set_title("cmd1 1C  end err  " + "  ".join(ends) + " mV")
    ax.set_ylabel("Ut / V")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, (full["uh"] - u) * 1e3, color="#2e7d32", lw=0.9, label="2RC")
    ax.plot(t, (full_bv["uh"] - u) * 1e3, color="#1565c0", lw=0.9, label="+BV")
    ax.plot(t, (full_ocp["uh"] - u) * 1e3, color="#6a1b9a", lw=0.95, label="OCP k")
    for rec, col, ls in extra:
        lab = "OCP kn/kp" if rec.get("split") and not rec.get("split_tau") else "OCP τn/τp"
        ax.plot(t, (rec["uh"] - u) * 1e3, color=col, ls=ls, lw=0.95, label=lab)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylabel("uh - Ut / mV")
    ax.set_xlabel("t / s")
    ax.set_title("full-seq residual")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(rest["t"], rest["eta"] * 1e3, color="k", lw=1.2, label="eta (bulk OCV)")
    ax.plot(rest["t"], rest["yhat"] * 1e3, color="#2e7d32", ls="-.", label=f"rest VarPro {rest['rmse_mv']:.2f} mV")
    ax.set_xlabel("rest t / s")
    ax.set_ylabel("eta / mV")
    tau_txt = f"OCP τs={full_ocp['tau_s']:.0f}s"
    if full_ocp_tau is not None:
        tau_txt = f"τn/τp={full_ocp_tau['tau_n']:.0f}/{full_ocp_tau['tau_p']:.0f}s"
    ax.set_title(f"rest  tau=({rest['tau1']:.1f},{rest['tau2']:.0f}) s  {tau_txt}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

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


def run_one(path: Path, *, quick: bool = False, plot: bool = True) -> dict:
    d = load(path)
    rest = rest_varpro(d)
    full = full_varpro(d, bv=False)
    full_bv = full_varpro(d, bv=True)
    full.update(segment_rmse(d, full["uh"]))
    full_bv.update(segment_rmse(d, full_bv["uh"]))
    full_ocp = full_varpro_ocp(d, tau_s=rest["tau2"], bv=True, split=False)
    full_ocp_split = full_varpro_ocp(d, tau_s=rest["tau2"], bv=True, split=True)
    full_ocp_tau = full_varpro_ocp(d, tau_s=rest["tau2"], bv=True, split=True, split_tau=True)
    full_ocp_nobv = full_varpro_ocp(d, tau_s=rest["tau2"], bv=False, split=False)
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
    joint.update(segment_rmse(d, joint["uh"]))
    if quick:
        starts = []
        rmses = np.array([joint["rmse_mv"]])
        tau2s = np.array([joint["tau2"]])
        valley_rest = [float("nan"), float("nan")]
        valley_full = [float("nan"), float("nan")]
    else:
        starts = multi_start(d, n=8, seed=7)
        rmses = np.array([s["rmse_mv"] for s in starts])
        tau2s = np.array([s["tau2"] for s in starts])
        land_rest = landscape(d, "rest")
        land_full = landscape(d, "full")
        if plot:
            plot_case(
                d,
                rest,
                full,
                full_bv,
                joint,
                land_rest,
                land_full,
                OUT / f"nls_{path.stem}.png",
                full_ocp=full_ocp,
            )

        def valley_tau2(z, tau2_grid):
            m = np.nanmin(z)
            ok = np.where(np.nanmin(z, axis=1) <= m + 0.3)[0]
            if ok.size == 0:
                return [float("nan"), float("nan")]
            return [float(tau2_grid[ok[0]]), float(tau2_grid[ok[-1]])]

        valley_rest = valley_tau2(land_rest[2], land_rest[1])
        valley_full = valley_tau2(land_full[2], land_full[1])
    if plot:
        plot_ocp_case(
            d,
            rest,
            full,
            full_bv,
            full_ocp,
            OUT / f"nls_ocp_{path.stem}.png",
            full_ocp_split=full_ocp_split,
            full_ocp_tau=full_ocp_tau,
        )

    def ocp_pack(rec):
        return {
            "tau1": rec["tau1"],
            "tau2": rec["tau2"],
            "tau_s": rec["tau_s"],
            "tau_n": rec.get("tau_n", rec["tau_s"]),
            "tau_p": rec.get("tau_p", rec["tau_s"]),
            "k": rec["k"],
            "kn": rec.get("kn", rec["k"]),
            "kp": rec.get("kp", rec["k"]),
            "split": bool(rec.get("split", False)),
            "split_tau": bool(rec.get("split_tau", False)),
            "coef_mohm": {key: val * 1e3 for key, val in rec["coef"].items()},
            "rmse_mv": rec["rmse_mv"],
            "nfev": rec["nfev"],
            "dis1c": rec["dis1c"],
            "rest2": rec["rest2"],
            "chg0p5": rec["chg0p5"],
            "dis2c": rec["dis2c"],
            "dis1c_end_mv": rec["dis1c_end_mv"],
        }

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
            "valley_tau2_0p3mV": valley_rest,
        },
        "full_varpro": {
            "tau1": full["tau1"],
            "tau2": full["tau2"],
            "coef_mohm": {k: v * 1e3 for k, v in full["coef"].items()},
            "rmse_mv": full["rmse_mv"],
            "cond_phi": full["cond_phi"],
            "nfev": full["nfev"],
            "ident_tau": slim_ident(full["ident_tau"]),
            "valley_tau2_0p3mV": valley_full,
            "dis1c": full["dis1c"],
            "dis1c_end_mv": full["dis1c_end_mv"],
        },
        "full_varpro_bv": {
            "tau1": full_bv["tau1"],
            "tau2": full_bv["tau2"],
            "coef_mohm": {k: v * 1e3 for k, v in full_bv["coef"].items()},
            "rmse_mv": full_bv["rmse_mv"],
            "nfev": full_bv["nfev"],
            "dis1c": full_bv["dis1c"],
            "dis1c_end_mv": full_bv["dis1c_end_mv"],
        },
        "full_varpro_ocp": ocp_pack(full_ocp),
        "full_varpro_ocp_split": ocp_pack(full_ocp_split),
        "full_varpro_ocp_tau": ocp_pack(full_ocp_tau),
        "full_varpro_ocp_nobv": ocp_pack(full_ocp_nobv),
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
    parser = argparse.ArgumentParser(description="NLS 估 2RC，并对照 BV / 表面 OCP")
    parser.add_argument("--only", default="", help="文件名子串过滤，如 s00_t04")
    parser.add_argument("--quick", action="store_true", help="跳过 landscape 和多初值")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    files = load_all()
    if args.only:
        files = [p for p in files if args.only in p.name]
        if not files:
            raise FileNotFoundError(f"没有匹配 --only {args.only!r} 的波形")
    print(f"共 {len(files)} 条波形，开始 NLS …", flush=True)
    reports = []
    for path in files:
        print("NLS", path.name, flush=True)
        rec = run_one(path, quick=args.quick, plot=not args.no_plot)
        reports.append(rec)
        ocp = rec["full_varpro_ocp"]
        sp = rec["full_varpro_ocp_split"]
        tp = rec["full_varpro_ocp_tau"]
        print(
            f"  rest {rec['rest']['rmse_mv']:.3f} mV  tau=({rec['rest']['tau1']:.1f},{rec['rest']['tau2']:.0f})  "
            f"full {rec['full_varpro']['rmse_mv']:.2f}  bv {rec['full_varpro_bv']['rmse_mv']:.2f}  "
            f"ocp {ocp['rmse_mv']:.2f} (k={ocp['k']:.2f}, end {ocp['dis1c_end_mv']:+.1f})  "
            f"kn/kp {sp['rmse_mv']:.2f} ({sp['kn']:.2f}/{sp['kp']:.2f}, 1C {sp['dis1c']:.1f}, end {sp['dis1c_end_mv']:+.1f})  "
            f"τn/τp {tp['rmse_mv']:.2f} ({tp['tau_n']:.0f}/{tp['tau_p']:.0f}, end {tp['dis1c_end_mv']:+.1f})"
        )
    rows = []
    for r in reports:
        ocp = r["full_varpro_ocp"]
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
                "full_dis1c": r["full_varpro"]["dis1c"],
                "full_dis1c_end": r["full_varpro"]["dis1c_end_mv"],
                "bv_rmse": r["full_varpro_bv"]["rmse_mv"],
                "bv_dis1c": r["full_varpro_bv"]["dis1c"],
                "bv_dis1c_end": r["full_varpro_bv"]["dis1c_end_mv"],
                "ocp_rmse": ocp["rmse_mv"],
                "ocp_k": ocp["k"],
                "ocp_tau_s": ocp["tau_s"],
                "ocp_dis1c": ocp["dis1c"],
                "ocp_rest2": ocp["rest2"],
                "ocp_dis1c_end": ocp["dis1c_end_mv"],
                "ocp_split_rmse": r["full_varpro_ocp_split"]["rmse_mv"],
                "ocp_kn": r["full_varpro_ocp_split"]["kn"],
                "ocp_kp": r["full_varpro_ocp_split"]["kp"],
                "ocp_split_dis1c": r["full_varpro_ocp_split"]["dis1c"],
                "ocp_split_rest2": r["full_varpro_ocp_split"]["rest2"],
                "ocp_split_end": r["full_varpro_ocp_split"]["dis1c_end_mv"],
                "ocp_tau_rmse": r["full_varpro_ocp_tau"]["rmse_mv"],
                "ocp_tau_n": r["full_varpro_ocp_tau"]["tau_n"],
                "ocp_tau_p": r["full_varpro_ocp_tau"]["tau_p"],
                "ocp_tau_kn": r["full_varpro_ocp_tau"]["kn"],
                "ocp_tau_kp": r["full_varpro_ocp_tau"]["kp"],
                "ocp_tau_dis1c": r["full_varpro_ocp_tau"]["dis1c"],
                "ocp_tau_rest2": r["full_varpro_ocp_tau"]["rest2"],
                "ocp_tau_end": r["full_varpro_ocp_tau"]["dis1c_end_mv"],
                "ocp_nobv_rmse": r["full_varpro_ocp_nobv"]["rmse_mv"],
                "joint_rmse": r["joint5"]["rmse_mv"],
                "joint_cond_jtj": r["joint5"]["ident"]["cond_jtj"],
                "ms_tau2_min": r["multistart"]["tau2_min"],
                "ms_tau2_max": r["multistart"]["tau2_max"],
                "ms_rmse_span": r["multistart"]["rmse_max"] - r["multistart"]["rmse_min"],
            }
        )
    if not args.only:
        pd.DataFrame(rows).to_csv(OUT / "nls_ocp_summary.csv", index=False, float_format="%.6g")
        print("wrote", OUT / "nls_ocp_summary.csv")
        if not args.quick:
            (OUT / "nls_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
            pd.DataFrame(rows).to_csv(OUT / "nls_summary.csv", index=False, float_format="%.6g")
            print("wrote", OUT / "nls_report.json")


if __name__ == "__main__":
    main()
