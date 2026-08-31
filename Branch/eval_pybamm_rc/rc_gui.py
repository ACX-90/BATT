"""2RC 参数调节 GUI：从 Data/grid_pybamm 选一条序列，调用 eval_nls 估出
默认 R0/R1/R2/tau1/tau2（rest VarPro + 边沿 R0 为推荐档，另存整段 VarPro、
五参数联立、整段 VarPro+BV 三组对照），滑块 / 输入框调参实时重绘预测与
实测电压对比，可一键回滚默认参数；BV 补偿开关在模型里加一只
I/sqrt(|I|/100+0.25) 形状的电流修正项（Rbv 可调，见 eval_nls §4）。

用法（仓库根目录）：

    python Branch/eval_pybamm_rc/rc_gui.py
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib

_use_orig = matplotlib.use
matplotlib.use = lambda *a, **k: None
try:
    import eval_nls
    import identify_rc
finally:
    matplotlib.use = _use_orig
matplotlib.use("TkAgg")

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

DATA = eval_nls.DATA
PARAMS = [
    ("r0", "R0 / mΩ", 1e-6, 1e-2, 1e3),
    ("r1", "R1 / mΩ", 1e-6, 1e-2, 1e3),
    ("tau1", "tau1 / s", 0.3, 40.0, 1.0),
    ("r2", "R2 / mΩ", 1e-6, 1e-2, 1e3),
    ("tau2", "tau2 / s", 12.0, 400.0, 1.0),
    ("rbv", "Rbv / mΩ", 1e-5, 1e-2, 1e3),
]
PARAM_KEYS = tuple(p[0] for p in PARAMS)
PRESETS = [
    ("rest", "rest VarPro + 边沿 R0（推荐）"),
    ("full", "整段 VarPro（整段电压最优）"),
    ("joint", "五参数联立（参数不可信，见 eval_nls §3）"),
    ("full_bv", "整段 VarPro + BV（边沿最优）"),
]
BV_PRESET_IDX = [n for n, _ in PRESETS].index("full_bv")


def bv_col(i: np.ndarray) -> np.ndarray:
    return i / np.sqrt(np.abs(i) / 100.0 + 0.25)


def list_csvs() -> list[Path]:
    return sorted(DATA.glob("nmc100ah_pybamm_s*_t*.csv"))


def spec_of(key: str) -> tuple[str, str, float, float, float]:
    for p in PARAMS:
        if p[0] == key:
            return p
    raise KeyError(key)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("eval_pybamm_rc — 2RC 参数调节")
        self.data = None
        self.presets = None
        self.params = {key: float(np.sqrt(lo * hi)) for key, _, lo, hi, _ in PARAMS}
        self.worker = None
        self.pending_path = None
        self.last_nonbv = 0
        self.q = queue.Queue()
        self._redraw_job = None
        self._scales = {}
        self._entries = {}

        top = ttk.Frame(root, padding=(8, 6, 8, 0))
        top.pack(fill="x")
        ttk.Label(top, text="CSV 序列:").pack(side="left")
        self.combo = ttk.Combobox(top, state="readonly", width=50)
        self.combo.pack(side="left", padx=(4, 6))
        self.combo.bind("<<ComboboxSelected>>", lambda e: self.start_fit())
        self.fit_btn = ttk.Button(top, text="评估 NLS 默认参数", command=self.start_fit)
        self.fit_btn.pack(side="left")

        row2 = ttk.Frame(root, padding=(8, 4, 8, 0))
        row2.pack(fill="x")
        ttk.Label(row2, text="默认参数组:").pack(side="left")
        self.preset_combo = ttk.Combobox(
            row2, state="readonly", width=38, values=[label for _, label in PRESETS]
        )
        self.preset_combo.current(0)
        self.preset_combo.pack(side="left", padx=(4, 6))
        self.reset_btn = ttk.Button(row2, text="回滚默认参数", command=self.rollback, state="disabled")
        self.reset_btn.pack(side="left")
        self.bv_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="BV 补偿", variable=self.bv_var, command=self.on_bv).pack(
            side="left", padx=(12, 0)
        )

        box = ttk.LabelFrame(root, text="RC 参数（拖动滑块，或在输入框改数值后回车）", padding=(8, 4))
        box.pack(fill="x", padx=8, pady=6)
        for row, (key, label, lo, hi, scale) in enumerate(PARAMS):
            ttk.Label(box, text=label, width=10, anchor="e").grid(
                row=row, column=0, sticky="e", padx=(0, 6), pady=2
            )
            sc = ttk.Scale(box, from_=0.0, to=1000.0, command=lambda v, k=key: self.on_slide(k))
            sc.grid(row=row, column=1, sticky="ew", pady=2)
            en = ttk.Entry(box, width=12, justify="right")
            en.grid(row=row, column=2, padx=(6, 0), pady=2)
            en.bind("<Return>", lambda e, k=key: self.on_entry(k))
            en.bind("<FocusOut>", lambda e, k=key: self.on_entry(k))
            self._scales[key] = sc
            self._entries[key] = en
        box.columnconfigure(1, weight=1)
        self.info_var = tk.StringVar(value="C1 = tau1/R1 = --    C2 = tau2/R2 = --")
        ttk.Label(box, textvariable=self.info_var, anchor="e").grid(
            row=len(PARAMS), column=0, columnspan=3, sticky="e", pady=(2, 0)
        )
        for key in PARAM_KEYS:
            self.sync_widgets(key)
        self.set_rbv_enabled(False)

        self.status_var = tk.StringVar(value="")
        ttk.Label(root, textvariable=self.status_var, padding=(8, 2), anchor="w").pack(fill="x")

        self.fig = Figure(figsize=(10.6, 5.6), dpi=100)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0], hspace=0.15)
        self.ax_v = self.fig.add_subplot(gs[0])
        self.ax_r = self.fig.add_subplot(gs[1], sharex=self.ax_v)
        for ax in (self.ax_v, self.ax_r):
            ax.grid(True, alpha=0.3)
        (self.line_u,) = self.ax_v.plot([], [], color="k", lw=1.1, label="实测 PyBaMM")
        (self.line_uh,) = self.ax_v.plot([], [], color="#e65100", lw=1.0, label="预测 2RC")
        self.ax_v.legend(loc="best", fontsize=8)
        self.ax_v.set_ylabel("Ut / V")
        self.ax_v.tick_params(labelbottom=False)
        (self.line_res,) = self.ax_r.plot([], [], color="#1565c0", lw=0.9)
        self.ax_r.axhline(0.0, color="0.5", lw=0.7)
        self.ax_r.set_xlabel("t / s")
        self.ax_r.set_ylabel("残差 / mV")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        self.toolbar.update()

        self.files = list_csvs()
        self.combo["values"] = [p.name for p in self.files]
        if self.files:
            self.combo.current(0)
            self.start_fit()
        else:
            self.status_var.set(f"未找到波形 CSV：{DATA}（先跑 nmc100ah_gen_grid.py --pybamm）")
        root.after(100, self.poll)

    def pos_of(self, key: str, v: float) -> float:
        _, _, lo, hi, _ = spec_of(key)
        v = min(max(v, lo), hi)
        return 1000.0 * (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))

    def val_of(self, key: str, pos: float) -> float:
        _, _, lo, hi, _ = spec_of(key)
        pos = min(max(pos, 0.0), 1000.0)
        return float(np.exp(np.log(lo) + pos / 1000.0 * (np.log(hi) - np.log(lo))))

    def fmt(self, key: str, v: float) -> str:
        _, _, _, _, scale = spec_of(key)
        return f"{v * scale:.4g}"

    def sync_widgets(self, key: str) -> None:
        v = self.params[key]
        self._scales[key].set(self.pos_of(key, v))
        en = self._entries[key]
        en.delete(0, "end")
        en.insert(0, self.fmt(key, v))

    def on_slide(self, key: str) -> None:
        self.params[key] = self.val_of(key, float(self._scales[key].get()))
        en = self._entries[key]
        en.delete(0, "end")
        en.insert(0, self.fmt(key, self.params[key]))
        self.schedule_redraw()

    def on_entry(self, key: str) -> None:
        _, _, lo, hi, scale = spec_of(key)
        try:
            v = float(self._entries[key].get().strip()) / scale
        except ValueError:
            self.sync_widgets(key)
            return
        self.params[key] = min(max(v, lo), hi)
        self.sync_widgets(key)
        self.schedule_redraw()

    def clamp(self, key: str, v: float) -> float:
        _, _, lo, hi, _ = spec_of(key)
        return min(max(v, lo), hi)

    def set_rbv_enabled(self, on: bool) -> None:
        self._scales["rbv"].state(("!disabled",) if on else ("disabled",))
        self._entries["rbv"].config(state="normal" if on else "disabled")

    def on_bv(self) -> None:
        if not self.presets:
            self.set_rbv_enabled(self.bv_var.get())
            self.schedule_redraw()
            return
        if self.bv_var.get():
            idx = self.preset_combo.current()
            if PRESETS[idx][0] != "full_bv":
                self.last_nonbv = idx
            self.preset_combo.current(BV_PRESET_IDX)
            self.apply_preset("full_bv")
        else:
            self.preset_combo.current(self.last_nonbv)
            self.apply_preset(PRESETS[self.last_nonbv][0])

    def rollback(self) -> None:
        if not self.presets:
            return
        name = PRESETS[self.preset_combo.current()][0]
        self.apply_preset(name)

    def apply_preset(self, name: str) -> None:
        p = self.presets[name]
        want_bv = name == "full_bv"
        self.bv_var.set(want_bv)
        self.set_rbv_enabled(want_bv)
        if not want_bv:
            self.last_nonbv = [n for n, _ in PRESETS].index(name)
        for key in PARAM_KEYS:
            self.params[key] = self.clamp(key, p[key])
            self.sync_widgets(key)
        self.schedule_redraw()

    def start_fit(self) -> None:
        if not self.files:
            return
        path = self.files[self.combo.current()]
        if self.worker is not None and self.worker.is_alive():
            self.pending_path = path
            self.status_var.set(f"上一条评估进行中，已排队：{path.name}")
            return
        self.status_var.set(f"评估 NLS：{path.name}（rest / 整段 / +BV / 五参数联立，约几秒）…")
        self.fit_btn.config(state="disabled")
        self.worker = threading.Thread(target=self._fit_worker, args=(path,), daemon=True)
        self.worker.start()

    def _fit_worker(self, path: Path) -> None:
        try:
            d = eval_nls.load(path)
            rest = eval_nls.rest_varpro(d)
            full = eval_nls.full_varpro(d, bv=False)
            full_bv = eval_nls.full_varpro(d, bv=True)
            joint = eval_nls.joint_nls(
                d,
                np.log(
                    np.array(
                        [
                            max(full["coef"].get("R0", 1e-3), 2e-4),
                            max(full["coef"].get("R1", 3e-4), 5e-5),
                            rest["tau1"],
                            max(full["coef"].get("R2", 4e-4), 5e-5),
                            rest["tau2"],
                        ]
                    )
                ),
            )
            r0_edge, _ = identify_rc.pick_r0_1c(identify_rc.edge_r0(d["i"], d["u"]))
            r0_src = "边沿"
            if not np.isfinite(r0_edge) or r0_edge <= 0.0:
                r0_edge = float(full["coef"].get("R0", joint["r0"]))
                r0_src = "整段 VarPro"
            rbv_default = float(full_bv["coef"].get("Rbv", 0.0))
            presets = {
                "rest": {
                    "r0": float(r0_edge),
                    "r1": rest["r1_ohm"],
                    "tau1": rest["tau1"],
                    "r2": rest["r2_ohm"],
                    "tau2": rest["tau2"],
                },
                "full": {
                    "r0": float(full["coef"].get("R0", 0.0)),
                    "r1": float(full["coef"].get("R1", 0.0)),
                    "tau1": full["tau1"],
                    "r2": float(full["coef"].get("R2", 0.0)),
                    "tau2": full["tau2"],
                },
                "joint": {
                    "r0": joint["r0"],
                    "r1": joint["r1"],
                    "tau1": joint["tau1"],
                    "r2": joint["r2"],
                    "tau2": joint["tau2"],
                },
                "full_bv": {
                    "r0": float(full_bv["coef"].get("R0", 0.0)),
                    "r1": float(full_bv["coef"].get("R1", 0.0)),
                    "tau1": full_bv["tau1"],
                    "r2": float(full_bv["coef"].get("R2", 0.0)),
                    "tau2": full_bv["tau2"],
                },
            }
            for p in presets.values():
                p["rbv"] = rbv_default
            self.q.put(
                ("ok", {"path": path, "d": d, "rest": rest, "full": full, "full_bv": full_bv,
                        "joint": joint, "presets": presets, "r0_src": r0_src})
            )
        except Exception:
            self.q.put(("err", traceback.format_exc()))

    def poll(self) -> None:
        try:
            kind, payload = self.q.get_nowait()
        except queue.Empty:
            pass
        else:
            if kind == "ok":
                self.on_fit_done(payload)
            else:
                self.fit_btn.config(state="normal")
                self.status_var.set("NLS 评估失败")
                messagebox.showerror("NLS 评估失败", payload)
        self.root.after(100, self.poll)

    def on_fit_done(self, payload: dict) -> None:
        self.data = payload["d"]
        self.presets = payload["presets"]
        rest, full, joint, full_bv = payload["rest"], payload["full"], payload["joint"], payload["full_bv"]
        d = self.data
        self.line_u.set_data(d["t"], d["u"])
        lo_u = float(np.min(d["u"]))
        hi_u = float(np.max(d["u"]))
        pad = 0.02 * (hi_u - lo_u) + 1e-3
        self.ax_v.set_xlim(float(d["t"][0]), float(d["t"][-1]))
        self.ax_v.set_ylim(lo_u - pad, hi_u + pad)
        self.ax_v.set_title(f"{d['name']}   SOC0={d['soc0']:.2f}  T={d['t_c']:.1f} C")
        self.reset_btn.config(state="normal")
        self.fit_btn.config(state="normal")
        self.status_var.set(
            f"rest RMSE={rest['rmse_mv']:.2f} mV  tau=({rest['tau1']:.1f},{rest['tau2']:.0f}) s | "
            f"整段 VarPro RMSE={full['rmse_mv']:.2f} mV | +BV RMSE={full_bv['rmse_mv']:.2f} mV | "
            f"五参数联立 RMSE={joint['rmse_mv']:.2f} mV | 默认 R0 取自{payload['r0_src']}"
        )
        self.apply_preset("rest")
        if self.pending_path is not None:
            path, self.pending_path = self.pending_path, None
            self.combo.current(self.files.index(path))
            self.start_fit()

    def update_info(self) -> None:
        p = self.params
        c1 = p["tau1"] / max(p["r1"], 1e-12)
        c2 = p["tau2"] / max(p["r2"], 1e-12)
        self.info_var.set(f"C1 = tau1/R1 = {c1:.4g} F        C2 = tau2/R2 = {c2:.4g} F")

    def predict(self, p: dict) -> np.ndarray:
        d = self.data
        up1 = eval_nls.unit_rc(d["i"], p["tau1"]) * p["r1"]
        up2 = eval_nls.unit_rc(d["i"], p["tau2"]) * p["r2"]
        uh = d["ocv"] - d["i"] * p["r0"] - up1 - up2
        if self.bv_var.get():
            uh = uh - p["rbv"] * bv_col(d["i"])
        return uh

    def schedule_redraw(self) -> None:
        if self._redraw_job is None:
            self._redraw_job = self.root.after(30, self.redraw)

    def redraw(self) -> None:
        self._redraw_job = None
        if self.data is None:
            return
        d = self.data
        uh = self.predict(self.params)
        rmse = eval_nls.rmse_mv(d["u"], uh)
        self.line_uh.set_data(d["t"], uh)
        self.line_uh.set_label("预测 2RC+BV" if self.bv_var.get() else "预测 2RC")
        self.ax_v.legend(loc="best", fontsize=8)
        self.line_res.set_data(d["t"], (uh - d["u"]) * 1e3)
        self.ax_r.relim()
        self.ax_r.autoscale_view()
        self.ax_r.set_title(f"预测 - 实测    当前 RMSE = {rmse:.2f} mV", fontsize=9)
        self.update_info()
        self.canvas.draw_idle()


def main() -> None:
    root = tk.Tk()
    root.geometry("1100x820")
    root.minsize(920, 660)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
