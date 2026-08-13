"""Plot 脚本共用路径、中文字体与读表。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PLOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PLOT_DIR.parent.parent
SIM_DIR = PLOT_DIR.parent / "Sim"
FIG_DIR = REPO_ROOT / "Fig"
DATA_DIR = REPO_ROOT / "Data"
DEFAULT_SIM_CSV = DATA_DIR / "nmc100ah_ecm_sim.csv"

if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))


def apply_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.30,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )


def save_figure(fig, path: Path, *, show: bool) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = FIG_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return path


def load_sim_csv(path: str | Path | None = None) -> dict[str, np.ndarray]:
    import csv

    csv_path = Path(path) if path else DEFAULT_SIM_CSV
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到仿真数据: {csv_path}，请先运行 python Src/Sim/nmc100ah_ecm_gen.py")

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        body = (line for line in fh if line.strip() and not line.lstrip().startswith("#"))
        reader = csv.DictReader(body)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 无表头: {csv_path}")
        columns = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in columns:
                columns[name].append(row[name])

    data: dict[str, np.ndarray] = {}
    for name, values in columns.items():
        if name == "mode":
            data[name] = np.asarray(values, dtype=str)
            continue
        try:
            data[name] = np.asarray(values, dtype=float)
        except ValueError:
            data[name] = np.asarray(values, dtype=str)
    return data


def mode_spans(time_s: np.ndarray, modes: np.ndarray) -> list[tuple[str, float, float]]:
    """把连续相同 mode 压成 (mode, t0, t1) 区间。"""
    spans: list[tuple[str, float, float]] = []
    if time_s.size == 0:
        return spans
    start = 0
    for i in range(1, time_s.size + 1):
        if i == time_s.size or modes[i] != modes[start]:
            t0 = float(time_s[start])
            t1 = float(time_s[i - 1] if i == time_s.size else time_s[i])
            spans.append((str(modes[start]), t0, t1))
            start = i
    return spans
