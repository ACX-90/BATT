"""读取 Data/grid 轨迹，堆成 (B, T) 张量。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import REPO_ROOT, TrainConfig


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        body = (line for line in fh if line.strip() and not line.lstrip().startswith("#"))
        reader = csv.DictReader(body)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 无表头: {path}")
        cols = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in cols:
                cols[name].append(row[name])
    out: dict[str, np.ndarray] = {}
    for name, values in cols.items():
        if name == "mode":
            out[name] = np.asarray(values, dtype=str)
        else:
            out[name] = np.asarray(values, dtype=float)
    return out


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureScaler":
        return cls(mean=np.asarray(data["mean"], dtype=float), std=np.asarray(data["std"], dtype=float))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class TrajectoryDataset(Dataset):
    def __init__(self, sequences: list[dict[str, np.ndarray]], scaler: FeatureScaler) -> None:
        self.sequences = sequences
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        feat = np.stack([seq["i"], seq["soc"], seq["t"]], axis=-1)
        feat_n = self.scaler.transform(feat).astype(np.float32)
        return {
            "x": torch.from_numpy(feat_n),
            "i": torch.from_numpy(seq["i"].astype(np.float32)),
            "u_ocv": torch.from_numpy(seq["u_ocv"].astype(np.float32)),
            "u_t": torch.from_numpy(seq["u_t"].astype(np.float32)),
            "r0": torch.from_numpy(seq["r0"].astype(np.float32)),
            "r1": torch.from_numpy(seq["r1"].astype(np.float32)),
            "c1": torch.from_numpy(seq["c1"].astype(np.float32)),
        }


def collate_traj(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch], dim=0) for k in keys}


def load_grid_sequences(cfg: TrainConfig) -> list[dict[str, np.ndarray]]:
    data_dir = cfg.data_path()
    index = data_dir / cfg.index_name
    paths: list[Path] = []
    if index.exists():
        with index.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rel = row.get("path") or row.get("file")
                if not rel:
                    continue
                p = Path(rel)
                paths.append(p if p.is_absolute() else REPO_ROOT / p)
    else:
        paths = sorted(data_dir.glob("*.csv"))
        paths = [p for p in paths if p.name != cfg.index_name]
    if not paths:
        raise FileNotFoundError(f"未找到轨迹 CSV：{data_dir}，请先运行 nmc100ah_gen_grid.py")

    i_key = "i_true_a" if cfg.use_true_inputs else "i_meas_a"
    soc_key = "soc_true" if cfg.use_true_inputs else "soc_meas"
    t_key = "t_true_c" if cfg.use_true_inputs else "t_meas_c"
    u_key = "u_t_meas_v" if cfg.voltage_target == "meas" else "u_t_true_v"

    seqs: list[dict[str, np.ndarray]] = []
    for path in paths:
        if not path.exists():
            alt = data_dir / path.name
            if not alt.exists():
                raise FileNotFoundError(path)
            path = alt
        raw = _load_csv(path)
        seqs.append(
            {
                "name": path.name,
                "i": raw[i_key],
                "soc": raw[soc_key],
                "t": raw[t_key],
                "u_ocv": raw["u_ocv_v"],
                "u_t": raw[u_key],
                "r0": raw["r0_ohm"],
                "r1": raw["r1_ohm"],
                "c1": raw["c1_f"],
            }
        )
    return seqs


def fit_scaler(sequences: list[dict[str, np.ndarray]]) -> FeatureScaler:
    feat = np.concatenate(
        [np.stack([s["i"], s["soc"], s["t"]], axis=-1) for s in sequences],
        axis=0,
    )
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return FeatureScaler(mean=mean, std=std)


def split_sequences(
    sequences: list[dict[str, np.ndarray]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(sequences))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(sequences) * val_ratio))) if len(sequences) > 1 else 0
    val_i = set(idx[:n_val].tolist())
    train = [sequences[i] for i in range(len(sequences)) if i not in val_i]
    val = [sequences[i] for i in range(len(sequences)) if i in val_i]
    if not train:
        train, val = sequences, []
    return train, val
