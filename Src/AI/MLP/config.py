"""MLP-ECM 训练配置。默认方案 B：只出 R0/R1，C1 固定。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


MLP_DIR = Path(__file__).resolve().parent
REPO_ROOT = MLP_DIR.parent.parent.parent


@dataclass
class TrainConfig:
    # A: MLP→R0,R1,C1   B: MLP→R0,R1，C1 固定   B+: 另学一个全局 C1
    scheme: str = "B"

    data_dir: str = "Data/grid"
    out_dir: str = "Data/ai_mlp"
    index_name: str = "index.csv"

    # 用真值列当网络输入（合成数据更干净）；现场可改 meas
    use_true_inputs: bool = True
    # 电压监督：meas=带噪声测量，true=无噪声真值
    voltage_target: str = "meas"

    dt_s: float = 0.1
    r0_min: float = 1.0e-5
    r1_min: float = 1.0e-5
    c1_min: float = 1.0e2
    c1_star: float = 2.8e4
    r0_ref: float = 8.0e-4
    r1_ref: float = 6.5e-4

    hidden: tuple[int, ...] = (64, 64)
    dropout: float = 0.0

    epochs: int = 40
    pretrain_epochs: int = 5  # 先用教师 R0/R1 预热，0 则跳过
    batch_size: int = 8
    lr: float = 2.0e-3
    lr_c1: float = 2.0e-4  # 仅 B+
    weight_decay: float = 1.0e-6
    grad_clip: float = 1.0
    lambda_smooth: float = 1.0e-3
    val_ratio: float = 0.2
    seed: int = 42

    # 截断 BPTT 窗口（步）。0 表示整条轨迹反传
    tbptt: int = 0

    device: str = "cpu"

    def resolve(self, rel: str) -> Path:
        path = Path(rel)
        return path if path.is_absolute() else REPO_ROOT / path

    def data_path(self) -> Path:
        return self.resolve(self.data_dir)

    def output_path(self) -> Path:
        return self.resolve(self.out_dir)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["hidden"] = list(self.hidden)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        if "hidden" in known:
            known["hidden"] = tuple(known["hidden"])
        return cls(**known)
