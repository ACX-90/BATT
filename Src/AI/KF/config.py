"""EKF / 融合滤波配置。离散化与 Src/Sim/nmc100ah_gen.py 一致。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

KF_DIR = Path(__file__).resolve().parent
REPO_ROOT = KF_DIR.parent.parent.parent


@dataclass
class KfConfig:
    dt_s: float = 0.1
    capacity_ah: float = 100.0  # 与仿真默认一致。容量错配走 run.py --capacity-scale，不要改这里

    # 过程噪声（每步方差）。s 略大才能靠电压纠安时漂。
    q_s: float = 1.0e-8
    q_up: float = 1.0e-6
    q_dr0: float = 1.0e-12

    # 测量噪声标准差 / V。仿真模板 0.5 mV，车上取 2~10 mV。
    rv_std: float = 0.5e-3

    # 初值方差
    p0_s: float = 2.5e-3  # (0.05)^2
    p0_up: float = 2.5e-3
    p0_dr0: float = 1.0e-8

    # 平台区 |dOCV/ds| 小时放大 Rv，避免猛纠 SOC
    schedule_rv: bool = True
    slope_min: float = 0.20
    rv_max_scale: float = 25.0

    # SOC 增益上限（每伏特创新）。0 表示不限制
    ks_max: float = 2.0

    # 慢变局部电阻残差，默认关（文档一期不做）
    estimate_dr0: bool = False

    soc_min: float = 0.0
    soc_max: float = 1.0

    def resolve(self, rel: str | Path) -> Path:
        path = Path(rel)
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def q_coulomb(self) -> float:
        return self.capacity_ah * 3600.0

    @property
    def rv(self) -> float:
        return float(self.rv_std) ** 2

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KfConfig":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)
