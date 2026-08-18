"""100 Ah NMC 电芯一阶 ECM 参数集。

数值对应 Doc/01-b-NMC100Ah_ECM参数规范.md，参考点为：
SOC = 50%，T = 25 °C，I = 100 A（1C 放电），BOL。

这些是通用方壳/软包 100 Ah NMC 的模板参数，不是某一款商用电芯的出厂值。
标定真实电芯时，只改本文件中的数值或从 JSON 加载，不必改模型结构。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
import json


PARAM_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class CellSpec:
    """电芯铭牌与符号约定。"""

    name: str = "NMC-100Ah-Generic"
    chemistry: str = "NMC"
    capacity_ah: float = 100.0
    v_min: float = 2.80
    v_nom: float = 3.67
    v_max: float = 4.20
    # 电流符号：放电为正，充电为负
    discharge_positive: bool = True


@dataclass(frozen=True)
class ValidityRange:
    """模型声明的有效输入域。超出后默认裁剪，strict 模式下报错。"""

    soc_min: float = 0.02
    soc_max: float = 0.98
    t_min_c: float = -20.0
    t_max_c: float = 55.0
    i_min_a: float = -300.0  # -3C
    i_max_a: float = 300.0   # +3C


@dataclass(frozen=True)
class ReferencePoint:
    """所有修正因子在该点归一化为 1。"""

    soc: float = 0.50
    t_celsius: float = 25.0
    i_a: float = 100.0  # 1C 放电


@dataclass(frozen=True)
class SocShape:
    """U 形 SOC 修正：两端指数翘起 + 二次项。

    raw(s) = 1 + a_low*exp(-s/s_low) + a_high*exp(-(1-s)/s_high)
             + quad*(s - 0.5)^2
    再除以 raw(s_ref)，使参考点处因子为 1。
    """

    a_low: float
    s_low: float
    a_high: float
    s_high: float
    quad: float


@dataclass(frozen=True)
class PhaseBump:
    """高镍高 SOC 相变附近的局部凸起（高斯）。"""

    amplitude: float
    center: float
    width: float

    def disabled(self) -> bool:
        return abs(self.amplitude) < 1e-15


@dataclass(frozen=True)
class ArrheniusTemp:
    """f_T = exp[Ea/R * (1/T - 1/T_ref)]，T 为开尔文。

    Ea > 0：降温电阻升高；Ea < 0：降温电容下降。
    """

    ea_j_per_mol: float


@dataclass(frozen=True)
class CurrentShape:
    """电流修正。kind 见模型实现：identity / linear_soft / bv_asinh。"""

    kind: str
    # linear_soft: 1 / (1 + k * |I|/I_1C)，再对参考电流归一化
    k: float = 0.0
    # bv_asinh: asinh(|I|/I_s) / (|I|/I_s)，再对参考电流归一化
    i_s_a: float = 50.0


@dataclass(frozen=True)
class DirectionShape:
    """充放不对称。各自在 SOC=s_ref 处归一化为 1。

    放电(I>=0): 1 + d_low * exp(-s / s_d)
    充电(I<0) : 1 + c_high * exp(-(1-s) / s_c)
    """

    d_low: float
    s_d: float
    c_high: float
    s_c: float


@dataclass(frozen=True)
class ParameterChannel:
    """单个输出通道：R0 / R1 / C1。"""

    name: str
    unit: str
    ref_value: float
    soc: SocShape
    phase: PhaseBump
    temperature: ArrheniusTemp
    current: CurrentShape
    direction: DirectionShape


@dataclass(frozen=True)
class ECMParamSet:
    """完整参数集，可 JSON 读写。"""

    schema_version: str = PARAM_SCHEMA_VERSION
    cell: CellSpec = field(default_factory=CellSpec)
    validity: ValidityRange = field(default_factory=ValidityRange)
    reference: ReferencePoint = field(default_factory=ReferencePoint)
    gas_constant: float = 8.314462618  # J/(mol·K)
    r0: ParameterChannel = field(
        default_factory=lambda: ParameterChannel(
            name="R0",
            unit="ohm",
            ref_value=8.0e-4,
            soc=SocShape(a_low=0.55, s_low=0.08, a_high=0.18, s_high=0.07, quad=0.20),
            phase=PhaseBump(amplitude=0.08, center=0.93, width=0.03),
            temperature=ArrheniusTemp(ea_j_per_mol=16_000.0),
            current=CurrentShape(kind="linear_soft", k=0.06, i_s_a=50.0),
            direction=DirectionShape(d_low=0.10, s_d=0.15, c_high=0.08, s_c=0.12),
        )
    )
    r1: ParameterChannel = field(
        default_factory=lambda: ParameterChannel(
            name="R1",
            unit="ohm",
            ref_value=6.5e-4,
            soc=SocShape(a_low=1.80, s_low=0.07, a_high=0.70, s_high=0.06, quad=0.35),
            phase=PhaseBump(amplitude=0.15, center=0.93, width=0.03),
            temperature=ArrheniusTemp(ea_j_per_mol=32_000.0),
            current=CurrentShape(kind="bv_asinh", k=0.0, i_s_a=50.0),
            direction=DirectionShape(d_low=0.45, s_d=0.12, c_high=0.55, s_c=0.10),
        )
    )
    c1: ParameterChannel = field(
        default_factory=lambda: ParameterChannel(
            name="C1",
            unit="farad",
            ref_value=2.8e4,
            soc=SocShape(a_low=-0.20, s_low=0.10, a_high=-0.12, s_high=0.08, quad=-0.10),
            phase=PhaseBump(amplitude=-0.05, center=0.93, width=0.03),
            temperature=ArrheniusTemp(ea_j_per_mol=-3_000.0),
            current=CurrentShape(kind="identity", k=0.0, i_s_a=50.0),
            direction=DirectionShape(d_low=0.0, s_d=0.20, c_high=0.0, s_c=0.20),
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=indent, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ECMParamSet":
        payload = dict(data)
        payload.pop("schema_version", None)
        return cls(
            schema_version=str(data.get("schema_version", PARAM_SCHEMA_VERSION)),
            cell=CellSpec(**payload["cell"]),
            validity=ValidityRange(**payload["validity"]),
            reference=ReferencePoint(**payload["reference"]),
            gas_constant=float(payload.get("gas_constant", 8.314462618)),
            r0=_channel_from_dict(payload["r0"]),
            r1=_channel_from_dict(payload["r1"]),
            c1=_channel_from_dict(payload["c1"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ECMParamSet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def channels(self) -> tuple[ParameterChannel, ParameterChannel, ParameterChannel]:
        return self.r0, self.r1, self.c1


def _channel_from_dict(data: dict[str, Any]) -> ParameterChannel:
    known = {item.name for item in fields(ParameterChannel)}
    extra = set(data) - known
    if extra:
        raise ValueError(f"ParameterChannel 含未知字段: {sorted(extra)}")
    return ParameterChannel(
        name=data["name"],
        unit=data["unit"],
        ref_value=float(data["ref_value"]),
        soc=SocShape(**data["soc"]),
        phase=PhaseBump(**data["phase"]),
        temperature=ArrheniusTemp(**data["temperature"]),
        current=CurrentShape(**data["current"]),
        direction=DirectionShape(**data["direction"]),
    )


def default_param_set() -> ECMParamSet:
    return ECMParamSet()
