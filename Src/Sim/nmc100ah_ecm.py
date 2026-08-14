"""100 Ah NMC 一阶 ECM 参数模型。

由 (电流 I, 温度 T, 荷电状态 SOC) 计算 (R0, R1, C1)。

    R_x(SOC, T, I) = R_x,ref · f_SOC · f_phase · f_T · f_I · f_dir
    C1 同结构。

因子在参考点 (SOC=50%, T=25°C, I=100 A 放电) 处均为 1。
方程与默认数值见 Doc/02-NMC100Ah_ECM参数规范.md。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warnings

import numpy as np

try:
    from nmc100ah_ecm_params import (
        ArrheniusTemp,
        CurrentShape,
        DirectionShape,
        ECMParamSet,
        ParameterChannel,
        PhaseBump,
        SocShape,
        default_param_set,
    )
except ImportError:  # python -m Src.Sim.nmc100ah_ecm
    from .nmc100ah_ecm_params import (
        ArrheniusTemp,
        CurrentShape,
        DirectionShape,
        ECMParamSet,
        ParameterChannel,
        PhaseBump,
        SocShape,
        default_param_set,
    )


ArrayLike = float | int | np.ndarray | Iterable[float]


@dataclass
class ECMResult:
    """一次或一批评估结果。标量输入则字段为 Python float。"""

    R0: float | np.ndarray
    R1: float | np.ndarray
    C1: float | np.ndarray
    tau1: float | np.ndarray
    soc: float | np.ndarray
    t_celsius: float | np.ndarray
    i_a: float | np.ndarray
    f_r0: dict[str, float | np.ndarray] | None = None
    f_r1: dict[str, float | np.ndarray] | None = None
    f_c1: dict[str, float | np.ndarray] | None = None

    @property
    def R0_mohm(self) -> float | np.ndarray:
        return self.R0 * 1e3

    @property
    def R1_mohm(self) -> float | np.ndarray:
        return self.R1 * 1e3

    def as_tuple(self) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
        return self.R0, self.R1, self.C1


class NMC100AhECM:
    """100 Ah NMC 电芯 ECM 参数映射。"""

    def __init__(self, params: ECMParamSet | None = None) -> None:
        self.params = params if params is not None else default_param_set()

    @classmethod
    def from_json(cls, path: str) -> "NMC100AhECM":
        return cls(ECMParamSet.from_json(path))

    def evaluate(
        self,
        i_a: ArrayLike | None = None,
        t_celsius: ArrayLike = 25.0,
        soc: ArrayLike = 0.5,
        *,
        i_c: ArrayLike | None = None,
        strict: bool = False,
        soc_in_percent: bool | None = None,
    ) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
        """计算 (R0[ohm], R1[ohm], C1[F])。

        Parameters
        ----------
        i_a
            电流，单位 A。放电为正，充电为负。与 i_c 二选一。
        t_celsius
            电芯温度，单位 °C。
        soc
            荷电状态。默认 0~1；若数值明显大于 1，按 0~100% 自动换算。
        i_c
            电流倍率。内部转换为 I = i_c * Qn。
        strict
            True 时输入超界抛错；False 时裁剪到有效域并告警。
        soc_in_percent
            True 强制按百分数；False 强制按 0~1；None 自动判断。
        """
        result = self.evaluate_full(
            i_a=i_a,
            t_celsius=t_celsius,
            soc=soc,
            i_c=i_c,
            strict=strict,
            soc_in_percent=soc_in_percent,
            return_factors=False,
        )
        return result.as_tuple()

    def evaluate_full(
        self,
        i_a: ArrayLike | None = None,
        t_celsius: ArrayLike = 25.0,
        soc: ArrayLike = 0.5,
        *,
        i_c: ArrayLike | None = None,
        strict: bool = False,
        soc_in_percent: bool | None = None,
        return_factors: bool = True,
    ) -> ECMResult:
        i_a = self._resolve_current(i_a, i_c)
        soc_arr = self._normalize_soc(soc, soc_in_percent)
        t_arr = np.asarray(t_celsius, dtype=float)
        i_arr = np.asarray(i_a, dtype=float)

        soc_b, t_b, i_b = np.broadcast_arrays(soc_arr, t_arr, i_arr)
        soc_u, t_u, i_u = self._enforce_domain(soc_b, t_b, i_b, strict=strict)
        scalar = soc_u.ndim == 0

        r0, f0 = self._eval_channel(self.params.r0, soc_u, t_u, i_u)
        r1, f1 = self._eval_channel(self.params.r1, soc_u, t_u, i_u)
        c1, f2 = self._eval_channel(self.params.c1, soc_u, t_u, i_u)
        tau1 = r1 * c1

        if scalar:
            r0, r1, c1, tau1 = (float(r0), float(r1), float(c1), float(tau1))
            soc_out, t_out, i_out = float(soc_u), float(t_u), float(i_u)
            if return_factors:
                f0 = {k: float(v) for k, v in f0.items()}
                f1 = {k: float(v) for k, v in f1.items()}
                f2 = {k: float(v) for k, v in f2.items()}
        else:
            soc_out, t_out, i_out = soc_u, t_u, i_u

        return ECMResult(
            R0=r0,
            R1=r1,
            C1=c1,
            tau1=tau1,
            soc=soc_out,
            t_celsius=t_out,
            i_a=i_out,
            f_r0=f0 if return_factors else None,
            f_r1=f1 if return_factors else None,
            f_c1=f2 if return_factors else None,
        )

    def build_map(
        self,
        soc: ArrayLike,
        t_celsius: ArrayLike,
        i_a: ArrayLike,
        *,
        strict: bool = False,
    ) -> dict[str, np.ndarray]:
        """在 (SOC, T, I) 网格上生成查找表。

        返回字典，每个字段形状为 (n_soc, n_t, n_i)。
        """
        soc_1d = np.atleast_1d(np.asarray(soc, dtype=float))
        t_1d = np.atleast_1d(np.asarray(t_celsius, dtype=float))
        i_1d = np.atleast_1d(np.asarray(i_a, dtype=float))
        soc_g, t_g, i_g = np.meshgrid(soc_1d, t_1d, i_1d, indexing="ij")
        result = self.evaluate_full(
            i_a=i_g, t_celsius=t_g, soc=soc_g, strict=strict, return_factors=False
        )
        return {
            "soc": soc_1d,
            "t_celsius": t_1d,
            "i_a": i_1d,
            "R0": np.asarray(result.R0),
            "R1": np.asarray(result.R1),
            "C1": np.asarray(result.C1),
            "tau1": np.asarray(result.tau1),
        }

    def export_csv(
        self,
        path: str,
        soc: ArrayLike,
        t_celsius: ArrayLike,
        i_a: ArrayLike,
        *,
        strict: bool = False,
    ) -> None:
        """导出扁平查找表，列：soc,t_celsius,i_a,R0,R1,C1,tau1。电阻单位 ohm。"""
        table = self.build_map(soc, t_celsius, i_a, strict=strict)
        soc_g, t_g, i_g = np.meshgrid(table["soc"], table["t_celsius"], table["i_a"], indexing="ij")
        header = "soc,t_celsius,i_a,R0_ohm,R1_ohm,C1_F,tau1_s"
        data = np.column_stack(
            [
                soc_g.ravel(),
                t_g.ravel(),
                i_g.ravel(),
                table["R0"].ravel(),
                table["R1"].ravel(),
                table["C1"].ravel(),
                table["tau1"].ravel(),
            ]
        )
        np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.8g")

    def _resolve_current(self, i_a: ArrayLike | None, i_c: ArrayLike | None) -> ArrayLike:
        if i_a is not None and i_c is not None:
            raise ValueError("i_a 与 i_c 只能提供一个")
        if i_a is None and i_c is None:
            return self.params.reference.i_a
        if i_c is not None:
            return np.asarray(i_c, dtype=float) * self.params.cell.capacity_ah
        return i_a  # type: ignore[return-value]

    def _normalize_soc(self, soc: ArrayLike, soc_in_percent: bool | None) -> np.ndarray:
        soc_arr = np.asarray(soc, dtype=float)
        if soc_in_percent is True:
            return soc_arr / 100.0
        if soc_in_percent is False:
            return soc_arr
        finite = soc_arr[np.isfinite(soc_arr)]
        if finite.size and np.nanmax(np.abs(finite)) > 1.5:
            return soc_arr / 100.0
        return soc_arr

    def _enforce_domain(
        self,
        soc: np.ndarray,
        t_celsius: np.ndarray,
        i_a: np.ndarray,
        *,
        strict: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lim = self.params.validity
        bounds = (
            ("SOC", soc, lim.soc_min, lim.soc_max),
            ("T", t_celsius, lim.t_min_c, lim.t_max_c),
            ("I", i_a, lim.i_min_a, lim.i_max_a),
        )
        clipped = []
        for name, arr, lo, hi in bounds:
            below = arr < lo
            above = arr > hi
            if np.any(below) or np.any(above):
                msg = f"{name} 超出有效域 [{lo}, {hi}]"
                if strict:
                    raise ValueError(msg)
                warnings.warn(msg + "，已裁剪", RuntimeWarning, stacklevel=3)
            clipped.append(np.clip(arr, lo, hi))
        return clipped[0], clipped[1], clipped[2]

    def _eval_channel(
        self,
        ch: ParameterChannel,
        soc: np.ndarray,
        t_celsius: np.ndarray,
        i_a: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        ref = self.params.reference
        f_soc = _normalized_soc(soc, ch.soc, ref.soc)
        f_phase = _phase_bump(soc, ch.phase)
        f_t = _arrhenius(t_celsius, ref.t_celsius, ch.temperature, self.params.gas_constant)
        f_i = _normalized_current(i_a, ref.i_a, ch.current, self.params.cell.capacity_ah)
        f_dir = _normalized_direction(soc, i_a, ch.direction, ref.soc)
        value = ch.ref_value * f_soc * f_phase * f_t * f_i * f_dir
        factors = {
            "soc": f_soc,
            "phase": f_phase,
            "temperature": f_t,
            "current": f_i,
            "direction": f_dir,
        }
        return value, factors


def _soc_raw(soc: np.ndarray, shape: SocShape) -> np.ndarray:
    return (
        1.0
        + shape.a_low * np.exp(-soc / shape.s_low)
        + shape.a_high * np.exp(-(1.0 - soc) / shape.s_high)
        + shape.quad * (soc - 0.5) ** 2
    )


def _normalized_soc(soc: np.ndarray, shape: SocShape, soc_ref: float) -> np.ndarray:
    raw = _soc_raw(soc, shape)
    ref = _soc_raw(np.asarray(soc_ref, dtype=float), shape)
    return raw / ref


def _phase_bump(soc: np.ndarray, bump: PhaseBump) -> np.ndarray:
    if bump.disabled():
        return np.ones_like(soc, dtype=float)
    z = (soc - bump.center) / bump.width
    return 1.0 + bump.amplitude * np.exp(-0.5 * z * z)


def _arrhenius(
    t_celsius: np.ndarray,
    t_ref_c: float,
    spec: ArrheniusTemp,
    gas_constant: float,
) -> np.ndarray:
    t_k = t_celsius + 273.15
    t_ref_k = t_ref_c + 273.15
    return np.exp(spec.ea_j_per_mol / gas_constant * (1.0 / t_k - 1.0 / t_ref_k))


def _current_raw(i_a: np.ndarray, spec: CurrentShape, capacity_ah: float) -> np.ndarray:
    kind = spec.kind.lower()
    if kind == "identity":
        return np.ones_like(i_a, dtype=float)
    abs_i = np.abs(i_a)
    if kind == "linear_soft":
        i_1c = max(capacity_ah, 1e-12)
        return 1.0 / (1.0 + spec.k * abs_i / i_1c)
    if kind == "bv_asinh":
        i_s = max(spec.i_s_a, 1e-12)
        x = abs_i / i_s
        return np.divide(
            np.arcsinh(x),
            x,
            out=np.ones_like(x, dtype=float),
            where=x >= 1e-12,
        )
    raise ValueError(f"未知电流修正类型: {spec.kind}")


def _normalized_current(
    i_a: np.ndarray,
    i_ref: float,
    spec: CurrentShape,
    capacity_ah: float,
) -> np.ndarray:
    raw = _current_raw(i_a, spec, capacity_ah)
    ref = _current_raw(np.asarray(i_ref, dtype=float), spec, capacity_ah)
    return raw / ref


def _direction_raw(soc: np.ndarray, i_a: np.ndarray, spec: DirectionShape) -> np.ndarray:
    discharge = 1.0 + spec.d_low * np.exp(-soc / spec.s_d)
    charge = 1.0 + spec.c_high * np.exp(-(1.0 - soc) / spec.s_c)
    return np.where(i_a >= 0.0, discharge, charge)


def _normalized_direction(
    soc: np.ndarray,
    i_a: np.ndarray,
    spec: DirectionShape,
    soc_ref: float,
) -> np.ndarray:
    raw = _direction_raw(soc, i_a, spec)
    # 充、放各自用本方向在 s_ref 处的值归一化，使中段 SOC 对齐参考点
    ref_dis = _direction_raw(np.asarray(soc_ref), np.asarray(1.0), spec)
    ref_chg = _direction_raw(np.asarray(soc_ref), np.asarray(-1.0), spec)
    ref = np.where(i_a >= 0.0, ref_dis, ref_chg)
    return raw / ref


__all__ = ["NMC100AhECM", "ECMResult"]
