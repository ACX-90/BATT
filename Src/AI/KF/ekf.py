"""最小 EKF：状态 [s, U_p]（可选慢变 δR0）。

预测用安时 + 一阶 ECM；测量是端电压。R0/R1 由调用方按预测 SOC 提供，
本拍不回代后验再算电阻，避免代数环（Doc/03-c §2.2）。
"""

# 让类型注解变成字符串延迟求值，支持前向引用、更干净的写法
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import KfConfig
from .ocv import docv_ds, ocv_nmc


@dataclass  # 装饰器，把普通类变成数据类
class EkfStep:
    s_pred: float
    s_post: float
    u_p_pred: float
    u_p_post: float
    d_r0: float
    r0_used: float
    r1: float
    c1: float
    u_ocv: float
    u_t_pri: float
    u_t_post: float
    e_pri: float
    e_post: float
    nis: float
    docv_ds: float
    k_s: float
    k_up: float
    alpha: float
    s: float  # HPH^T + Rv


def ecm1_step(
    i_a: float,
    u_p: float,
    r0: float,
    r1: float,
    c1: float,
    u_ocv: float,
    dt_s: float,
) -> tuple[float, float, float]:
    """一阶 Thevenin 单步。放电电流为正。

    U_p[k] = α U_p[k-1] + R1 (1-α) I
    U_t    = OCV − I R0 − U_p
    α = exp(−Δt / (R1 C1))
    """
    tau = max(float(r1) * float(c1), 1.0e-6)
    alpha = float(np.exp(-dt_s / tau))
    u_p_next = alpha * float(u_p) + float(r1) * (1.0 - alpha) * float(i_a)
    u_t = float(u_ocv) - float(i_a) * float(r0) - u_p_next
    return u_p_next, u_t, alpha


class SocUpEKF:
    # KfConfig | None：3.10+ 联合类型，None 则内部 new 默认配置。
    def __init__(self, cfg: KfConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else KfConfig()
        self.n = 3 if self.cfg.estimate_dr0 else 2
        self.s = 0.5
        self.u_p = 0.0
        self.d_r0 = 0.0
        self.P = np.eye(self.n)  # 单位矩阵
        self._i = 0.0  # 单下划线前缀：约定“内部使用”，外部尽量别直接改
        self._s_pred = 0.5
        self.reset(0.5, 0.0)

    def reset(self, s0: float, u_p0: float = 0.0, d_r0: float = 0.0) -> None:
        cfg = self.cfg
        self.s = float(np.clip(s0, cfg.soc_min, cfg.soc_max))
        self.u_p = float(u_p0)
        self.d_r0 = float(d_r0)
        diag = [cfg.p0_s, cfg.p0_up]
        if self.n == 3:
            diag.append(cfg.p0_dr0)
        self.P = np.diag(diag).astype(float)  # 对角协方差矩阵，并强制为浮点

    # 先验SOC，SOC[k+1] = SOC[k] - I*dt/Q，基本库仑计公式
    def predict_soc(self, i_a: float) -> float:
        """安时预测 s_{k|k-1}。MLP 必须用这个 SOC，不要用后验。"""
        cfg = self.cfg
        self._i = float(i_a)
        # np.clip 把 SOC 限制在合法区间，防止数值越界
        self._s_pred = float(
            np.clip(self.s - self._i * cfg.dt_s / cfg.q_coulomb, cfg.soc_min, cfg.soc_max)
        )
        return self._s_pred

    # 状态更新，必须用predict_soc产生先验SOC，输出给MLP做参数估算
    # 再用MLP参数输入给update，计算后验状态
    # 若用后验 SOC 回代再算电阻，会形成代数环（同一拍计算自己依赖自己，导致方程无法按时间顺序求解）
    def update(self, t_celsius: float, u_meas: float, r0: float, r1: float, c1: float) -> EkfStep:
        cfg = self.cfg
        i_a = self._i
        s_pred = self._s_pred

        ##### 先验状态预测 #####
        # 使用已估算的慢变 δR0，吸收「这一趟比表偏 50 µΩ」，不改曲面，Doc/04-a §7.4
        r0_used = float(r0) + (self.d_r0 if cfg.estimate_dr0 else 0.0)
        # SOC查表获取OCV
        u_ocv = float(ocv_nmc(s_pred, t_celsius))
        # dOCV/dSOC斜率，用于计算雅可比
        slope = float(docv_ds(s_pred, t_celsius))
        # ECM状态推进
        u_p_pred, u_t_pri, alpha = ecm1_step(
            i_a, self.u_p, r0_used, r1, c1, u_ocv, cfg.dt_s
        )
        # 创新 e = U_meas − U_t⁻（先验电压残差）
        e_pri = float(u_meas) - u_t_pri
        # 先验状态误差协方差 P⁻ = F P Fᵀ + Q（对状态估计有多不确定）
        F = np.eye(self.n)  # 状态转移矩阵 F
        # 对角其余为 1：SOC 是运动学、δR0 是随机游走；只有 Up 有 RC 动力学，所以 F[1,1]=α
        F[1, 1] = alpha
        # 过程噪声 Q：每拍允许 SOC / Up / δR0 各自漂多少（对角、互不相关）
        Q = np.diag([cfg.q_s, cfg.q_up] + ([cfg.q_dr0] if self.n == 3 else []))
        P_pri = F @ self.P @ F.T + Q

        ##### 测量方程处理 #####
        # 测量噪声调度，OCV 斜率很小时（平台区），电压对 SOC 不敏感，
        # 人为放大测量噪声，防止滤波器过度相信电压。
        rv = cfg.rv
        if cfg.schedule_rv:
            scale = (cfg.slope_min / max(abs(slope), 1.0e-6)) ** 2
            rv = rv * min(max(scale, 1.0), cfg.rv_max_scale)
        # 测量雅可比 H = ∂Ut/∂[s, Up, (δR0)] = [slope, −1, (−I)]
        # |slope| 大：同样 SOC 误差变成更大电压残差，电压里 SOC 信息多；平台区近 0，信息很少
        # 「更信电压 / 更信安时」不单看 slope：平台区真正压低电压权重的是上面的 Rv 调度
        if self.n == 3:
            H = np.array([[slope, -1.0, -i_a]], dtype=float)
        else:
            H = np.array([[slope, -1.0]], dtype=float)
        # 创新协方差 S = H P⁻ Hᵀ + Rv：预期电压残差有多大波动
        # 两部分：状态不确定度投影到电压上，再加上测量噪声 Rv（不是「单纯的测量不确定度」）
        s_innov = float(np.asarray(H @ P_pri @ H.T).reshape(-1)[0] + rv)
        s_innov = max(s_innov, 1.0e-18)

        ##### 卡尔曼核心公式 #####
        # 卡尔曼增益 K = P⁻ Hᵀ / S；|K| 大更信测量，|K| 小更信预测
        K = (P_pri @ H.T) / s_innov
        if cfg.ks_max > 0.0:
            # 只限幅 SOC 增益 K_s，避免平台区 / 大创新时一步把 s 拉飞；K_up、K_δR0 不裁
            K[0, 0] = float(np.clip(K[0, 0], -cfg.ks_max, cfg.ks_max))
        # x⁺ = x⁻ + K e：SOC、Up、δR0 同一套线性修正
        dx = (K * e_pri).reshape(-1)
        s_post = float(np.clip(s_pred + dx[0], cfg.soc_min, cfg.soc_max))
        u_p_post = float(u_p_pred + dx[1])
        d_r0_post = float(self.d_r0 + dx[2]) if self.n == 3 else 0.0
        # 后验协方差更新（数值更稳定的Joseph形式）
        # P_pri：我对当前状态有多不确定
        # I-KH：电压测量告诉我一部分信息，不确定性被压缩
        # KRK^T：测量本身也有噪声，不能把不确定性压到 0
        i_kh = np.eye(self.n) - K @ H
        self.P = i_kh @ P_pri @ i_kh.T + (K * rv) @ K.T

        ##### 统一更新EKF状态，计算后验输出 #####
        self.s = s_post
        self.u_p = u_p_post
        self.d_r0 = d_r0_post
        # 后验 δR0 是否叠进 R0，由 estimate_dr0 决定（这里不是「再估一次」）
        r0_post = float(r0) + (d_r0_post if cfg.estimate_dr0 else 0.0)
        # 用后验 OCV(s⁺)、R0⁺、Up⁺ 算诊断端电压；不回代 MLP
        u_t_post = float(ocv_nmc(s_post, t_celsius)) - i_a * r0_post - u_p_post

        ##### 诊断日志 #####
        # e_post、NIS = e²/S：滤波器健康度，只记账，不参与滤波、不当 MLP 损失
        e_post = float(u_meas) - u_t_post
        nis = (e_pri * e_pri) / s_innov

        return EkfStep(
            s_pred=s_pred,
            s_post=s_post,
            u_p_pred=u_p_pred,
            u_p_post=u_p_post,
            d_r0=d_r0_post,
            r0_used=r0_used,
            r1=float(r1),
            c1=float(c1),
            u_ocv=u_ocv,
            u_t_pri=u_t_pri,
            u_t_post=u_t_post,
            e_pri=e_pri,
            e_post=e_post,
            nis=float(nis),
            docv_ds=slope,
            k_s=float(K[0, 0]),
            k_up=float(K[1, 0]),
            alpha=alpha,
            s=s_innov,
        )


def selftest() -> dict[str, float]:
    """静置时用电压把错误的 SOC 初值拉回来。"""
    cfg = KfConfig(rv_std=0.5e-3, q_s=1.0e-8, schedule_rv=False)
    ekf = SocUpEKF(cfg)
    s_true = 0.80
    t_c = 25.0
    r0, r1, c1 = 8.0e-4, 6.5e-4, 2.8e4
    u_meas = float(ocv_nmc(s_true, t_c))
    ekf.reset(0.70, 0.0)
    for _ in range(80):
        ekf.predict_soc(0.0)
        step = ekf.update(t_c, u_meas, r0, r1, c1)
    err0 = abs(0.70 - s_true)
    err1 = abs(step.s_post - s_true)
    if err1 >= 0.4 * err0:
        raise RuntimeError(f"EKF 静置纠偏失败: s={step.s_post:.4f}  期望靠近 {s_true:.2f}")
    return {"s_post": step.s_post, "e_pri": step.e_pri, "err0": err0, "err1": err1}
