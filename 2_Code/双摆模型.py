"""
双摆物理模型

双自由度耦合非线性系统。由 Lagrange 力学导出运动方程，支持无耗散和
阻尼驱动两种工况。

AI 使用声明: DeepSeek-V4-pro 辅助推导 ODE 求解公式与能量表达式。
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PendulumParams:
    """双摆物理参数集中定义"""
    m1: float = 1.0       # 摆1质量 [kg]
    m2: float = 1.0       # 摆2质量 [kg]
    l1: float = 1.0       # 摆1长度 [m]
    l2: float = 1.0       # 摆2长度 [m]
    g:  float = 9.81      # 重力加速度 [m/s²]
    b:  float = 0.0       # 阻尼系数 [1/s], 0 表示无耗散
    A:  float = 0.0       # 驱动力振幅 [rad/s²], 0 表示无驱动
    Omega: float = 0.0    # 驱动力角频率 [rad/s]


class DoublePendulum:
    """
    混沌双摆系统 (含可选阻尼驱动拓展)

    状态向量 y = [θ1, θ2, ω1, ω2]ᵀ，一阶 ODE 系统:
        dθ1/dt = ω1
        dθ2/dt = ω2
        dω1/dt = α1(θ1,θ2,ω1,ω2)
        dω2/dt = α2(θ1,θ2,ω1,ω2)

    其中 α1, α2 由 Euler-Lagrange 方程导出。
    """

    def __init__(self, params: PendulumParams | None = None, **kwargs):
        if params is None:
            params = PendulumParams(**kwargs)
        self.p = params

    # ------------------------------------------------------------------
    # 运动方程
    # ------------------------------------------------------------------
    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        计算状态导数 dy/dt = f(t, y)

        参数
        ----------
        t : 时间 (用于含时驱动力项)
        state : [θ1, θ2, ω1, ω2]

        返回
        -------
        [dθ1/dt, dθ2/dt, dω1/dt, dω2/dt]
        """
        θ1, θ2, ω1, ω2 = state
        Δ = θ1 - θ2
        sinΔ, cosΔ = np.sin(Δ), np.cos(Δ)

        m1, m2 = self.p.m1, self.p.m2
        l1, l2 = self.p.l1, self.p.l2
        g = self.p.g

        # 质量矩阵行列式 × l1·l2 的公因子
        denom = l1 * (m1 + m2 - m2 * cosΔ * cosΔ)

        # 耦合 ODE 右端项 (不含 θ̈ 的部分)
        # (m1+m2)l1 θ̈1 + m2 l2 θ̈2 cosΔ + m2 l2 ω2² sinΔ + (m1+m2)g sinθ1 = 0
        # l1 θ̈1 cosΔ + l2 θ̈2 - l1 ω1² sinΔ + g sinθ2 = 0
        b1 = -m2 * l2 * ω2 * ω2 * sinΔ - (m1 + m2) * g * np.sin(θ1)
        b2 = l1 * ω1 * ω1 * sinΔ - g * np.sin(θ2)

        # 解 2×2 线性方程组求角加速度
        α1 = (b1 - m2 * cosΔ * b2) / denom
        α2 = ((m1 + m2) * b2 - cosΔ * b1) / denom

        # 阻尼项: -b ω_i (仅在 b > 0 时生效)
        if self.p.b > 0:
            α1 -= self.p.b * ω1
            α2 -= self.p.b * ω2

        # 周期驱动力: A sin(Ω t) (仅在 A > 0 时生效)
        if self.p.A > 0:
            drive = self.p.A * np.sin(self.p.Omega * t)
            α1 += drive
            α2 += drive

        return np.array([ω1, ω2, α1, α2])

    # ------------------------------------------------------------------
    # 能量与守恒量
    # ------------------------------------------------------------------
    def kinetic_energy(self, state: np.ndarray) -> float:
        """动能 T = ½(m1+m2)l1²ω1² + ½m2l2²ω2² + m2 l1 l2 ω1 ω2 cosΔ"""
        θ1, θ2, ω1, ω2 = state
        m1, m2, l1, l2 = self.p.m1, self.p.m2, self.p.l1, self.p.l2
        return (0.5 * (m1 + m2) * l1 * l1 * ω1 * ω1
                + 0.5 * m2 * l2 * l2 * ω2 * ω2
                + m2 * l1 * l2 * ω1 * ω2 * np.cos(θ1 - θ2))

    def potential_energy(self, state: np.ndarray) -> float:
        """势能 V = -(m1+m2)g l1 cosθ1 - m2 g l2 cosθ2 (零势能面在 y=0)"""
        θ1, θ2, _, _ = state
        m1, m2, l1, l2, g = self.p.m1, self.p.m2, self.p.l1, self.p.l2, self.p.g
        return -(m1 + m2) * g * l1 * np.cos(θ1) - m2 * g * l2 * np.cos(θ2)

    def total_energy(self, state: np.ndarray) -> float:
        """总机械能 E = T + V (无耗散时应守恒)"""
        return self.kinetic_energy(state) + self.potential_energy(state)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def mass_ratio(self) -> float:
        """质量比 m2 / m1"""
        return self.p.m2 / self.p.m1 if self.p.m1 > 0 else np.inf

    def is_dissipative(self) -> bool:
        """是否含阻尼或驱动"""
        return self.p.b > 0 or self.p.A > 0

    def reduced_state(self, state: np.ndarray) -> np.ndarray:
        """退化到单摆: 将 (θ1, ω1) 映射为等效单摆的 (θ, ω)。

        m2→0 时双摆退化为单摆，用于与解析解对比验证。"""
        θ1, _, ω1, _ = state
        return np.array([θ1, ω1])
