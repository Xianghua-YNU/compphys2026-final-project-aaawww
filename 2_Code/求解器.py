"""
ODE 求解器集合

实现以下数值方法用于双摆系统的时间积分:

  - forward_euler: 前向 Euler 法, O(h) 显式, 基准对照
  - rk4: 四阶 Runge-Kutta, O(h⁴), 主算法
  - symplectic_euler: 辛 Euler (Euler-Cromer), O(h) 但能量行为好
  - velocity_verlet: Störmer-Verlet, O(h²), 辛格式
  - rk45_reference: SciPy 自适应 RK45 参考解 (rtol=1e-9)

所有求解器返回统一格式的 SolverResult。

AI 使用声明: DeepSeek-V4-pro 辅助编写求解器实现代码。
"""

import time
from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

from 双摆模型 import DoublePendulum


@dataclass
class SolverResult:
    """求解器返回结果"""
    t: np.ndarray              # 时间序列 [N]
    y: np.ndarray              # 状态序列 [N, 4], 列: θ1, θ2, ω1, ω2
    n_steps: int               # 总步数
    n_eval: int                # 右端函数求值次数
    cpu_time: float            # CPU 耗时 [s]
    method_name: str           # 方法名称

    @property
    def theta1(self) -> np.ndarray: return self.y[:, 0]

    @property
    def theta2(self) -> np.ndarray: return self.y[:, 1]

    @property
    def omega1(self) -> np.ndarray: return self.y[:, 2]

    @property
    def omega2(self) -> np.ndarray: return self.y[:, 3]

    @property
    def state(self) -> np.ndarray: return self.y

    def energy_history(self, pendulum: DoublePendulum) -> np.ndarray:
        """计算每一帧的总机械能 E(t)"""
        return np.array([pendulum.total_energy(yi) for yi in self.y])


# ══════════════════════════════════════════════════════════════════════
# 求解器实现
# ══════════════════════════════════════════════════════════════════════

def forward_euler(
    pendulum: DoublePendulum,
    t_span: tuple[float, float],
    y0: np.ndarray,
    dt: float,
) -> SolverResult:
    """
    前向 Euler 法 (一阶显式)

    迭代格式:
        y_{n+1} = y_n + h f(t_n, y_n)

    精度 O(h), 作为对照基准。预期在非线性系统中快速发散。
    """
    t0, tf = t_span
    n_steps = int(np.ceil((tf - t0) / dt))
    dt = (tf - t0) / n_steps

    t = np.empty(n_steps + 1)
    y = np.empty((n_steps + 1, 4))
    t[0] = t0
    y[0] = y0

    f = pendulum.derivatives
    t_start = time.perf_counter()

    for i in range(n_steps):
        ti = t[i]
        yi = y[i]
        y[i + 1] = yi + dt * f(ti, yi)
        t[i + 1] = ti + dt

    cpu_time = time.perf_counter() - t_start
    return SolverResult(t=t, y=y, n_steps=n_steps, n_eval=n_steps,
                        cpu_time=cpu_time, method_name="Forward Euler")


def rk4(
    pendulum: DoublePendulum,
    t_span: tuple[float, float],
    y0: np.ndarray,
    dt: float,
) -> SolverResult:
    """
    经典四阶 Runge-Kutta 法

    迭代格式:
        k1 = h f(t_n, y_n)
        k2 = h f(t_n + h/2, y_n + k1/2)
        k3 = h f(t_n + h/2, y_n + k2/2)
        k4 = h f(t_n + h,   y_n + k3)
        y_{n+1} = y_n + (k1 + 2k2 + 2k3 + k4) / 6

    精度 O(h⁴), 适合混沌系统中短时间积分, 是本项目的主算法。
    """
    t0, tf = t_span
    n_steps = int(np.ceil((tf - t0) / dt))
    dt = (tf - t0) / n_steps

    t = np.empty(n_steps + 1)
    y = np.empty((n_steps + 1, 4))
    t[0] = t0
    y[0] = y0

    f = pendulum.derivatives
    t_start = time.perf_counter()

    for i in range(n_steps):
        ti = t[i]
        yi = y[i]
        half_dt = 0.5 * dt

        k1 = f(ti, yi)
        k2 = f(ti + half_dt, yi + half_dt * k1)
        k3 = f(ti + half_dt, yi + half_dt * k2)
        k4 = f(ti + dt, yi + dt * k3)

        y[i + 1] = yi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t[i + 1] = ti + dt

    cpu_time = time.perf_counter() - t_start
    return SolverResult(t=t, y=y, n_steps=n_steps, n_eval=4 * n_steps,
                        cpu_time=cpu_time, method_name="RK4")


def symplectic_euler(
    pendulum: DoublePendulum,
    t_span: tuple[float, float],
    y0: np.ndarray,
    dt: float,
) -> SolverResult:
    """
    辛 Euler 法 / Euler-Cromer (一阶辛格式)

    迭代格式:
        ω_{n+1} = ω_n + h α(θ_n, ω_n)    ← 先更新速度
        θ_{n+1} = θ_n + h ω_{n+1}        ← 用新速度更新位置

    虽精度仅 O(h), 但对 Hamilton 系统天然保持相空间体积,
    长时间积分中能量仅有界振动而不会单调漂移。

    注: 双摆在 (θ,ω) 坐标下 Hamilton 量不可分, 此处用半隐式
    近似。实测能量行为仍远优于前向 Euler。
    """
    t0, tf = t_span
    n_steps = int(np.ceil((tf - t0) / dt))
    dt = (tf - t0) / n_steps

    t = np.empty(n_steps + 1)
    y = np.empty((n_steps + 1, 4))
    t[0] = t0
    y[0] = y0

    f = pendulum.derivatives
    t_start = time.perf_counter()

    for i in range(n_steps):
        ti = t[i]
        θ1, θ2, ω1, ω2 = y[i]

        # f(t, y) 返回 [ω1, ω2, α1, α2]
        deriv = f(ti, y[i])
        α1_old, α2_old = deriv[2], deriv[3]

        # 先更新角速度
        ω1_new = ω1 + dt * α1_old
        ω2_new = ω2 + dt * α2_old

        # 用新角速度更新角度
        θ1_new = θ1 + dt * ω1_new
        θ2_new = θ2 + dt * ω2_new

        y[i + 1] = np.array([θ1_new, θ2_new, ω1_new, ω2_new])
        t[i + 1] = ti + dt

    cpu_time = time.perf_counter() - t_start
    return SolverResult(t=t, y=y, n_steps=n_steps, n_eval=n_steps,
                        cpu_time=cpu_time, method_name="Symplectic Euler")


def velocity_verlet(
    pendulum: DoublePendulum,
    t_span: tuple[float, float],
    y0: np.ndarray,
    dt: float,
) -> SolverResult:
    """
    Velocity-Verlet 法 (Störmer-Verlet, 二阶辛格式)

    对含速度依赖的力场 a(θ, ω, t) 推广形式:

        a_n = α(θ_n, ω_n, t_n)
        θ_{n+1} = θ_n + h ω_n + (h²/2) a_n           ← (1) 位置更新
        ω_predict = ω_n + h a_n                        ← (2) 速度预估
        a_{n+1} = α(θ_{n+1}, ω_predict, t_n + h)      ← (3) 用预估值算新力
        ω_{n+1} = ω_n + (h/2)(a_n + a_{n+1})          ← (4) 速度校正

    精度 O(h²), 广泛应用于分子动力学模拟。
    """
    t0, tf = t_span
    n_steps = int(np.ceil((tf - t0) / dt))
    dt = (tf - t0) / n_steps
    half_dt = 0.5 * dt
    half_dt2 = 0.5 * dt * dt

    t = np.empty(n_steps + 1)
    y = np.empty((n_steps + 1, 4))
    t[0] = t0
    y[0] = y0

    f = pendulum.derivatives
    t_start = time.perf_counter()

    for i in range(n_steps):
        ti = t[i]
        θ1, θ2, ω1, ω2 = y[i]

        # 当前加速度
        deriv = f(ti, y[i])
        α1_n, α2_n = deriv[2], deriv[3]

        # 位置更新
        θ1_new = θ1 + dt * ω1 + half_dt2 * α1_n
        θ2_new = θ2 + dt * ω2 + half_dt2 * α2_n

        # 速度预估
        ω1_pred = ω1 + dt * α1_n
        ω2_pred = ω2 + dt * α2_n

        # 用预估值评估加速度
        y_pred = np.array([θ1_new, θ2_new, ω1_pred, ω2_pred])
        deriv_new = f(ti + dt, y_pred)
        α1_new, α2_new = deriv_new[2], deriv_new[3]

        # 速度校正
        ω1_new = ω1 + half_dt * (α1_n + α1_new)
        ω2_new = ω2 + half_dt * (α2_n + α2_new)

        y[i + 1] = np.array([θ1_new, θ2_new, ω1_new, ω2_new])
        t[i + 1] = ti + dt

    cpu_time = time.perf_counter() - t_start
    return SolverResult(t=t, y=y, n_steps=n_steps, n_eval=2 * n_steps,
                        cpu_time=cpu_time, method_name="Velocity Verlet")


def rk45_reference(
    pendulum: DoublePendulum,
    t_span: tuple[float, float],
    y0: np.ndarray,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    max_step: float = 0.01,
) -> SolverResult:
    """
    SciPy 自适应 RK45 参考解

    使用 scipy.integrate.solve_ivp 的 RK45 (Dormand-Prince) 方法,
    设定严格容差得到高精度参考解, 用于衡量其他方法的全局误差。

    精度: O(h⁵) (嵌入式 4/5 阶对)
    """
    t0, tf = t_span
    t_start = time.perf_counter()

    sol = solve_ivp(
        pendulum.derivatives,
        [t0, tf],
        y0,
        method='RK45',
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )

    cpu_time = time.perf_counter() - t_start
    y_out = sol.y.T
    return SolverResult(
        t=sol.t, y=y_out,
        n_steps=len(sol.t) - 1,
        n_eval=sol.nfev,
        cpu_time=cpu_time,
        method_name="RK45 (SciPy reference)",
    )


# ══════════════════════════════════════════════════════════════════════
# 求解器注册表 (便于批量对比)
# ══════════════════════════════════════════════════════════════════════

SOLVER_REGISTRY: dict[str, Callable] = {
    "euler":             forward_euler,
    "rk4":               rk4,
    "symplectic_euler":  symplectic_euler,
    "verlet":            velocity_verlet,
    "rk45_ref":          rk45_reference,
}


def solve_all(
    pendulum: DoublePendulum,
    t_span: tuple[float, float],
    y0: np.ndarray,
    dt: float,
    methods: list[str] | None = None,
) -> dict[str, SolverResult]:
    """
    批量运行多个求解器, 返回 {名称: SolverResult} 字典。

    参数
    ----------
    methods : 求解器名称列表, 默认全部 (不含 rk45_ref)
    """
    if methods is None:
        methods = ["euler", "rk4", "symplectic_euler", "verlet"]

    results = {}
    for name in methods:
        solver = SOLVER_REGISTRY[name]
        if name == "rk45_ref":
            results[name] = solver(pendulum, t_span, y0)
        else:
            results[name] = solver(pendulum, t_span, y0, dt)
    return results
