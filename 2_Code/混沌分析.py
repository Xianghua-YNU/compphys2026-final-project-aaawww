"""
分析与混沌诊断模块

负责处理原始模拟数据、计算混沌特征量:
  - FFT 功率谱分析 (周期 vs 混沌运动的频域区分)
  - 最大 Lyapunov 指数 (轨道分离法 / Benettin 算法)
  - Poincaré 截面
  - 参数空间混沌地图

AI 使用声明: DeepSeek-V4-pro 辅助编写。
"""

import numpy as np
from 双摆模型 import DoublePendulum, PendulumParams
from 求解器 import SolverResult


# ══════════════════════════════════════════════════════════════════════
# FFT 功率谱分析
# ══════════════════════════════════════════════════════════════════════

def compute_power_spectrum(
    signal: np.ndarray,
    dt: float,
    apply_window: bool = True,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算信号的 FFT 功率谱密度 S(f) = |FFT{θ}(f)|²

    参数
    ----------
    signal : 等间距采样的时域信号 (θ1 或 θ2)
    dt : 采样时间间隔 [s]
    apply_window : 是否加 Hanning 窗减少频谱泄漏
    normalize : 是否归一化 (最大值 = 1)

    返回
    -------
    freqs : 频率数组 [Hz] (仅正频率)
    power : 功率谱密度 (仅正频率)
    """
    n = len(signal)
    signal_centered = signal - np.mean(signal)  # 去直流分量

    if apply_window:
        signal_centered = signal_centered * np.hanning(n)

    fft = np.fft.rfft(signal_centered)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n, dt)

    if normalize and np.max(power) > 0:
        power = power / np.max(power)

    return freqs, power


def compute_fft_from_result(
    result: SolverResult,
    theta_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从 SolverResult 提取信号并计算 FFT 功率谱。

    参数
    ----------
    result : 求解器结果
    theta_index : 0 → θ1, 1 → θ2

    返回
    -------
    freqs, power : 同 compute_power_spectrum
    """
    dt = result.t[1] - result.t[0]
    signal = result.y[:, theta_index]
    return compute_power_spectrum(signal, dt)


def find_peak_frequencies(
    freqs: np.ndarray,
    power: np.ndarray,
    threshold: float = 0.05,
    min_distance: int = 5,
    max_peaks: int = 10,
) -> np.ndarray:
    """
    从功率谱中提取峰值频率。

    阈值: 峰值需 ≥ threshold * max(power)。
    """
    from scipy.signal import find_peaks
    peaks, props = find_peaks(
        power, height=threshold, distance=min_distance
    )
    order = np.argsort(power[peaks])[::-1]
    return freqs[peaks[order][:max_peaks]]


def compute_fft_all_methods(
    results: dict[str, SolverResult],
    theta_index: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """对多个求解器结果批量计算 FFT 功率谱。"""
    spectra = {}
    for name, result in results.items():
        spectra[name] = compute_fft_from_result(result, theta_index)
    return spectra


# ══════════════════════════════════════════════════════════════════════
# Lyapunov 指数
# ══════════════════════════════════════════════════════════════════════

def compute_lyapunov_benettin(
    pendulum: DoublePendulum,
    y0: np.ndarray,
    T: float,
    dt: float,
    d0: float = 1e-8,
    renormalize_interval: int = 50,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    用 Benettin 轨道分离法计算最大 Lyapunov 指数。

    原理:
        1. 从 y0 和 y0 + d0 出发分别积分
        2. 每隔 renormalize_interval 步测量分离距离 d
        3. 将扰动向量重新归一化到 |d0|
        4. λ_max = (1/t) Σ ln(d_i / d0)

    参数
    ----------
    pendulum : 双摆系统
    y0 : 初始状态 [4]
    T : 总积分时间 [s]
    dt : 时间步长 [s]
    d0 : 初始扰动大小
    renormalize_interval : 重归一化间隔 (步数)

    返回
    -------
    lyap : 最大 Lyapunov 指数估计值 [1/s]
    t_vals : Lyapunov 指数收敛的时间序列
    lyap_vals : Lyapunov 指数收敛值序列
    """
    from 求解器 import rk4

    n_steps = int(T / dt)
    y_ref = y0.copy()
    y_pert = y0.copy()
    y_pert[0] += d0

    d_current = d0
    sum_log = 0.0
    n_renorm = 0

    lyap_vals = []
    t_vals = []

    for i in range(n_steps):
        t_i = i * dt
        y_ref = y_ref + dt * pendulum.derivatives(t_i, y_ref)
        y_pert = y_pert + dt * pendulum.derivatives(t_i, y_pert)

        if (i + 1) % renormalize_interval == 0:
            delta = y_pert - y_ref
            d = np.linalg.norm(delta)
            if d > 1e-15:
                sum_log += np.log(d / d_current)
                n_renorm += 1
                y_pert = y_ref + (d0 / d) * delta
                d_current = d0

            t_vals.append((i + 1) * dt)
            lyap_val = sum_log / ((i + 1) * dt) if n_renorm > 0 else 0.0
            lyap_vals.append(lyap_val)

    lyap = sum_log / (n_steps * dt) if n_renorm > 0 else 0.0
    return lyap, np.array(t_vals), np.array(lyap_vals)


# ══════════════════════════════════════════════════════════════════════
# Poincaré 截面
# ══════════════════════════════════════════════════════════════════════

def poincare_section(
    result: SolverResult,
    section_var: str = "theta1",
    section_value: float = 0.0,
    direction: str = "positive",
) -> np.ndarray:
    """
    从轨线中提取 Poincaré 截面点。

    当 section_var 穿过 section_value 时, 记录其余相空间坐标。

    参数
    ----------
    result : 求解器结果
    section_var : 截面变量 ("theta1" 或 "theta2")
    section_value : 截面值 [rad]
    direction : "positive" (正向穿过) 或 "both"

    返回
    -------
    points : [N, 2] 截面上的 (other_theta, other_omega) 坐标
    """
    θ1, θ2 = result.theta1, result.theta2
    ω1, ω2 = result.omega1, result.omega2

    if section_var == "theta1":
        x, y_plot = θ1, θ2
        o_plot = ω2
    elif section_var == "theta2":
        x, y_plot = θ2, θ1
        o_plot = ω1
    else:
        raise ValueError(f"未知截面变量: {section_var}")

    points = []
    for i in range(1, len(x)):
        prev_val = x[i - 1] - section_value
        curr_val = x[i] - section_value
        crossed = prev_val * curr_val < 0
        if not crossed:
            continue
        if direction == "positive" and x[i] < x[i - 1]:
            continue
        frac = -prev_val / (curr_val - prev_val)
        y_cross = y_plot[i - 1] + frac * (y_plot[i] - y_plot[i - 1])
        o_cross = o_plot[i - 1] + frac * (o_plot[i] - o_plot[i - 1])
        points.append([y_cross, o_cross])

    return np.array(points)


# ══════════════════════════════════════════════════════════════════════
# 参数空间扫描
# ══════════════════════════════════════════════════════════════════════

def _initial_from_energy(pendulum: DoublePendulum, target_E: float) -> np.ndarray:
    """二分查找初始角度 [θ1, θ2, 0, 0] 使 total_energy ≈ target_E。"""
    lo, hi = 0.0, np.pi
    for _ in range(60):
        mid = (lo + hi) / 2.0
        state = np.array([mid, mid * 0.5, 0.0, 0.0])
        if pendulum.total_energy(state) < target_E:
            lo = mid
        else:
            hi = mid
    theta = (lo + hi) / 2.0
    return np.array([theta, theta * 0.5, 0.0, 0.0])


def scan_parameter_space(
    mass_ratios: np.ndarray,
    energies: np.ndarray,
    T: float = 40.0,
    dt: float = 0.01,
) -> np.ndarray:
    """
    扫描 (能量, 质量比) 参数空间, 计算每个格点的最大 Lyapunov 指数。

    参数
    ----------
    mass_ratios : [N_R] 质量比 m2/m1 格点
    energies : [N_E] 初始能量格点
    T : 每次积分时长 [s]
    dt : 时间步长 [s]

    返回
    -------
    lyap_map : [N_E, N_R] Lyapunov 指数矩阵
    """
    from 求解器 import rk4

    n_E, n_R = len(energies), len(mass_ratios)
    lyap_map = np.full((n_E, n_R), np.nan)

    for j, R in enumerate(mass_ratios):
        pend = DoublePendulum(m1=1.0, m2=R)
        for i, E in enumerate(energies):
            E_max = (1.0 + R) * pend.p.g * pend.p.l1 + R * pend.p.g * pend.p.l2
            if E >= E_max - 0.1:
                continue

            try:
                y0 = _initial_from_energy(pend, E)
            except Exception:
                continue

            lyap, _, _ = compute_lyapunov_benettin(pend, y0, T, dt)
            lyap_map[i, j] = lyap

    return lyap_map


# ══════════════════════════════════════════════════════════════════════
# 分岔图数据采集 (受迫阻尼双摆)
# ══════════════════════════════════════════════════════════════════════

def bifurcation_data(
    A_vals: np.ndarray,
    b: float = 0.5,
    Omega: float = 2.0,
    T_transient: float = 120.0,
    T_sample: float = 150.0,
    dt: float = 0.02,
) -> list[np.ndarray]:
    """
    采集受迫阻尼双摆的分岔图数据 (stroboscopic sampling)。

    对每个驱动振幅 A:
      1. 积分 T_transient 秒, 丢弃瞬态
      2. 继续积分 T_sample 秒, 每隔驱动力周期 T_drive = 2π/Ω 记录 θ1
      3. 返回该 A 下所有采样点

    参数
    ----------
    A_vals : 驱动振幅数组
    b : 阻尼系数 [1/s]
    Omega : 驱动力角频率 [rad/s]
    T_transient : 瞬态丢弃时长 [s]
    T_sample : 采样时长 [s]
    dt : 时间步长 [s]

    返回
    -------
    samples : 列表, samples[i] 为 A_vals[i] 对应的 θ1 采样数组
    """
    from 求解器 import rk4

    T_drive = 2.0 * np.pi / Omega
    all_samples = []

    for A in A_vals:
        params = PendulumParams(b=b, A=A, Omega=Omega)
        pend = DoublePendulum(params)
        y0 = np.array([1.0, 0.5, 0.0, 0.0])

        # 瞬态
        n_trans = int(T_transient / dt)
        y = y0.copy()
        for i in range(n_trans):
            y = y + dt * pend.derivatives(i * dt, y)

        # 采样
        n_sample = int(T_sample / dt)
        samples_A = []
        for i in range(n_sample):
            t_i = i * dt
            y = y + dt * pend.derivatives(T_transient + t_i, y)
            # 判断是否接近驱动周期整数倍
            phase = (T_transient + t_i) % T_drive
            if phase < dt or T_drive - phase < dt:
                samples_A.append(y[0])

        all_samples.append(np.array(samples_A))

    return all_samples
