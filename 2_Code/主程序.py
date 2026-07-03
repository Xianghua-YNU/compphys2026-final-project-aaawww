"""
入口模块: 混沌双摆数值模拟

负责设置物理参数、初始化系统、调用求解器并启动分析流程。
所有重要物理参数在此集中定义, 严禁在子模块中出现魔法数字。

用法:
    python main.py                 # 运行全部对比 & 验证
    python main.py --quick         # 快速验证模式
    python main.py --full          # 完整模拟 (长时间积分)

AI 使用声明: DeepSeek-V4-pro 辅助编写主流程代码。
"""

import argparse
import io
import sys
from pathlib import Path

# Windows 终端 UTF-8 编码修复
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头模式, 不弹窗
import matplotlib.pyplot as plt

import 图表配置 as fc  # 论文图表样式 (600 DPI / Times New Roman / cm 数学字体)

import 混沌分析
from 双摆模型 import DoublePendulum, PendulumParams
from 求解器 import (
    forward_euler, rk4, symplectic_euler, velocity_verlet,
    rk45_reference, solve_all, SolverResult,
)
import 可视化 as viz

# ══════════════════════════════════════════════════════════════════════
# 物理参数 (集中定义)
# ══════════════════════════════════════════════════════════════════════

# 默认双摆参数
DEFAULT_PARAMS = PendulumParams(
    m1=1.0,     # 摆1质量 [kg]
    m2=1.0,     # 摆2质量 [kg]
    l1=1.0,     # 摆1长度 [m]
    l2=1.0,     # 摆2长度 [m]
    g=9.81,     # 重力加速度 [m/s²]
    b=0.0,      # 阻尼系数 [1/s], 0 = 无耗散
    A=0.0,      # 驱动力振幅 [rad/s²], 0 = 无驱动
    Omega=0.0,  # 驱动力频率 [rad/s]
)

# 初始条件: 两摆从不同偏转角静止释放 (E₀ ≠ 0, 避免能量归一化发散)
INITIAL_STATE = np.array([np.pi / 2 + 0.3, np.pi / 2 - 0.3, 0.0, 0.0])

# 微扰初始条件 (用于初值敏感性演示, Δθ = 1e-6 rad)
INITIAL_STATE_PERTURBED = np.array(
    [np.pi / 2 + 0.3 + 1e-6, np.pi / 2 - 0.3, 0.0, 0.0]
)


# ══════════════════════════════════════════════════════════════════════
# 验证模块
# ══════════════════════════════════════════════════════════════════════

def verify_single_pendulum():
    """
    退化验证: m2→0 时双摆退化为单摆, 对比解析解。

    单摆小角度近似周期: T₀ = 2π √(l/g)
    本测试用大角度 (θ₀=π/2) 精确周期的级数展开作为参考。
    """
    print("=" * 60)
    print("验证 1: 退化到单摆 (m2 → 0)")
    print("=" * 60)

    params = PendulumParams(m1=1.0, m2=1e-12, l1=1.0, l2=1.0, g=9.81)
    pendulum = DoublePendulum(params)
    y0 = np.array([np.pi / 4, 0.1, 0.0, 0.0])  # θ1=45°, θ2 微小
    dt = 0.001
    t_span = (0.0, 20.0)

    result = rk4(pendulum, t_span, y0, dt)

    # 大角度单摆精确周期 (级数展开到 θ₀⁴)
    theta0 = np.pi / 4
    T_exact = 2 * np.pi * np.sqrt(params.l1 / params.g)
    T_corrected = T_exact * (1 + theta0**2 / 16 + 11 * theta0**4 / 3072)

    # 从数据中找 θ1 过零点来估计周期
    theta1 = result.theta1
    zero_crossings = []
    for i in range(1, len(theta1)):
        if theta1[i - 1] * theta1[i] < 0 and theta1[i - 1] > 0:
            # 线性插值
            frac = -theta1[i - 1] / (theta1[i] - theta1[i - 1])
            t_cross = result.t[i - 1] + frac * (result.t[i] - result.t[i - 1])
            zero_crossings.append(t_cross)

    if len(zero_crossings) >= 3:
        T_numerical = (zero_crossings[-1] - zero_crossings[0]) / (len(zero_crossings) - 1)
        error = abs(T_numerical - T_corrected) / T_corrected * 100
        print(f"  理论周期 (θ₀=45°, 含修正): {T_corrected:.4f} s")
        print(f"  数值周期 (RK4):              {T_numerical:.4f} s")
        print(f"  相对误差:                     {error:.4f}%")
        if error < 0.5:
            print("  ✓ 退化到单摆验证通过")
            return True
        else:
            print(f"  ⚠ 误差偏大 ({error:.2f}%), 请检查")
            return False
    else:
        print("  ✗ 未找到足够的过零点, 增加积分时长")
        return False


def verify_convergence():
    """
    收敛性验证: 绘制 log(error) vs log(h) 图, 验证各方法收敛阶。

    使用退化单摆 (m2→0, 无混沌) 消除混沌指数发散对收敛阶的干扰。
    以 RK45 (rtol=1e-9) 为参考解, 计算各方法在不同步长下的
    全局误差 ‖y(T) - y_ref(T)‖₂。
    预期: Euler O(h), RK4 O(h⁴), Symplectic Euler O(h), Verlet O(h²)
    """
    print("\n" + "=" * 60)
    print("验证 2: 收敛阶测试 (退化单摆, T=1s)")
    print("=" * 60)

    # 退化到单摆: m2 → 0, 无混沌干扰
    params = PendulumParams(m1=1.0, m2=1e-12, l1=1.0, l2=1.0, g=9.81)
    pendulum = DoublePendulum(params)
    y0 = np.array([np.pi / 4, 0.1, 0.0, 0.0])  # θ1=45°, θ2 微小
    T = 1.0
    h_values = [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025]

    # 参考解
    ref = rk45_reference(pendulum, (0.0, T), y0)
    y_ref = ref.y[-1]
    print(f"  参考解 (RK45): {ref.n_steps} 步, {ref.cpu_time:.3f}s")

    solvers_to_test = {
        "Forward Euler": forward_euler,
        "RK4": rk4,
        "Symplectic Euler": symplectic_euler,
        "Velocity Verlet": velocity_verlet,
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, solver in solvers_to_test.items():
        errors = []
        for h in h_values:
            result = solver(pendulum, (0.0, T), y0, h)
            error = np.linalg.norm(result.y[-1] - y_ref)
            errors.append(max(error, 1e-16))

        errors = np.array(errors)
        h_arr = np.array(h_values)
        ax.loglog(h_arr, errors, "o-", label=name, markersize=5)

        # 线性拟合求收敛阶 (用后 4 个最小步长)
        coeffs = np.polyfit(np.log(h_arr[-4:]), np.log(errors[-4:]), 1)
        order = coeffs[0]
        print(f"  {name:20s}: 实测阶数 ≈ {order:.2f}")

    # 参考线
    h_ref = np.array([h_values[-1], h_values[0]])
    ax.loglog(h_ref, 0.5 * h_ref, "k--", alpha=0.3, label=r"$O(h)$")
    ax.loglog(h_ref, 0.1 * h_ref**2, "k-.", alpha=0.3, label=r"$O(h^2)$")
    ax.loglog(h_ref, 0.02 * h_ref**4, "k:", alpha=0.3, label=r"$O(h^4)$")

    ax.set_xlabel("Step size h [s]")
    ax.set_ylabel(r"Global error $\|y(T) - y_{\rm ref}(T)\|_2$")
    ax.set_title("Convergence Order Verification (single pendulum, T = 1 s)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    fc.save_figure_both(fig, "convergence_test.png")
    return True


def verify_energy():
    """
    能量守恒性验证: 积分 T=100s, 绘制 ΔE/E₀ 和 |ΔE| 随时间变化。

    无耗散双摆总机械能应守恒。symplectic 方法预期能量有界振动,
    而非 symplectic 方法 (Euler, RK4) 预期存在长期漂移。
    """
    print("\n" + "=" * 60)
    print("验证 3: 能量守恒性测试")
    print("=" * 60)

    pendulum = DoublePendulum(DEFAULT_PARAMS)
    T = 100.0
    dt = 0.005
    E0 = pendulum.total_energy(INITIAL_STATE)
    e0_abs = abs(E0) if abs(E0) > 1e-12 else 1.0
    print(f"  初始能量 E₀ = {E0:.4f} J")

    methods = {
        "Forward Euler": forward_euler,
        "RK4": rk4,
        "Symplectic Euler": symplectic_euler,
        "Velocity Verlet": velocity_verlet,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, solver in methods.items():
        result = solver(pendulum, (0.0, T), INITIAL_STATE, dt)
        E = result.energy_history(pendulum)

        # 左图: 相对偏差
        delta_rel = (E - E0) / e0_abs
        ax1.plot(result.t, delta_rel, label=name, alpha=0.8, lw=0.8)
        print(f"  {name:20s}: max|ΔE/E₀| = {np.max(np.abs(delta_rel)):.2e}")

        # 右图: 绝对偏差 (对数)
        delta_abs = np.abs(E - E0) + 1e-16
        ax2.semilogy(result.t, delta_abs, label=name, alpha=0.8, lw=0.8)

    ax1.set_xlabel("Time t [s]")
    ax1.set_ylabel(r"Rel. energy error $\Delta E / |E_0|$")
    ax1.set_title(f"Relative Energy Error (T = {T} s, h = {dt})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="k", lw=0.5)

    ax2.set_xlabel("Time t [s]")
    ax2.set_ylabel(r"Abs. energy error $|\Delta E|$ [J]")
    ax2.set_title(f"Absolute Energy Error — log scale (T = {T} s, h = {dt})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    fc.save_figure_both(fig, "energy_conservation.png")


# ══════════════════════════════════════════════════════════════════════
# 模拟与可视化
# ══════════════════════════════════════════════════════════════════════

def run_trajectory_comparison():
    """
    短时间轨迹对比: 所有方法在相同初始条件下积分, 对比 θ1(t) 轨线。
    """
    print("\n" + "=" * 60)
    print("轨迹对比: 所有方法 θ1(t) 轨线")
    print("=" * 60)

    pendulum = DoublePendulum(DEFAULT_PARAMS)
    T, dt = 20.0, 0.01

    methods = {
        "Forward Euler": forward_euler,
        "RK4": rk4,
        "Symplectic Euler": symplectic_euler,
        "Velocity Verlet": velocity_verlet,
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

    for (name, solver), ax in zip(methods.items(), axes.flat):
        result = solver(pendulum, (0.0, T), INITIAL_STATE, dt)
        ax.plot(result.t, result.theta1, lw=0.5, alpha=0.9)
        ax.set_title(name, fontsize=11)
        ax.set_ylabel(r"$\theta_1$ [rad]")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("t [s]")
    axes[-1, 1].set_xlabel("t [s]")
    fig.suptitle(f"Double Pendulum $\\theta_1(t)$ — Method Comparison (h={dt})", fontsize=13)
    fig.tight_layout()

    fc.save_figure_both(fig, "trajectory_comparison.png")


def run_sensitivity_demo():
    """
    初值敏感性演示: 对同一系统分别以初始条件和微扰条件积分,
    绘制两轨线 θ1(t) 之差 Δθ(t), 展示指数级分离 (蝴蝶效应)。
    """
    print("\n" + "=" * 60)
    print("初值敏感性演示 (蝴蝶效应)")
    print("=" * 60)

    pendulum = DoublePendulum(DEFAULT_PARAMS)
    T, dt = 30.0, 0.002

    result_ref = rk4(pendulum, (0.0, T), INITIAL_STATE, dt)
    result_pert = rk4(pendulum, (0.0, T), INITIAL_STATE_PERTURBED, dt)

    delta_theta = np.sqrt(
        (result_ref.theta1 - result_pert.theta1) ** 2
        + (result_ref.theta2 - result_pert.theta2) ** 2
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 上: 两条轨线
    ax1.plot(result_ref.t, result_ref.theta1, lw=0.5, alpha=0.8,
             label=r"$\theta_1(0)$ = 1.8708 rad")
    ax1.plot(result_pert.t, result_pert.theta1, lw=0.5, alpha=0.8,
             label=r"$\theta_1(0)$ = 1.8708 + $10^{-6}$ rad")
    ax1.set_ylabel(r"$\theta_1$ [rad]")
    ax1.set_title("Sensitivity to Initial Conditions (Double Pendulum)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 下: Δθ(t) 半对数图 (指数分离在混沌系统中应近似为直线)
    ax2.semilogy(result_ref.t, delta_theta + 1e-16, lw=0.8)
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel(r"$\|\Delta\theta(t)\|_2$")
    ax2.set_title("Trajectory Separation (semi-log scale)")
    ax2.grid(True, alpha=0.3)

    # 拟合 Lyapunov 指数 (用后半段数据)
    half = len(result_ref.t) // 2
    valid = delta_theta[half:] > 1e-12
    if np.sum(valid) > 10:
        coeffs = np.polyfit(result_ref.t[half:][valid],
                            np.log(delta_theta[half:][valid]), 1)
        lyap_est = coeffs[0]
        print(f"  估计最大 Lyapunov 指数: λ ≈ {lyap_est:.3f} [1/s]")
        ax2.plot(result_ref.t[half:][valid],
                 np.exp(coeffs[1]) * np.exp(coeffs[0] * result_ref.t[half:][valid]),
                 "r--", lw=1, alpha=0.6,
                 label=f"Exp. fit: λ ≈ {lyap_est:.3f}")

    ax2.legend(fontsize=8)
    fig.tight_layout()

    fc.save_figure_both(fig, "sensitivity_demo.png")
    print(f"  初始微扰: Δθ₁ = 1e-6 rad")


# ══════════════════════════════════════════════════════════════════════
# 第 2 阶段: 相图、FFT、长时间稳定性、精度基准
# ══════════════════════════════════════════════════════════════════════

def run_phase_portraits():
    """
    四方法相空间轨迹对比: 2×2 面板, 时间颜色编码。

    所有方法使用相同的初始条件和步长, 展示不同数值方法
    在相空间中的轨迹差异。
    """
    print("\n" + "=" * 60)
    print("第 2 阶段 · 相空间轨迹对比 (4 方法 × 2×2)")
    print("=" * 60)

    pendulum = DoublePendulum(DEFAULT_PARAMS)
    T, dt = 25.0, 0.002

    results = solve_all(pendulum, (0.0, T), INITIAL_STATE, dt,
                        methods=["euler", "rk4", "symplectic_euler", "verlet"])

    viz.plot_phase_portrait_4panel(results, "phase_portrait_4panel.png")
    print("  ✓ 相空间对比图已生成")


def run_fft_analysis():
    """
    FFT 功率谱分析: (1) 周期 vs 混沌对比 (2) 四方法叠加对比。

    周期运动 → 离散尖峰; 混沌运动 → 宽频连续谱。
    这是区分运动类型最直观的频域判据。
    """
    print("\n" + "=" * 60)
    print("第 2 阶段 · FFT 功率谱分析")
    print("=" * 60)

    pendulum = DoublePendulum(DEFAULT_PARAMS)
    T, dt = 200.0, 0.002

    # 周期运动: 小角度初始条件
    y0_periodic = np.array([0.4, 0.3, 0.0, 0.0])
    result_periodic = rk4(pendulum, (0.0, T), y0_periodic, dt)
    print(f"  周期运动: θ1(0)={y0_periodic[0]}, θ2(0)={y0_periodic[1]}")
    print(f"    E₀ = {pendulum.total_energy(y0_periodic):.3f} J")

    # 混沌运动: 大角度初始条件
    y0_chaotic = INITIAL_STATE
    result_chaotic = rk4(pendulum, (0.0, T), y0_chaotic, dt)
    print(f"  混沌运动: θ1(0)={y0_chaotic[0]:.2f}, θ2(0)={y0_chaotic[1]:.2f}")
    print(f"    E₀ = {pendulum.total_energy(y0_chaotic):.3f} J")

    # 周期 vs 混沌对比图
    viz.plot_fft_periodic_vs_chaotic(
        result_periodic, result_chaotic, "fft_periodic_vs_chaotic.png"
    )

    # 四方法叠加对比图
    results_all = solve_all(pendulum, (0.0, T), y0_chaotic, dt,
                            methods=["euler", "rk4", "symplectic_euler", "verlet"])
    viz.plot_fft_method_comparison(results_all, "fft_method_comparison.png")

    # 打印峰值频率
    print("\n  混沌运动 θ1 峰值频率:")
    for name, result in results_all.items():
        freqs, power = analysis.compute_fft_from_result(result, 0)
        peaks = analysis.find_peak_frequencies(freqs, power, threshold=0.03)
        peaks_str = ", ".join(f"{p:.3f}" for p in peaks[:5])
        print(f"    {name:20s}: {peaks_str} Hz")

    print("  ✓ FFT 分析完成")


def run_energy_stability_extended():
    """
    长时间能量守恒性测试: T=2000s, 四种方法对比。

    预期: Symplectic 方法能量有界振动; 非 symplectic 方法单调漂移。
    这是 symplectic 格式在 Hamilton 系统中优势的核心证据。
    """
    print("\n" + "=" * 60)
    print("第 2 阶段 · 长时间能量稳定性 (T = 2000 s)")
    print("=" * 60)

    pendulum = DoublePendulum(DEFAULT_PARAMS)
    T, dt = 2000.0, 0.005
    E0 = pendulum.total_energy(INITIAL_STATE)

    results = solve_all(pendulum, (0.0, T), INITIAL_STATE, dt,
                        methods=["euler", "rk4", "symplectic_euler", "verlet"])

    # 统计各方法能量漂移
    for name, result in results.items():
        E = np.array([pendulum.total_energy(yi) for yi in result.y])
        delta_rel = (E - E0) / abs(E0) if abs(E0) > 1e-12 else E - E0
        drift_rate = delta_rel[-1] / T  # 平均漂移速率
        rms_dev = np.sqrt(np.mean(delta_rel ** 2))
        print(f"  {name:20s}: |drift rate| = {abs(drift_rate):.2e} /s, "
              f"RMS = {rms_dev:.2e}")

    viz.plot_energy_stability_extended(results, pendulum,
                                       "energy_stability_extended.png")
    print("  ✓ 长时间能量稳定性图已生成")


def run_precision_benchmark():
    """
    精度-效率基准测试: 各方法在多种步长下测量全局误差和 CPU 时间。

    产出双面板图: (左) 误差 vs 步长, (右) 误差 vs CPU 时间。
    揭示"为达到同等精度, 不同方法需要多少计算资源"。
    """
    print("\n" + "=" * 60)
    print("第 2 阶段 · 精度-效率基准测试")
    print("=" * 60)

    pendulum = DoublePendulum(DEFAULT_PARAMS)
    T = 10.0
    h_values = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

    # 参考解
    ref = rk45_reference(pendulum, (0.0, T), INITIAL_STATE)
    y_ref = ref.y[-1]
    print(f"  参考解 (RK45): {ref.n_steps} 步, {ref.cpu_time:.4f}s")

    solvers_to_test = {
        "Forward Euler": forward_euler,
        "RK4": rk4,
        "Symplectic Euler": symplectic_euler,
        "Velocity Verlet": velocity_verlet,
    }

    benchmark = {}
    for name, solver in solvers_to_test.items():
        dt_list, err_list, cpu_list = [], [], []
        for h in h_values:
            result = solver(pendulum, (0.0, T), INITIAL_STATE, h)
            error = np.linalg.norm(result.y[-1] - y_ref)
            dt_list.append(h)
            err_list.append(max(error, 1e-16))
            cpu_list.append(result.cpu_time)
        benchmark[name] = {"dt": dt_list, "error": err_list, "cpu": cpu_list}
        print(f"  {name:20s}: dt={h_values[-1]:.4f}s → error={err_list[-1]:.2e}, "
              f"CPU={cpu_list[-1]:.5f}s")

    viz.plot_precision_benchmark(benchmark, "precision_benchmark.png")
    print("  ✓ 精度-效率基准图已生成")


def run_animation():
    """
    生成双摆运动动画 — 四种数值方法对比。

    参数:
      dt=0.005, T=40s, skip=5, fps=12, duration=60s
      → 展示最后 ~18s 模拟数据, 完整轨迹拖尾
    """
    print("\n" + "=" * 60)
    print("第 2 阶段 · 双摆运动动画 (四方法)")
    print("=" * 60)

    pendulum = DoublePendulum(DEFAULT_PARAMS)
    T, dt = 40.0, 0.005

    methods = [
        ("Forward Euler", forward_euler),
        ("RK4", rk4),
        ("Symplectic Euler", symplectic_euler),
        ("Velocity Verlet", velocity_verlet),
    ]

    for method_name, solver in methods:
        print(f"\n  [{method_name}]")
        result = solver(pendulum, (0.0, T), INITIAL_STATE, dt)
        print(f"    积分: {result.n_steps} 步, {result.cpu_time:.2f}s")

        fname = f"anim_{method_name.lower().replace(' ', '_')}.mp4"
        viz.animate_double_pendulum(result, pendulum=pendulum,
                                    l1=DEFAULT_PARAMS.l1, l2=DEFAULT_PARAMS.l2,
                                    save_name=fname,
                                    fps=12, skip=5, trail_length=9999, duration=60.0)

    print("\n  ✓ 全部 4 个动画生成完成")


# ══════════════════════════════════════════════════════════════════════
# Phase 3: 混沌诊断
# ══════════════════════════════════════════════════════════════════════

def run_poincare():
    """
    Poincaré 截面: 四种方法在 θ1=0 截面上的 (θ2, ω2) 分布。
    """
    print("\n" + "=" * 60)
    print("第 3 阶段 · Poincaré 截面")
    print("=" * 60)

    pendulum = DoublePendulum(DEFAULT_PARAMS)
    T, dt = 200.0, 0.005

    methods = {
        "Forward Euler": forward_euler,
        "RK4": rk4,
        "Symplectic Euler": symplectic_euler,
        "Velocity Verlet": velocity_verlet,
    }

    results = {}
    for name, solver in methods.items():
        print(f"  [{name}] 积分 T={T}s ...")
        results[name] = solver(pendulum, (0.0, T), INITIAL_STATE, dt)

    viz.plot_poincare_sections(results, section_var="theta1", section_value=0.0)
    print("  ✓ Poincaré 截面图完成")


def run_chaos_map():
    """
    参数空间混沌地图: 扫描 (质量比, 能量) 参数空间, 计算 Lyapunov 指数。
    """
    print("\n" + "=" * 60)
    print("第 3 阶段 · 参数空间混沌地图")
    print("=" * 60)

    mass_ratios = np.linspace(0.2, 3.0, 18)
    energies = np.linspace(-20.0, 18.0, 16)

    print(f"  质量比: {mass_ratios[0]:.1f} ~ {mass_ratios[-1]:.1f} ({len(mass_ratios)} 格点)")
    print(f"  能量:   {energies[0]:.1f} ~ {energies[-1]:.1f} ({len(energies)} 格点)")
    print(f"  共计 {len(mass_ratios) * len(energies)} 个参数组, 预计耗时较长 ...")

    lyap_map = analysis.scan_parameter_space(mass_ratios, energies, T=40.0, dt=0.01)

    viz.plot_chaos_map(lyap_map, mass_ratios, energies)
    print("  ✓ 混沌地图完成")


def run_bifurcation():
    """
    受迫阻尼双摆分岔图: 驱动振幅 A 从 0.3 到 1.8, stroboscopic 采样。
    """
    print("\n" + "=" * 60)
    print("第 3 阶段 · 分岔图")
    print("=" * 60)

    A_vals = np.linspace(0.3, 1.8, 80)
    b, Omega = 0.5, 2.0

    print(f"  驱动振幅 A: {A_vals[0]:.1f} ~ {A_vals[-1]:.1f} ({len(A_vals)} 个值)")
    print(f"  阻尼 b={b}, 驱动频率 Ω={Omega}")
    print(f"  每个 A 值: 瞬态 120s + 采样 150s, 预计耗时较长 ...")

    samples = analysis.bifurcation_data(A_vals, b=b, Omega=Omega)

    viz.plot_bifurcation(A_vals, samples)
    print("  ✓ 分岔图完成")


# ══════════════════════════════════════════════════════════════════════
# Phase 4: 关联维数、运行时基准、高分辨率动画、加密分岔
# ══════════════════════════════════════════════════════════════════════

def run_correlation_dimension():
    """
    关联维数分析: 使用 Grassberger-Procaccia 算法计算混沌吸引子的 D2。
    """
    print("\n" + "=" * 60)
    print("第 4 阶段 · 关联维数 (Grassberger-Procaccia)")
    print("=" * 60)

    from 关联维数 import compute_correlation_dimension

    pendulum = DoublePendulum(DEFAULT_PARAMS)
    T, dt = 200.0, 0.005

    print("  积分混沌轨线 (T=200s, dt=0.005)...")
    result = rk4(pendulum, (0.0, T), INITIAL_STATE, dt)

    print("  计算关联维数 (m_max=8, tau=10)...")
    corr_info = compute_correlation_dimension(result, theta_index=0,
                                              m_max=8, tau=10)

    for m, d2 in zip(corr_info["m_vals"], corr_info["D2_vals"]):
        if not np.isnan(d2):
            print(f"    m={m}: D2 ≈ {d2:.3f}")
        else:
            print(f"    m={m}: D2 = NaN (拟合失败)")

    viz.plot_correlation_dimension(corr_info, "correlation_dimension.png")
    print("  ✓ 关联维数分析完成")


def run_runtime_benchmark():
    """
    运行效率基准: 测量四种方法在相同条件下的纯计算时间。
    """
    print("\n" + "=" * 60)
    print("第 4 阶段 · 运行效率基准测试")
    print("=" * 60)

    import time

    pendulum = DoublePendulum(DEFAULT_PARAMS)
    T, dt = 40.0, 0.005
    n_steps_expected = int(T / dt)
    n_trials = 5

    methods = [
        ("Forward Euler", forward_euler),
        ("RK4", rk4),
        ("Symplectic Euler", symplectic_euler),
        ("Velocity Verlet", velocity_verlet),
    ]

    method_names = []
    avg_times = []
    step_counts = []

    for name, solver in methods:
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            result = solver(pendulum, (0.0, T), INITIAL_STATE, dt)
            times.append(time.perf_counter() - t0)
        avg_t = np.mean(times)
        method_names.append(name)
        avg_times.append(avg_t)
        step_counts.append(result.n_steps)
        print(f"  {name:20s}: {avg_t:.4f}s avg ({n_trials} trials), "
              f"{result.n_steps} steps, {result.n_steps / avg_t:.0f} steps/s")

    viz.plot_runtime_benchmark(method_names, avg_times, step_counts,
                               "runtime_benchmark.png")
    print("  ✓ 运行效率基准图完成")


def run_animation_hires():
    """
    高分辨率动画: 更高 DPI, 更长轨迹拖尾。
    """
    print("\n" + "=" * 60)
    print("第 4 阶段 · 高分辨率动画 (高DPI)")
    print("=" * 60)

    pendulum = DoublePendulum(DEFAULT_PARAMS)
    T, dt = 40.0, 0.005

    methods = [
        ("Forward Euler", forward_euler),
        ("RK4", rk4),
        ("Symplectic Euler", symplectic_euler),
        ("Velocity Verlet", velocity_verlet),
    ]

    for method_name, solver in methods:
        print(f"\n  [{method_name}]")
        result = solver(pendulum, (0.0, T), INITIAL_STATE, dt)
        print(f"    积分: {result.n_steps} 步, {result.cpu_time:.2f}s")

        fname = f"anim_{method_name.lower().replace(' ', '_')}_hires.gif"
        viz.animate_double_pendulum(result, pendulum=pendulum,
                                    l1=DEFAULT_PARAMS.l1, l2=DEFAULT_PARAMS.l2,
                                    save_name=fname,
                                    fps=15, skip=3, trail_length=9999,
                                    duration=120.0)

    print("\n  ✓ 高分辨率动画生成完成")


def run_bifurcation_dense():
    """
    加密分岔图: 200 个 A 值, 更高参数分辨率以捕捉周期窗口。
    """
    print("\n" + "=" * 60)
    print("第 4 阶段 · 加密分岔图 (200 A 值)")
    print("=" * 60)

    A_vals = np.linspace(0.3, 1.8, 200)
    b, Omega = 0.5, 2.0

    print(f"  驱动振幅 A: {A_vals[0]:.1f} ~ {A_vals[-1]:.1f} ({len(A_vals)} 个值)")
    print(f"  比默认多 2.5 倍采样点, 预计耗时较长 (~15-20 min) ...")

    samples = analysis.bifurcation_data(A_vals, b=b, Omega=Omega)

    viz.plot_bifurcation(A_vals, samples, save_name="bifurcation_dense.png")
    print("  ✓ 加密分岔图完成")


# ══════════════════════════════════════════════════════════════════════
# 命令行入口
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="混沌双摆数值模拟 — 计算物理期末项目"
    )
    parser.add_argument("--quick", action="store_true",
                        help="快速验证模式 (仅收敛性 + 能量守恒)")
    parser.add_argument("--full", action="store_true",
                        help="Phase 1 完整模式 (验证 + 轨迹 + 敏感性)")
    parser.add_argument("--phase2", action="store_true",
                        help="Phase 2 分析模式 (相图 + FFT + 稳定性 + 精度基准)")
    parser.add_argument("--phase3", action="store_true",
                        help="Phase 3 混沌诊断 (Poincaré + 混沌地图 + 分岔图)")
    parser.add_argument("--phase4", action="store_true",
                        help="Phase 4 高级分析 (关联维数 + 效率基准 + 加密分岔 + 高清动画)")
    parser.add_argument("--all", action="store_true",
                        help="全部运行 (Phase 1 + Phase 2 + Phase 3 + Phase 4)")
    parser.add_argument("--verify-only", action="store_true",
                        help="仅运行三个验证模块")
    args = parser.parse_args()

    print("=" * 60)
    print("  混沌双摆数值模拟")
    print("  计算物理期末项目 — 王翔熙")
    print("=" * 60)
    print(f"\n物理参数: m1={DEFAULT_PARAMS.m1}, m2={DEFAULT_PARAMS.m2}, "
          f"l1={DEFAULT_PARAMS.l1}, l2={DEFAULT_PARAMS.l2}, g={DEFAULT_PARAMS.g}")
    print(f"初始条件: θ1={INITIAL_STATE[0]:.2f}, θ2={INITIAL_STATE[1]:.2f}, "
          f"ω1={INITIAL_STATE[2]}, ω2={INITIAL_STATE[3]}")
    print()

    run_all_phases = args.all
    run_phase1 = args.full or run_all_phases
    run_phase2 = args.phase2 or run_all_phases
    run_phase3 = args.phase3 or run_all_phases
    run_phase4 = args.phase4 or run_all_phases

    # 始终运行验证
    verify_single_pendulum()
    verify_convergence()
    verify_energy()

    if args.verify_only:
        print("\n✓ 验证完成")
        return

    if args.quick:
        print("\n[quick 模式] 完成")
        return

    # Phase 1
    if run_phase1:
        run_trajectory_comparison()
        run_sensitivity_demo()

    # Phase 2
    if run_phase2:
        run_phase_portraits()
        run_fft_analysis()
        run_energy_stability_extended()
        run_precision_benchmark()
        run_animation()

    # Phase 3
    if run_phase3:
        run_poincare()
        run_chaos_map()
        run_bifurcation()

    # Phase 4
    if run_phase4:
        run_correlation_dimension()
        run_runtime_benchmark()
        run_bifurcation_dense()
        run_animation_hires()

    if not run_phase1 and not run_phase2 and not run_phase3 and not run_phase4 and not args.quick:
        print("\n请指定运行模式: --full / --phase2 / --phase3 / --phase4 / --all")
        return

    print("\n✓ 全部完成")


if __name__ == "__main__":
    main()
