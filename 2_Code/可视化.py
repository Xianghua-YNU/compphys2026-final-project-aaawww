"""
可视化模块

负责生成论文所需的全部图表与动画:
  - 四方法相空间轨迹对比 (2×2 面板)
  - FFT 功率谱分析 (周期 vs 混沌)
  - 能量守恒性对比 (长时间)
  - 精度-效率基准测试
  - 双摆运动轨迹动画

AI 使用声明: DeepSeek-V4-pro 辅助编写。
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

import 图表配置 as fc
from 求解器 import SolverResult
from 混沌分析 import compute_fft_from_result, find_peak_frequencies

# 四种方法统一配色方案
METHOD_COLORS = {
    "Forward Euler":      "#d62728",  # 红色
    "RK4":                "#1f77b4",  # 蓝色
    "Symplectic Euler":   "#ff7f0e",  # 橙色
    "Velocity Verlet":    "#2ca02c",  # 绿色
    "RK45 (SciPy ref.)":  "#7f7f7f",  # 灰色
}
METHOD_COLORS_SHORT = {
    "Euler":    "#d62728",
    "RK4":      "#1f77b4",
    "SympEul":  "#ff7f0e",
    "Verlet":   "#2ca02c",
}


def _get_color(name: str) -> str:
    """根据求解器名称返回统一颜色。"""
    for key, color in METHOD_COLORS.items():
        if key.lower() in name.lower():
            return color
    for key, color in METHOD_COLORS_SHORT.items():
        if key.lower() in name.lower():
            return color
    return "#333333"


# ══════════════════════════════════════════════════════════════════════
# 1. 四方法相空间轨迹对比 (核心图表)
# ══════════════════════════════════════════════════════════════════════

def plot_phase_portrait_4panel(
    results: dict[str, SolverResult],
    save_name: str = "phase_portrait_4panel.png",
) -> plt.Figure:
    """
    2×2 面板: 四种方法在相同初始条件和步长下的 θ1-θ2 相空间轨迹。

    每张图用时间颜色编码 (viridis), 起始为紫色, 结束为黄色。
    直观展示不同数值方法在相空间中的轨迹差异。
    """
    fig, axes = plt.subplots(2, 2, figsize=(fc.DOUBLE_COL_WIDTH,
                                            fc.DOUBLE_COL_WIDTH * 0.95))
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    for (name, result), ax in zip(results.items(), axes.flat):
        theta1 = result.theta1
        theta2 = result.theta2
        t_norm = (result.t - result.t[0]) / (result.t[-1] - result.t[0])

        # 用 LineCollection 实现时间颜色编码
        points = np.column_stack([theta1, theta2])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="viridis", array=t_norm[:-1],
                            alpha=0.75, lw=0.4)
        ax.add_collection(lc)
        ax.autoscale()

        # 标注起点
        ax.scatter(theta1[0], theta2[0], c="blue", s=20, zorder=5,
                   marker="o", label="Start")
        ax.scatter(theta1[-1], theta2[-1], c="red", s=20, zorder=5,
                   marker="s", label="End")

        ax.set_title(name, fontsize=fc.TITLE_SIZE, fontweight="bold")
        ax.set_xlabel(r"$\theta_1$ [rad]")
        ax.set_ylabel(r"$\theta_2$ [rad]")
        ax.axhline(0, color="gray", lw=0.3)
        ax.axvline(0, color="gray", lw=0.3)
        ax.set_aspect("equal")
        ax.legend(fontsize=6, loc="upper right")

    # 统一颜色条
    cbar = fig.colorbar(lc, ax=list(axes.flat), shrink=0.92, pad=0.02)
    cbar.set_label("Normalized time", fontsize=fc.LABEL_SIZE)

    fig.suptitle("Phase Portraits: Method Comparison", fontsize=fc.TITLE_SIZE + 1,
                 fontweight="bold", y=1.01)
    fc.save_figure_both(fig, save_name)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 2. FFT 功率谱: 周期 vs 混沌
# ══════════════════════════════════════════════════════════════════════

def plot_fft_periodic_vs_chaotic(
    result_periodic: SolverResult,
    result_chaotic: SolverResult,
    save_name: str = "fft_periodic_vs_chaotic.png",
) -> plt.Figure:
    """
    2×2 面板: 周期运动 vs 混沌运动 FFT 功率谱对比。

    上排: 周期运动 → 离散尖峰
    下排: 混沌运动 → 宽频连续谱
    坐标轴自适应数据范围。
    """
    fig, axes = plt.subplots(2, 2, figsize=(fc.DOUBLE_COL_WIDTH,
                                            fc.DOUBLE_COL_WIDTH * 0.9))
    fig.subplots_adjust(hspace=0.38, wspace=0.22)

    configs = [
        (result_periodic, "Periodic Regime"),
        (result_chaotic, "Chaotic Regime"),
    ]
    colors_theta = {0: ("#2980b9", "#3498db"), 1: ("#c0392b", "#e74c3c")}

    for row, (result, label) in enumerate(configs):
        for col, theta_idx in enumerate([0, 1]):
            ax = axes[row, col]
            freqs, power = compute_fft_from_result(result, theta_idx)

            # 自适应频率范围
            significant = np.where(power > 1e-4)[0]
            f_cut = min(freqs[significant[-1]] * 1.3, freqs[-1]) if len(significant) > 0 else 5.0
            mask = freqs <= f_cut

            fill_c, line_c = colors_theta[theta_idx]
            p_plot = power[mask] + 1e-12

            ax.fill_between(freqs[mask], 1e-8, p_plot, color=fill_c, alpha=0.18)
            ax.semilogy(freqs[mask], p_plot, lw=0.8, alpha=0.9, color=line_c)

            # 标注峰值
            peaks = find_peak_frequencies(freqs, power, threshold=0.03, max_peaks=5)
            for pk in peaks:
                pk_idx = np.argmin(np.abs(freqs - pk))
                ax.axvline(x=pk, color=line_c, lw=0.5, alpha=0.5,
                           linestyle="--", ymax=0.85)

            # 自适应 y 轴
            p_pos = power[power > 1e-15]
            if len(p_pos) > 0:
                y_lo = max(np.percentile(p_pos, 2), 1e-6)
                y_hi = min(np.max(p_pos) * 2, 1e2)
                ax.set_ylim(y_lo, y_hi)

            ax.set_xlim(0, f_cut * 1.05)
            ax.set_title(fr"$\theta_{theta_idx+1}$ — {label}", fontsize=fc.TITLE_SIZE)
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Power")
            ax.grid(True, alpha=0.25, which="both")

    fig.suptitle("FFT Power Spectrum: Periodic vs Chaotic Motion",
                 fontsize=fc.TITLE_SIZE + 1, fontweight="bold", y=1.02)
    fc.save_figure_both(fig, save_name)
    return fig


def plot_fft_method_comparison(
    results: dict[str, SolverResult],
    save_name: str = "fft_method_comparison.png",
) -> plt.Figure:
    """
    2×2 面板: 四种方法的 FFT 功率谱 (混沌运动), 各自自适应缩放。

    每个面板:
      - 蓝色填充区域 = θ1 功率谱
      - 红色填充区域 = θ2 功率谱
      - 虚竖线标注主峰值频率
      - y 轴自动缩放到数据范围
    """
    fig, axes = plt.subplots(2, 2, figsize=(fc.DOUBLE_COL_WIDTH,
                                            fc.DOUBLE_COL_WIDTH * 0.9))
    fig.subplots_adjust(hspace=0.38, wspace=0.22)

    colors_theta = {0: ("#2980b9", "#3498db"),  # θ1: 深蓝填充 / 浅蓝线
                    1: ("#c0392b", "#e74c3c")}  # θ2: 深红填充 / 浅红线

    for (name, result), ax in zip(results.items(), axes.flat):
        f_max = 0.0
        for theta_idx, theta_label in [(0, r"$\theta_1$"), (1, r"$\theta_2$")]:
            freqs, power = compute_fft_from_result(result, theta_idx)

            # 自适应: 截断到有意义的频率范围 (功率 > 1e-4 的最高频率 × 1.2)
            significant = np.where(power > 1e-4)[0]
            if len(significant) > 0:
                f_cut = min(freqs[significant[-1]] * 1.3, freqs[-1])
            else:
                f_cut = 5.0
            f_max = max(f_max, f_cut)

            mask = freqs <= f_cut
            f_plot = freqs[mask]
            p_plot = power[mask] + 1e-12

            fill_c, line_c = colors_theta[theta_idx]
            # 半透明填充
            ax.fill_between(f_plot, 1e-8, p_plot, color=fill_c, alpha=0.18)
            ax.semilogy(f_plot, p_plot, lw=0.8, alpha=0.9, color=line_c,
                        label=theta_label)

            # 标记峰值 (竖虚线)
            peaks = find_peak_frequencies(freqs, power, threshold=0.03, max_peaks=4)
            for pk in peaks:
                pk_idx = np.argmin(np.abs(freqs - pk))
                pk_power = power[pk_idx]
                ax.axvline(x=pk, color=line_c, lw=0.5, alpha=0.5,
                           linestyle="--", ymax=0.85)

        # 自适应坐标轴
        ax.set_xlim(0, f_max * 1.05)
        y_data = []
        for ti in [0, 1]:
            _, p = compute_fft_from_result(result, ti)
            y_data.append(p[p > 0])
        if y_data:
            all_p = np.concatenate(y_data)
            y_lo = max(np.percentile(all_p, 1), 1e-6)
            y_hi = min(np.max(all_p) * 2, 1e2)
            ax.set_ylim(y_lo, y_hi)

        ax.set_title(name, fontsize=fc.TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
        ax.grid(True, alpha=0.25, which="both")

    fig.suptitle("FFT Power Spectrum — Method Comparison (Chaotic Regime)",
                 fontsize=fc.TITLE_SIZE + 1, fontweight="bold", y=1.02)
    fc.save_figure_both(fig, save_name)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 3. 长时间能量守恒性对比
# ══════════════════════════════════════════════════════════════════════

def plot_energy_stability_extended(
    results: dict[str, SolverResult],
    pendulum,
    save_name: str = "energy_stability_extended.png",
) -> plt.Figure:
    """
    上下分面板: 长时间 (T=2000s) 能量漂移对比。

    上: Euler (大漂移, 独立 y 轴)
    下: RK4 + Symplectic Euler + Verlet (放大细节)

    预期: Symplectic 方法能量有界振动, 非 symplectic 方法单调漂移。
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(fc.DOUBLE_COL_WIDTH, fc.DOUBLE_COL_WIDTH * 0.85),
        sharex=True
    )
    fig.subplots_adjust(hspace=0.12)

    # 分离 Euler 和其他方法
    euler_result = None
    other_results = {}
    for name, result in results.items():
        if "euler" in name.lower() and "symplectic" not in name.lower():
            euler_result = (name, result)
        else:
            other_results[name] = result

    # ── 上: Euler ──
    if euler_result:
        name, result = euler_result
        E = np.array([pendulum.total_energy(yi) for yi in result.y])
        E0 = E[0]
        e0_abs = abs(E0) if abs(E0) > 1e-12 else 1.0
        delta_rel = (E - E0) / e0_abs

        # 用双 y 轴: 左=相对偏差, 右=绝对偏差
        ax_top_r = ax_top.twinx()
        ax_top.plot(result.t, delta_rel, lw=0.6, color="#d62728", alpha=0.85)
        ax_top_r.semilogy(result.t, np.abs(E - E0) + 1e-16, lw=0.4,
                          color="#d62728", alpha=0.3)

        ax_top.set_ylabel(r"$\Delta E / |E_0|$", color="#d62728")
        ax_top_r.set_ylabel(r"$|\Delta E|$ [J]", color="#d62728", alpha=0.5)
        ax_top.tick_params(axis='y', labelcolor="#d62728")
        ax_top_r.tick_params(axis='y', labelcolor="#d62728", labelsize=7)
        ax_top.set_title("Forward Euler (rapid energy divergence)", fontsize=fc.TITLE_SIZE)
        ax_top.axhline(0, color="k", lw=0.3, alpha=0.5)
        ax_top.grid(True, alpha=0.2)
        ax_top_r.grid(False)

    # ── 下: 其余方法 ──
    for name, result in other_results.items():
        E = np.array([pendulum.total_energy(yi) for yi in result.y])
        E0 = E[0]
        e0_abs = abs(E0) if abs(E0) > 1e-12 else 1.0
        delta_rel = (E - E0) / e0_abs
        color = _get_color(name)

        ax_bot.plot(result.t, delta_rel, lw=0.7, alpha=0.9, color=color,
                    label=name)

    ax_bot.set_xlabel("Time t [s]")
    ax_bot.set_ylabel(r"$\Delta E / |E_0|$")
    ax_bot.set_title("RK4, Symplectic Euler, Velocity Verlet (zoomed detail)",
                     fontsize=fc.TITLE_SIZE)
    ax_bot.axhline(0, color="k", lw=0.4)
    ax_bot.legend(fontsize=8, ncol=3, loc="upper right")
    ax_bot.grid(True, alpha=0.25)

    fig.suptitle("Long-Term Energy Conservation (T = 2000 s)",
                 fontsize=fc.TITLE_SIZE + 1, fontweight="bold", y=1.01)
    fc.save_figure_both(fig, save_name)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 4. 精度-效率基准测试
# ══════════════════════════════════════════════════════════════════════

def plot_precision_benchmark(
    benchmark_data: dict,
    save_name: str = "precision_benchmark.png",
) -> plt.Figure:
    """
    精度-效率散点图: 各方法在不同步长下的全局误差 vs CPU 时间。

    参数
    ----------
    benchmark_data : {method_name: {"dt": [...], "error": [...], "cpu": [...]}}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fc.DOUBLE_COL_WIDTH * 1.1,
                                                   fc.DOUBLE_COL_WIDTH * 0.55))

    markers = {"Forward Euler": "o", "RK4": "s", "Symplectic Euler": "D",
               "Velocity Verlet": "^"}

    for name, data in benchmark_data.items():
        color = _get_color(name)
        marker = markers.get(name, "o")
        dt_arr = np.array(data["dt"])
        error_arr = np.array(data["error"])
        cpu_arr = np.array(data["cpu"])

        # 左: Error vs dt
        ax1.loglog(dt_arr, error_arr, marker=marker, color=color,
                   label=name, markersize=5, lw=1, alpha=0.85)

        # 右: Error vs CPU
        ax2.loglog(cpu_arr, error_arr, marker=marker, color=color,
                   label=name, markersize=5, lw=1, alpha=0.85)

        # 标注 dt 值
        for dt_val, err_val in zip(dt_arr, error_arr):
            ax1.annotate(f"{dt_val:.3f}", (dt_val, err_val),
                         textcoords="offset points", xytext=(0, -12),
                         fontsize=5, ha="center", alpha=0.6)

    ax1.set_xlabel("Step size h [s]")
    ax1.set_ylabel(r"Global error $\|y(T) - y_{\rm ref}\|$")
    ax1.set_title("Accuracy vs Step Size")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, which="both")

    ax2.set_xlabel("CPU time [s]")
    ax2.set_ylabel(r"Global error $\|y(T) - y_{\rm ref}\|$")
    ax2.set_title("Accuracy vs Computational Cost")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle("Precision–Efficiency Benchmark (T = 10 s)",
                 fontsize=fc.TITLE_SIZE + 1, fontweight="bold")
    fc.save_figure_both(fig, save_name)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 5. 双摆运动动画
# ══════════════════════════════════════════════════════════════════════

def animate_double_pendulum(
    result: SolverResult,
    pendulum=None,
    l1: float = 1.0,
    l2: float = 1.0,
    save_name: str = "double_pendulum_anim.mp4",
    fps: int = 20,
    skip: int = 2,
    trail_length: int = 120,
    duration: float = 12.0,
) -> animation.FuncAnimation:
    """
    创建高质量双摆运动动画。

    视觉要素:
      - 双色摆臂 (臂1 深蓝, 臂2 深红)
      - 三个质点 (支点黑, m1 蓝, m2 红)
      - m2 拖尾 (渐变透明度)
      - 实时能量显示 (T / V / E)
      - 时间戳

    参数
    ----------
    result : 求解器结果
    pendulum : DoublePendulum 对象 (用于计算能量)
    l1, l2 : 摆长 [m]
    save_name : 输出文件名
    fps : 帧率
    skip : 采样间隔
    trail_length : 拖尾点数
    duration : 动画时长 [s]
    """
    # 下采样
    t_full = result.t[::skip]
    theta1_full = result.theta1[::skip]
    theta2_full = result.theta2[::skip]
    y_full = result.y[::skip]

    # duration = 动画时长 [s]; 选取最后 duration*speed_factor 秒模拟数据
    sim_per_frame = (result.t[1] - result.t[0]) * skip  # 每帧对应模拟秒数
    sim_duration = duration * sim_per_frame * fps       # 总共展示的模拟秒数
    n_anim_frames = int(duration * fps)

    if n_anim_frames < len(t_full):
        idx_end = len(t_full)
        idx_start = max(0, idx_end - n_anim_frames)
    else:
        idx_start = 0
        idx_end = len(t_full)
        n_anim_frames = idx_end - idx_start

    theta1 = theta1_full[idx_start:idx_end]
    theta2 = theta2_full[idx_start:idx_end]
    t_arr = t_full[idx_start:idx_end]
    y_arr = y_full[idx_start:idx_end]
    n_frames = len(theta1)

    # 笛卡尔坐标
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    # 预计算能量
    energy_vals = None
    if pendulum is not None:
        energy_vals = np.array([pendulum.total_energy(yi) for yi in y_arr])
        T_vals = np.array([pendulum.kinetic_energy(yi) for yi in y_arr])
        V_vals = np.array([pendulum.potential_energy(yi) for yi in y_arr])

    # 创建图形
    fig = plt.figure(figsize=(8, 6), dpi=120)
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    limit = l1 + l2 + 0.35
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_facecolor("#fafafa")
    ax.grid(True, alpha=0.2, lw=0.4)
    ax.set_xlabel("x [m]", fontsize=10)
    ax.set_ylabel("y [m]", fontsize=10)
    ax.set_title(f"Chaotic Double Pendulum  |  {result.method_name}",
                 fontsize=13, fontweight="bold", pad=12)

    # 支点
    (pivot,) = ax.plot([0], [0], "o", color="#2c3e50", markersize=6,
                        zorder=10, markeredgecolor="white", markeredgewidth=0.5)

    # 摆臂 1 (支点 → m1)
    (arm1,) = ax.plot([], [], "-", lw=3.0, color="#2980b9", solid_capstyle="round",
                       zorder=4)
    # 摆臂 2 (m1 → m2)
    (arm2,) = ax.plot([], [], "-", lw=2.5, color="#c0392b", solid_capstyle="round",
                       zorder=4)

    # 质点
    (mass1,) = ax.plot([], [], "o", color="#3498db", markersize=10,
                        zorder=5, markeredgecolor="#1a5276", markeredgewidth=1.2)
    (mass2,) = ax.plot([], [], "o", color="#e74c3c", markersize=12,
                        zorder=5, markeredgecolor="#7b241c", markeredgewidth=1.2)

    # 拖尾 (m2) — 红色不消失轨迹
    (trail,) = ax.plot([], [], lw=0.7, alpha=0.8, color="#e74c3c",
                        zorder=3, solid_capstyle="round")

    # 时间框
    time_text = ax.text(
        0.02, 0.97, "",
        transform=ax.transAxes, va="top", fontsize=11,
        fontfamily="monospace", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85)
    )

    # 能量框
    energy_text = ax.text(
        0.98, 0.97, "",
        transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85)
    )

    def init():
        arm1.set_data([], [])
        arm2.set_data([], [])
        mass1.set_data([], [])
        mass2.set_data([], [])
        trail.set_data([], [])
        time_text.set_text("")
        energy_text.set_text("")
        return arm1, arm2, mass1, mass2, trail, time_text, energy_text

    def animate(i):
        # 摆臂
        arm1.set_data([0, x1[i]], [0, y1[i]])
        arm2.set_data([x1[i], x2[i]], [y1[i], y2[i]])

        # 质点
        mass1.set_data([x1[i]], [y1[i]])
        mass2.set_data([x2[i]], [y2[i]])

        # 拖尾 (不消失, 完整轨迹)
        start = max(0, i - trail_length)
        trail.set_data(x2[start:i + 1], y2[start:i + 1])

        # 时间
        time_text.set_text(f"t = {t_arr[i]:5.2f} s")

        # 能量
        if energy_vals is not None:
            e0 = energy_vals[0]
            de = energy_vals[i] - e0
            sign = "+" if de >= 0 else "-"
            energy_text.set_text(
                f"T  = {T_vals[i]:6.3f} J\n"
                f"V  = {V_vals[i]:6.3f} J\n"
                f"E  = {energy_vals[i]:6.3f} J\n"
                f"ΔE = {sign}{abs(de):.2e} J"
            )

        return arm1, arm2, mass1, mass2, trail, time_text, energy_text

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames,
        interval=1000 / fps, blit=True,
    )

    # 保存
    fc.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    path = fc.ASSETS_DIR / save_name
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
        ani.save(str(path), writer=writer)
        print(f"  [动画] {path}")
    except (FileNotFoundError, RuntimeError):
        print("  [警告] 未找到 ffmpeg, 保存为 GIF (压缩模式)...")
        path_gif = fc.ASSETS_DIR / save_name.replace(".mp4", ".gif")
        fig.set_size_inches(5, 3.8)
        fig.set_dpi(60)
        ani.save(str(path_gif), writer="pillow", fps=10, dpi=60)
        print(f"  [动画] GIF 已保存: {path_gif}")

    plt.close(fig)
    return ani


# ══════════════════════════════════════════════════════════════════════
# 6. 收敛阶增强图 (含标注)
# ══════════════════════════════════════════════════════════════════════

def plot_convergence_enhanced(
    results_data: dict[str, dict],
    save_name: str = "convergence_enhanced.png",
) -> plt.Figure:
    """
    增强版收敛阶图: log-log 误差 vs 步长, 标注实测阶数。

    参数
    ----------
    results_data : {name: {"h": [...], "error": [...], "order": float}}
    """
    fig, ax = plt.subplots(figsize=(fc.SINGLE_COL_WIDTH * 1.6,
                                    fc.SINGLE_COL_WIDTH * 1.4))

    for name, data in results_data.items():
        h = np.array(data["h"])
        err = np.array(data["error"])
        order = data["order"]
        color = _get_color(name)

        ax.loglog(h, err, "o-", color=color, label=f"{name} (O(h^{order:.1f}))",
                  markersize=4, lw=0.9, alpha=0.85)

    # 理论参考线
    h_ref = np.array([results_data[list(results_data.keys())[0]]["h"][-1],
                       results_data[list(results_data.keys())[0]]["h"][0]])
    for order, style, label in [(1, "--", r"O(h)"), (2, "-.", r"O(h$^2$)"),
                                 (4, ":", r"O(h$^4$)")]:
        scale = 0.3 / (10 ** order)
        ax.loglog(h_ref, scale * h_ref ** order, "k" + style,
                  alpha=0.25, lw=0.8, label=label)

    ax.set_xlabel("Step size h [s]")
    ax.set_ylabel(r"Global error $\|y(T)-y_{\rm ref}(T)\|$")
    ax.set_title("Convergence Order Verification")
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fc.save_figure_both(fig, save_name)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Phase 3: Poincaré 截面
# ══════════════════════════════════════════════════════════════════════

def plot_poincare_sections(
    results: dict[str, SolverResult],
    section_var: str = "theta1",
    section_value: float = 0.0,
    save_name: str = "poincare_sections.png",
) -> plt.Figure:
    """
    Poincaré 截面 2×2 面板: 四种方法在截面上的相空间点。

    当 θ1 穿过 0 时, 记录 (θ2, ω2)。
    周期运动 → 少数孤立点; 混沌运动 → 弥散分布。
    每个面板独立自适应缩放。
    """
    from 混沌分析 import poincare_section as poincare

    fig, axes = plt.subplots(2, 2, figsize=(fc.DOUBLE_COL_WIDTH,
                                             fc.DOUBLE_COL_WIDTH * 0.95))
    axes = list(axes.flat)

    if section_var == "theta1":
        xlab, ylab = r"$\theta_2$ [rad]", r"$\omega_2$ [rad/s]"
    else:
        xlab, ylab = r"$\theta_1$ [rad]", r"$\omega_1$ [rad/s]"

    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        pts = poincare(result, section_var=section_var, section_value=section_value)

        if len(pts) > 1:
            color = _get_color(name)
            ax.scatter(pts[:, 0], pts[:, 1], s=3.0, c=color,
                       alpha=0.75, edgecolors="none", linewidths=0)

            # 自适应轴范围: 使用 2-98 百分位 + 10% 余量
            x_lo, x_hi = np.percentile(pts[:, 0], [1, 99])
            y_lo, y_hi = np.percentile(pts[:, 1], [1, 99])
            x_margin = max((x_hi - x_lo) * 0.15, 0.1)
            y_margin = max((y_hi - y_lo) * 0.15, 0.1)
            ax.set_xlim(x_lo - x_margin, x_hi + x_margin)
            ax.set_ylim(y_lo - y_margin, y_hi + y_margin)
        else:
            ax.text(0.5, 0.5, "No crossings", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="gray")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f"{name}  ({len(pts)} pts)", fontsize=fc.LABEL_SIZE)
        ax.grid(True, alpha=0.2, lw=0.3)

    fig.suptitle(f"Poincaré Section ($\\theta_1={{{section_value}}}$)",
                 fontsize=fc.TITLE_SIZE + 1, y=1.01)
    fig.tight_layout()
    fc.save_figure_both(fig, save_name)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Phase 3: 参数空间混沌地图
# ══════════════════════════════════════════════════════════════════════

def plot_chaos_map(
    lyap_map: np.ndarray,
    mass_ratios: np.ndarray,
    energies: np.ndarray,
    save_name: str = "chaos_map.png",
) -> plt.Figure:
    """
    参数空间 Lyapunov 指数热力图。

    参数
    ----------
    lyap_map : [N_E, N_R] λ 值矩阵
    mass_ratios : [N_R] 质量比轴
    energies : [N_E] 能量轴
    """
    fig, ax = plt.subplots(figsize=(fc.DOUBLE_COL_WIDTH,
                                     fc.DOUBLE_COL_WIDTH * 0.72))

    # 使用对称的 diverging colormap: 蓝色=规则, 红色=混沌
    vmax = max(np.nanmax(lyap_map), 0.01)
    vmin = -vmax * 0.05

    im = ax.pcolormesh(mass_ratios, energies, lyap_map,
                       cmap="RdBu_r", shading="auto",
                       vmin=vmin, vmax=vmax, rasterized=True)

    cbar = fig.colorbar(im, ax=ax, label=r"$\lambda_{\rm max}$ [1/s]",
                        pad=0.02)
    cbar.ax.tick_params(labelsize=fc.TICK_SIZE)

    ax.set_xlabel(r"Mass ratio $R = m_2 / m_1$")
    ax.set_ylabel(r"Initial energy $E_0$ [J]")
    ax.set_title("Chaos Map: Maximum Lyapunov Exponent in Parameter Space")
    ax.grid(False)

    fig.tight_layout()
    fc.save_figure_both(fig, save_name)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Phase 3: 分岔图
# ══════════════════════════════════════════════════════════════════════

def plot_bifurcation(
    A_vals: np.ndarray,
    samples: list[np.ndarray],
    save_name: str = "bifurcation.png",
) -> plt.Figure:
    """
    受迫阻尼双摆分岔图 (Stroboscopic Poincaré map)。

    参数
    ----------
    A_vals : [N_A] 驱动振幅数组
    samples : [N_A] 列表, 每个元素为该 A 值下的 θ1 采样数组
    """
    fig, ax = plt.subplots(figsize=(fc.DOUBLE_COL_WIDTH,
                                     fc.DOUBLE_COL_WIDTH * 0.7))

    for A, pts in zip(A_vals, samples):
        if len(pts) > 0:
            ax.plot(np.full_like(pts, A), pts, ".", color="#1f77b4",
                    alpha=0.6, markersize=1.2)

    # 自适应 y 轴
    all_pts = np.concatenate([p for p in samples if len(p) > 0])
    if len(all_pts) > 0:
        y_lo, y_hi = np.percentile(all_pts, [0.5, 99.5])
        y_m = (y_hi - y_lo) * 0.1
        ax.set_ylim(y_lo - y_m, y_hi + y_m)

    ax.set_xlabel(r"Driving amplitude $A$ [rad/s$^2$]")
    ax.set_ylabel(r"$\theta_1$ (stroboscopic) [rad]")
    ax.set_title("Bifurcation Diagram: Damped-Driven Double Pendulum")
    ax.grid(True, alpha=0.2, lw=0.3)

    fig.tight_layout()
    fc.save_figure_both(fig, save_name)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 关联维数可视化
# ══════════════════════════════════════════════════════════════════════

def plot_correlation_dimension(
    corr_info: dict,
    save_name: str = "correlation_dimension.png",
) -> plt.Figure:
    """
    关联维数分析图：左图 C(ε) vs ε (log-log)，右图 D2 vs m。
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(fc.DOUBLE_COL_WIDTH, fc.DOUBLE_COL_WIDTH * 0.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1])

    m_vals = corr_info["m_vals"]
    D2_vals = corr_info["D2_vals"]
    slopes_by_m = corr_info["slopes_by_m"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(m_vals)))

    # 左图: log C(ε) vs log ε
    ax1 = fig.add_subplot(gs[0])
    for idx, m in enumerate(m_vals):
        if m not in slopes_by_m:
            continue
        log_eps, log_C, mask, slope = slopes_by_m[m]
        ax1.plot(log_eps, log_C, color=colors[idx], lw=1.0, alpha=0.8,
                 label=f"$m={m}$")
        # 拟合线
        if not np.isnan(slope):
            fit_line = slope * log_eps[mask] + (log_C[mask][0] - slope * log_eps[mask][0])
            ax1.plot(log_eps[mask], fit_line, "--", color=colors[idx],
                     lw=0.6, alpha=0.5)

    ax1.set_xlabel(r"$\log_{10}\varepsilon$")
    ax1.set_ylabel(r"$\log_{10} C(\varepsilon)$")
    ax1.set_title("Correlation Integral")
    ax1.legend(fontsize=6, ncol=2, loc="lower right")
    ax1.grid(True, alpha=0.2, lw=0.3)

    # 右图: D2 vs m
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(m_vals, D2_vals, "o-", color="#2166ac", lw=1.5, ms=5,
             markerfacecolor="white", markeredgewidth=1.2)
    ax2.set_xlabel("Embedding dimension $m$")
    ax2.set_ylabel("Correlation dimension $D_2(m)$")
    ax2.set_title("$D_2$ Convergence")
    ax2.grid(True, alpha=0.2, lw=0.3)

    # D2 估计值 (取 m 最大时的值)
    valid = ~np.isnan(D2_vals)
    if np.any(valid):
        d2_est = D2_vals[valid][-1]
        ax2.axhline(y=d2_est, color="gray", linestyle=":", lw=0.8, alpha=0.6)
        ax2.text(m_vals[-1], d2_est + 0.1, f"$D_2 \\approx {d2_est:.2f}$",
                 fontsize=8, color="gray", ha="right")

    fig.tight_layout()
    fc.save_figure_both(fig, save_name)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 运行时基准测试可视化
# ══════════════════════════════════════════════════════════════════════

def plot_runtime_benchmark(
    methods: list[str],
    times: list[float],
    n_steps: list[int],
    save_name: str = "runtime_benchmark.png",
) -> plt.Figure:
    """
    数值方法运行效率对比柱状图。
    """
    fig, ax = plt.subplots(figsize=(fc.SINGLE_COL_WIDTH,
                                     fc.SINGLE_COL_WIDTH * 0.9))

    colors = ["#d73027", "#2166ac", "#f46d43", "#74add1"]
    bars = ax.bar(methods, times, color=colors, edgecolor="k", lw=0.5, width=0.6)

    # 标注运行时间
    for bar, t, n in zip(bars, times, n_steps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.02,
                f"{t:.3f} s", ha="center", va="bottom", fontsize=7)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{n} steps", ha="center", va="center", fontsize=6,
                color="white", fontweight="bold")

    ax.set_ylabel("Wall-clock time [s]")
    ax.set_title("Runtime Comparison ($T=40$ s, $\\Delta t=0.005$ s)")
    ax.grid(True, alpha=0.2, lw=0.3, axis="y")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=7)

    fig.tight_layout()
    fc.save_figure_both(fig, save_name)
    return fig
