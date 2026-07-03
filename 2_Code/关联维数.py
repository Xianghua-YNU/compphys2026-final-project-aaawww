# ══════════════════════════════════════════════════════════════════════
# 关联维数 (Grassberger-Procaccia 算法)
# ══════════════════════════════════════════════════════════════════════

import numpy as np
from 求解器 import SolverResult


def time_delay_embed(
    signal: np.ndarray,
    m: int,
    tau: int = 1,
) -> np.ndarray:
    """
    时延嵌入重构相空间 (Takens 定理)。

    参数
    ----------
    signal : 1D 时域信号 [N]
    m : 嵌入维数
    tau : 时延 (采样点数)

    返回
    -------
    embedded : [N - (m-1)*tau, m] 重构相空间向量
    """
    n = len(signal) - (m - 1) * tau
    embedded = np.zeros((n, m))
    for i in range(m):
        embedded[:, i] = signal[i * tau : i * tau + n]
    return embedded


def correlation_sum(
    embedded: np.ndarray,
    epsilons: np.ndarray,
    max_pairs: int = 50000,
) -> np.ndarray:
    """
    计算关联求和 C(ε) = (2/N(N-1)) Σ Θ(ε - |x_i - x_j|)
    """
    n = len(embedded)
    rng = np.random.default_rng(42)
    idx_i = rng.integers(0, n, max_pairs)
    idx_j = rng.integers(0, n, max_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    diffs = embedded[idx_i] - embedded[idx_j]

    dists = np.linalg.norm(diffs, axis=1)
    dists = dists[dists > 0]

    C = np.array([np.mean(dists < eps) for eps in epsilons])
    return C


def compute_correlation_dimension(
    result: SolverResult,
    theta_index: int = 0,
    m_max: int = 8,
    tau: int = 10,
    n_eps: int = 40,
) -> dict:
    """
    计算关联维数 D2 (Grassberger-Procaccia 算法)。

    对每个嵌入维数 m 在 log-log 图的线性区拟合斜率 → D2(m)。
    D2 随 m 增大而收敛 → 真实关联维数。

    返回
    -------
    info : 包含 m_vals, D2_vals, epsilons, slopes_by_m 的字典
    """
    signal = result.y[:, theta_index]
    n_total = len(signal)

    # 使用信号后半段 (去除瞬态)
    signal = signal[n_total // 3:]
    n_use = min(len(signal), 4000)
    signal = signal[-n_use:]

    # 归一化
    signal = (signal - np.mean(signal)) / np.std(signal)

    # 距离阈值范围 (对数等间距)
    epsilons = np.logspace(-2.5, 0.8, n_eps)

    D2_vals = []
    m_vals = list(range(2, m_max + 1))
    slopes_by_m = {}

    for m in m_vals:
        embedded = time_delay_embed(signal, m, tau)
        C = correlation_sum(embedded, epsilons)

        log_eps = np.log10(epsilons)
        log_C = np.log10(C + 1e-15)

        # 自动选取线性区: C 在 [0.005, 0.5] 范围
        mask = (C > 0.005) & (C < 0.5)
        if np.sum(mask) < 5:
            D2_vals.append(np.nan)
            slopes_by_m[m] = (log_eps, log_C, mask, np.nan)
            continue

        coeffs = np.polyfit(log_eps[mask], log_C[mask], 1)
        D2_vals.append(coeffs[0])
        slopes_by_m[m] = (log_eps.copy(), log_C.copy(), mask.copy(), coeffs[0])

    return {
        "m_vals": np.array(m_vals),
        "D2_vals": np.array(D2_vals),
        "epsilons": epsilons,
        "slopes_by_m": slopes_by_m,
    }
