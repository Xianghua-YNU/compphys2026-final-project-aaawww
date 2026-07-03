"""
论文图表统一配置

为《大学物理》期刊定制 matplotlib 图表风格:
  - 文本字体: Times New Roman (与模板一致)
  - 数学字体: Computer Modern (matplotlib 内置, 无需外部 LaTeX)
  - 输出分辨率: 600 DPI (模板最低要求)
  - 图表尺寸: 单栏 8cm / 双栏 16cm 宽度 (模板要求)

用法:
    from 图表配置 import apply_paper_style, save_figure
    apply_paper_style()          # 全局生效
    fig = plt.figure(...)
    save_figure(fig, "output.png")

AI 使用声明: DeepSeek-V4-pro 辅助编写。
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# 输出目录
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "3_Data"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "1_论文" / "assets"

# 论文图表尺寸 (《大学物理》模板要求)
# 单栏: 8 cm, 双栏: 16 cm
SINGLE_COL_WIDTH = 3.15   # inches (8 cm / 2.54)
DOUBLE_COL_WIDTH = 6.30   # inches (16 cm / 2.54)
GOLDEN_RATIO = 0.618       # 图表高宽比

# 字体大小 (pt)
TITLE_SIZE = 10
LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 7.5


def _has_latex() -> bool:
    """检测系统是否安装 LaTeX (pylatex 可用)。"""
    try:
        import shutil
        return shutil.which("latex") is not None
    except Exception:
        return False


def apply_paper_style():
    """
    应用论文级 matplotlib 全局样式。

    优先级: 若系统有 LaTeX → 使用 LaTeX 渲染;
            否则 → 使用 matplotlib 内置 cm 数学字体 + Times New Roman 文本。
    """
    if _has_latex():
        _apply_latex_style()
    else:
        _apply_native_style()


def _apply_latex_style():
    """LaTeX 渲染 (系统已安装 LaTeX 时使用)。"""
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": (
            r"\usepackage{amsmath}"
            r"\usepackage{newtxtext,newtxmath}"  # Times-like font for LaTeX
            r"\usepackage[T1]{fontenc}"
        ),
        "font.family": "serif",
        "font.size": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.which": "both",
        "grid.alpha": 0.25,
        "grid.linewidth": 0.3,
        "lines.linewidth": 0.8,
        "lines.markersize": 4.0,
        "mathtext.fontset": "custom",
    })


def _apply_native_style():
    """
    matplotlib 原生渲染 (无需外部 LaTeX)。

    使用内置 Computer Modern (cm) 数学字体 + Times New Roman 文本,
    与《大学物理》模板的排版风格一致。
    """
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "SimSun", "serif"],
        "font.size": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.which": "both",
        "grid.alpha": 0.25,
        "grid.linewidth": 0.3,
        "lines.linewidth": 0.8,
        "lines.markersize": 4.0,
        # 内置 Computer Modern 数学字体 (Bakoma 字体, 随 matplotlib 分发)
        "mathtext.fontset": "cm",
        "mathtext.default": "it",
    })


def single_column_figure(aspect: float = GOLDEN_RATIO) -> tuple[plt.Figure, plt.Axes]:
    """创建单栏宽度 (8cm) 图。"""
    fig, ax = plt.subplots(
        figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * aspect)
    )
    return fig, ax


def double_column_figure(aspect: float = GOLDEN_RATIO) -> tuple[plt.Figure, plt.Axes]:
    """创建双栏宽度 (16cm) 图。"""
    fig, ax = plt.subplots(
        figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * aspect)
    )
    return fig, ax


def save_figure(
    fig: plt.Figure,
    name: str,
    out_dir: str = "data",
    close: bool = True,
):
    """
    保存图表到 3_Data/ 或 1_论文/assets/。

    参数
    ----------
    fig : matplotlib Figure
    name : 文件名 (如 "convergence_test.png")
    out_dir : "data" → 3_Data/, "paper" → 1_论文/assets/
    close : 保存后是否关闭图表
    """
    if out_dir == "paper":
        base = ASSETS_DIR
        label = "论文插图"
    else:
        base = OUTPUT_DIR
        label = "数据"
    base.mkdir(parents=True, exist_ok=True)

    path = base / name
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    print(f"  [{label}] {path}")

    if close:
        plt.close(fig)


def save_figure_both(fig: plt.Figure, name: str, close: bool = True):
    """同时保存到 3_Data/ 和 1_论文/assets/。"""
    save_figure(fig, name, out_dir="data", close=False)
    save_figure(fig, name, out_dir="paper", close=close)


# ══════════════════════════════════════════════════════════════════════
# 初始化: 模块导入时自动应用论文样式
# ══════════════════════════════════════════════════════════════════════

apply_paper_style()
