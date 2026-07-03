# 2_Code/ - 混沌双摆数值模拟源代码

## 文件说明

| 文件 | 功能 |
|------|------|
| `double_pendulum.py` | 物理模型: 双摆 ODE 系统 (含阻尼驱动拓展)、能量计算 |
| `solvers.py` | ODE 求解器集合: Euler / RK4 / Symplectic Euler / Verlet / RK45 |
| `main.py` | **入口模块**: 参数配置、验证流程、轨迹对比、初值敏感性演示 |
| `analysis.py` | 混沌诊断: FFT 功率谱、Lyapunov 指数、Poincaré 截面 |
| `visualization.py` | 可视化: 相空间轨迹图、双摆运动动画、能量漂移图 |
| `requirements.txt` | Python 环境依赖清单 |

## 环境配置

```bash
pip install -r requirements.txt
```

核心依赖: `numpy`, `scipy`, `matplotlib`

## 如何运行

### 快速验证 (推荐首次运行)
```bash
python main.py --quick
```
- 退化到单摆验证
- 收敛阶测试 (log-log 图)
- 能量守恒性对比

### 完整模拟
```bash
python main.py --full
```
额外运行:
- 各方法轨迹对比 (θ1(t) 轨线)
- 初值敏感性演示 (蝴蝶效应 + Lyapunov 指数估算)

### 仅运行验证模块
```bash
python main.py --verify-only
```

## 模块结构

```
main.py  ──→  double_pendulum.py  (物理模型, ODE 右端)
         ──→  solvers.py          (四种求解器 + RK45 参考解)
         ──→  analysis.py         (混沌诊断分析)
         ──→  visualization.py    (图表与动画)
```

## 代码规范

- 所有物理参数在 `main.py` 顶部集中定义 (`PendulumParams` 数据类)
- 求解器保持独立性, 不含绘图逻辑
- 变量命名具有物理含义 (theta, omega, alpha 等)
- 核心方程与算法步骤包含物理注释

## AI 使用声明

本项目使用 DeepSeek-V4-pro 辅助完成: ODE 求解公式推导、求解器代码实现、
主流程代码编写。所有 AI 辅助部分在相应文件头部已注明。
