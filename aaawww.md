[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/PbwMKW4u)

# 计算物理期末项目开题报告 (Proposal)

## 1. 项目基本信息
- **论文暂定标题**: 混沌双摆的数值模拟与动力学分析
- **小组名称**: 独立项目
- **完成人**: 王翔熙 (学号: [在此处填写])

## 2. 问题描述 (Why & What)

### 物理背景
双摆（Double Pendulum）是最简单的混沌系统之一：一个摆的末端连接第二个摆，构成两自由度的耦合系统。尽管系统本身完全确定（运动由 Newton/Lagrange 力学精确描述），在适当参数下其运动表现出对初始条件的极端敏感性，即"蝴蝶效应"。混沌双摆是经典力学通向非线性动力学的桥梁问题，在机器人运动规划、天体力学和分子动力学中均有对应模型。

### 控制方程
取广义坐标为两摆角 $\theta_1, \theta_2$，由 Lagrange 量 $L = T - V$ 出发：

系统的动能为：
$$
T = \frac{1}{2}(m_1+m_2)l_1^2\dot\theta_1^2 + \frac{1}{2}m_2l_2^2\dot\theta_2^2 + m_2l_1l_2\dot\theta_1\dot\theta_2\cos(\theta_1-\theta_2)
$$

势能为：
$$
V = -(m_1+m_2)gl_1\cos\theta_1 - m_2gl_2\cos\theta_2
$$

代入 Euler-Lagrange 方程后得到两个耦合的二阶 ODE：

$$
\begin{aligned}
(m_1+m_2)l_1\ddot\theta_1 &+ m_2l_2\ddot\theta_2\cos(\theta_1-\theta_2) + m_2l_2\dot\theta_2^2\sin(\theta_1-\theta_2) + (m_1+m_2)g\sin\theta_1 = 0 \\
l_2\ddot\theta_2 &+ l_1\ddot\theta_1\cos(\theta_1-\theta_2) - l_1\dot\theta_1^2\sin(\theta_1-\theta_2) + g\sin\theta_2 = 0
\end{aligned}
$$

消去耦合项后化为四个一阶 ODE：

$$
\frac{d}{dt}
\begin{pmatrix}
\theta_1 \\ \theta_2 \\ \omega_1 \\ \omega_2
\end{pmatrix}
=
\begin{pmatrix}
\omega_1 \\ \omega_2 \\ f_1(\theta_1, \theta_2, \omega_1, \omega_2) \\ f_2(\theta_1, \theta_2, \omega_1, \omega_2)
\end{pmatrix}
$$

此外，为丰富动力学行为，本课题还将拓展研究**阻尼驱动双摆**模型，在运动方程中加入阻尼项 $-b\dot\theta_i$ 和周期驱动力项 $A\sin(\Omega t)$：

$$
\ddot\theta_i \rightarrow \ddot\theta_i + b\dot\theta_i + A\sin(\Omega t)
$$

这将使系统出现倍周期分岔、极限环、受迫混沌等更丰富的非线性现象。

### 研究目标
1. 模拟混沌双摆的运动轨迹，直观展示混沌系统对初值的敏感性；
2. 系统对比 Euler / RK4 / Symplectic Euler / Verlet 四种方法的精度、能量守恒性与长期稳定性；
3. 通过 Lyapunov 指数、Poincaré 截面和 FFT 功率谱三种手段定量刻画混沌程度，区分周期运动与混沌运动；
4. 绘制参数空间混沌地图（质量比 vs 能量），揭示系统从规则到混沌的转变边界；
5. 拓展至阻尼驱动双摆，观察倍周期分岔通向混沌的经典路径。

## 3. 数值方法 (How)

### 拟采用的算法

本项目的核心工作之一是**系统对比不同数值方法**在混沌双摆问题上的表现，从精度、能量守恒性和长期稳定性三个维度进行评估。

#### (a) 前向 Euler 法
一阶显式格式，$O(h)$ 精度。作为基准对照方法，预期在双摆这一非线性系统中会很快发散、严重违背能量守恒。

#### (b) 四阶 Runge-Kutta 法 (RK4)
$O(h^4)$ 精度，是课程中重点学习的 ODE 求解器。适合混沌系统的中等时间积分，是本项目的主算法。

#### (c) Symplectic Euler 法
一阶 symplectic 格式，虽精度不高，但专门为 Hamiltonian 系统设计，能天然保持相空间体积（Liouville 定理），长时间积分中能量漂移远小于前向 Euler。

#### (d) Störmer-Verlet (Velocity-Verlet) 法
二阶 symplectic 格式，$O(h^2)$ 精度。广泛应用于分子动力学模拟。预期在长时间模拟中能量守恒性显著优于 RK4。

#### (e) 自适应步长 RK45 (scipy.integrate.solve_ivp)
作为"参考答案"：使用 SciPy 内置的 RK45 自适应步长求解器，设定严格的容差 (rtol=1e-9) 得到高精度参考解，用于衡量其他方法的误差。

### 对比方案

从四个维度系统对比上述方法：

| 对比维度 | 具体做法 | 预期结论 |
|---------|---------|---------|
| **精度对比** | 固定步长 $h$，以 RK45 参考解为基准，计算各方法在 $t=10$ 时的全局误差，绘制 $\log E$ vs $\log h$ 图，验证收敛阶 | Euler $O(h)$，RK4 $O(h^4)$，Verlet $O(h^2)$ |
| **能量守恒性** | 长时间积分（$t=1000$），绘制 $\Delta E/E_0$ 随时间变化曲线，对比各方法的能量漂移斜率 | Symplectic 方法能量有界振动而非漂移 |
| **长期稳定性** | 检查各方法能否在 $t=10000$ 时间范围内保持有界轨道 | 非 symplectic 方法可能发散 |
| **计算效率** | 记录达到相同精度所需的 CPU 时间与步数 | RK4 在中等精度下效率最优 |

### 验证方案
1. **能量守恒检查**：无耗散情形下总机械能应守恒，对比各方法的能量漂移率；
2. **与单摆对比**：将 $m_2 \to 0$ 退化到单摆，对比解析解验证代码正确性；
3. **收敛性测试**：绘制不同步长下的全局误差 log-log 图，验证收敛阶与理论值一致；
4. **守恒量检验**：检查 Hamiltonian 在长时间积分中的漂移量，对比 symplectic 与非 symplectic 方法。

### 混沌诊断与分析方法

#### (a) FFT 功率谱分析
对 $\theta_1(t)$ 和 $\theta_2(t)$ 做快速傅里叶变换，计算功率谱密度 $S(f) = |\hat\theta(f)|^2$。周期运动表现为离散尖峰，混沌运动则呈现宽频连续谱——这是区分周期与混沌最直观的频域判据。

#### (b) 参数空间混沌地图
固定系统几何参数，扫描 $(E, m_1/m_2)$ 二维参数平面。对每个格点计算最大 Lyapunov 指数 $\lambda_{\max}$，用颜色编码映射到二维平面上生成"混沌地图"。这能系统揭示系统从规则运动（$\lambda_{\max} \approx 0$，蓝色）到混沌运动（$\lambda_{\max} > 0$，红色）的转变边界。

#### (c) 倍周期分岔图（阻尼驱动双摆）
对阻尼驱动双摆，以驱动力振幅 $A$ 为分岔参数，在每一 $A$ 值下取 Poincaré 截面上的 $\theta_1$ 值绘制分岔图。预期观察到倍周期分岔序列（周期 $1 \to 2 \to 4 \to$ 混沌），这是混沌动力学的经典特征，也是对 Feigenbaum 常数的间接验证。

## 4. 拟实现的目标 (Objectives & Deliverables)

### 基础目标 (Must-have)
- 实现双摆运动方程的 Euler、RK4、Symplectic Euler、Störmer-Verlet 四种求解器
- 绘制精度对比图（收敛阶 log-log 图，验证各方法理论阶数）
- 绘制能量漂移对比图（长时间积分下的能量守恒性比较）
- 生成双摆轨迹动画与 $\theta_1$-$\theta_2$ 相图
- 初值敏感性演示（微小扰动下的轨迹指数级分离）

### 进阶目标 (Nice-to-have)
- 计算最大 Lyapunov 指数（采用轨道分离法）
- 绘制 Poincaré 截面，区分周期区与混沌区
- FFT 功率谱分析，从频域表征混沌运动
- 参数空间混沌地图（质量比 vs 能量，颜色编码 Lyapunov 指数）
- 拓展至阻尼驱动双摆，绘制倍周期分岔图
- 代码工程：面向对象封装、单元测试、命令行参数接口，确保结果可复现

### 预期交付物
- 完整 Python 代码（RK4 求解器 + 可视化 + 分析）
- 结题论文
- 现场汇报 PPT（含动画演示）

## 5. 时间规划
- **第 16 周**：选题确认与文献调研、运动方程推导；实现 Euler / RK4 / Symplectic Euler / Verlet 四种求解器、面向对象封装、单元测试
- **第 17 周**：精度对比（收敛阶验证）、能量守恒性对比（长时间漂移分析）；轨迹动画与相图可视化；初值敏感性演示；FFT 功率谱分析
- **第 18 周**：进阶目标（Lyapunov 指数 / Poincaré 截面 / 参数空间混沌地图）；阻尼驱动双摆拓展（分岔图）；论文撰写与汇报准备

## 6. 参考文献
1. Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering* (2nd ed.). CRC Press. — 第 5-6 章关于混沌与 Lyapunov 指数
2. Shinbrot, T., Grebogi, C., Wisdom, J., & Yorke, J. A. (1992). Chaos in a double pendulum. *American Journal of Physics*, 60(6), 491–499. — 双摆混沌的经典文献
3. Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007). *Numerical Recipes: The Art of Scientific Computing* (3rd ed.). Cambridge University Press. — 第 16-17 章关于 ODE 求解与 RK4
4. Benettin, G., Galgani, L., Giorgilli, A., & Strelcyn, J. M. (1980). Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems; a method for computing all of them. *Meccanica*, 15(1), 9–20. — Lyapunov 指数计算方法
5. Tél, T., & Gruiz, M. (2006). *Chaotic Dynamics: An Introduction Based on Classical Mechanics*. Cambridge University Press. — 第 3-4 章关于分岔、功率谱与混沌诊断
6. Feigenbaum, M. J. (1978). Quantitative universality for a class of nonlinear transformations. *Journal of Statistical Physics*, 19(1), 25–52. — 倍周期分岔的 Feigenbaum 常数
