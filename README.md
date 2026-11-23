# 稀疏线性方程组迭代求解器（纯 Python + NumPy）

面向结构仿真 / 热分析中"网格一细未知数到几百万、直接分解内存几十 GB 撑不住"的
场景：一组**纯 NumPy 实现、不依赖任何现成稀疏矩阵库**的 Krylov 迭代求解器，
覆盖 CG / BiCGSTAB / GMRES(m)，配合对角缩放、不完全分解（IC(0)/ILU(0)）、
SSOR 预条件，逆卡氏重排，以及基于矩阵特征 + 小规模试跑的**自动方案选择**。

## 快速开始

```bash
# 1) 生成页面数据（小规模问题会与稠密直接解对比）
python3 scripts/run.py export            # -> web/data/

# 2) 跑完整验收，输出 reports/acceptance.json
python3 scripts/run.py acceptance

# 3) 把验收结果转成可读的 Markdown
python3 scripts/report.py reports/acceptance.json > reports/REPORT.md

# 4) 看页面
python3 scripts/run.py serve             # http://127.0.0.1:8765/index.html

# 5) 单元测试 / 快速冒烟
python3 -m pytest tests/ -q
python3 scripts/smoke.py

# 6) 单问题基准
python3 scripts/run.py bench --key poisson2d_large
```

依赖：Python ≥ 3.8、NumPy ≥ 1.20。**不需要 scipy**（测试中可选使用，仅作参考对照）。

## 模块

| 文件 | 内容 |
|---|---|
| `sparsesolve/csr.py` | CSR 存储、`reduceat` 矩阵向量乘、转置、结构指标 |
| `sparsesolve/parallel.py` | 按行分块 + 共享内存的多进程 SpMV |
| `sparsesolve/trisolve.py` | 层调度（level scheduling）三角求解 |
| `sparsesolve/precond/` | Jacobi / IC(0) / ILU(0) / SSOR |
| `sparsesolve/krylov.py` | CG、BiCGSTAB、GMRES(m)（右预条件） |
| `sparsesolve/reorder.py` | 逆卡氏重排、带宽 / 轮廓统计 |
| `sparsesolve/generators.py` | 工程型测试矩阵生成器 |
| `sparsesolve/autoselect.py` | 自动选择求解器 + 预条件 |
| `sparsesolve/acceptance.py` | 验收测试与报告 |
| `sparsesolve/export_web.py` | 页面数据导出 |

## 关键技术点

**CSR 矩阵向量乘** 用 `np.add.reduceat` 对每行做分段求和，避免 Python 逐行
循环；2D 1M 未知数时一次 SpMV 约 18 ms。

**多进程并行 SpMV**：按行分块，矩阵与向量放在 POSIX 共享内存中，工作进程用
`np.ndarray(buffer=...)` 零拷贝映射。8 进程在 1M 未知数上约 2.1× 加速（受
Python 进程间同步开销限制）。数值与串行完全一致。

**三角求解**：把稀疏三角结构的行按依赖关系分层（Kahn 拓扑排序），同层内
各行互不依赖，因此每层所有行的行内点积合并成一次 gather + `reduceat`。
二维 Poisson 图上层数约 O(√n)，求解一次约 23 ms（n=1M），而不是逐行循环的
数秒。IC(0)/ILU(0)/SSOR 的每次应用都建立在这之上。

**不完全分解**：right-looking IKJ，纯 Python 列表原地更新（每行更新量小，
NumPy 逐行内核开销反而更大）。1M 未知数 / 5M 非零元约 3–5 s 建好 IC(0)（含分层
结构预计算），此后每次应用约 50 ms（约 3 次 SpMV 的开销）。
IC(0) 默认做对称对角缩放 D^{-1/2}AD^{-1/2}，显著降低非结构网格上出现
负主元而失效的概率。

**GMRES 用右预条件**：解 `A M⁻¹ y = b, x = M⁻¹ y`，这样内部残差估计与
真实残差一致，停止判据可靠（左预条件会让估计值失真，实测会导致"假收敛"）。
修正 Gram-Schmidt + 二次正交化。

**逆卡氏重排**在验收里呈现了细致的结果：规则网格的带宽本已最优（RCM 不再改变
带宽，只压缩约 1/3 轮廓）；不规则网格带宽可从 35 万降到约 1900（185 倍）；
而弹性体问题 RCM 反而增大带宽，却把 IC(0) 迭代从 184 降到 75——因为 RCM 按
拓扑连通性聚类，把高刚度区域收拢到相邻编号。**是否值得重排必须实测**，
不能只看带宽，这正是自动选择对不完全分解额外试跑 RCM 版本的原因。

**收敛判据**统一为真实相对残差 `‖b − Ax‖/‖b‖ ≤ tol`，每 `check_every` 步
用一次真实残差复核，其余迭代用递推残差做停止估计，兼顾可信性与开销。
不收敛时返回 `converged=False` 并给出明确原因（`maxiter` / `breakdown` /
`nonfinite` / `precond_not_spd` 等）。

## 测试矩阵（生成器）

| key | 类型 | 特征 |
|---|---|---|
| `poisson2d_*` / `poisson3d_*` | 规则网格 Poisson（5/7 点） | SPD，条件数 O(h⁻²) |
| `irregular_*` | 扰动三角网格 P1 有限元 | SPD，节点编号随机、带宽大 |
| `elastic_1e6` / `elastic_1e8` | 平面应力弹性体，刚度比 1e6/1e8 | SPD，条件数 10⁹–10¹¹ |
| `layered_1e4` | 层状各向异性热传导 | SPD，导热系数差 10⁴ 倍 |
| `conv_pe2` / `conv_pe20` | 对流-扩散（迎风） | 非对称，Pe 控制失对称程度 |

规模从 1.6k 到 1M 未知数。右端项由已知解生成（`b = A x*`），可精确统计误差。

## 自动选择策略

**阶段一（静态筛选）**：看非对称度 `‖A−Aᵀ‖_F/‖A‖_F`、结构对称性、对角
正负与占优率。对称 + 对角全正 → CG；非对称 → BiCGSTAB/GMRES；对角含零/负
→ 排除 CG。

**阶段二（试跑打分）**：对每个候选跑少量步数，用“平均每步 log10 残差下降”
刻画收敛速度，外推到达容差所需步数；再乘以试跑实测的**每步耗时**（这是最
可靠的每步成本，已含 SpMV、预条件应用、正交化等全部开销），加上按规模放大
的预条件建立耗时，得到**预测总耗时**并据此排序。对不完全分解额外尝试 RCM
重排版本。大规模问题（n > 12000）先在 RCM 顺序的连通块子问题（约 1.2 万未知数）
上试跑，结论再拿去执行——相当于工程上“先在小模型上调参”；对选中的不完全
分解还会在完整矩阵上做一次主元健康检查，退化则降级到 SSOR/Jacobi。

页面会展示每个候选的试跑结果（下降率、终点残差、评分）和最终决策理由，
方便工程师理解“为什么选这个”。

## 验收

`python3 scripts/run.py acceptance` 生成 `reports/acceptance.json`，覆盖：

1. 小规模问题（n ≤ 12000）与 `numpy.linalg.solve` 的对比误差，n ≤ 1500 时另与
   自带的纯 Python 稠密高斯消元交叉验证；
2. 大规模问题的最终相对残差是否满足判据；
3. 各预条件的建立耗时、内存、迭代次数、总耗时；
4. 自动选择与枚举最快方案的迭代数 / 耗时差距；
5. 不收敛场景能否正确报告（迭代上限、极紧容差、奇异矩阵）；
6. RCM 前后的带宽、轮廓与 IC(0) 迭代数变化。

## 已知限制

* 纯 NumPy 的性能天花板：SpMV 每非零元约 3–4 ns，比编译型稀疏内核慢一个
  数量级。多进程并行能补一部分，但进程间同步限制在 2× 左右。
* **纯 Python 的分层三角求解比一次 SpMV 慢数倍**（实测 IC(0)/SSOR 每步应用
  约是 SpMV 的 3–7 倍），因此在中等问题上 IC(0) 虽然把迭代数降 3–5 倍，
  总耗时未必最低。自动选择因此以“预测总耗时”而非“最少迭代”为目标——
  这也是为什么页面上会看到某些问题自动选了 Jacobi/无预条件，而不是迭代最少的 IC(0)。
  有编译型内核（如 C/C++ 的稀疏三角求解）时，IC(0) 的优势会明显得多。
* IC(0)/ILU(0) 对强非对称矩阵（如 Pe=20 的对流-扩散）可能退化，自动选择
  会退回到更稳的 GMRES + 无预条件或 Jacobi，并在完整矩阵上做兜底检查。
* 大规模问题的 RCM 与不完全分解仍需 O(nnz) 的临时内存。
* 自动选择的探测计时在毫秒级问题上有抖动；对“none / Jacobi 在常数对角
  Poisson 上等价”这类真实存在的情况，二者排序可能互换，但差距可忽略。
* 对条件数极大（如 1e11）的问题，小相对残差**不等于**小解误差（误差可放大
  cond·rel_residual）。验收里 `elastic_1e8` 用无预条件时残差 1e-11 但解误差
  仍约 1e-2——这是问题本身的病态，不是求解器的错误；换用 Jacobi/IC(0) 可把
  误差降低一个数量级。页面同时给出“对精确解误差”正是为了暴露这一点。
