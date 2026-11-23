# 稀疏迭代求解器 · 验收报告

共 13 个测试问题。

## 1. 小规模问题 vs 稠密直接法

| 问题 | n | 稠密交叉校验 | 方案 | 收敛 | 迭代 | 相对残差 | 相对稠密解误差 |
|---|---|---|---|---|---|---|---|
| poisson2d_small | 1600 | — | cg+none | ✓ | 129 | 8.8e-11 | 3.8e-10 |
| poisson2d_small | 1600 | — | cg+jacobi | ✓ | 129 | 8.8e-11 | 3.8e-10 |
| poisson2d_small | 1600 | — | cg+ic0 | ✓ | 44 | 9.8e-11 | 3.6e-10 |
| poisson2d_small | 1600 | — | cg+ssor | ✓ | 52 | 7.0e-11 | 4.3e-10 |
| poisson2d_small | 1600 | — | cg+ic0+RCM | ✓ | 45 | 4.5e-11 | 1.7e-10 |
| elastic_1e8 | 7200 | — | cg+none | ✓ | 8179 | 9.5e-11 | 0.02 |
| elastic_1e8 | 7200 | — | cg+jacobi | ✓ | 452 | 9.9e-11 | 2.2e-03 |
| elastic_1e8 | 7200 | — | cg+ic0 | ✓ | 174 | 7.7e-11 | 2.2e-03 |
| elastic_1e8 | 7200 | — | cg+ssor | ✓ | 172 | 9.1e-11 | 2.2e-03 |
| elastic_1e8 | 7200 | — | cg+ic0+RCM | ✓ | 176 | 9.0e-11 | 2.9e-03 |
| conv_pe2 | 10000 | — | bicgstab+none | ✓ | 157 | 5.2e-11 | 2.6e-09 |
| conv_pe2 | 10000 | — | bicgstab+ilu0 | ✓ | 29 | 5.3e-12 | 1.3e-10 |
| conv_pe2 | 10000 | — | gmres+none | ✓ | 655 | 8.7e-11 | 1.1e-09 |
| conv_pe2 | 10000 | — | gmres+ilu0 | ✓ | 42 | 3.7e-11 | 1.4e-10 |
| conv_pe2 | 10000 | — | gmres+ilu0+RCM | ✓ | 42 | 3.7e-11 | 1.4e-10 |
| conv_pe20 | 10000 | — | bicgstab+none | ✓ | 153 | 1.5e-11 | 1.9e-10 |
| conv_pe20 | 10000 | — | bicgstab+ilu0 | ✓ | 10 | 8.1e-12 | 1.3e-10 |
| conv_pe20 | 10000 | — | gmres+none | ✓ | 620 | 9.9e-11 | 1.1e-09 |
| conv_pe20 | 10000 | — | gmres+ilu0 | ✓ | 14 | 6.7e-11 | 2.8e-10 |
| conv_pe20 | 10000 | — | gmres+ilu0+RCM | ✓ | 14 | 6.7e-11 | 2.8e-10 |

## 2. 大规模问题残差判据

| 问题 | n | 方案 | 收敛 | 迭代 | 相对残差 | 满足判据 | 耗时 s | 内存 MB |
|---|---|---|---|---|---|---|---|---|
| poisson2d_med | 40000 | cg+ic0 | ✓ | 134 | 8.8e-09 | ✓ | 2.81 | 4.7 |
| poisson2d_large | 1000000 | cg+ic0 | ✓ | 457 | 9.4e-09 | ✓ | 71.91 | 118.2 |
| poisson3d_med | 64000 | cg+ic0 | ✓ | 41 | 6.4e-09 | ✓ | 0.77 | 9.8 |
| poisson3d_large | 1000000 | cg+ic0 | ✓ | 78 | 9.3e-09 | ✓ | 17.53 | 155.3 |
| irregular_med | 14400 | cg+ssor | ✓ | 170 | 8.0e-09 | ✓ | 0.42 | 2.3 |
| irregular_large | 360000 | cg+ssor | ✓ | 1066 | 9.9e-09 | ✓ | 35.06 | 61.0 |
| elastic_1e6 | 12800 | cg+ic0 | ✓ | 184 | 9.8e-09 | ✓ | 0.66 | 4.5 |
| elastic_1e8 | 7200 | cg+ic0 | ✓ | 55 | 7.5e-09 | ✓ | 0.18 | 2.5 |
| layered_1e4 | 40000 | cg+ic0 | ✓ | 176 | 9.9e-09 | ✓ | 0.72 | 4.7 |
| conv_pe2 | 10000 | gmres+ilu0 | ✓ | 37 | 7.7e-09 | ✓ | 0.08 | 1.4 |
| conv_pe20 | 10000 | gmres+ilu0 | ✓ | 12 | 5.3e-09 | ✓ | 0.05 | 1.4 |
| conv_pe20_large | 250000 | gmres+ilu0 | ✓ | 27 | 6.9e-09 | ✓ | 1.37 | 35.2 |

## 3. 自动选择 vs 枚举最快方案

| 问题 | 自动选择 | 自动耗时 s | 枚举最快 | 最快耗时 s | 迭代差 | 耗时差 | 候选数 |
|---|---|---|---|---|---|---|---|
| poisson2d_small | cg+none | 4.42e-03 | cg+none | 4.52e-03 | +0% | -2% | 5 |
| poisson2d_med | cg+none | 0.287 | cg+jacobi | 0.280 | +0% | +2% | 5 |
| poisson2d_large | cg+jacobi | 25.882 | cg+jacobi | 27.029 | +0% | -4% | 2 (受限) |
| poisson3d_med | cg+none | 0.132 | cg+jacobi | 0.129 | +0% | +2% | 5 |
| poisson3d_large | cg+none | 4.416 | cg+none | 4.372 | +0% | +1% | 2 (受限) |
| irregular_med | cg+jacobi | 0.107 | cg+jacobi | 0.107 | +0% | +0% | 5 |
| irregular_large | cg+jacobi | 19.598 | cg+jacobi | 19.658 | +0% | -0% | 2 (受限) |
| elastic_1e6 | cg+jacobi | 0.080 | cg+jacobi | 0.080 | +0% | +0% | 5 |
| elastic_1e8 | cg+jacobi | 0.033 | cg+jacobi | 0.032 | +0% | +1% | 5 |
| layered_1e4 | cg+jacobi | 0.339 | cg+jacobi | 0.335 | +0% | +1% | 5 |
| conv_pe2 | gmres+ilu0 | 0.071 | bicgstab+none | 0.048 | -76% | +47% | 7 |
| conv_pe20 | gmres+ilu0 | 0.038 | gmres+ilu0+RCM | 0.037 | +0% | +4% | 7 |
| conv_pe20_large | gmres+ilu0+RCM | 1.043 | gmres+ilu0+RCM | 1.039 | +0% | +0% | 7 |

## 4. 不收敛报告

| 场景 | 正确报告 | 收敛标志 | 原因 |
|---|---|---|---|
| maxiter=3, tol=1e-12 | ✓ | False | maxiter |
| gmres+ilu0 maxiter=5 tol=1e-15 | ✓ | False | maxiter |
| singular (zero diagonal) | ✓ | False | pAp_nonpositive |

## 5. 重排序（RCM）效果

| 问题 | 带宽 前后 | 轮廓 前后 | IC0 迭代 未重排→重排 |
|---|---|---|---|
| poisson2d_small | 40 → 40 | 62439 → 43420 | 34 → 36 |
| poisson2d_med | 200 → 200 | 7960199 → 5353100 | 134 → 134 |
| poisson2d_large | 1000 → 1000 | 999000999 → 667165500 | 400 → 400 |
| poisson3d_med | 1600 → 1220 | 99902439 → 57451498 | 41 → 41 |
| poisson3d_large | 10000 → 7550 | 9900990099 → 5545171995 | 78 → 83 |
| irregular_med | 14257 → 399 | 73867577 → 2054635 | 1 → 1 |
| irregular_large | 359480 → 1943 | 47949690418 → 268708371 | 3 → 1 |
| elastic_1e6 | 163 → 316 | 2015525 → 2599249 | 184 → 75 |
| elastic_1e8 | 123 → 236 | 845645 → 1078629 | 55 → 56 |
| layered_1e4 | 200 → 200 | 7960199 → 5353100 | 176 → 114 |

