"""sparsesolve —— 纯 Python + NumPy 的稀疏线性方程组迭代求解器。

限制 BLAS/OpenMP 线程数，避免 multiprocessing fork 后线程状态复制导致的死锁。
必须在导入 numpy 之前设置，因此放在包初始化文件的最前面。
"""
from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

__version__ = "0.1.0"

from .csr import CSRMatrix  # noqa: E402
from .problem import Problem  # noqa: E402

__all__ = ["CSRMatrix", "Problem", "__version__"]
