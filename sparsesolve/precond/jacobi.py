"""对角缩放（Jacobi）预条件。"""
from __future__ import annotations

import time

import numpy as np

from ..csr import CSRMatrix
from .base import Preconditioner, _nbytes

__all__ = ["JacobiPrecond"]


class JacobiPrecond(Preconditioner):
    """M = diag(A)，M^{-1} r = r / diag。

    对角元为 0 或过小（相对最大对角）时以 1 兜底并计数，保证不产生 Inf/NaN；
    这类退化会体现在 ``stats["bad_diag"]`` 中，供自动选择排除该方案。
    """

    name = "jacobi"
    kind = "general"
    symmetric = True

    def __init__(self, floor: float = 1e-14):
        super().__init__()
        self.floor = float(floor)

    def build(self, A: CSRMatrix):
        t0 = time.perf_counter()
        d = A.diag.copy()
        scale = float(np.max(np.abs(d))) if d.size else 1.0
        bad = ~(np.abs(d) > self.floor * max(scale, 1e-300))
        self.bad_diag = int(bad.sum())
        self.zero_diag = int(np.sum(d == 0.0))
        self.negative_diag = int(np.sum(d < 0.0))
        dd = np.where(bad, 1.0, d)
        self.invd = 1.0 / dd
        self.stats = {
            "build_s": time.perf_counter() - t0,
            "nnz": int(A.n),
            "mem_bytes": _nbytes(self.invd),
            "zero_diag": self.zero_diag,
            "bad_diag": self.bad_diag,
            "negative_diag": self.negative_diag,
            "breakdown": bool(self.bad_diag > 0),
            "min_pivot": float(np.min(np.abs(dd))),
        }
        if self.bad_diag:
            self.reason = f"{self.bad_diag} 个对角元为 0 或过小"
        self._built = True
        return self.stats

    def apply(self, r):
        return self.invd * r
