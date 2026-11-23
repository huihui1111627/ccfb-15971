"""对称逐次超松弛（SSOR / 对称 Gauss-Seidel）预条件。"""
from __future__ import annotations

import time

import numpy as np

from ..csr import CSRMatrix
from ..trisolve import (LayerStruct, lower_csr_pattern, upper_csr_pattern,
                        transpose_strict_pattern)
from .base import Preconditioner, _nbytes

__all__ = ["SSORPrecond"]


class SSORPrecond(Preconditioner):
    """SSOR 预条件（要求 A 对称、对角为正）。

    取 A = D + L + U（L、U 严格三角），
        M = (1/(ω(2-ω))) (D/ω + L) D^{-1} (D/ω + U)。
    应用 M^{-1} r 等价于：
        s = (D + ωL)^{-1} r,   t = D^{-1} s,
        w = (D + ωU)^{-1} (D t),
        z = ω(2-ω) w。
    即两次分层三角求解。
    """

    name = "ssor"
    kind = "spd"
    symmetric = True

    def __init__(self, omega: float = 1.0):
        super().__init__()
        if not (0.0 < omega < 2.0):
            raise ValueError("SSOR 的 ω 必须在 (0, 2) 内")
        self.omega = float(omega)

    def build(self, A: CSRMatrix):
        t0 = time.perf_counter()
        n = A.n
        if not A.is_square:
            raise ValueError("SSOR 需要方阵")
        self.n = n
        d = A.diag.copy()
        scale = float(np.max(np.abs(d))) if d.size else 1.0
        bad = ~(np.abs(d) > 1e-14 * max(scale, 1e-300))
        self.bad_diag = int(bad.sum())
        self.d = np.where(bad, 1.0, d)
        self.d_omega = self.d / self.omega
        w = self.omega
        lptr, lind, lmask = lower_csr_pattern(A.indices, A.indptr, n)
        self.lind = lind
        lval = A.data[lmask].copy()
        self.lower_layers = LayerStruct.build(lptr, lind, n)
        uptr, uind, umask = upper_csr_pattern(A.indices, A.indptr, n)
        self.uind = uind
        uval = A.data[umask].copy()
        self.upper_layers = LayerStruct.build(uptr, uind, n)
        # 求解 (D + ωL) s = r：s = D^{-1/2} 形式不做变换，直接用带对角的层求解。
        # 层求解器内部按行"先减内积再除以对角"：需要 val 为 ωL 的原值。
        self.lval = lval
        self.uval = uval
        mem = _nbytes(self.lind, lval, uind, uval, self.d,
                      self.lower_layers.level, self.upper_layers.level)
        ok = self.lower_layers.ok and self.upper_layers.ok
        self.stats = {
            "build_s": time.perf_counter() - t0,
            "nnz": int(lind.size + uind.size),
            "mem_bytes": mem,
            "omega": w,
            "lower_levels": self.lower_layers.nlevels,
            "upper_levels": self.upper_layers.nlevels,
            "levels_ok": bool(ok),
            "breakdown": bool(self.bad_diag > 0),
            "min_pivot": float(np.min(np.abs(self.d))),
        }
        if not ok:
            self.reason = "三角结构无法分层"
        elif self.bad_diag:
            self.reason = f"{self.bad_diag} 个对角元为 0 或过小"
        self._built = True
        return self.stats

    def apply(self, r):
        """M^{-1} r = ω(2-ω) (D/ω+U)^{-1} D (D/ω+L)^{-1} r。

        M = (1/(ω(2-ω))) (D/ω+L) D^{-1} (D/ω+U)
        => M^{-1} = ω(2-ω) (D/ω+U)^{-1} D (D/ω+L)^{-1}。
        两次分层三角求解。ω=1 时即对称 Gauss-Seidel。
        """
        w = self.omega
        s = self.lower_layers.solve_lower_scaled(self.lval, self.d_omega, r)
        t = self.d * s                      # 中间的 D
        z = self.upper_layers.solve_upper_scaled(self.uval, self.d_omega, t)
        return z * (w * (2.0 - w))
