"""预条件器基类与工厂。

接口::

    P = make_preconditioner("ic0")
    P.build(A)      # 返回统计字典，含建立耗时、非零元数、内存字节数
    z = P.apply(r)  # z ≈ M^{-1} r

预条件器只会用到 ``CSRMatrix`` 的公开属性与 ``trisolve.LayerStruct``。
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..csr import CSRMatrix

__all__ = ["Preconditioner", "IdentityPrecond", "make_preconditioner",
           "PRECONDITIONER_NAMES", "preconditioner_display_name"]


def _nbytes(*arrays) -> int:
    return int(sum(a.nbytes for a in arrays if a is not None))


class Preconditioner:
    name = "base"
    kind = "general"          # "spd" / "general"
    symmetric = False          # M 是否对称正定（CG 需要）

    def __init__(self):
        self.stats: Dict = {}
        self._built = False
        self.reason = ""

    def build(self, A: CSRMatrix) -> Dict:
        raise NotImplementedError

    def apply(self, r: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> Dict:
        return {"name": self.name, "kind": self.kind, **self.stats}

    @property
    def mem_bytes(self) -> int:
        return int(self.stats.get("mem_bytes", 0))

    @property
    def available(self) -> bool:
        return self._built and not self.reason


class IdentityPrecond(Preconditioner):
    name = "none"
    kind = "general"
    symmetric = True

    def build(self, A):
        self.stats = {"build_s": 0.0, "nnz": 0, "mem_bytes": 0, "breakdown": False}
        self._built = True
        return self.stats

    def apply(self, r):
        return r.copy()


def make_preconditioner(name: str, **kw) -> Preconditioner:
    name = name.lower()
    if name in ("none", "identity"):
        return IdentityPrecond()
    if name in ("jacobi", "diag", "diagonal"):
        from .jacobi import JacobiPrecond
        return JacobiPrecond(**kw)
    if name in ("ssor", "sgs"):
        from .ssor import SSORPrecond
        return SSORPrecond(**kw)
    if name in ("ic0", "ic", "ichol0"):
        from .incomplete import IC0Precond
        return IC0Precond(**kw)
    if name in ("ilu0", "ilu"):
        from .incomplete import ILU0Precond
        return ILU0Precond(**kw)
    raise ValueError(f"未知预条件: {name}")


PRECONDITIONER_NAMES = ("none", "jacobi", "ic0", "ilu0", "ssor")

_DISPLAY = {
    "none": "无预条件",
    "jacobi": "对角缩放 (Jacobi)",
    "ic0": "不完全 Cholesky IC(0)",
    "ilu0": "不完全 LU ILU(0)",
    "ssor": "对称 Gauss-Seidel / SSOR (ω=1)",
}


def preconditioner_display_name(name: str) -> str:
    return _DISPLAY.get(name, name)
