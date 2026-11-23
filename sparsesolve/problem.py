"""线性方程组问题容器。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .csr import CSRMatrix

__all__ = ["Problem"]


@dataclass
class Problem:
    """一个待求解的稀疏线性方程组 A x = b。

    Attributes
    ----------
    A : CSRMatrix
    b : ndarray
    name : str
        唯一名称，用作结果文件与页面中的标识。
    kind : str
        问题类型：``spd_poisson`` / ``spd_elastic`` / ``nonsym`` / ``spd_irregular`` 等。
    x_exact : ndarray, optional
        已知精确解（生成时构造），用于误差统计。
    meta : dict
        生成参数（网格尺寸、对比度、Peclet 数等），供页面展示。
    """

    A: CSRMatrix
    b: np.ndarray
    name: str
    kind: str = "general"
    x_exact: Optional[np.ndarray] = None
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.b = np.ascontiguousarray(self.b, dtype=np.float64)
        if self.b.shape != (self.A.n,):
            raise ValueError(f"b 维度不符: {self.b.shape} vs n={self.A.n}")

    @property
    def n(self) -> int:
        return self.A.n

    @property
    def nnz(self) -> int:
        return self.A.nnz

    @property
    def norm_b(self) -> float:
        return float(np.linalg.norm(self.b))

    def residual(self, x: np.ndarray) -> np.ndarray:
        return self.b - self.A.matvec(x)

    def relative_residual(self, x: np.ndarray) -> float:
        nb = self.norm_b
        r = self.residual(x)
        return float(np.linalg.norm(r) / nb) if nb > 0 else float(np.linalg.norm(r))

    def relative_error(self, x: np.ndarray) -> Optional[float]:
        if self.x_exact is None:
            return None
        den = np.linalg.norm(self.x_exact)
        return float(np.linalg.norm(x - self.x_exact) / den) if den > 0 else None

    def summary(self) -> dict:
        asym, struct_sym = self.A.symmetry_metrics()
        rate, avg, dmin, dmax = self.A.diagonal_dominance()
        return {
            "name": self.name,
            "kind": self.kind,
            "n": self.n,
            "nnz": self.nnz,
            "density": self.nnz / max(1, self.n * self.n),
            "asymmetry": asym,
            "structurally_symmetric": bool(struct_sym),
            "diag_dominant_fraction": rate,
            "avg_dominance": avg,
            "abs_diag_min": dmin,
            "abs_diag_max": dmax,
            "has_exact": self.x_exact is not None,
            "meta": dict(self.meta),
        }
