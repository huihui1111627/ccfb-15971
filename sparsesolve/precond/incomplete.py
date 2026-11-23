"""不完全分解（零填充）：IC(0) 与 ILU(0)。

约定
----
* **IC(0)**：A ≈ L D Lᵀ，L 为严格下三角（单位对角），D 为对角。
  L 的非零结构恰为 A 的严格下三角结构，D 的对角位置恰为 A 的对角位置，
  即“零填充”。
* **ILU(0)**：A ≈ L U，L 单位下三角（结构 = A 的严格下三角），
  U 上三角含对角（结构 = A 的严格上三角 + 对角）。

分解算法：right-looking IKJ 变体。按行 i 由左至右，用已算完的行 j 更新
行 i 中“行列结构都有”的位置；因为每行列号升序，双重循环用归并指针，
无需查找。分解天然串行（行 i 依赖所有前序行）。

性能：每行的更新量很小，纯 NumPy 逐行内核开销大于收益；实测在
1M 未知数 / 5M 非零元时纯 Python 列表实现约 1.2 秒，而逐行 NumPy 需
数秒。因此这里把数组转成 Python 列表就地更新，最后再转回 NumPy。

数值失败（主元为 0 或过小）会在 ``stats["breakdown"]`` 中报告，并把
主元抬到下限，预条件器仍可应用（保证不产生 NaN/Inf）。
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from ..csr import CSRMatrix
from ..trisolve import LayerStruct, transpose_strict_pattern
from .base import Preconditioner, _nbytes

__all__ = ["IC0Precond", "ILU0Precond"]


class _ZeroFillBase(Preconditioner):
    """零填充分解的公共骨架。"""

    symmetric = False
    factor_kind = "ic0"          # "ic0" / "ilu0"

    def __init__(self, drop_tol: float = 0.0, pivot_floor: float = 1e-14,
                 scale_diag: bool = True):
        super().__init__()
        self.drop_tol = float(drop_tol)
        self.pivot_floor = float(pivot_floor)
        self.scale_diag = bool(scale_diag)
        self.breakdown = False
        self.zero_pivots = 0
        self.min_pivot = 0.0
        self.max_abs_l = 0.0
        self.scale_vec = None
        self.scale_diag_used = False
        self.scale_s = 0.0

    # ------------------------------------------------------------------
    def _extract(self, A: CSRMatrix):
        n = A.n
        if not A.is_square:
            raise ValueError("不完全分解需要方阵")
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(A.indptr))
        cols = A.indices.astype(np.int64)
        lower = cols < rows
        diag = cols == rows
        if int(diag.sum()) != n:
            missing = int(n - diag.sum())
            raise ValueError(f"A 缺对角元（缺 {missing} 个），请先补零对角")
        self.diag_a = A.data[diag].copy()
        lptr = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(rows[lower], minlength=n), out=lptr[1:])
        self.lptr = lptr
        self.lind = cols[lower]
        self.lval0 = A.data[lower].copy()
        if self.factor_kind == "ilu0":
            upper = cols > rows
            uptr = np.zeros(n + 1, np.int64)
            np.cumsum(np.bincount(rows[upper], minlength=n), out=uptr[1:])
            self.uptr = uptr
            self.uind = cols[upper]
            self.uval0 = A.data[upper].copy()

    # ------------------------------------------------------------------
    def build(self, A: CSRMatrix):
        t0 = time.perf_counter()
        n = A.n
        self.n = n
        # IC(0) 采用对称对角缩放 D^{-1/2} A D^{-1/2} 作为默认策略：
        # 位移有限元刚度阵的对角量级可跨越多个数量级，未缩放时 IC(0) 容易
        # 出现负主元而失效；缩放后主元为正的概率大幅提高，且不改变零填充
        # 结构。缩放因子在 apply 中通过 z = D^{-1/2} M̃^{-1} D^{-1/2} r 还原。
        t_scale0 = time.perf_counter()
        if self.factor_kind == "ic0" and self.scale_diag:
            d0 = A.diag
            dmax0 = float(np.max(np.abs(d0))) if d0.size else 1.0
            if np.any(~(d0 > 0)):
                # 对角含非正值，无法做对称缩放；退回原矩阵并记录
                self.scale_diag_used = False
                self.scale_vec = None
                A_use = A
            else:
                s = 1.0 / np.sqrt(d0)
                self.scale_vec = s
                self.scale_diag_used = True
                A_use = _sym_scale(A, s)
        else:
            self.scale_diag_used = False
            self.scale_vec = None
            A_use = A
        self.scale_s = time.perf_counter() - t_scale0
        self._extract(A_use)
        lptr_l = self.lptr.tolist()
        lind_l = self.lind.tolist()
        lv = self.lval0.tolist()
        diag_a = self.diag_a.tolist()
        dmax = max((abs(v) for v in diag_a), default=1.0) or 1.0
        floor = self.pivot_floor * dmax
        tol = self.drop_tol
        if self.factor_kind == "ic0":
            d = self._factor_ic0(n, lptr_l, lind_l, lv, diag_a, floor, tol)
            self.d = d
            self.lval = np.asarray(lv, dtype=np.float64)
            self.lower_layers = LayerStruct.build(self.lptr, self.lind.astype(np.int32), n)
            # Lᵀ 的严格上三角结构；其数值是 L 数值按列重排后的结果
            uptr, uind, src = transpose_strict_pattern(self.lptr, self.lind, n)
            self.upper_layers = LayerStruct.build(uptr, uind, n)
            self.lval_T = self.lval[src]
            self.dinv = _safe_inverse(d, floor)
            mem = _nbytes(self.lval, self.lind, self.d, self.dinv,
                          self.lower_layers.level, self.upper_layers.level)
            nnz = int(self.lind.size * 2 + n)
        else:
            uptr_l = self.uptr.tolist()
            uind_l = self.uind.tolist()
            uv = self.uval0.tolist()
            d = self._factor_ilu0(n, lptr_l, lind_l, lv, uptr_l, uind_l, uv,
                                  diag_a, floor, tol)
            self.d = d
            self.lval = np.asarray(lv, dtype=np.float64)
            self.uval = np.asarray(uv, dtype=np.float64)
            self.lower_layers = LayerStruct.build(self.lptr, self.lind.astype(np.int32), n)
            self.upper_layers = LayerStruct.build(self.uptr, self.uind.astype(np.int32), n)
            mem = _nbytes(self.lval, self.lind, self.uval, self.uind, self.d,
                          self.lower_layers.level, self.upper_layers.level)
            nnz = int(self.lind.size + self.uind.size + n)
        self.stats = {
            "build_s": time.perf_counter() - t0,
            "nnz": nnz,
            "mem_bytes": mem,
            "breakdown": self.breakdown,
            "zero_pivots": self.zero_pivots,
            "min_pivot": self.min_pivot,
            "max_abs_l": self.max_abs_l,
            "lower_levels": self.lower_layers.nlevels,
            "upper_levels": self.upper_layers.nlevels,
            "levels_ok": bool(self.lower_layers.ok and self.upper_layers.ok),
            "drop_tol": self.drop_tol,
            "scale_diag": self.scale_diag_used,
            "scale_s": self.scale_s,
        }
        self._built = True
        return self.stats

    # ------------------------------------------------------------------
    def _factor_ic0(self, n, lptr, lind, lv, diag_a, floor, tol):
        d = [0.0] * n
        zero_piv = 0
        max_l = 0.0
        for i in range(n):
            s = lptr[i]
            e = lptr[i + 1]
            acc = 0.0
            for p in range(s, e):
                j = lind[p]
                v = lv[p]
                if v == 0.0:
                    continue
                dj = d[j]
                if dj == 0.0:
                    lv[p] = 0.0
                    continue
                l = v / dj
                if tol > 0.0 and abs(l) * abs(dj) ** 0.5 < tol:
                    l = 0.0
                lv[p] = l
                if l == 0.0:
                    continue
                al = l if l >= 0 else -l
                if al > max_l:
                    max_l = al
                acc += l * l * dj
                lDj = l * dj
                a = p + 1
                q = lptr[j]
                je = lptr[j + 1]
                while a < e and q < je:
                    ka = lind[a]
                    kq = lind[q]
                    if ka == kq:
                        lv[a] -= lDj * lv[q]
                        a += 1
                        q += 1
                    elif ka < kq:
                        a += 1
                    else:
                        q += 1
            di = diag_a[i] - acc
            if not (abs(di) > floor) or not (di == di):
                zero_piv += 1
                self.breakdown = True
                di = floor
            d[i] = di
        self.zero_pivots = zero_piv
        self.max_abs_l = max_l
        self.min_pivot = float(min(d)) if d else 0.0
        return np.asarray(d, dtype=np.float64)

    def _factor_ilu0(self, n, lptr, lind, lv, uptr, uind, uv, diag_a, floor, tol):
        # ud[j] 存 U 的对角；uv 只存严格上三角
        ud = [0.0] * n
        zero_piv = 0
        max_l = 0.0
        for i in range(n):
            s = lptr[i]
            e = lptr[i + 1]
            s_u = uptr[i]
            e_u = uptr[i + 1]
            for p in range(s, e):
                j = lind[p]
                v = lv[p]
                if v == 0.0:
                    continue
                ujj = ud[j]
                if ujj == 0.0:
                    lv[p] = 0.0
                    continue
                l = v / ujj
                if tol > 0.0 and abs(l) < tol:
                    l = 0.0
                lv[p] = l
                if l == 0.0:
                    continue
                al = l if l >= 0 else -l
                if al > max_l:
                    max_l = al
                a = s_u
                q = uptr[j]
                je = uptr[j + 1]
                while a < e_u and q < je:
                    ka = uind[a]
                    kq = uind[q]
                    if ka == kq:
                        uv[a] -= l * uv[q]
                        a += 1
                        q += 1
                    elif ka < kq:
                        a += 1
                    else:
                        q += 1
            # U_ii = A_ii - sum_{j<i} L_ij U_ji
            di = diag_a[i]
            a = s_u
            for p in range(s, e):
                j = lind[p]
                q = uptr[j]
                je = uptr[j + 1]
                while q < je and uind[q] < i:
                    q += 1
                if q < je and uind[q] == i:
                    di -= lv[p] * uv[q]
            if not (abs(di) > floor) or not (di == di):
                zero_piv += 1
                self.breakdown = True
                di = floor
            ud[i] = di
        self.zero_pivots = zero_piv
        self.max_abs_l = max_l
        self.min_pivot = float(min(ud)) if ud else 0.0
        return np.asarray(ud, dtype=np.float64)

    # ------------------------------------------------------------------
    def apply(self, r):
        if not (self.lower_layers.ok and self.upper_layers.ok):
            return r.copy()
        if self.factor_kind == "ic0":
            # 若做过对称缩放 A -> S A S（S=diag(s)），则
            #   M = S L D L^T S，M^{-1} r = S L^{-T} D^{-1} L^{-1} S r
            if self.scale_diag_used:
                r = self.scale_vec * r
            y = self.lower_layers.solve_lower(self.lval, r)
            y = self.dinv * y
            z = self.upper_layers.solve_upper(self.lval_T, y)
            if self.scale_diag_used:
                z = self.scale_vec * z
            return z
        # ILU(0)：U 含对角，解 (diag(U) + strictU) z = y
        y = self.lower_layers.solve_lower(self.lval, r)
        return self.upper_layers.solve_upper_scaled(self.uval, self.d, y)


def _safe_inverse(d, floor):
    scale = max(float(np.max(np.abs(d))), 1e-300)
    bad = np.abs(d) <= floor
    dd = np.where(bad, 1.0, d)
    out = 1.0 / dd
    out[bad] = 0.0
    return out


def _transpose_layers(lptr, lind, n):
    """构造 Lᵀ 严格上三角结构的层分解（行依赖列号更大的行，反向做）。"""
    uptr, uind, src = transpose_strict_pattern(lptr, lind, n)
    # Lᵀ 的第 k 行包含原行 i>k 的列 k。求解 Lᵀ z = y 时
    # z_i 依赖 z_j (j>i)，因此从层号大的往小的做；层结构用 L 的转置依赖图。
    layers = LayerStruct.build(uptr, uind, n)
    # 数值映射：Lᵀ 结构第 k 行的数值 = lval[src[该行块]]
    layers.upper_src = src
    return layers


class IC0Precond(_ZeroFillBase):
    """IC(0)：A ≈ L D Lᵀ。"""

    name = "ic0"
    kind = "spd"
    symmetric = True
    factor_kind = "ic0"


class ILU0Precond(_ZeroFillBase):
    """ILU(0)：A ≈ L U。"""

    name = "ilu0"
    kind = "general"
    symmetric = False
    factor_kind = "ilu0"


def _sym_scale(A: CSRMatrix, s: np.ndarray) -> CSRMatrix:
    """返回 D^{1/2}... 的具体实现：B = S A S，S = diag(s)。

    只对结构做逐元素乘：B_ij = s_i A_ij s_j。
    """
    rows = np.repeat(np.arange(A.n, dtype=np.int64), A.row_nnz)
    vals = A.data * s[rows] * s[A.indices.astype(np.int64)]
    return CSRMatrix(vals, A.indices, A.indptr, A.n, A.ncols)
