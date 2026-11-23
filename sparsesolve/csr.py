"""压缩行（CSR）稀疏矩阵：纯 NumPy 实现，不依赖任何稀疏矩阵库。

约定
----
* ``indptr`` 长度为 ``n+1``，int64；``indices`` 为 int32（支持 n < 2^31）；
* 每一行的列下标严格升序，行内无重复；
* 允许零长度行。

矩阵向量乘用 ``np.add.reduceat`` 做行内分段求和，避免 Python 逐行循环。

注意：macOS Accelerate 后端下 ``np.matmul`` 处理大向量点积会发出伪警告
（divide by zero / overflow），而 ``np.dot`` 不会，因此数值内核统一用
``np.dot``。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["CSRMatrix", "from_coo", "from_dense", "coo_to_csr"]


def _validate(data, indices, indptr, n, ncols):
    data = np.ascontiguousarray(data, dtype=np.float64)
    if indices.dtype != np.int32:
        if indices.size and int(indices.max()) >= 2**31 - 1:
            raise ValueError("列下标超出 int32 范围")
        indices = indices.astype(np.int32)
    else:
        indices = np.ascontiguousarray(indices)
    indptr = np.ascontiguousarray(indptr, dtype=np.int64)
    nnz = data.size
    if indices.size != nnz or indptr.size != n + 1 or indptr[-1] != nnz:
        raise ValueError("CSR 数组长度不一致")
    if ncols is None:
        ncols = n
    if nnz and (indices.min() < 0 or indices.max() >= ncols):
        raise ValueError("列下标越界")
    if np.any(np.diff(indptr) < 0):
        raise ValueError("indptr 必须单调不减")
    return data, indices, indptr, int(n), int(ncols)


class CSRMatrix:
    """不可变 CSR 稀疏矩阵。"""

    __slots__ = ("data", "indices", "indptr", "n", "ncols",
                 "_csc", "_diag", "_row_nnz", "_sorted")

    def __init__(self, data, indices, indptr, n: int, ncols: Optional[int] = None):
        self.data, self.indices, self.indptr, self.n, self.ncols = _validate(
            data, indices, indptr, int(n), ncols)
        self._csc = None
        self._diag = None
        self._row_nnz = None
        self._sorted = True

    # ------------------------------------------------------------ 基本属性
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n, self.ncols)

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def row_nnz(self) -> np.ndarray:
        if self._row_nnz is None:
            self._row_nnz = np.diff(self.indptr)
        return self._row_nnz

    @property
    def is_square(self) -> bool:
        return self.n == self.ncols

    def row(self, i: int):
        s, e = self.indptr[i], self.indptr[i + 1]
        return self.indices[s:e], self.data[s:e]

    @property
    def diag(self) -> np.ndarray:
        """主对角线（缺失的对角元记 0）。"""
        if self._diag is None:
            d = np.zeros(self.n)
            rows = np.repeat(np.arange(self.n), self.row_nnz)
            hit = self.indices == rows
            d[rows[hit]] = self.data[hit]
            self._diag = d
        return self._diag

    def transpose(self) -> "CSRMatrix":
        """返回转置（CSR 表示），结果缓存。"""
        data, indices, indptr, _ = self.to_csc_arrays()
        return CSRMatrix(data, indices, indptr, self.ncols, self.n)

    # ------------------------------------------------------------ CSC 视图
    def to_csc_arrays(self):
        """返回以本矩阵的转置为 CSR 的 (data, indices, indptr)。

        即 CSR(T)：行是原矩阵的列。用于按列访问的算法（IC 分解、行松弛）。
        """
        if self._csc is None:
            n, nnz = self.n, self.nnz
            order = np.argsort(self.indices.astype(np.int64), kind="stable")
            counts = np.bincount(self.indices.astype(np.int64), minlength=self.ncols)
            cptr = np.zeros(self.ncols + 1, np.int64)
            np.cumsum(counts, out=cptr[1:])
            # order 按列分组；组内保持原行序（行号升序，因为 CSR 行遍历顺序）
            self._csc = (self.data[order].copy(), None, cptr, order)
        return _csc_materialize(self)

    # ------------------------------------------------------------ 基本运算
    def matvec(self, x: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """y = A x，行内 reduceat 累加。"""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.ncols,):
            raise ValueError(f"matvec 维度不符: A={self.shape}, x={x.shape}")
        y = np.empty(self.n) if out is None else out
        self._matvec_inplace(x, y)
        return y

    def _matvec_inplace(self, x, y):
        v = self.data * x[self.indices]
        ptr = self.indptr
        cnt = np.diff(ptr)
        nz = cnt > 0
        y.fill(0.0)
        if not nz.any():
            return
        # 非空行使用 reduceat；空行保持 0
        starts = ptr[:-1][nz]
        acc = np.add.reduceat(v, starts)
        y[nz] = acc

    def rmatvec(self, x: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """y = A^T x。"""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.n,):
            raise ValueError("rmatvec 维度不符")
        data, indices, indptr, _ = self.to_csc_arrays()
        y = np.empty(self.ncols) if out is None else out
        v = data * x[indices]
        cnt = np.diff(indptr)
        nz = cnt > 0
        y.fill(0.0)
        if nz.any():
            y[nz] = np.add.reduceat(v, indptr[:-1][nz])
        return y

    def matvec_block(self, x, r0: int, r1: int, out=None) -> np.ndarray:
        """只计算 [r0, r1) 行，用于按行分块并行。"""
        ptr = self.indptr
        s, e = ptr[r0], ptr[r1]
        v = self.data[s:e] * x[self.indices[s:e]]
        cnt = np.diff(ptr[r0:r1 + 1])
        nz = cnt > 0
        if out is None:
            out = np.zeros(r1 - r0)
        else:
            out.fill(0.0)
        if nz.any():
            out[nz] = np.add.reduceat(v, ptr[r0:r1][nz] - s)
        return out

    def scale(self, d: np.ndarray) -> "CSRMatrix":
        """返回行缩放矩阵 diag(d) @ A。"""
        rows = np.repeat(np.arange(self.n), self.row_nnz)
        return CSRMatrix(self.data * d[rows], self.indices, self.indptr,
                         self.n, self.ncols)

    def copy(self) -> "CSRMatrix":
        return CSRMatrix(self.data.copy(), self.indices.copy(), self.indptr.copy(),
                         self.n, self.ncols)

    def astype(self, dtype) -> "CSRMatrix":
        return CSRMatrix(self.data.astype(dtype), self.indices, self.indptr,
                         self.n, self.ncols)

    def to_dense(self) -> np.ndarray:
        D = np.zeros(self.shape)
        rows = np.repeat(np.arange(self.n), self.row_nnz)
        D[rows, self.indices.astype(np.int64)] = self.data
        return D

    def symmetry_metrics(self):
        """返回 (相对 Frobenius 非对称度, 结构对称?)。

        相对非对称度 = ||A - A^T||_F / ||A||_F，无需显式做差。
        """
        if not self.is_square:
            return 1.0, False
        rows = np.repeat(np.arange(self.n), self.row_nnz).astype(np.int32)
        # 在 (col, row) 处找 A
        key_t = rows.astype(np.int64) * self.n + self.indices.astype(np.int64)
        key_a = self.indices.astype(np.int64) * self.n + rows.astype(np.int64)
        # 结构对称：每个 (i,j) 都有 (j,i)
        uniq_a = np.unique(key_a)
        uniq_t = np.unique(key_t)
        struct_sym = uniq_a.size == uniq_t.size and np.array_equal(uniq_a, uniq_t)
        if struct_sym:
            order = np.argsort(key_t, kind="stable")
            # 对 A 中的每个 (i,j)，取 A^T 中的同位置值 A[j,i]
            kt_sorted = key_t[order]
            pos = np.searchsorted(kt_sorted, key_a)
            pos = np.clip(pos, 0, kt_sorted.size - 1)
            ok = kt_sorted[pos] == key_a
            at = self.data[order][pos[ok]]
            aa = self.data[ok]
            diff = aa - at
            nrm = np.linalg.norm(self.data)
            asym = float(np.linalg.norm(diff) / nrm) if nrm > 0 else 0.0
            return asym, True
        nrm = np.linalg.norm(self.data)
        # 无法逐项对齐，给一个上界估计
        return 1.0, False

    def diagonal_dominance(self):
        """返回 (行对角占优率, 平均占优比, 最小/最大 |diag|)。

        行 i 的占优比 rho_i = (|a_ii| - sum_{j!=i}|a_ij|) / sum_{j!=i}|a_ij|，
        sum 为 0 时记 +inf（该行仅对角）。占优率 = ratio > 0 的行占比。
        """
        absrow = np.abs(self.data)
        sumrow = np.add.reduceat(absrow, self.indptr[:-1]) if self.nnz else np.zeros(self.n)
        rows = np.repeat(np.arange(self.n), self.row_nnz)
        hit = self.indices == rows
        d = np.zeros(self.n)
        d[rows[hit]] = absrow[hit]
        off = sumrow - d
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = np.where(off > 0, (d - off) / off, np.where(d > 0, np.inf, -1.0))
        rate = float(np.mean(rho > 0))
        finite = rho[np.isfinite(rho)]
        avg = float(finite.mean()) if finite.size else float("inf")
        return rate, avg, float(d.min()), float(d.max())

    def __repr__(self):
        return (f"CSRMatrix(n={self.n}, ncols={self.ncols}, nnz={self.nnz}, "
                f"density={self.nnz / max(1, self.n * self.ncols):.4g})")


def _csc_materialize(A: CSRMatrix):
    cached = A._csc
    data, indices, indptr, order = cached
    if indices is None:
        rows = np.repeat(np.arange(A.n, dtype=np.int32), A.row_nnz)
        indices = rows[order].copy()
        A._csc = (data, indices, indptr, order)
    else:
        data = cached[0]
    return data, indices, indptr, order


def coo_to_csr(rows, cols, vals, n: int, ncols: Optional[int] = None,
               duplicate: str = "sum") -> CSRMatrix:
    """COO -> CSR。

    duplicate="sum" 时重复项求和（有限元组装常见），"last" 时后者覆盖。
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    if rows.size != cols.size or rows.size != vals.size:
        raise ValueError("COO 三个数组长度不一致")
    if duplicate not in ("sum", "last"):
        raise ValueError("duplicate 只支持 'sum' 或 'last'")
    # 复合键排序：key = row * ncols + col。当 n*ncols < 2^53 时 int64 无精度损失；
    # 这比 lexsort 快 3~5 倍（本机 6.5M 条目 lexsort 8.6s vs 复合键 0.6s）。
    # 需要稳定排序：duplicate="last" 依赖原始顺序，因此用稳定排序。
    if ncols is None:
        ncols = int(cols.max()) + 1 if cols.size else 1
    if n * ncols < (1 << 53):
        key = rows * ncols + cols
        kind = "stable" if duplicate == "last" else "quicksort"
        order = np.argsort(key, kind=kind)
    else:
        order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    vals = vals[order]
    if rows.size:
        same = np.empty(rows.size, bool)
        same[0] = False
        same[1:] = (rows[1:] == rows[:-1]) & (cols[1:] == cols[:-1])
        if duplicate == "sum":
            if same.any():
                grp = np.cumsum(~same) - 1
                vals = np.bincount(grp, weights=vals, minlength=grp[-1] + 1)
                keep = ~same
                rows = rows[keep]
                cols = cols[keep]
        else:
            keep = np.empty(rows.size, bool)
            keep[-1] = True
            keep[:-1] = ~same[1:]
            rows, cols, vals = rows[keep], cols[keep], vals[keep]
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    return CSRMatrix(vals, cols.astype(np.int32), indptr, n, ncols)


def from_coo(rows, cols, vals, n, ncols=None, duplicate="sum") -> CSRMatrix:
    return coo_to_csr(rows, cols, vals, n, ncols, duplicate)


def from_dense(D: np.ndarray) -> CSRMatrix:
    D = np.asarray(D, dtype=np.float64)
    n, m = D.shape
    rows, cols = np.nonzero(D)
    return CSRMatrix(D[rows, cols], cols.astype(np.int32), _ptr_from_counts(rows, n),
                     n, m)


def _ptr_from_counts(rows, n):
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    return indptr
