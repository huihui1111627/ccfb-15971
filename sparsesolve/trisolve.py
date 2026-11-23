"""三角求解：层调度（level scheduling）+ 每层内向量化。

对稀疏严格下（上）三角结构解三角方程组。按依赖关系把行划分成层（DAG 的
最长路径分层），同层内各行互不依赖，因此可以把一层所有行的行内点积合并成
一次 gather + reduceat + scatter，避免 Python 的逐行循环。二维 Poisson 图
上层数约为 O(sqrt(n))，远小于 n。

每层的切片、行号、段起点在构造时预计算，求解时每层只有 2 次向量化内核调用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

__all__ = [
    "LayerStruct", "lower_csr_pattern", "upper_csr_pattern",
    "transpose_strict_pattern",
]


def lower_csr_pattern(indices: np.ndarray, indptr: np.ndarray, n: int):
    """从 A 的 CSR 提取严格下三角结构，返回 (lptr, lind, mask)。"""
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    mask = indices.astype(np.int64) < rows
    cnt = np.bincount(rows[mask], minlength=n)
    lptr = np.zeros(n + 1, np.int64)
    np.cumsum(cnt, out=lptr[1:])
    return lptr, indices[mask], mask


def upper_csr_pattern(indices: np.ndarray, indptr: np.ndarray, n: int):
    """从 A 的 CSR 提取严格上三角结构，返回 (uptr, uind, mask)。"""
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    mask = indices.astype(np.int64) > rows
    cnt = np.bincount(rows[mask], minlength=n)
    uptr = np.zeros(n + 1, np.int64)
    np.cumsum(cnt, out=uptr[1:])
    return uptr, indices[mask], mask


def transpose_strict_pattern(lptr: np.ndarray, lind: np.ndarray, n: int):
    """严格下三角结构 → 严格上三角的 CSR。

    返回 (uptr, uind, src)：
    * ``uptr``/``uind`` 是转置结构（即 Lᵀ）的 CSR，uind 为原行号；
    * ``src`` 把转置结构数值数组映射回原数组：``uval = lval[src]``。
    """
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(lptr))
    uptr = np.zeros(n + 1, np.int64)
    if lptr[-1]:
        order = np.argsort(lind.astype(np.int64), kind="stable")
        np.cumsum(np.bincount(lind.astype(np.int64), minlength=n), out=uptr[1:])
        uind = rows[order].astype(np.int32)
        src = order.astype(np.int64)
    else:
        uind = np.zeros(0, np.int32)
        src = np.zeros(0, np.int64)
    return uptr, uind, src


@dataclass
class _LevelSlices:
    idx: np.ndarray        # 该层非零元在全局数值数组中的位置
    rows: np.ndarray       # 该层有非零元的行
    starts: np.ndarray     # 每行的段起点（拼接数组内）
    nholes: int = 0


@dataclass
class LayerStruct:
    """三角结构的层分解，可被多次求解复用。"""

    n: int
    nlevels: int
    level: np.ndarray
    lptr: np.ndarray
    lind: np.ndarray
    ok: bool = True
    reason: str = ""
    levels: List[_LevelSlices] = field(default_factory=list)

    @property
    def nnz(self) -> int:
        return int(self.lptr[-1])

    @staticmethod
    def build(lptr: np.ndarray, lind: np.ndarray, n: int) -> "LayerStruct":
        """Kahn 拓扑排序求最长路径分层，复杂度 O(nnz + n)。"""
        nnz = int(lptr[-1])
        if n == 0:
            return LayerStruct(0, 0, np.zeros(0, np.int32), lptr, lind)
        row_of_entry = np.repeat(np.arange(n, dtype=np.int64), np.diff(lptr))
        if nnz:
            order = np.argsort(lind.astype(np.int64), kind="stable")
            cptr = np.zeros(n + 1, np.int64)
            np.cumsum(np.bincount(lind.astype(np.int64), minlength=n), out=cptr[1:])
            child = row_of_entry[order]
        else:
            child = np.zeros(0, np.int64)
            cptr = np.zeros(n + 1, np.int64)
        rem = np.diff(lptr).astype(np.int64)
        level = np.full(n, -1, np.int32)
        cur = np.nonzero(rem == 0)[0]
        k = 0
        while cur.size:
            level[cur] = k
            k += 1
            s = cptr[cur]
            e = cptr[cur + 1]
            cnt = e - s
            total = int(cnt.sum())
            rem[cur] = -1
            if total:
                offsets = np.repeat(np.cumsum(cnt) - cnt, cnt)
                sel = np.repeat(s, cnt) + (np.arange(total) - offsets)
                np.subtract.at(rem, child[sel], 1)
            cur = np.nonzero(rem == 0)[0]
        if np.any(level < 0):
            return LayerStruct(n, 0, level, lptr, lind, False,
                               "三角结构含环（不是严格三角），无法分层求解")
        lv_entry = level[row_of_entry]
        order = np.argsort(lv_entry, kind="stable").astype(np.int64)
        rows_sorted = np.argsort(level, kind="stable").astype(np.int64)
        rstart = np.zeros(k + 1, np.int64)
        np.cumsum(np.bincount(level, minlength=k), out=rstart[1:])
        estart = np.zeros(k + 1, np.int64)
        np.cumsum(np.bincount(lv_entry, minlength=k), out=estart[1:])
        lens = np.diff(lptr)
        levels: List[_LevelSlices] = []
        for kk in range(k):
            r = rows_sorted[rstart[kk]:rstart[kk + 1]]
            e = order[estart[kk]:estart[kk + 1]]
            ln = lens[r]
            nz = ln > 0
            starts = np.empty(r.size, np.int64)
            starts[0] = 0
            if r.size > 1:
                np.cumsum(ln[:-1], out=starts[1:])
            levels.append(_LevelSlices(idx=e, rows=r[nz], starts=starts[nz],
                                       nholes=int((~nz).sum())))
        return LayerStruct(n, k, level, lptr, lind, True, "", levels)

    # ------------------------------------------------------------------
    def _sweep(self, val, y):
        """逐层做 y_i -= sum_j val_ij y_j，按层号升序。

        Kahn 分层把“没有依赖的行”放在层 0 并逐层解锁，因此层号升序就是
        合法的拓扑序——对严格下三角和严格上三角都成立（上三角中，行 i 依赖
        列 j>i 的行 j，而行 j 没有对 i 的依赖，同样从叶子开始被解锁）。
        """
        lind = self.lind
        for lev in self.levels:
            if lev.idx.size == 0:
                continue
            y[lev.rows] -= np.add.reduceat(val[lev.idx] * y[lind[lev.idx]], lev.starts)
        return y

    def row_of_entry(self) -> np.ndarray:
        """每个非零元所属的行号（与数值数组对齐）。"""
        return np.repeat(np.arange(self.n, dtype=np.int64), np.diff(self.lptr))

    def scale_rows(self, val: np.ndarray, inv_diag: np.ndarray) -> np.ndarray:
        """把严格三角数值按行除以对角，用于解 (D + T) y = b。

        (D + T) y = b  <=>  (I + D^{-1}T) y = D^{-1}b。返回 D^{-1}T 的数值。
        """
        return val * inv_diag[self.row_of_entry()]

    def solve_lower(self, val: np.ndarray, b: np.ndarray) -> np.ndarray:
        """解 (I + L) y = b，L 严格下三角。"""
        if not self.ok:
            raise ValueError(self.reason)
        return self._sweep(val, np.array(b, dtype=np.float64, copy=True))

    def solve_upper(self, val: np.ndarray, b: np.ndarray) -> np.ndarray:
        """解 (I + U) y = b，U 严格上三角。"""
        if not self.ok:
            raise ValueError(self.reason)
        return self._sweep(val, np.array(b, dtype=np.float64, copy=True))

    def solve_lower_scaled(self, val: np.ndarray, diag: np.ndarray,
                           b: np.ndarray) -> np.ndarray:
        """解 (diag + L) y = b，L 不含对角。"""
        if not self.ok:
            raise ValueError(self.reason)
        dinv = 1.0 / diag
        return self._sweep(self.scale_rows(val, dinv),
                           np.array(b, dtype=np.float64, copy=True) * dinv)

    def solve_upper_scaled(self, val: np.ndarray, diag: np.ndarray,
                           b: np.ndarray) -> np.ndarray:
        """解 (diag + U) y = b，U 不含对角。"""
        if not self.ok:
            raise ValueError(self.reason)
        dinv = 1.0 / diag
        return self._sweep(self.scale_rows(val, dinv),
                           np.array(b, dtype=np.float64, copy=True) * dinv)
