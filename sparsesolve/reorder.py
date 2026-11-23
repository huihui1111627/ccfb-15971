"""稀疏矩阵重排序：逆卡氏（RCM）、带宽/轮廓统计。

逆卡氏重排（Reverse Cuthill–McKee）
--------------------------------
1. 把矩阵结构视为无向图（去掉对角元，列共享边）；
2. 选一个“伪外围”节点作为起点（用两次 BFS 近似求外围）；
3. 按 BFS 层次、层内按度升序排列，得到 Cuthill–McKee 序；
4. 反转该序得到 RCM。

RCM 显著减小带宽和轮廓，从而减少 IC(0)/ILU(0) 的“近期填充需求”，
对不完全分解预条件的迭代数有明显影响——本项目会把重排前后的带宽、
IC(0) 迭代数、建立代价一并报告。

实现是纯 NumPy + 有限 Python 循环（BFS 逐层推进，层内向量化）。
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

from .csr import CSRMatrix

__all__ = ["bandwidth_stats", "rcm_order", "permute", "reverse_cuthill_mckee"]


def _adjacency(A: CSRMatrix, drop_diag: bool = True):
    """把 CSR 结构转成邻接表（CSR 形式的索引数组）。"""
    indptr = A.indptr.astype(np.int64)
    indices = A.indices.astype(np.int64)
    if drop_diag:
        rows = np.repeat(np.arange(A.n, dtype=np.int64), np.diff(indptr))
        mask = indices != rows
        # 去重（结构可能已有重复列，但不应该有）
        cnt = np.bincount(rows[mask], minlength=A.n)
        adj_ptr = np.zeros(A.n + 1, np.int64)
        np.cumsum(cnt, out=adj_ptr[1:])
        adj = indices[mask]
    else:
        adj = indices
        adj_ptr = indptr
    return adj_ptr, adj


def bandwidth_stats(A: CSRMatrix) -> Dict[str, float]:
    """返回带宽、平均带宽、轮廓（profile）等结构指标。

    * 带宽 bandwidth = max |i - j|（结构非零）；
    * 平均行带宽 = mean(max_j - min_j)；
    * profile = sum_i (i - min_j_in_row) （下三角轮廓）。
    """
    rows = np.repeat(np.arange(A.n, dtype=np.int64), np.diff(A.indptr))
    cols = A.indices.astype(np.int64)
    if cols.size == 0:
        return {"bandwidth": 0, "avg_row_span": 0.0, "profile": 0,
                "max_row_nnz": 0, "avg_row_nnz": 0.0}
    bw = int(np.max(np.abs(rows - cols)))
    rowmin = np.full(A.n, np.iinfo(np.int64).max, dtype=np.int64)
    rowmax = np.full(A.n, -1, dtype=np.int64)
    np.minimum.at(rowmin, rows, cols)
    np.maximum.at(rowmax, rows, cols)
    has = rowmax >= 0
    span = (rowmax[has] - rowmin[has])
    # 下三角轮廓：只统计 j <= i 的最小列
    lower = cols <= rows
    if lower.any():
        rl = rows[lower]
        cl = cols[lower]
        lmin = np.full(A.n, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(lmin, rl, cl)
        prof = int(np.sum(np.where(lmin == np.iinfo(np.int64).max, 0,
                                   np.arange(A.n) - lmin)))
    else:
        prof = 0
    return {"bandwidth": bw,
            "avg_row_span": float(span.mean()) if span.size else 0.0,
            "profile": prof,
            "max_row_nnz": int(A.row_nnz.max()),
            "avg_row_nnz": float(A.row_nnz.mean())}


def _pseudo_peripheral(adj_ptr, adj, n, start=None, rng=None):
    """两次 BFS 近似外围节点：从任意节点出发取最远点，再从该点取最远点。"""
    if n == 0:
        return 0
    if start is None:
        # 取度数最小的节点之一作为起点（更接近外围）
        deg = np.diff(adj_ptr)
        start = int(np.argmin(deg))
    far = _bfs_farthest(adj_ptr, adj, n, start)
    far2 = _bfs_farthest(adj_ptr, adj, n, far)
    return far2


def _bfs_farthest(adj_ptr, adj, n, start):
    dist = np.full(n, -1, dtype=np.int64)
    dist[start] = 0
    q = deque([start])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[adj_ptr[u]:adj_ptr[u + 1]]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(int(v))
    reach = dist >= 0
    if not reach.any():
        return start
    # 最远层中的最小度节点更接近外围
    dmax = dist[reach].max()
    cand = np.nonzero(reach & (dist == dmax))[0]
    deg = np.diff(adj_ptr)
    return int(cand[np.argmin(deg[cand])])


def rcm_order(A: CSRMatrix) -> np.ndarray:
    """计算逆卡氏重排序：返回 new_order，满足 new = old[order]。"""
    n = A.n
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if not A.is_square:
        return np.arange(n, dtype=np.int64)
    adj_ptr, adj = _adjacency(A)
    rng = np.random.default_rng(0)
    start = _pseudo_peripheral(adj_ptr, adj, n, rng=rng)
    order = _cm_order(adj_ptr, adj, n, start)
    return order[::-1].copy()


def _cm_order(adj_ptr, adj, n, start):
    """Cuthill–McKee 序（BFS 逐层，层内按度升序）。"""
    visited = np.zeros(n, dtype=bool)
    deg = np.diff(adj_ptr)
    order = []
    # 处理不连通分量：start 所在分量优先，其余按度升序补齐
    remaining_start = [start]
    while True:
        while remaining_start:
            s = remaining_start.pop()
            if visited[s]:
                continue
            # BFS 一层
            queue = deque([s])
            visited[s] = True
            while queue:
                # 当前层的所有节点
                level = list(queue)
                queue.clear()
                for u in level:
                    order.append(u)
                # 收集本层邻居
                cand = []
                for u in level:
                    for v in adj[adj_ptr[u]:adj_ptr[u + 1]]:
                        v = int(v)
                        if not visited[v]:
                            visited[v] = True
                            cand.append(v)
                if cand:
                    # 层内按度升序
                    cand.sort(key=lambda x: (deg[x], x))
                    queue.extend(cand)
        if len(order) == n:
            break
        # 下一个未访问分量：取最小度节点
        unv = np.nonzero(~visited)[0]
        if unv.size == 0:
            break
        remaining_start.append(int(unv[np.argmin(deg[unv])]))
    return np.asarray(order, dtype=np.int64)


def permute(A: CSRMatrix, order: np.ndarray) -> CSRMatrix:
    """返回置换后的矩阵 P A P^T，其中 new_index = old_index[order]。

    即新矩阵的 i 行对应旧矩阵的 order[i] 行。逆置换 inv[order] = arange。
    """
    n = A.n
    order = np.asarray(order, dtype=np.int64)
    if order.size != n:
        raise ValueError("置换向量长度不匹配")
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n, dtype=np.int64)
    rows = np.repeat(np.arange(n, dtype=np.int64), A.row_nnz)
    new_rows = inv[rows]                # 旧行 -> 新行
    new_cols = inv[A.indices.astype(np.int64)]
    # 结构上按新行重排；行内仍需按新列排序
    from .csr import coo_to_csr
    return coo_to_csr(new_rows, new_cols, A.data.copy(), n)


def reverse_cuthill_mckee(A: CSRMatrix) -> Tuple[CSRMatrix, np.ndarray, Dict]:
    """返回 (重排后的矩阵, 置换向量, 带宽统计对比)。"""
    order = rcm_order(A)
    before = bandwidth_stats(A)
    B = permute(A, order)
    after = bandwidth_stats(B)
    stats = {"before": before, "after": after,
             "bandwidth_reduction": (1.0 - after["bandwidth"] /
                                     max(1.0, before["bandwidth"])),
             "profile_reduction": (1.0 - after["profile"] /
                                   max(1.0, before["profile"]))}
    return B, order, stats
