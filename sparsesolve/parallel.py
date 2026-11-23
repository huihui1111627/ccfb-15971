"""并行稀疏矩阵-向量乘：按行分块，多进程 + POSIX 共享内存。

为什么用进程而不是线程：CPython 的 GIL 会让纯 Python 层的调度无法并行；
NumPy 的 gather（``x[indices]``）本身会释放 GIL，但 ``reduceat`` 与写回
不会，多进程是唯一能真正并行这些内核的方式。共享内存避免每次调用都
pickle 整个矩阵。

内存布局
--------
data/indices/indptr/x/y 全部放在独立的 SharedMemory 段中；每个工作进程
用 ``np.ndarray(..., buffer=shm.buf)`` 零拷贝映射。矩阵数组只读，x/y 由
主进程在每次 matvec 前写入 / 之后读取。

宿主若不支持 fork，会自动退回串行执行。
"""
from __future__ import annotations

import multiprocessing as mp
import os
import warnings
from typing import Optional

import numpy as np

from .csr import CSRMatrix

__all__ = ["ParallelSpMV", "available_workers"]


def available_workers() -> int:
    try:
        n = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        n = os.cpu_count() or 1
    return max(1, int(n))


_WORKER_STATE = {}


def _shm_open(name, shape, dtype):
    """打开共享内存并构造 ndarray（不保留 SharedMemory 引用以免提前 close）。"""
    from multiprocessing import shared_memory as shm
    handle = shm.SharedMemory(name=name)
    return np.ndarray(shape, dtype=dtype, buffer=handle.buf), handle


def _worker_init(payload):
    names = payload["names"]
    shapes = payload["shapes"]
    dtypes = payload["dtypes"]
    state = {}
    handles = []
    for key in ("data", "indices", "indptr", "x", "y"):
        arr, h = _shm_open(names[key], shapes[key], dtypes[key])
        state[key] = arr
        handles.append(h)
    state["bounds"] = payload["bounds"]
    state["row_nnz"] = state["indptr"]  # 别名，减少查找
    _WORKER_STATE.clear()
    _WORKER_STATE.update(state)
    _WORKER_STATE["handles"] = handles


def _worker_matvec(args):
    ii, r0, r1 = args
    st = _WORKER_STATE
    ptr = st["indptr"]
    s, e = int(ptr[r0]), int(ptr[r1])
    v = st["data"][s:e] * st["x"][st["indices"][s:e]]
    y = st["y"]
    cnt = np.diff(ptr[r0:r1 + 1])
    nz = cnt > 0
    y[r0:r1].fill(0.0)
    if nz.any():
        y[r0:r1][nz] = np.add.reduceat(v, ptr[r0:r1][nz] - s)
    return 0


class ParallelSpMV:
    """把一个 CSRMatrix 包装成可并行 matvec 的算子。

    参数
    ----
    A : CSRMatrix
    workers : int
        工作进程数；1 表示串行。
    """

    def __init__(self, A: CSRMatrix, workers: Optional[int] = None):
        self.A = A
        self.n = A.n
        self.ncols = A.ncols
        if workers is None:
            workers = min(4, max(1, available_workers() - 1))
        self.workers = int(max(1, min(workers, self.n)))
        self._pool = None
        self._shm = []
        self._x = None
        self._y = None
        self.calls = 0
        self.fallback_reason = None
        if self.workers > 1:
            if mp.get_start_method(allow_none=True) not in (None, "fork"):
                self.workers = 1
                self.fallback_reason = "non-fork start method"
            else:
                try:
                    self._setup()
                except Exception as exc:  # pragma: no cover - 平台相关
                    warnings.warn(f"并行 SpMV 初始化失败，回退串行: {exc}")
                    self.fallback_reason = str(exc)
                    self._teardown()
                    self.workers = 1

    # ------------------------------------------------------------------
    def _setup(self):
        from multiprocessing import shared_memory as shm
        A = self.A
        ctx = mp.get_context("fork")
        self._ctx = ctx
        arrays = {
            "data": (A.data, np.float64),
            "indices": (A.indices, np.int32),
            "indptr": (A.indptr, np.int64),
            "x": (np.zeros(A.ncols), np.float64),
            "y": (np.zeros(A.n), np.float64),
        }
        names, shapes, dtypes = {}, {}, {}
        for key, (arr, dt) in arrays.items():
            handle = shm.SharedMemory(create=True, size=max(1, arr.nbytes))
            view = np.ndarray(arr.shape, dtype=dt, buffer=handle.buf)
            view[...] = arr
            self._shm.append(handle)
            names[key] = handle.name
            shapes[key] = arr.shape
            dtypes[key] = np.dtype(dt).str
        self._x, _ = self._shm_view(names["x"], shapes["x"])
        self._y, _ = self._shm_view(names["y"], shapes["y"])
        self._names = names
        # 分块边界：尽量让每块非零元数目接近
        nnz_ptr = A.indptr
        total = int(nnz_ptr[-1])
        bounds = []
        target = total / self.workers
        start = 0
        for w in range(self.workers - 1):
            want = int(round((w + 1) * target))
            r = int(np.searchsorted(nnz_ptr, want, side="left"))
            r = max(r, start + 1)
            bounds.append((start, r))
            start = r
        bounds.append((start, A.n))
        bounds = [(a, b) for a, b in bounds if b > a]
        self.bounds = bounds
        payload = {"names": names, "shapes": shapes, "dtypes": dtypes, "bounds": bounds}
        self._pool = ctx.Pool(processes=len(bounds), initializer=_worker_init,
                              initargs=(payload,))

    def _shm_view(self, name, shape):
        from multiprocessing import shared_memory as shm
        handle = shm.SharedMemory(name=name)
        self._extra_handles = getattr(self, "_extra_handles", [])
        self._extra_handles.append(handle)
        return np.ndarray(shape, dtype=np.float64, buffer=handle.buf), handle

    def _teardown(self):
        pool = getattr(self, "_pool", None)
        if pool is not None:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass
            self._pool = None
        for attr in ("_extra_handles",):
            for h in getattr(self, attr, []):
                try:
                    h.close()
                except Exception:
                    pass
            setattr(self, attr, [])
        for h in self._shm:
            try:
                h.close()
                h.unlink()
            except Exception:
                pass
        self._shm = []

    def close(self):
        self._teardown()

    def __del__(self):  # pragma: no cover
        try:
            self._teardown()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._pool is not None

    def matvec(self, x: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        self.calls += 1
        if self._pool is None:
            y = self.A.matvec(x, out=out)
            return y
        np.copyto(self._x, x)
        self._pool.map(_worker_matvec, [(i, a, b) for i, (a, b) in enumerate(self.bounds)],
                       chunksize=1)
        y = out if out is not None else np.empty(self.n)
        np.copyto(y, self._y)
        return y

    def rmatvec(self, x: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        # 转置乘没有行分块的天然优势（需要按列归约），直接串行
        return self.A.rmatvec(x, out=out)

    def tune_workers(self, x_test: Optional[np.ndarray] = None) -> int:
        """对本矩阵测试不同进程数，选择实际最快者；返回选中的进程数。"""
        if not self.active:
            return 1
        if x_test is None:
            x_test = np.ones(self.ncols)
        # 简单两步：比较当前配置与串行
        import time
        self.matvec(x_test)
        t_par = time.perf_counter()
        for _ in range(3):
            self.matvec(x_test)
        t_par = (time.perf_counter() - t_par) / 3
        t_ser = time.perf_counter()
        for _ in range(3):
            self.A.matvec(x_test)
        t_ser = (time.perf_counter() - t_ser) / 3
        self.perf = {"parallel_s": t_par, "serial_s": t_ser}
        if t_ser < t_par:
            self.close()
            self.workers = 1
            return 1
        return self.workers
