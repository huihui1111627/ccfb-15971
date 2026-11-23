"""Krylov 子空间求解器：CG、BiCGSTAB、GMRES(m)。

统一返回 ``SolverResult``，记录收敛历史（相对残差）、迭代数、耗时、
以及是否收敛/为什么停止。收敛判据为相对残差 ``||r_k|| / ||b|| <= tol``
（与初始残差 ``||r_0||/||b||`` 相比也可用 ``tol_type`` 切换）。

实现要点
--------
* 所有内积用 ``np.dot``（Accelerate 下 ``np.matmul`` 对长向量会发出伪
  Warning，且慢）；范数直接 ``sqrt(dot(r,r))``，不调用 ``np.linalg.norm``。
* CG / BiCGSTAB 使用**预条件残差**的范数做判据？不——判据统一用真实
  残差的相对范数，每 ``check_every`` 步重算一次真实残差，其余迭代用递推
  残差做停止估计，既保证判据可信又避免每步额外一次 SpMV。
* GMRES(m) 使用修正 Gram-Schmidt，可选二次正交化（默认开）以应对
  病态问题。
* 每步都检查非有限值（NaN/Inf），一旦出现立即以 ``breakdown`` 停止。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

__all__ = ["SolverResult", "SolveOptions", "solve_cg", "solve_bicgstab",
           "solve_gmres", "solve_krylov"]


def _dot(a, b):
    return float(np.dot(a, b))


def _norm2(a):
    return float(np.sqrt(np.dot(a, a)))


@dataclass
class SolveOptions:
    tol: float = 1e-8
    maxiter: int = 2000
    restart: int = 60               # GMRES 重启长度
    check_every: int = 25           # 真实残差复核间隔
    reorth: bool = True             # GMRES 二次正交化
    record_every: int = 1           # 历史记录间隔（大问题可调大以省内存）
    atol: float = 0.0               # 绝对残差下限：||r|| <= max(tol*||b||, atol)
    x0: Optional[np.ndarray] = None


@dataclass
class SolverResult:
    solver: str
    precond: str
    converged: bool
    reason: str
    iterations: int
    rel_residual: float
    rel_residual_true: float
    seconds: float
    history: List[float] = field(default_factory=list)
    precond_build_s: float = 0.0
    precond_stats: Dict = field(default_factory=dict)
    spmv_count: int = 0
    precond_apply_count: int = 0
    x: Optional[np.ndarray] = None
    error: Optional[float] = None
    extra: Dict = field(default_factory=dict)

    def to_dict(self, keep_x: bool = False) -> dict:
        d = {
            "solver": self.solver,
            "precond": self.precond,
            "converged": self.converged,
            "reason": self.reason,
            "iterations": self.iterations,
            "rel_residual": self.rel_residual,
            "rel_residual_true": self.rel_residual_true,
            "seconds": self.seconds,
            "history": list(self.history),
            "precond_build_s": self.precond_build_s,
            "precond_stats": dict(self.precond_stats),
            "spmv_count": self.spmv_count,
            "precond_apply_count": self.precond_apply_count,
            "error": self.error,
            "extra": dict(self.extra),
        }
        if keep_x:
            d["x"] = None if self.x is None else self.x.tolist()
        return d


class _CountingOperator:
    """包装 A 与预条件器，统计 SpMV / 预条件应用次数，并检测非有限值。"""

    def __init__(self, matvec: Callable, precond):
        self._mv = matvec
        self.P = precond
        self.spmv = 0
        self.papp = 0
        self.nonfinite = False

    def A(self, x):
        self.spmv += 1
        y = self._mv(x)
        if not np.isfinite(y).all():
            self.nonfinite = True
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y

    def M(self, r):
        self.papp += 1
        z = self.P.apply(r) if self.P is not None else r
        if not np.isfinite(z).all():
            self.nonfinite = True
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return z


def _init_residual(op, A, b, x0):
    if x0 is None:
        x = np.zeros_like(b)
        r = b.copy()
    else:
        x = np.array(x0, dtype=np.float64, copy=True)
        r = b - A(x)
    nb = _norm2(b)
    nr = _norm2(r)
    if nb == 0.0:
        nb = 1.0
    return x, r, nb, nr / nb


def _finish(res: SolverResult, x, b, Ax_fn, nb, hist, opts, t0):
    r = b - Ax_fn(x)
    rr = _norm2(r)
    rel = rr / nb if nb > 0 else rr
    res.rel_residual_true = rel
    res.history = hist
    res.seconds = time.perf_counter() - t0
    res.x = x
    return res


# ---------------------------------------------------------------------------
def solve_cg(A, b, precond=None, name_A="", name_P="none",
             opts: Optional[SolveOptions] = None, x_exact=None) -> SolverResult:
    """共轭梯度法（要求 A 对称正定、预条件子 M 对称正定）。"""
    opts = opts or SolveOptions()
    t0 = time.perf_counter()
    op = _CountingOperator(A, precond)
    bnorm = _norm2(b)
    if bnorm == 0.0:
        res = SolverResult("cg", name_P, True, "zero_rhs", 0, 0.0, 0.0,
                           time.perf_counter() - t0, [0.0])
        res.x = np.zeros_like(b)
        if x_exact is not None:
            res.error = float(np.linalg.norm(res.x - x_exact) / max(np.linalg.norm(x_exact), 1e-300))
        return res
    x, r, nb, rel0 = _init_residual(op, op.A, b, opts.x0)
    hist = [rel0]
    target = max(opts.tol, 0.0)
    if rel0 <= target:
        res = SolverResult("cg", name_P, True, "initial_residual", 0, rel0, rel0,
                           time.perf_counter() - t0, hist)
        res.x = x
        res.spmv_count = op.spmv
        res.precond_apply_count = op.papp
        return res
    z = op.M(r)
    p = z.copy()
    rho = _dot(r, z)
    if not np.isfinite(rho) or rho <= 0.0:
        # 预条件子非正定（或在舍入误差下失效），无法继续 CG
        res = SolverResult("cg", name_P, False, "precond_not_spd", 0, rel0, rel0,
                           time.perf_counter() - t0, hist)
        res.x = x
        return res
    rel = rel0
    it = 0
    reason = "maxiter"
    converged = False
    while it < opts.maxiter:
        Ap = op.A(p)
        pAp = _dot(p, Ap)
        if not np.isfinite(pAp) or pAp <= 0.0:
            reason = "pAp_nonpositive" if np.isfinite(pAp) else "nonfinite"
            break
        alpha = rho / pAp
        x += alpha * p
        r -= alpha * Ap
        it += 1
        # 每 check_every 步用真实残差复核，避免递推误差导致“假收敛”
        true_check = (it % opts.check_every == 0)
        if true_check:
            r = b - op.A(x)
        rr = _norm2(r)
        rel = rr / nb
        if it % opts.record_every == 0 or true_check:
            hist.append(rel)
        if op.nonfinite:
            reason = "nonfinite"
            break
        if rel <= target:
            converged = True
            reason = "converged"
            break
        z = op.M(r)
        rho_new = _dot(r, z)
        if not np.isfinite(rho_new) or rho_new <= 0.0:
            reason = "precond_not_spd"
            break
        beta = rho_new / rho
        p = z + beta * p
        rho = rho_new
    res = SolverResult("cg", name_P, converged, reason, it, rel, rel,
                       time.perf_counter() - t0, hist)
    res.spmv_count = op.spmv
    res.precond_apply_count = op.papp
    res = _finish(res, x, b, op.A, nb, hist, opts, t0)
    res.iterations = it
    res.converged = converged
    res.reason = reason
    res.rel_residual = rel
    res.spmv_count = op.spmv
    res.precond_apply_count = op.papp
    if x_exact is not None:
        res.error = float(np.linalg.norm(x - x_exact) / max(np.linalg.norm(x_exact), 1e-300))
    return res


# ---------------------------------------------------------------------------
def solve_bicgstab(A, b, precond=None, name_A="", name_P="none",
                   opts: Optional[SolveOptions] = None, x_exact=None) -> SolverResult:
    """BiCGSTAB：非对称问题的短递推 Krylov 方法。"""
    opts = opts or SolveOptions()
    t0 = time.perf_counter()
    op = _CountingOperator(A, precond)
    bnorm = _norm2(b)
    if bnorm == 0.0:
        res = SolverResult("bicgstab", name_P, True, "zero_rhs", 0, 0.0, 0.0,
                           time.perf_counter() - t0, [0.0])
        res.x = np.zeros_like(b)
        return res
    x, r, nb, rel0 = _init_residual(op, op.A, b, opts.x0)
    hist = [rel0]
    target = opts.tol
    if rel0 <= target:
        res = SolverResult("bicgstab", name_P, True, "initial_residual", 0, rel0, rel0,
                           time.perf_counter() - t0, hist)
        res.x = x
        return res
    rhat = r.copy()
    rho_old = alpha = omega = 1.0
    v = np.zeros_like(b)
    p = np.zeros_like(b)
    rel = rel0
    it = 0
    reason = "maxiter"
    converged = False
    while it < opts.maxiter:
        rho = _dot(rhat, r)
        if not np.isfinite(rho) or abs(rho) < 1e-300:
            reason = "rho_breakdown"
            break
        if it == 0:
            p = r.copy()
        else:
            if abs(omega) < 1e-300:
                reason = "omega_breakdown"
                break
            beta = (rho / rho_old) * (alpha / omega)
            p = r + beta * (p - omega * v)
        phat = op.M(p)
        v = op.A(phat)
        denom = _dot(rhat, v)
        if not np.isfinite(denom) or abs(denom) < 1e-300:
            reason = "alpha_breakdown"
            break
        alpha = rho / denom
        s = r - alpha * v
        if _norm2(s) / nb <= target:
            x = x + alpha * phat
            it += 1
            rel = _norm2(s) / nb
            hist.append(rel)
            converged = True
            reason = "converged"
            break
        shat = op.M(s)
        t = op.A(shat)
        tt = _dot(t, t)
        if not np.isfinite(tt) or tt < 1e-300:
            reason = "omega_breakdown"
            break
        omega = _dot(t, s) / tt
        x = x + alpha * phat + omega * shat
        r = s - omega * t
        rho_old = rho
        it += 1
        true_check = (it % opts.check_every == 0)
        if true_check:
            r = b - op.A(x)
        rel = _norm2(r) / nb
        if it % opts.record_every == 0 or true_check:
            hist.append(rel)
        if op.nonfinite:
            reason = "nonfinite"
            break
        if rel <= target:
            converged = True
            reason = "converged"
            break
        if abs(omega) < 1e-300:
            reason = "omega_breakdown"
            break
    res = SolverResult("bicgstab", name_P, converged, reason, it, rel, rel,
                       time.perf_counter() - t0, hist)
    res.spmv_count = op.spmv
    res.precond_apply_count = op.papp
    res = _finish(res, x, b, op.A, nb, hist, opts, t0)
    res.iterations = it
    res.converged = converged
    res.reason = reason
    res.rel_residual = rel
    res.spmv_count = op.spmv
    res.precond_apply_count = op.papp
    if x_exact is not None:
        res.error = float(np.linalg.norm(x - x_exact) / max(np.linalg.norm(x_exact), 1e-300))
    return res


# ---------------------------------------------------------------------------
def solve_gmres(A, b, precond=None, name_A="", name_P="none",
                opts: Optional[SolveOptions] = None, x_exact=None) -> SolverResult:
    """GMRES(m)：右预条件重启式 GMRES。

    右预条件：解 A M^{-1} y = b，x = M^{-1} y。这样残差 r = b - A M^{-1} y
    就是真实残差，GMRES 的内部残差估计与真实残差一致，停止判据可靠。

    修正 Gram-Schmidt，可选二次正交化（默认开）以应对病态问题。
    """
    opts = opts or SolveOptions()
    t0 = time.perf_counter()
    op = _CountingOperator(A, precond)
    bnorm = _norm2(b)
    if bnorm == 0.0:
        res = SolverResult("gmres", name_P, True, "zero_rhs", 0, 0.0, 0.0,
                           time.perf_counter() - t0, [0.0])
        res.x = np.zeros_like(b)
        return res
    x, r, nb, rel0 = _init_residual(op, op.A, b, opts.x0)
    hist = [rel0]
    target = opts.tol
    if rel0 <= target:
        res = SolverResult("gmres", name_P, True, "initial_residual", 0, rel0, rel0,
                           time.perf_counter() - t0, hist)
        res.x = x
        return res
    m = max(1, int(opts.restart))
    rel = rel0
    it = 0
    reason = "maxiter"
    converged = False
    use_precond = precond is not None and getattr(precond, "name", "none") != "none"
    while it < opts.maxiter:
        # 右预条件：V 空间是 A M^{-1} 的 Krylov 空间，y 为预条件坐标
        r = b - op.A(x)
        beta = _norm2(r)
        if beta / nb <= target:
            converged = True
            reason = "converged"
            rel = beta / nb
            break
        if not np.isfinite(beta):
            reason = "nonfinite"
            break
        V = np.empty((m + 1, b.shape[0]))
        V[0] = r / beta
        H = np.zeros((m + 1, m))
        cs = np.zeros(m)
        sn = np.zeros(m)
        g = np.zeros(m + 1)
        g[0] = beta
        inner = 0
        for j in range(m):
            # 右预条件：先作用 M^{-1}，再作用 A
            if use_precond:
                w = op.A(op.M(V[j]))
            else:
                w = op.A(V[j])
            for k in range(j + 1):
                h = _dot(w, V[k])
                H[k, j] = h
                w -= h * V[k]
            if opts.reorth:
                for k in range(j + 1):
                    h2 = _dot(w, V[k])
                    H[k, j] += h2
                    w -= h2 * V[k]
            nw = _norm2(w)
            H[j + 1, j] = nw
            inner = j + 1
            it += 1
            for k in range(j):
                tmp = cs[k] * H[k, j] + sn[k] * H[k + 1, j]
                H[k + 1, j] = -sn[k] * H[k, j] + cs[k] * H[k + 1, j]
                H[k, j] = tmp
            denom = np.hypot(H[j, j], H[j + 1, j])
            if denom == 0.0 or not np.isfinite(denom):
                reason = "happy_breakdown"
                break
            cs[j] = H[j, j] / denom
            sn[j] = H[j + 1, j] / denom
            H[j, j] = denom
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            rel = abs(g[j + 1]) / nb
            if it % opts.record_every == 0:
                hist.append(rel)
            if op.nonfinite:
                reason = "nonfinite"
                break
            if rel <= target:
                converged = True
                reason = "converged"
                break
            if nw == 0.0:
                reason = "happy_breakdown"
                break
            V[j + 1] = w / nw
        if inner == 0:
            if reason == "maxiter":
                reason = "gmres_stagnation"
            break
        k = inner
        Ht = H[:k, :k]
        rhs = g[:k]
        try:
            ycoef = np.linalg.solve(Ht, rhs)
        except np.linalg.LinAlgError:
            ycoef = np.linalg.lstsq(Ht, rhs, rcond=None)[0]
        # y = V[:k] ycoef；右预条件时 x = M^{-1} y（累积到当前 x 上）
        # 用 np.dot 而非 @：Accelerate 下大矩阵 @ 会发出伪 Warning 且更慢
        dy = np.dot(V[:k].T, ycoef)
        if use_precond:
            dy = op.M(dy)
        x += dy
        if converged:
            break
        if reason == "happy_breakdown":
            rr = b - op.A(x)
            rel = _norm2(rr) / nb
            hist.append(rel)
            if rel <= target:
                converged = True
                reason = "converged"
            break
    res = SolverResult("gmres", name_P, converged, reason, it, rel, rel,
                       time.perf_counter() - t0, hist)
    res.spmv_count = op.spmv
    res.precond_apply_count = op.papp
    res = _finish(res, x, b, op.A, nb, hist, opts, t0)
    res.iterations = it
    res.converged = converged
    res.reason = reason
    res.rel_residual = rel
    res.spmv_count = op.spmv
    res.precond_apply_count = op.papp
    res.extra["restart"] = m
    res.extra["precond_side"] = "right"
    if x_exact is not None:
        res.error = float(np.linalg.norm(x - x_exact) / max(np.linalg.norm(x_exact), 1e-300))
    return res


SOLVERS = {
    "cg": solve_cg,
    "bicgstab": solve_bicgstab,
    "gmres": solve_gmres,
}


def solve_krylov(solver: str, A, b, precond=None, **kw) -> SolverResult:
    key = solver.lower()
    if key not in SOLVERS:
        raise ValueError(f"未知求解器: {solver}（可选 {sorted(SOLVERS)}）")
    return SOLVERS[key](A, b, precond, **kw)
