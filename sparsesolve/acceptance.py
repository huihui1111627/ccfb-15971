"""验收测试与报告生成。

验收项（对应题目要求）
----------------------
1. **小规模问题 vs 稠密直接法**：对几千未知数以内的问题，把迭代解与
   ``numpy.linalg.solve``（或我们自带的稠密高斯消元）比较，报告相对误差。
2. **大规模问题**：检查最终相对残差是否满足判据（默认 1e-8）。
3. **预条件对比**：同一问题下各预条件（无/Jacobi/IC(0)/ILU(0)/SSOR）的
   建立耗时、内存、迭代次数、总耗时。
4. **自动选择 vs 枚举最优**：自动选择的结果与“逐一试出最快方案”比较，
   统计迭代次数与总耗时的差距。
5. **不收敛报告**：构造一个必然不收敛的场景（极低容差/极少迭代上限，
   或病态矩阵），验证求解器正确报告 ``converged=False`` 与原因。
6. **重排序效果**：RCM 前后的带宽、轮廓、IC(0) 迭代数与建立代价。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .autoselect import auto_select, analyze_matrix
from .csr import CSRMatrix, from_dense
from .generators import PROBLEM_SPECS, make_problem
from .krylov import SolveOptions, solve_bicgstab, solve_cg, solve_gmres
from .precond import make_preconditioner
from .problem import Problem
from .reorder import bandwidth_stats, permute, rcm_order

__all__ = ["dense_reference_solve", "solve_one", "run_acceptance", "benchmark_preconds",
           "dense_direct_solve_own"]

_SOLVER_FN = {"cg": solve_cg, "bicgstab": solve_bicgstab, "gmres": solve_gmres}


def dense_direct_solve_own(D: np.ndarray, b: np.ndarray) -> np.ndarray:
    """不依赖 LAPACK 的稠密高斯消元（带部分主元），用于交叉验证。

    仅用于小规模问题的参考解——O(n^3)，n 大于约 2000 会很慢。
    """
    A = np.array(D, dtype=np.float64, copy=True)
    bb = np.array(b, dtype=np.float64, copy=True)
    n = A.shape[0]
    for k in range(n):
        p = k + int(np.argmax(np.abs(A[k:, k])))
        if p != k:
            A[[k, p]] = A[[p, k]]
            bb[[k, p]] = bb[[p, k]]
        piv = A[k, k]
        if piv == 0.0:
            raise np.linalg.LinAlgError("奇异矩阵")
        A[k + 1:, k] /= piv
        A[k + 1:, k + 1:] -= np.outer(A[k + 1:, k], A[k, k + 1:])
        bb[k + 1:] -= bb[k] * A[k + 1:, k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (bb[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x


def dense_reference_solve(prob: Problem) -> np.ndarray:
    """用 numpy.linalg.solve 求稠密参考解。"""
    D = prob.A.to_dense()
    return np.linalg.solve(D, prob.b)


# ---------------------------------------------------------------------------
@dataclass
class SolveRecord:
    problem: str
    solver: str
    precond: str
    reorder: Optional[str]
    converged: bool
    reason: str
    iterations: int
    rel_residual: float
    seconds: float
    build_s: float
    total_s: float
    mem_bytes: int
    error_vs_dense: Optional[float]
    error_vs_exact: Optional[float]
    history: List[float]
    spmv: int
    precond_apply: int
    stats: dict

    def to_dict(self, keep_history=True):
        d = dict(self.__dict__)
        if not keep_history:
            d.pop("history", None)
        return d


def solve_one(prob: Problem, solver: str, precond: str,
              tol: float = 1e-8, maxiter: int = 5000,
              reorder: Optional[str] = None,
              parallel_workers: int = 1,
              x_ref: Optional[np.ndarray] = None,
              record_every: int = 1,
              spmv_op=None,
              fallback: bool = False) -> SolveRecord:
    """求解一次并记录完整统计。

    ``fallback=True`` 时启用工程级兜底：若 IC(0)/SSOR/ILU(0) 分解出现负主元
    或 CG 因预条件非正定而立刻失败，自动降级到 Jacobi（对称问题）或
    ILU(0)→Jacobi（非对称），并把降级过程记录在 ``stats["fallback_chain"]``。
    正式“按自动选择求解”用 fallback=True；验收里逐方案对比用 fallback=False
    以保持每个方案的真实表现。
    """
    A = prob.A
    b = prob.b
    perm = None
    if reorder == "rcm":
        perm = rcm_order(A)
        A = permute(A, perm)
        b = prob.b[perm]
    chain = []
    t0 = time.perf_counter()
    P = make_preconditioner(precond)
    st = P.build(A)
    build_s = time.perf_counter() - t0
    opts = SolveOptions(tol=tol, maxiter=maxiter, record_every=record_every,
                        check_every=min(200, max(1, maxiter // 10)))
    matvec = A.matvec if spmv_op is None else spmv_op.matvec
    fn = _SOLVER_FN[solver]
    res = fn(matvec, b, P, name_P=precond, opts=opts)
    if fallback and not res.converged and precond in ("ic0", "ssor", "ilu0"):
        bad = st.get("zero_pivots", 0) > 0 or res.reason == "precond_not_spd"
        if bad:
            # 降级：IC(0)/SSOR -> Jacobi；ILU(0) -> 保持求解器但仍换 Jacobi 预条件
            chain.append({"from": precond, "to": "jacobi",
                          "why": f"分解退化 (zero_pivots={st.get('zero_pivots', 0)}, "
                                 f"reason={res.reason})"})
            t1 = time.perf_counter()
            P2 = make_preconditioner("jacobi")
            st2 = P2.build(A)
            build_s += time.perf_counter() - t1
            res2 = fn(matvec, b, P2, name_P="jacobi", opts=opts)
            if res2.converged or res2.iterations >= res.iterations:
                res = res2
                st = {**st2, "fallback_chain": chain}
                precond = "jacobi"
    x = res.x
    if perm is not None:
        xo = np.empty_like(x)
        xo[perm] = x
        x = xo
    total_s = build_s + res.seconds
    err_dense = None
    if x_ref is not None:
        err_dense = float(np.linalg.norm(x - x_ref) /
                          max(np.linalg.norm(x_ref), 1e-300))
    err_exact = prob.relative_error(x)
    stats = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in st.items()}
    if chain:
        stats["fallback_chain"] = chain
    return SolveRecord(
        problem=prob.name, solver=solver, precond=precond, reorder=reorder,
        converged=res.converged, reason=res.reason, iterations=res.iterations,
        rel_residual=res.rel_residual_true, seconds=res.seconds,
        build_s=build_s, total_s=total_s,
        mem_bytes=int(st.get("mem_bytes", 0)) + int(A.data.nbytes +
                                                   A.indices.nbytes +
                                                   A.indptr.nbytes),
        error_vs_dense=err_dense, error_vs_exact=err_exact,
        history=list(res.history), spmv=res.spmv_count,
        precond_apply=res.precond_apply_count,
        stats=stats,
    )


def benchmark_preconds(prob: Problem, solvers=None, preconds=None,
                       tol: float = 1e-8, maxiter: int = 5000,
                       x_ref: Optional[np.ndarray] = None,
                       record_every: int = 1) -> List[SolveRecord]:
    """在给定问题上逐一尝试 (solver, precond) 组合。"""
    solvers = solvers or ["cg"]
    preconds = preconds or ["none", "jacobi", "ic0", "ssor"]
    out = []
    for s in solvers:
        for p in preconds:
            try:
                out.append(solve_one(prob, s, p, tol=tol, maxiter=maxiter,
                                     x_ref=x_ref, record_every=record_every))
            except Exception as exc:
                out.append(SolveRecord(
                    problem=prob.name, solver=s, precond=p, reorder=None,
                    converged=False, reason=f"exception: {exc}", iterations=0,
                    rel_residual=float("inf"), seconds=0.0, build_s=0.0,
                    total_s=0.0, mem_bytes=0, error_vs_dense=None,
                    error_vs_exact=None, history=[], spmv=0, precond_apply=0,
                    stats={}))
    return out


# ---------------------------------------------------------------------------
def _acceptance_small(problems, tol=1e-10, maxiter=10000) -> Dict:
    """小规模问题与稠密直接法对比。"""
    rows = []
    for prob in problems:
        if prob.n > 4000:
            continue
        xd = dense_reference_solve(prob)
        xd_own = dense_direct_solve_own(prob.A.to_dense(), prob.b)
        dense_cross = float(np.linalg.norm(xd - xd_own) /
                            max(np.linalg.norm(xd), 1e-300))
        recs = []
        feats = analyze_matrix(prob.A)
        if feats["looks_spd"]:
            combos = [("cg", "none"), ("cg", "jacobi"), ("cg", "ic0"),
                      ("cg", "ssor"), ("cg", "ic0", "rcm")]
        else:
            combos = [("bicgstab", "none"), ("bicgstab", "ilu0"),
                      ("gmres", "none"), ("gmres", "ilu0"),
                      ("gmres", "ilu0", "rcm")]
        for combo in combos:
            s, p = combo[0], combo[1]
            ro = combo[2] if len(combo) > 2 else None
            r = solve_one(prob, s, p, tol=tol, maxiter=maxiter, x_ref=xd,
                          reorder=ro)
            recs.append(r)
        rows.append({"problem": prob.name, "n": prob.n,
                     "dense_cross_check": dense_cross,
                     "records": [r.to_dict() for r in recs]})
    return {"items": rows}


def _acceptance_large(problems, tol=1e-8, maxiter=20000) -> Dict:
    """大规模问题的残差判据检查。

    SPD 问题优先用 IC(0)（1M 未知数时建造约 2–5 s，仍远低于迭代本身），
    若非 SPD 或 IC(0) 退化则退回 Jacobi；非对称问题用 GMRES + ILU(0)。
    """
    rows = []
    for prob in problems:
        if prob.n <= 4000:
            continue
        feats = analyze_matrix(prob.A)
        if feats["looks_spd"]:
            combo = ("cg", "ic0")
        else:
            combo = ("gmres", "ilu0")
        try:
            r = solve_one(prob, combo[0], combo[1], tol=tol, maxiter=maxiter,
                          record_every=max(1, maxiter // 200))
            if not r.converged and feats["looks_spd"] and combo[1] != "jacobi":
                # IC(0) 退化时退回 Jacobi，模拟自动选择的降级行为
                r = solve_one(prob, "cg", "jacobi", tol=tol, maxiter=maxiter,
                              record_every=max(1, maxiter // 200))
                combo = ("cg", "jacobi")
            rows.append({"problem": prob.name, "n": prob.n, "nnz": prob.nnz,
                         "solver": combo[0], "precond": combo[1],
                         "converged": r.converged, "reason": r.reason,
                         "iterations": r.iterations,
                         "rel_residual": r.rel_residual,
                         "meets_criterion": bool(r.converged and r.rel_residual <= tol),
                         "seconds": r.total_s, "mem_bytes": r.mem_bytes})
        except Exception as exc:
            rows.append({"problem": prob.name, "n": prob.n, "nnz": prob.nnz,
                         "solver": combo[0], "precond": combo[1],
                         "converged": False, "reason": f"exception: {exc}",
                         "meets_criterion": False})
    return {"items": rows}


def _acceptance_auto_vs_bruteforce(problems, tol=1e-8, maxiter=5000) -> Dict:
    """自动选择 vs 枚举最快方案。"""
    rows = []
    for prob in problems:
        feats = analyze_matrix(prob.A)
        if feats["looks_spd"]:
            combos = [("cg", p) for p in ("none", "jacobi", "ic0", "ssor")]
            combos += [("cg", "ic0", "rcm")]
        else:
            combos = [("bicgstab", p) for p in ("none", "jacobi", "ilu0")]
            combos += [("gmres", p) for p in ("none", "jacobi", "ilu0")]
            combos += [("gmres", "ilu0", "rcm")]
        brute = []
        for combo in combos:
            s, p = combo[0], combo[1]
            ro = combo[2] if len(combo) > 2 else None
            try:
                r = solve_one(prob, s, p, tol=tol, maxiter=maxiter, reorder=ro)
            except Exception:
                continue
            brute.append(r)
        ok_brute = [r for r in brute if r.converged]
        if not ok_brute:
            continue
        fastest = min(ok_brute, key=lambda r: r.total_s)
        fewest = min(ok_brute, key=lambda r: r.iterations)
        choice = auto_select(prob, tol=tol, maxiter=maxiter)
        try:
            auto_r = solve_one(prob, choice.solver, choice.precond, tol=tol,
                               maxiter=maxiter, reorder=choice.reorder)
        except Exception as exc:
            auto_r = None
        rows.append({
            "problem": prob.name, "n": prob.n,
            "auto": {"solver": choice.solver, "precond": choice.precond,
                     "reorder": choice.reorder,
                     "confidence": choice.confidence,
                     "probe_s": choice.probe_s,
                     "converged": None if auto_r is None else auto_r.converged,
                     "iterations": None if auto_r is None else auto_r.iterations,
                     "total_s": None if auto_r is None else auto_r.total_s,
                     "rel_residual": None if auto_r is None else auto_r.rel_residual},
            "fastest": {"solver": fastest.solver, "precond": fastest.precond,
                        "reorder": fastest.reorder,
                        "iterations": fastest.iterations,
                        "total_s": fastest.total_s},
            "fewest_iters": {"solver": fewest.solver, "precond": fewest.precond,
                             "reorder": fewest.reorder,
                             "iterations": fewest.iterations,
                             "total_s": fewest.total_s},
            "candidates": [r.to_dict(keep_history=False) for r in brute],
            "reasoning": choice.reasoning,
        })
    return {"items": rows}


def _acceptance_failure_report() -> Dict:
    """验证不收敛时能正确报告。"""
    from .generators import elasticity_2d
    A, b, xe, meta = elasticity_2d(nx=40, contrast=1e8)
    from .problem import Problem
    prob = Problem(A=A, b=b, name="fail_case", kind="spd", x_exact=xe, meta=meta)
    out = []
    # 1) 迭代上限极小
    r = solve_one(prob, "cg", "none", tol=1e-12, maxiter=3)
    out.append({"scenario": "maxiter=3, tol=1e-12",
                "converged": r.converged, "reason": r.reason,
                "iterations": r.iterations,
                "rel_residual": r.rel_residual,
                "correctly_reports_failure": bool(not r.converged and r.reason)})
    # 2) 用非对称求解器 + 极紧容差
    r2 = solve_one(prob, "gmres", "ilu0", tol=1e-15, maxiter=5)
    out.append({"scenario": "gmres+ilu0 maxiter=5 tol=1e-15",
                "converged": r2.converged, "reason": r2.reason,
                "iterations": r2.iterations,
                "rel_residual": r2.rel_residual,
                "correctly_reports_failure": bool(not r2.converged and r2.reason)})
    # 3) 奇异矩阵（零对角）
    n = 50
    rows = np.repeat(np.arange(n), 3)
    cols = np.concatenate([np.arange(n), (np.arange(n) + 1) % n,
                           (np.arange(n) - 1) % n])
    vals = np.concatenate([np.zeros(n), np.ones(n) * -1.0, np.ones(n) * -1.0])
    As = CSRMatrix(vals, cols.astype(np.int32),
                   np.arange(0, 3 * n + 1, 3, dtype=np.int64), n)
    prob_s = Problem(A=As, b=np.ones(n), name="singular", kind="spd")
    r3 = solve_one(prob_s, "cg", "jacobi", tol=1e-10, maxiter=200)
    out.append({"scenario": "singular (zero diagonal)",
                "converged": r3.converged, "reason": r3.reason,
                "iterations": r3.iterations,
                "correctly_reports_failure": bool(not r3.converged)})
    return {"items": out}


def _acceptance_reordering(problems, tol=1e-8) -> Dict:
    """重排序（RCM）对带宽与 IC(0) 效果的影响。"""
    rows = []
    for prob in problems:
        feats = analyze_matrix(prob.A)
        if not feats["looks_spd"]:
            continue
        order = rcm_order(prob.A)
        B = permute(prob.A, order)
        before = bandwidth_stats(prob.A)
        after = bandwidth_stats(B)
        entry = {"problem": prob.name, "n": prob.n,
                 "before": before, "after": after}
        for ro, label in ((None, "plain"), ("rcm", "rcm")):
            try:
                r = solve_one(prob, "cg", "ic0", tol=tol, maxiter=3000,
                              reorder=ro)
                entry[label] = {"converged": r.converged,
                                "iterations": r.iterations,
                                "build_s": r.build_s,
                                "total_s": r.total_s,
                                "rel_residual": r.rel_residual,
                                "reason": r.reason}
            except Exception as exc:
                entry[label] = {"converged": False, "reason": str(exc)}
        # SSOR 作为对照
        try:
            rs = solve_one(prob, "cg", "ssor", tol=tol, maxiter=3000)
            entry["ssor"] = {"converged": rs.converged,
                             "iterations": rs.iterations,
                             "total_s": rs.total_s}
        except Exception:
            pass
        rows.append(entry)
    return {"items": rows}


# ---------------------------------------------------------------------------
def _sparse_structure_image(A: CSRMatrix, max_points: int = 20000) -> Dict:
    """稀疏结构散点（用于页面绘图），行/列归一化到 0..1。"""
    rows = np.repeat(np.arange(A.n), A.row_nnz).astype(np.float64)
    cols = A.indices.astype(np.float64)
    nnz = rows.size
    if nnz > max_points:
        sel = np.linspace(0, nnz - 1, max_points).astype(np.int64)
        rows = rows[sel]
        cols = cols[sel]
    n = max(1, A.n)
    return {"n": A.n, "nnz": A.nnz, "shown": int(rows.size),
            "x": (cols / n).round(5).tolist(),
            "y": (rows / n).round(5).tolist(),
            "bandwidth": bandwidth_stats(A)["bandwidth"]}


def run_acceptance(out_dir: str, keys: Optional[List[str]] = None,
                   seed: int = 0) -> Dict:
    """跑完整验收，写出 JSON 报告，返回汇总字典。"""
    os.makedirs(out_dir, exist_ok=True)
    keys = keys or [s["key"] for s in PROBLEM_SPECS]
    problems = []
    for k in keys:
        try:
            problems.append(make_problem(k, seed=seed))
        except Exception as exc:
            print(f"跳过 {k}: {exc}")
    print(f"已生成 {len(problems)} 个问题")
    summary = {"generated": [], "small": None, "large": None,
               "auto_vs_brute": None, "failure": None, "reorder": None,
               "structure": {}}
    for prob in problems:
        summary["generated"].append(prob.summary())
    print("小规模 vs 稠密直接法 ...")
    summary["small"] = _acceptance_small(problems)
    print("大规模残差判据 ...")
    summary["large"] = _acceptance_large(problems)
    print("自动选择 vs 枚举 ...")
    summary["auto_vs_brute"] = _acceptance_auto_vs_bruteforce(problems)
    print("不收敛报告 ...")
    summary["failure"] = _acceptance_failure_report()
    print("重排序效果 ...")
    summary["reorder"] = _acceptance_reordering(problems)
    print("稀疏结构图数据 ...")
    for prob in problems:
        if prob.n <= 300000:
            summary["structure"][prob.name] = {
                "plain": _sparse_structure_image(prob.A),
            }
            try:
                order = rcm_order(prob.A)
                summary["structure"][prob.name]["rcm"] = _sparse_structure_image(
                    permute(prob.A, order))
            except Exception:
                pass
    with open(os.path.join(out_dir, "acceptance.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    return summary
