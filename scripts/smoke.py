#!/usr/bin/env python3
"""流水线冒烟测试：不跑大规模，验证生成器→求解→自动选择→导出全链路可用。

用法::

    python3 scripts/smoke.py

适合快速自检（约 30 秒）或接入 CI。
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sparsesolve.acceptance import dense_reference_solve, solve_one
from sparsesolve.autoselect import auto_select, analyze_matrix
from sparsesolve.export_web import export_problem
from sparsesolve.generators import make_problem
from sparsesolve.reorder import rcm_order, permute


def main():
    t0 = time.perf_counter()
    ok = True

    cases = ["poisson2d_small", "conv_pe2", "elastic_1e6", "irregular_med"]
    for key in cases:
        prob = make_problem(key)
        xref = dense_reference_solve(prob) if prob.n <= 12000 else None
        feats = analyze_matrix(prob.A)
        choice = auto_select(prob)
        r = solve_one(prob, choice.solver, choice.precond, tol=1e-8,
                      maxiter=5000, reorder=choice.reorder, fallback=True)
        status = "✓" if r.converged else "✗"
        if not r.converged:
            ok = False
        extra = ""
        if xref is not None:
            x = _get_x(prob, choice)
            e = np.linalg.norm(x - xref) / np.linalg.norm(xref)
            extra = f" |x-x_dense|={e:.1e}"
        print(f"  {status} {key:16s} n={prob.n:6d} "
              f"auto={choice.solver}+{choice.precond}"
              f"{'+RCM' if choice.reorder else ''} "
              f"it={r.iterations:5d} rel={r.rel_residual:.1e}{extra}")

    # 导出链路
    with tempfile.TemporaryDirectory() as d:
        prob = make_problem("poisson2d_small")
        export_problem(prob, d)
        files = os.listdir(d)
        assert any(f.startswith("problem_") for f in files), "导出未生成文件"
        print(f"  ✓ 导出链路（{files[0]}）")

    # RCM 正确性
    prob = make_problem("poisson2d_small")
    o = rcm_order(prob.A)
    B = permute(prob.A, o)
    from sparsesolve.reorder import bandwidth_stats
    assert bandwidth_stats(B)["profile"] <= bandwidth_stats(prob.A)["profile"], \
        "RCM 未改善轮廓"
    print(f"  ✓ RCM 轮廓改善")

    dt = time.perf_counter() - t0
    print(f"\n冒烟测试{'通过' if ok else '失败'}，用时 {dt:.1f}s")
    return 0 if ok else 1


def _get_x(prob, choice):
    """用自动选择方案求解并返回解向量（用于与稠密解对比）。"""
    from sparsesolve.krylov import SOLVERS, SolveOptions
    from sparsesolve.precond import make_preconditioner
    from sparsesolve.reorder import permute, rcm_order
    A, b = prob.A, prob.b
    perm = None
    if choice.reorder == "rcm":
        perm = rcm_order(A)
        A = permute(A, perm)
        b = prob.b[perm]
    P = make_preconditioner(choice.precond)
    P.build(A)
    res = SOLVERS[choice.solver](A.matvec, b, P,
                                 opts=SolveOptions(tol=1e-10, maxiter=10000,
                                                   check_every=200))
    x = res.x
    if perm is not None:
        xo = np.empty_like(x)
        xo[perm] = x
        x = xo
    return x


if __name__ == "__main__":
    sys.exit(main())
