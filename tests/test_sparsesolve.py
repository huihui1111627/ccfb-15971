"""pytest 测试：数值正确性、预条件性质、求解器、自动选择。"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from sparsesolve.acceptance import dense_direct_solve_own
from sparsesolve.csr import CSRMatrix, coo_to_csr, from_dense
from sparsesolve.generators import (anisotropic_layered_2d, convection_diffusion_2d,
                                    elasticity_2d, make_problem, poisson_2d,
                                    poisson_3d, poisson_irregular_2d)
from sparsesolve.krylov import (SolveOptions, solve_bicgstab, solve_cg,
                                solve_gmres)
from sparsesolve.precond import make_preconditioner
from sparsesolve.reorder import bandwidth_stats, permute, rcm_order
from sparsesolve.trisolve import LayerStruct, lower_csr_pattern


# --------------------------------------------------------------- CSR 基础
def test_csr_matvec_matches_dense():
    rng = np.random.default_rng(0)
    n = 80
    D = rng.random((n, n))
    D[D < 0.8] = 0.0
    D[np.arange(n), np.arange(n)] += 5.0
    A = from_dense(D)
    x = rng.random(n)
    assert np.allclose(A.matvec(x), D @ x)
    assert np.allclose(A.rmatvec(x), D.T @ x)


def test_csr_shape_and_diag():
    n = 30
    D = np.diag(np.arange(1.0, n + 1)) + np.diag(np.ones(n - 1), 1)
    A = from_dense(D)
    assert A.shape == (n, n)
    assert np.allclose(A.diag, np.arange(1.0, n + 1))
    assert np.allclose(A.transpose().to_dense(), D.T)


def test_coo_duplicate_sum():
    rows = np.array([0, 0, 1])
    cols = np.array([0, 0, 1])
    vals = np.array([1.0, 2.0, 3.0])
    A = coo_to_csr(rows, cols, vals, 2)
    assert A.to_dense()[0, 0] == 3.0


def test_csr_unsorted_rejected():
    with pytest.raises(ValueError):
        CSRMatrix(np.ones(3), np.array([1, 2, 3], np.int32),
                  np.array([0, 3, 3], np.int64), 2)


# --------------------------------------------------------------- 三角求解
def test_lower_solve_correct():
    A = poisson_2d(10)[0]
    n = A.n
    lptr, lind, mask = lower_csr_pattern(A.indices, A.indptr, n)
    ls = LayerStruct.build(lptr, lind, n)
    assert ls.ok
    lval = np.random.default_rng(0).uniform(0.01, 0.1, len(lind))
    b = np.random.default_rng(1).random(n)
    y = ls.solve_lower(lval, b)
    # 参考：显式 (I+L) y = b
    L = np.eye(n)
    rows = np.repeat(np.arange(n), np.diff(lptr))
    L[rows, lind] = lval
    assert np.allclose(y, np.linalg.solve(L, b), atol=1e-12)


def test_upper_solve_matches_dense():
    A = poisson_2d(8)[0]
    n = A.n
    from sparsesolve.trisolve import upper_csr_pattern
    uptr, uind, mask = upper_csr_pattern(A.indices, A.indptr, n)
    ls = LayerStruct.build(uptr, uind, n)
    uval = np.random.default_rng(0).uniform(0.01, 0.1, len(uind))
    b = np.random.default_rng(1).random(n)
    y = ls.solve_upper(uval, b)
    U = np.eye(n)
    rows = np.repeat(np.arange(n), np.diff(uptr))
    U[rows, uind] = uval
    assert np.allclose(y, np.linalg.solve(U, b), atol=1e-12)


# --------------------------------------------------------------- 预条件
def _apply_matrix(P, n):
    Z = np.column_stack([P.apply(e) for e in np.eye(n)])
    return np.linalg.inv(Z)


@pytest.mark.parametrize("omega", [1.0, 1.4])
def test_ssor_is_symmetric_and_correct(omega):
    A = poisson_2d(9)[0]
    n = A.n
    P = make_preconditioner("ssor", omega=omega)
    P.build(A)
    Z = np.column_stack([P.apply(e) for e in np.eye(n)])
    Dm = A.to_dense()
    d = np.diag(Dm).copy()
    D = np.diag(d)
    L = np.tril(Dm, -1)
    U = np.triu(Dm, 1)
    # M = 1/(ω(2-ω)) (D/ω+L) D^{-1} (D/ω+U)
    # => M^{-1} = ω(2-ω) (D/ω+U)^{-1} D (D/ω+L)^{-1}
    ref = omega * (2.0 - omega) * np.linalg.solve(D / omega + U,
                                                  D @ np.linalg.solve(
                                                      D / omega + L, np.eye(n)))
    assert np.allclose(Z, ref, atol=1e-10)


@pytest.mark.parametrize("scale_diag", [False, True])
def test_ic0_factorization_quality(scale_diag):
    A = poisson_2d(12)[0]
    n = A.n
    P = make_preconditioner("ic0", scale_diag=scale_diag)
    P.build(A)
    Z = np.column_stack([P.apply(e) for e in np.eye(n)])
    # 重构 L D L^T（若启用对角缩放，M = S L D L^T S）
    L = np.eye(n)
    rows = np.repeat(np.arange(n), np.diff(P.lptr))
    L[rows, P.lind] = P.lval
    M = np.dot(np.dot(L, np.diag(P.d)), L.T)
    if P.scale_diag_used:
        # apply 内部先乘 s 再乘 s：z = S M̃^{-1} S r，故 M = S^{-1} M̃ S^{-1}
        S = np.diag(1.0 / P.scale_vec)
        M = np.dot(np.dot(S, M), S)
    assert np.allclose(Z, np.linalg.inv(M), atol=1e-9)


def test_ilu0_factorization_quality():
    A = convection_diffusion_2d(12, pe=2.0)[0]
    n = A.n
    P = make_preconditioner("ilu0")
    P.build(A)
    Z = np.column_stack([P.apply(e) for e in np.eye(n)])
    L = np.eye(n)
    rows = np.repeat(np.arange(n), np.diff(P.lptr))
    L[rows, P.lind] = P.lval
    U = np.diag(P.d).astype(float)
    rows_u = np.repeat(np.arange(n), np.diff(P.uptr))
    U[rows_u, P.uind] = P.uval
    assert np.allclose(Z, np.linalg.inv(np.dot(L, U)), atol=1e-8)


# --------------------------------------------------------------- 生成器
@pytest.mark.parametrize("gen", [poisson_2d, poisson_3d])
def test_generators_spd_and_consistent(gen):
    A, b, xe, meta = gen(8)
    asym, sym = A.symmetry_metrics()
    assert sym and asym < 1e-10
    assert np.all(A.diag > 0)
    assert np.allclose(A.matvec(xe), b, atol=1e-10)


def test_irregular_generator_spd():
    A, b, xe, meta = poisson_irregular_2d(10, jitter=0.1)
    asym, sym = A.symmetry_metrics()
    assert sym and asym < 1e-10
    # 最小特征值应显著为正
    eig = np.linalg.eigvalsh(A.to_dense())
    assert eig.min() > 1e-6 * eig.max()


def test_elasticity_spd():
    A, b, xe, meta = elasticity_2d(6, contrast=1e4)
    eig = np.linalg.eigvalsh(A.to_dense())
    assert eig.min() > 0


def test_convection_is_nonsymmetric():
    A, b, xe, meta = convection_diffusion_2d(10, pe=10.0)
    asym, sym = A.symmetry_metrics()
    # 五点格式的结构是对称的（(i,j) 与 (j,i) 同时存在），但数值不对称
    assert sym and asym > 0.1


def test_anisotropic_spd():
    A, b, xe, meta = anisotropic_layered_2d(12, contrast=1e3)
    eig = np.linalg.eigvalsh(A.to_dense())
    # 允许数值噪声级别的微小负值
    assert eig.min() > -1e-10 * eig.max()


# --------------------------------------------------------------- 求解器
def test_cg_matches_dense():
    from sparsesolve.generators import poisson_2d
    A, b, xe, meta = poisson_2d(60)
    from sparsesolve.problem import Problem
    prob = Problem(A=A, b=b, name="p60", kind="spd", x_exact=xe)
    xref = np.linalg.solve(prob.A.to_dense(), prob.b)
    P = make_preconditioner("ic0")
    P.build(prob.A)
    r = solve_cg(prob.A.matvec, prob.b, P, opts=SolveOptions(tol=1e-10,
                                                            maxiter=3000))
    assert r.converged
    assert np.linalg.norm(r.x - xref) / np.linalg.norm(xref) < 1e-7


def test_bicgstab_and_gmres_nonsymmetric():
    prob = make_problem("conv_pe2")
    xref = np.linalg.solve(prob.A.to_dense(), prob.b)
    P = make_preconditioner("ilu0")
    P.build(prob.A)
    r1 = solve_bicgstab(prob.A.matvec, prob.b, P,
                        opts=SolveOptions(tol=1e-9, maxiter=3000))
    r2 = solve_gmres(prob.A.matvec, prob.b, P,
                     opts=SolveOptions(tol=1e-9, maxiter=3000, restart=50))
    assert r1.converged and r2.converged
    for r in (r1, r2):
        assert np.linalg.norm(r.x - xref) / np.linalg.norm(xref) < 1e-6


def test_nonconvergence_reported():
    prob = make_problem("elastic_1e8")
    r = solve_cg(prob.A.matvec, prob.b, None,
                 opts=SolveOptions(tol=1e-14, maxiter=2))
    assert not r.converged
    assert r.reason  # 非空原因


def test_zero_rhs():
    prob = make_problem("poisson2d_small")
    r = solve_cg(prob.A.matvec, np.zeros(prob.n), None)
    assert r.converged and r.iterations == 0


# --------------------------------------------------------------- 重排序
def test_rcm_reduces_bandwidth_irregular():
    A = poisson_irregular_2d(60, jitter=0.15)[0]
    order = rcm_order(A)
    B = permute(A, order)
    b0 = bandwidth_stats(A)["bandwidth"]
    b1 = bandwidth_stats(B)["bandwidth"]
    assert b1 < b0


def test_permute_preserves_matrix():
    A = poisson_2d(12)[0]
    order = rcm_order(A)
    B = permute(A, order)
    D = A.to_dense()
    assert np.allclose(B.to_dense(), D[np.ix_(order, order)])


# --------------------------------------------------------------- 自动选择
def test_auto_select_spd_uses_cg():
    from sparsesolve.autoselect import auto_select
    prob = make_problem("poisson2d_med")
    ch = auto_select(prob)
    assert ch.solver == "cg"
    assert ch.reasoning


def test_auto_select_nonsym_avoids_cg():
    from sparsesolve.autoselect import auto_select
    prob = make_problem("conv_pe20")
    ch = auto_select(prob)
    assert ch.solver in ("bicgstab", "gmres")


def test_dense_direct_solver_matches_numpy():
    rng = np.random.default_rng(0)
    n = 60
    D = rng.random((n, n)) + n * np.eye(n)
    b = rng.random(n)
    x1 = dense_direct_solve_own(D, b)
    x2 = np.linalg.solve(D, b)
    assert np.allclose(x1, x2, atol=1e-9)


# --------------------------------------------------------------- 并行
def test_parallel_spmv_matches_serial():
    from sparsesolve.parallel import ParallelSpMV
    A = poisson_2d(80)[0]
    x = np.random.default_rng(0).random(A.n)
    y0 = A.matvec(x)
    with ParallelSpMV(A, workers=2) as P:
        if P.active:
            assert np.allclose(P.matvec(x), y0, atol=1e-12)
