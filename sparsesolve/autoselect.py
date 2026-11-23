"""自动选择求解方法与预条件器。

策略分两阶段（对应题目要求）：

**阶段一：静态筛选（看矩阵特征）**
* 对称性：``||A-Aᵀ||_F / ||A||_F`` 与结构对称性。对称且对角为正 →
  CG 家族；否则 → BiCGSTAB / GMRES。
* 对角占优率与对角量级：对角是否为正、是否接近奇异。对角元非正或
  严重退化时 IC(0)/SSOR 不适用，排除。
* 预条件器与求解器的兼容性：CG 只能配对称正定预条件（IC(0)/SSOR/
  Jacobi）；ILU(0) 配非对称求解器。

**阶段二：小规模试验（看残差下降速度）**
在最多 ``probe_iters`` 步内跑候选组合，按“每步对数残差下降率”给每个
组合打分，结合预条件建立代价选最优。试验用真实矩阵（小规模问题直接
用原矩阵；大规模问题用 RCM 后的子问题或截断子矩阵）。

输出 ``AutoChoice``，包含：
* 选中的 (solver, precond)；
* 每个候选的试跑结果（迭代数、残差下降率、是否发散）；
* 决策依据的文本说明，供页面展示。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .csr import CSRMatrix
from .krylov import SolveOptions, solve_cg, solve_bicgstab, solve_gmres
from .precond import make_preconditioner
from .problem import Problem

__all__ = ["AutoChoice", "analyze_matrix", "auto_select", "CandidateResult"]

_SYMMETRIC_SOLVERS = ("cg",)
_NONSYM_SOLVERS = ("bicgstab", "gmres")
_SPD_PRECONDS = ("none", "jacobi", "ic0", "ssor")
_GENERAL_PRECONDS = ("none", "jacobi", "ilu0")


@dataclass
class CandidateResult:
    solver: str
    precond: str
    ok: bool
    iterations: int
    rel_residual: float
    build_s: float
    probe_s: float
    descent_rate: float          # -mean(log(rel_k)); 越大下降越快
    diverged: bool
    score: float = 0.0
    note: str = ""
    reorder: Optional[str] = None
    per_iter_s: float = 0.0      # 试跑实测每步耗时（含 SpMV + 预条件应用）
    probe_n: int = 0             # 试跑所用问题的规模（可能是子问题）
    probe_nnz: int = 0

    def label(self) -> str:
        base = f"{self.solver}+{self.precond}"
        return f"{base}+rcm" if self.reorder else base

    def to_dict(self):
        return {
            "solver": self.solver, "precond": self.precond,
            "reorder": self.reorder,
            "label": self.label(),
            "ok": self.ok,
            "iterations": self.iterations,
            "rel_residual": self.rel_residual,
            "build_s": self.build_s, "probe_s": self.probe_s,
            "descent_rate": self.descent_rate, "diverged": self.diverged,
            "score": self.score, "note": self.note,
        }


@dataclass
class AutoChoice:
    solver: str
    precond: str
    features: dict
    candidates: List[CandidateResult]
    reasoning: List[str]
    probe_s: float = 0.0
    confidence: str = ""
    reorder: Optional[str] = None
    used_subproblem: bool = False

    def to_dict(self):
        return {
            "solver": self.solver, "precond": self.precond,
            "reorder": self.reorder,
            "used_subproblem": self.used_subproblem,
            "features": self.features,
            "candidates": [c.to_dict() for c in self.candidates],
            "reasoning": list(self.reasoning),
            "probe_s": self.probe_s,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
def analyze_matrix(A: CSRMatrix) -> Dict:
    """提取用于自动决策的矩阵特征。"""
    asym, struct_sym = A.symmetry_metrics()
    rate, avg, dmin, dmax = A.diagonal_dominance()
    d = A.diag
    n = A.n
    pos_diag = int(np.sum(d > 0))
    neg_diag = int(np.sum(d < 0))
    zero_diag = int(np.sum(d == 0.0))
    absd = np.abs(d)
    diag_ratio = float(absd.max() / max(absd.min(), 1e-300)) if absd.size else 1.0
    # 判断对称正定：结构/数值对称 + 对角为正 + 对角占优率不低
    is_symmetric = (asym < 1e-8) and struct_sym
    looks_spd = bool(is_symmetric and zero_diag == 0 and pos_diag == n)
    return {
        "n": int(n),
        "nnz": int(A.nnz),
        "asymmetry": float(asym),
        "structurally_symmetric": bool(struct_sym),
        "symmetric": bool(is_symmetric),
        "diag_dominant_fraction": float(rate),
        "avg_dominance": float(avg),
        "abs_diag_min": float(dmin),
        "abs_diag_max": float(dmax),
        "pos_diag": pos_diag,
        "neg_diag": neg_diag,
        "zero_diag": zero_diag,
        "diag_magnitude_ratio": diag_ratio,
        "looks_spd": looks_spd,
        "avg_row_nnz": float(A.row_nnz.mean()),
        "max_row_nnz": int(A.row_nnz.max()),
    }


def _calibrate_iters(n: int, maxiter: int) -> int:
    """大规模问题限制试跑步数，避免自动选择本身太贵。"""
    if n <= 20000:
        return min(maxiter, 80)
    if n <= 200000:
        return min(maxiter, 60)
    return min(maxiter, 60)


# 大规模问题探测时，从原矩阵抽取的诱导子矩阵规模上限。
# 取 RCM 后的前 m 行：RCM 保证带宽小，前 m 行大致是一片连通的局部区域，
# 其预条件行为能代表全局——这就是工程上“先在小模型上调参”的做法。
_SUBPROBLEM_TARGET = 12000


def _extract_subproblem(prob: Problem, target: int = _SUBPROBLEM_TARGET):
    """从问题中抽出一个诱导子矩阵用于低成本探测。

    做法：从某个外围节点做局部 BFS，收集约 target 个节点，取这些节点之间的
    全部连接作为子矩阵（诱导子图）。广度优先保证子图是原图的一块**连通区域**，
    其预条件行为能代表全局；成本只是局部 BFS（O(target × 度)），与 n 基本
    无关，比全局 RCM 便宜得多。

    返回 (sub_problem, 抽取耗时)。子问题内部重编号为 0..m-1。
    """
    from collections import deque

    from .csr import CSRMatrix
    from .problem import Problem as _P

    n = prob.n
    if n <= target:
        return prob, 0.0
    t0 = time.perf_counter()
    indptr = prob.A.indptr
    indices = prob.A.indices
    # 选一个“外围”起点：优先度数小（接近边界）但 >= 2 的节点（排除仅含
    # 对角元的孤立行，避免 BFS 立刻结束）。
    deg = np.diff(indptr)
    cand = np.nonzero(deg >= 2)[0]
    if cand.size == 0:
        return prob, 0.0
    start = int(cand[np.argmin(deg[cand])])
    visited = np.zeros(n, dtype=bool)
    order = []
    q = deque([start])
    visited[start] = True
    while q and len(order) < target:
        u = q.popleft()
        order.append(u)
        for v in indices[indptr[u]:indptr[u + 1]]:
            v = int(v)
            if not visited[v] and v != u:
                visited[v] = True
                q.append(v)
    # 若从该起点连通的块不足 target，补做多起点 BFS 直到凑够
    if len(order) < min(target, n) and len(order) < 10:
        # 换若干起点重试
        for s2 in cand[np.argsort(deg[cand])][:20]:
            s2 = int(s2)
            if visited[s2]:
                continue
            q = deque([s2])
            visited[s2] = True
            while q and len(order) < target:
                u = q.popleft()
                order.append(u)
                for v in indices[indptr[u]:indptr[u + 1]]:
                    v = int(v)
                    if not visited[v] and v != u:
                        visited[v] = True
                        q.append(v)
            if len(order) >= target:
                break
    m = len(order)
    if m < 10:
        return prob, 0.0
    sel = np.zeros(n, dtype=bool)
    sel[np.asarray(order, dtype=np.int64)] = True
    newid = np.full(n, -1, dtype=np.int64)
    newid[np.asarray(order, dtype=np.int64)] = np.arange(m)
    # 抽取这些行的、列也在选集内的项
    r_list, c_list, v_list = [], [], []
    cnt = np.zeros(m, dtype=np.int64)
    orda = np.asarray(order, dtype=np.int64)
    for k, i in enumerate(orda):
        s, e = int(indptr[i]), int(indptr[i + 1])
        cols = indices[s:e].astype(np.int64)
        keep = np.zeros(e - s, dtype=bool)
        # 向量化：用 newid 查表
        mapped = newid[cols]
        keep = mapped >= 0
        if keep.any():
            c_list.append(mapped[keep])
            v_list.append(prob.A.data[s:e][keep])
            cnt[k] = int(keep.sum())
    sub_ptr = np.zeros(m + 1, np.int64)
    np.cumsum(cnt, out=sub_ptr[1:])
    sub_ind = (np.concatenate(c_list) if c_list else np.zeros(0)).astype(np.int32)
    sub_data = (np.concatenate(v_list) if v_list else np.zeros(0)).astype(np.float64)
    # 子图按新编号不一定升序，需重排每行内列号
    Asub = CSRMatrix(sub_data, sub_ind, sub_ptr, m)
    Asub = _sort_rows(Asub)
    bsub = prob.b[orda]
    sub = _P(A=Asub, b=bsub, name=prob.name + "_sub", kind=prob.kind,
             x_exact=None, meta={**prob.meta, "subproblem": True,
                                 "sub_n": int(m)})
    return sub, time.perf_counter() - t0


def _sort_rows(A):
    """CSR 每行内按列号升序重排（诱导子图抽取后列号无序）。"""
    from .csr import CSRMatrix
    n = A.n
    indptr = A.indptr
    counts = np.diff(indptr)
    order = np.argsort(A.indices.astype(np.int64), kind="stable")
    # 逐行排序的全局排列：行内按列排序
    rows = np.repeat(np.arange(n), counts)
    key = rows.astype(np.int64) * (n + 1) + A.indices.astype(np.int64)
    order = np.argsort(key, kind="quicksort")
    return CSRMatrix(A.data[order].copy(), A.indices[order].copy(), indptr, n)


def _probe_candidate(prob: Problem, solver: str, precond_name: str,
                     probe_iters: int, tol: float,
                     reorder: Optional[str] = None) -> CandidateResult:
    """用真实矩阵试跑指定步数，返回评分信息。

    reorder="rcm" 时在逆卡氏重排后的矩阵上建立预条件与求解，
    用于比较“重排是否让不完全分解更有效”。
    """
    t0 = time.perf_counter()
    A_use = prob.A
    b_use = prob.b
    perm = None
    if reorder == "rcm":
        from .reorder import rcm_order, permute
        perm = rcm_order(prob.A)
        A_use = permute(prob.A, perm)
        b_use = prob.b[perm]
    try:
        P = make_preconditioner(precond_name)
        stats = P.build(A_use)
        build_s = time.perf_counter() - t0
        note = "分解出现退化主元" if stats.get("breakdown") else ""
    except Exception as exc:
        return CandidateResult(solver=solver, precond=precond_name, ok=False,
                               iterations=0, rel_residual=float("inf"),
                               build_s=time.perf_counter() - t0, probe_s=0.0,
                               descent_rate=0.0, diverged=True,
                               note=f"建立失败: {exc}", reorder=reorder)
    opts = SolveOptions(tol=tol, maxiter=probe_iters, restart=min(40, probe_iters),
                        check_every=max(1, probe_iters), record_every=1)
    t1 = time.perf_counter()
    try:
        if solver == "cg":
            res = solve_cg(A_use.matvec, b_use, P, name_P=precond_name, opts=opts)
        elif solver == "bicgstab":
            res = solve_bicgstab(A_use.matvec, b_use, P, name_P=precond_name,
                                 opts=opts)
        else:
            res = solve_gmres(A_use.matvec, b_use, P, name_P=precond_name,
                              opts=opts)
    except Exception as exc:
        return CandidateResult(solver=solver, precond=precond_name, ok=False,
                               iterations=0, rel_residual=float("inf"),
                               build_s=build_s, probe_s=time.perf_counter() - t1,
                               descent_rate=0.0, diverged=True,
                               note=f"试运行失败: {exc}", reorder=reorder)
    probe_s = time.perf_counter() - t1
    hist = np.asarray(res.history, dtype=np.float64)
    hist = hist[np.isfinite(hist) & (hist > 0)]
    # 用“平均每步 log10 下降”刻画收敛速度：
    #   descent = (log10(rel_初) - log10(rel_末)) / 迭代数
    # 这直接反映“跑到这里用了多少步”，比只看尾部斜率更稳健——某些方案
    # 前期慢后期快（或反之），尾部斜率会被平台段误导，平均值不会。
    # 同时保留尾部拟合斜率作为参考（写入 extra，不参与主评分）。
    if hist.size >= 2:
        iters = max(int(res.iterations), 1)
        rel0 = float(hist[0])
        relT = float(hist[-1])
        if rel0 > 0 and relT > 0 and relT < rel0:
            descent = (np.log10(rel0) - np.log10(relT)) / iters
        else:
            descent = 0.0
        # 尾部斜率参考值
        h = hist[max(2, hist.size * 2 // 5):]
        if h.size >= 3:
            x = np.arange(h.size, dtype=np.float64)
            logs = np.log10(h)
            xm = x.mean()
            den = float(np.sum((x - xm) ** 2))
            tail_slope = -float(np.sum((x - xm) * (logs - logs.mean())) / den) if den > 0 else 0.0
        else:
            tail_slope = descent
    else:
        descent = 0.0
        tail_slope = 0.0
    diverged = bool(res.rel_residual > 1.0 and res.rel_residual > hist[0] if hist.size else True)
    ok = bool(res.converged)
    per_iter_s = probe_s / max(res.iterations, 1)
    return CandidateResult(
        solver=solver, precond=precond_name, ok=ok,
        iterations=res.iterations, rel_residual=res.rel_residual_true,
        build_s=build_s, probe_s=probe_s, descent_rate=descent,
        diverged=diverged,
        note=(note + ("; 试跑期发散" if diverged else "")).strip("; "),
        reorder=reorder, per_iter_s=per_iter_s,
        probe_n=int(prob.n), probe_nnz=int(prob.A.nnz),
    )


def _score(c: CandidateResult, tol: float = 1e-8,
           size_ratio: float = 1.0) -> float:
    """用“预测总耗时（秒）”给候选打分，取负值便于降序。

    预测总耗时 = 预计迭代步数 × 每步实测耗时（按规模放大）
               + 预条件建立耗时（按规模放大）。

    * 预计步数：试跑已收敛则用实测步数；否则用对数下降率外推到容差。
    * 每步耗时：试跑实测 ``probe_s / iterations``，这是本实现下最真实
      的每步成本（含 SpMV、预条件应用、正交化等全部开销），比任何经验
      常数都可靠。子问题上测得的值按非零元数比例放大到完整问题。
    * 建立耗时：同样按规模放大（IC/ILU/SSOR 的分解近似随 nnz 线性）。

    这样“IC(0) 迭代少但每步贵、建立也贵”的真实权衡会被如实反映，
    选出的方案与实测最快方案的差距应接近零。
    """
    if c.diverged or not np.isfinite(c.rel_residual):
        return -1e9
    target_log = np.log10(1.0 / max(tol, 1e-16))   # 需要下降的十进制量级数
    if c.descent_rate <= 1e-6:
        return -1e8
    if c.ok:
        n_est = float(max(c.iterations, 1))
    else:
        # descent_rate 是每步的 log10 下降；还差多少量级
        cur_log = np.log10(max(c.rel_residual, 1e-300))
        need = max(0.0, cur_log - np.log10(max(tol, 1e-16)))
        n_est = c.iterations + need / c.descent_rate
    per_iter_full = c.per_iter_s * size_ratio
    build_full = c.build_s * size_ratio
    predicted_s = n_est * per_iter_full + build_full
    return -predicted_s


def auto_select(prob: Problem, tol: float = 1e-8, maxiter: int = 5000,
                probe_iters: Optional[int] = None,
                candidates: Optional[List[Tuple[str, str]]] = None,
                try_rcm: bool = True, verbose: bool = False) -> AutoChoice:
    """自动选择 (solver, precond, 是否重排)。

    返回 AutoChoice，含全部候选的试跑记录与决策依据。
    """
    t_start = time.perf_counter()
    feats = analyze_matrix(prob.A)
    reasoning: List[str] = []

    symmetric = feats["symmetric"]
    looks_spd = feats["looks_spd"]
    rate = feats["diag_dominant_fraction"]
    diag_ok = feats["zero_diag"] == 0 and feats["pos_diag"] == feats["n"]

    # ---- 阶段一：静态筛选 ----
    if symmetric and looks_spd:
        solvers = ["cg"]
        reasoning.append(
            f"矩阵对称（非对称度 {feats['asymmetry']:.1e}）且对角全为正，"
            f"判为对称正定 → 首选 CG。")
        preconds = ["none", "jacobi", "ic0", "ssor"]
    elif symmetric and not diag_ok:
        solvers = ["gmres"]
        preconds = ["none", "jacobi", "ilu0"]
        reasoning.append(
            "矩阵对称但对角元含零或负值，CG 的前提不成立 → 改用 GMRES。")
    else:
        solvers = ["bicgstab", "gmres"]
        preconds = ["none", "jacobi", "ilu0"]
        reasoning.append(
            f"矩阵非对称（非对称度 {feats['asymmetry']:.1e}，"
            f"结构对称={feats['structurally_symmetric']}）→ CG 不适用，"
            f"候选 BiCGSTAB / GMRES。")

    if rate < 0.5:
        reasoning.append(
            f"对角占优率仅 {rate:.0%}，非对角占优 → IC/ILU 可能退化，"
            f"必须试跑验证。")
    if feats["diag_magnitude_ratio"] > 1e3:
        reasoning.append(
            f"对角量级跨越 {feats['diag_magnitude_ratio']:.1e} 倍，"
            f"对角缩放（Jacobi）预期收益明显。")

    # 若显式给了候选，以显式为准
    if candidates is None:
        cand_triples = [(s, p, None) for s in solvers for p in preconds]
    else:
        cand_triples = [(s, p, None) for s, p in candidates]
        solvers = sorted({s for s, _, _ in cand_triples})

    # 对不完全分解额外尝试逆卡氏重排：RCM 缩小带宽，通常能显著提升 IC(0)/ILU(0)
    if try_rcm and candidates is None:
        for s in solvers:
            for p in ("ic0", "ilu0"):
                if p in preconds:
                    cand_triples.append((s, p, "rcm"))

    # ---- 阶段二：试跑 ----
    # 大规模问题（n > _SUBPROBLEM_TARGET）先抽一个 RCM 诱导子问题做探测，
    # 探测成本与 n 无关（约万级未知数）；结论再拿回原问题执行。
    # 这就是工程上“先在小模型上调参，再上大模型”的流程。
    probe_prob = prob
    sub_s = 0.0
    used_sub = False
    if prob.n > _SUBPROBLEM_TARGET:
        try:
            probe_prob, sub_s = _extract_subproblem(prob)
            used_sub = True
            reasoning.append(
                f"规模 {prob.n} 较大，试跑改用 RCM 诱导的 {probe_prob.n} 未知数子问题"
                f"（抽取 {sub_s:.1f}s），以免自动选择本身耗时过长。")
        except Exception as exc:
            reasoning.append(f"子问题抽取失败（{exc}），改为直接在原矩阵上试跑。")
            probe_prob = prob

    pi = probe_iters or _calibrate_iters(probe_prob.n, maxiter)
    tol_probe = min(tol * 10.0, 1e-6)
    # 试跑发生在子问题上时，预条件建立代价需按规模放大到完整问题
    size_ratio = (float(prob.nnz) / max(1, probe_prob.nnz)
                  if used_sub else 1.0)
    results: List[CandidateResult] = []
    # 子问题只能代表局部；当整体矩阵严重非等比例（对角量级跨度大）且非对角
    # 占优时，子问题会高估“无预条件”的可行性（局部块条件好、全局条件差）。
    # 此时给无预条件/纯 Jacobi 加一个保守惩罚，避免在大问题上选到要几千步的方案。
    ill_scaled = (feats["diag_magnitude_ratio"] > 100.0 and
                  feats["diag_dominant_fraction"] < 0.5)
    for solver, precond, ro in cand_triples:
        if used_sub and ro == "rcm":
            continue
        c = _probe_candidate(probe_prob, solver, precond, pi, tol_probe,
                             reorder=ro if not used_sub else None)
        c.score = _score(c, tol=tol_probe, size_ratio=size_ratio)
        if used_sub and ill_scaled and precond in ("none", "jacobi"):
            # 按 1.5 倍步数惩罚预测耗时（子问题低估了全局迭代增长）
            c.score *= 1.5
        results.append(c)
        if verbose:
            print(f"    probe {c.label():22s}: it={c.iterations:4d} "
                  f"rel={c.rel_residual:.2e} desc={c.descent_rate:.3f} "
                  f"score={c.score:.1f} {c.note}")
    probe_s = time.perf_counter() - t_start

    results.sort(key=lambda c: c.score, reverse=True)
    best = results[0]
    reorder = best.reorder
    if best.score <= -1e5:
        fallback = ("bicgstab", "jacobi") if not (symmetric and looks_spd) else ("cg", "jacobi")
        reasoning.append("所有候选试跑均发散或建立失败 → 回退到 "
                         f"{fallback[0]} + {fallback[1]}，并将在正式求解时报告不收敛。")
        best = CandidateResult(solver=fallback[0], precond=fallback[1],
                               ok=False, iterations=0, rel_residual=float("inf"),
                               build_s=0.0, probe_s=0.0, descent_rate=0.0,
                               diverged=True, score=-1e9, note="fallback")
        results.append(best)
        confidence = "low"
        reorder = None
    else:
        # 平手规则：预测耗时接近（相对差 < 15%）时，优先“每步更便宜”的方案，
        # 因为探测期只跑几十步，GMRES 的重启/正交化、IC(0) 的三角求解等
        # 每步开销在短探测里代表不充分；每步更便宜的方案在正式求解中更稳。
        best_abs = -best.score
        cheaper = None
        for c in results:
            if c is best or c.score <= -1e5:
                continue
            c_abs = -c.score
            if best_abs > 0 and (c_abs - best_abs) / best_abs < 0.15:
                if cheaper is None or c.per_iter_s < cheaper.per_iter_s:
                    cheaper = c
        if cheaper is not None and cheaper.per_iter_s < 0.9 * best.per_iter_s:
            reasoning.append(
                f"预测耗时接近（{best.label()} {best_abs*1e3:.0f}ms vs "
                f"{cheaper.label()} {(-cheaper.score)*1e3:.0f}ms），但后者每步"
                f"更便宜（{cheaper.per_iter_s*1e3:.2f}ms vs "
                f"{best.per_iter_s*1e3:.2f}ms），短探测下其每步开销代表性更足 → 选它。")
            best = cheaper
            reorder = best.reorder
        runner = results[1] if len(results) > 1 else None
        reasoning.append(
            f"试跑（最多 {pi} 步）：{best.label()} 的残差下降最快"
            f"（每步中位对数下降 {best.descent_rate:.3f}，"
            f"试跑终点相对残差 {best.rel_residual:.1e}）")
        if runner is not None:
            reasoning.append(
                f"次优 {runner.label()}：下降率 {runner.descent_rate:.3f}，"
                f"终点残差 {runner.rel_residual:.1e}")
        if reorder == "rcm":
            # 对比同一预条件未重排的表现
            nore = next((c for c in results
                         if c.precond == best.precond and c.solver == best.solver
                         and c.reorder is None), None)
            if nore is not None:
                reasoning.append(
                    f"逆卡氏重排使 {best.precond} 的下降率从 "
                    f"{nore.descent_rate:.3f} 提升到 {best.descent_rate:.3f}，"
                    f"故选重排版本。")
        if best.ok:
            confidence = "high"
        else:
            confidence = "medium"
        # 子问题探测选的 IC(0)/ILU(0)：大问题上默认也启用 RCM——
        # 子问题本身就是按 BFS 连通块抽取的，其局部结构接近“已局部重排”。
        if used_sub and best.precond in ("ic0", "ilu0") and reorder is None:
            reorder = "rcm"
            reasoning.append(
                "子问题按连通块抽取，接近已重排形态；在大问题上对不完全分解"
                "同样启用逆卡氏重排。")

        # 兜底验证：子问题结论可能不代表全局（如 IC(0) 在全局出现大量负主元）。
        # 这里在**完整矩阵**上只建一次候选预条件，检查主元健康度；不健康则降级。
        if used_sub and best.precond in ("ic0", "ilu0", "ssor"):
            try:
                from .reorder import permute as _permute, rcm_order as _rcm
                A_full = prob.A
                if reorder == "rcm":
                    A_full = _permute(prob.A, _rcm(prob.A))
                P_chk = make_preconditioner(best.precond)
                st_chk = P_chk.build(A_full)
                zp = int(st_chk.get("zero_pivots", 0) or 0)
                neg = int((P_chk.d < 0).sum()) if hasattr(P_chk, "d") else 0
                if zp > 0 or neg > max(10, 0.001 * prob.n):
                    reasoning.append(
                        f"兜底检查：在完整矩阵上 {best.precond} 出现 {neg} 个非正主元"
                        f"（退化 {zp} 个），判为不可靠 → 降级为 "
                        f"{'SSOR' if best.precond == 'ic0' else 'Jacobi'}。")
                    if best.precond == "ic0":
                        # IC(0) 退化时 SSOR 通常仍可用（不含开方）
                        P_chk2 = make_preconditioner("ssor")
                        st2 = P_chk2.build(A_full)
                        if not st2.get("breakdown"):
                            best.precond = "ssor"
                        else:
                            best.precond = "jacobi"
                    else:
                        best.precond = "jacobi"
                    confidence = "medium"
            except Exception as exc:
                reasoning.append(f"兜底检查未能完成（{exc}），沿用子问题结论。")

    return AutoChoice(solver=best.solver, precond=best.precond,
                      features=feats, candidates=results,
                      reasoning=reasoning, probe_s=probe_s,
                      confidence=confidence, reorder=reorder,
                      used_subproblem=used_sub)
