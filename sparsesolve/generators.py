"""测试矩阵生成器：模拟真实工程结构仿真 / 热分析中的线性方程组。

覆盖
----
1. 规则网格 Poisson（2D 五点、3D 七点）—— 对称正定，条件数 O(h^-2)；
2. 不规则网格 Poisson —— 扰动节点 + P1 三角形有限元组装，节点编号随机化，
   带宽大、无规则结构；
3. 材料刚度差别大的结构问题 —— 二维平面应力弹性体，含高刚度夹杂，
   刚度对比可到 1e6，条件数巨大；
4. 层状各向异性热传导 —— 导热系数相差 1e4 倍；
5. 非对称问题 —— 对流-扩散（迎风），Peclet 数控制非对称程度。

规模从数千到数百万未知数。除另有说明，右端项由已知解 x_exact 生成
（b = A x_exact），便于精确统计误差。

所有矩阵用 COO 组装 + 重复项求和（有限元叠加），再转 CSR。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .csr import CSRMatrix, coo_to_csr

__all__ = [
    "poisson_2d", "poisson_3d", "poisson_irregular_2d", "elasticity_2d",
    "anisotropic_layered_2d", "convection_diffusion_2d",
    "PROBLEM_SPECS", "make_problem", "generate_suite",
]

GenResult = Tuple[CSRMatrix, np.ndarray, Optional[np.ndarray], dict]


def _smooth_solution(n: int, rng) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    x = np.sin(idx * 0.037 + 0.3) * np.cos(idx * 0.011 + 1.1) + 0.5
    x += 0.05 * rng.standard_normal(n)
    return x


def _finish(rows, cols, vals, n, kind, meta, seed=0) -> GenResult:
    A = coo_to_csr(rows, cols, vals, n)
    rng = np.random.default_rng(seed)
    x_exact = _smooth_solution(n, rng)
    b = A.matvec(x_exact)
    return A, b, x_exact, meta


# ---------------------------------------------------------------------------
def poisson_2d(nx: int, ny: Optional[int] = None, seed: int = 0) -> GenResult:
    """二维规则网格 Poisson，五点差分，Dirichlet 边界。"""
    ny = ny or nx
    n = nx * ny
    idx = np.arange(n)
    r = idx // ny
    c = idx % ny
    rows = [idx]
    cols = [idx]
    vals = [4.0 * np.ones(n)]
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        ok = (rr >= 0) & (rr < nx) & (cc >= 0) & (cc < ny)
        rows.append(idx[ok])
        cols.append((rr * ny + cc)[ok])
        vals.append(-np.ones(int(ok.sum())))
    meta = {"grid": [nx, ny], "stencil": "5-point", "bc": "Dirichlet"}
    return _finish(np.concatenate(rows), np.concatenate(cols),
                   np.concatenate(vals), n, "spd_poisson", meta, seed)


def poisson_3d(nx: int, ny: Optional[int] = None, nz: Optional[int] = None,
               seed: int = 0) -> GenResult:
    """三维规则网格 Poisson，七点差分。"""
    ny = ny or nx
    nz = nz or nx
    n = nx * ny * nz
    idx = np.arange(n)
    z = idx % nz
    y = (idx // nz) % ny
    x = idx // (ny * nz)
    rows = [idx]
    cols = [idx]
    vals = [6.0 * np.ones(n)]
    for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0),
                       (0, 1, 0), (0, 0, -1), (0, 0, 1)):
        xx, yy, zz = x + dx, y + dy, z + dz
        ok = ((xx >= 0) & (xx < nx) & (yy >= 0) & (yy < ny) &
              (zz >= 0) & (zz < nz))
        rows.append(idx[ok])
        cols.append(((xx * ny + yy) * nz + zz)[ok])
        vals.append(-np.ones(int(ok.sum())))
    meta = {"grid": [nx, ny, nz], "stencil": "7-point", "bc": "Dirichlet"}
    return _finish(np.concatenate(rows), np.concatenate(cols),
                   np.concatenate(vals), n, "spd_poisson3d", meta, seed)


# ---------------------------------------------------------------------------
def poisson_irregular_2d(nx: int, ny: Optional[int] = None, jitter: float = 0.30,
                         seed: int = 0) -> GenResult:
    """不规则三角网格上的 Poisson（P1 有限元）。

    每个矩形单元劈成两个三角形，节点坐标随机扰动，节点编号随机置换。
    矩阵 SPD、结构对称，但带宽大——用于检验重排序对 IC(0) 的影响。
    """
    ny = ny or nx
    n = nx * ny
    rng = np.random.default_rng(seed)
    # 扰动幅度限制在 <0.4 单元，避免三角形接近退化（面积→0 会让刚度阵病态）
    jitter = float(min(max(jitter, 0.0), 0.4))
    xs = np.repeat(np.arange(nx), ny).astype(np.float64)
    ys = np.tile(np.arange(ny), nx).astype(np.float64)
    xs += jitter * rng.standard_normal(n)
    ys += jitter * rng.standard_normal(n)
    # 用同一扰动场保证相邻单元的公共边节点坐标一致（no-op，因为按节点扰动）
    # 沿较短对角线剖分（Delaunay 式）：对角线 ad 与 bc 中取短者，
    # 这样尽量避免钝角三角形，保证刚度阵是 M 矩阵（IC(0) 才能稳定）。
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b = (i + 1) * ny + j
            c = i * ny + j + 1
            d = (i + 1) * ny + j + 1
            diag_ad = ((xs[a] - xs[d]) ** 2 + (ys[a] - ys[d]) ** 2)
            diag_bc = ((xs[b] - xs[c]) ** 2 + (ys[b] - ys[c]) ** 2)
            if diag_ad <= diag_bc:
                tris.append((a, b, d))
                tris.append((a, d, c))
            else:
                tris.append((a, b, c))
                tris.append((b, d, c))
    tris = np.asarray(tris, dtype=np.int64)
    # 随机置换节点编号：把“旧编号 t”映射成“新编号 inv[t]”，
    # 这样坐标数组保持旧编号顺序即可（xs[旧编号]），三角形则用新编号。
    perm = rng.permutation(n)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n, dtype=np.int64)
    tris = inv[tris]
    # 置换后新编号 k 对应旧节点 perm[k]，所以重排坐标数组与新编号对齐
    xs = xs[perm]
    ys = ys[perm]

    # P1 单元刚度（向量化）：K_e[a,d] = (b_a b_d + c_a c_d) / (4 area)
    p0x, p0y = xs[tris[:, 0]], ys[tris[:, 0]]
    p1x, p1y = xs[tris[:, 1]], ys[tris[:, 1]]
    p2x, p2y = xs[tris[:, 2]], ys[tris[:, 2]]
    area = np.abs(0.5 * ((p1x - p0x) * (p2y - p0y) - (p1y - p0y) * (p2x - p0x)))
    area = np.maximum(area, 1e-12)
    bcoef = np.column_stack([ys[tris[:, 1]] - ys[tris[:, 2]],
                             ys[tris[:, 2]] - ys[tris[:, 0]],
                             ys[tris[:, 0]] - ys[tris[:, 1]]])
    ccoef = np.column_stack([xs[tris[:, 2]] - xs[tris[:, 1]],
                             xs[tris[:, 0]] - xs[tris[:, 2]],
                             xs[tris[:, 1]] - xs[tris[:, 0]]])
    scale = 1.0 / (4.0 * area)
    # 一次性展开 9 个 (a,d) 组合：tris 为 (ntri,3)
    a_idx = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    d_idx = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    rr_rows = tris[:, a_idx].ravel()
    cc_cols = tris[:, d_idx].ravel()
    vv_vals = (scale[:, None] * (bcoef[:, a_idx] * bcoef[:, d_idx] +
                                 ccoef[:, a_idx] * ccoef[:, d_idx])).ravel()
    meta = {"grid": [nx, ny], "elements": "P1 triangles",
            "ntri": int(tris.shape[0]), "jitter": jitter,
            "node_order": "randomized"}
    A, b, xe, meta = _finish(rr_rows, cc_cols, vv_vals, n,
                             "spd_poisson_irregular", meta, seed + 1)
    # 判定物理边界节点：在随机置换之前，原始索引中 i==0/i==nx-1/j==0/j==ny-1
    # 的节点就是计算域边界。置换后取回其新编号。
    ii0 = np.repeat(np.arange(nx), ny)
    jj0 = np.tile(np.arange(ny), nx)
    bnd_old = ((ii0 == 0) | (ii0 == nx - 1) | (jj0 == 0) | (jj0 == ny - 1))
    on_bnd = np.zeros(n, dtype=bool)
    on_bnd[perm[bnd_old]] = True
    if on_bnd.any():
        A, b = _apply_dirichlet(A, b, on_bnd)
        xe = xe.copy()
        xe[on_bnd] = 0.0
        b = A.matvec(xe)
    meta["dirichlet_nodes"] = int(on_bnd.sum())
    return A, b, xe, meta


def _apply_dirichlet(A: CSRMatrix, b: np.ndarray, fixed: np.ndarray):
    """对称施加 Dirichlet：固定行/列清零，对角置 1，b 对应分量置 0。

    保证 SPD（单位行对应已知零解）。返回新 (A, b)。
    """
    n = A.n
    rows = np.repeat(np.arange(n, dtype=np.int64), A.row_nnz)
    cols = A.indices.astype(np.int64)
    is_diag = rows == cols
    fixed_row = fixed[rows]
    fixed_col = fixed[cols]
    # 保留：对角元全部保留；非对角元只在两端都不固定时保留。
    keep = is_diag | ~(fixed_row | fixed_col)
    r_k = rows[keep]
    c_k = cols[keep]
    v_k = A.data[keep].copy()
    kd = r_k == c_k
    # 固定的对角元置 1
    v_k[np.logical_and(kd, fixed[r_k])] = 1.0
    # 若某个固定节点在结构中没有对角元，补一个
    have = np.zeros(n, dtype=bool)
    have[r_k[kd]] = True
    miss = fixed & ~have
    if miss.any():
        mi = np.nonzero(miss)[0]
        r_k = np.concatenate([r_k, mi])
        c_k = np.concatenate([c_k, mi])
        v_k = np.concatenate([v_k, np.ones(mi.size)])
    # 若某个非固定节点缺对角元（结构上本不该），补 0 占位
    miss2 = (~fixed) & (~have)
    if miss2.any():
        mi = np.nonzero(miss2)[0]
        r_k = np.concatenate([r_k, mi])
        c_k = np.concatenate([c_k, mi])
        v_k = np.concatenate([v_k, np.zeros(mi.size)])
    A2 = coo_to_csr(r_k, c_k, v_k, n)
    b2 = np.array(b, dtype=np.float64, copy=True)
    b2[fixed] = 0.0
    return A2, b2

# ---------------------------------------------------------------------------
def _q1_elasticity_E(Ecell: np.ndarray, nx: int, ny: int, nu: float,
                     fixed: np.ndarray):
    """组装 Q1 平面应力刚度矩阵；Ecell 为 (nx-1, ny-1)。"""
    gp = 1.0 / np.sqrt(3.0)
    gauss = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    ndof = 2 * nx * ny
    rr_all, cc_all, vv_all = [], [], []

    def node(i, j):
        return i * ny + j

    for i in range(nx - 1):
        for j in range(ny - 1):
            nodes = [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
            dofs = np.array([2 * nd for nd in nodes] +
                            [2 * nd + 1 for nd in nodes])
            E = Ecell[i, j]
            c = E / (1.0 - nu * nu)
            D = c * np.array([[1.0, nu, 0.0],
                              [nu, 1.0, 0.0],
                              [0.0, 0.0, 0.5 * (1.0 - nu)]])
            Ke = np.zeros((8, 8))
            for xi, eta in gauss:
                dN = 0.25 * np.array([
                    [-(1 - eta), -(1 - xi)],
                    [(1 - eta), -(1 + xi)],
                    [(1 + eta), (1 + xi)],
                    [-(1 + eta), (1 - xi)],
                ])
                B = np.zeros((3, 8))
                for a in range(4):
                    B[0, 2 * a] = dN[a, 0]
                    B[1, 2 * a + 1] = dN[a, 1]
                    B[2, 2 * a] = dN[a, 1]
                    B[2, 2 * a + 1] = dN[a, 0]
                Ke += B.T @ D @ B
            rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
            rr_all.append(rr.ravel())
            cc_all.append(cc.ravel())
            vv_all.append(Ke.ravel())
    rows = np.concatenate(rr_all)
    cols = np.concatenate(cc_all)
    vals = np.concatenate(vv_all)
    A = coo_to_csr(rows, cols, vals, ndof)
    b0 = np.zeros(ndof)
    A2, _ = _apply_dirichlet(A, b0, fixed)
    return A2


def elasticity_2d(nx: int, ny: Optional[int] = None, contrast: float = 1e6,
                  seed: int = 0, nu: float = 0.30) -> GenResult:
    """二维平面应力弹性体，中心区域刚度对比 contrast。

    刚度比直接进入条件数，不做预条件的 CG 几乎无法收敛——这是“材料刚度
    差别很大的结构问题”的核心特征。
    """
    ny = ny or nx
    Ecell = np.ones((nx - 1, ny - 1))
    i0, i1 = nx // 3, 2 * nx // 3
    j0, j1 = ny // 3, 2 * ny // 3
    Ecell[i0:i1, j0:j1] = contrast
    n = nx * ny
    ndof = 2 * n

    def node(i, j):
        return i * ny + j

    fixed = np.zeros(ndof, dtype=bool)
    for j in range(ny):
        fixed[2 * node(0, j)] = True
        fixed[2 * node(0, j) + 1] = True
    for i in range(nx):
        fixed[2 * node(i, 0) + 1] = True
    A = _q1_elasticity_E(Ecell, nx, ny, nu, fixed)
    rng = np.random.default_rng(seed)
    x_exact = _smooth_solution(ndof, rng)
    x_exact[fixed] = 0.0
    b = A.matvec(x_exact)
    meta = {"grid": [nx, ny], "dof": "2 (ux,uy)", "element": "Q1 plane stress",
            "E_contrast": contrast, "nu": nu, "fixed_dofs": int(fixed.sum())}
    return A, b, x_exact, meta


# ---------------------------------------------------------------------------
def anisotropic_layered_2d(nx: int, ny: Optional[int] = None,
                           contrast: float = 1e4, seed: int = 0) -> GenResult:
    """层状各向异性热传导：层间导热系数相差 contrast 倍（SPD）。"""
    ny = ny or nx
    n = nx * ny
    r = np.repeat(np.arange(nx), ny)
    c = np.tile(np.arange(ny), nx)
    idx = np.arange(n)
    nlayer = 8
    layer_id = (np.arange(nx) * nlayer // nx) % 2
    kx = np.where(layer_id == 1, contrast, 1.0)      # (nx,)
    ky = kx.copy()
    rows = [idx]
    cols = [idx]
    diag = np.zeros(n)
    vals_extra = []
    for dr in (-1, 1):
        rr = r + dr
        ok = (rr >= 0) & (rr < nx)
        src = idx[ok]
        dst = (rr * ny + c)[ok]
        kface = 0.5 * (kx[r[ok]] + kx[rr[ok]])
        rows.append(src)
        cols.append(dst)
        vals_extra.append(-kface)
        diag += np.bincount(src, weights=kface, minlength=n)
    for dc in (-1, 1):
        cc = c + dc
        ok = (cc >= 0) & (cc < ny)
        src = idx[ok]
        dst = (r * ny + cc)[ok]
        kface = ky[r[ok]]
        rows.append(src)
        cols.append(dst)
        vals_extra.append(-kface)
        diag += np.bincount(src, weights=kface, minlength=n)
    vals = [diag] + vals_extra
    meta = {"grid": [nx, ny], "k_contrast": contrast, "layers": nlayer}
    return _finish(np.concatenate(rows), np.concatenate(cols),
                   np.concatenate(vals), n, "spd_anisotropic", meta, seed)


# ---------------------------------------------------------------------------
def convection_diffusion_2d(nx: int, ny: Optional[int] = None, pe: float = 5.0,
                            seed: int = 0, wind=(1.0, 0.5)) -> GenResult:
    """二维对流-扩散，一阶迎风差分；Peclet 数 pe 控制非对称程度。"""
    ny = ny or nx
    n = nx * ny
    idx = np.arange(n)
    r = idx // ny
    c = idx % ny
    cx, cy = wind
    # 一阶迎风：-Δu + b·∇u = f。
    #   x 方向：bx>0 时用左邻，bx<0 时用右邻；
    #   y 方向同理。
    # 单元 Peclet 数 pe 即 b·h/ε。离散后（乘 h²）：
    #   对角 = 4 + pe(|bx| + |by|)
    #   上游邻元系数 = -(1 + pe·|b_dir|)，下游邻元系数 = -1。
    absx, absy = abs(cx), abs(cy)
    diag = 4.0 + pe * (absx + absy)
    if cx >= 0:
        wx_neg, wx_pos = -(1.0 + pe * absx), -1.0
    else:
        wx_neg, wx_pos = -1.0, -(1.0 + pe * absx)
    if cy >= 0:
        wy_neg, wy_pos = -(1.0 + pe * absy), -1.0
    else:
        wy_neg, wy_pos = -1.0, -(1.0 + pe * absy)
    rows = [idx]
    cols = [idx]
    vals = [diag * np.ones(n)]
    for dr, dc, w in ((-1, 0, wx_neg), (1, 0, wx_pos),
                      (0, -1, wy_neg), (0, 1, wy_pos)):
        rr, cc = r + dr, c + dc
        ok = (rr >= 0) & (rr < nx) & (cc >= 0) & (cc < ny)
        rows.append(idx[ok])
        cols.append((rr * ny + cc)[ok])
        vals.append(w * np.ones(int(ok.sum())))
    meta = {"grid": [nx, ny], "scheme": "upwind", "peclet": pe,
            "wind": list(wind)}
    return _finish(np.concatenate(rows), np.concatenate(cols),
                   np.concatenate(vals), n, "nonsym_convection", meta, seed)


# ---------------------------------------------------------------------------
# 问题清单与工厂
# ---------------------------------------------------------------------------
PROBLEM_SPECS = [
    # (key, 显示名, kind, 生成器, 参数字典)
    {"key": "poisson2d_small", "label": "2D Poisson 40×40", "kind": "spd",
     "family": "poisson", "gen": "poisson_2d", "args": {"nx": 40},
     "scale": "small", "expected": "SPD"},
    {"key": "poisson2d_med", "label": "2D Poisson 200×200", "kind": "spd",
     "family": "poisson", "gen": "poisson_2d", "args": {"nx": 200},
     "scale": "medium", "expected": "SPD"},
    {"key": "poisson2d_large", "label": "2D Poisson 1000×1000", "kind": "spd",
     "family": "poisson", "gen": "poisson_2d", "args": {"nx": 1000},
     "scale": "large", "expected": "SPD"},
    {"key": "poisson3d_med", "label": "3D Poisson 40³", "kind": "spd",
     "family": "poisson3d", "gen": "poisson_3d", "args": {"nx": 40},
     "scale": "medium", "expected": "SPD"},
    {"key": "poisson3d_large", "label": "3D Poisson 100³", "kind": "spd",
     "family": "poisson3d", "gen": "poisson_3d", "args": {"nx": 100},
     "scale": "large", "expected": "SPD"},
    {"key": "irregular_med", "label": "不规则网格 Poisson 120×120", "kind": "spd",
     "family": "irregular", "gen": "poisson_irregular_2d", "args": {"nx": 120},
     "scale": "medium", "expected": "SPD, 带宽大"},
    {"key": "irregular_large", "label": "不规则网格 Poisson 600×600", "kind": "spd",
     "family": "irregular", "gen": "poisson_irregular_2d", "args": {"nx": 600},
     "scale": "large", "expected": "SPD, 带宽大"},
    {"key": "elastic_1e6", "label": "弹性体 刚度比 1e6 80×80", "kind": "spd",
     "family": "elasticity", "gen": "elasticity_2d",
     "args": {"nx": 80, "contrast": 1e6}, "scale": "medium",
     "expected": "SPD, 条件数极大"},
    {"key": "elastic_1e8", "label": "弹性体 刚度比 1e8 60×60", "kind": "spd",
     "family": "elasticity", "gen": "elasticity_2d",
     "args": {"nx": 60, "contrast": 1e8}, "scale": "medium",
     "expected": "SPD, 条件数极大"},
    {"key": "layered_1e4", "label": "层状各向异性 热传导 200×200", "kind": "spd",
     "family": "anisotropic", "gen": "anisotropic_layered_2d",
     "args": {"nx": 200, "contrast": 1e4}, "scale": "medium",
     "expected": "SPD, 强各向异性"},
    {"key": "conv_pe2", "label": "对流-扩散 Pe=2 100×100", "kind": "nonsym",
     "family": "convection", "gen": "convection_diffusion_2d",
     "args": {"nx": 100, "pe": 2.0}, "scale": "medium",
     "expected": "非对称, 对角占优"},
    {"key": "conv_pe20", "label": "对流-扩散 Pe=20 100×100", "kind": "nonsym",
     "family": "convection", "gen": "convection_diffusion_2d",
     "args": {"nx": 100, "pe": 20.0}, "scale": "medium",
     "expected": "非对称, 强对流"},
    {"key": "conv_pe20_large", "label": "对流-扩散 Pe=20 500×500", "kind": "nonsym",
     "family": "convection", "gen": "convection_diffusion_2d",
     "args": {"nx": 500, "pe": 20.0}, "scale": "large",
     "expected": "非对称, 强对流"},
]

_GENERATORS = {
    "poisson_2d": poisson_2d,
    "poisson_3d": poisson_3d,
    "poisson_irregular_2d": poisson_irregular_2d,
    "elasticity_2d": elasticity_2d,
    "anisotropic_layered_2d": anisotropic_layered_2d,
    "convection_diffusion_2d": convection_diffusion_2d,
}


def make_problem(key: str, seed: int = 0):
    """按 key 生成 Problem 对象。"""
    from .problem import Problem
    spec = next((s for s in PROBLEM_SPECS if s["key"] == key), None)
    if spec is None:
        raise KeyError(f"未知问题: {key}")
    gen = _GENERATORS[spec["gen"]]
    A, b, xe, meta = gen(seed=seed, **spec["args"])
    return Problem(A=A, b=b, name=spec["key"], kind=spec["kind"],
                   x_exact=xe, meta={**spec["args"], **meta,
                                     "label": spec["label"],
                                     "family": spec["family"],
                                     "scale": spec["scale"],
                                     "expected": spec["expected"]})


def generate_suite(keys=None, seed: int = 0):
    keys = keys or [s["key"] for s in PROBLEM_SPECS]
    return [make_problem(k, seed=seed) for k in keys]
