"""导出页面所需数据：每个问题的求解曲线、预条件对比、自动选择依据、
重排前后的稀疏结构图。

产出 web/data/*.json：
* ``index.json``   —— 问题列表与摘要
* ``problem_<key>.json`` —— 单问题的详细结果（运行页面时才计算，
  这里提供预生成的小规模数据供离线查看）
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from .acceptance import (benchmark_preconds, dense_reference_solve, solve_one,
                         _sparse_structure_image)
from .autoselect import auto_select, analyze_matrix
from .generators import PROBLEM_SPECS, make_problem
from .precond import preconditioner_display_name
from .reorder import bandwidth_stats, permute, rcm_order


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(f"不可序列化: {type(o)}")


def export_problem(prob, out_dir, tol=1e-8, maxiter=5000,
                   record_every=1, include_dense=True):
    feats = analyze_matrix(prob.A)
    x_ref = None
    if include_dense and prob.n <= 3000:
        x_ref = dense_reference_solve(prob)
    if feats["looks_spd"]:
        solvers = ["cg"]
        preconds = ["none", "jacobi", "ic0", "ssor"]
    else:
        solvers = ["bicgstab", "gmres"]
        preconds = ["none", "jacobi", "ilu0"]
    # 大规模问题：只跑自动选择推荐的组合 + 无预条件对照，避免导出过慢
    if prob.n > 300000:
        ch = auto_select(prob, tol=tol, maxiter=maxiter)
        seen = set()
        records = []
        combos = [(ch.solver, ch.precond, ch.reorder),
                  (ch.solver, "none", None)]
        for s, p, ro in combos:
            k = (s, p, ro)
            if k in seen:
                continue
            seen.add(k)
            try:
                records.append(solve_one(prob, s, p, tol=tol, maxiter=maxiter,
                                         x_ref=x_ref, reorder=ro,
                                         fallback=True,
                                         record_every=max(1, maxiter // 300)))
            except Exception:
                pass
        # 收敛安全网：子问题试跑可能低估了完整问题的难度（尤其不规则网格，
        # 条件数随规模增长）。若推荐方案在完整问题上未收敛，补一个稳健方案。
        if not any(r.converged for r in records):
            robust = ("cg", "jacobi", None) if feats["looks_spd"] else \
                     ("gmres", "ilu0", "rcm")
            if robust not in seen:
                try:
                    records.append(solve_one(prob, robust[0], robust[1], tol=tol,
                                             maxiter=maxiter, x_ref=x_ref,
                                             reorder=robust[2], fallback=True,
                                             record_every=max(1, maxiter // 300)))
                except Exception:
                    pass
        data_auto = ch.to_dict()
    else:
        records = benchmark_preconds(prob, solvers=solvers, preconds=preconds,
                                     tol=tol, maxiter=maxiter, x_ref=x_ref,
                                     record_every=record_every)
        # RCM 版本（对 IC0/ILU0）
        for s in solvers:
            p = "ic0" if s == "cg" else "ilu0"
            try:
                records.append(solve_one(prob, s, p, tol=tol, maxiter=maxiter,
                                         x_ref=x_ref, reorder="rcm",
                                         record_every=record_every))
            except Exception:
                pass
        data_auto = auto_select(prob, tol=tol, maxiter=maxiter).to_dict()
    # 重排结构图
    structure = {"plain": _sparse_structure_image(prob.A)}
    bw = {"plain": bandwidth_stats(prob.A)}
    try:
        t = time.perf_counter()
        order = rcm_order(prob.A)
        B = permute(prob.A, order)
        structure["rcm"] = _sparse_structure_image(B)
        bw["rcm"] = bandwidth_stats(B)
        bw["rcm"]["order_s"] = time.perf_counter() - t
    except Exception:
        pass
    # 自动选择
    data = {
        "name": prob.name,
        "summary": prob.summary(),
        "features": feats,
        "records": [r.to_dict() for r in records],
        "structure": structure,
        "bandwidth": bw,
        "auto": data_auto,
        "precond_names": {p: preconditioner_display_name(p) for p in
                          ("none", "jacobi", "ic0", "ilu0", "ssor")},
        "tol": tol,
    }
    path = os.path.join(out_dir, f"problem_{prob.name}.json")
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, default=_jsonable)
    return data


def _rebuild_index(out_dir):
    """根据目录下所有 problem_*.json 重建 index.json（按 PROBLEM_SPECS 顺序）。

    这样即使只导出部分问题（--keys 子集），索引也始终反映目录里的全部问题。
    """
    import glob
    index = []
    for spec in PROBLEM_SPECS:
        key = spec["key"]
        path = os.path.join(out_dir, f"problem_{key}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        conv = [r for r in d["records"] if r["converged"]]
        best = min(conv, key=lambda r: r["total_s"]) if conv else None
        index.append({
            "key": key,
            "label": d["summary"]["meta"].get("label", key),
            "kind": d["summary"]["kind"],
            "n": d["summary"]["n"],
            "nnz": d["summary"]["nnz"],
            "family": d["summary"]["meta"].get("family", ""),
            "scale": d["summary"]["meta"].get("scale", ""),
            "auto": {"solver": d["auto"]["solver"],
                     "precond": d["auto"]["precond"],
                     "reorder": d["auto"]["reorder"],
                     "confidence": d["auto"]["confidence"]},
            "best": None if best is None else {
                "solver": best["solver"], "precond": best["precond"],
                "reorder": best["reorder"], "iterations": best["iterations"],
                "total_s": best["total_s"]},
        })
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"problems": index}, f, ensure_ascii=False, indent=1,
                  default=_jsonable)
    return index


def export_all(out_dir, keys=None, seed=0, tol=1e-8, maxiter=5000):
    os.makedirs(out_dir, exist_ok=True)
    keys = keys or [s["key"] for s in PROBLEM_SPECS]
    for key in keys:
        t0 = time.perf_counter()
        prob = make_problem(key, seed=seed)
        print(f"[{key}] n={prob.n} nnz={prob.nnz} 求解中 ...", flush=True)
        export_problem(prob, out_dir, tol=tol, maxiter=maxiter)
        dt = time.perf_counter() - t0
        print(f"   完成，用时 {dt:.1f}s", flush=True)
    return _rebuild_index(out_dir)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "web/data"
    keys = sys.argv[2:] or None
    export_all(out, keys=keys)
