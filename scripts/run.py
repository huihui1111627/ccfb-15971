#!/usr/bin/env python3
"""命令行入口：生成数据、跑验收、启动本地页面服务。

用法::

    python3 scripts/run.py export            # 生成 web/data（页面数据）
    python3 scripts/run.py acceptance        # 跑验收，输出 reports/
    python3 scripts/run.py serve             # 起本地静态服务器看页面
    python3 scripts/run.py export --keys poisson2d_med conv_pe20
    python3 scripts/run.py bench --key poisson2d_large   # 单问题详尽基准
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparsesolve.acceptance import run_acceptance  # noqa: E402
from sparsesolve.export_web import export_all  # noqa: E402
from sparsesolve.generators import PROBLEM_SPECS, make_problem  # noqa: E402


def cmd_export(args):
    export_all(args.out, keys=args.keys or None, tol=args.tol,
               maxiter=args.maxiter)


def cmd_acceptance(args):
    out = args.out or "reports"
    summary = run_acceptance(out, keys=args.keys or None)
    _print_summary(summary)


def _print_summary(s):
    print("\n=== 小规模 vs 稠密直接法 ===")
    for it in s["small"]["items"]:
        dc = it["dense_cross_check"]
        dcs = f"{dc:.1e}" if dc is not None else "—(n>1500，仅对 numpy)"
        print(f"  {it['problem']:22s} n={it['n']:6d} 稠密交叉校验={dcs}")
        for r in it["records"]:
            status = "✓" if r["converged"] else "✗"
            err = r["error_vs_dense"]
            print(f"     {status} {r['solver']:9s}+{r['precond']:6s} "
                  f"{'⟲RCM ' if r['reorder'] else '      '}"
                  f"it={r['iterations']:5d} 残差={r['rel_residual']:.1e} "
                  f"|x-x_d|={('%.1e' % err) if err is not None else '—'}")
    print("\n=== 大规模残差判据 ===")
    for it in s["large"]["items"]:
        if "meets_criterion" not in it:
            print(f"  {it['problem']:22s} 异常: {it.get('reason')}")
            continue
        mark = "✓" if it["meets_criterion"] else "✗"
        print(f"  {mark} {it['problem']:22s} n={it['n']:8d} "
              f"{it['solver']}+{it['precond']:6s} it={it['iterations']:5d} "
              f"残差={it['rel_residual']:.2e} {it['seconds']:.1f}s "
              f"{it['mem_bytes']/1048576:.0f}MB")
    print("\n=== 自动选择 vs 枚举最优 ===")
    for it in s["auto_vs_brute"]["items"]:
        a = it["auto"]; f = it["fastest"]
        gap_it = (a["iterations"] / f["iterations"] - 1) * 100 if a["iterations"] and f["iterations"] else None
        gap_t = (a["total_s"] / f["total_s"] - 1) * 100 if a["total_s"] and f["total_s"] else None
        print(f"  {it['problem']:22s} 自动={a['solver']}+{a['precond']}"
              f"{'+RCM' if a['reorder'] else '':5s} it={a['iterations']} "
              f"{a['total_s']:.2f}s | 最快={f['solver']}+{f['precond']}"
              f"{'+RCM' if f['reorder'] else ''} it={f['iterations']} {f['total_s']:.2f}s"
              f" | 迭代差={('%+.0f%%' % gap_it) if gap_it is not None else '—'}"
              f" 耗时差={('%+.0f%%' % gap_t) if gap_t is not None else '—'}")
    print("\n=== 不收敛报告 ===")
    for it in s["failure"]["items"]:
        print(f"  {it['scenario']:34s} 正确报告={it['correctly_reports_failure']} "
              f"(converged={it['converged']}, reason={it.get('reason')})")
    print("\n=== 重排序效果 ===")
    for it in s["reorder"]["items"]:
        b, a2 = it["before"], it["after"]
        p, r = it.get("plain", {}), it.get("rcm", {})
        print(f"  {it['problem']:22s} 带宽 {b['bandwidth']:6d}→{a2['bandwidth']:6d} "
              f"轮廓 {b['profile']:10d}→{a2['profile']:10d} | "
              f"IC0 未重排 it={p.get('iterations','—')} "
              f"重排 it={r.get('iterations','—')}")


def cmd_serve(args):
    import functools
    import http.server
    import socketserver
    os.chdir(args.root)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=".")
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"页面: http://127.0.0.1:{args.port}/index.html")
        print("Ctrl-C 退出")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def cmd_bench(args):
    from sparsesolve.acceptance import benchmark_preconds, dense_reference_solve
    from sparsesolve.autoselect import auto_select, analyze_matrix
    prob = make_problem(args.key)
    feats = analyze_matrix(prob.A)
    print(f"{args.key}: n={prob.n} nnz={prob.nnz} SPD={feats['looks_spd']} "
          f"非对称度={feats['asymmetry']:.1e}")
    x_ref = dense_reference_solve(prob) if prob.n <= 3000 else None
    solvers = ["cg"] if feats["looks_spd"] else ["bicgstab", "gmres"]
    preconds = ["none", "jacobi", "ic0", "ssor"] if feats["looks_spd"] else ["none", "jacobi", "ilu0"]
    recs = benchmark_preconds(prob, solvers=solvers, preconds=preconds,
                              x_ref=x_ref)
    recs.sort(key=lambda r: (not r.converged, r.total_s))
    for r in recs:
        print(f"  {'✓' if r.converged else '✗'} {r.solver:9s}+{r.precond:6s} "
              f"it={r.iterations:6d} 残差={r.rel_residual:.2e} "
              f"建立={r.build_s:.3f}s 总={r.total_s:.2f}s "
              f"内存={r.mem_bytes/1048576:.0f}MB")
    ch = auto_select(prob, verbose=True)
    print(f"自动选择: {ch.solver}+{ch.precond}{'+RCM' if ch.reorder else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="生成页面数据")
    pe.add_argument("--out", default="web/data")
    pe.add_argument("--keys", nargs="*")
    pe.add_argument("--tol", type=float, default=1e-8)
    pe.add_argument("--maxiter", type=int, default=5000)
    pe.set_defaults(fn=cmd_export)

    pa = sub.add_parser("acceptance", help="跑验收")
    pa.add_argument("--out", default="reports")
    pa.add_argument("--keys", nargs="*")
    pa.set_defaults(fn=cmd_acceptance)

    ps = sub.add_parser("serve", help="启动页面服务")
    ps.add_argument("--root", default="web")
    ps.add_argument("--port", type=int, default=8765)
    ps.set_defaults(fn=cmd_serve)

    pb = sub.add_parser("bench", help="单问题基准")
    pb.add_argument("--key", default="poisson2d_med")
    pb.set_defaults(fn=cmd_bench)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
