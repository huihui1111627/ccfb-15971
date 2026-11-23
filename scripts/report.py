"""把验收结果 acceptance.json 渲染成可读的 Markdown 报告。

用法::

    python3 scripts/report.py reports/acceptance.json > reports/REPORT.md
"""
from __future__ import annotations

import json
import sys


def _f(x, d=2):
    if x is None:
        return "—"
    if isinstance(x, float):
        if x != x:
            return "NaN"
        if x == 0:
            return "0"
        if abs(x) >= 1e4 or abs(x) < 1e-2:
            return f"{x:.{d}e}"
        return f"{x:.{d+1}f}"
    return str(x)


def main(path):
    s = json.load(open(path))
    out = []
    out.append("# 稀疏迭代求解器 · 验收报告\n")
    out.append(f"共 {len(s['generated'])} 个测试问题。\n")

    out.append("## 1. 小规模问题 vs 稠密直接法\n")
    out.append("| 问题 | n | 稠密交叉校验 | 方案 | 收敛 | 迭代 | 相对残差 | 相对稠密解误差 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for it in s["small"]["items"]:
        for r in it["records"]:
            name = f"{r['solver']}+{r['precond']}" + ("+RCM" if r["reorder"] else "")
            out.append(f"| {it['problem']} | {it['n']} | {_f(it['dense_cross_check'],1)} | "
                       f"{name} | {'✓' if r['converged'] else '✗'} | {r['iterations']} | "
                       f"{_f(r['rel_residual'],1)} | {_f(r['error_vs_dense'],1)} |")
    out.append("")

    out.append("## 2. 大规模问题残差判据\n")
    out.append("| 问题 | n | 方案 | 收敛 | 迭代 | 相对残差 | 满足判据 | 耗时 s | 内存 MB |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for it in s["large"]["items"]:
        if "meets_criterion" not in it:
            out.append(f"| {it['problem']} | — | — | ✗ | — | — | ✗ | — | — |")
            continue
        out.append(f"| {it['problem']} | {it['n']} | {it['solver']}+{it['precond']} | "
                   f"{'✓' if it['converged'] else '✗'} | {it['iterations']} | "
                   f"{_f(it['rel_residual'],1)} | {'✓' if it['meets_criterion'] else '✗'} | "
                   f"{_f(it['seconds'],1)} | {_f(it.get('mem_bytes',0)/1048576,0)} |")
    out.append("")

    out.append("## 3. 自动选择 vs 枚举最快方案\n")
    out.append("| 问题 | 自动选择 | 自动耗时 s | 枚举最快 | 最快耗时 s | 迭代差 | 耗时差 | 候选数 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for it in s["auto_vs_brute"]["items"]:
        a = it["auto"]
        f = it["fastest"]
        an = f"{a['solver']}+{a['precond']}" + ("+RCM" if a["reorder"] else "")
        fn = f"{f['solver']}+{f['precond']}" + ("+RCM" if f["reorder"] else "")
        gap_i = (a["iterations"] / f["iterations"] - 1) * 100 if a["iterations"] and f["iterations"] else None
        gap_t = (a["total_s"] / f["total_s"] - 1) * 100 if a["total_s"] and f["total_s"] else None
        lim = " (受限)" if it.get("limited") else ""
        out.append(f"| {it['problem']} | {an} | {_f(a['total_s'])} | {fn} | "
                   f"{_f(f['total_s'])} | "
                   f"{('%+.0f%%' % gap_i) if gap_i is not None else '—'} | "
                   f"{('%+.0f%%' % gap_t) if gap_t is not None else '—'} | "
                   f"{it.get('n_candidates_tried','?')}{lim} |")
    out.append("")

    out.append("## 4. 不收敛报告\n")
    out.append("| 场景 | 正确报告 | 收敛标志 | 原因 |")
    out.append("|---|---|---|---|")
    for it in s["failure"]["items"]:
        out.append(f"| {it['scenario']} | {'✓' if it['correctly_reports_failure'] else '✗'} | "
                   f"{it['converged']} | {it.get('reason','')} |")
    out.append("")

    out.append("## 5. 重排序（RCM）效果\n")
    out.append("| 问题 | 带宽 前后 | 轮廓 前后 | IC0 迭代 未重排→重排 |")
    out.append("|---|---|---|---|")
    for it in s["reorder"]["items"]:
        b, a2 = it["before"], it["after"]
        p, r = it.get("plain", {}), it.get("rcm", {})
        out.append(f"| {it['problem']} | {b['bandwidth']} → {a2['bandwidth']} | "
                   f"{b['profile']} → {a2['profile']} | "
                   f"{p.get('iterations','—')} → {r.get('iterations','—')} |")
    out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports/acceptance.json")
