#!/usr/bin/env python3
"""Compare the result files of a full re-run (under $SCAFFOLD_ROOT) with the archived results used in the
manuscript (results/ in this repository). Usage: SCAFFOLD_ROOT=<re-run root> python3 verification/compare_archive.py For every numeric leaf: bit-identical, within 1e-9 relative, or different (max relative diff).
Non-numeric leaves (e.g. checkpoint paths with timestamps) are listed separately and not counted as failures."""
import json, math, os, sys
ARCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
NEW = os.environ.get("SCAFFOLD_ROOT", "/workspace")
SKIP_KEYS = ("checkpoint", "_us_per_user", "time", "seconds")


def leaves(d, pre=""):
    if isinstance(d, dict):
        for k, v in d.items(): yield from leaves(v, f"{pre}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d): yield from leaves(v, f"{pre}[{i}]")
    else: yield pre, d


tot_bad = 0
for root, _, files in os.walk(ARCH):
    for fn in sorted(files):
        if not fn.endswith(".json"): continue
        rel = os.path.relpath(os.path.join(root, fn), ARCH)
        a = dict(leaves(json.load(open(os.path.join(ARCH, rel)))))
        p = os.path.join(NEW, rel)
        if not os.path.exists(p): print(f"{rel}: NOT PRODUCED"); tot_bad += 1; continue
        b = dict(leaves(json.load(open(p))))
        exact = close = 0; diff = []; other = []
        for k, va in a.items():
            if any(s in k for s in SKIP_KEYS): continue
            vb = b.get(k, "<missing>")
            if isinstance(va, (int, float)) and not isinstance(va, bool) and isinstance(vb, (int, float)) and not isinstance(vb, bool):
                if va == vb: exact += 1
                elif math.isclose(va, vb, rel_tol=1e-9, abs_tol=1e-12): close += 1
                else: diff.append((k, va, vb, abs(va - vb) / max(abs(va), 1e-12)))
            elif va != vb: other.append((k, va, vb))
            else: exact += 1
        extra = [k for k in b if k not in a and not any(s in k for s in SKIP_KEYS)]
        n = exact + close + len(diff)
        msg = f"{rel}: {n} numeric/equal values; {exact} identical, {close} within 1e-9, {len(diff)} differ"
        if diff: msg += f" (max rel diff {max(d[3] for d in diff):.2e})"
        if other: msg += f"; {len(other)} non-numeric differences"
        if extra: msg += f"; {len(extra)} extra keys"
        print(msg)
        for k, va, vb, r in sorted(diff, key=lambda x: -x[3])[:5]: print(f"    {k}: archived={va} new={vb} rel={r:.2e}")
        for k, va, vb in other[:3]: print(f"    (non-numeric) {k}: archived={str(va)[:60]} new={str(vb)[:60]}")
        tot_bad += len(diff)
print("ALL IDENTICAL (up to 1e-9)" if tot_bad == 0 else f"{tot_bad} values differ")
sys.exit(0 if tot_bad == 0 else 1)
