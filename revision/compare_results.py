#!/usr/bin/env python3
"""Compare freshly generated revision results with the reference results used in the manuscript.

Checks, for each result file:
  * the set of result keys equals the reference key set (missing or extra keys are failures);
  * every value equals the reference value; integers and strings exactly, floats within a relative
    tolerance of 1e-9 (absolute 1e-12), which absorbs only the last-bit differences of floating-point
    summation order; the stochastic parts of the pipeline are seeded, so any real difference is reported.
Timing fields (wall-clock microseconds per user) are excluded from both checks because they depend on the machine.
Exit code 0 if everything matches, 1 otherwise. Also prints the software environment.

Usage: python3 compare_results.py      (run in $SCAFFOLD_ROOT/paper/rev1 after the three analysis scripts,
                                        with the files of revision/reference/ copied there)"""
import json, math, platform, sys
PAIRS = [("rev1_mitigation_result.json", "reference_rev1_mitigation_result.json"),
         ("rev1_variants_result.json", "reference_rev1_variants_result.json"),
         ("rev1_junyi_outcome_result.json", "reference_rev1_junyi_outcome_result.json")]
TIMING = ("filter_us_per_user", "rerank_us_per_user")


def leaves(d, pre=""):
    if isinstance(d, dict):
        for k, v in d.items(): yield from leaves(v, f"{pre}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d): yield from leaves(v, f"{pre}[{i}]")
    else: yield pre, d


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return a == b
    if isinstance(a, int) and isinstance(b, int): return a == b
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def env():
    out = {"python": platform.python_version()}
    for m in ("numpy", "scipy", "torch", "recbole"):
        try: out[m] = __import__(m).__version__
        except Exception: out[m] = "not installed"
    return out


print("environment:", json.dumps(env()))
failures = 0
for new, ref in PAIRS:
    try:
        a = {k: v for k, v in leaves(json.load(open(new))) if not any(t in k for t in TIMING)}
    except FileNotFoundError:
        print(f"{new}: MISSING (run the analysis script first)"); failures += 1; continue
    b = {k: v for k, v in leaves(json.load(open(ref))) if not any(t in k for t in TIMING)}
    missing, extra = sorted(b.keys() - a.keys()), sorted(a.keys() - b.keys())
    diff = [k for k in b.keys() & a.keys() if not same(a[k], b[k])]
    exact = sum(1 for k in b.keys() & a.keys() if a[k] == b[k])
    for k in missing[:10]: print(f"  MISSING KEY {new}{k}")
    for k in extra[:10]: print(f"  EXTRA KEY   {new}{k}")
    for k in sorted(diff)[:10]: print(f"  MISMATCH    {new}{k}: new={a[k]} ref={b[k]}")
    n_bad = len(missing) + len(extra) + len(diff)
    print(f"{new}: {len(b)} reference values; {len(b) - len(missing) - len(diff)} match "
          f"({exact} bit-identical); {len(missing)} missing, {len(extra)} extra, {len(diff)} differ")
    failures += n_bad
print("ALL MATCH" if failures == 0 else f"FAILED: {failures} problems")
sys.exit(0 if failures == 0 else 1)
