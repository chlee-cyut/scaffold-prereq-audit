#!/usr/bin/env python3
"""Paper-precision check for a full re-run.

`compare_results.py` and `compare_archive.py` are strict: they report any difference beyond 1e-9. After re-training,
small GPU-related differences are expected. This script lists every value that differs from the archived result
and checks whether it still rounds to the same number at the precision printed in the manuscript:
  * PVR values: percent with 2 decimals for the three-seed results (Table 4) and 1 decimal for the revision
    analyses (Tables 11, 13, 14 print 1 decimal); a value that agrees at the printed precision but not at 2
    decimals is listed as NOTE;
  * NDCG / HR values and their standard deviations: 4 decimals (Table 4);
  * NDCG changes in Tables 13 and 14: percent with 1 decimal, recomputed from the re-run values;
  * PVR standard deviations: percent with 2 decimals.
Values that do not appear in the manuscript are checked with the same rules. Exit code 0 if every differing value
rounds identically.
Usage: SCAFFOLD_ROOT=<re-run root> python3 verification/paper_precision_check.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.join(HERE, "..", "results")
REF = os.path.join(HERE, "..", "revision", "reference")
NEW = os.environ.get("SCAFFOLD_ROOT", "/workspace")
PAIRS = [(os.path.join(ARCH, "paper", "threeseed_result.json"), "paper/threeseed_result.json"),
         (os.path.join(ARCH, "mooccube", "seeds_result.json"), "mooccube/seeds_result.json"),
         (os.path.join(ARCH, "mooccube", "sasrec_result.json"), "mooccube/sasrec_result.json"),
         (os.path.join(REF, "reference_rev1_mitigation_result.json"), "paper/rev1/rev1_mitigation_result.json"),
         (os.path.join(REF, "reference_rev1_variants_result.json"), "paper/rev1/rev1_variants_result.json")]
SKIP = ("_us_per_user", "checkpoint")


def leaves(d, pre=""):
    if isinstance(d, dict):
        for k, v in d.items(): yield from leaves(v, f"{pre}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d): yield from leaves(v, f"{pre}[{i}]")
    else: yield pre, d


def rule(key, rel):
    k = key.lower()
    if "pvr" in k or "inconsist" in k or "viol" in k:
        if rel.startswith("paper/rev1/"): return "PVR % (1 dp, as printed)", lambda x: round(100 * x, 1)
        return "PVR % (2 dp, as printed)", lambda x: round(100 * x, 2)
    return "4 dp", lambda x: round(x, 4)


n_diff = n_bad = 0; n_note = [0]
for arch_path, rel in PAIRS:
    a = dict(leaves(json.load(open(arch_path)))); b = dict(leaves(json.load(open(os.path.join(NEW, rel)))))
    for key, va in a.items():
        vb = b.get(key)
        if any(s in key for s in SKIP) or not isinstance(va, float) or not isinstance(vb, float) or va == vb: continue
        n_diff += 1; name, r = rule(key, rel); ok = r(va) == r(vb)
        n_bad += not ok
        strict = round(100 * va, 2) == round(100 * vb, 2) if "PVR %" in name else True
        tag = "FAIL" if not ok else ("NOTE" if not strict else "OK  ")
        n_note[0] += ok and not strict
        print(f"{tag} {rel}{key}: archived={va:.8f} re-run={vb:.8f} | {name}: {r(va)} vs {r(vb)}"
              + ("" if strict else f" (2 dp: {round(100 * va, 2)} vs {round(100 * vb, 2)})"))
    # NDCG change of each strategy relative to the unfiltered list (Tables 13 and 14), percent with 1 decimal
    for key, va in a.items():
        if key.endswith("NDCG@10") and "|raw/" not in key and "|" in key:
            base = key.rsplit("|", 1)[0] + "|raw/NDCG@10"
            if base in a and base in b and key in b:
                ca, cb = round(100 * (va / a[base] - 1), 1), round(100 * (b[key] / b[base] - 1), 1)
                if (va, a[base]) != (b[key], b[base]):
                    n_diff += 1; ok = ca == cb; n_bad += not ok
                    print(f"{'OK  ' if ok else 'FAIL'} {rel}{key} (change vs raw): {ca}% vs {cb}%")
print(f"\n{n_diff} differing values or derived changes checked; {n_diff - n_bad} round to the same number at the printed "
      f"precision, {n_bad} do not; {n_note[0]} of them differ at 2 decimals (NOTE)")
sys.exit(0 if n_bad == 0 else 1)
