#!/usr/bin/env python3
"""Revision 1 (R2 Q4): are prerequisite-inconsistent attempts associated with worse performance?

Junyi records correctness. For every learner's FIRST attempt at an exercise that has prerequisites,
compute readiness from the learner's earlier attempts (attempted mastery, as in the paper) and compare
first-attempt correctness for ready vs. unready attempts (tau = 0.5). To control for exercise difficulty,
we also compare within exercises (exercises with >= 30 ready and >= 30 unready first attempts) and
report the attempt-weighted mean within-exercise difference with a sign test over exercises.
Observational evidence only (no causal claim)."""
import os
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")   # root of the data/working-directory layout (see README)
import json, collections, sys, time
from tqdm import tqdm
from scipy.stats import binomtest, wilcoxon
TAU = 0.5
idx2name = {}
for line in open(BASE + "/junyi/vertex_id2idx", encoding="utf-8"):
    line = line.rstrip("\n")
    if line: n, i = line.rsplit(",", 1); idx2name[int(i)] = n
assets = json.load(open(BASE + "/junyi/junyi_pvr_assets.json"))
pre = collections.defaultdict(set)
for A, B in assets["clean_edges"]: pre[B].add(A)

def run(mastery):
    g = {True: [0, 0], False: [0, 0]}; per_ex = collections.defaultdict(lambda: {True: [0, 0], False: [0, 0]})
    for path in (BASE + "/junyi/train.json", BASE + "/junyi/test.json"):
        n_lines = sum(1 for _ in open(path, encoding="utf-8"))
        for line in tqdm(open(path, encoding="utf-8"), total=n_lines, unit="learner", file=sys.stderr, mininterval=1.0,
                         desc=f"  {mastery:9s} mastery: {path.split('/')[-1]}"):
            line = line.strip()
            if not line: continue
            att, cor, seen = set(), set(), set()
            for ex, c in json.loads(line):
                e = idx2name.get(ex)
                if e is None: continue
                if e not in seen and pre.get(e):
                    M = att if mastery == "attempted" else cor
                    ready = len(pre[e] & M) / len(pre[e]) >= TAU
                    g[ready][0] += c; g[ready][1] += 1
                    per_ex[e][ready][0] += c; per_ex[e][ready][1] += 1
                seen.add(e); att.add(e)
                if c == 1: cor.add(e)
    diffs, w = [], []
    for e, d in per_ex.items():
        if d[True][1] >= 30 and d[False][1] >= 30:
            diffs.append(d[True][0] / d[True][1] - d[False][0] / d[False][1]); w.append(d[True][1] + d[False][1])
    pos = sum(x > 0 for x in diffs)
    return {"ready_acc": g[True][0] / g[True][1], "unready_acc": g[False][0] / g[False][1],
            "n_ready": g[True][1], "n_unready": g[False][1], "n_exercises_within": len(diffs),
            "within_weighted_mean_diff": sum(a * b for a, b in zip(diffs, w)) / sum(w),
            "within_median_diff": sorted(diffs)[len(diffs) // 2],
            "exercises_ready_better": pos,
            "sign_test_p": binomtest(pos, len(diffs), 0.5).pvalue,
            "wilcoxon_p": wilcoxon(diffs).pvalue}

T0 = time.time(); out = {}
for i, m in enumerate(("attempted", "correct"), 1):
    print(f"\n=== [{i}/2] mastery = {m}", file=sys.stderr, flush=True); out[m] = run(m)
for m, r in out.items(): print(m, json.dumps(r))
json.dump(out, open(BASE + "/paper/rev1/rev1_junyi_outcome_result.json", "w"), indent=1)
print(f"saved rev1_junyi_outcome_result.json  (total {time.time() - T0:.0f}s)")
