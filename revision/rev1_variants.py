#!/usr/bin/env python3
"""Revision 1: graph-provenance and mastery-inference variants (R1 graph-audit point; R2 Q6).

Same seed-2020 checkpoints and top-50 pools as rev1_mitigation.py. For each dataset x model:
  graph   clean (paper; MOOCCube 803 audited edges / Junyi expert DAG) or raw (MOOCCube 905 golden edges)
  mastery attempted  M = concepts of consumed items (paper)
          1hop       M plus the direct prerequisites of concepts in M
          closure    M plus all transitive prerequisites of concepts in M
Reports PVR@10 (raw list), NDCG@10 raw / hard filter / budget-1, and the inconsistency of the true next item.
"""
import os
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")   # root of the data/working-directory layout (see README)
import csv, json, math, collections, sys, time
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev1_mitigation import load_graph, rank_pools, closure_map, stage, DS_NAME, ASSETS, N, K, TAU

RAW_EDGES = BASE + "/mooccube/prereq_edges_audit.csv"


def graph_from_edges(edges):
    direct = collections.defaultdict(set)
    for A, B in edges: direct[B].add(A)
    return direct, closure_map(direct)


def evaluate(pools, vid_pre, direct, trans, desc="  evaluate"):
    rc = {}
    def req(t):
        if t not in rc:
            r = set()
            for c in vid_pre.get(t, ()): r |= direct.get(c, set())
            rc[t] = r
        return rc[t]
    acc = {m: collections.Counter() for m in ("attempted", "1hop", "closure")}
    tr = collections.Counter()          # transitive-prerequisite PVR under attempted mastery (Table 9)
    rqt = {}
    def req_t(t):
        if t not in rqt:
            r = set()
            for c in vid_pre.get(t, ()): r |= trans.get(c, set())
            rqt[t] = r
        return rqt[t]
    for pool, true_t, hist in tqdm(pools, desc=desc, unit="user", file=sys.stderr, mininterval=1.0, leave=True):
        M = set()
        for t in hist: M |= vid_pre.get(t, set())
        M1 = set(M)
        for c in M: M1 |= direct.get(c, set())
        Mc = set(M)
        for c in M: Mc |= trans.get(c, set())
        for t in pool[:K]:
            r = req_t(t)
            if r: tr["judge"] += 1; tr["viol"] += len(r & M) / len(r) < TAU
        for m, MM in (("attempted", M), ("1hop", M1), ("closure", Mc)):
            a = acc[m]
            un = lambda t: bool(req(t)) and len(req(t) & MM) / len(req(t)) < TAU
            flags = [un(t) for t in pool]
            if req(true_t): a["tj"] += 1; a["tv"] += un(true_t)
            a["u"] += 1
            for j in range(K):
                if req(pool[j]): a["judge"] += 1; a["viol"] += flags[j]
            def score(idx, key):
                toks = [pool[j] for j in idx]
                if true_t in toks: a[key] += 1 / math.log2(toks.index(true_t) + 2)
            score(range(K), "ndcg_raw")
            score([j for j in range(N) if not flags[j]][:K], "ndcg_hard")
            ch = []; nb = 0
            for j in range(N):
                if flags[j]:
                    if nb < 1: ch.append(j); nb += 1
                else: ch.append(j)
                if len(ch) == K: break
            score(ch, "ndcg_b1")
    out = {}
    for m, a in acc.items():
        out[m] = {"PVR@10": a["viol"] / max(1, a["judge"]), "true_inconsistency": a["tv"] / max(1, a["tj"]),
                  "true_inconsistency_all_users": a["tv"] / a["u"],
                  "NDCG_raw": a["ndcg_raw"] / a["u"], "NDCG_hard": a["ndcg_hard"] / a["u"], "NDCG_budget1": a["ndcg_b1"] / a["u"]}
    out["PVR_transitive_attempted"] = tr["viol"] / max(1, tr["judge"])
    return out


if __name__ == "__main__":
    res = {}; JOB = [0]; T_START = time.time()
    raw_edges = [(r["prereq_A"], r["dependent_B"]) for r in csv.DictReader(open(RAW_EDGES, encoding="utf-8-sig"))]
    for ds, ap in ASSETS.items():
        vid_pre, direct, trans = load_graph(ap)
        graphs = {"clean": (direct, trans)}
        if ds.startswith("mooccube"): graphs["raw"] = graph_from_edges(raw_edges)
        for mn in ("Pop", "ItemKNN", "BPR", "LightGCN", "SASRec"):
            JOB[0] += 1; stage(f"[{JOB[0]}/{len(ASSETS) * 5}] {DS_NAME[ds]} · {mn}: loading checkpoint and ranking top-{N}")
            t0 = time.time(); pools = rank_pools(mn, ds)
            for g, (d_, t_) in graphs.items():
                r = evaluate(pools, vid_pre, d_, t_, desc=f"  evaluate {DS_NAME[ds]} {mn} ({g} graph)"); res[f"{ds}|{mn}|{g}"] = r
                print(f"[{ds} {mn} {g}] " + f"PVR_trans={r['PVR_transitive_attempted']*100:.1f}% | " + " | ".join(
                    f"{m}: PVR={v['PVR@10']*100:.1f}% inc={v['true_inconsistency']*100:.1f}% "
                    f"NDCG {v['NDCG_raw']:.4f}->hard {v['NDCG_hard']:.4f}/b1 {v['NDCG_budget1']:.4f}" for m, v in r.items() if isinstance(v, dict)),
                    flush=True)
            json.dump(res, open(BASE + "/paper/rev1/rev1_variants_result.json", "w"), indent=1)
            print(f"   ({time.time()-t0:.0f}s)", flush=True)
    print(f"saved rev1_variants_result.json  (total {time.time() - T_START:.0f}s)")
