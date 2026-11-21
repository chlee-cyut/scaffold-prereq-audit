#!/usr/bin/env python3
"""Revision 1 (IEEE Access, Access-2026-42504): analyses requested by Reviewer 2.

For each dataset x model (seed 2020 checkpoints used in the paper; no retraining), take each
test user's top-N (N=50) candidate pool and evaluate re-ranking strategies that trade accuracy
against prerequisite soundness (R2 Q5/Q9), an inferred-mastery signal (R2 Q6), and the
run-time cost of the readiness filter (R2 Q10).

Strategies on the same pool (K=10):
  raw            original top-K
  hard           drop unready items, no backfill (as in the paper; lists may be shorter than K)
  hard+backfill  drop unready items, then fill empty slots with the highest-ranked unready items
  budget-m       allow at most m unready items in the top-K (highest-ranked ones), m=1,2,3
  demote-d       move unready items down by d rank positions (stable sort on rank + d*unready)
Mastery variants:
  attempted      M_u = union of concepts of consumed items (paper definition)
  inferred       M_u+ = M_u plus all transitive prerequisites of concepts in M_u
                 (a learner who has worked on c is assumed to have mastered c's ancestors)
"""
import os
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")   # root of the data/working-directory layout (see README)
import os, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")          # silence TensorFlow start-up messages
warnings.filterwarnings("ignore", category=FutureWarning)
import torch, functools, time, json, math, collections, sys
from tqdm import tqdm
DS_NAME = {"mooccube_csmath_t": "MOOCCube", "junyi": "Junyi"}
def stage(msg): print(f"\n=== {msg}", file=sys.stderr, flush=True)
torch.load = functools.partial(torch.load, weights_only=False)
import numpy as np
for _a, _t in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _a): setattr(np, _a, _t)
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_model

DEV = "cuda:0"; N = 50; K = 10; TAU = 0.5
R = BASE + "/mooccube/recbole/"
ASSETS = {"mooccube_csmath_t": BASE + "/mooccube/pvr_assets.json",
          "junyi": BASE + "/junyi/junyi_pvr_assets.json"}
# Checkpoints written by experiments/revision_metrics.py (saved_rev: Pop, ItemKNN, BPR, LightGCN) and
# experiments/sasrec_run.py (saved_sasrec: SASRec). Both store the two datasets in the same directory, so the
# newest file of each model whose stored config names the dataset is used.
def find_ckpt(ds, model):
    import glob
    sub = "saved_sasrec/" if model == "SASRec" else "saved_rev/"
    for f in sorted(glob.glob(R + sub + model + "-*.pth"), key=os.path.getmtime, reverse=True):
        if torch.load(f, map_location="cpu", weights_only=False)["config"]["dataset"] == ds: return f
    raise FileNotFoundError(f"no {model} checkpoint for {ds} in {R + sub}; run the experiment script first")
CKPT = {(ds, m): None for ds in ASSETS for m in ("Pop", "ItemKNN", "BPR", "LightGCN", "SASRec")}
STRATS = ["raw", "hard", "hard+backfill", "budget-1", "budget-2", "budget-3", "demote-5", "demote-10", "demote-20"]


def closure_map(direct):
    """Transitive prerequisites by reachability (BFS over prerequisite edges).
    Order-independent and well defined on cyclic graphs (a concept on a cycle is its own ancestor)."""
    nodes = set(direct) | {x for v in direct.values() for x in v}
    out = {}
    for c in nodes:
        seen, stack = set(), list(direct.get(c, ()))
        while stack:
            p = stack.pop()
            if p not in seen:
                seen.add(p); stack.extend(direct.get(p, ()))
        out[c] = seen
    return out


def load_graph(path):
    a = json.load(open(path)); vid_pre = {k: set(v) for k, v in a["vid_pre"].items()}
    direct = collections.defaultdict(set)
    for A, B in a["clean_edges"]: direct[B].add(A)
    return vid_pre, direct, closure_map(direct)


def config_for(model, ds):
    cfg = {"data_path": R + "dataset/", "checkpoint_dir": R + "saved_rev1/",
           "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
           "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id", "TIME_FIELD": "timestamp",
           "user_inter_num_interval": "[5,inf)", "item_inter_num_interval": "[5,inf)",
           "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO", "mode": "full", "group_by": "user"},
           "metrics": ["NDCG", "Hit"], "topk": [10], "valid_metric": "NDCG@10",
           "train_batch_size": 8192, "eval_batch_size": 4096, "device": DEV, "gpu_id": 0,
           "seed": 2020, "reproducibility": True, "show_progress": False}
    if model == "SASRec":
        cfg.update({"MAX_ITEM_LIST_LENGTH": 50, "loss_type": "CE", "train_neg_sample_args": None,
                    "train_batch_size": 4096})
    return Config(model=model, dataset=ds, config_dict=cfg)


def rank_pools(model_name, ds):
    """Return list of (pool_tokens[N], true_token, history_tokens) for every test user."""
    config = config_for(model_name, ds); init_seed(2020, True)
    dataset = create_dataset(config); train_data, valid_data, test_data = data_preparation(config, dataset)
    IID = config["ITEM_ID_FIELD"]
    model = get_model(model_name)(config, train_data._dataset).to(DEV)
    if CKPT[(ds, model_name)] is None: CKPT[(ds, model_name)] = find_ckpt(ds, model_name)
    ck = torch.load(CKPT[(ds, model_name)], map_location=DEV)
    model.load_state_dict(ck["state_dict"]); model.load_other_parameter(ck.get("other_parameter")); model.eval()
    tok = lambda ids: dataset.id2token(IID, np.asarray(ids))
    out = []; NEG = torch.finfo(torch.float).min
    seq_field = IID + config["LIST_SUFFIX"] if model_name == "SASRec" else None
    u2hist = None if model_name == "SASRec" else test_data.uid2history_item
    with torch.no_grad():
        for interaction, history_index, positive_u, positive_i in tqdm(test_data, desc=f"  ranking  {DS_NAME[ds]} {model_name}", unit="step", file=sys.stderr, leave=True):
            interaction = interaction.to(DEV)
            if model_name == "SASRec":
                scores = model.full_sort_predict(interaction).view(-1, dataset.item_num)
            else:
                scores = model.full_sort_predict(interaction).view(-1, dataset.item_num)
            scores[:, 0] = NEG
            if history_index is not None: scores[history_index] = NEG
            topi = torch.topk(scores, N, dim=1)[1].cpu().numpy()
            pos = positive_i.cpu().numpy(); uids = interaction[config["USER_ID_FIELD"]].cpu().numpy()
            if model_name == "SASRec": seqs = interaction[seq_field].cpu().numpy()
            for b in range(topi.shape[0]):
                if model_name == "SASRec":     # same history as the paper's SASRec analysis
                    h = [x for x in seqs[b] if x > 0]
                else:
                    h = u2hist[uids[b]]; h = h[h > 0].cpu().numpy() if torch.is_tensor(h) else h
                out.append((list(tok(topi[b])), tok([pos[b]])[0], list(tok(h))))
    return out


def evaluate(pools, vid_pre, direct, trans, desc="  evaluate"):
    req_cache = {}
    def req(t):
        if t not in req_cache:
            r = set()
            for c in vid_pre.get(t, ()): r |= direct.get(c, set())
            req_cache[t] = r
        return req_cache[t]
    res = {}; inc = {"attempted": [0, 0], "inferred": [0, 0]}; t_filter = 0.0; t_e2e = 0.0
    acc = {(m, s): collections.Counter() for m in ("attempted", "inferred") for s in STRATS}
    for pool, true_t, hist in tqdm(pools, desc=desc, unit="user", file=sys.stderr, mininterval=1.0, leave=True):
        te = time.perf_counter()                              # end-to-end timer: mastery set + readiness + budget-1 list
        M = set()
        for t in hist: M |= vid_pre.get(t, set())
        t_e2e += time.perf_counter() - te
        Mp = set(M)
        for c in M: Mp |= trans.get(c, set())
        for mname, MM in (("attempted", M), ("inferred", Mp)):
            def unready(t):
                rq = req(t); return bool(rq) and len(rq & MM) / len(rq) < TAU
            rt = req(true_t)
            if rt:
                inc[mname][1] += 1; inc[mname][0] += unready(true_t)
            t0 = time.perf_counter()
            flags = [unready(t) for t in pool]
            if mname == "attempted":
                t_filter += time.perf_counter() - t0
                t1 = time.perf_counter()                          # budget-1 selection with backfill (Algorithm 2)
                sel = []; nb1 = 0
                for j in range(N):
                    if flags[j]:
                        if nb1 < 1: sel.append(j); nb1 += 1
                    else: sel.append(j)
                    if len(sel) == K: break
                if len(sel) < K: sel += [j for j in range(N) if flags[j] and j not in sel][:K - len(sel)]
                t_e2e += (t1 - t0) + (time.perf_counter() - t1)
            ready_idx = [j for j in range(N) if not flags[j]]
            bad_idx = [j for j in range(N) if flags[j]]
            lists = {"raw": list(range(K)), "hard": ready_idx[:K]}
            lists["hard+backfill"] = sorted(ready_idx[:K] + bad_idx[:max(0, K - len(ready_idx[:K]))])
            for m in (1, 2, 3):
                chosen = []; nb = 0
                for j in range(N):
                    if flags[j]:
                        if nb < m: chosen.append(j); nb += 1
                    else: chosen.append(j)
                    if len(chosen) == K: break
                if len(chosen) < K: chosen += [j for j in bad_idx if j not in chosen][:K - len(chosen)]
                lists[f"budget-{m}"] = sorted(chosen)
            for d in (5, 10, 20):
                lists[f"demote-{d}"] = sorted(range(N), key=lambda j: (j + d * flags[j], j))[:K]
            for s, idx in lists.items():
                a = acc[(mname, s)]; toks = [pool[j] for j in idx]
                a["users"] += 1; a["len"] += len(toks)
                if true_t in toks:
                    r = toks.index(true_t); a["hit"] += 1; a["ndcg"] += 1 / math.log2(r + 2)
                for j in idx:
                    if req(pool[j]): a["judge"] += 1; a["viol"] += flags[j]
    for (mname, s), a in acc.items():
        res[f"{mname}|{s}"] = {"HR@10": a["hit"] / a["users"], "NDCG@10": a["ndcg"] / a["users"],
                               "PVR@10": a["viol"] / max(1, a["judge"]), "avg_len": a["len"] / a["users"]}
    res["true_item_inconsistency"] = {k: v[0] / max(1, v[1]) for k, v in inc.items()}
    res["filter_us_per_user"] = 1e6 * t_filter / len(pools)          # readiness checks of the N-item pool only
    res["rerank_us_per_user"] = 1e6 * t_e2e / len(pools)             # mastery set + readiness + budget-1 list
    res["n_users"] = len(pools)
    return res


if __name__ == "__main__":
    import os
    RES = BASE + "/paper/rev1/rev1_mitigation_result.json"
    JOB = [0]; T_START = time.time()
    out = json.load(open(RES)) if ("--resume" in sys.argv and os.path.exists(RES)) else {}   # fresh run unless --resume
    for ds, ap in ASSETS.items():
        vid_pre, direct, trans = load_graph(ap)
        for mn in ("BPR", "LightGCN", "SASRec"):
            if f"{ds}|{mn}" in out: continue
            JOB[0] += 1; stage(f"[{JOB[0]}/{len(ASSETS) * 3}] {DS_NAME[ds]} · {mn}: loading checkpoint and ranking top-{N}")
            t0 = time.time(); pools = rank_pools(mn, ds)
            r = evaluate(pools, vid_pre, direct, trans, desc=f"  evaluate {DS_NAME[ds]} {mn}"); out[f"{ds}|{mn}"] = r
            raw, hard = r["attempted|raw"], r["attempted|hard"]
            print(f"[{ds} {mn}] users={r['n_users']} raw NDCG={raw['NDCG@10']:.4f} PVR={raw['PVR@10']:.3f} | "
                  f"hard NDCG={hard['NDCG@10']:.4f} | inc={r['true_item_inconsistency']} | "
                  f"filter {r['filter_us_per_user']:.1f} us/user | {time.time() - t0:.0f}s", flush=True)
            json.dump(out, open(BASE + "/paper/rev1/rev1_mitigation_result.json", "w"), indent=1)
    print(f"saved rev1_mitigation_result.json  (total {time.time() - T_START:.0f}s)")
