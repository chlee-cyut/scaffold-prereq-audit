#!/usr/bin/env python3
"""G2/G3 rigorous version: temporal LOO checkpoint, post-processing comparison with consistent τ."""
import os
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")   # root of the data/working-directory layout (see README)
import torch, functools
torch.load = functools.partial(torch.load, weights_only=False)
import numpy as np
for _a,_t in [("float",float),("int",int),("bool",bool),("object",object)]:
    if not hasattr(np,_a): setattr(np,_a,_t)
import json, glob, os, collections, math
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.trainer import Trainer
from recbole.utils import init_seed
from recbole.utils.case_study import full_sort_topk

CKPT = max(glob.glob(BASE + "/mooccube/recbole/saved_t/LightGCN-*.pth"), key=os.path.getmtime)
print("checkpoint:", CKPT)
ckpt = torch.load(CKPT, map_location="cpu")
config = ckpt["config"]; config["device"] = "cuda"
init_seed(config["seed"], config["reproducibility"])
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)
device = config["device"]; IID = config["ITEM_ID_FIELD"]
model = LightGCN(config, train_data.dataset).to(device)
model.load_state_dict(ckpt["state_dict"]); model.restore_user_e = None; model.restore_item_e = None
model.eval()

trainer = Trainer(config, model)
test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
print("\n=== G2 accuracy (temporal LOO, full ranking) ===")
for k, v in test_result.items(): print(f"  {k}: {v}")

assets = json.load(open(BASE + "/mooccube/pvr_assets.json"))
vid_pre = {k: set(v) for k, v in assets["vid_pre"].items()}
prereqs_of = collections.defaultdict(set)
for A, B in assets["clean_edges"]: prereqs_of[B].add(A)
def required_of(tok):
    r = set()
    for c in vid_pre.get(tok, ()): r |= prereqs_of.get(c, set())
    return r

uid_list = np.asarray(test_data.uid_list)
u2hist = test_data.uid2history_item; u2pos = test_data.uid2positive_item
def ids2tok(ids): return dataset.id2token(IID, np.asarray(ids))
print(f"Test users: {len(uid_list):,}")

K, POOL, BATCH = 10, 50, 4096
TAUS = [1.0, 0.5]
# Per τ: raw violations, judgeable count, pf violations, pf judgeable, pf NDCG
stat = {t: dict(tot=0, viol=0, pf_tot=0, pf_viol=0, ndcg_pf=0.0) for t in TAUS}
ndcg_raw = 0.0; hr_raw = 0; nuser = 0

for s in range(0, len(uid_list), BATCH):
    bu = uid_list[s:s+BATCH]
    _, topk = full_sort_topk(bu, model, test_data, POOL, device=device)
    topk = topk.cpu().numpy()
    for row, uid in enumerate(bu):
        toks = ids2tok(topk[row]); reqs = [required_of(t) for t in toks]
        h = u2hist[uid]; h = h[h > 0].cpu().numpy() if torch.is_tensor(h) else h
        M = set()
        for t in ids2tok(h): M |= vid_pre.get(t, set())
        p = u2pos[uid]; p = p[p > 0].cpu().numpy() if torch.is_tensor(p) else p
        pos = set(ids2tok(p)); nuser += 1
        idcg = sum(1/math.log2(i+2) for i in range(min(K, len(pos))))
        ready = [ (len(reqs[j] & M)/len(reqs[j]) if reqs[j] else None) for j in range(POOL) ]
        # Raw top-K accuracy (independent of τ)
        raw = list(toks[:K])
        dcg = sum(1/math.log2(i+2) for i,t in enumerate(raw) if t in pos)
        if idcg: ndcg_raw += dcg/idcg
        if pos & set(raw): hr_raw += 1
        for t in TAUS:
            st = stat[t]
            # raw PVR@K at τ
            for j in range(K):
                if ready[j] is not None:
                    st["tot"] += 1
                    if ready[j] < t: st["viol"] += 1
            # Post-processing: filter with the same τ
            kept = []
            for j in range(POOL):
                if ready[j] is not None and ready[j] < t: continue
                kept.append(toks[j])
                if len(kept) == K: break
            for j2, tk in enumerate(kept):
                rq = required_of(tk)
                if rq:
                    st["pf_tot"] += 1
                    if len(rq & M)/len(rq) < t: st["pf_viol"] += 1
            dcg = sum(1/math.log2(i+2) for i,tk in enumerate(kept) if tk in pos)
            if idcg: st["ndcg_pf"] += dcg/idcg

nd_raw = ndcg_raw/max(1,nuser); hr = hr_raw/max(1,nuser)
print("\n" + "="*60)
print(f"G2 model PVR + G3 post-processing (consistent τ)  own NDCG@{K}={nd_raw:.4f} HR@{K}={hr:.4f}")
print("="*60)
out = {"test_result":{k:float(v) for k,v in test_result.items()},
       "ndcg_self":nd_raw,"hr_self":hr,"n_users":nuser,"checkpoint":CKPT,"tau":{}}
for t in TAUS:
    st = stat[t]
    raw_pvr = st["viol"]/max(1,st["tot"])
    pf_pvr = st["pf_viol"]/max(1,st["pf_tot"])
    nd_pf = st["ndcg_pf"]/max(1,nuser)
    drop = (nd_pf-nd_raw)/max(1e-9,nd_raw)*100
    label = "strict(Ready<1)" if t==1.0 else f"τ={t}"
    print(f"\n[{label}]")
    print(f"  LightGCN PVR@{K}   = {raw_pvr*100:.2f}%")
    print(f"  Post-proc PVR@{K}  = {pf_pvr*100:.2f}%")
    print(f"  NDCG@{K}: {nd_raw:.4f} -> {nd_pf:.4f}  ({drop:+.1f}%)  <- accuracy cost of removing PVR via post-processing")
    out["tau"][str(t)] = {"raw_pvr":raw_pvr,"pf_pvr":pf_pvr,"pf_ndcg":nd_pf,"ndcg_drop_pct":drop}
json.dump(out, open(BASE + "/mooccube/g2g3_t_result.json","w"), indent=2)
print("\nSaved g2g3_t_result.json")
