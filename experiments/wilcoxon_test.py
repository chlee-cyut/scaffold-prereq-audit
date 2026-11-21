#!/usr/bin/env python3
"""Wilcoxon signed-rank: post-processing vs LightGCN, per-user paired PVR@10 and NDCG@10. Temporal LOO."""
import os
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")   # root of the data/working-directory layout (see README)
import torch, functools
torch.load = functools.partial(torch.load, weights_only=False)
import numpy as np
for _a,_t in [("float",float),("int",int),("bool",bool),("object",object)]:
    if not hasattr(np,_a): setattr(np,_a,_t)
import json, glob, os, collections, math
from scipy.stats import wilcoxon
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.utils import init_seed
from recbole.utils.case_study import full_sort_topk

DEV="cuda"; N=50; K=10; TAU=0.5
CKPT=max(glob.glob(BASE + "/mooccube/recbole/saved_t/LightGCN-*.pth"),key=os.path.getmtime)
ckpt=torch.load(CKPT,map_location="cpu"); config=ckpt["config"]; config["device"]=DEV
init_seed(config["seed"],config["reproducibility"])
dataset=create_dataset(config)
train_data,valid_data,test_data=data_preparation(config,dataset)
IID=config["ITEM_ID_FIELD"]
model=LightGCN(config,train_data.dataset).to(DEV)
model.load_state_dict(ckpt["state_dict"]); model.restore_user_e=None; model.restore_item_e=None
model.eval()

assets=json.load(open(BASE + "/mooccube/pvr_assets.json"))
vid_pre={k:set(v) for k,v in assets["vid_pre"].items()}
prereqs_of=collections.defaultdict(set)
for A,B in assets["clean_edges"]: prereqs_of[B].add(A)
def required_of(tok):
    r=set()
    for c in vid_pre.get(tok,()): r|=prereqs_of.get(c,set())
    return r

uid_list=np.asarray(test_data.uid_list)
u2hist=test_data.uid2history_item; u2pos=test_data.uid2positive_item
def ids2tok(ids): return dataset.id2token(IID,np.asarray(ids))

# Per user: pvr_raw, pvr_pf (paired, both must have judgeable items); ndcg_raw, ndcg_pf (all users)
pvr_raw=[]; pvr_pf=[]; nd_raw=[]; nd_pf=[]
B=4096
for s in range(0,len(uid_list),B):
    bu=uid_list[s:s+B]
    _,ids=full_sort_topk(bu,model,test_data,N,device=DEV); ids=ids.cpu().numpy()
    for r,uid in enumerate(bu):
        toks=list(ids2tok(ids[r])); reqs=[required_of(t) for t in toks]
        h=u2hist[uid]; h=h[h>0].cpu().numpy() if torch.is_tensor(h) else h
        M=set()
        for t in ids2tok(h): M|=vid_pre.get(t,set())
        p=u2pos[uid]; p=p[p>0].cpu().numpy() if torch.is_tensor(p) else p
        pos=set(ids2tok(p))
        # raw top-K
        raw=toks[:K]
        tot=v=0
        for j in range(K):
            if reqs[j]:
                tot+=1
                if len(reqs[j]&M)/len(reqs[j])<TAU: v+=1
        d=next((1/math.log2(i+2) for i,t in enumerate(raw) if t in pos),0.0)
        nd_raw.append(d)
        # post-processing top-K
        kept=[]
        for j in range(N):
            if reqs[j] and len(reqs[j]&M)/len(reqs[j])<TAU: continue
            kept.append(toks[j])
            if len(kept)==K: break
        tot2=v2=0
        for t in kept:
            rq=required_of(t)
            if rq:
                tot2+=1
                if len(rq&M)/len(rq)<TAU: v2+=1
        d2=next((1/math.log2(i+2) for i,t in enumerate(kept) if t in pos),0.0)
        nd_pf.append(d2)
        if tot>0 and tot2>0:
            pvr_raw.append(v/tot); pvr_pf.append(v2/tot2)

pvr_raw=np.array(pvr_raw); pvr_pf=np.array(pvr_pf)
nd_raw=np.array(nd_raw); nd_pf=np.array(nd_pf)

def report(name, a, b, better_lower):
    # test the a vs b difference; pratt handles zero differences
    diff=b-a
    try:
        stat,pval=wilcoxon(a,b,zero_method="pratt",alternative="two-sided")
    except ValueError as e:
        stat,pval=float("nan"),float("nan")
    print(f"\n[{name}]  n={len(a):,}")
    print(f"  LightGCN mean={a.mean():.4f}  post-proc mean={b.mean():.4f}  Δmean={b.mean()-a.mean():+.4f}")
    print(f"  median diff (post-proc − LightGCN)={np.median(diff):+.4f}")
    print(f"  Wilcoxon signed-rank: W={stat:.1f}  p={pval:.3e}")
    win=int((diff< -1e-12).sum() if better_lower else (diff>1e-12).sum())
    worse=int((diff>1e-12).sum() if better_lower else (diff< -1e-12).sum())
    print(f"  users better with post-proc={win:,}  worse={worse:,}  tied={int((np.abs(diff)<=1e-12).sum()):,}")
    return {"n":len(a),"mean_lightgcn":float(a.mean()),"mean_postproc":float(b.mean()),
            "median_diff":float(np.median(diff)),"W":float(stat),"p":float(pval)}

print("="*60); print("Wilcoxon: post-processing vs LightGCN (temporal LOO, τ=0.5)"); print("="*60)
res={}
res["PVR@10"]=report("PVR@10 (lower is better)", pvr_raw, pvr_pf, better_lower=True)
res["NDCG@10"]=report("NDCG@10 (higher is better; measures accuracy cost)", nd_raw, nd_pf, better_lower=False)
json.dump(res,open(BASE + "/mooccube/wilcoxon_result.json","w"),indent=2)
print("\nSaved wilcoxon_result.json")
