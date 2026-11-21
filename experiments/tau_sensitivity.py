#!/usr/bin/env python3
"""τ sensitivity: how LightGCN's PVR@10, post-processing NDCG@10, and the 39% ceiling vary with τ. Temporal LOO."""
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
from recbole.utils import init_seed
from recbole.utils.case_study import full_sort_topk

DEV="cuda"; N=50; K=10
TAUS=[0.3,0.4,0.5,0.6,0.7,0.8,1.0]
NOPRE=2.0   # sentinel for no prerequisite requirement (never < τ≤1)

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
U=len(uid_list)

# Collect: per-candidate ready-ratio (NOPRE = no prerequisites), pos position, holdout ready/haspre
READ=np.full((U,N),NOPRE,np.float32); POS=np.full(U,-1,np.int64)
HO_READ=np.full(U,NOPRE,np.float32)
B=4096
for s in range(0,U,B):
    bu=uid_list[s:s+B]
    _,ids=full_sort_topk(bu,model,test_data,N,device=DEV); ids=ids.cpu().numpy()
    for r,uid in enumerate(bu):
        gi=s+r; toks=list(ids2tok(ids[r]))
        h=u2hist[uid]; h=h[h>0].cpu().numpy() if torch.is_tensor(h) else h
        M=set()
        for t in ids2tok(h): M|=vid_pre.get(t,set())
        for j,t in enumerate(toks):
            req=required_of(t)
            if req: READ[gi,j]=len(req&M)/len(req)
        p=u2pos[uid]; p=p[p>0].cpu().numpy() if torch.is_tensor(p) else p
        postoks=ids2tok(p); posset=set(postoks)
        for j,t in enumerate(toks):
            if t in posset: POS[gi]=j; break
        if len(postoks):
            req=required_of(postoks[0])
            if req: HO_READ[gi]=len(req&M)/len(req)

gain=np.array([1/math.log2(i+2) for i in range(K)])
def eval_tau(tau):
    top=READ[:,:K]; hasp=top<NOPRE
    viol=(top<tau)&hasp
    pvr=viol.sum()/max(1,hasp.sum())
    # Post-processing: per user drop prerequisite items with READ<tau, take top K, compute NDCG (single pos)
    ndcg_raw=0.0; ndcg_pf=0.0
    for u in range(U):
        pos=POS[u]
        if pos>=0 and pos<K: ndcg_raw+=gain[pos]
        # post-processing keep mask
        keep=~((READ[u]<tau)&(READ[u]<NOPRE))
        kept_idx=np.nonzero(keep)[0][:K]
        rank=np.nonzero(kept_idx==pos)[0]
        if pos>=0 and len(rank): ndcg_pf+=gain[rank[0]]
    ndcg_raw/=U; ndcg_pf/=U
    ho_viol=((HO_READ<tau)&(HO_READ<NOPRE)).mean()
    return dict(tau=tau,pvr=float(pvr),ndcg_raw=ndcg_raw,ndcg_pf=ndcg_pf,
                ndcg_drop=(ndcg_pf-ndcg_raw)/ndcg_raw*100,holdout_viol=float(ho_viol))

rows=[eval_tau(t) for t in TAUS]
print("="*72)
print("τ sensitivity (LightGCN, temporal LOO)")
print("="*72)
print(f"{'τ':>5} | {'PVR@10':>8} | {'PP_NDCG':>10} {'Drop':>7} | {'HOViol(ceiling)':>18}")
for r in rows:
    print(f"{r['tau']:>5.1f} | {r['pvr']*100:>7.2f}% | {r['ndcg_pf']:>10.4f} {r['ndcg_drop']:>6.1f}% | {r['holdout_viol']*100:>16.1f}%")
json.dump(rows,open(BASE + "/mooccube/tau_sensitivity_result.json","w"),indent=2)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
t=[r['tau'] for r in rows]
ax[0].plot(t,[r['pvr']*100 for r in rows],"o-",label="LightGCN PVR@10")
ax[0].plot(t,[r['holdout_viol']*100 for r in rows],"s--",color="#dc2626",label="Held-out item violation rate (ceiling driver)")
ax[0].set_xlabel("τ (prerequisite readiness threshold)"); ax[0].set_ylabel("%"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title("PVR and behavioral violation rate vs τ")
ax[1].plot(t,[r['ndcg_raw'] for r in rows],"o-",label="LightGCN NDCG@10")
ax[1].plot(t,[r['ndcg_pf'] for r in rows],"^-",color="#dc2626",label="NDCG@10 after post-processing")
ax[1].set_xlabel("τ"); ax[1].set_ylabel("NDCG@10"); ax[1].legend(); ax[1].grid(alpha=.3)
ax[1].set_title("Post-processing accuracy cost vs τ")
plt.tight_layout(); plt.savefig(BASE + "/mooccube/tau_sensitivity.png",dpi=130)
print("\nSaved tau_sensitivity_result.json and tau_sensitivity.png")
