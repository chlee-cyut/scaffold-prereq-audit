#!/usr/bin/env python3
"""
Cheap improvements + decisive diagnostics:
1) Enlarge candidate pool to N=200 (RL greedy curve compared with post-processing on the same pool)
2) Diagnostic: share of test users' actual next video that is itself a prerequisite violation
3) Theoretical NDCG ceiling at PVR=0 (best achievable NDCG@10 when recommending only non-violating items)
   -> if post-processing already approaches the ceiling, no method (incl. PGPR) can beat it easily -> pivot to empirical study
"""
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

DEV="cuda"; N=200; K=10; TAU=0.5
LAMBDAS=[0.0,0.25,0.5,1.0,2.0,4.0,8.0]

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
def ids2tok(ids): return dataset.id2token(IID,np.asarray(ids))

uid_list=np.asarray(test_data.uid_list)
u2hist=test_data.uid2history_item; u2pos=test_data.uid2positive_item
U=len(uid_list); print(f"Test users: {U:,}  candidate pool N={N}")

SCORE=np.zeros((U,N),np.float32); READY=np.ones((U,N),np.float32)
HASP=np.zeros((U,N),np.float32);  POSIDX=np.full(U,-1,np.int64)
# Diagnostic: whether the real held-out item is a violation
holdout_hasp=np.zeros(U,np.float32); holdout_ready=np.ones(U,np.float32)
holdout_inpool=np.zeros(U,np.float32)

B=4096
for s in range(0,U,B):
    bu=uid_list[s:s+B]
    sc,ids=full_sort_topk(bu,model,test_data,N,device=DEV)
    sc=sc.cpu().numpy(); ids=ids.cpu().numpy()
    for r,uid in enumerate(bu):
        gi=s+r; toks=ids2tok(ids[r]); SCORE[gi]=sc[r]
        h=u2hist[uid]; h=h[h>0].cpu().numpy() if torch.is_tensor(h) else h
        M=set()
        for t in ids2tok(h): M|=vid_pre.get(t,set())
        for j,t in enumerate(toks):
            req=required_of(t)
            if req:
                HASP[gi,j]=1.0; READY[gi,j]=len(req&M)/len(req)
        p=u2pos[uid]; p=p[p>0].cpu().numpy() if torch.is_tensor(p) else p
        postoks=ids2tok(p); posset=set(postoks)
        for j,t in enumerate(toks):
            if t in posset: POSIDX[gi]=j; holdout_inpool[gi]=1.0; break
        # violation status of the held-out item (first pos)
        pt=postoks[0] if len(postoks) else None
        if pt is not None:
            req=required_of(pt)
            if req:
                holdout_hasp[gi]=1.0; holdout_ready[gi]=len(req&M)/len(req)

smin=SCORE.min(1,keepdims=True); smax=SCORE.max(1,keepdims=True)
RREL=(SCORE-smin)/np.clip(smax-smin,1e-6,None)
RREL_t=torch.tensor(RREL,device=DEV); READY_t=torch.tensor(READY,device=DEV)
HASP_t=torch.tensor(HASP,device=DEV); POS=torch.tensor(POSIDX,device=DEV)

def ndcg_pvr(order):
    topk=order[:,:K]; pos=POS.unsqueeze(1)
    match=(topk==pos)&(pos>=0)
    gain=1.0/torch.log2(torch.arange(K,device=DEV).float()+2.0)
    ndcg=(match.float()*gain).sum(1).mean().item()
    rs=torch.gather(READY_t,1,topk); hs=torch.gather(HASP_t,1,topk)
    viol=((rs<TAU)&(hs>0)).float().sum(); tot=(hs>0).float().sum()
    pvr=(viol/torch.clamp(tot,min=1)).item()
    return ndcg,pvr

# greedy trade-off curve (N=200)
print(f"\n=== greedy trade-off curve (N={N}) ===")
print(f"{'λ':>6} | {'NDCG@10':>8} {'PVR@10':>8}")
curve=[]
for lam in LAMBDAS:
    order=torch.argsort(RREL_t+lam*READY_t,dim=1,descending=True)
    nd,pv=ndcg_pvr(order)
    curve.append({"lambda":lam,"ndcg":nd,"pvr":pv})
    print(f"{lam:>6.2f} | {nd:>8.4f} {pv*100:>7.2f}%")

# Post-processing (N=200, hard filter at τ)
BIG=-1e9
masked=RREL_t.clone()
masked[(READY_t<TAU)&(HASP_t>0)]=BIG          # push violating items to the bottom
order_pf=torch.argsort(masked,dim=1,descending=True)
pf_nd,pf_pv=ndcg_pvr(order_pf)
print(f"\nPost-processing (N={N}): NDCG={pf_nd:.4f}  PVR={pf_pv*100:.2f}%")

# Diagnostics
frac_ho_viol = ((holdout_hasp>0)&(holdout_ready<TAU)).mean()
frac_ho_hasp = (holdout_hasp>0).mean()
frac_ho_inpool = holdout_inpool.mean()
# NDCG ceiling at PVR=0: if the held-out item is non-violating it can be ranked 1 -> contributes 1; if violating it must be excluded -> 0
ho_nonviol = ~((holdout_hasp>0)&(holdout_ready<TAU))
ceiling = ho_nonviol.mean()      # upper bound (assumes non-violating held-out item can be ranked first)
print("\n" + "="*56)
print("Decisive diagnostics")
print("="*56)
print(f"Share of held-out items with prerequisite requirements: {frac_ho_hasp*100:.1f}%")
print(f"Share of held-out items in the top-{N} pool:             {frac_ho_inpool*100:.1f}%")
print(f">>> Held-out items that are themselves prerequisite violations: {frac_ho_viol*100:.1f}%")
print(f">>> Theoretical NDCG@10 ceiling at PVR=0:   {ceiling:.4f}")
print(f"    (vs LightGCN NDCG≈0.107, post-processing NDCG={pf_nd:.4f})")

out={"N":N,"tau":TAU,"curve":curve,"postproc":{"ndcg":pf_nd,"pvr":pf_pv},
     "diag":{"holdout_violation_frac":float(frac_ho_viol),
             "holdout_hasprereq_frac":float(frac_ho_hasp),
             "holdout_inpool_frac":float(frac_ho_inpool),
             "pvr0_ndcg_ceiling":float(ceiling)}}
json.dump(out,open(BASE + "/mooccube/scaffold_rl_v2_result.json","w"),indent=2)

# Plot (overlay the ceiling and the old N=50 curve for comparison)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(7.5,5.2))
plt.plot([c["pvr"]*100 for c in curve],[c["ndcg"] for c in curve],"o-",
         color="#2563eb",label=f"SCAFFOLD-RL greedy (N={N})",zorder=3)
for c in curve:
    plt.annotate(f"λ={c['lambda']}",(c["pvr"]*100,c["ndcg"]),
                 textcoords="offset points",xytext=(5,4),fontsize=7,color="#2563eb")
plt.scatter([pf_pv*100],[pf_nd],marker="^",s=120,color="#dc2626",
            label=f"Post-processing (N={N})",zorder=4)
plt.axhline(ceiling,ls="--",color="#16a34a",lw=1.3,
            label=f"PVR=0 NDCG ceiling = {ceiling:.3f}")
plt.axhline(0.1066,ls=":",color="#666",lw=1,label="LightGCN NDCG = 0.107")
plt.xlabel("PVR@10 (%)  ← lower is better"); plt.ylabel("NDCG@10  ↑ higher is better")
plt.title(f"Trade-off with larger pool (N={N}) + PVR=0 ceiling")
plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(BASE + "/mooccube/tradeoff_curve_v2.png",dpi=130)
print("\nSaved scaffold_rl_v2_result.json and tradeoff_curve_v2.png")
