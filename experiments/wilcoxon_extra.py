#!/usr/bin/env python3
"""Wilcoxon only: SASRec (two datasets) + Junyi LightGCN, per-user PVR & NDCG (raw vs post-processing). seed 2020."""
import os
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")   # root of the data/working-directory layout (see README)
import torch, functools
torch.load = functools.partial(torch.load, weights_only=False)
import numpy as np
for _a,_t in [("float",float),("int",int),("bool",bool),("object",object)]:
    if not hasattr(np,_a): setattr(np,_a,_t)
import json, collections, math
from scipy.stats import wilcoxon
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, get_model
from recbole.utils.case_study import full_sort_topk

DEV="cuda"; N=50; K=10; TAU=0.5
DS={"mooccube_csmath_t":BASE + "/mooccube/pvr_assets.json","junyi":BASE + "/junyi/junyi_pvr_assets.json"}
def load(p):
    a=json.load(open(p)); vp={k:set(v) for k,v in a["vid_pre"].items()}
    pr=collections.defaultdict(set)
    for A,B in a["clean_edges"]: pr[B].add(A)
    return vp,pr
def cfg(ds,seq):
    c={"data_path":BASE + "/mooccube/recbole/dataset/","checkpoint_dir":BASE + "/mooccube/recbole/saved_wx/",
       "load_col":{"inter":["user_id","item_id","timestamp"]},"USER_ID_FIELD":"user_id","ITEM_ID_FIELD":"item_id","TIME_FIELD":"timestamp",
       "user_inter_num_interval":"[5,inf)","item_inter_num_interval":"[5,inf)",
       "eval_args":{"split":{"LS":"valid_and_test"},"order":"TO","mode":"full","group_by":"user"},
       "metrics":["NDCG"],"topk":[10],"valid_metric":"NDCG@10","epochs":50,"train_batch_size":8192,
       "eval_batch_size":20000000,"stopping_step":5,"device":DEV,"gpu_id":0,"seed":2020,"reproducibility":True,"show_progress":False}
    if seq: c.update({"MAX_ITEM_LIST_LENGTH":50,"loss_type":"CE","train_neg_sample_args":None,"train_batch_size":4096,"eval_batch_size":4096})
    return c
def train(ds,mn,seq):
    config=Config(model=mn,dataset=ds,config_dict=cfg(ds,seq)); init_seed(2020,True)
    dataset=create_dataset(config); tr,va,te=data_preparation(config,dataset)
    m=get_model(mn)(config,tr._dataset).to(DEV); t=Trainer(config,m)
    t.fit(tr,va,saved=True,show_progress=False); t.evaluate(te,load_best_model=True,show_progress=False)
    if hasattr(m,"restore_user_e"): m.restore_user_e=None
    if hasattr(m,"restore_item_e"): m.restore_item_e=None
    m.eval(); return m,dataset,te,config
def rows_general(m,dataset,te,IID):
    ul=np.asarray(te.uid_list); u2h=te.uid2history_item; u2p=te.uid2positive_item
    def i2t(x): return dataset.id2token(IID,np.asarray(x))
    out=[]; B=4096
    for s in range(0,len(ul),B):
        bu=ul[s:s+B]; _,ids=full_sort_topk(bu,m,te,N,device=DEV); ids=ids.cpu().numpy()
        for r,uid in enumerate(bu):
            h=u2h[uid]; h=h[h>0].cpu().numpy() if torch.is_tensor(h) else h
            p=u2p[uid]; p=p[p>0].cpu().numpy() if torch.is_tensor(p) else p
            out.append((list(i2t(ids[r])),list(i2t(h)),set(i2t(p))))
    return out
def rows_seq(m,dataset,te,config):
    IID=config["ITEM_ID_FIELD"]; ISEQ=IID+config["LIST_SUFFIX"]; ni=dataset.item_num
    def i2t(x): return dataset.id2token(IID,np.asarray(x))
    out=[]; NEG=torch.finfo(torch.float).min
    with torch.no_grad():
        for bd in te:
            inter,hidx,pu,pi=bd; inter=inter.to(DEV)
            sc=m.full_sort_predict(inter).view(-1,ni); sc[:,0]=NEG
            if hidx is not None: sc[hidx]=NEG
            _,ti=torch.topk(sc,N,dim=1); ti=ti.cpu().numpy(); seq=inter[ISEQ].cpu().numpy(); pos=pi.cpu().numpy()
            for b in range(ti.shape[0]):
                out.append((list(i2t(ti[b])),list(i2t([x for x in seq[b] if x>0])),{i2t([pos[b]])[0]}))
    return out
def pu(rows,vp,pr):
    def req(t):
        r=set()
        for c in vp.get(t,()): r|=pr.get(c,set())
        return r
    pr_r=[]; pr_f=[]; ndr=[]; ndp=[]
    for toks,hist,pos in rows:
        M=set()
        for t in hist: M|=vp.get(t,set())
        raw=toks[:K]; tot=v=0
        for t in raw:
            q=req(t)
            if q:
                tot+=1
                if len(q&M)/len(q)<TAU: v+=1
        kept=[]
        for t in toks:
            q=req(t)
            if q and len(q&M)/len(q)<TAU: continue
            kept.append(t)
            if len(kept)==K: break
        tot2=v2=0
        for t in kept:
            q=req(t)
            if q:
                tot2+=1
                if len(q&M)/len(q)<TAU: v2+=1
        ndr.append(next((1/math.log2(i+2) for i,t in enumerate(raw) if t in pos),0.0))
        ndp.append(next((1/math.log2(i+2) for i,t in enumerate(kept) if t in pos),0.0))
        if tot>0 and tot2>0: pr_r.append(v/tot); pr_f.append(v2/tot2)
    return map(np.array,(pr_r,pr_f,ndr,ndp))
def wil(a,b):
    try: W,p=wilcoxon(a,b,zero_method="pratt"); return float(W),float(p)
    except Exception as e: return float("nan"),float("nan")

RES={}
for ds,ap in DS.items():
    vp,pr=load(ap); m,dataset,te,config=train(ds,"SASRec",True)
    prr,prf,ndr,ndp=pu(rows_seq(m,dataset,te,config),vp,pr)
    Wp,pp=wil(prr,prf); Wn,pn=wil(ndr,ndp)
    RES[f"SASRec_{ds}"]={"pvr_raw":float(prr.mean()),"pvr_pf":float(prf.mean()),"pvr_p":pp,
                         "ndcg_raw":float(ndr.mean()),"ndcg_pf":float(ndp.mean()),"ndcg_W":Wn,"ndcg_p":pn,"n":len(ndr)}
    print(f"[SASRec {ds}] PVR {prr.mean():.3f}->{prf.mean():.3f} p={pp:.2e} | NDCG {ndr.mean():.4f}->{ndp.mean():.4f} W={Wn:.0f} p={pn:.2e}")
vp,pr=load(DS["junyi"]); m,dataset,te,config=train("junyi","LightGCN",False)
prr,prf,ndr,ndp=pu(rows_general(m,dataset,te,config["ITEM_ID_FIELD"]),vp,pr)
Wn,pn=wil(ndr,ndp); Wp,pp=wil(prr,prf)
RES["LightGCN_junyi"]={"pvr_raw":float(prr.mean()),"pvr_pf":float(prf.mean()),"pvr_p":pp,
                       "ndcg_raw":float(ndr.mean()),"ndcg_pf":float(ndp.mean()),"ndcg_W":Wn,"ndcg_p":pn,"n":len(ndr)}
print(f"[LightGCN junyi] PVR {prr.mean():.3f}->{prf.mean():.3f} p={pp:.2e} | NDCG {ndr.mean():.4f}->{ndp.mean():.4f} W={Wn:.0f} p={pn:.2e}")
json.dump(RES,open(BASE + "/paper/wilcoxon_extra_result.json","w"),indent=2)
print("Saved wilcoxon_extra_result.json")
