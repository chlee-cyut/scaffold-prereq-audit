#!/usr/bin/env python3
"""(1) shared-denominator ConceptCov (catalog-wide ready concepts as denominator);
(2) Wilcoxon: SASRec (two datasets) + Junyi LightGCN, per-user PVR & NDCG vs post-processing. seed 2020."""
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

def load(path):
    a=json.load(open(path)); vp={k:set(v) for k,v in a["vid_pre"].items()}
    pr=collections.defaultdict(set)
    for A,B in a["clean_edges"]: pr[B].add(A)
    allc=set()
    for v in vp.values(): allc|=v
    return vp,pr,allc

def cfg(ds,seq,seed=2020):
    c={"data_path":BASE + "/mooccube/recbole/dataset/","checkpoint_dir":BASE + "/mooccube/recbole/saved_b2/",
       "load_col":{"inter":["user_id","item_id","timestamp"]},"USER_ID_FIELD":"user_id","ITEM_ID_FIELD":"item_id","TIME_FIELD":"timestamp",
       "user_inter_num_interval":"[5,inf)","item_inter_num_interval":"[5,inf)",
       "eval_args":{"split":{"LS":"valid_and_test"},"order":"TO","mode":"full","group_by":"user"},
       "metrics":["NDCG","Hit"],"topk":[10],"valid_metric":"NDCG@10","epochs":50,
       "train_batch_size":8192,"eval_batch_size":20000000,"stopping_step":5,"device":DEV,"gpu_id":0,
       "seed":seed,"reproducibility":True,"show_progress":False}
    if seq: c.update({"MAX_ITEM_LIST_LENGTH":50,"loss_type":"CE","train_neg_sample_args":None,"train_batch_size":4096,"eval_batch_size":4096})
    return c

def train(ds,model_name,seq):
    config=Config(model=model_name,dataset=ds,config_dict=cfg(ds,seq)); init_seed(2020,True)
    dataset=create_dataset(config); tr,va,te=data_preparation(config,dataset)
    m=get_model(model_name)(config,tr._dataset).to(DEV)
    t=Trainer(config,m); t.fit(tr,va,saved=True,show_progress=False); t.evaluate(te,load_best_model=True,show_progress=False)
    if hasattr(m,"restore_user_e"): m.restore_user_e=None
    if hasattr(m,"restore_item_e"): m.restore_item_e=None
    m.eval(); return m,dataset,te,config

def topk_general(m,dataset,te,IID):
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

def topk_seq(m,dataset,te,config):
    IID=config["ITEM_ID_FIELD"]; ISEQ=IID+config["LIST_SUFFIX"]; ni=dataset.item_num
    def i2t(x): return dataset.id2token(IID,np.asarray(x))
    out=[]; NEG=torch.finfo(torch.float).min
    with torch.no_grad():
        for bd in te:
            inter,hidx,pu,pi=bd; inter=inter.to(DEV)
            sc=m.full_sort_predict(inter).view(-1,ni); sc[:,0]=NEG
            if hidx is not None: sc[hidx]=NEG
            _,ti=torch.topk(sc,N,dim=1); ti=ti.cpu().numpy()
            seq=inter[ISEQ].cpu().numpy(); pos=pi.cpu().numpy()
            for b in range(ti.shape[0]):
                hist=[x for x in seq[b] if x>0]
                out.append((list(i2t(ti[b])),list(i2t(hist)),{i2t([pos[b]])[0]}))
    return out

RES={}
# ---- (1) shared-denominator ConceptCov: LightGCN two datasets ----
for ds,ap in DS.items():
    vp,pr,allc=load(ap)
    def ready_concept(c,M):
        p=pr.get(c,None)
        return True if not p else len(p&M)/len(p)>=TAU
    def NR(toks,M):
        s=set()
        for t in toks:
            for c in vp.get(t,()):
                if c not in M and ready_concept(c,M): s.add(c)
        return s
    m,dataset,te,config=train(ds,"LightGCN",False)
    rows=topk_general(m,dataset,te,config["ITEM_ID_FIELD"])
    cc_pool=cc_glob=n=0
    for toks,hist,pos in rows:
        M=set()
        for t in hist: M|=vp.get(t,set())
        nr_top=NR(toks[:K],M); nr_pool=NR(toks,M)
        nr_glob={c for c in allc if c not in M and ready_concept(c,M)}
        if nr_pool: cc_pool+=len(nr_top)/len(nr_pool)
        if nr_glob: cc_glob+=len(nr_top)/len(nr_glob)
        n+=1
    RES[f"conceptcov_{ds}"]={"pool_denom":cc_pool/n,"global_denom":cc_glob/n,"n":n}
    print(f"[ConceptCov {ds} LightGCN] pool-denom={cc_pool/n:.3f}  global-denom={cc_glob/n:.4f}")

# ---- (2) Wilcoxon: SASRec two datasets + Junyi LightGCN ----
def per_user_pvr_ndcg(rows,vp,pr):
    def req(t):
        r=set()
        for c in vp.get(t,()): r|=pr.get(c,set())
        return r
    praw=[]; ppf=[]; ndr=[]; ndp=[]
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
        d=next((1/math.log2(i+2) for i,t in enumerate(raw) if t in pos),0.0)
        d2=next((1/math.log2(i+2) for i,t in enumerate(kept) if t in pos),0.0)
        ndr.append(d); ndp.append(d2)
        if tot>0 and tot2>0: praw.append(v/tot); ppf.append(v2/tot2)
    return np.array(praw),np.array(ppf),np.array(ndr),np.array(ndp)

for ds,ap in DS.items():
    vp,pr,_=load(ap)
    m,dataset,te,config=train(ds,"SASRec",True)
    rows=topk_seq(m,dataset,te,config)
    pr_,pf_,ndr,ndp=per_user_pvr_ndcg(rows,vp,pr)
    def wil(a,b):
        try: W,p=wilcoxon(a,b,zero_method="pratt"); return float(W),float(p)
        except: return float("nan"),float("nan")
    Wp,pp=wil(pr_,pf_); Wn,pn=wil(ndr,ndp)
    RES[f"wilcoxon_SASRec_{ds}"]={"pvr_raw":float(pr_.mean()),"pvr_pf":float(pf_.mean()),"pvr_W":Wp,"pvr_p":pp,
                                  "ndcg_raw":float(ndr.mean()),"ndcg_pf":float(ndp.mean()),"ndcg_W":Wn,"ndcg_p":pn,"n":len(ndr)}
    print(f"[Wilcoxon SASRec {ds}] PVR {pr_.mean():.3f}->{pf_.mean():.3f} p={pp:.2e} | NDCG {ndr.mean():.4f}->{ndp.mean():.4f} p={pn:.2e}")

# Junyi LightGCN Wilcoxon
vp,pr,_=load(DS["junyi"])
m,dataset,te,config=train("junyi","LightGCN",False)
rows=topk_general(m,dataset,te,config["ITEM_ID_FIELD"])
pr_,pf_,ndr,ndp=per_user_pvr_ndcg(rows,vp,pr)
def wil(a,b):
    try: W,p=wilcoxon(a,b,zero_method="pratt"); return float(W),float(p)
    except: return float("nan"),float("nan")
Wn,pn=wil(ndr,ndp)
RES["wilcoxon_LightGCN_junyi"]={"ndcg_raw":float(ndr.mean()),"ndcg_pf":float(ndp.mean()),"ndcg_W":Wn,"ndcg_p":pn,"n":len(ndr)}
print(f"[Wilcoxon LightGCN junyi] NDCG {ndr.mean():.4f}->{ndp.mean():.4f} p={pn:.2e}")

json.dump(RES,open(BASE + "/paper/revision_batch2_result.json","w"),indent=2)
print("Saved revision_batch2_result.json")
