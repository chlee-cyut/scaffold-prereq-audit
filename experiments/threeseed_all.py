#!/usr/bin/env python3
"""All trainable models x two datasets x 3 seeds: HR/NDCG/PVR/post-proc NDCG, reported as mean±std.
BPR/LightGCN use general evaluation; SASRec uses sequential evaluation. Pop/ItemKNN are deterministic (not here)."""
import os
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")   # root of the data/working-directory layout (see README)
import torch, functools
torch.load = functools.partial(torch.load, weights_only=False)
import numpy as np
for _a,_t in [("float",float),("int",int),("bool",bool),("object",object)]:
    if not hasattr(np,_a): setattr(np,_a,_t)
import json, collections, math, statistics
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, get_model
from recbole.utils.case_study import full_sort_topk

DEV="cuda"; N=50; K=10; TAU=0.5; SEEDS=[2020,2021,2022]
DATASETS={"mooccube_csmath_t":BASE + "/mooccube/pvr_assets.json",
          "junyi":BASE + "/junyi/junyi_pvr_assets.json"}

def assets(path):
    a=json.load(open(path)); vp={k:set(v) for k,v in a["vid_pre"].items()}
    pr=collections.defaultdict(set)
    for A,B in a["clean_edges"]: pr[B].add(A)
    return vp,pr

def base_cfg(seq):
    c={"data_path":BASE + "/mooccube/recbole/dataset/","checkpoint_dir":BASE + "/mooccube/recbole/saved_3s/",
       "load_col":{"inter":["user_id","item_id","timestamp"]},
       "USER_ID_FIELD":"user_id","ITEM_ID_FIELD":"item_id","TIME_FIELD":"timestamp",
       "user_inter_num_interval":"[5,inf)","item_inter_num_interval":"[5,inf)",
       "eval_args":{"split":{"LS":"valid_and_test"},"order":"TO","mode":"full","group_by":"user"},
       "metrics":["NDCG","Hit"],"topk":[10],"valid_metric":"NDCG@10",
       "epochs":50,"train_batch_size":8192,"eval_batch_size":20000000,
       "stopping_step":5,"device":DEV,"gpu_id":0,"reproducibility":True,"show_progress":False}
    if seq: c.update({"MAX_ITEM_LIST_LENGTH":50,"loss_type":"CE","train_neg_sample_args":None,
                      "train_batch_size":4096,"eval_batch_size":4096})
    return c

def eval_general(model,dataset,test_data,IID,vp,pr):
    def req(t):
        r=set()
        for c in vp.get(t,()): r|=pr.get(c,set())
        return r
    ul=np.asarray(test_data.uid_list); u2h=test_data.uid2history_item; u2p=test_data.uid2positive_item
    def i2t(x): return dataset.id2token(IID,np.asarray(x))
    tot=v=pf_t=pf_v=0; nd=hr=ndpf=0.0; nu=0; B=4096
    for s in range(0,len(ul),B):
        bu=ul[s:s+B]; _,ids=full_sort_topk(bu,model,test_data,N,device=DEV); ids=ids.cpu().numpy()
        for r,uid in enumerate(bu):
            toks=list(i2t(ids[r])); rq=[req(t) for t in toks]
            h=u2h[uid]; h=h[h>0].cpu().numpy() if torch.is_tensor(h) else h
            M=set()
            for t in i2t(h): M|=vp.get(t,set())
            p=u2p[uid]; p=p[p>0].cpu().numpy() if torch.is_tensor(p) else p
            pos=set(i2t(p)); nu+=1; idcg=sum(1/math.log2(i+2) for i in range(min(K,len(pos))))
            raw=toks[:K]; d=sum(1/math.log2(i+2) for i,t in enumerate(raw) if t in pos)
            if idcg: nd+=d/idcg
            if pos&set(raw): hr+=1
            for j in range(K):
                if rq[j]:
                    tot+=1
                    if len(rq[j]&M)/len(rq[j])<TAU: v+=1
            kept=[]
            for j in range(N):
                if rq[j] and len(rq[j]&M)/len(rq[j])<TAU: continue
                kept.append(toks[j])
                if len(kept)==K: break
            for t in kept:
                q=req(t)
                if q:
                    pf_t+=1
                    if len(q&M)/len(q)<TAU: pf_v+=1
            d=sum(1/math.log2(i+2) for i,t in enumerate(kept) if t in pos)
            if idcg: ndpf+=d/idcg
    return dict(hr=hr/nu,ndcg=nd/nu,pvr=v/max(1,tot),pf_ndcg=ndpf/nu)

def eval_seq(model,dataset,test_data,config,vp,pr):
    def req(t):
        r=set()
        for c in vp.get(t,()): r|=pr.get(c,set())
        return r
    IID=config["ITEM_ID_FIELD"]; ISEQ=IID+config["LIST_SUFFIX"]; n_items=dataset.item_num
    def i2t(x): return dataset.id2token(IID,np.asarray(x))
    tot=v=pf_t=pf_v=0; nd=hr=ndpf=0.0; nu=0; NEG=torch.finfo(torch.float).min
    with torch.no_grad():
        for bd in test_data:
            inter,hist_idx,pu,pi=bd; inter=inter.to(DEV)
            sc=model.full_sort_predict(inter).view(-1,n_items); sc[:,0]=NEG
            if hist_idx is not None: sc[hist_idx]=NEG
            _,ti=torch.topk(sc,N,dim=1); ti=ti.cpu().numpy()
            seq=inter[ISEQ].cpu().numpy(); pos_i=pi.cpu().numpy()
            for b in range(ti.shape[0]):
                toks=list(i2t(ti[b])); rq=[req(t) for t in toks]
                M=set()
                for t in i2t([x for x in seq[b] if x>0]): M|=vp.get(t,set())
                ptok=i2t([pos_i[b]])[0]; nu+=1; raw=toks[:K]
                if ptok in raw: nd+=1/math.log2(raw.index(ptok)+2); hr+=1
                for j in range(K):
                    if rq[j]:
                        tot+=1
                        if len(rq[j]&M)/len(rq[j])<TAU: v+=1
                kept=[]
                for j in range(N):
                    if rq[j] and len(rq[j]&M)/len(rq[j])<TAU: continue
                    kept.append(toks[j])
                    if len(kept)==K: break
                for t in kept:
                    q=req(t)
                    if q:
                        pf_t+=1
                        if len(q&M)/len(q)<TAU: pf_v+=1
                if ptok in kept: ndpf+=1/math.log2(kept.index(ptok)+2)
    return dict(hr=hr/nu,ndcg=nd/nu,pvr=v/max(1,tot),pf_ndcg=ndpf/nu)

def run(ds,ap):
    vp,pr=assets(ap); res={}
    for mname in ["BPR","LightGCN","SASRec"]:
        seq=(mname=="SASRec"); runs=[]
        for sd in SEEDS:
            cfg=base_cfg(seq); cfg["seed"]=sd
            config=Config(model=mname,dataset=ds,config_dict=cfg); init_seed(sd,True)
            dataset=create_dataset(config); tr_d,va_d,te_d=data_preparation(config,dataset)
            model=get_model(mname)(config,tr_d._dataset).to(DEV)
            t=Trainer(config,model); t.fit(tr_d,va_d,saved=True,show_progress=False)
            t.evaluate(te_d,load_best_model=True,show_progress=False)
            if hasattr(model,"restore_user_e"): model.restore_user_e=None
            if hasattr(model,"restore_item_e"): model.restore_item_e=None
            model.eval()
            r=eval_seq(model,dataset,te_d,config,vp,pr) if seq else eval_general(model,dataset,te_d,config["ITEM_ID_FIELD"],vp,pr)
            runs.append(r); print(f"  {ds}/{mname}/seed{sd}: HR{r['hr']:.4f} NDCG{r['ndcg']:.4f} PVR{r['pvr']*100:.1f}%")
        agg={k:(statistics.mean([x[k] for x in runs]),statistics.pstdev([x[k] for x in runs])) for k in runs[0]}
        res[mname]=agg
        print(f">>> {ds}/{mname}: HR {agg['hr'][0]:.4f}±{agg['hr'][1]:.4f} NDCG {agg['ndcg'][0]:.4f}±{agg['ndcg'][1]:.4f} PVR {agg['pvr'][0]*100:.2f}±{agg['pvr'][1]*100:.2f}%")
    return res

OUT={}
for ds,ap in DATASETS.items():
    print(f"=== {ds} ==="); OUT[ds]=run(ds,ap)
json.dump({d:{m:{k:list(v) for k,v in ag.items()} for m,ag in md.items()} for d,md in OUT.items()},
          open(BASE + "/paper/threeseed_result.json","w"),indent=2)
print("Saved threeseed_result.json")
