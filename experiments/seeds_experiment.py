#!/usr/bin/env python3
"""3-seed robustness: BPR/LightGCN report mean±std; Pop/ItemKNN are deterministic, run once.
For each (model,seed) compute HR@10, NDCG@10, PVR@10 (τ=.5), post-proc NDCG@10 + PVR. Temporal LOO."""
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

DEV="cuda"; N=50; K=10; TAU=0.5
PLAN = {"Pop":([2020],1), "ItemKNN":([2020],1),
        "BPR":([2020,2021,2022],50), "LightGCN":([2020,2021,2022],50)}

assets=json.load(open(BASE + "/mooccube/pvr_assets.json"))
vid_pre={k:set(v) for k,v in assets["vid_pre"].items()}
prereqs_of=collections.defaultdict(set)
for A,B in assets["clean_edges"]: prereqs_of[B].add(A)
def required_of(tok):
    r=set()
    for c in vid_pre.get(tok,()): r|=prereqs_of.get(c,set())
    return r

def run(model_name, seed, epochs):
    cfg={"data_path":BASE + "/mooccube/recbole/dataset/","checkpoint_dir":BASE + "/mooccube/recbole/saved_seed/",
         "load_col":{"inter":["user_id","item_id","timestamp"]},
         "USER_ID_FIELD":"user_id","ITEM_ID_FIELD":"item_id","TIME_FIELD":"timestamp",
         "user_inter_num_interval":"[5,inf)","item_inter_num_interval":"[5,inf)",
         "eval_args":{"split":{"LS":"valid_and_test"},"order":"TO","mode":"full","group_by":"user"},
         "metrics":["NDCG","Hit"],"topk":[10],"valid_metric":"NDCG@10",
         "epochs":epochs,"train_batch_size":8192,"eval_batch_size":20000000,
         "stopping_step":5,"device":DEV,"gpu_id":0,"seed":seed,"reproducibility":True,"show_progress":False}
    config=Config(model=model_name,dataset="mooccube_csmath_t",config_dict=cfg)
    init_seed(seed,True)
    dataset=create_dataset(config)
    train_data,valid_data,test_data=data_preparation(config,dataset)
    IID=config["ITEM_ID_FIELD"]
    model=get_model(config["model"])(config,train_data.dataset).to(DEV)
    trainer=Trainer(config,model)
    trainer.fit(train_data,valid_data,saved=True,show_progress=False)
    trainer.evaluate(test_data,load_best_model=True,show_progress=False)
    if hasattr(model,"restore_user_e"): model.restore_user_e=None
    if hasattr(model,"restore_item_e"): model.restore_item_e=None
    model.eval()
    uid_list=np.asarray(test_data.uid_list)
    u2hist=test_data.uid2history_item; u2pos=test_data.uid2positive_item
    def ids2tok(ids): return dataset.id2token(IID,np.asarray(ids))
    tot=viol=0; pf_tot=pf_viol=0; ndcg=hr=ndcg_pf=0.0; nu=0; B=4096
    for s in range(0,len(uid_list),B):
        bu=uid_list[s:s+B]
        _,ids=full_sort_topk(bu,model,test_data,N,device=DEV); ids=ids.cpu().numpy()
        for r,uid in enumerate(bu):
            toks=ids2tok(ids[r]); reqs=[required_of(t) for t in toks]
            h=u2hist[uid]; h=h[h>0].cpu().numpy() if torch.is_tensor(h) else h
            M=set()
            for t in ids2tok(h): M|=vid_pre.get(t,set())
            p=u2pos[uid]; p=p[p>0].cpu().numpy() if torch.is_tensor(p) else p
            pos=set(ids2tok(p)); nu+=1
            idcg=sum(1/math.log2(i+2) for i in range(min(K,len(pos))))
            raw=list(toks[:K])
            d=sum(1/math.log2(i+2) for i,t in enumerate(raw) if t in pos)
            if idcg: ndcg+=d/idcg
            if pos & set(raw): hr+=1
            for j in range(K):
                if reqs[j]:
                    tot+=1
                    if len(reqs[j]&M)/len(reqs[j])<TAU: viol+=1
            kept=[]
            for j in range(N):
                if reqs[j] and len(reqs[j]&M)/len(reqs[j])<TAU: continue
                kept.append(toks[j])
                if len(kept)==K: break
            for t in kept:
                rq=required_of(t)
                if rq:
                    pf_tot+=1
                    if len(rq&M)/len(rq)<TAU: pf_viol+=1
            d=sum(1/math.log2(i+2) for i,t in enumerate(kept) if t in pos)
            if idcg: ndcg_pf+=d/idcg
    return {"hr":hr/nu,"ndcg":ndcg/nu,"pvr":viol/max(1,tot),
            "pf_ndcg":ndcg_pf/nu,"pf_pvr":pf_viol/max(1,pf_tot)}

def ms(xs):  # mean±std
    m=statistics.mean(xs); s=statistics.pstdev(xs) if len(xs)>1 else 0.0
    return m,s

results={}
for mn,(seeds,ep) in PLAN.items():
    runs=[]
    for sd in seeds:
        print(f">>> {mn} seed={sd}")
        try: runs.append(run(mn,sd,ep))
        except Exception as e: print(f"  failed {mn}/{sd}: {e}")
    if runs:
        agg={k:ms([r[k] for r in runs]) for k in runs[0]}
        results[mn]={"n_seeds":len(runs),"agg":agg,"raw":runs}
        print(f"  {mn}: HR {agg['hr'][0]:.4f}±{agg['hr'][1]:.4f}  "
              f"NDCG {agg['ndcg'][0]:.4f}±{agg['ndcg'][1]:.4f}  PVR {agg['pvr'][0]*100:.2f}±{agg['pvr'][1]*100:.2f}%")

print("\n"+"="*76)
print(f"{'Model':>10} | {'HR@10':>13} {'NDCG@10':>13} | {'PVR@10(τ=.5)':>14} | {'PostProcNDCG':>12}")
print("="*76)
for mn in PLAN:
    if mn not in results: continue
    a=results[mn]["agg"]
    print(f"{mn:>10} | {a['hr'][0]:.4f}±{a['hr'][1]:.4f} {a['ndcg'][0]:.4f}±{a['ndcg'][1]:.4f} | "
          f"{a['pvr'][0]*100:5.2f}±{a['pvr'][1]*100:.2f}% | {a['pf_ndcg'][0]:.4f}±{a['pf_ndcg'][1]:.4f}")
json.dump(results,open(BASE + "/mooccube/seeds_result.json","w"),indent=2)
print("\nSaved seeds_result.json")
