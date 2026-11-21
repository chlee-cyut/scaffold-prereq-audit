#!/usr/bin/env python3
"""Junyi second dataset: multi-model PVR + post-processing + behavioral ceiling diagnostics. Temporal LOO, τ=0.5."""
import os
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")   # root of the data/working-directory layout (see README)
import torch, functools
torch.load = functools.partial(torch.load, weights_only=False)
import numpy as np
for _a,_t in [("float",float),("int",int),("bool",bool),("object",object)]:
    if not hasattr(np,_a): setattr(np,_a,_t)
import json, collections, math
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, get_model
from recbole.utils.case_study import full_sort_topk

DEV="cuda"; N=50; K=10; TAU=0.5
MODELS={"Pop":1,"ItemKNN":1,"BPR":50,"LightGCN":50}

assets=json.load(open(BASE + "/junyi/junyi_pvr_assets.json"))
vid_pre={k:set(v) for k,v in assets["vid_pre"].items()}
prereqs_of=collections.defaultdict(set)
for A,B in assets["clean_edges"]: prereqs_of[B].add(A)
def required_of(tok):
    r=set()
    for c in vid_pre.get(tok,()): r|=prereqs_of.get(c,set())
    return r

def run(model_name,epochs,want_diag=False):
    cfg={"data_path":BASE + "/mooccube/recbole/dataset/","checkpoint_dir":BASE + "/mooccube/recbole/saved_junyi/",
         "load_col":{"inter":["user_id","item_id","timestamp"]},
         "USER_ID_FIELD":"user_id","ITEM_ID_FIELD":"item_id","TIME_FIELD":"timestamp",
         "user_inter_num_interval":"[5,inf)","item_inter_num_interval":"[5,inf)",
         "eval_args":{"split":{"LS":"valid_and_test"},"order":"TO","mode":"full","group_by":"user"},
         "metrics":["NDCG","Hit"],"topk":[10],"valid_metric":"NDCG@10",
         "epochs":epochs,"train_batch_size":8192,"eval_batch_size":20000000,
         "stopping_step":5,"device":DEV,"gpu_id":0,"seed":2020,"reproducibility":True,"show_progress":False}
    config=Config(model=model_name,dataset="junyi",config_dict=cfg)
    init_seed(2020,True)
    dataset=create_dataset(config)
    if model_name=="Pop": print(dataset)
    train_data,valid_data,test_data=data_preparation(config,dataset)
    IID=config["ITEM_ID_FIELD"]
    model=get_model(config["model"])(config,train_data.dataset).to(DEV)
    tr=Trainer(config,model); tr.fit(train_data,valid_data,saved=True,show_progress=False)
    tr.evaluate(test_data,load_best_model=True,show_progress=False)
    if hasattr(model,"restore_user_e"): model.restore_user_e=None
    if hasattr(model,"restore_item_e"): model.restore_item_e=None
    model.eval()
    uid_list=np.asarray(test_data.uid_list); u2hist=test_data.uid2history_item; u2pos=test_data.uid2positive_item
    def ids2tok(ids): return dataset.id2token(IID,np.asarray(ids))
    tot=v=pf_tot=pf_v=0; ndcg=hr=ndcg_pf=0.0; nu=0
    ho_viol=ho_hasp=ho_n=0
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
            postoks=ids2tok(p); pos=set(postoks); nu+=1
            idcg=sum(1/math.log2(i+2) for i in range(min(K,len(pos))))
            raw=toks[:K]
            d=sum(1/math.log2(i+2) for i,t in enumerate(raw) if t in pos)
            if idcg: ndcg+=d/idcg
            if pos&set(raw): hr+=1
            for j in range(K):
                if reqs[j]:
                    tot+=1
                    if len(reqs[j]&M)/len(reqs[j])<TAU: v+=1
            kept=[]
            for j in range(N):
                if reqs[j] and len(reqs[j]&M)/len(reqs[j])<TAU: continue
                kept.append(toks[j])
                if len(kept)==K: break
            for t in kept:
                rq=required_of(t)
                if rq:
                    pf_tot+=1
                    if len(rq&M)/len(rq)<TAU: pf_v+=1
            d=sum(1/math.log2(i+2) for i,t in enumerate(kept) if t in pos)
            if idcg: ndcg_pf+=d/idcg
            if want_diag and len(postoks):
                req=required_of(postoks[0])
                if req:
                    ho_hasp+=1
                    if len(req&M)/len(req)<TAU: ho_viol+=1
                ho_n+=1
    out={"model":model_name,"hr":hr/nu,"ndcg":ndcg/nu,"pvr":v/max(1,tot),
         "pf_ndcg":ndcg_pf/nu,"pf_pvr":pf_v/max(1,pf_tot),"judgable":tot}
    if want_diag:
        out["holdout_viol_frac"]=ho_viol/max(1,ho_n)
        out["holdout_hasprereq_frac"]=ho_hasp/max(1,ho_n)
    return out

rows=[]
for mn,ep in MODELS.items():
    print(f">>> {mn}")
    try:
        r=run(mn,ep,want_diag=(mn=="LightGCN")); rows.append(r)
        extra=f"  held-out item violation={r['holdout_viol_frac']*100:.1f}%" if "holdout_viol_frac" in r else ""
        print(f"  {mn}: HR@10={r['hr']:.4f} NDCG@10={r['ndcg']:.4f} PVR@10={r['pvr']*100:.2f}% post-proc NDCG={r['pf_ndcg']:.4f}{extra}")
    except Exception as e:
        import traceback; print(f"  {mn} failed: {e}"); traceback.print_exc()

print("\n"+"="*70)
print(f"{'Model':>10} | {'HR@10':>7} {'NDCG@10':>8} | {'PVR@10':>8} | {'PP_NDCG':>10}")
print("="*70)
for r in rows:
    print(f"{r['model']:>10} | {r['hr']:>7.4f} {r['ndcg']:>8.4f} | {r['pvr']*100:>7.2f}% | {r['pf_ndcg']:>10.4f}")
ld=[r for r in rows if r['model']=='LightGCN']
if ld: print(f"\nBehavioral ceiling: held-out items with prerequisites {ld[0]['holdout_hasprereq_frac']*100:.1f}%, of which violating {ld[0]['holdout_viol_frac']*100:.1f}%")
json.dump(rows,open(BASE + "/junyi/junyi_result.json","w"),indent=2)
print("Saved junyi_result.json")
