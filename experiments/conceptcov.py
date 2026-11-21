#!/usr/bin/env python3
"""ConceptCov@10 (+PVR) multi-model measurement. Temporal LOO, τ=0.5, candidate pool N=50. seed=2020.
Also compares: LightGCN raw vs post-filtered ConceptCov (does post-processing also raise concept coverage?)."""
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

assets=json.load(open(BASE + "/mooccube/pvr_assets.json"))
vid_pre={k:set(v) for k,v in assets["vid_pre"].items()}   # video -> concept set
prereqs_of=collections.defaultdict(set)
for A,B in assets["clean_edges"]: prereqs_of[B].add(A)

def concept_ready(c,M):
    pre=prereqs_of.get(c,None)
    if not pre: return True                     # root concept: ready by default
    return len(pre & M)/len(pre) >= TAU
def item_required(concepts):
    r=set()
    for c in concepts: r|=prereqs_of.get(c,set())
    return r
def NR(item_tokens,M):                          # set of newly ready concepts
    s=set()
    for t in item_tokens:
        for c in vid_pre.get(t,()):
            if c not in M and concept_ready(c,M): s.add(c)
    return s

def run(model_name,epochs):
    cfg={"data_path":BASE + "/mooccube/recbole/dataset/","checkpoint_dir":BASE + "/mooccube/recbole/saved_cc/",
         "load_col":{"inter":["user_id","item_id","timestamp"]},
         "USER_ID_FIELD":"user_id","ITEM_ID_FIELD":"item_id","TIME_FIELD":"timestamp",
         "user_inter_num_interval":"[5,inf)","item_inter_num_interval":"[5,inf)",
         "eval_args":{"split":{"LS":"valid_and_test"},"order":"TO","mode":"full","group_by":"user"},
         "metrics":["NDCG"],"topk":[10],"valid_metric":"NDCG@10",
         "epochs":epochs,"train_batch_size":8192,"eval_batch_size":20000000,
         "stopping_step":5,"device":DEV,"gpu_id":0,"seed":2020,"reproducibility":True,"show_progress":False}
    config=Config(model=model_name,dataset="mooccube_csmath_t",config_dict=cfg)
    init_seed(2020,True)
    dataset=create_dataset(config)
    train_data,valid_data,test_data=data_preparation(config,dataset)
    IID=config["ITEM_ID_FIELD"]
    model=get_model(config["model"])(config,train_data.dataset).to(DEV)
    tr=Trainer(config,model); tr.fit(train_data,valid_data,saved=True,show_progress=False)
    tr.evaluate(test_data,load_best_model=True,show_progress=False)
    if hasattr(model,"restore_user_e"): model.restore_user_e=None
    if hasattr(model,"restore_item_e"): model.restore_item_e=None
    model.eval()
    uid_list=np.asarray(test_data.uid_list); u2hist=test_data.uid2history_item
    def ids2tok(ids): return dataset.id2token(IID,np.asarray(ids))
    cc_sum=cc_n=0.0; nr_top_sum=0.0; nu=0
    cc_pf_sum=cc_pf_n=0.0                        # ConceptCov after post-processing
    B=4096
    for s in range(0,len(uid_list),B):
        bu=uid_list[s:s+B]
        _,ids=full_sort_topk(bu,model,test_data,N,device=DEV); ids=ids.cpu().numpy()
        for r,uid in enumerate(bu):
            toks=list(ids2tok(ids[r]))
            h=u2hist[uid]; h=h[h>0].cpu().numpy() if torch.is_tensor(h) else h
            M=set()
            for t in ids2tok(h): M|=vid_pre.get(t,set())
            nr_pool=NR(toks,M)
            if not nr_pool: continue
            nr_top=NR(toks[:K],M)
            cc_sum+=len(nr_top)/len(nr_pool); cc_n+=1; nr_top_sum+=len(nr_top); nu+=1
            # Post-processing: drop violating items from pool (item Ready<τ), take top K, then compute ConceptCov
            kept=[]
            for t in toks:
                req=item_required(vid_pre.get(t,()))
                if req and len(req&M)/len(req)<TAU: continue
                kept.append(t)
                if len(kept)==K: break
            nr_top_pf=NR(kept,M)
            cc_pf_sum+=len(nr_top_pf)/len(nr_pool); cc_pf_n+=1
    return {"model":model_name,"conceptcov@10":cc_sum/max(1,cc_n),
            "avg_new_ready@10":nr_top_sum/max(1,nu),
            "conceptcov@10_postproc":cc_pf_sum/max(1,cc_pf_n),"n_users":nu}

rows=[]
for mn,ep in MODELS.items():
    print(f">>> {mn}")
    try:
        r=run(mn,ep); rows.append(r)
        print(f"  {mn}: ConceptCov@10={r['conceptcov@10']:.4f}  "
              f"avg newly-ready concepts={r['avg_new_ready@10']:.2f}  post-proc ConceptCov={r['conceptcov@10_postproc']:.4f}")
    except Exception as e:
        print(f"  {mn} failed: {e}")

print("\n"+"="*70)
print(f"{'Model':>10} | {'ConceptCov@10':>13} | {'avgNewRdy@10':>12} | {'PP_ConceptCov':>15}")
print("="*70)
for r in rows:
    print(f"{r['model']:>10} | {r['conceptcov@10']:>13.4f} | {r['avg_new_ready@10']:>12.2f} | {r['conceptcov@10_postproc']:>15.4f}")
json.dump(rows,open(BASE + "/mooccube/conceptcov_result.json","w"),indent=2)
print("\nSaved conceptcov_result.json")
