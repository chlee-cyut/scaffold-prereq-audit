#!/usr/bin/env python3
"""Multi-model baselines: show that high PVR is common (not specific to LightGCN). Temporal LOO, τ=0.5."""
import os
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")   # root of the data/working-directory layout (see README)
import torch, functools
torch.load = functools.partial(torch.load, weights_only=False)
import numpy as np
for _a,_t in [("float",float),("int",int),("bool",bool),("object",object)]:
    if not hasattr(np,_a): setattr(np,_a,_t)
import json, collections
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, get_model
from recbole.utils.case_study import full_sort_topk

DEV="cuda"; N=50; K=10; TAU=0.5
MODELS = {"Pop":1, "ItemKNN":1, "BPR":50}   # LightGCN reuses saved results

assets=json.load(open(BASE + "/mooccube/pvr_assets.json"))
vid_pre={k:set(v) for k,v in assets["vid_pre"].items()}
prereqs_of=collections.defaultdict(set)
for A,B in assets["clean_edges"]: prereqs_of[B].add(A)
def required_of(tok):
    r=set()
    for c in vid_pre.get(tok,()): r|=prereqs_of.get(c,set())
    return r

def eval_model(model_name, epochs):
    cfg = {
        "data_path":BASE + "/mooccube/recbole/dataset/",
        "checkpoint_dir":BASE + "/mooccube/recbole/saved_base/",
        "load_col":{"inter":["user_id","item_id","timestamp"]},
        "USER_ID_FIELD":"user_id","ITEM_ID_FIELD":"item_id","TIME_FIELD":"timestamp",
        "user_inter_num_interval":"[5,inf)","item_inter_num_interval":"[5,inf)",
        "eval_args":{"split":{"LS":"valid_and_test"},"order":"TO","mode":"full","group_by":"user"},
        "metrics":["NDCG","Hit"],"topk":[10],"valid_metric":"NDCG@10",
        "epochs":epochs,"train_batch_size":8192,"eval_batch_size":20000000,
        "stopping_step":5,"device":DEV,"gpu_id":0,"seed":2020,"reproducibility":True,
        "show_progress":False,
    }
    config=Config(model=model_name,dataset="mooccube_csmath_t",config_dict=cfg)
    init_seed(config["seed"],config["reproducibility"])
    dataset=create_dataset(config)
    train_data,valid_data,test_data=data_preparation(config,dataset)
    IID=config["ITEM_ID_FIELD"]
    model=get_model(config["model"])(config,train_data.dataset).to(DEV)
    trainer=Trainer(config,model)
    trainer.fit(train_data,valid_data,saved=True,show_progress=False)
    test_result=trainer.evaluate(test_data,load_best_model=True,show_progress=False)
    for m in (model,):
        if hasattr(m,"restore_user_e"): m.restore_user_e=None
        if hasattr(m,"restore_item_e"): m.restore_item_e=None
    model.eval()
    # PVR@10
    uid_list=np.asarray(test_data.uid_list)
    u2hist=test_data.uid2history_item
    def ids2tok(ids): return dataset.id2token(IID,np.asarray(ids))
    tot=viol=0; B=4096
    for s in range(0,len(uid_list),B):
        bu=uid_list[s:s+B]
        _,ids=full_sort_topk(bu,model,test_data,K,device=DEV)
        ids=ids.cpu().numpy()
        for r,uid in enumerate(bu):
            toks=ids2tok(ids[r])
            h=u2hist[uid]; h=h[h>0].cpu().numpy() if torch.is_tensor(h) else h
            M=set()
            for t in ids2tok(h): M|=vid_pre.get(t,set())
            for t in toks:
                req=required_of(t)
                if req:
                    tot+=1
                    if len(req&M)/len(req)<TAU: viol+=1
    pvr=viol/max(1,tot)
    return {"model":model_name,"ndcg@10":float(test_result.get("ndcg@10",0)),
            "hit@10":float(test_result.get("hit@10",0)),"pvr@10":pvr,"judgable":tot}

rows=[]
for mn,ep in MODELS.items():
    print(f"\n>>> {mn} training...")
    try:
        r=eval_model(mn,ep); rows.append(r)
        print(f"  {mn}: HR@10={r['hit@10']:.4f} NDCG@10={r['ndcg@10']:.4f} PVR@10={r['pvr@10']*100:.2f}%")
    except Exception as e:
        print(f"  {mn} failed: {e}")

# Merge in LightGCN
g=json.load(open(BASE + "/mooccube/g2g3_t_result.json"))
rows.append({"model":"LightGCN","ndcg@10":g["ndcg_self"],"hit@10":g["hr_self"],
             "pvr@10":g["tau"]["0.5"]["raw_pvr"],"judgable":None})

print("\n"+"="*54)
print(f"{'Model':>10} | {'HR@10':>7} {'NDCG@10':>8} | {'PVR@10(τ=.5)':>12}")
print("="*54)
for r in rows:
    print(f"{r['model']:>10} | {r['hit@10']:>7.4f} {r['ndcg@10']:>8.4f} | {r['pvr@10']*100:>11.2f}%")
json.dump(rows,open(BASE + "/mooccube/baselines_pvr.json","w"),indent=2)
print("\nSaved baselines_pvr.json")
