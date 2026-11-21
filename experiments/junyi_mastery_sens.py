#!/usr/bin/env python3
"""Mastery-definition sensitivity (Junyi): mastery = "answered correctly" (correct=1) vs the original "attempted". Recompute PVR/post-processing/ceiling.
Interactions (for training recommenders) unchanged; only the mastery set M used for PVR changes. Temporal LOO, τ=0.5."""
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

# idx->name
idx2name={}
for line in open(BASE + "/junyi/vertex_id2idx",encoding="utf-8"):
    line=line.rstrip("\n")
    if line: n,i=line.rsplit(",",1); idx2name[int(i)]=n
# user -> exercises answered correctly (correct=1)
user_correct=collections.defaultdict(set)
for u,line in enumerate(open(BASE + "/junyi/train.json",encoding="utf-8")):
    line=line.strip()
    if not line: continue
    for ex,corr in json.loads(line):
        if corr==1:
            nm=idx2name.get(ex)
            if nm: user_correct[f"JU_{u}"].add(nm)
print(f"Users with correct-answer records: {len(user_correct):,}")

assets=json.load(open(BASE + "/junyi/junyi_pvr_assets.json"))
vid_pre={k:set(v) for k,v in assets["vid_pre"].items()}
prereqs_of=collections.defaultdict(set)
for A,B in assets["clean_edges"]: prereqs_of[B].add(A)
def required_of(tok):
    r=set()
    for c in vid_pre.get(tok,()): r|=prereqs_of.get(c,set())
    return r

def run(model_name,epochs):
    cfg={"data_path":BASE + "/mooccube/recbole/dataset/","checkpoint_dir":BASE + "/mooccube/recbole/saved_junyi_ms/",
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
    train_data,valid_data,test_data=data_preparation(config,dataset)
    IID=config["ITEM_ID_FIELD"]; UID=config["USER_ID_FIELD"]
    model=get_model(config["model"])(config,train_data._dataset).to(DEV)
    tr=Trainer(config,model); tr.fit(train_data,valid_data,saved=True,show_progress=False)
    tr.evaluate(test_data,load_best_model=True,show_progress=False)
    if hasattr(model,"restore_user_e"): model.restore_user_e=None
    if hasattr(model,"restore_item_e"): model.restore_item_e=None
    model.eval()
    uid_list=np.asarray(test_data.uid_list); u2hist=test_data.uid2history_item; u2pos=test_data.uid2positive_item
    def ids2tok(ids): return dataset.id2token(IID,np.asarray(ids))
    def uid2tok(u): return dataset.id2token(UID,np.asarray([u]))[0]
    tot=v=pf_tot=pf_v=0; ndcg=hr=ndcg_pf=0.0; nu=0; ho_v=ho_n=0; B=4096
    for s in range(0,len(uid_list),B):
        bu=uid_list[s:s+B]
        _,ids=full_sort_topk(bu,model,test_data,N,device=DEV); ids=ids.cpu().numpy()
        for r,uid in enumerate(bu):
            toks=list(ids2tok(ids[r])); reqs=[required_of(t) for t in toks]
            h=u2hist[uid]; h=h[h>0].cpu().numpy() if torch.is_tensor(h) else h
            hist_tok=set(ids2tok(h))
            M=hist_tok & user_correct.get(uid2tok(uid),set())   # mastery = history exercises answered correctly
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
            if len(postoks):
                req=required_of(postoks[0])
                if req:
                    ho_n+=1
                    if len(req&M)/len(req)<TAU: ho_v+=1
    return {"model":model_name,"hr":hr/nu,"ndcg":ndcg/nu,"pvr":v/max(1,tot),
            "pf_ndcg":ndcg_pf/nu,"holdout_viol":ho_v/max(1,ho_n)}

rows=[]
for mn,ep in MODELS.items():
    print(f">>> {mn}")
    try:
        r=run(mn,ep); rows.append(r)
        print(f"  {mn}: HR={r['hr']:.4f} PVR@10(correct mastery)={r['pvr']*100:.2f}% post-proc NDCG={r['pf_ndcg']:.4f} held-out item violation={r['holdout_viol']*100:.1f}%")
    except Exception as e:
        import traceback; print(f"  {mn} failed: {e}"); traceback.print_exc()

att={r['model']:r for r in json.load(open(BASE + "/junyi/junyi_result.json"))}
print("\n"+"="*72)
print(f"{'Model':>9} | {'PVR(tried)':>11} {'PVR(corr)':>10} | {'HOViol(corr)':>14}")
print("="*72)
for r in rows:
    a=att.get(r['model'],{})
    print(f"{r['model']:>9} | {a.get('pvr',0)*100:>10.1f}% {r['pvr']*100:>9.1f}% | {r['holdout_viol']*100:>13.1f}%")
json.dump(rows,open(BASE + "/junyi/junyi_mastery_sens_result.json","w"),indent=2)
print("Saved junyi_mastery_sens_result.json")
