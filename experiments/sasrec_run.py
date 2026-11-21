#!/usr/bin/env python3
"""SASRec sequential baseline (two datasets): HR/NDCG@10 + PVR@10 (τ=.5) + post-processing. Temporal LOO, full ranking."""
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

DEV="cuda"; N=50; K=10; TAU=0.5
DATASETS={
 "mooccube_csmath_t":BASE + "/mooccube/pvr_assets.json",
 "junyi":BASE + "/junyi/junyi_pvr_assets.json",
}

def load_assets(path):
    a=json.load(open(path))
    vid_pre={k:set(v) for k,v in a["vid_pre"].items()}
    pr=collections.defaultdict(set)
    for A,B in a["clean_edges"]: pr[B].add(A)
    return vid_pre,pr

def run(dsname,assets_path):
    vid_pre,prereqs_of=load_assets(assets_path)
    def required_of(tok):
        r=set()
        for c in vid_pre.get(tok,()): r|=prereqs_of.get(c,set())
        return r
    cfg={"data_path":BASE + "/mooccube/recbole/dataset/","checkpoint_dir":BASE + "/mooccube/recbole/saved_sasrec/",
         "load_col":{"inter":["user_id","item_id","timestamp"]},
         "USER_ID_FIELD":"user_id","ITEM_ID_FIELD":"item_id","TIME_FIELD":"timestamp",
         "user_inter_num_interval":"[5,inf)","item_inter_num_interval":"[5,inf)",
         "MAX_ITEM_LIST_LENGTH":50,"loss_type":"CE","train_neg_sample_args":None,
         "eval_args":{"split":{"LS":"valid_and_test"},"order":"TO","mode":"full","group_by":"user"},
         "metrics":["NDCG","Hit"],"topk":[10],"valid_metric":"NDCG@10",
         "epochs":50,"train_batch_size":4096,"eval_batch_size":4096,
         "stopping_step":5,"device":DEV,"gpu_id":0,"seed":2020,"reproducibility":True,"show_progress":False}
    config=Config(model="SASRec",dataset=dsname,config_dict=cfg)
    init_seed(2020,True)
    dataset=create_dataset(config)
    train_data,valid_data,test_data=data_preparation(config,dataset)
    IID=config["ITEM_ID_FIELD"]; ISEQ=IID+config["LIST_SUFFIX"]
    n_items=dataset.item_num
    model=get_model("SASRec")(config,train_data._dataset).to(DEV)
    tr=Trainer(config,model)
    tr.fit(train_data,valid_data,saved=True,show_progress=False)
    test_result=tr.evaluate(test_data,load_best_model=True,show_progress=False)
    model.eval()
    def ids2tok(ids): return dataset.id2token(IID,np.asarray(ids))

    tot=v=pf_tot=pf_v=0; ndcg=hr=ndcg_pf=0.0; nu=0
    NEG=torch.finfo(torch.float).min
    with torch.no_grad():
        for batched in test_data:
            interaction,history_index,positive_u,positive_i=batched
            interaction=interaction.to(DEV)
            scores=model.full_sort_predict(interaction).view(-1,n_items)
            scores[:,0]=NEG
            if history_index is not None: scores[history_index]=NEG
            topv,topi=torch.topk(scores,N,dim=1)      # [B,N]
            topi=topi.cpu().numpy()
            seq=interaction[ISEQ].cpu().numpy()        # [B,L] history item internal ids
            pos_i=positive_i.cpu().numpy()             # [B] held-out item internal ids
            for b in range(topi.shape[0]):
                toks=list(ids2tok(topi[b])); reqs=[required_of(t) for t in toks]
                hist_ids=[x for x in seq[b] if x>0]
                M=set()
                for t in ids2tok(hist_ids): M|=vid_pre.get(t,set())
                postok=ids2tok([pos_i[b]])[0]; nu+=1
                # accuracy (single positive)
                raw=toks[:K]
                if postok in raw:
                    r=raw.index(postok); ndcg+=1/math.log2(r+2); hr+=1
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
                if postok in kept:
                    ndcg_pf+=1/math.log2(kept.index(postok)+2)
    return {"dataset":dsname,"model":"SASRec","hr":hr/nu,"ndcg":ndcg/nu,
            "pvr":v/max(1,tot),"pf_ndcg":ndcg_pf/nu,"pf_pvr":pf_v/max(1,pf_tot),
            "official":{k:float(x) for k,x in test_result.items()}}

out=[]
for ds,ap in DATASETS.items():
    print(f">>> SASRec on {ds}")
    try:
        r=run(ds,ap); out.append(r)
        print(f"  {ds}: HR@10={r['hr']:.4f} NDCG@10={r['ndcg']:.4f} PVR@10={r['pvr']*100:.2f}% post-proc NDCG={r['pf_ndcg']:.4f}")
    except Exception as e:
        import traceback; print(f"  {ds} failed: {e}"); traceback.print_exc()
json.dump(out,open(BASE + "/mooccube/sasrec_result.json","w"),indent=2)
print("Saved sasrec_result.json")
