#!/usr/bin/env python3
"""Required reviewer additions: judgeability, ViolationAll, candidate sufficiency, direct vs transitive PVR.
Generic recommenders x two datasets, seed 2020, temporal LOO, τ=0.5, pool N=50."""
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
DATASETS={"mooccube_csmath_t":BASE + "/mooccube/pvr_assets.json",
          "junyi":BASE + "/junyi/junyi_pvr_assets.json"}
MODELS={"Pop":1,"ItemKNN":1,"BPR":50,"LightGCN":50}

def load_graph(path):
    a=json.load(open(path)); vid_pre={k:set(v) for k,v in a["vid_pre"].items()}
    direct=collections.defaultdict(set)
    for A,B in a["clean_edges"]: direct[B].add(A)
    # Transitive prerequisites by reachability (BFS), identical to closure_map() in revision/rev1_mitigation.py.
    # (An earlier memoized DFS with a cycle guard was order-dependent on the cyclic MOOCCube graph.)
    trans={}
    for c in set(direct)|{x for v in direct.values() for x in v}:
        seen,stack=set(),list(direct.get(c,()))
        while stack:
            q=stack.pop()
            if q not in seen: seen.add(q); stack.extend(direct.get(q,()))
        trans[c]=seen
    return vid_pre, direct, trans

def run(model_name,epochs,dsname,vid_pre,direct,trans):
    def req(concepts, table):
        r=set()
        for c in concepts: r|=table.get(c,set())
        return r
    cfg={"data_path":BASE + "/mooccube/recbole/dataset/","checkpoint_dir":BASE + "/mooccube/recbole/saved_rev/",
         "load_col":{"inter":["user_id","item_id","timestamp"]},
         "USER_ID_FIELD":"user_id","ITEM_ID_FIELD":"item_id","TIME_FIELD":"timestamp",
         "user_inter_num_interval":"[5,inf)","item_inter_num_interval":"[5,inf)",
         "eval_args":{"split":{"LS":"valid_and_test"},"order":"TO","mode":"full","group_by":"user"},
         "metrics":["NDCG","Hit"],"topk":[10],"valid_metric":"NDCG@10",
         "epochs":epochs,"train_batch_size":8192,"eval_batch_size":20000000,
         "stopping_step":5,"device":DEV,"gpu_id":0,"seed":2020,"reproducibility":True,"show_progress":False}
    config=Config(model=model_name,dataset=dsname,config_dict=cfg)
    init_seed(2020,True)
    dataset=create_dataset(config); train_data,valid_data,test_data=data_preparation(config,dataset)
    IID=config["ITEM_ID_FIELD"]
    model=get_model(config["model"])(config,train_data._dataset).to(DEV)
    tr=Trainer(config,model); tr.fit(train_data,valid_data,saved=True,show_progress=False)
    tr.evaluate(test_data,load_best_model=True,show_progress=False)
    if hasattr(model,"restore_user_e"): model.restore_user_e=None
    if hasattr(model,"restore_item_e"): model.restore_item_e=None
    model.eval()
    uid_list=np.asarray(test_data.uid_list); u2hist=test_data.uid2history_item
    def ids2tok(ids): return dataset.id2template if False else dataset.id2token(IID,np.asarray(ids))
    tot=viol=viol_t=0; K_slots=0; excluded=0; nu=0
    ready_counts=[]; users_lt_K=0; B=4096
    for s in range(0,len(uid_list),B):
        bu=uid_list[s:s+B]
        _,ids=full_sort_topk(bu,model,test_data,N,device=DEV); ids=ids.cpu().numpy()
        for r,uid in enumerate(bu):
            toks=list(ids2tok(ids[r]))
            h=u2hist[uid]; h=h[h>0].cpu().numpy() if torch.is_tensor(h) else h
            M=set()
            for t in ids2tok(h): M|=vid_pre.get(t,set())
            nu+=1
            top=toks[:K]
            judge=0
            for t in top:
                rq=req(vid_pre.get(t,()),direct)
                if rq:
                    judge+=1; tot+=1
                    if len(rq&M)/len(rq)<TAU: viol+=1
                    rqt=req(vid_pre.get(t,()),trans)
                    if rqt and len(rqt&M)/len(rqt)<TAU: viol_t+=1
            K_slots+=K
            if judge==0: excluded+=1
            # sufficiency: number of ready items in top-N (non-violating and with prerequisites)
            ready=0
            for t in toks:
                rq=req(vid_pre.get(t,()),direct)
                if not rq: ready+=1                       # no prerequisites -> recommendable
                elif len(rq&M)/len(rq)>=TAU: ready+=1
            ready_counts.append(ready)
            if ready<K: users_lt_K+=1
    rc=np.array(ready_counts)
    return {"dataset":dsname,"model":model_name,
            "PVR_micro":viol/max(1,tot),
            "ViolationAll@10":viol/max(1,K_slots),
            "JudgeableRate@10":tot/max(1,K_slots),
            "excluded_user_rate":excluded/max(1,nu),
            "PVR_transitive":viol_t/max(1,tot),
            "avg_ready_in_top50":float(rc.mean()),
            "frac_users_lt_K_ready":users_lt_K/max(1,nu)}

out=[]
for ds,ap in DATASETS.items():
    vid_pre,direct,trans=load_graph(ap)
    tt=sum(len(v) for v in trans.values()); dd=sum(len(v) for v in direct.values())
    print(f"[{ds}] direct prerequisite relations {dd}, transitive closure {tt}")
    for mn,ep in MODELS.items():
        try:
            r=run(mn,ep,ds,vid_pre,direct,trans); out.append(r)
            print(f"  {mn}: PVR={r['PVR_micro']*100:.1f}% ViolAll={r['ViolationAll@10']*100:.1f}% "
                  f"Judge={r['JudgeableRate@10']*100:.1f}% excl={r['excluded_user_rate']*100:.1f}% "
                  f"PVR_trans={r['PVR_transitive']*100:.1f}% avgReady={r['avg_ready_in_top50']:.1f} "
                  f"<K={r['frac_users_lt_K_ready']*100:.1f}%")
        except Exception as e:
            import traceback; print(f"  {mn} failed: {e}"); traceback.print_exc()
json.dump(out,open(BASE + "/paper/revision_metrics_result.json","w"),indent=2)
print("Saved revision_metrics_result.json")
