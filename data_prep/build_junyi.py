#!/usr/bin/env python3
"""Build the Junyi RecBole dataset + prerequisite assets. Exercise = item = concept; prerequisite graph = expert-annotated DAG; sequence position = temporal order."""
import os
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")   # root of the data/working-directory layout (see README)
import json, os, collections
OUT=BASE + "/mooccube/recbole/dataset/junyi/"   # same data_path as MOOCCube for reuse
os.makedirs(OUT, exist_ok=True)

idx2name={}
for line in open(BASE + "/junyi/vertex_id2idx",encoding="utf-8"):
    line=line.rstrip("\n")
    if line:
        name,idx=line.rsplit(",",1); idx2name[int(idx)]=name

pre=json.load(open(BASE + "/junyi/prerequisite.json"))
clean_edges=[[idx2name[a],idx2name[b]] for a,b in pre if a in idx2name and b in idx2name]
print(f"Prerequisite edges: {len(clean_edges)}")

# vid_pre: exercise -> its own concept (identity)
all_ex=set(idx2name.values())
vid_pre={ex:[ex] for ex in all_ex}
json.dump({"vid_pre":vid_pre,"clean_edges":clean_edges},
          open(BASE + "/junyi/junyi_pvr_assets.json","w"),ensure_ascii=False)

# Interactions: one student per line of train.json; first-occurrence position = timestamp
nrow=0
with open(OUT+"junyi.inter","w",encoding="utf-8") as fo:
    fo.write("user_id:token\titem_id:token\ttimestamp:float\n")
    for u,line in enumerate(open(BASE + "/junyi/train.json",encoding="utf-8")):
        line=line.strip()
        if not line: continue
        seq=json.loads(line)
        seen=set(); pos=0
        for ex,corr in seq:
            name=idx2name.get(ex)
            if name is None: continue
            if name not in seen:
                seen.add(name)
                fo.write(f"JU_{u}\t{name}\t{pos}\n"); pos+=1; nrow+=1
print(f"Wrote interactions: {nrow:,} -> {OUT}junyi.inter")
print(f"assets -> {BASE}/junyi/junyi_pvr_assets.json")
