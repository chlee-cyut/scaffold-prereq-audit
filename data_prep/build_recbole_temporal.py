#!/usr/bin/env python3
"""Build a timestamped RecBole dataset (S1 CS/Math) for temporal leave-one-out.
Timestamps come from local_start_time in user_video_act.json (48k-user subset)."""
import os
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")   # root of the data/working-directory layout (see README)
import collections, os, json
from datetime import datetime
ROOT=BASE + "/mooccube/MOOCCube/"
OUT=BASE + "/mooccube/recbole/dataset/mooccube_csmath_t/"
os.makedirs(OUT, exist_ok=True)
CSMATH={"\u8ba1\u7b97\u673a\u79d1\u5b66\u6280\u672f","\u6570\u5b66"}  # MOOCCube discipline names (data): "Computer Science and Technology", "Mathematics"
EPOCH=datetime(1970,1,1)
def fld(c): return c.rsplit("_",1)[-1] if c.startswith("K_") else None
def to_sec(s):
    try: return (datetime.strptime(s,"%Y-%m-%d %H:%M:%S")-EPOCH).total_seconds()
    except Exception: return None

# S1 videos (courses whose main field is CS/Math)
cf=collections.defaultdict(collections.Counter)
for line in open(ROOT+"relations/course-concept.json",encoding="utf-8",errors="replace"):
    p=line.rstrip("\n").split("\t")
    if len(p)>=2 and fld(p[1].strip()): cf[p[0].strip()][fld(p[1].strip())]+=1
csmath_crs={c for c,cnt in cf.items() if cnt.most_common(1)[0][0] in CSMATH}
s1=set()
for line in open(ROOT+"relations/course-video.json",encoding="utf-8",errors="replace"):
    p=line.rstrip("\n").split("\t")
    if len(p)>=2 and p[0].strip() in csmath_crs: s1.add(p[1].strip())
print(f"S1 videos: {len(s1):,}")

# Stream user_video_act: earliest time per (user,video)
best={}   # (u,v) -> epoch
nrec=0
for line in open(ROOT+"additional_information/user_video_act.json",encoding="utf-8",errors="replace"):
    line=line.strip()
    if not line: continue
    try: r=json.loads(line)
    except json.JSONDecodeError: continue
    u=r.get("id")
    for a in r.get("activity",[]):
        v=a.get("video_id"); t=a.get("local_start_time")
        if not u or not v or v not in s1 or not t: continue
        sec=to_sec(t)
        if sec is None: continue
        k=(u,v); nrec+=1
        if k not in best or sec<best[k]: best[k]=sec
print(f"Scanned interactions: {nrec:,}  deduplicated (u,v): {len(best):,}")

with open(OUT+"mooccube_csmath_t.inter","w",encoding="utf-8") as fo:
    fo.write("user_id:token\titem_id:token\ttimestamp:float\n")
    for (u,v),sec in best.items():
        fo.write(f"{u}\t{v}\t{sec:.0f}\n")
print(f"Wrote {len(best):,} rows -> {OUT}mooccube_csmath_t.inter")
