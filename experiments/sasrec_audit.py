#!/usr/bin/env python3
"""SASRec leakage audit + history stratification(MOOCCube CS+Math).
(1) repeat rate; (2) same-course transition rate; (3) first-order Markov baseline HR@10;
(4) behavioral inconsistency rate vs history length. Uses user_video_act sequences (consistent with the temporal dataset)."""
import os
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")   # root of the data/working-directory layout (see README)
import json, collections, math
from datetime import datetime
ROOT=BASE + "/mooccube/MOOCCube/"
def to_sec(s):
    try: return (datetime.strptime(s,"%Y-%m-%d %H:%M:%S")-datetime(1970,1,1)).total_seconds()
    except: return None

# S1 videos (main field CS/Math) - same logic as dataset construction
CSMATH={"\u8ba1\u7b97\u673a\u79d1\u5b66\u6280\u672f","\u6570\u5b66"}  # MOOCCube discipline names (data): "Computer Science and Technology", "Mathematics"
def fld(c): return c.rsplit("_",1)[-1] if c.startswith("K_") else None
cf=collections.defaultdict(collections.Counter)
for line in open(ROOT+"relations/course-concept.json",encoding="utf-8",errors="replace"):
    p=line.rstrip("\n").split("\t")
    if len(p)>=2 and fld(p[1].strip()): cf[p[0].strip()][fld(p[1].strip())]+=1
csmath_crs={c for c,cnt in cf.items() if cnt.most_common(1)[0][0] in CSMATH}
s1=set(); vid_course=collections.defaultdict(set)
for line in open(ROOT+"relations/course-video.json",encoding="utf-8",errors="replace"):
    p=line.rstrip("\n").split("\t")
    if len(p)>=2:
        vid_course[p[1].strip()].add(p[0].strip())
        if p[0].strip() in csmath_crs: s1.add(p[1].strip())

assets=json.load(open(BASE + "/mooccube/pvr_assets.json"))
vid_pre={k:set(v) for k,v in assets["vid_pre"].items()}
prereqs_of=collections.defaultdict(set)
for A,B in assets["clean_edges"]: prereqs_of[B].add(A)
def required_of(tok):
    r=set()
    for c in vid_pre.get(tok,()): r|=prereqs_of.get(c,set())
    return r

# Read each user's S1 video temporal sequence (first occurrence)
seqs=[]
tot_events=0; repeat_events=0
for line in open(ROOT+"additional_information/user_video_act.json",encoding="utf-8",errors="replace"):
    line=line.strip()
    if not line: continue
    try: r=json.loads(line)
    except: continue
    acts=[]
    for a in r.get("activity",[]):
        v=a.get("video_id"); t=a.get("local_start_time")
        if v in s1 and t:
            sec=to_sec(t)
            if sec is not None: acts.append((sec,v))
    if not acts: continue
    acts.sort()
    seen=set(); order=[]
    for sec,v in acts:
        tot_events+=1
        if v in seen: repeat_events+=1
        else: seen.add(v); order.append(v)
    if len(order)>=2: seqs.append(order)

print(f"S1 user sequences: {len(seqs):,}")
print(f"(1) repeat rate: {repeat_events/max(1,tot_events)*100:.1f}%  (share of repeat-view events)")

# (2) same-course transition rate: whether adjacent videos share a course
sc=0; sc_tot=0
for o in seqs:
    for a,b in zip(o,o[1:]):
        sc_tot+=1
        if vid_course.get(a,set()) & vid_course.get(b,set()): sc+=1
print(f"(2) same-course transition rate: {sc/max(1,sc_tot)*100:.1f}%  (adjacent views in the same course)")

# (3) first-order Markov baseline: train = full sequence except last item; bigram counts; test = last item
trans=collections.defaultdict(collections.Counter); pop=collections.Counter()
tests=[]
for o in seqs:
    *tr,last=o
    for a,b in zip(tr,tr[1:]): trans[a][b]+=1
    for v in tr: pop[v]+=1
    tests.append((tr,last))
hit=0; nu=0
for tr,last in tests:
    if not tr: continue
    nu+=1; seen=set(tr)
    cand=trans.get(tr[-1],None)
    ranked=[v for v,_ in cand.most_common() if v not in seen][:10] if cand else []
    if len(ranked)<10:  # pad with popularity
        for v,_ in pop.most_common():
            if v not in seen and v not in ranked:
                ranked.append(v)
                if len(ranked)==10: break
    if last in ranked: hit+=1
print(f"(3) first-order Markov+pop baseline HR@10: {hit/max(1,nu)*100:.1f}%  (vs SASRec 70.9%)")

# (4) behavioral inconsistency rate vs history length
bins=[(5,10),(11,20),(21,50),(51,10**9)]
bstat={b:[0,0] for b in bins}   # [viol, judgeable]
for o in seqs:
    *tr,last=o
    hl=len(tr)
    if hl<5: continue
    b=next((bb for bb in bins if bb[0]<=hl<=bb[1]),None)
    if not b: continue
    M=set()
    for v in tr: M|=vid_pre.get(v,set())
    req=required_of(last)
    if req:
        bstat[b][1]+=1
        if len(req&M)/len(req)<0.5: bstat[b][0]+=1
print("(4) behavioral inconsistency rate vs history length:")
strat=[]
for b in bins:
    viol,jud=bstat[b]
    rate=viol/max(1,jud)*100
    lbl=f"{b[0]}-{b[1] if b[1]<10**9 else '+'}"
    print(f"    history {lbl:>7}: {rate:.1f}%  (n={jud:,})")
    strat.append({"bin":lbl,"rate":rate,"n":jud})

json.dump({"repeat_rate":repeat_events/max(1,tot_events),
           "same_course_rate":sc/max(1,sc_tot),
           "markov_hr10":hit/max(1,nu),
           "history_strat":strat},
          open(BASE + "/paper/sasrec_audit_result.json","w"),indent=2)
print("Saved sasrec_audit_result.json")
