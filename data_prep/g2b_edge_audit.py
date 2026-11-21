#!/usr/bin/env python3
"""
Prerequisite edge audit: distinguish whether high-violation edges are "reversed direction" or "co-taught in the same course (order meaningless)", and recompute PVR after cleaning.

For each edge A->B (A is the prerequisite) compute:
  appl      : number of users who consumed both concepts
  viol      : number of users who saw B before A (t[B] < t[A])
  equal     : A and B first consumed at the same time (usually the same video)
  same_vid  : number of users whose first contact with A and B is the same video (co-teaching signal)
  co_occ    : number of videos structurally containing both A and B (from video-concept)
  gap_days  : median |t[A]-t[B]| in days (non-simultaneous only)

Classes:
  CO_TAUGHT     same_vid/appl ≥ .5  or  high co_occ share  → order meaningless, removed when cleaning
  REVERSED      viol/appl ≥ .80 and not CO_TAUGHT       → direction likely reversed, flipped when cleaning
  HIGH_VIOL     .60 ≤ viol/appl < .80                  → kept, but high
  OK            everything else
"""
import os
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")   # root of the data/working-directory layout (see README)
import json, collections, statistics
from datetime import datetime

ROOT = BASE + "/mooccube/MOOCCube/"
EPOCH = datetime(1970, 1, 1)
def to_sec(s):
    try: return (datetime.strptime(s, "%Y-%m-%d %H:%M:%S") - EPOCH).total_seconds()
    except Exception: return None

# Prerequisite edges
edges = []
pre_concepts = set()
with open(ROOT+"relations/prerequisite-dependency.json", encoding="utf-8", errors="replace") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[0].strip() != p[1].strip() and p[0].strip() and p[1].strip():
            A, B = p[0].strip(), p[1].strip()
            edges.append((A, B)); pre_concepts.add(A); pre_concepts.add(B)
edges = list(dict.fromkeys(edges))
prereqs_of = collections.defaultdict(set)
for A, B in edges: prereqs_of[B].add(A)

# video -> prerequisite concepts ; concept -> set of videos (structural co-occurrence)
vid_pre = collections.defaultdict(set)
concept_vids = collections.defaultdict(set)
with open(ROOT+"relations/video-concept.json", encoding="utf-8", errors="replace") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[1].strip() in pre_concepts:
            v, c = p[0].strip(), p[1].strip()
            vid_pre[v].add(c); concept_vids[c].add(v)
co_occ = {(A, B): len(concept_vids[A] & concept_vids[B]) for A, B in edges}

# Stream activity
appl = collections.Counter(); viol = collections.Counter()
equal = collections.Counter(); same_vid = collections.Counter()
gaps = collections.defaultdict(list)
with open(ROOT+"additional_information/user_video_act.json", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        first_t = {}; first_v = {}
        for a in r.get("activity", []):
            v = a.get("video_id"); t = a.get("local_start_time")
            if not v or not t or v not in vid_pre: continue
            for c in vid_pre[v]:
                if c not in first_t or t < first_t[c]:
                    first_t[c] = t; first_v[c] = v
        if not first_t: continue
        for B in first_t:
            tB = first_t[B]
            for A in prereqs_of.get(B, ()):
                if A in first_t:
                    tA = first_t[A]; e = (A, B)
                    appl[e] += 1
                    if tB < tA: viol[e] += 1
                    if tB == tA: equal[e] += 1
                    if first_v[A] == first_v[B]: same_vid[e] += 1
                    else:
                        sa, sb = to_sec(tA), to_sec(tB)
                        if sa is not None and sb is not None:
                            gaps[e].append(abs(sa - sb) / 86400.0)

# Classification
cls = {}
for e in edges:
    ap = appl[e]
    if ap == 0: cls[e] = "UNUSED"; continue
    vr = viol[e] / ap
    svr = same_vid[e] / ap
    # co_occ share: co-occurring videos / union of the two concepts' videos
    A, B = e
    union = len(concept_vids[A] | concept_vids[B]) or 1
    cor = co_occ[e] / union
    if svr >= 0.5 or cor >= 0.5: cls[e] = "CO_TAUGHT"
    elif vr >= 0.80: cls[e] = "REVERSED"
    elif vr >= 0.60: cls[e] = "HIGH_VIOL"
    else: cls[e] = "OK"

counts = collections.Counter(cls.values())
print("=== Edge class counts ===")
for k in ["OK","HIGH_VIOL","REVERSED","CO_TAUGHT","UNUSED"]:
    print(f"  {k:10}: {counts.get(k,0)}")

def agg(pred_viol):
    tv = ta = 0
    for e in edges:
        if cls[e] in ("CO_TAUGHT","UNUSED"): continue   # removed when cleaning
        ta += appl[e]; tv += pred_viol(e)
    return tv, ta

raw_v = sum(viol[e] for e in edges); raw_a = sum(appl[e] for e in edges)
print(f"\nRaw behavioral PVR: {raw_v/max(1,raw_a)*100:.2f}%  ({raw_v:,}/{raw_a:,}, all edges)")

# Cleaning: remove CO_TAUGHT/UNUSED; flip REVERSED (new violations = appl - viol - equal)
def cleaned_viol(e):
    if cls[e] == "REVERSED": return appl[e] - viol[e] - equal[e]
    return viol[e]
cv, ca = agg(cleaned_viol)
print(f"Cleaned PVR: {cv/max(1,ca)*100:.2f}%  ({cv:,}/{ca:,}, co-taught edges removed + reversed edges flipped)")

# Conservative: only remove CO_TAUGHT/REVERSED (no flipping), to see the lower bound
def conserv_viol(e):
    return 0 if cls[e] == "REVERSED" else viol[e]
def agg2():
    tv = ta = 0
    for e in edges:
        if cls[e] in ("CO_TAUGHT","UNUSED","REVERSED"): continue
        ta += appl[e]; tv += viol[e]
    return tv, ta
c2v, c2a = agg2()
print(f"Most conservative PVR: {c2v/max(1,c2a)*100:.2f}%  ({c2v:,}/{c2a:,}, suspicious edges dropped as well)")

# Samples: representative edges per class
print("\n=== REVERSED: likely reversed direction (appl≥500, top 8 by violation rate) ===")
rev = sorted([e for e in edges if cls[e]=="REVERSED" and appl[e]>=500],
             key=lambda e: viol[e]/appl[e], reverse=True)[:8]
for e in rev:
    A,B=e; g=statistics.median(gaps[e]) if gaps[e] else 0
    print(f"  viol {viol[e]/appl[e]*100:4.0f}%  co_occ videos {co_occ[e]:3}  median gap {g:5.1f} days  {A} -> {B}")

print("\n=== CO_TAUGHT: likely co-taught in same course (appl≥500, top 6 by co-teaching rate) ===")
ct = sorted([e for e in edges if cls[e]=="CO_TAUGHT" and appl[e]>=500],
            key=lambda e: same_vid[e]/appl[e], reverse=True)[:6]
for e in ct:
    A,B=e
    print(f"  same-video first contact {same_vid[e]/appl[e]*100:4.0f}%  co-occurring videos {co_occ[e]:3}  {A} -> {B}")

# ---- Export per-edge audit details to CSV (for manual adjudication) ----
import csv
out = BASE + "/mooccube/prereq_edges_audit.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as fo:
    w = csv.writer(fo)
    w.writerow(["class","prereq_A","dependent_B","appl","viol","viol_rate",
                "same_vid_first","same_vid_rate","co_occ_videos","median_gap_days","suggested_action"])
    action = {"REVERSED":"Manually check direction (is it semantically B->A?); if confirmed, flip or delete",
              "CO_TAUGHT":"Co-taught in same course, order meaningless; excluded from PVR",
              "UNUSED":"No user consumption, no PVR contribution; can be ignored",
              "HIGH_VIOL":"Keep; violation rate is high, worth noting in the paper",
              "OK":"Keep"}
    # Sort by class (suspicious first), then by applicable count
    order = {"REVERSED":0,"CO_TAUGHT":1,"UNUSED":2,"HIGH_VIOL":3,"OK":4}
    for e in sorted(edges, key=lambda e:(order[cls[e]], -appl[e])):
        A,B = e; ap = appl[e]
        vr = viol[e]/ap if ap else 0
        svr = same_vid[e]/ap if ap else 0
        g = statistics.median(gaps[e]) if gaps[e] else 0
        w.writerow([cls[e],A,B,ap,viol[e],f"{vr:.3f}",same_vid[e],f"{svr:.3f}",
                    co_occ[e],f"{g:.2f}",action[cls[e]]])
print(f"\nExported audit details: {out}  ({len(edges)} edges total)")
