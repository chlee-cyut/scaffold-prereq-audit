#!/usr/bin/env python3
"""
G2 early check: behavioral PVR (prerequisite violation rate)
Question: do users consume concept B, which depends on A, before consuming prerequisite concept A?

Definition (based on "first consumption time of a concept"; clean and reproducible):
  Prerequisite edge A -> B means "A is a prerequisite of B".
  For a given user, if the user consumed both concepts, the edge is "applicable";
  if first_time[B] < first_time[A] (saw B before learning A) => violation.
  The same video teaching both A and B (equal times) => not a violation.

  Behavioral PVR = violated applicable edge instances / all applicable edge instances (summed over users).

Decision thresholds (proposal): >30% story holds; 15-30% publishable but conservative; <15% premise fails, change story.
"""
import os
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")   # root of the data/working-directory layout (see README)
import json, collections, statistics

ROOT = BASE + "/mooccube/MOOCCube/"

# ---- Prerequisite graph: col0 = prerequisite A, col1 = dependent B ----
prereqs_of = collections.defaultdict(set)   # B -> {A...}
pre_concepts = set()
with open(ROOT+"relations/prerequisite-dependency.json", encoding="utf-8", errors="replace") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[0] and p[1] and p[0] != p[1]:
            A, B = p[0].strip(), p[1].strip()
            prereqs_of[B].add(A)
            pre_concepts.add(A); pre_concepts.add(B)
n_edges = sum(len(v) for v in prereqs_of.values())
print(f"Prerequisite edges: {n_edges}  concepts: {len(pre_concepts)}")

# ---- video -> prerequisite concepts it covers ----
vid_pre = collections.defaultdict(set)
with open(ROOT+"relations/video-concept.json", encoding="utf-8", errors="replace") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            v, c = p[0].strip(), p[1].strip()
            if c in pre_concepts:
                vid_pre[v].add(c)
print(f"Videos with prerequisite concepts: {len(vid_pre):,}")

# ---- Stream user_video_act.json, one user per line ----
tot_applicable = 0
tot_violation = 0
users_scored = 0
user_pvr_list = []            # per-user PVR for each user with applicable edges
users_gt0 = 0                 # number of users with at least one violation
edge_viol = collections.Counter()
edge_appl = collections.Counter()

processed = 0
with open(ROOT+"additional_information/user_video_act.json", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        processed += 1
        # first consumption time of each prerequisite concept (ISO strings compare directly)
        first_t = {}
        for a in r.get("activity", []):
            v = a.get("video_id"); t = a.get("local_start_time")
            if not v or not t or v not in vid_pre:
                continue
            for c in vid_pre[v]:
                if c not in first_t or t < first_t[c]:
                    first_t[c] = t
        if not first_t:
            continue
        # check applicable edges
        u_appl = 0; u_viol = 0
        for B, tB in first_t.items():
            for A in prereqs_of.get(B, ()):
                if A in first_t:
                    u_appl += 1
                    tot_applicable += 1
                    edge_appl[(A, B)] += 1
                    if tB < first_t[A]:          # saw B before A => violation
                        u_viol += 1
                        tot_violation += 1
                        edge_viol[(A, B)] += 1
        if u_appl:
            users_scored += 1
            user_pvr_list.append(u_viol / u_appl)
            if u_viol:
                users_gt0 += 1

print(f"Users processed: {processed:,}")
print("\n" + "="*56)
print("Behavioral PVR results")
print("="*56)
pvr = tot_violation / max(1, tot_applicable)
print(f"Applicable prerequisite edge instances (across users): {tot_applicable:,}")
print(f"Of which violated: {tot_violation:,}")
print(f">>> Overall behavioral PVR = {pvr*100:.2f}% <<<")
print()
print(f"Users with applicable edges: {users_scored:,}")
print(f"  Users with at least one violation: {users_gt0:,} ({users_gt0/max(1,users_scored)*100:.1f}%)")
if user_pvr_list:
    print(f"  Mean per-user PVR: {statistics.mean(user_pvr_list)*100:.2f}%")
    print(f"  Median per-user PVR: {statistics.median(user_pvr_list)*100:.2f}%")

# Decision
print("\nDecision (proposal thresholds):")
if pvr >= 0.30:
    print(f"  ✅ {pvr*100:.1f}% ≥ 30% — premise holds, proceed as planned")
elif pvr >= 0.15:
    print(f"  🟡 15% ≤ {pvr*100:.1f}% < 30% — publishable but argue conservatively")
else:
    print(f"  ❌ {pvr*100:.1f}% < 15% — premise fails, change story (cold start / cross-domain subgroup)")

# Edge level: which prerequisite edges are violated most (high rate and many samples)
print("\nPrerequisite edges with highest violation rate (applicable ≥500 times):")
rows = [((A,B), edge_viol[(A,B)], edge_appl[(A,B)]) for (A,B) in edge_appl if edge_appl[(A,B)]>=500]
rows.sort(key=lambda x: x[1]/x[2], reverse=True)
for (A,B), v, ap in rows[:8]:
    print(f"  {v/ap*100:5.1f}%  ({v:,}/{ap:,})  {A} -> {B}")
