#!/usr/bin/env python3
"""Build MOOCCube PVR assets: item->concept map (vid_pre) and the audited prerequisite graph (clean_edges).
vid_pre: for videos of CS/Math courses (same scope as build_recbole_temporal.py), video-concept relations
restricted to the concepts of the golden prerequisite file (425 concepts);
clean_edges: golden edges minus those the audit labels REVERSED, CO_TAUGHT, or UNUSED (prereq_edges_audit.csv)."""
import os
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")   # root of the data/working-directory layout (see README)
import csv, json, collections, os
BASE = os.environ.get("SCAFFOLD_ROOT", BASE)
ROOT = BASE + "/mooccube/MOOCCube/"
audit = list(csv.DictReader(open(BASE + "/mooccube/prereq_edges_audit.csv", encoding="utf-8-sig")))
concepts = {x for r in audit for x in (r["prereq_A"], r["dependent_B"])}
clean = [[r["prereq_A"], r["dependent_B"]] for r in audit if r["class"] not in ("REVERSED", "CO_TAUGHT", "UNUSED")]
CSMATH = {"\u8ba1\u7b97\u673a\u79d1\u5b66\u6280\u672f", "\u6570\u5b66"}  # MOOCCube discipline names (data): "Computer Science and Technology", "Mathematics"
fld = lambda c: c.rsplit("_", 1)[-1] if c.startswith("K_") else None
cf = collections.defaultdict(collections.Counter)
for line in open(ROOT + "relations/course-concept.json", encoding="utf-8", errors="replace"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2 and fld(p[1].strip()): cf[p[0].strip()][fld(p[1].strip())] += 1
courses = {c for c, cnt in cf.items() if cnt.most_common(1)[0][0] in CSMATH}
scope = set()
for line in open(ROOT + "relations/course-video.json", encoding="utf-8", errors="replace"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2 and p[0].strip() in courses: scope.add(p[1].strip())
vid = collections.defaultdict(list)
for line in open(ROOT + "relations/video-concept.json", encoding="utf-8", errors="replace"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2 and p[0].strip() in scope and p[1].strip() in concepts and p[1].strip() not in vid[p[0].strip()]:
        vid[p[0].strip()].append(p[1].strip())
json.dump({"vid_pre": dict(vid), "clean_edges": clean}, open(BASE + "/mooccube/pvr_assets.json", "w"), ensure_ascii=False)
print(f"videos {len(vid)}, concepts {len(concepts)}, clean edges {len(clean)}")
