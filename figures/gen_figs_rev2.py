#!/usr/bin/env python3
"""Revision 2: regenerate all data figures as vector PDFs (Type-42 fonts, no in-figure titles).
Values come from the result files used in the paper (Table 2: three-seed means; Fig. tradeoff and tau: seed 2020)."""
import json, os, matplotlib
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch, Ellipse
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
                            "font.size": 9, "axes.titlesize": 9, "legend.fontsize": 8})
OUT = BASE + "/paper/figs/"
BLUE, RED, GREEN, GREY = "#1f5fbf", "#c62828", "#2e7d32", "#555555"

# ---------- accuracy vs PVR (Table 2 values) ----------
rows = {"MOOCCube": [("Popularity", 0.0097, 56.3), ("ItemKNN", 0.1458, 46.3), ("BPR", 0.1081, 46.89),
                     ("LightGCN", 0.1061, 47.55), ("SASRec", 0.5905, 61.66)],
        "Junyi": [("Popularity", 0.0288, 91.8), ("ItemKNN", 0.1582, 72.1), ("BPR", 0.1348, 77.59),
                  ("LightGCN", 0.0553, 90.89), ("SASRec", 0.2761, 70.56)]}
offs = {("MOOCCube", "BPR"): (-32, -14), ("MOOCCube", "LightGCN"): (22, 8), ("MOOCCube", "ItemKNN"): (18, -10),
        ("MOOCCube", "Popularity"): (6, 4), ("MOOCCube", "SASRec"): (-14, 8),
        ("Junyi", "Popularity"): (6, 4), ("Junyi", "LightGCN"): (6, -10), ("Junyi", "BPR"): (6, 4),
        ("Junyi", "ItemKNN"): (6, 4), ("Junyi", "SASRec"): (6, 4)}
LEADER = {("MOOCCube", "BPR"), ("MOOCCube", "LightGCN"), ("MOOCCube", "ItemKNN")}
fig, ax = plt.subplots(figsize=(3.5, 2.5))
for ds, mk, col in (("MOOCCube", "o", BLUE), ("Junyi", "s", RED)):
    xs = [r[1] for r in rows[ds]]; ys = [r[2] for r in rows[ds]]
    ax.scatter(xs, ys, marker=mk, s=28, color=col, label=ds, zorder=3)
    for name, x, y in rows[ds]:
        ax.annotate(name, (x, y), xytext=offs[(ds, name)], textcoords="offset points", fontsize=7.5, color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.6, shrinkA=0, shrinkB=3) if (ds, name) in LEADER else None)
ax.set_xlabel("NDCG@10 (accuracy)"); ax.set_ylabel("PVR@10 (%)")
ax.set_xlim(-0.02, 0.66); ax.set_ylim(35, 100); ax.grid(alpha=.3)
ax.legend(loc="lower right", frameon=True)
fig.tight_layout(); fig.savefig(OUT + "pvr_bars.pdf"); plt.close(fig)

# ---------- accuracy-prerequisite trade-off ----------
r = json.load(open(BASE + "/mooccube/scaffold_rl_v2_result.json"))
cur = r["curve"]; pp = r["postproc"]; ceil = r["diag"]["pvr0_ndcg_ceiling"]
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.plot([100 * c["pvr"] for c in cur], [c["ndcg"] for c in cur], "-o", color=BLUE, ms=5, lw=1.6,
        label=r"Soft-score reranker ($\lambda$ sweep)")
lab = {0.0: (r"$\lambda$=0", (0, -13)), 0.25: (r"$\lambda$=0.25", (0, -13)), 0.5: (r"$\lambda$=0.5", (0, -13)),
       1.0: (r"$\lambda$=1", (10, -11)), 2.0: (r"$\lambda\geq$2", (16, -6))}
for c in cur:
    if c["lambda"] in lab:
        t, off = lab[c["lambda"]]
        ax.annotate(t, (100 * c["pvr"], c["ndcg"]), xytext=off, textcoords="offset points", ha="center",
                    fontsize=7.5, color=BLUE)
ax.plot([100 * pp["pvr"]], [pp["ndcg"]], "^", color=RED, ms=9, label="Hard prerequisite filter", zorder=4)
ax.annotate("hard filter", (100 * pp["pvr"], pp["ndcg"]), xytext=(-6, 9), textcoords="offset points", fontsize=7.5, color=RED)
ax.axhline(cur[0]["ndcg"], ls=":", color=GREY, lw=1.4, label="LightGCN, unfiltered")
ax.set_xlabel("PVR@10 (%), lower is better"); ax.set_ylabel("NDCG@10, higher is better")
ax.set_xlim(-3, 52); ax.set_ylim(0.05, 0.125); ax.grid(alpha=.3)
ax.legend(loc="upper left", fontsize=7, framealpha=0.95)
ax.text(0.98, 0.04, f"PVR = 0 oracle ceiling: NDCG {ceil:.2f} (off scale)", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=7, color=GREEN)
fig.tight_layout(); fig.savefig(OUT + "tradeoff.pdf"); plt.close(fig)

# ---------- tau sensitivity ----------
t = json.load(open(BASE + "/mooccube/tau_sensitivity_result.json"))
tau = [x["tau"] for x in t]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
a1.plot(tau, [100 * x["pvr"] for x in t], "-o", color=BLUE, ms=4, label="LightGCN PVR@10")
a1.plot(tau, [100 * x["holdout_viol"] for x in t], "--s", color=RED, ms=4, label="True next item inconsistent")
a1.set_xlabel(r"Readiness threshold $\tau$"); a1.set_ylabel("%"); a1.grid(alpha=.3); a1.legend(loc="upper left")
a1.text(-0.16, 1.02, "(a)", transform=a1.transAxes, fontweight="bold")
a2.plot(tau, [x["ndcg_raw"] for x in t], "-o", color=BLUE, ms=4, label="Unfiltered")
a2.plot(tau, [x["ndcg_pf"] for x in t], "-^", color=RED, ms=4, label="Hard filter")
a2.set_xlabel(r"Readiness threshold $\tau$"); a2.set_ylabel("NDCG@10"); a2.grid(alpha=.3); a2.legend(loc="center right")
a2.text(-0.18, 1.02, "(b)", transform=a2.transAxes, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT + "tau.pdf"); plt.close(fig)

# ---------- worked example ----------
import math
fig, ax = plt.subplots(figsize=(3.5, 2.2)); ax.axis("off"); ax.set_xlim(0, 14.6); ax.set_ylim(0, 6.3)
EW, EH = 2.6, 1.35                                    # ellipse width and height (data units)
pos = {"c1": (1.35, 3), "c2": (4.15, 4.45), "c3": (4.15, 1.55), "c4": (6.95, 3), "c5": (10.05, 3)}
mastered = {"c1", "c2", "c3"}
label = {"c1": "$c_1$\naddition", "c2": "$c_2$\nmultiplication", "c3": "$c_3$\nfractions", "c4": "$c_4$\nalgebra", "c5": "$c_5$\nequations"}


def rim(p, q):
    """Point on the ellipse around p in the direction of q (arrows start and end on the outline)."""
    dx, dy = q[0] - p[0], q[1] - p[1]; t = 1 / math.sqrt((dx / (EW / 2)) ** 2 + (dy / (EH / 2)) ** 2)
    return (p[0] + t * dx, p[1] + t * dy)


for a, b in [("c1", "c2"), ("c1", "c3"), ("c2", "c4"), ("c3", "c4"), ("c4", "c5")]:
    ax.add_patch(FancyArrowPatch(rim(pos[a], pos[b]), rim(pos[b], pos[a]), arrowstyle="-|>", mutation_scale=9,
                                 color="#888", lw=1, shrinkA=1, shrinkB=1, zorder=1))
for c, (x, y) in pos.items():
    ax.add_patch(Ellipse((x, y), EW, EH, facecolor="#c8f0d2" if c in mastered else "white", edgecolor="#333", lw=1, zorder=2))
    ax.text(x, y, label[c], ha="center", va="center", fontsize=6.0, zorder=3)
for txt, fc, ec, tgt in (("item A: $c_4$", "#dbe7fb", BLUE, "c4"), ("item B: $c_5$", "#fbe0e0", RED, "c5")):
    cx = pos[tgt][0]
    ax.add_patch(FancyBboxPatch((cx - 1.2, 5.05), 2.4, 0.75, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1))
    ax.text(cx, 5.43, txt, ha="center", va="center", fontsize=6.4, color=ec)
    ax.add_patch(FancyArrowPatch((cx, 5.0), (cx, pos[tgt][1] + EH / 2), arrowstyle="-|>", mutation_scale=8, color=ec,
                                 lw=.8, ls="--", shrinkA=0, shrinkB=1))
ax.add_patch(Circle((0.5, 0.45), 0.25, facecolor="#c8f0d2", edgecolor="#333"))
ax.text(0.9, 0.45, "mastered ($M_u$)", fontsize=6.8, va="center")
ax.add_patch(Circle((5.4, 0.45), 0.25, facecolor="white", edgecolor="#333"))
ax.text(5.8, 0.45, "not mastered", fontsize=6.8, va="center")
ax.text(11.6, 4.6, "Ready(A) = 2/2 = 1\n$\\Rightarrow$ ready", fontsize=6.6, color=BLUE, va="center")
ax.text(11.6, 1.5, "Ready(B) = 0/1 = 0\n$\\Rightarrow$ violation", fontsize=6.6, color=RED, va="center")
fig.tight_layout(pad=0.2); fig.savefig(OUT + "worked_example.pdf"); plt.close(fig)
print("saved pvr_bars.pdf, tradeoff.pdf, tau.pdf, worked_example.pdf")
