# The Accuracy–Prerequisite Tension in Educational Recommendation — code

Code for the article *The Accuracy–Prerequisite Tension in Educational Recommendation: A Cross-Dataset Audit*
(C.-H. Lee, C.-W. Hsu, Y.-C. Lee; IEEE Access, under review).

The code implements a prerequisite-aware evaluation layer for recommender systems: readiness, the
Prerequisite Violation Rate (PVR), prerequisite-aware Concept Coverage (ConceptCov), and readiness-aware
re-ranking with a violation budget (Algorithms 1 and 2 of the paper). It also contains every script used to
produce the tables and figures.

## Environment
Python 3.12; PyTorch 2.11 (CUDA); RecBole 1.2.0; NumPy 1.26; SciPy 1.12; scikit-learn 1.5; pandas 2.2;
matplotlib 3.11; tqdm. One GPU is used for training and scoring. See `requirements.txt`.

## Running time and requirements
Measured on the machine used for the paper (Intel Xeon w5-3433, 32 threads, 251 GB RAM; one NVIDIA RTX PRO 6000;
Ubuntu, Python 3.12, CUDA 12.8) in a full re-run from the raw data on 2026-11-21:

| Stage | Scripts | Time |
|---|---|---|
| Preprocessing | 4 scripts in `data_prep/` | 1.5 min |
| LightGCN checkpoint | `train_lightgcn_t.py` | 8 min |
| Quick analyses on that checkpoint | `run_g2g3_t.py`, `scaffold_rl_v2.py`, `tau_sensitivity.py`, `wilcoxon_test.py` | 0.5 min each |
| Baselines | `generic_baselines.py` | 5 min |
| Three-seed runs | `seeds_experiment.py` / `threeseed_all.py` | 34 min / **1 h 55 min** |
| SASRec | `sasrec_run.py` / `wilcoxon_extra.py` | 26 min / 27 min |
| ConceptCov | `conceptcov.py` / `junyi_conceptcov.py` | 12 min / 1 min |
| Junyi analyses | `junyi_pipeline.py`, `junyi_mastery_sens.py` | 1 min each |
| Metric validity, SASRec audit | `revision_metrics.py` / `sasrec_audit.py` | 13 min / 0.5 min |
| Revision analyses | 3 scripts in `revision/` | 7.5 min |
| **Total (full re-run)** | | **about 4.3 h** |
| Quick check only (revision analyses on existing checkpoints) | | about 8 min |

Peak memory observed while monitoring the run (including SASRec training): 6.1 GB RAM and 5.3 GB GPU memory;
preprocessing needs 0.5 GB RAM. Disk: 3.9 GB for the extracted MOOCCube files and about 1.7 GB for the
generated datasets, checkpoints, and results. A CUDA GPU is required for training (`device: cuda`).

## Data
| Dataset | Source | Licence |
|---|---|---|
| MOOCCube | https://github.com/thukg/MOOCCube (archive `http://lfs.aminer.cn/misc/moocdata/data/MOOCCube.zip`; files `relations/`, `additional_information/user_video_act.json`) | CC0-1.0 |
| Junyi Academy | USTC mirror `http://base.ustc.edu.cn/data/ktbd/junyi/` (`train.json`, `test.json`, `vertex_id2idx`, `prerequisite.json`) | see source |

The data are not redistributed here. SHA-256 checksums of the files used are listed in `verification/verification_report.md`.

## Directory layout
All scripts read and write under a working root given by the environment variable `SCAFFOLD_ROOT`
(default `/workspace`):
```
$SCAFFOLD_ROOT/mooccube/MOOCCube/          raw MOOCCube files
$SCAFFOLD_ROOT/junyi/                      raw Junyi files
$SCAFFOLD_ROOT/mooccube/recbole/dataset/   RecBole atomic files (written by data_prep)
$SCAFFOLD_ROOT/mooccube/, junyi/, paper/   result JSON files and checkpoints (written by experiments)
```

## Pipeline
1. **Preprocessing** (`data_prep/`, paper Sec. IV-C)
   - `build_recbole_temporal.py`: MOOCCube CS/Math scope, earliest view per (learner, video), RecBole atomic file.
   - `g2b_edge_audit.py`: audits the 905 golden prerequisite edges (reversed, co-taught, unused); writes `prereq_edges_audit.csv`.
   - `build_pvr_assets.py`: item–concept map and the audited 803-edge graph (`pvr_assets.json`).
   - `build_junyi.py`: Junyi atomic file (first attempt per exercise, sequence position as time) and assets.
2. **Experiments** (`experiments/`; temporal leave-one-out and 5-core filtering are configured in each script)

   Run `train_lightgcn_t.py` first: it trains the seed-2020 LightGCN checkpoint (`recbole/saved_t/`) that
   `run_g2g3_t.py`, `scaffold_rl_v2.py`, `tau_sensitivity.py`, and `wilcoxon_test.py` load. The revision scripts
   load the checkpoints written by `revision_metrics.py` and `sasrec_run.py` (newest file per model and dataset).

| Paper item | Script(s) |
|---|---|
| Table 4 (accuracy and PVR; three seeds) and Table 8 (hard-filter cost) | `threeseed_all.py`, `seeds_experiment.py`, `generic_baselines.py`, `junyi_pipeline.py`, `sasrec_run.py` |
| Table 5 (behavioral ceiling) and the PVR = 0 oracle | `scaffold_rl_v2.py`, `junyi_pipeline.py` |
| Tables 6–7 (ConceptCov); catalog-wide ConceptCov (Sec. V-C)¹ | `conceptcov.py`, `junyi_conceptcov.py`, `revision_batch2.py` |
| Wilcoxon tests (Sec. V-D) | `wilcoxon_test.py`, `wilcoxon_extra.py` |
| Table 9 and Fig. 4 (re-ranking frontier) | `scaffold_rl_v2.py` |
| Table 10 and Fig. 5 (threshold); correctness-based mastery (Sec. V-F) | `tau_sensitivity.py`, `junyi_mastery_sens.py` |
| Table 11 (metric validity and candidate sufficiency) | `revision_metrics.py` |
| Table 12 (history length) and the SASRec audit (Sec. V-G) | `sasrec_audit.py` |
| Raw vs. audited graph inconsistency (Appendix A) | `g2_behavioral_pvr.py` |
| G2/G3 checkpoint used by several analyses | `run_g2g3_t.py` |

¹ The catalog-wide ConceptCov values come from part (1) of `revision_batch2.py`; its part (2) was superseded by
`wilcoxon_extra.py`, so that run left no JSON file and the values are kept in `results/paper/revision_batch2_conceptcov.log`.

The result files used for the manuscript are in `results/` (same relative paths as under `$SCAFFOLD_ROOT`).

3. **Revision analyses** (`revision/`; use the seed-2020 checkpoints, no retraining)

| Paper item | Script |
|---|---|
| Table 13 (mastery inference, raw vs. audited graph), transitive PVR in Table 11 | `rev1_variants.py` |
| Table 14 (budgeted and backfilled re-ranking), runtime (Sec. V-I) | `rev1_mitigation.py` |
| Readiness and correctness on Junyi (Sec. V-J) | `rev1_junyi_outcome.py` |
| Check a re-run against the reference results (`revision/reference/`; copy them into `$SCAFFOLD_ROOT/paper/rev1/` first). A full re-run with this repository reproduced all 755 reference values bit-identically | `compare_results.py` |

4. **Figures** (`figures/`): `gen_figs_rev2.py` (Figs. 2–5, vector PDF) and `workflow.tex` (Fig. 1, TikZ).

## Reproducibility notes
- Trainable models use seeds 2020–2022 (RecBole `reproducibility=True`); deterministic models are run once.
- Transitive prerequisites are computed by reachability, which is order-independent on graphs with cycles
  (the audited MOOCCube graph retains 31 cycle-closing edges).
- `revision/compare_results.py` compares a fresh run of the three revision scripts with the reference
  outputs in `revision/reference/`. It requires identical key sets (missing or extra fields fail), compares
  integers exactly and floats with a relative tolerance of 1e-9, reports how many values are bit-identical,
  prints the software versions, and exits with status 1 on any problem. Timing fields are excluded.
- Timing fields in `rev1_mitigation.py`: `filter_us_per_user` measures only the readiness checks of the
  50-item pool; `rerank_us_per_user` measures the whole post-processing step for one learner (mastery set,
  readiness checks, and the budget-1 list of Algorithm 2). Model scoring and candidate retrieval are excluded.
- Not included: the raw datasets (see Data) and the trained checkpoints; the scripts retrain them with fixed seeds.
- GPU non-determinism: re-training on the same machine and software reproduced all preprocessing outputs and most
  results bit-for-bit; a few SASRec and LightGCN values differed in the sixth significant digit (largest absolute
  difference 0.004 percentage points), which does not change any number printed in the paper. Such differences
  are consistent with non-deterministic floating-point accumulation on the GPU and may be larger on other GPUs or CUDA versions.
  The strict checkers (`revision/compare_results.py`, `verification/compare_archive.py`) report them as differences;
  `verification/paper_precision_check.py` checks whether they still round to the numbers printed in the paper.
- `results/paper/revision_metrics_result.json` was regenerated after replacing an order-dependent transitive
  closure in `revision_metrics.py` with the reachability closure used elsewhere; only its `PVR_transitive` field
  changed, and it now equals the transitive PVR of `rev1_variants.py` reported in Table 11.

## Verification
`verification/verification_report.md` records two verifications with this repository: a quick re-run of the
revision analyses and a full re-run from the raw data (environment, input checksums, scope, results), with logs.

Three checkers compare a re-run with the results used in the paper. The two in `verification/` can be run from
any directory; they locate the archived results relative to their own path.

| Checker | Compares | When to use | Pass criterion |
|---|---|---|---|
| `revision/compare_results.py` | the three revision result files with `revision/reference/` | after the revision scripts (run it in `$SCAFFOLD_ROOT/paper/rev1`, see above) | strict: identical keys, floats within 1e-9 |
| `verification/compare_archive.py` | all 15 main result files with `results/` | after a full re-run | strict: floats within 1e-9 |
| `verification/paper_precision_check.py` | every value the two strict checkers report as different | after a full re-run, when a strict checker reports differences | each value rounds to the same number at the precision printed in the paper |

```bash
export SCAFFOLD_ROOT=/path/to/rerun_root          # the root used for the full re-run
python3 verification/compare_archive.py            # strict; exit 1 if any value differs
python3 verification/paper_precision_check.py      # exit 0 if all differences vanish at the printed precision
```

`paper_precision_check.py` reads the archived values from `results/` and `revision/reference/` and the re-run
values from `$SCAFFOLD_ROOT` (three-seed, seed, and SASRec results, and `paper/rev1/rev1_mitigation_result.json`
and `rev1_variants_result.json`, which depend on re-trained checkpoints). For every value that differs it prints one line:

* `OK` - the value rounds to the same number at the printed precision: PVR in percent with 2 decimals
  (Table 4) or 1 decimal (Tables 11, 13, 14), NDCG/HR with 4 decimals, NDCG change versus the unfiltered
  list in percent with 1 decimal (recomputed from the re-run values);
* `NOTE` - same at the printed precision but different at 2 decimals (for values printed with 1 decimal);
* `FAIL` - the printed number would change.

It ends with a summary line and exits with 0 if there is no `FAIL`. Values that do not appear in the paper are
checked with the same rules. Our full re-run gave 35 checked values: 34 `OK`, 1 `NOTE`, 0 `FAIL`
(`verification/paper_precision_check_full_B_261121.log`). If you obtain a `FAIL`, the difference is larger than
what we observed on our machine; please open an issue with the output and your GPU and CUDA versions.

## Licence
MIT (see `LICENSE`).
