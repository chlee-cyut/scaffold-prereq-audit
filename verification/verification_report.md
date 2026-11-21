# Verification report

## Re-run of the revision analyses (2026-11-20)

| Item | Value |
|---|---|
| Date | 2026-11-20 (UTC+8) |
| Code | this repository, `revision/rev1_mitigation.py`, `rev1_variants.py`, `rev1_junyi_outcome.py`, run unchanged from the released files |
| Environment | Python 3.12.3, NumPy 1.26.4, SciPy 1.12.0, PyTorch 2.11.0+cu128, RecBole 1.2.0 (printed by the checker); Intel Xeon w5-3433, NVIDIA RTX PRO 6000 |
| Inputs | raw data with the SHA-256 checksums below; seed-2020 checkpoints trained by the experiment scripts |
| Check | `revision/compare_results.py` against the values reported in the manuscript |
| Result | 755 of 755 reference values match, all bit-identical: mitigation 450/450, variants 285/285, Junyi outcome 20/20; 0 missing, 0 extra keys; exit code 0 |
| Logs | `compare_results_261120.log` (checker output), `run_*_261120.log` (script output, progress bars removed) |

The checker allows a relative tolerance of 1e-9 for floats, but the log reports how many values are bit-identical;
in this run all were. Timing fields are excluded from the comparison because they depend on the machine.

## Full re-run from the raw data (2026-11-21)

| Item | Value |
|---|---|
| Setup | fresh directory (`SCAFFOLD_ROOT`), code unpacked from the release archive, raw data only (checksums below); Ubuntu 24.04.1 |
| Steps | 4 preprocessing scripts, 16 experiment scripts, 3 revision scripts; all exited with code 0; total 4.3 h (`run_full_B_progress_261121.txt`) |
| Preprocessing outputs | MOOCCube and Junyi interaction files: byte-identical to the originals; item-concept maps and audited graphs: identical in content; graph audit: identical in every field (the free-text `suggested_action` column has since been translated to English) |
| Main results (`verification/compare_archive.py`) | 15 result files, 481 numeric values: 471 bit-identical, 10 differ (`compare_archive_full_B_261121.log`) |
| Revision analyses on the re-trained checkpoints (`revision/compare_results.py`) | 755 values: 731 bit-identical, 24 differ (`compare_results_full_B_261121.log`) |
| Effect on the manuscript | none at the printed precision (see below) |

### Exact match versus match at the paper's precision

| Check | Values | Result |
|---|---|---|
| Strict comparison, main results (`compare_archive.py`) | 481 | 471 bit-identical, 10 differ; the script reports FAILED (exit 1) by design |
| Strict comparison, revision analyses (`revision/compare_results.py`) | 755 | 731 bit-identical, 24 differ; the script reports FAILED (exit 1) by design |
| Total | 1,236 | 1,202 bit-identical (97.2%), 34 differ; largest absolute difference 3.9e-5 (0.004 percentage points) |
| Paper-precision check (`paper_precision_check.py`) | 34 differing values + 1 derived NDCG change | all 35 round to the same number at the precision printed in the manuscript (exit 0); `paper_precision_check_full_B_261121.log` |

The strict checkers are kept strict on purpose; they are not relaxed to report a pass. The paper-precision check
applies the printed precision of each table (PVR: 2 decimals in Table 4, 1 decimal in Tables 11, 13, 14; NDCG:
4 decimals; NDCG change: 1 decimal). One value also differs at the second decimal: the SASRec PVR on MOOCCube
under demote-10 (44.29% vs 44.28%), which is not printed in the manuscript (Table 14 shows demote-20).

All 34 differing values come from re-trained SASRec models and from one LightGCN seed (seed 2022 in
`seeds_result.json`); every analysis that loads a fixed checkpoint (quick check above) is bit-identical. This
pattern is consistent with non-deterministic floating-point accumulation on the GPU during training, but the logs
alone do not prove the cause.

One script problem was found and fixed during this run: `revision_metrics.py` computed its `PVR_transitive` field
with an order-dependent closure on the cyclic MOOCCube graph; it now uses the reachability closure, its output
equals the transitive PVR of `rev1_variants.py` (Table 11), and the archived file was regenerated (all other
fields unchanged).

## Scope

| Covered | How |
|---|---|
| Tables 11 (transitive PVR), 13, 14; Sec. V-I; Sec. V-J | automatic re-run and value-by-value comparison (above) |
| Tables 4–12 and the numbers quoted in Sec. V | every table cell was traced by hand to the stored result files in `results/` (three rounding or transcription slips were found and corrected in the manuscript) |
| Tables 4–12 re-trained from the raw data | full re-run (section above): every printed number reproduced |

## Input data checksums (SHA-256)

MOOCCube (`http://lfs.aminer.cn/misc/moocdata/data/MOOCCube.zip`, linked from https://github.com/thukg/MOOCCube):
```
51f4619ed41bfa2dc1bdfd2e05cf4c08df3a0eb5b0e687c3cb4ddbfd38659f14  relations/prerequisite-dependency.json
4ee112ef110039475ba9750cdbc152c60157b39b87b8b89addcce5e96c824377  relations/video-concept.json
bfc18b50ffc5c4d7f96e19f7528fd96f27852cb30b513e3648ac792ce7837f90  relations/course-concept.json
33f878cb0908b2a50309d0cf203f9cb84b7b5f82e7de4843c4836aedc1bdb65f  relations/course-video.json
d1643e2720274f6e9e339959d2d9749d945c1b89f740a8a613c9c90091a83f97  additional_information/user_video_act.json
```
Junyi Academy (USTC mirror, `http://base.ustc.edu.cn/data/ktbd/junyi/`):
```
9a50d9bdbfbc19cd4682cc0303fb0ff5bdb017a955c10d6f1922a60636d494c1  train.json
c55f0714d9479d35afb25564ef2faf57c17ca46d944a0e509cb3f26780934973  test.json
553271b2804315b879adcd04e7696db61477b0b53fd9033be3e1413136bf4c2c  vertex_id2idx
cd3eaf22c22ebbeb3165068233b7fe16962d1910573f5546628f7612cef435b0  prerequisite.json
```

## Measured running time (seed 2020, N = 50, one CPU core, from `reference_rev1_mitigation_result.json`)

| Data / model | readiness checks (µs/user) | complete post-processing (µs/user) |
|---|---|---|
| MOOCCube BPR | 31.6 | 62.2 |
| MOOCCube LightGCN | 31.2 | 62.0 |
| MOOCCube SASRec | 28.5 | 43.5 |
| Junyi BPR | 17.7 | 26.6 |
| Junyi LightGCN | 16.8 | 26.4 |
| Junyi SASRec | 18.3 | 26.7 |

Complete post-processing = mastery set, readiness checks, and the budget-1 list of Algorithm 2; model scoring and
candidate retrieval are excluded. The manuscript reports both ranges (Sec. V-I; the abstract uses 26–62 µs).
