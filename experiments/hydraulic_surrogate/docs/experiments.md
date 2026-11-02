# Experiment log

All scores below are local chronological validation unless explicitly marked Kaggle. No Kaggle submission yet. Raw data remains unchanged. Generated artifacts live under ignored `artifacts/`.

## setup-001 — 2026-11-02

Hypothesis: create an isolated environment for reproducible modeling. Command: `python3 -m venv .venv && .venv/bin/python -m pip install numpy pandas scikit-learn lightgbm catboost pypdf`. Python 3.14.6 on macOS arm64. Outcome: failed during dependency resolution because pip could not resolve pypi.org; no models trained, seeds/splits/scores/artifacts not applicable. User then specified uv; subsequent environment setup uses uv. An initial `.venv/bin/python -m unittest discover -s tests -v` likewise failed for missing numpy; syntax checks passed. Original scorer SHA256: `6ba062eeb8c2983a490cad411071dfc77831e3f2233278e5a67b68810df90553`.

Each training invocation appends its exact command, configuration, validation metrics and manifest location below. Manifests contain environment/hardware, source snapshots and hashes, input and artifact hashes, case IDs and dates, feature definitions, seeds, hyperparameters and timings, including failure records.

## lgb-basic-001 — 2026-11-02T09:42:16.627301+00:00

Status: failed. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-basic-001 --years 2021 2022 --feature-set basic --trees 180 --sample-fraction 0.3 --threads 6`. Runtime: 0.8s.

Manifest and source snapshot: `artifacts/lgb-basic-001/manifest.json`. Feature ablation: basic; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.


Failure: `              dtype='datetime64[ns, UTC]', freq=None)`

Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## lgb-basic-002 — 2026-11-02T09:42:49.166676+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-basic-002 --years 2021 2022 --feature-set basic --trees 180 --sample-fraction 0.3 --threads 6`. Runtime: 72.6s.

Manifest and source snapshot: `artifacts/lgb-basic-002/manifest.json`. Feature ablation: basic; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.

- 2021: micro AUC **0.87088230**, prevalence baseline 0.54351580; train ['2015-01-01 00:00:00+00:00', '2020-12-25 00:00:00+00:00'], validation ['2021-01-01 00:00:00+00:00', '2021-12-31 00:00:00+00:00']; six-day purge.
- 2022: micro AUC **0.84465480**, prevalence baseline 0.55445194; train ['2015-01-01 00:00:00+00:00', '2021-12-25 00:00:00+00:00'], validation ['2022-01-01 00:00:00+00:00', '2022-12-31 00:00:00+00:00']; six-day purge.

Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## environment-uv-001 — 2026-11-02

User-directed uv environment setup completed in `.venv-ml`, Python 3.11.15. Exact commands, hardware/library versions, network failures, wheel provenance/checksums and successful import/fit checks are recorded in `docs/environment.md`; dependencies pinned in `pyproject.toml`. No competition model or score in these tiny package installation smoke checks. All 16 scorer and feature-integration tests pass with `.venv-ml/bin/python -m unittest discover -s tests -v` (0.296 s). Feature builder's first smoke failed because pandas3 dates used microseconds; explicit nanosecond conversion corrected it, then full data feature smoke passed (612,696 ×231 float32,1.81 s). Same defect was caught and logged in `lgb-basic-001` before training.

## lgb-context-001 — 2026-11-02T09:43:22.065699+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-context-001 --years 2021 2022 --feature-set full --trees 350 --sample-fraction 0.4 --threads 6`. Runtime: 439.2s.

Manifest and source snapshot: `artifacts/lgb-context-001/manifest.json`. Feature ablation: full; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.

- 2021: micro AUC **0.98065317**, prevalence baseline 0.54351580; train ['2015-01-01 00:00:00+00:00', '2020-12-25 00:00:00+00:00'], validation ['2021-01-01 00:00:00+00:00', '2021-12-31 00:00:00+00:00']; six-day purge.
- 2022: micro AUC **0.98435803**, prevalence baseline 0.55445194; train ['2015-01-01 00:00:00+00:00', '2021-12-25 00:00:00+00:00'], validation ['2022-01-01 00:00:00+00:00', '2022-12-31 00:00:00+00:00']; six-day purge.

Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## postprocess-context-001 — 2026-11-02T09:51:33.183558+00:00

Status: complete. Hypothesis: startup costs favor persistent within-case schedules; near-nested Vinje labels support an optional probability ordering projection. Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/analyze_predictions.py lgb-context-001 --run-id postprocess-context-001`. Runtime 8.9s.

No training, seeds or early stopping. Same case splits, input data and base models as source manifests. Centered edge-normalized probability means over windows 3,7,13,25; weights 0,.25,.5,1; optional L2 projection onto Vinje G1 >= G2 >= G3. Select solely on 2021; freeze before reading 2022 labels. 2022 grid scores are diagnostics, never selection criteria.

- lgb-context-001: selected config {'window': 3, 'weight': 0.0, 'projection': True}. 2021 baseline=0.98065317, selected=0.98069869. 2022 baseline=0.98435803, selected=0.98438025. 

Complete grid, environment, source snapshots, source/artifact checksums and selected predictions: `artifacts/postprocess-context-001/analysis.json`. Kaggle submission ID, score, rank and observation time: none.

## lgb-physics-001 — 2026-11-02T09:47:10.768161+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-physics-001 --years 2021 2022 --feature-set physics --trees 350 --sample-fraction 0.4 --threads 6`. Runtime: 600.0s.

Manifest and source snapshot: `artifacts/lgb-physics-001/manifest.json`. Feature ablation: physics; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.

- 2021: micro AUC **0.98470043**, prevalence baseline 0.54351580; train ['2015-01-01 00:00:00+00:00', '2020-12-25 00:00:00+00:00'], validation ['2021-01-01 00:00:00+00:00', '2021-12-31 00:00:00+00:00']; six-day purge.
- 2022: micro AUC **0.98607836**, prevalence baseline 0.55445194; train ['2015-01-01 00:00:00+00:00', '2021-12-25 00:00:00+00:00'], validation ['2022-01-01 00:00:00+00:00', '2022-12-31 00:00:00+00:00']; six-day purge.

Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## postprocess-physics-001 — 2026-11-02T09:57:47.375691+00:00

Status: complete. Hypothesis: startup costs favor persistent within-case schedules; near-nested Vinje labels support an optional probability ordering projection. Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/analyze_predictions.py lgb-physics-001 --run-id postprocess-physics-001`. Runtime 8.7s.

No training, seeds or early stopping. Same case splits, input data and base models as source manifests. Centered edge-normalized probability means over windows 3,7,13,25; weights 0,.25,.5,1; optional L2 projection onto Vinje G1 >= G2 >= G3. Select solely on 2021; freeze before reading 2022 labels. 2022 grid scores are diagnostics, never selection criteria.

- lgb-physics-001: selected config {'window': 3, 'weight': 0.0, 'projection': True}. 2021 baseline=0.98470043, selected=0.98475739. 2022 baseline=0.98607836, selected=0.98609080. 

Complete grid, environment, source snapshots, source/artifact checksums and selected predictions: `artifacts/postprocess-physics-001/analysis.json`. Kaggle submission ID, score, rank and observation time: none.

## case-calibration-001 — 2026-11-02

Completed independent auxiliary case-fraction regression experiment; **rejected**. Fixed chronological models (200 trees,15 leaves,min_child_samples20,lr0.04,seed4173) predicted weekly on-fractions from legal case inputs, then shifted hourly logits to match fractions. Selection on2021 chose a50% blend for Lio/Haukeli only: micro AUC0.980653166→0.982402139. Frozen selection degraded2022 from0.984358034→0.983619831. No final calibrators trained. Exact command, all hyperparameters, environment, source/data/model/prediction checksums, case IDs and diagnostics are in `docs/case_calibration.md` and `artifacts/case-calibration-001/manifest.json`. Runtime69s; no Kaggle submission.

## economic-probe-001 — 2026-11-02T10:04:13.149832+00:00

Rejected linear/spline models for the two weak units. Fixed C=0.1, L-BFGS400 iterations, tolerance1e-5,39 features/unit,80,000 purged training rows/fold, two threads; all eight fits converged in27.329s. A10% Haukeli linear blend selected on2021 gives0.98499348 versus physics baseline0.98470043, but the frozen2022 score declines to0.98591413 versus0.98607836. Complete command/environment/seeds/splits/source/data/artifact hashes: `artifacts/economic-probe-001/manifest.json`; discussion: `docs/economic_probe.md`. No Kaggle submission.

## lgb-physics-final-001 — 2026-11-02T10:02:41.654357+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-physics-final-001 --years --feature-set physics --trees 350 --sample-fraction 0.4 --threads 3 --final`. Runtime: 649.4s.

Manifest and source snapshot: `artifacts/lgb-physics-final-001/manifest.json`. Feature ablation: physics; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.


Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## lgb-normalized-001 — 2026-11-02T10:00:55.540053+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-001 --years 2021 2022 --feature-set normalized --trees 350 --sample-fraction 0.4 --threads 5`. Runtime: 772.0s.

Manifest and source snapshot: `artifacts/lgb-normalized-001/manifest.json`. Feature ablation: normalized; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.

- 2021: micro AUC **0.98954045**, prevalence baseline 0.54351580; train ['2015-01-01 00:00:00+00:00', '2020-12-25 00:00:00+00:00'], validation ['2021-01-01 00:00:00+00:00', '2021-12-31 00:00:00+00:00']; six-day purge.
- 2022: micro AUC **0.98895423**, prevalence baseline 0.55445194; train ['2015-01-01 00:00:00+00:00', '2021-12-25 00:00:00+00:00'], validation ['2022-01-01 00:00:00+00:00', '2022-12-31 00:00:00+00:00']; six-day purge.

Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## postprocess-normalized-001 — 2026-11-02T10:14:01.514078+00:00

Status: complete. Hypothesis: startup costs favor persistent within-case schedules; near-nested Vinje labels support an optional probability ordering projection. Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/analyze_predictions.py lgb-normalized-001 --run-id postprocess-normalized-001`. Runtime 8.8s.

No training, seeds or early stopping. Same case splits, input data and base models as source manifests. Centered edge-normalized probability means over windows 3,7,13,25; weights 0,.25,.5,1; optional L2 projection onto Vinje G1 >= G2 >= G3. Select solely on 2021; freeze before reading 2022 labels. 2022 grid scores are diagnostics, never selection criteria.

- lgb-normalized-001: selected config {'window': 3, 'weight': 0.0, 'projection': True}. 2021 baseline=0.98954045, selected=0.98955299. 2022 baseline=0.98895423, selected=0.98896450. 

Complete grid, environment, source snapshots, source/artifact checksums and selected predictions: `artifacts/postprocess-normalized-001/analysis.json`. Kaggle submission ID, score, rank and observation time: none.

## submission-physics-001 — 2026-11-02T10:14:19.418951+00:00

Export of `lgb-physics-final-001` using the postprocessing frozen on 2021 from `postprocess-physics-001`: `{'window': 3, 'weight': 0.0, 'projection': True}`. Training/data/parameters inherited from the referenced manifests. Exact command: `.venv-ml/bin/python scripts/export_submission.py --model-run lgb-physics-final-001 --selection-run postprocess-physics-001 --validation-model lgb-physics-001 --run-id submission-physics-001`. No refitting. Shape: [725, 2353]; probability range [1.1480146895337384e-06, 0.9999996423721313]; readback precision and exact schema/IDs verified. SHA256 `2bf904ad8337c1f5884be8385ababb9aa761ec9fc098de0761dbd6c04ae40590`. Full provenance: `artifacts/submission-physics-001/manifest.json`. Runtime 1.52s. Not yet submitted at export time.

## replay-physics-final-001 — 2026-11-02T10:14:17.484134+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id replay-physics-final-001 --years --feature-set physics --inference-from artifacts/lgb-physics-final-001 --threads 3`. Runtime: 23.5s.

Manifest and source snapshot: `artifacts/replay-physics-final-001/manifest.json`. Feature ablation: physics; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.


Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## Kaggle observation — submission-physics-001 — 2026-11-02T10:16:50Z

Submitted the independently replayed physics350 model with the validation-selected Vinje projection through Kaggle UI under the user's explicit authorization. **Public score0.99116**, statusComplete. This is below the existing team reference0.99172 (predating this session). The submission ID was not exposed in the inspected UI; description includes run ID and full checksum, and filename is `submission.csv`. Team rank91 was observed before submission; no new candidate-specific rank is asserted. Exact record: `docs/kaggle_scores.json`, `artifacts/submission-physics-001/manifest.json`. Artifact SHA256 `2bf904ad8337c1f5884be8385ababb9aa761ec9fc098de0761dbd6c04ae40590`. Raw saved-checkpoint replay in `replay-physics-final-001` was bitwise equal to training predictions and reproduced raw CSV checksum `2086795bdc86e9e68d43a1e69176d4ae213e68487151fb1458be9ca0e227db91`. Competition goal0.999 is not achieved.

## lgb-normalized-context-001 — 2026-11-02T10:14:11.218841+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-context-001 --years 2021 2022 --feature-set normalized-context --trees 350 --sample-fraction 0.4 --threads 5`. Runtime: 709.8s.

Manifest and source snapshot: `artifacts/lgb-normalized-context-001/manifest.json`. Feature ablation: normalized-context; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.

- 2021: micro AUC **0.98984982**, prevalence baseline 0.54351580; train ['2015-01-01 00:00:00+00:00', '2020-12-25 00:00:00+00:00'], validation ['2021-01-01 00:00:00+00:00', '2021-12-31 00:00:00+00:00']; six-day purge.
- 2022: micro AUC **0.98946942**, prevalence baseline 0.55445194; train ['2015-01-01 00:00:00+00:00', '2021-12-25 00:00:00+00:00'], validation ['2022-01-01 00:00:00+00:00', '2022-12-31 00:00:00+00:00']; six-day purge.

Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## ensemble-001 — 2026-11-02T10:27:00.251200+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/ensemble_compare.py --run-id ensemble-001`. Runtime 4.5s.

Hypothesis: normalized, normalized-context and physics models make complementary errors. Six fixed convex mixtures; common Vinje projection, no smoothing; no fitting/early stopping/seeds. Source models determine exact data, preprocessing, model settings and purged splits. All 168 hours and 14 units retained. Select by 2021 only and lock before reading 2022.

Sources: ['lgb-normalized-context-001', 'lgb-normalized-001', 'lgb-physics-001']. Weights: {'context': [1.0, 0.0, 0.0], 'normalized': [0.0, 1.0, 0.0], 'physics': [0.0, 0.0, 1.0], 'context75_normalized25': [0.75, 0.25, 0.0], 'context50_normalized50': [0.5, 0.5, 0.0], 'context75_physics25': [0.75, 0.0, 0.25]}. Selected: context75_physics25.

- 2021: selected micro AUC 0.9899660815115779.
- 2022: selected micro AUC 0.9897040280060725.

Complete environment/source/artifact provenance and score grid: `artifacts/ensemble-001/analysis.json`. Kaggle submission ID/score/rank/observation: none.

## postprocess-normalized-context-001 — 2026-11-02T10:27:00.251285+00:00

Status: complete. Hypothesis: startup costs favor persistent within-case schedules; near-nested Vinje labels support an optional probability ordering projection. Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/analyze_predictions.py lgb-normalized-context-001 --run-id postprocess-normalized-context-001`. Runtime 8.9s.

No training, seeds or early stopping. Same case splits, input data and base models as source manifests. Centered edge-normalized probability means over windows 3,7,13,25; weights 0,.25,.5,1; optional L2 projection onto Vinje G1 >= G2 >= G3. Select solely on 2021; freeze before reading 2022 labels. 2022 grid scores are diagnostics, never selection criteria.

- lgb-normalized-context-001: selected config {'window': 3, 'weight': 0.0, 'projection': True}. 2021 baseline=0.98984982, selected=0.98985442. 2022 baseline=0.98946942, selected=0.98947692. 

Complete grid, environment, source snapshots, source/artifact checksums and selected predictions: `artifacts/postprocess-normalized-context-001/analysis.json`. Kaggle submission ID, score, rank and observation time: none.

## prefix-selection-001 — 2026-11-02

First attempt failed before inference because LightGBM canonicalizes feature-name spaces to underscores; preserved failure manifest in `artifacts/prefix-selection-001/manifest.json`. Corrected validation and reran under `artifacts/prefix-selection-001/attempt-002` with source snapshot. No model refitting: evaluated tree prefixes100/150/200/250/350, selecting between fixed350, per-unit AUC, and per-unit log-loss rules on2021; locked the winning log-loss rule before2022. 2021micro AUC0.989849818435→0.990343518048;2022degraded0.989469422412→0.989269099257. **Rejected**. Full350 prefix reconstruction exactly matches saved predictions. Complete provenance, counts, environment, commands and comparisons in `docs/prefix_selection.md` and attempt manifest. No Kaggle submission.

## legacy-reference recovery — 2026-11-02

Downloaded the team's own previous `submission_round5_lgb_generators.csv` through the authenticated Kaggle UI, which displays public score0.99172. Saved unchanged at `artifacts/legacy-reference/`; SHA256 `5b736f51f14028eaee5382124314fd4c368b72f546116d0a6ac053fafa086a29`. Exact725 IDs/2353 columns, finite[0,1] probabilities verified. No historical models or validation predictions recovered. Prediction correlation with physics001 is0.97464694 and mean absolute difference0.04047443; these are unlabeled-test agreement statistics, not accuracy estimates. Preserve this reference and its provenance; no model fitting and no new submission in this recovery step.

## lgb-normalized-context-final-001 — 2026-11-02T10:28:33.382031+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-context-final-001 --years --feature-set normalized-context --trees 350 --sample-fraction 0.4 --threads 3 --final`. Runtime: 623.3s.

Manifest and source snapshot: `artifacts/lgb-normalized-context-final-001/manifest.json`. Feature ablation: normalized-context; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.


Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## submission-ensemble-001 — 2026-11-02T10:39:06.630110+00:00

Status:complete. Exact command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/export_ensemble.py --run-id submission-ensemble-001`. Runtime:3.0s.

Export-only hypothesis:reproduce frozen ensemble selection on full-fit saved test predictions. No fitting/inference/seeds/early stopping. Source training manifests carry data paths/checksums, environment, seeds, sampling and full model settings. Frozen weights75%normalized-context+25%physics; Vinje projection; no smoothing. Full source/code/data/checkpoint integrity and CSV readback checks. Independent checkpoint replay is separate.

Reused validation for75/25 recipe:{'2021': 0.9899660815115779, '2022': 0.9897040280060725}. Legacy research mode:False; legacy blend validation unavailable.

Complete provenance:`artifacts/submission-ensemble-001/manifest.json`. Kaggle submission ID/score/rank/observation:none.

## User threshold update — 2026-11-02

The user raised the main-repository transfer threshold from 0.99 to **0.998 on both chronological validation years**. This supersedes earlier migration references; Kaggle submissions remain authorized before migration. The active public-leaderboard target remains 0.999, with an explicit desire for top three; rank must be observed on Kaggle, not inferred from local AUC.

## replay-context-final-001 — 2026-11-02T10:40:08.248096+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id replay-context-final-001 --years --feature-set normalized-context --inference-from artifacts/lgb-normalized-context-final-001 --threads 2`. Runtime: 30.3s.

Manifest and source snapshot: `artifacts/replay-context-final-001/manifest.json`. Feature ablation: normalized-context; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.


Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

Saved-checkpoint verification: `replay-context-final-001` reproduces all raw `lgb-normalized-context-final-001` test probabilities bitwise and CSV SHA256 `4b53d338bac9e8c0aca9dad69727a648c6de423b43351f6142337803b7a3add3`; comparison recorded alongside replay manifest.


## Hydraulic LP cache and diagnostic ledger — 2026-11-02

These input-only optimization runs test whether relaxed hydraulic dispatch and water shadow prices improve a supervised surrogate. They do not fit labels, use early stopping, tune ensemble weights or constitute submissions. Each cited manifest is the authoritative exact command, selected IDs/dates, mathematical specification, source snapshot/hashes, data paths/checksums, OS/Python/package versions, runtime and artifact hashes. Runs used an Apple M2 Max (12 CPU cores, 32 GB RAM), HiGHS one thread, no random seed. The LP uses all legal 168-hour inputs, initial volume, terminal water values at start+7 days, and no commitment labels or future historical volumes. Methods and limitations: `docs/lp_features.md`; primary sources: `docs/web_research.md`.

| Run and UTC start | Outcome | Exact manifest |
| --- | --- | --- |
| `lp-benchmark-001`, 10:17:27 | Failed before first solve: YAML loss coefficient was a string; numeric casting corrected afterward. Runtime 1.322 s; failed source snapshot retained. | `artifacts/lp-benchmark-001/manifest.json` |
| `lp-benchmark-002`, 10:18:02 | Sixteen spread-out cases passed; runtime 2.726 s, median 0.08688 s/case. | `artifacts/lp-benchmark-002/manifest.json` |
| `lp-full-001`, 10:20:10 | All 3,647 cases passed, 3,647×168×78 features; runtime 376.342 s, median 0.08209 s/case. All constraint slacks zero. | `artifacts/lp-full-001/manifest.json` |
| `lp-diagnostic-001`, 10:28:19 | Failed before label loading: object-dtype column metadata could not be read without pickle. Failed script and error retained; duration unavailable. | `artifacts/lp-diagnostic-001/manifest.json` |
| `lp-diagnostic-002`, 10:28:37 | Fixed monotone q/Qmin score; 2021 micro AUC 0.9469648687, 2022 0.9522510823. Runtime 1.174 s. | `artifacts/lp-diagnostic-002/manifest.json` |
| `lp-v2-benchmark-001`, 10:41:09 | Isolated directed Haukeli reservoirs; 16 cases passed in 3.041 s, median 0.09663 s/case; shape16×168×81. Four wet error cases reduce false negatives453→142 with20 false positives; four dry exceptions unchanged. Selected-case diagnostic, not holdout AUC. | `artifacts/lp-v2-benchmark-001/manifest.json` |

The full-v1 cache metadata repair is recorded inside its manifest: `scripts/repair_lp_cache_schema.py artifacts/lp-full-001` changed column metadata to Unicode while verifying identical numeric bytes; original and repaired archive hashes and repair source are retained. Current feature archive SHA256: `0ebba7d10ffb29294e36e5e8742c73165e202ac1d96e6d15fbef7ea9d7cca7b9`. Frozen v1 modules/build script have not been changed for v2. Nine v1 tests and four v2 tests passed. V2 source/benchmark/assumptions and exact selected-case comparisons are described in `docs/lp_v2_features.md`.

Diagnostic002 evaluates all365 cases in each of2021 and2022, all168hours and14units, using p=(q/Qmin)/(1+q/Qmin). It has no training split because no learned parameters exist. Its intended use is as supervised features, not forced labels. The v2 benchmark selects eight spread controls and eight Haukeli error cases, then reads their known labels only after all optimization/features are complete. Every listed run has Kaggle submission ID/score/rank/observation time: none. Supervised LP model results are recorded separately by the root coordinator. No migration follows from this benchmark; the latest user gate is0.998 on both chronological years.

## Kaggle observation — submission-ensemble-001 — 2026-11-02T10:43:25Z

Public score **0.99254**, team rank **82**, displayed new personal best (previous0.99172). Uploaded under explicit user authorization. SHA256 `cd972841fa1e8da50a0fbfd07d30dece5b695ea893d867e68b83df97283600d6`. Both component raw test predictions were replayed from saved checkpoints bitwise; frozen75/25weights and Vinje projection exported with exactschema/ID/probability checks. Submission ID not exposed in UI; description and checksum preserved in `docs/kaggle_scores.json`. The three leading public scores observed concurrently were0.99820,0.99808,0.99805. No final-selection checkboxes changed. Active0.999/top-three leaderboard goal and0.998/both-years transfer gate remain unmet.

## lgb-normalized-lp-001 — 2026-11-02T10:29:59.310295+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-lp-001 --years 2021 2022 --feature-set normalized-lp --trees 350 --sample-fraction 0.4 --threads 4`. Runtime: 1014.3s.

Manifest and source snapshot: `artifacts/lgb-normalized-lp-001/manifest.json`. Feature ablation: normalized-lp; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.

- 2021: micro AUC **0.99437001**, prevalence baseline 0.54351580; train ['2015-01-01 00:00:00+00:00', '2020-12-25 00:00:00+00:00'], validation ['2021-01-01 00:00:00+00:00', '2021-12-31 00:00:00+00:00']; six-day purge.
- 2022: micro AUC **0.99334854**, prevalence baseline 0.55445194; train ['2015-01-01 00:00:00+00:00', '2021-12-25 00:00:00+00:00'], validation ['2022-01-01 00:00:00+00:00', '2022-12-31 00:00:00+00:00']; six-day purge.

Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## postprocess-normalized-lp-001 — 2026-11-02T10:48:02.533319+00:00

Status: complete. Hypothesis: startup costs favor persistent within-case schedules; near-nested Vinje labels support an optional probability ordering projection. Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/analyze_predictions.py lgb-normalized-lp-001 --run-id postprocess-normalized-lp-001`. Runtime 9.1s.

No training, seeds or early stopping. Same case splits, input data and base models as source manifests. Centered edge-normalized probability means over windows 3,7,13,25; weights 0,.25,.5,1; optional L2 projection onto Vinje G1 >= G2 >= G3. Select solely on 2021; freeze before reading 2022 labels. 2022 grid scores are diagnostics, never selection criteria.

- lgb-normalized-lp-001: selected config {'window': 3, 'weight': 0.0, 'projection': True}. 2021 baseline=0.99437001, selected=0.99437222. 2022 baseline=0.99334854, selected=0.99335146. 

Complete grid, environment, source snapshots, source/artifact checksums and selected predictions: `artifacts/postprocess-normalized-lp-001/analysis.json`. Kaggle submission ID, score, rank and observation time: none.


## lp-v2-full-001 and lp-v2-diagnostic-001 — 2026-11-02

The root approved the complete directed-Haukeli cache after the16-case benchmark. Full build started10:43:41UTC, exact command `.venv-ml/bin/python scripts/build_lp_features_v2.py --run-id lp-v2-full-001 --all`. All3647cases succeeded in425.189s, median0.09185s, oneHiGHSthread. All soft slacks zero, maximum aggregate mass residual1.47e-11Mm3. Shape3647×168×81. Complete source snapshots, exact case IDs/dates, data/source/artifact hashes, parameters and Python/OS/package environment: `artifacts/lp-v2-full-001/manifest.json`. Feature archive SHA256:`5dc566d5dd2f2e07b5dc4e5ce8ec85430a45c79cf439b3ecbeff4e1a32709084`. V1 remains frozen.

Fixed diagnostic started10:51:16UTC, exact command `.venv-ml/bin/python scripts/score_lp_v2_diagnostic.py --run-id lp-v2-diagnostic-001`. It took1.097s. No fitting, seed, early stopping or tuned weights. Evaluate all2021/2022cases using the previously fixed score p=(q/Qmin)/(1+q/Qmin). Micro AUC2021:0.947609525214162;2022:0.9541151178522858. Haukeli AUC2021:0.8825280084964002;2022:0.7880384158415946. Both improve on v1 diagnostics, but these are untrained LP ranks rather than supervised model scores. Exact predictions, labels, case IDs, commands, source/data/cache/environment and artifact provenance: `artifacts/lp-v2-diagnostic-001/manifest.json`.

A verification smoke loaded the complete cache with exact case/column order, checked finiteness and normalized the base+physics+v2 table: shape612696×385,3.659s. The independent startup-DP feature candidate adds28dimensionless columns in1.596s (shape612696×413); no model trained for this candidate here. Report with source/topology/DP hashes and unit parameters: `artifacts/lp-v2-full-001-loader-verification.json`. Six v2 tests and four LP-switching tests pass. Methods/assumptions: `docs/lp_v2_features.md`, `docs/lp_switching_features.md`. All three checks have no Kaggle submission ID,score,rank or observationtime. Supervised matched ablations belong to separate root-run manifests; migration gate remains0.998bothyears.

## lgb-physics-large-001 — 2026-11-02T09:51:49.819894+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-physics-large-001 --years 2021 2022 --feature-set physics --trees 650 --leaves 63 --sample-fraction 0.7 --threads 5 --final`. Runtime: 3654.4s.

Manifest and source snapshot: `artifacts/lgb-physics-large-001/manifest.json`. Feature ablation: physics; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.

- 2021: micro AUC **0.98431056**, prevalence baseline 0.54351580; train ['2015-01-01 00:00:00+00:00', '2020-12-25 00:00:00+00:00'], validation ['2021-01-01 00:00:00+00:00', '2021-12-31 00:00:00+00:00']; six-day purge.
- 2022: micro AUC **0.98575100**, prevalence baseline 0.55445194; train ['2015-01-01 00:00:00+00:00', '2021-12-25 00:00:00+00:00'], validation ['2022-01-01 00:00:00+00:00', '2022-12-31 00:00:00+00:00']; six-day purge.

Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## artifacts/lp-error-analysis-001 — diagnostic only

Status: complete. Manifest: `artifacts/lp-error-analysis-001/manifest.json`. No model fitting or LP solving. Complete timestamp, command, source/data/model/cache hashes, fixed splits, environment, runtime and output checksums are retained in the manifest.

## artifacts/lp-error-analysis-001/lp2021 — diagnostic only

Status: complete. Manifest: `artifacts/lp-error-analysis-001/lp2021/manifest.json`. No model fitting or LP solving. Complete timestamp, command, source/data/model/cache hashes, fixed splits, environment, runtime and output checksums are retained in the manifest.

## artifacts/lp-error-analysis-001/residual-2021 — diagnostic only

Status: complete. Manifest: `artifacts/lp-error-analysis-001/residual-2021/manifest.json`. No model fitting or LP solving. Complete timestamp, command, source/data/model/cache hashes, fixed splits, environment, runtime and output checksums are retained in the manifest.

## artifacts/lp-error-analysis-001/residual-2022 — diagnostic only

Status: complete. Manifest: `artifacts/lp-error-analysis-001/residual-2022/manifest.json`. No model fitting or LP solving. Complete timestamp, command, source/data/model/cache hashes, fixed splits, environment, runtime and output checksums are retained in the manifest.

The exact LP prediction error decomposition assigns about41.07% of2021 and39.06% of2022 pooled AUC discordance to Haukeli/Hogga/Lio combined; mixed weeks dominate. Physical directionality explains wet Haukeli examples only, with documented dry counterexamples. The initial interactive date-unit preflight failed before diagnostics; its unrecorded start time is not invented. Evidence and formulas: `docs/lp_error_analysis.md`. No Kaggle submission.

## lp-ensemble-001 — 2026-11-02T10:54:29.459298+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/lp_ensemble_compare.py --run-id lp-ensemble-001`. Runtime 3.6s.

Hypothesis: LP features and the earlier normalized-context/physics ensemble make complementary errors. Four fixed convex mixtures; common Vinje projection, no smoothing; no fitting/early stopping/seeds. Source models determine exact data, preprocessing, model settings and purged splits. All 168 hours and 14 units retained. Select by 2021 only and lock before reading 2022.

Sources: ['lgb-normalized-lp-001', 'lgb-normalized-context-001', 'lgb-physics-001']. Weights: {'lp': [1.0, 0.0, 0.0], 'prior_ensemble': [0.0, 0.75, 0.25], 'lp90_prior10': [0.9, 0.075, 0.025], 'lp75_prior25': [0.75, 0.1875, 0.0625]}. Selected: lp90_prior10.

- 2021: selected micro AUC 0.9944673352759986.
- 2022: selected micro AUC 0.99344637149997.

Complete environment/source/artifact provenance and score grid: `artifacts/lp-ensemble-001/analysis.json`. Kaggle submission ID/score/rank/observation: none.

## xgb-lp-specialists-001 — 2026-11-02T10:52:51.135412+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/xgb_lp_specialists.py --run-id xgb-lp-specialists-001`. Runtime:254.7s.

Hypothesis:complementary XGBoost specialists for['Haukeli_G1', 'Lio_G1', 'Hogga_G1'] improve the normalized-LP v1 LightGBM matrix. Parameters:{'n_estimators': 600, 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 20, 'reg_lambda': 4.0, 'reg_alpha': 0.0, 'max_bin': 128, 'subsample': 0.9, 'colsample_bytree': 0.9, 'colsample_bylevel': 1.0, 'colsample_bynode': 1.0, 'random_state': 4173, 'n_jobs': 2, 'objective': 'binary:logistic', 'tree_method': 'hist', 'device': 'cpu', 'eval_metric': 'logloss', 'grow_policy': 'depthwise', 'max_delta_step': 0.0, 'validate_parameters': True, 'verbosity': 0}. No early stopping. Identical baseline purged case IDs and stored sampledtrainingrowindices; no preprocessing fit. Weights[0.0, 0.25, 0.5, 1.0] jointly replace the threeunits; select2021 before2022 validation access and fitting. Fullmatrix microAUC, no projection or smoothing.

- 2021: baseline=0.9943700122907179; selected=0.9945393237576134; selectedweight=0.5.
- 2022: baseline=0.9933485421643476; selected=0.9934238349385737; selectedweight=0.5.

Complete source snapshots/data and artifact checksums, environment, splits, samples, checkpoints and grid:`artifacts/xgb-lp-specialists-001/manifest.json`. Kaggle submission ID/score/rank/observation:none.

## lgb-normalized-lp-final-001 — 2026-11-02T10:48:02.543958+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-lp-final-001 --years --feature-set normalized-lp --trees 350 --sample-fraction 0.4 --threads 3 --final`. Runtime: 1574.4s.

Manifest and source snapshot: `artifacts/lgb-normalized-lp-final-001/manifest.json`. Feature ablation: normalized-lp; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.


Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## submission-lp-blend-001 — 2026-11-02T11:24:11.250559+00:00

Status: complete. Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/export_selected_blend.py --run-id submission-lp-blend-001 --selection-run lp-ensemble-001 --final-runs lgb-normalized-lp-final-001 lgb-normalized-context-final-001 lgb-physics-final-001`. No fitting or checkpoint inference; verified selected full-fit recipes, all source/checkpoint/data/artifact hashes, schema and probability serialization. Independent replay is separate. Manifest: `artifacts/submission-lp-blend-001/manifest.json`. Runtime 3.24s. Kaggle ID/score/rank/observation: none at export.

## replay-lp-final-001 — 2026-11-02T11:24:11.198205+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id replay-lp-final-001 --years --feature-set normalized-lp --inference-from artifacts/lgb-normalized-lp-final-001 --threads 2`. Runtime: 17.9s.

Manifest and source snapshot: `artifacts/replay-lp-final-001/manifest.json`. Feature ablation: normalized-lp; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.


Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

Independent replay `replay-lp-final-001` reproduces every `lgb-normalized-lp-final-001` raw test probability bitwise and raw CSV SHA256 `83b759ec4cd29f74aa527bbc016f3d0f40e3fe74743479b2f4299afa0da5178c`. This verifies the new LP component for the selected90/7.5/2.5 blend export; earlier context/physics components also have exact saved-model replays.

## Kaggle observation — submission-lp-blend-001 — 2026-11-02T11:27:19Z

Public AUC **0.99580**, team rank **34**, new personal best over0.99254. Frozen blend90%normalized-LP+7.5%normalized-context+2.5%physics, followed by Vinje projection. Selection2021 micro0.994467335276; unchanged2022 micro0.993446371500. Every component raw test array replayed bitwise from checkpoints; exact725-ID/2353-column schema and finite[0,1] serialization verified. SHA256 `ff6c52f78c000d5235f460ca4ea814403f0fe5096249fb4b8f2a85a4e31b9e58`. Submission ID not exposed in inspected UI; exact description/checksum and observation in `docs/kaggle_scores.json`. No final-selection checkbox changed. Both current goals remain unmet.

## xgb-lp-specialists-final-001 — 2026-11-02T11:26:14.205559+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/xgb_lp_specialists_final.py --run-id xgb-lp-specialists-final-001`. Runtime:144.5s.

Hypothesis:reproduce validated complementary3-unit XGBoost recipe for fullfit. Exact2021checkpoint/fullmatrix replay required before fitting. Parameters:{'n_estimators': 600, 'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 20, 'reg_lambda': 4.0, 'reg_alpha': 0.0, 'max_bin': 128, 'subsample': 0.9, 'colsample_bytree': 0.9, 'colsample_bylevel': 1.0, 'colsample_bynode': 1.0, 'random_state': 4173, 'n_jobs': 2, 'objective': 'binary:logistic', 'tree_method': 'hist', 'device': 'cpu', 'eval_metric': 'logloss', 'grow_policy': 'depthwise', 'max_delta_step': 0.0, 'validate_parameters': True, 'verbosity': 0}; noearlystopping. Samefull-fit storedsamplerows aslgb-normalized-lp-final-001; all2922trainingcases eligible, no testlabels.

Inherited selectedrecipe validation:{'2021': 0.9945393237576134, '2022': 0.9934238349385737}; this is not a fullfit orKagglescore. Source/data/code/checkpoint/sample/predictionhashes, exactIDs, featureformulas and environment:`artifacts/xgb-lp-specialists-final-001/manifest.json`.

Kaggle submission ID/score/rank/observation:none. Component predictions only; no finalblendCSV created.

## lgb-normalized-lp-context-001 — 2026-11-02T10:44:39.008857+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-lp-context-001 --years 2021 2022 --feature-set normalized-lp-context --trees 350 --sample-fraction 0.4 --threads 3`. Runtime: 2698.7s.

Manifest and source snapshot: `artifacts/lgb-normalized-lp-context-001/manifest.json`. Feature ablation: normalized-lp-context; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.

- 2021: micro AUC **0.99462502**, prevalence baseline 0.54351580; train ['2015-01-01 00:00:00+00:00', '2020-12-25 00:00:00+00:00'], validation ['2021-01-01 00:00:00+00:00', '2021-12-31 00:00:00+00:00']; six-day purge.
- 2022: micro AUC **0.99310791**, prevalence baseline 0.55445194; train ['2015-01-01 00:00:00+00:00', '2021-12-25 00:00:00+00:00'], validation ['2022-01-01 00:00:00+00:00', '2022-12-31 00:00:00+00:00']; six-day purge.

Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## vinje-count-001 — 2026-11-02T10:53:36.727754+00:00

Status: complete. Hypothesis: structured active-unit count shares information across three nearly nested Vinje targets. Same normalized-LP features, purged dates and exact sampled rows as baseline; LightGBM350trees/31leaves/.06learning-rate, four-class objective,2threads,seed4173,noearlystopping. Frozen50% probability replacement selected on2021 from[0,.25,.5,1]. MicroAUC2021 .994408340132;2022 .993382225812, versusbaseline .994370012291/.993348542164. All three Vinje unit AUCs improved in both years. Nonnested label exceptions retained and documented. Runtime565.354s. Exact command/source/data/checkpoints/splits/artifacts/versions in `artifacts/vinje-count-001/manifest.json`; independent artifact and metric checks passed. NoKaggle submission.

## lgb-normalized-lp-v2-001 — 2026-11-02T10:53:01.438777+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-lp-v2-001 --years 2021 2022 --feature-set normalized-lp-v2 --trees 350 --sample-fraction 0.4 --threads 2`. Runtime: 2611.2s.

Manifest and source snapshot: `artifacts/lgb-normalized-lp-v2-001/manifest.json`. Feature ablation: normalized-lp-v2; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.

- 2021: micro AUC **0.99438996**, prevalence baseline 0.54351580; train ['2015-01-01 00:00:00+00:00', '2020-12-25 00:00:00+00:00'], validation ['2021-01-01 00:00:00+00:00', '2021-12-31 00:00:00+00:00']; six-day purge.
- 2022: micro AUC **0.99336744**, prevalence baseline 0.55445194; train ['2015-01-01 00:00:00+00:00', '2021-12-25 00:00:00+00:00'], validation ['2022-01-01 00:00:00+00:00', '2022-12-31 00:00:00+00:00']; six-day purge.

Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## lp-candidate-comparison-001 — 2026-11-02T11:36:44.735927+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/lp_candidate_compare.py --run-id lp-candidate-comparison-001`. Runtime:4.3s.

Hypothesis:LP temporalcontext orLPv2 may add complementary input-only scheduling information. Eightfixed recipes:{'base': '100% raw LPv1 baseline', 'context': '100% raw LP temporal-context model', 'v2': '100% raw LPv2 model', 'base50_context50': '50% base +50% context, all units', 'context75_base25': '75% context +25% base, all units', 'base_haukeli_v2_50': 'Base, replacing only Haukeli_G1 by50%base+50%v2', 'base_haukeli_v2_100': 'Base, replacing only Haukeli_G1 by100%v2', 'combined_specialist_count': 'Existing frozen50%XGB replacement forHaukeli/Lio/Hogga and50%Vinje-count replacement, without new tuning'}. CommonVinjeprojection, nosmoothing. No model fits,seeds orearlystopping; source manifests preserve exact splits/modelsettings/features. Select2021only, lock before2022, accept only if2022 improves overfixedpriorcombined_specialist_count; otherwise retainprior, never retuneusing2022. Reuseddevelopmentfolds.

- 2021: selected=0.9947811119052823; prior=0.9945774047171779; accepted=0.9947811119052823.
- 2022: selected=0.9934702459184567; prior=0.99345819682712; accepted=0.9934702459184567.

Complete source/data/artifact hashes, IDs, units, source/model snapshots and environment:`artifacts/lp-candidate-comparison-001/manifest.json`. Kaggle submission ID/score/rank/observation:none.


## Head and shared-loss physical ablations — 2026-11-02

All three runs completed successfully as computations; scientific negative/mixed findings are preserved. One HiGHS thread per solve on the same Apple M2 Max,12-core,32GB machine. No model fit, random seed, threshold tuning, early stopping, ensemble selection, Kaggle submission or full cache. All legal inputs and source/cache/model digests, code snapshots, exact commands, selected IDs/dates, OS/Python/package versions, solver configuration and artifact hashes are in these manifests:

| Run | UTC start | Runtime | Exact provenance |
| --- | --- | ---: | --- |
| `lp-head-update-001` | 2026-11-02T11:27:14.908747+00:00 | 5.076s | `artifacts/lp-head-update-001/manifest.json` |
| `lp-shared-loss-001` | 2026-11-02T11:33:38.895959+00:00 | 4.820s | `artifacts/lp-shared-loss-001/manifest.json` |
| `lp-shared-loss-iterations-001` | 2026-11-02T11:37:10.690449+00:00 | 15.484s | `artifacts/lp-shared-loss-iterations-001/manifest.json` |

`lp-head-update-001`: quantify all365 cases in2021 using v1's predicted Bandak/Vinjevatn storage, then rebuild only Hogga hourly PQ curves and solve once on16 predeclared spread/error/high-head-change cases. Initial-head terminal values stay fixed; no pooled reservoir allocation. Hogga maximum within-week head change has median0.611m,p951.587m,max1.941m. Selected LP-only micro AUC worsens0.9501469713→0.9456230152; Hogga0.8109969707→0.7303571244. Each of the three selected subgroups worsens. **Reject full-cache expansion.** A float64 accumulation audit explains the original aggregate diagnostic's float32 reporting roundoff: true conservation residual<=7.05e-12Mm3; original manifest remains unchanged. Audit artifact:`artifacts/lp-head-update-001-conservation-review.json`.

`lp-shared-loss-001`: isolate Tokke/Vinje shared main/penstock losses at frozen v1 flows, subtract only extra shared loss before the existing own-loss curve construction. Initial heads and terminal conversion unchanged. Same16 selected2021 cases plus16 independently evenly spaced training IDs1..2186. Fixed proxy p=(q/Qmin)/(1+q/Qmin). Selected micro AUC0.9501469713→0.9555522117, independent subset0.9327837983→0.9304519753. First-step post-solve loss inconsistency reaches21.504m. Preserve both outcomes; no general improvement claimed.

`lp-shared-loss-iterations-001`: exactly4 iterations, fixed damping0.5 on reference flows; no iteration selection. Selected micro AUC trend baseline/1/2/3/4:0.9501469713,0.9555522117,0.9564357573,0.9573457426,0.9563965308. Independent trend:0.9327837983,0.9304519753,0.9334538926,0.9340829419,0.9330893487. Mean loss inconsistency falls0.880→0.489→0.541→0.477m, but maximum21.504→18.480→12.096→18.480m and approximately3% of unit-hour states continue changing. All32 first iterations exactly reproduce the prior benchmark. Fourth-iteration positive diagnostics do not establish physical convergence. **Do not expand this construction to a full cache under the consistency gate.** All slacks zero and precise aggregate conservation<=1.26e-11Mm3 across128 updated solves.

Five hourly-head/negative-price/conservation/3D-coefficient tests and two shared-loss double-counting/isolation tests pass. The actual32-case sample has no negative-price hours; the negative-price result is a synthetic independent test, not claimed from the real sample. Full formulas, selection rules, primary sources, score tables and limitations are in `docs/lp_head_update.md`. No raw data, v1/v2 frozen source/cache or trained model was modified. Kaggle submission ID/score/rank/observation time:none for all runs.

## vinje-count-final-001 — 2026-11-02T11:28:43.511685+00:00

Status: complete. Complete exact command, source/data/model hashes, runtime, environment and reproduction in `artifacts/vinje-count-final-001/manifest.json` and `docs/vinje_count.md`. Final count and all three XGB checkpoints replayed exactly from reconstructed input features. Combined OOF composition has no new weight search and was proposed after seeing both component-year results; it is reused development evidence.

## vinje-count-combined-001 — 2026-11-02T11:29:05.387213+00:00

Status: complete. Complete exact command, source/data/model hashes, runtime, environment and reproduction in `artifacts/vinje-count-combined-001/manifest.json` and `docs/vinje_count.md`. Final count and all three XGB checkpoints replayed exactly from reconstructed input features. Combined OOF composition has no new weight search and was proposed after seeing both component-year results; it is reused development evidence.

## vinje-count-combined-final-001 — 2026-11-02T11:34:33.646458+00:00

Status: complete. Complete exact command, source/data/model hashes, runtime, environment and reproduction in `artifacts/vinje-count-combined-final-001/manifest.json` and `docs/vinje_count.md`. Final count and all three XGB checkpoints replayed exactly from reconstructed input features. Combined OOF composition has no new weight search and was proposed after seeing both component-year results; it is reused development evidence.

## Kaggle observation — vinje-count-combined-final-001 — 2026-11-02T11:41:14Z

Public AUC **0.99584**, team rank **33**, new personal best; user independently confirmed newest best. Local micro AUC0.994577404717/0.993458196827. SHA256 `e769f2c77336ad958f14c6c38b18248707c55b248453d9012cf76fbc07ea4039`. Count and three XGB final checkpoints replayed exactly; full LP base had separate exact replay. Exact schema and finite probability readback checked. All model/selection and submission provenance retained. Third place still0.99805 at observation. No final-selection checkbox changed. Transfer gate0.998/both-years and active0.999/top-three goal remain unmet.

## lgb-normalized-lp-switching-001 — 2026-11-02T11:24:11.216628+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-lp-switching-001 --years 2021 2022 --feature-set normalized-lp-switching --trees 350 --sample-fraction 0.4 --threads 3`. Runtime: 1118.4s.

Manifest and source snapshot: `artifacts/lgb-normalized-lp-switching-001/manifest.json`. Feature ablation: normalized-lp-switching; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.

- 2021: micro AUC **0.99384015**, prevalence baseline 0.54351580; train ['2015-01-01 00:00:00+00:00', '2020-12-25 00:00:00+00:00'], validation ['2021-01-01 00:00:00+00:00', '2021-12-31 00:00:00+00:00']; six-day purge.
- 2022: micro AUC **0.99342362**, prevalence baseline 0.55445194; train ['2015-01-01 00:00:00+00:00', '2021-12-25 00:00:00+00:00'], validation ['2022-01-01 00:00:00+00:00', '2022-12-31 00:00:00+00:00']; six-day purge.

Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## lp-locality-001 — 2026-11-02T11:45:30.802655+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/lp_locality_ablation.py --run-id lp-locality-001`. Runtime:77.3s.

Hypothesis:localLP/physical/market inputs reduce calendar and distantreservoir shortcuts onHaukeli/Lio/Hogga. Exactapproved81/83/86column sets; same350treeLPv1LightGBM settings,seed4173 and saved sampleindices; onlycolumns change exceptauthorized CPUthreadcap2. Noearlystopping. Jointweights0,.25,.5,1 selected2021 thenlocked2022; noproj/smoothing.

- 2021: baseline=0.9943700122907179; selected=0.9944432091079239; jointweight=0.25.
- 2022: baseline=0.9933485421643476; selected=0.9934137578705272; jointweight=0.25.

All parameters,features,IDs,source/data/checkpoint/sample/output hashes and environment:`artifacts/lp-locality-001/manifest.json`. No fullfit. Kaggle submission ID/score/rank/observation:none.

## lgb-normalized-lp-context-final-001 — 2026-11-02T11:38:19.273992+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-lp-context-final-001 --years --feature-set normalized-lp-context --trees 350 --sample-fraction 0.4 --threads 3 --final`. Runtime: 908.3s.

Manifest and source snapshot: `artifacts/lgb-normalized-lp-context-final-001/manifest.json`. Feature ablation: normalized-lp-context; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.


Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## replay-lp-context-final-001 — 2026-11-02T11:55:20.069381+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id replay-lp-context-final-001 --years --feature-set normalized-lp-context --inference-from artifacts/lgb-normalized-lp-context-final-001 --threads 2`. Runtime: 17.5s.

Manifest and source snapshot: `artifacts/replay-lp-context-final-001/manifest.json`. Feature ablation: normalized-lp-context; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.


Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## submission-lp-context-blend-001 — 2026-11-02T11:56:20.074546+00:00

Status: complete. Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/export_selected_blend.py --run-id submission-lp-context-blend-001 --selection-run lp-candidate-comparison-001 --final-runs lgb-normalized-lp-final-001 lgb-normalized-lp-context-final-001`. No fitting or checkpoint inference; verified selected full-fit recipes, all source/checkpoint/data/artifact hashes, schema and probability serialization. Independent replay is separate. Manifest: `artifacts/submission-lp-context-blend-001/manifest.json`. Runtime 2.56s. Kaggle ID/score/rank/observation: none at export.

## verify-lp-context-blend-001 — 2026-11-02T11:57:28.811012+00:00

Status: failed. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/verify_lp_context_export.py`. Independent LP-context checkpoint replay compared bitwise with the original full-fit output; raw CSV hashes compared. The locked 75% context / 25% base export was reconstructed from saved arrays, with fixed Vinje projection, exact schema/ID checks and CSV readback. Proof: `artifacts/verify-lp-context-blend-001/manifest.json`. No fit or Kaggle submission. Chronological folds have been reused for research; both scores remain below 0.998.

## verify-lp-context-blend-002 — 2026-11-02T11:57:43.009520+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/verify_lp_context_export.py --run-id verify-lp-context-blend-002`. Independent LP-context checkpoint replay compared bitwise with the original full-fit output; raw CSV hashes compared. The locked 75% context / 25% base export was reconstructed from saved arrays, with fixed Vinje projection, exact schema/ID checks and CSV readback. Proof: `artifacts/verify-lp-context-blend-002/manifest.json`. No fit or Kaggle submission. Chronological folds have been reused for research; both scores remain below 0.998.

Verification correction: `verify-lp-context-blend-001` failed because the independent checker initially multiplied float32 source arrays before promoting to float64; the exporter correctly promotes first. The checker was corrected to the documented float64 export arithmetic and rerun separately as `verify-lp-context-blend-002`. The failed manifest and original checker snapshot are preserved. No model, prediction, export, selection, or label changed.

## mps-tcn-smoke-001 — 2026-11-02T12:02:14.845664+00:00

Status: blocked. Command: `/Users/heap/programming/tdt4173/.venv-mps/bin/python scripts/smoke_mps_tcn.py`. Synthetic-only MPS TCN feasibility; no real feature table, labels, or fit. Proof: `artifacts/mps-tcn-smoke-001/manifest.json`. Blocked/failed checks do not establish GPU speed or memory consumption. No Kaggle submission.

## Kaggle observation — submission-lp-context-blend-001 — 2026-11-02T12:02:42Z

Public AUC **0.99604**, team rank **28**, new personal best over0.99584. Frozen75%LP-context+25%LP-base with Vinje projection, selected on2021 from fixed8grid and accepted unchanged on2022. LocalAUC .994781111905/.993470245918. Every component checkpoint replayed exactly, and independent float64 blend/projection/readback proof passed (`artifacts/verify-lp-context-blend-002/manifest.json`). Initial verification failure due to prematurefloat32 weighting is preserved; exported predictions never changed. SHA256 `36aa5a2f9378139916ee79aa710e9003875e3a81ce419f74f8fe1f75b50f3c98`. Numeric submission ID unknown; exact description and observation retained. No final-selection checkbox changed. Transfer and leaderboard goals remain unmet.

Neural feasibility setup: uv created `.venv-mps` with Python 3.11.15, but offline torch resolution failed because torch is absent from the cache. Ordinary official-PyPI curl metadata lookup failed DNS; its identical escalated invocation was rejected by the approval policy (`sandbox_approval: false`). No alternative download route was attempted. Hardware reports M2 Max / 38 GPU cores / 32 GB / macOS 27.0.1; MPS runtime remains unverified. Exact commands, official references, architecture, split/normalization/blend protocol, and unmeasured resource limits are in `docs/neural_tcn_feasibility.md` and `docs/neural_tcn_protocol.json`. Only blocked synthetic smoke and syntax compilation ran; no real data table or fit.


## lp-joint-tests-001 and lp-joint-plant-001 — 2026-11-02

A test preflight had one invalid expectation: it required retaining a collinear lower-hull point. The unchanged implementation correctly removed that point. The fixture now places it strictly below the edge. Failure source and explanation are preserved in `artifacts/lp-joint-tests-001/manifest.json`; recording time is captured, exact original start time unavailable. The other four initial tests passed. After correction and adding a full-network feature-schema check, all six joint-model tests pass.

Benchmark `lp-joint-plant-001` started 2026-11-02T11:57:09.967664+00:00; command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/benchmark_lp_joint_plant.py --run-id lp-joint-plant-001`; status complete, runtime 8.329s, median0.2021s/case. Same Apple M2 Max12-core32GB hardware, one HiGHS thread. No learned parameters, seed, early stopping, grid tuning or score-based selection. Exact128 positive fractions1/128..1, all15/7 active-unit subsets for Tokke/Vinje. Physically evaluate simultaneous main/assigned-penstock losses, reject invalid head/flow/Pmin/Pmax points, retain both upper/lower convex-hull vertices and optimize convex mixtures with total mass<=1. Old independent Tokke/Vinje variables disabled. Other seven plants and initial-head terminal values unchanged.

Use the exact same16 selected2021 cases and16 independent evenly spaced2015–2020 cases as the prior shared-loss benchmark. Fixed q/Qmin micro AUC improves0.9501469713→0.9647306043 on selected cases and0.9327837983→0.9516931354 independently. A separately predeclared on-fraction hybrid is weaker than the q proxy:0.9454439731/0.9400889164; preserve these scores rather than selecting them away. Count MAE Tokke0.3030→0.2025 and0.3549→0.1450; Vinje0.2515→0.1655 and0.5076→0.2895. Unit gains are mixed: independent TokkeG3 decreases0.98637→0.97509. No full-validation or leaderboard claim.

All32 solves successful, zero soft slacks, physical generator caps and allocation identities passed, maximum convex mass1, disabled old dispatch exactly0, maximum aggregate float64 conservation1.21e-11Mm3. Retained vertices29–35 forTokke/18–35 forVinje; augmented variables27384–30240. Complete commands, exact IDs/dates, raw input/source/cache checksums, source snapshots, environment, solver configuration, all schedules/convex weights/vertex arrays, full unit scores and artifact hashes: `artifacts/lp-joint-plant-001/manifest.json`. Methods/limitations: `docs/lp_joint_plant.md`.

After reviewing both predetermined benchmark subsets and physical tests, the root authorized full input-only cache `lp-joint-full-001`, command `.venv-ml/bin/python scripts/build_lp_joint_features.py --run-id lp-joint-full-001 --all`, one solver thread. At this ledger entry it is running; the authoritative final status/timing/artifacts will be in `artifacts/lp-joint-full-001/manifest.json`. No supervised training or Kaggle submission launched by this investigation. Kaggle submission ID/score/rank/observation:none. V1/v2 remain unchanged.

## past-state-audit-001 — 2026-11-02T12:12:30.423887+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_past_state.py`. Past-only input availability audit; no labels, hourly feature matrix, training, selection, or Kaggle submission. Manifest with data/source/environment hashes: `artifacts/past-state-audit-001/manifest.json`. Runtime: 0.91s. Pure transform boundary tests recorded separately.

Past-state transform verification: `.venv-ml/bin/python -m unittest discover -s tests -p 'test_past_state_features.py'` passed all7 tests in0.028s, including invariance to changed/deleted future storage and current/future inflow. Proposed first ablation excludes documented half-full constant reservoirs Hyljelihyl/Vatjern:15 reservoirs×3 windows×5 fields=225 case-level columns; all395 LP-v1 baseline columns would be retained (620 total). Details and formula/legitimacy caveats: `docs/past_state_features.md`. No integration or fit has run.


## lp-joint-full-001, verification and complete-year diagnostic — 2026-11-02

Full cache completed: start 2026-11-02T12:03:46.740473+00:00; command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/build_lp_joint_features.py --run-id lp-joint-full-001 --all`; runtime 792.912s, median 0.18634s/case. All3647 cases successful, shape3647×168×87, zero soft slacks,268 cases with negative-price hours. Maximum aggregate conservation1.42e-9Mm3. One HiGHS thread; same Apple M2 Max12-core32GB environment. No training, random seed, early stopping, tuning or ensemble. Feature archive SHA256 `5fd1de4d08bf5db95892dc66a870431f38c3571e1bcc65ec62cbbe8882eb5db6`. Exact input/source/artifact digests, legal case dates/IDs, source snapshots, raw schedules and joint vertex allocations are in `artifacts/lp-joint-full-001/manifest.json`. The complete source closure excludes trainer/export code and preserves v1/v2 unchanged.

`lp-joint-verification-001`: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/verify_lp_joint_cache.py --run-id lp-joint-verification-001`; runtime 2.272s. Strict loader validates every case-hour, all87 feature columns, provenance, finiteness, on fractions/counts and preservation by monetary normalization. Lightweight hour/price+joint matrix612696×89, avoiding an extra large frame while other fits run. Expected full normalized/context frame404columns is documented rather than falsely claimed built by this check. Report/source snapshot:`artifacts/lp-joint-verification-001/manifest.json`.

`lp-joint-diagnostic-001`: start 2026-11-02T12:17:58.774560+00:00; `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/score_lp_joint_diagnostic.py --run-id lp-joint-diagnostic-001`; runtime 1.329s. Evaluate every2021/2022 case-hour-unit with the same fixed p=(q/Qmin)/(1+q/Qmin), no fit/tuning. Micro AUC2021 0.962105338442,2022 0.960666737595; baselinev1 .946964868706/.952251082318. The weaker fixed on-fraction hybrid is retained: 0.946684840903/0.952597441995. These are LP-only ranks, not supervised validation or leaderboard results. All predictions, labels, IDs, unit scores, cache/label/source hashes and environment:`artifacts/lp-joint-diagnostic-001/manifest.json`.

Limitations and counterexamples remain in `docs/lp_joint_plant.md`: TokkeG3 can decline, Haukeli directionality remains unresolved in this v1-based cache, and the dry all-on Haukeli cases largely remain zero-flow. No Kaggle submission ID/score/rank/observation for these runs; no supervised training launched by the audit agent. Root owns the matched model ablation and memory-cap scheduling.

## lp-past-state-specialists-001 — 2026-11-02T12:23:12.399142+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/lp_past_state_specialists.py --run-id lp-past-state-specialists-001`. Runtime: 482.0s. Matched 350-tree three-unit LP-v1 specialist ablation: retain395 baseline columns, add225 strictly past-state fields; same saved sample rows and parameters, two CPU threads. No early stopping, smoothing or projection. Joint weights0/.25/.5/1 chosen on2021 before2022 labels; no fullfit or Kaggle submission. All code/data/checkpoint/sample/environment provenance: `artifacts/lp-past-state-specialists-001/manifest.json`.

- 2021: baseline 0.9943700122907179; locked weight 0.5; selected 0.9944142704862281.

- 2022: baseline 0.9933485421643476; locked weight 0.5; selected 0.9932906222455605.

## lp-past-state-specialists-001-verification — 2026-11-02T12:34:23.193262+00:00

Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/verify_specialist_ablation.py lp-past-state-specialists-001`. Complete: artifact/source/data/checkpoint hashes, exact baseline sample identity, labels/IDs/units, bitwise specialist blend arithmetic and recomputed two-fold micro AUC. No checkpoint replay, fitting or feature loading. Proof: `artifacts/lp-past-state-specialists-001-verification/manifest.json`.

## catboost-lp-specialists-001 — 2026-11-02T12:07:49.440481Z

Status: complete, rejected. Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/catboost_specialists.py --run-id catboost-lp-specialists-001`. Runtime633.4286s. Hypothesis: a different fixed tree learner supplies complementary errors on Haukeli, Lio and Hogga. Same395 normalizedLP-v1 features, exact stored baseline40% training samples, seed4173, six-day-purged2021/2022 chronological splits. Six models, each CatBoost600 iterations/depth6/learning_rate.08/l2_leaf_reg5/border_count128/random_strength1/Bayesian bootstrap/bagging_temperature1/Plain boosting/Logloss;2CPUthreads, no categorical features, evaluation set or early stopping. Expanded defaults for every checkpoint are retained.

One joint specialist weight selected from[0,.25,.5,1] on2021; scores[.994370012291,.994504231894,.994525542721,.994301821433]. Weight0.5 was frozen at2026-11-02T12:12:45.661566Z before2022 prediction/label decoding. Locked2022 score.993245083821 versus baseline.993348542164, a decline of.000103458343; reject without posthoc per-unit selection. Both years are reused development folds. No final fit, submission or repository transfer. The two-year0.998 gate is unmet.

Environment: macOS27.0.1 arm64, Python3.11.15, AppleM2Max12-core/32GB, CatBoost1.2.10; full package versions, code snapshots, data/cache/checkpoint/sample hashes, exact IDs/dates, preprocessing formulas, full configuration and selected arrays are in `artifacts/catboost-lp-specialists-001/manifest.json`, SHA256 `32b6e5155450ad56e71f8486c6622ecdd1e0d36e45f0a4678dee3483e4c65a53`. Detailed report:`docs/catboost_specialists.md`. Kaggle submission ID/score/rank/observation:none.

### CatBoost verification failure and correction — 2026-11-02T12:21:33.175329Z

The first post-run verification command `.venv-ml/bin/python artifacts/catboost-lp-specialists-001/verify_artifacts.py` failed exact equality of reloaded expanded parameter metadata. The failed source snapshot and `verification_failure_001.json` are retained. CatBoost serialization rounded two floating metadata values by at most9.89e-12; no prediction, model, label or blend changed. Corrected verifier uses absolute1e-9 tolerance only for floating parameter metadata. Every other comparison stays exact.

At2026-11-02T12:21:54.451028Z corrected verification passed all48 recorded hashes, exact sample bytes, label/ID/unit alignment, unchanged other units, exact frozen mixing, bounded probabilities, independent pooled AUC, all six600-tree structures and expanded settings. Runtime0.4199s. All six in-run first64-row saved-checkpoint prediction replays had already been exact. Corrected proof:`artifacts/catboost-lp-specialists-001/verification.json`.

verification_failure_001.json SHA256 `d84da773348a30e361dd45f7a17898cf2497efe16828a9621b23b6fc21ab14f1`.

verification.json SHA256 `1bee836d3086f7b2ab5cdb331b9abedf4eebb7fbf341a3680d515e30867df66c`.

## lp-joint-uncertainty-2021-001 — 2026-11-02T12:39:36.740564+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/paired_auc_uncertainty.py --run-id lp-joint-uncertainty-2021-001 --baseline artifacts/lp-candidate-comparison-001/validation_2021.npz --candidate artifacts/lp-joint-comparison-001/selected_validation_2021.npz`. Runtime 4.84s. Fixed-model paired28-day circular case bootstrap, seed4173, 200replicates; no fitting, selection, early stopping or Kaggle submission. Difference 0.00010737984581143944; descriptive95%percentile interval [6.773604515860344e-05, 0.00015745222507125843]. Adaptive research reuse, training uncertainty and distribution shift are not accounted for. Full source/input/artifact hashes, IDs, dates, versions and bootstrap draws: `artifacts/lp-joint-uncertainty-2021-001/manifest.json`.

## lp-large-blend-001 queued — 2026-11-02T12:41:18.668965+00:00

Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/compare_lp_large.py --run-id lp-large-blend-001`. Predeclared all-unit large-LP weights0/.25/.5/.75/1 versus fixed context75/base25 prior, one Vinje projection after raw blending. Waiting for2021 source fold; no2022 prediction file decoded. Protocol/source snapshot: `artifacts/lp-large-blend-001/manifest.json`.

## cpu-mlp-smoke-001 — 2026-11-02T12:44:25.892385+00:00

Status complete. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/smoke_cpu_mlp.py`. Synthetic4096×395 inputs and14 random binary outputs only; no competition data or GPU, no packages installed. MLP128/64 ReLU, Adam.001, batch512, three fixedepochs, seed4173, two BLASthreads. Fit seconds 0.04151754197664559. Expanded settings, versions, warnings and source hash: `artifacts/cpu-mlp-smoke-001/manifest.json`. No Kaggle submission.

## lgb-normalized-lp-joint-001 — 2026-11-02T12:27:52.353020+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-lp-joint-001 --years 2021 2022 --feature-set normalized-lp-joint --trees 350 --sample-fraction 0.4 --threads 3`. Runtime: 998.9s.

Manifest and source snapshot: `artifacts/lgb-normalized-lp-joint-001/manifest.json`. Feature ablation: normalized-lp-joint; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.

- 2021: micro AUC **0.99424256**, prevalence baseline 0.54351580; train ['2015-01-01 00:00:00+00:00', '2020-12-25 00:00:00+00:00'], validation ['2021-01-01 00:00:00+00:00', '2021-12-31 00:00:00+00:00']; six-day purge.
- 2022: micro AUC **0.99340202**, prevalence baseline 0.55445194; train ['2015-01-01 00:00:00+00:00', '2021-12-25 00:00:00+00:00'], validation ['2022-01-01 00:00:00+00:00', '2022-12-31 00:00:00+00:00']; six-day purge.

Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## lp-joint-comparison-001 — 2026-11-02T12:33:49.249171+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/lp_joint_compare.py --run-id lp-joint-comparison-001`. Runtime including waiting for external training: 649.17s.

Hypothesis: joint plant hydraulic features supply complementary scheduling information. Fixed global joint weights [0,.25,.5,.75,1], mixed with the current75%context/25%base recipe and followed by unchanged Vinje projection. Select2021, persist immutable lock before2022 arrays, then apply one fixed2022 acceptance gate; no alternate2022 grid. Both years are reused development folds. No model training, seed, early stopping or package change in this comparison.

- 2021: selected=0.994888491751, current=0.994781111905, matched_base_raw=0.994370012291, matched_base_projected=0.994372219552, joint_raw=0.994242557230, joint_projected=0.994247311938
- 2022: selected=0.993641964015, current=0.993470245918, matched_base_raw=0.993348542164, matched_base_projected=0.993351463186, joint_raw=0.993402020319, joint_projected=0.993403139230

Decision: 2021-selected joint blend improves current recipe on2022.. Full source/input/data/checkpoint-manifest hashes, selection timing, exact IDs, per-unit/month/lead-day diagnostics, environment and failure trace: `artifacts/lp-joint-comparison-001/manifest.json`. Kaggle submission ID/score/rank/observation: none.

## verify-lp-joint-comparison-001 — 2026-11-02T12:44:45.250469+00:00

Status: passed. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/verify_lp_joint_comparison.py --comparison-run lp-joint-comparison-001 --run-id verify-lp-joint-comparison-001`. Runtime0.46s. Independent saved-array verification of source/input/data/artifact hashes, immutable2021 lock, exact flattened convex mixture/projection, labels/IDs/units, pooled AUC and fixed acceptance rule. Also verifies the lock timestamp precedes creation of the joint2022 NPZ. No fitting or changed weights. Proof/source snapshot:`artifacts/verify-lp-joint-comparison-001/manifest.json`. Kaggle submission ID/score/rank/observation:none.

## lp-joint-audit-001 — 2026-11-02, lightweight frozen-cache audit

Status: complete in 1.533 s. Command: `.venv-ml/bin/python scripts/audit_lp_joint.py --run-id lp-joint-audit-001`. No model fit, LP solve, full feature matrix or score selection. Fixed IDs 1, 2008, 2193, 2461, 2653, 2690, 2923, 3647; source/data/archive hashes, exact vertex reconstruction, allocation/cap constraints, independent reservoir balances/objective reconstruction, guarded label metadata reads and in-memory non-start-volume poisoning all pass. Maximum reconstructed pool balance error 3.87e-12 Mm3; objective difference 2.57e-6 currency units. Current trainer manifest's exact joint feature spec and 612696×404 shape match. Full command, source hash, input/cache hashes, environment, per-case results and limitations: `artifacts/lp-joint-audit-001/manifest.json`, `docs/lp_joint_audit.md`. No Kaggle submission or new validation score. Frozen cache and source closure remain unchanged.

## lp-joint-uncertainty-2022-001 — 2026-11-02T12:45:44.049074+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/paired_auc_uncertainty.py --run-id lp-joint-uncertainty-2022-001 --baseline artifacts/lp-candidate-comparison-001/validation_2022.npz --candidate artifacts/lp-joint-comparison-001/selected_validation_2022.npz`. Runtime 5.11s. Fixed-model paired28-day circular case bootstrap, seed4173, 200replicates; no fitting, selection, early stopping or Kaggle submission. Difference 0.00017171809659999315; descriptive95%percentile interval [8.962547370469365e-05, 0.00028069014039029677]. Adaptive research reuse, training uncertainty and distribution shift are not accounted for. Full source/input/artifact hashes, IDs, dates, versions and bootstrap draws: `artifacts/lp-joint-uncertainty-2022-001/manifest.json`.

## CPU MLP preparation and small-fixture tests — 2026-11-02

Prepared fixed protocol in`docs/cpu_mlp.md`, runner`scripts/cpu_mlp.py`, pure normalization/target-boundary/portable-inference helpers`sintef_project/mlp_helpers.py`. No real-data feature matrix or fit was launched during preparation. Existing uv-created environment only; no package/network action. Command`.venv-ml/bin/python -m unittest discover -s tests -p 'test_cpu_mlp.py'` passed5 tests in0.033s, including blocked2022 target decoding, frozen float64 population statistics and clipping, invalid feature rejection, unchanged zero-weight anchor, and exact pickle/portable replay. The only test fit used128 synthetic rows×4 features, hidden5/3,14 random binary targets,3 epochs and2 threads. Syntax checks for runner/helpers passed. Root owns heavy-process launch. No Kaggle submission ID/score/rank/observation.

## lgb-normalized-lp-joint-final-001 — 2026-11-02T12:45:42.929514+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-lp-joint-final-001 --years --feature-set normalized-lp-joint --trees 350 --sample-fraction 0.4 --threads 3 --final`. Runtime: 629.4s.

Manifest and source snapshot: `artifacts/lgb-normalized-lp-joint-final-001/manifest.json`. Feature ablation: normalized-lp-joint; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.


Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

Preparation test/source/environment provenance:`artifacts/cpu-mlp-preflight-001/manifest.json`; original exact test start time was not captured and remains unknown.

## lp-joint-switching-tests-001 and lp-joint-switching-001 — 2026-11-02

Six targeted tests pass (3.948 s unittest, 4.912 s process): free state boundaries, exact physical on-fraction/power/water link, exact fractional transition cost, negative-price forced release, preservation of every feasible active subset and zero-cost objective equivalence. Test source snapshots/output: `artifacts/lp-joint-switching-tests-001/manifest.json`.

Benchmark command: `.venv-ml/bin/python scripts/benchmark_lp_joint_switching.py --run-id lp-joint-switching-001`. Completed in 171.634 s, one HiGHS thread, no fitted model/threshold or grid/cost tuning. Protocol/source hashes were recorded before labels; the same 16 selected 2021 and 16 spread 2015–2020 IDs from `lp-shared-loss-001` were used. Fixed 128-point grid, per-subset upper/lower hulls, exact vertex on-fractions and supplied adjacent-hour start/stop costs; initial/final transitions uncharged because states are unavailable. All other physics remained frozen. Full source/data/selection/artifact hashes, environment, exact IDs/dates, raw solutions/vertices and scores: `artifacts/lp-joint-switching-001/manifest.json`; derivation and limitations: `docs/lp_joint_switching.md`.

Primary q-proxy micro AUC (selected/independent): frozen global hull 0.9647306043/0.9516931354; subset without cost 0.9528390432/0.9251546286; subset with switching 0.9579212988/0.9401603021. Secondary on-fraction hybrid is also weaker than frozen: 0.9373916812/0.9168231502 versus 0.9454439731/0.9400889164. Positive counterpoint retained: Tokke count MAE 0.2025→0.1815 and 0.1450→0.0976; Vinje 0.1655→0.1570 and 0.2895→0.2267. Total fractional transitions 1168.67→533.68. All soft slacks zero, maximum conservation error 1.21e-11 Mm3; zero-cost objectives agree, showing allocation/timing degeneracy can worsen unit diagnostics without changing economics. Median switching solve 4.343 s/case, maximum122458 variables. Recommendation: no full cache for this slower unit-dispatch branch. No supervised or Kaggle result; frozen v1/v2/joint caches unchanged.

## replay-lp-joint-final-001 — 2026-11-02T12:57:36.858745+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id replay-lp-joint-final-001 --years --feature-set normalized-lp-joint --inference-from artifacts/lgb-normalized-lp-joint-final-001 --threads 3`. Runtime: 13.3s.

Manifest and source snapshot: `artifacts/replay-lp-joint-final-001/manifest.json`. Feature ablation: normalized-lp-joint; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.


Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## submission-lp-joint-blend-001 — 2026-11-02T12:58:43.634447+00:00

Status: complete. Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/export_selected_blend.py --run-id submission-lp-joint-blend-001 --selection-run lp-joint-comparison-001 --final-runs lgb-normalized-lp-final-001 lgb-normalized-lp-context-final-001 lgb-normalized-lp-joint-final-001`. No fitting or checkpoint inference; verified selected full-fit recipes, all source/checkpoint/data/artifact hashes, schema and probability serialization. Independent replay is separate. Manifest: `artifacts/submission-lp-joint-blend-001/manifest.json`. Runtime 2.93s. Kaggle ID/score/rank/observation: none at export.

Joint export verifier prepared: `scripts/verify_lp_joint_export.py`, default output run `verify-lp-joint-export-001`. Three targeted tests pass (`.venv-ml/bin/python -m unittest discover -s tests -p 'test_verify_lp_joint_export.py'`): float64 promotion before source-order weighting, byte-level equality including signed zeros/dtype, and invalid coefficient rejection. No full artifact verification ran yet because the joint final/replay/export are not all complete. The root will invoke the verifier after export; no third model or inference job was started.

## cpu-mlp-lp-001 — 2026-11-02T12:59:10.542038+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/cpu_mlp.py --run-id cpu-mlp-lp-001`. Runtime25.90s. Fixed CPU multilabelMLP128/64,14 logistic outputs,24epochs/Adam.001/batch512/alpha1e-4/seed4173/two threads. Exact395LP-v1 features and stored40% training samples; per-fold train-only float64 normalization thenclip[-10,10]/float32. NN jointweights0/.1/.25/.5 versus frozen acceptedjointblend, selected2021 before2022 target decoding; no early stopping or posthoc weights. Both folds are reused development evidence.

- 2021: rawNN=0.9756756531089388; anchor=0.9948884917510937; selected=0.9948884917510937; weight=0.0.

- 2022: rawNN=0.9738789690831663; anchor=0.9936419640150564; selected=0.9936419640150564; weight=0.0.

All source/data/cache/normalization/sample/pickle/portable-array hashes, complete model parameters, warnings, hardware/software, exact split IDs/dates and checkpoint replay proofs:`artifacts/cpu-mlp-lp-001/manifest.json`. Accepted:False. No full-data fit or Kaggle submission; ID/score/rank/observation:none.

## verify-lp-joint-export-001 — 2026-11-02T12:59:52.782787+00:00

Status: failed. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/verify_lp_joint_export.py --run-id verify-lp-joint-export-001`. No fit, inference, LP solve or feature table. Verified frozen joint selection and source/cache/data/checkpoint/artifact hashes, all three saved raw checkpoint replays and CSV bytes, existing replay/selection proofs, float64 blend reconstruction and exact template/readback. Proof: `artifacts/verify-lp-joint-export-001/manifest.json`. Runtime 4.13s. No Kaggle submission.

## tokke-exchangeability-001 — 2026-11-02, training-only ordering analysis

Command: `.venv-ml/bin/python scripts/analyze_tokke_exchangeability.py --run-id tokke-exchangeability-001`. Complete in 0.204 s. Read exactly first2192 training rows (IDs1–2192, 2015-01-01 through2020-12-31), only TokkeG1/G2 labels plus metadata; no holdout predictions or2021/2022 label values parsed. Raw generator attributes are identical, but G1-only78,819 versus G2-only120 case-hours gives99.848% G1 share of discordance, persistent every training year. Fourteen cases contain exceptions. Overlapping weekly counts are descriptive, not independent statistical evidence. This rejects the exchangeable-noise motivation for unconditional pair-mean shrinkage; optional alpha{0,.25,.5,1} validation search was deliberately not run. No prediction/postprocessing/cache changed. Exact source/input hashes, IDs, dates, environment and pair arrays: `artifacts/tokke-exchangeability-001/manifest.json`; interpretation: `docs/tokke_exchangeability.md`. No validation or Kaggle score.

## verify-cpu-mlp-lp-001 — 2026-11-02T13:01:52.586903+00:00

Status:passed. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/verify_cpu_mlp.py --source-run cpu-mlp-lp-001 --run-id verify-cpu-mlp-lp-001`. Runtime0.74s. Small-array CPU-MLP verification: source/data/input/model hashes, exact sample rows, normalization metadata, trusted pickle/portable parameter identity and64 synthetic-row replay, exact frozen probability mixture/labels/IDs/units/AUC, immutable2021 lock and fixed acceptance rule. Full validation checkpoint replay remains the source-run check; no large feature reconstruction or new fit here. Proof:`artifacts/verify-cpu-mlp-lp-001/manifest.json`. No Kaggle submission.

## verify-lp-joint-export-002 — 2026-11-02T13:03:30.855628+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/verify_lp_joint_export.py --run-id verify-lp-joint-export-002`. No fit, inference, LP solve or feature table. Verified frozen joint selection and source/cache/data/checkpoint/artifact hashes, all three saved raw checkpoint replays and CSV bytes, existing replay/selection proofs, float64 blend reconstruction and exact template/readback. Proof: `artifacts/verify-lp-joint-export-002/manifest.json`. Runtime 4.76s. No Kaggle submission.

## tokke-order-comparison-001 — 2026-11-02, soft violation-only pair correction

Command: `.venv-ml/bin/python scripts/compare_tokke_order.py --run-id tokke-order-comparison-001`. Separate authorized experiment after rejecting unconditional mean shrinkage. Three unit tests pass for named-pair scope, convex arithmetic/endpoints, input immutability and invalid inputs. Starting arrays are accepted `lp-joint-comparison-001` with raw weights [.1875,.5625,.25] and existing Vinje projection. Only Tokke G1<G2 probability violations move toward their pair mean; all other probabilities remain bitwise unchanged. Fixed alpha grid0/.25/.5/1, select pooled2021 AUC and lock before decoding2022; accept only strict improvement in both years. No model training, seed, early stopping, subset rule or postprocess.py edit.

2021 micro AUC for alpha0/.25/.5/1: 0.9948884917510937 / 0.9948884168869958 / 0.9948883816529702 / 0.9948883737389923. Alpha0 selected; 2022 unchanged at0.9936419640150564, and no nonzero alpha evaluated there. G2 individual AUC rises slightly, but does not override the pooled-metric decision. These tiny differences are not claimed statistically meaningful. Result rejected under the strict gate; accepted predictions/submission remain unchanged. Protocol/source/input/proof/raw-data hashes, exact fold IDs, selection/decode timestamps, all candidate scores, arrays and artifact digests: `artifacts/tokke-order-comparison-001/manifest.json`, `protocol.json`, `selection_2021.json`. Interpretation: `docs/tokke_exchangeability.md`. No Kaggle submission or migration claim.

## Context-plus-joint comparison preflight — 2026-11-02

One small-array fixture test passed in0.001s: exactzero-weight anchor and exporter-compatible four-source accumulation/projection for every nonzero grid weight. Command`.venv-ml/bin/python -m unittest discover -s tests -p 'test_lp_context_joint_comparison.py'`. No fit, competition-label decode or large feature matrix. Source/environment/timing provenance:`artifacts/context-joint-comparison-preflight-001/manifest.json`. Original exact test start time not captured. Comparison protocol also records metadata-only verification of608columns, preserved521-column prefix and both source/cache closures. No Kaggle submission.

## Kaggle observation — submission-lp-joint-blend-001 — 2026-11-02T13:12:22.956714+00:00

Public AUC **0.99625**, team rank **24**, new personal best over0.99604. Frozen weights base.1875/context.5625/joint.25 plus Vinje projection, selected2021 and locked before2022 predictions existed. Purged localAUC .994888491751/.993641964015. All three checkpoint replay arrays and rawCSV hashes match exactly; independent blend reconstruction/readback passed in verify-lp-joint-export-002. Verifier001 failed only on a prior export manifest updated with Kaggle metadata; historical bytes were reconstructed by restoring its four-null Kaggle object, with all model/data/weight checks unchanged. New exported manifest remains byteimmutable; actual observation is a separate kaggle_observation.json. SHA256 `911ec76d34b7bc2c1134286537a6bf068662effb80f93b200ddb6ec1fc1b110c`. Numeric submission ID/private score unknown. Observed top3 .99820/.99808/.99805. No final-selection checkbox changed. Transfer>=.998both and leaderboard>=.999/top3 goals remain unmet.

## kaggle-neural-bundle-001 — 2026-11-02, local review bundle only

Command: `.venv-ml/bin/python scripts/build_kaggle_neural_bundle.py --run-id kaggle-neural-bundle-001`. Complete in2.694s. Exact allowlist:10 raw/input-schema files, original LPv1 features/manifest/schema-repair copy, strict source closure plus normalized/context/scoring dependencies, frozen neural protocol and referenced baseline manifest. Twenty-four payload files plus bundle manifest;153925168 uncompressed payload bytes. ZIP62925128bytes, SHA256 `2dd110c37ae3bd81cda27104e09386a016bd88a8972e9c8708ee4518d410db44`. Every archive member passed streamed hash/size/allowlist readback; feature header and small IDs/units/columns verified without loading numeric feature matrix. No secrets, environments, checkpoints, unrelated caches/scripts or prediction arrays. No model fit, GPU test, upload, remote execution, validation score or Kaggle submission. Full filelist/hashes/sizes, environment, command and builder snapshot: `artifacts/kaggle-neural-bundle-001/manifest.json`; inner deployment contract: `payload/bundle_manifest.json`; explanatory recipe: `docs/kaggle_neural_bundle.md`. Original raw paths and strict LP provenance remain unchanged.

## Private Kaggle GPU authorization — 2026-11-02

User explicitly approved uploading the prepared63MB bundle (competition inputs/labels, frozenLPfeatures and source) as a PRIVATE dataset to their Kaggle account for GPU training: "Yes, use private Kaggle GPU training". This authorization persists. Root owns upload/session UI; environment agent prepares a portable CUDA TCN with a synthetic smoke gate and separate chronological fold commands. Draft notebook `https://www.kaggle.com/code/ravinav/tdt4173-private-neural-experiment/edit` was verified private; its session was off and Accelerator None when inspected. T4x2 appears as an available option; actual CUDA runtime/remaining quota are not yet measured. The competition attachment contains only sample_submission.csv, motivating the additional explicit upload authorization. No new local package download is attempted.

## lp-context-joint-uncertainty-2021-001 — 2026-11-02T13:16:47.126104+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/paired_auc_uncertainty.py --run-id lp-context-joint-uncertainty-2021-001 --baseline artifacts/lp-joint-comparison-001/selected_validation_2021.npz --candidate artifacts/lp-context-joint-comparison-001/selected_validation_2021.npz`. Runtime 5.16s. Fixed-model paired28-day circular case bootstrap, seed4173, 200replicates; no fitting, selection, early stopping or Kaggle submission. Difference 4.9226918053424384e-05; descriptive95%percentile interval [5.692710195809305e-06, 0.00010071092948416851]. Adaptive research reuse, training uncertainty and distribution shift are not accounted for. Full source/input/artifact hashes, IDs, dates, versions and bootstrap draws: `artifacts/lp-context-joint-uncertainty-2021-001/manifest.json`.

## Continuation regression suite — 2026-11-02T13:17UTC

Command `.venv-ml/bin/python -m unittest discover -s tests`;108tests passed in9.687s. Log: `artifacts/tests-continuation-20261102-1318.log`. Covers scorer/alignment, transforms/LPphysics, chronology, pairordering, weightedAUC, comparisonlocks andexportverification. Pandas3 deprecation warnings are retained; frozen feature sources were not changed just to silence them.

## Private Kaggle dataset created — 2026-11-02T13:27:54.855764+00:00

User-authorized63MBbundle uploaded and dataset verifiedPrivate with25files: https://www.kaggle.com/datasets/ravinav/tdt4173-private-hydraulic-training-inputs. Originalcreationdialog remainedspinning; independentYourWorklist+datasetpage confirmedcreation, no duplicatecreated. Archivehash `2dd110c37ae3bd81cda27104e09386a016bd88a8972e9c8708ee4518d410db44`. Observationstoredseparatelyat`artifacts/kaggle-neural-bundle-001/kaggle_observation.json` tokeepbundlemanifestimmutable. NoGPUcompute started atthisobservation.

## lgb-normalized-lp-context-joint-001 — 2026-11-02T13:02:41.002632+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-lp-context-joint-001 --years 2021 2022 --feature-set normalized-lp-context-joint --trees 350 --sample-fraction 0.4 --threads 3`. Runtime: 1626.9s.

Manifest and source snapshot: `artifacts/lgb-normalized-lp-context-joint-001/manifest.json`. Feature ablation: normalized-lp-context-joint; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.

- 2021: micro AUC **0.99467849**, prevalence baseline 0.54351580; train ['2015-01-01 00:00:00+00:00', '2020-12-25 00:00:00+00:00'], validation ['2021-01-01 00:00:00+00:00', '2021-12-31 00:00:00+00:00']; six-day purge.
- 2022: micro AUC **0.99340406**, prevalence baseline 0.55445194; train ['2015-01-01 00:00:00+00:00', '2021-12-25 00:00:00+00:00'], validation ['2022-01-01 00:00:00+00:00', '2022-12-31 00:00:00+00:00']; six-day purge.

Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## lp-context-joint-comparison-001 — 2026-11-02T13:08:41.203991+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/lp_context_joint_compare.py --run-id lp-context-joint-comparison-001`. Runtime including waiting for external training: 1279.51s.

Hypothesis: retaining521LP-context features and appending87joint hydraulic features improves scheduling information. Fixed global combined weights [0,.25,.5,.75,1], mixed with the current18.75%base/56.25%context/25%joint recipe and followed by unchanged Vinje projection. Select2021, persist immutable lock before2022 arrays, then apply one fixed2022 acceptance gate; no alternate2022 grid. Both years are reused development folds. No model training, seed, early stopping or package change in this comparison.

- 2021: selected=0.994937718669, current=0.994888491751, matched_base_raw=0.994370012291, matched_base_projected=0.994372219552, combined_raw=0.994678494898, combined_projected=0.994680317304
- 2022: selected=0.993691487954, current=0.993641964015, matched_base_raw=0.993348542164, matched_base_projected=0.993351463186, combined_raw=0.993404061184, combined_projected=0.993404731337

Decision: 2021-selected combined blend improves current recipe on2022.. Full source/input/data/checkpoint-manifest hashes, selection timing, exact IDs, per-unit/month/lead-day diagnostics, environment and failure trace: `artifacts/lp-context-joint-comparison-001/manifest.json`. Kaggle submission ID/score/rank/observation: none.

## verify-lp-context-joint-comparison-001 — 2026-11-02T13:54:30.278643+00:00

Status: passed. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/verify_lp_context_joint_comparison.py --comparison-run lp-context-joint-comparison-001 --run-id verify-lp-context-joint-comparison-001`. Runtime0.50s. Independent saved-array verification of source/input/data/artifact hashes, immutable2021 lock, exact flattened convex mixture/projection, labels/IDs/units, pooled AUC and fixed acceptance rule. Also verifies the lock timestamp precedes creation of the combined2022 NPZ. No fitting or changed weights. Proof/source snapshot:`artifacts/verify-lp-context-joint-comparison-001/manifest.json`. Kaggle submission ID/score/rank/observation:none.

## lp-context-joint-uncertainty-2022-001 — 2026-11-02T13:54:30.204877+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/paired_auc_uncertainty.py --run-id lp-context-joint-uncertainty-2022-001 --baseline artifacts/lp-joint-comparison-001/selected_validation_2022.npz --candidate artifacts/lp-context-joint-comparison-001/selected_validation_2022.npz`. Runtime 5.19s. Fixed-model paired28-day circular case bootstrap, seed4173, 200replicates; no fitting, selection, early stopping or Kaggle submission. Difference 4.9523938595141814e-05; descriptive95%percentile interval [-2.238309110815517e-05, 0.00011566176528635958]. Adaptive research reuse, training uncertainty and distribution shift are not accounted for. Full source/input/artifact hashes, IDs, dates, versions and bootstrap draws: `artifacts/lp-context-joint-uncertainty-2022-001/manifest.json`.

## cuda-tcn-code-001 — 2026-11-02T14:02:36.999596+00:00

Portable CUDA trainer and two importable notebooks prepared locally. Default smoke-only; separate explicit2021/2022 folds; externally hashed local2021 selection lock required for2022; frozen395LP inputs,24epochs,batch16,seed4173,train-onlyfloat64stats, fixed architecture and fresh checkpoint OOF replay. Independent audit002 passed9fixturegroups. No localTorch/GPU/realfeaturebuild/fit. Exact package file list/hash/zipreadback: `artifacts/cuda-tcn-code-001/manifest.json`; execution guide: `docs/cuda_tcn_execution.md`. Root owns authorized private upload/hosted execution and its subsequent measurements.

## cuda-tcn-audit-001 / cuda-tcn-audit-002 — 2026-11-02, independent lightweight runner review

Commands: `.venv-ml/bin/python scripts/audit_cuda_tcn.py --run-id cuda-tcn-audit-001`, then `--run-id cuda-tcn-audit-002` after the environment agent added replay, uv requirements and strict staged reuse. Both complete; final fixture runtime0.052s. Final nine fixture groups cover exact archive staging/reuse, path traversal/unlisted/tampered content rejection,2021 future-target redaction, invalid2022 selection locks, heldout-invariant float64 normalization and axis/clipping rules, CUDA-absent refusal, protocol lineage and exact notebook embedded bytes/default controls. No real raw labels/features, Torch/model/GPU execution or training occurred in this audit.

Final reviewed runner SHA256 `1afae39884da9d210a8ef695fb0c99636536bc688ce79961d502bf4c4b743e57`; protocol `d55bee5f39fc50eb5dc9ed48b99f244b7cad75aff1c9bc91c10c3a29de8cdd88`. Source review confirms purged full-case splits, training-only stats, fixed final epoch24/no validation-guided epoch choice, strict frozen feature/cache provenance with only explicit cache-root path canonicalization, genuine CUDA checks/no CPU fallback, offline uv system-site-packages launcher and fresh checkpoint/saved-stat byte-identical OOF replay requirement. Actual hosted CUDA success/replay remains unmeasured. Exact source snapshots/hashes, notebooks and fixture results: `artifacts/cuda-tcn-audit-001/manifest.json`, `artifacts/cuda-tcn-audit-002/manifest.json`; report: `docs/cuda_tcn_audit.md`. No implementation/UI changes or remote action by audit agent; root owns authorized private runtime actions.

## mps-tcn-smoke-002 — 2026-11-02T14:08:25.715360+00:00

Status: blocked. Command: `/Users/heap/programming/tdt4173/.venv-mps/bin/python scripts/smoke_mps_tcn.py --run-id mps-tcn-smoke-002`. Synthetic-only MPS TCN feasibility; no real feature table, labels, or fit. Proof: `artifacts/mps-tcn-smoke-002/manifest.json`. Blocked/failed checks do not establish GPU speed or memory consumption. No Kaggle submission.

## mps-environment-002 / mps-tcn-smoke-002 — 2026-11-02, retry after reported permission change

Root reported that the user enabled full access/network and authorized a normal, non-escalated install in the existing isolated `.venv-mps`. Command: `uv pip install --no-config --index-url https://pypi.org/simple --cache-dir /private/tmp/tdt4173-uv --python .venv-mps/bin/python torch==2.11.0 numpy==2.4.6`. uv0.11.28, Python3.11.15, macOS27.0.1 arm64. Package-index environment overrides were removed without printing their values; the explicit official PyPI index was used. This child command environment still returned DNS lookup failure for `https://pypi.org/simple/numpy/` after3retries; exit2 after8.087s. This is connectivity evidence, not package-version incompatibility. No sandbox override, approval request, alternate index/version, package download workaround or credential change occurred.

Exact command, complete resolver output and environment inventory: `artifacts/mps-environment-002/manifest.json`, `install.log`, `installed_versions.json`, `command_args.json`. The isolated environment inventory remains empty; `.venv-ml` was untouched. Requested synthetic command `.venv-mps/bin/python scripts/smoke_mps_tcn.py --run-id mps-tcn-smoke-002` then recorded blocked at the missing-Torch preflight. No tensor allocation, optimizer step, real feature table/label, GPU measurement or model fit occurred. Preserved source/metadata: `artifacts/mps-tcn-smoke-002/manifest.json`; prior blocked001 unchanged. Root was notified that the new network state may not have propagated to this existing child tool session.

## mps-tcn-smoke-003 — 2026-11-02T14:11:36.816083+00:00

Status: blocked. Command: `/Users/heap/programming/tdt4173/.venv-mps/bin/python scripts/smoke_mps_tcn.py --run-id mps-tcn-smoke-003`. Synthetic-only MPS TCN feasibility; no real feature table, labels, or fit. Proof: `artifacts/mps-tcn-smoke-003/manifest.json`. Blocked/failed checks do not establish GPU speed or memory consumption. No Kaggle submission.

## lgb-normalized-lp-context-joint-final-001 — 2026-11-02T13:55:15.885684+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-lp-context-joint-final-001 --years --feature-set normalized-lp-context-joint --trees 350 --sample-fraction 0.4 --threads 3 --final`. Runtime: 992.5s.

Manifest and source snapshot: `artifacts/lgb-normalized-lp-context-joint-final-001/manifest.json`. Feature ablation: normalized-lp-context-joint; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.


Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## mps-environment-root-003 — 2026-11-02, exact pinned installation succeeds in root session

Root normal command `uv pip install --python .venv-mps/bin/python --index-url https://pypi.org/simple torch==2.11.0 numpy==2.4.6` succeeded after the reported permission/network change, without escalation or bypass. Official PyPI, uv0.11.28, isolated Python3.11.15 arm64 environment; installed requested torch2.11.0/numpy2.4.6 and9dependencies. Exact11-package inventory, command, platform, observation timestamp and log hash are in `artifacts/mps-environment-root-003/manifest.json`; root log `artifacts/mps-environment-root-003.log` SHA256 `469fb1114e80ab3021c665e194c6c7095b58d0530bc84ba71f8478b75b784150`. The previous child-session DNS failure remains preserved, demonstrating different observed session access rather than version incompatibility. No real fit.

The independently requested synthetic child probe `mps-tcn-smoke-003` imported Torch2.11.0, found `mps_built=True` but `mps_available=False`, and stopped before tensors after5.105s. Its source/metadata manifest is `artifacts/mps-tcn-smoke-003/manifest.json`; runtime/memory remain null and no real inputs/labels/model fit were loaded. Root was notified that its own normal tool session may expose a different Metal availability state, as happened with network access. Earlier blocked001/002 were not overwritten. See `docs/neural_tcn_feasibility.md` for current versus historical evidence.

## mps-tcn-smoke-004 — 2026-11-02T14:13:47.759393+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-mps/bin/python scripts/smoke_mps_tcn.py --run-id mps-tcn-smoke-004`. Synthetic-only MPS TCN feasibility; no real feature table, labels, or fit. Proof: `artifacts/mps-tcn-smoke-004/manifest.json`. Blocked/failed checks do not establish GPU speed or memory consumption. No Kaggle submission.

## cuda-xgb-package-001 — 2026-11-02T14:15:42.715323+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/build_cuda_xgb_package.py --run-id cuda-xgb-package-001`. Fixed full14-unit GPU-XGB preparation only; no remote launch, upload, package install, full matrix or model fit. Reuses sealed24-file data bundle, freezes shared-loader/source/protocol hashes and originalsample-SHA checks. CUDA:1,600/depth6/fixedpriorparameters; local2021weight lock gates separate2022. Code-only archive36324bytes, SHA`29116fc622237b83065478fb407e67320dd45e263736c0af535eb0269d188139`. ProtocolSHA`7317e241c9a579531def94aec898b4cb3ae3192a5f5818062ee12b94db5b9b7c`. Every ZIP member readback/hash checked. Allsource/provenance:`artifacts/cuda-xgb-package-001/manifest.json`. No Kaggle submission.

## replay-lp-context-joint-final-001 — 2026-11-02T14:17:55.573601+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id replay-lp-context-joint-final-001 --years --feature-set normalized-lp-context-joint --inference-from artifacts/lgb-normalized-lp-context-joint-final-001 --threads 2`. Runtime: 19.4s.

Manifest and source snapshot: `artifacts/replay-lp-context-joint-final-001/manifest.json`. Feature ablation: normalized-lp-context-joint; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.


Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## submission-lp-context-joint-blend-001 — 2026-11-02T14:18:27.010624+00:00

Status: complete. Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/export_selected_blend.py --run-id submission-lp-context-joint-blend-001 --selection-run lp-context-joint-comparison-001 --final-runs lgb-normalized-lp-final-001 lgb-normalized-lp-context-final-001 lgb-normalized-lp-joint-final-001 lgb-normalized-lp-context-joint-final-001`. No fitting or checkpoint inference; verified selected full-fit recipes, all source/checkpoint/data/artifact hashes, schema and probability serialization. Independent replay is separate. Manifest: `artifacts/submission-lp-context-joint-blend-001/manifest.json`. Runtime 3.51s. Kaggle ID/score/rank/observation: none at export.

## GPU XGBoost preparation review correction — 2026-11-02

Before any remote use, peer review identified missing local checks for anchorNPZ identity, returnedshared-loader identity and exact frozen case-ID vectors. The selector now verifies all three; the GPU learning recipe is unchanged. Addedtampered-anchor fixture; fourtests passed0.005s using`.venv-ml/bin/python -m unittest discover -s tests -p 'test_cuda_xgb.py'`. NoGPUfit/smoke or largefeaturematrix. Originalthree-test source snapshot preserved, revisedsource/test proof:`artifacts/cuda-xgb-preflight-001/review_fix/manifest.json`. Exact originalteststart time unknown. No Kaggle submission.

## verify-lp-context-joint-export-001 — 2026-11-02T14:19:02.648169+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/verify_lp_context_joint_export.py --run-id verify-lp-context-joint-export-001`. No fit, inference, LP solve or feature table. Verified frozen joint selection and source/cache/data/checkpoint/artifact hashes, all four saved raw checkpoint replays and CSV bytes, existing replay/selection proofs, float64 blend reconstruction and exact template/readback. Proof: `artifacts/verify-lp-context-joint-export-001/manifest.json`. Runtime 5.51s. No Kaggle submission.

## cuda-tcn-selection-preflight-001 — 2026-11-02T14:24:48.647574+00:00

Six synthetic integrity/alignment/probability-bound/byte-replay/tie/lock tests passed in0.023s. Command: `.venv-ml/bin/python -m unittest discover -s tests -p 'test_cuda_tcn_selection.py'`. Original test start time was not recorded. Added local-only CUDA TCN artifact verifier and frozen neural blend selector, preserving audited trainer/protocol bytes, prior projected joint anchor and [0,.1,.25,.5] grid. No Torch import, real feature/label array, model fit, package installation or remote action. Source snapshots, hashes, environment and exact proof: `artifacts/cuda-tcn-selection-preflight-001/manifest.json`. Independent review requested before real selection. No Kaggle submission/score/rank.

## cuda-tcn-selector-check-001 — 2026-11-02T14:24:55.926484+00:00

Seven synthetic download-verifier tests passed. Command: `.venv-ml/bin/python -m unittest tests.test_select_cuda_tcn_blend -v`. No Torch, GPU, real feature table or model fit. Checks checksum-before-decode, OOF replay bytes, saved training-only normalizer scope, all epoch-case coverage, narrow portable cache path, fixed-weight scope. Source/environment/proof: `artifacts/cuda-tcn-selector-check-001/manifest.json`. Fixed selection protocol: `docs/cuda_tcn_selection.md`. No quality or Kaggle score claimed.

## verify-lp-context-joint-export-002 — 2026-11-02T14:25:19.641385+00:00

Status: failed. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/verify_lp_context_joint_export.py --run-id verify-lp-context-joint-export-002`. No fit, inference, LP solve or feature table. Verified frozen context-plus-joint selection and source/cache/data/checkpoint/artifact hashes, all four saved raw checkpoint replays and CSV bytes, existing replay/selection proofs, float64 blend reconstruction and exact template/readback. Proof: `artifacts/verify-lp-context-joint-export-002/manifest.json`. Runtime 5.58s. No Kaggle submission.

## lgb-normalized-lp-large-001 — 2026-11-02T12:35:14.412422+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-lp-large-001 --years 2021 2022 --feature-set normalized-lp --trees 700 --leaves 63 --learning-rate 0.04 --min-child-samples 100 --sample-fraction 0.7 --seed 4173 --threads 2`. Runtime: 6633.0s.

Manifest and source snapshot: `artifacts/lgb-normalized-lp-large-001/manifest.json`. Feature ablation: normalized-lp; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.

- 2021: micro AUC **0.99417944**, prevalence baseline 0.54351580; train ['2015-01-01 00:00:00+00:00', '2020-12-25 00:00:00+00:00'], validation ['2021-01-01 00:00:00+00:00', '2021-12-31 00:00:00+00:00']; six-day purge.
- 2022: micro AUC **0.99333404**, prevalence baseline 0.55445194; train ['2015-01-01 00:00:00+00:00', '2021-12-25 00:00:00+00:00'], validation ['2022-01-01 00:00:00+00:00', '2022-12-31 00:00:00+00:00']; six-day purge.

Kaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.

## lp-large-blend-001 result — 2026-11-02T14:26:02.890938+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/compare_lp_large.py --run-id lp-large-blend-001`. Selected: lp_large_w0.25; accepted: True; accepted candidate: lp_large_w0.25. No fitting or Kaggle submission. Full provenance, scalar grid,2021 lock, raw/base comparisons and accepted/selected predictions: `artifacts/lp-large-blend-001/manifest.json`.

- 2021: selected 0.9948175945039104; prior 0.9947811119052823; large raw 0.9941794368650142; baseline raw 0.9943700122907179.

- 2022: selected 0.993542917367456; prior 0.9934702459184567; large raw 0.9933340358239988; baseline raw 0.9933485421643476.

## verify-lp-context-joint-export-003 — 2026-11-02T14:26:21.299897+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/verify_lp_context_joint_export.py --run-id verify-lp-context-joint-export-003`. No fit, inference, LP solve or feature table. Verified frozen context-plus-joint selection and source/cache/data/checkpoint/artifact hashes, all four saved raw checkpoint replays and CSV bytes, existing replay/selection proofs, float64 blend reconstruction and exact template/readback. Proof: `artifacts/verify-lp-context-joint-export-003/manifest.json`. Runtime 5.68s. No Kaggle submission.

## verify-lp-context-joint-export-preparation-001 — 2026-11-02T14:27:17.196262+00:00

Initial four synthetic tests passed (0.001s); final five tests passed (0.002s), command `.venv-ml/bin/python -m unittest tests.test_verify_lp_context_joint_export`. Guards cover float64-before-weighting, one projection, repeated shared-cache references, exact current hash versus the sole pinned historical Kaggle-only exception, invalid inputs and byte comparison. Duplicate command `.venv-ml/bin/python scripts/verify_lp_context_joint_export.py --run-id verify-lp-context-joint-export-001` exited1 with FileExistsError after0.848s because root had already completed an earlier verifier snapshot. That proof was preserved. Stronger revision002 failed only because its helper rejected an exact already-observed context-manifest reference; the equality branch was corrected without widening the sole historical exception. Revision003 passed all extended checks and preserves002. Full preparation/duplicate/source provenance: `artifacts/verify-lp-context-joint-export-preparation-001/manifest.json`; final proof: `artifacts/verify-lp-context-joint-export-003/manifest.json`. No fit, feature construction, inference or Kaggle action.

## Kaggle observation — 2026-11-02T14:28:26.911798+00:00

submission-lp-context-joint-blend-001 scored **0.99631** in authenticated Kaggle UI; user reported current team rank **23**. Exact provenance and observation: docs/kaggle_scores.json and separate artifact kaggle_observation.json. Verified export manifest unchanged; all four replays passed. No final-selection checkboxes changed. Both thresholds remain unmet.

## cuda-tcn-selector-independent-review-001 — 2026-11-02T14:28:31.976632+00:00

Status: pending reconciliation. No model fitting or real blend selection. SHA-verified downloaded archive and original remote fold/smoke manifests; safe fresh extraction in `artifacts/cuda-tcn-download-001/`. Independent static audit identified live source/test mismatch: seven current synthetic tests passed; other suite failed importing absent helpers. Failed test output retained in `artifacts/cuda-tcn-selector-review-001-tests.log`; exact command and source snapshots in `artifacts/cuda-tcn-selector-independent-review-001/manifest.json`. Reconcile selector and tests in a fresh immutable package before selection. No Kaggle score/rank/submission.

## cuda-xgb-audit-recovery-001 and verify-lp-large-recovery-001 — 2026-11-02T14:28:36.995494+00:00

Independent recovery review only; no model fit, feature-table construction, upload, submission or repository transfer. GPU-XGB package/source/anchor closure, every ZIP member, original sampled rows (146,899/171,427) and four CPU protocol fixtures passed; immutable code archive SHA256 `29116fc622237b83065478fb407e67320dd45e263736c0af535eb0269d188139`. Audit took 1.414s and prepared separate hosted setup/smoke/2021 cells; actual CUDA smoke remains pending. Initial `uv run --offline --no-project --python .venv-ml/bin/python python -m unittest tests.test_cuda_xgb -v` failed on default cache permissions; the same check passed using `UV_CACHE_DIR=/private/tmp/tdt4173-xgb-uv-cache`, with no install. Full commands, source snapshots, package/environment/hardware information, seed4173, inherited data/protocol paths/checksums and artifact hashes: `artifacts/cuda-xgb-audit-recovery-001/manifest.json`.

The existing large-LP training and watcher completed by14:26:23UTC without restart. Frozen25%large blend accepted against its original context75/base25 anchor:2021 `0.9948175945039104` vs `0.9947811119052823`;2022 `0.993542917367456` vs `0.9934702459184567`. Independent1.135s verifier checked all source/data/model artifact hashes and reproduced chosen/anchor arrays exactly, confirmed2021lock and unchanged final configuration. Commands and full provenance: `artifacts/verify-lp-large-recovery-001/manifest.json`; original hyperparameters/splits/training runtime/checkpoints remain in `artifacts/lgb-normalized-lp-large-001/manifest.json`, comparison in `artifacts/lp-large-blend-001/manifest.json`. Both years are reused development evidence. This accepted ablation trails both later joint and608feature recipes on both years, so no replacement fullfit is recommended. Transfergate false; Kaggle ID/score/rank/observation are null for these runs. Operational report: `docs/xgb_large_recovery.md`.

## lp-large-blend-verification-001 — 2026-11-02T14:28:55.067750+00:00

Status: complete. Command: `.venv-ml/bin/python scripts/verify_lp_large_comparison.py`. All comparison source/data/artifact hashes checked; independent float64 source-order composition and Vinje projection reproduce both saved folds byte for byte. No model fit or checkpoint inference. Proof: `artifacts/lp-large-blend-verification-001/manifest.json`. No full fit: locked blend improves its original context/base prior but remains below the newer accepted joint blend on both reused development years.

## cuda-tcn-blend-2021-001 — 2026-11-02T14:30:17.440259+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python artifacts/tcn_audit_reconciled_20261102T142849Z_a892a747/scripts/select_cuda_tcn_blend.py --year 2021 --gpu-run artifacts/cuda-tcn-download-001/cuda-tcn-2021-001 --manifest-sha256 73e9df6ccd4ab7b7d94f39c0dda471b53f4509a2443ce51e30c6e8b968ffc49e --smoke-manifest artifacts/cuda-tcn-download-001/cuda-tcn-smoke-001/manifest.json --run-id cuda-tcn-blend-2021-001`. Local CUDA TCN download verification and fixed [0,.1,.25,.5] selection; no model fit. Frozen accepted joint anchor; 2021 lock precedes separate 2022 fit. Selected weight 0.0, micro AUC 0.9948884917510937, anchor 0.9948884917510937. Accepted: None. Complete hashes, environment, split IDs and replay provenance: `artifacts/cuda-tcn-blend-2021-001/manifest.json`. Reused chronological development folds; no Kaggle submission or public-score tuning.

## cuda-return-audit-001 — 2026-11-02T14:30:26.669885+00:00

Independent read-only review of portable GPU XGBoost runner/launcher/selector and TCN returned-result selector. Eleven small synthetic fixtures passed (0.141s), command `.venv-ml/bin/python -m unittest tests.test_cuda_xgb tests.test_select_cuda_tcn_blend`. Initial XGB anchor-byte, chronological-ID and shared-loader checks were corrected before package freeze and verified in the reviewed revision. No remaining blocking source finding; actual GPU smoke/real returned fold checks remain separate. No model execution, real feature table, checkpoint deserialization or remote action. Full reviewed source snapshots/hashes/environment: `artifacts/cuda-return-audit-001/manifest.json`; findings: `docs/cuda_return_audit.md`.

## cuda-tcn-selector-independent-review-001 reconciliation — 2026-11-02T14:30:32.380482+00:00

Approved immutable selector `artifacts/tcn_audit_reconciled_20261102T142849Z_a892a747/scripts/select_cuda_tcn_blend.py`, SHA`b14edb9f2e609038020202fd57c7c279b33835ba7c330b8608c8476bee51ad5f`, after independent diff review and seven snapshot tests. Required external manifest/original smoke hashes, exact allowlist, pinned protocol/dependencies and2022 lock argmax consistency were added. Separate returned2021-fold verification passed:365cases×168hours×14units, rawAUC0.9851992438242643, originalsmoke, IDs/labels, normalizer scope and fresh checkpoint replay bytes verified. No blend grid evaluated by this reviewer; tcn_audit owns selection. Proof: `artifacts/cuda-tcn-selector-independent-review-001/reconciled_approval.json`. No Kaggle submission or score.

## CUDA TCN 2021 rejection and selector reconciliation — 2026-11-02T14:33:17.686330+00:00

`cuda-tcn-blend-2021-001` selected zero neural weight. Raw2021 AUC0.9851992438242643; grid weights0/.1/.25/.5 give0.9948884917510937/0.9947939462246643/0.9944392967058471/0.9932144346578071. All positive blends fail2021; root rejected branch and did not launch2022. No neural2022 or Kaggle result claimed. LockSHA256 `9c59e49b3954c9c4e11d668316fe9c76fa9732f848cfbff166001e82162cd9d3`; decision is separate `artifacts/cuda-tcn-blend-2021-001/rejection_decision.json`, preserving verified manifests.

Two conflicting live selector/test variants appeared during implementation. Real selection used only independently reviewed unique snapshot `artifacts/tcn_audit_reconciled_20261102T142849Z_a892a747`, selectorSHA256 `b14edb9f2e609038020202fd57c7c279b33835ba7c330b8608c8476bee51ad5f`. Nine targeted synthetic snapshot tests passed0.173s. Source conflicts, first reconciliation patch failure, unmatched historical tests, exact commands, environment, hashes and peer review are preserved in its manifest. No extra real fit or local feature process. `docs/tcn_recovery.md` is the current branch report.

## cuda-tcn-live-reconciliation-001 — 2026-11-02T14:36:32.988539+00:00

Status: complete. Live TCN selector now dispatches to exact independently reviewed snapshot SHA`b14edb9f2e609038020202fd57c7c279b33835ba7c330b8608c8476bee51ad5f` and pins dependencies; importing its helpers preserves unrelated package resolution. Nine matching synthetic tests pass0.171s. Fullsuite command:`UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python -m unittest discover -s tests`:126tests passed9.544s. Initial uv invocation failed before tests due unwritable defaultcache; failure retained, then writable taskcache used offline. Fullsourcehashes, before/after snapshots and logs:`artifacts/cuda-tcn-live-reconciliation-001/manifest.json`. No package installs, model training, probability/lock changes, Kaggle submission or leaderboard score.

## Hosted XGBoost smoke and2021 launch — 2026-11-02T14:37:25.537097+00:00

Private code dataset created and privacy verified; attached to existing private neural notebook. Every code member hash verified. Synthetic cuda-xgb-smoke-001 completed oncuda:1 in2.101924s in offline uv environment. Explicit2021 fit cuda-xgb-2021-001 launched PID197;2022notlaunched. Source/protocol/data closure and exact commands in cuda-xgb-package-001 and cuda-xgb-audit-recovery-001; browser observation in cuda-xgb-hosted-observation-001. No leaderboard/validation score inferred; download/checkpoint verification pending.

## cuda-tcn-live-reconciliation-001 source vendoring — 2026-11-02T14:38:56.044845+00:00

Status:complete. Removed normal-source dependency on ignoredartifacts: copied exact approved selector/trainer/projection files into `scripts/reviewed_tcn_selector_v1/` and repointed hash-pinned shim. Selector remains SHA`b14edb9f2e609038020202fd57c7c279b33835ba7c330b8608c8476bee51ad5f`; historical lock interpretation unchanged. Targeted9tests passed0.169s viauv offline, importprobe confirms package source, restoredsys.path and unrelatedfeatureimports. Prior126-test fullsuite remains recorded; no unnecessary fullrerun after this source-only relocation. Completehashes:`artifacts/cuda-tcn-live-reconciliation-001/source_vendor.json`. No modeltraining, selection, submission or scoreboardclaim.

## xgb_return_preflight_20261102T144214Z_d148bf91 — 2026-11-02T14:42:51.585268+00:00

Three synthetic transfer/native-config validation tests passed0.007s. Command:`.venv-ml/bin/python -m unittest discover -s tests -p 'test_verify_cuda_xgb_return.py'`. External verifier pins hosted ZIP/fold/smoke hashes and independently checks14model/configs, exact sample/splits/source/bundle/OOF before invoking original protocol-pinned selector. Selector2182b0… and training protocol7317e… remain unchanged. Unique wrapper/source/test snapshot and environment:`artifacts/xgb_return_preflight_20261102T144214Z_d148bf91/manifest.json`. Root downloaded2021ZIP; source review pending before real selection. No model fit or2022label read.

## cuda-xgb-select-2021-001 — 2026-11-02T14:43:11.044220+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python /Users/heap/programming/tdt4173/artifacts/verify-cuda-xgb-2021-return-001/frozen_source/scripts/select_cuda_xgb_blend.py --year 2021 --gpu-run /Users/heap/programming/tdt4173/artifacts/verify-cuda-xgb-2021-return-001/download/cuda-xgb-2021-001 --protocol /Users/heap/programming/tdt4173/docs/cuda_xgb_protocol.json --run-id cuda-xgb-select-2021-001`. Local GPU-XGB selection/evaluation only, no fit. Fixed global weights0/.1/.25/.5 on2021, immutable lock before separately authorized2022 fit, strict both-year improvement against frozen joint anchor; no leaderboard retuning. Selected0.25, microAUC0.9949355929850593, anchor0.9948884917510937. Full hashes/schema/checkpoint-replay provenance:`artifacts/cuda-xgb-select-2021-001/manifest.json`. No Kaggle submission.

## verify-cuda-xgb-2021-return-001 — 2026-11-02T14:43:10.682939+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python artifacts/xgb_return_preflight_20261102T144214Z_d148bf91/verify_cuda_xgb_return.py --workspace /Users/heap/programming/tdt4173 --archive artifacts/cuda-xgb-2021-001-download.zip --archive-sha256 4f8109ef9062feb41350dfd9874d6b6d93c95182345306f8438c173d65bae1d8 --manifest-sha256 a20279e00a5fbdff4c8c6f9c096f58440fbb259ab950b4d85b22bd3f103a5d2a --smoke-manifest-sha256 f31b812df3cdea770b5f04861608d884faa8bbcaccebbe078a6e95c6736e7e4f --year 2021 --gpu-run-name cuda-xgb-2021-001 --run-id verify-cuda-xgb-2021-return-001 --selection-run-id cuda-xgb-select-2021-001`. Strict returned GPU-XGB archive/manifest/smoke/checkpoint/config/sample/source/OOF verifier, then exact protocol-pinned selector; no model fit. Frozen earlier joint anchor. Scores:{'0.0': 0.9948884917510937, '0.1': 0.9949209266933028, '0.25': 0.9949355929850593, '0.5': 0.9948926897265898}; selected:0.25. Full source snapshots, hashes, environment and verification: `artifacts/verify-cuda-xgb-2021-return-001/manifest.json`. No Kaggle submission.

## GPU XGBoost 2021 selection lock and private transfer preparation — 2026-11-02T14:45:09.267590+00:00

Full returned-model verification and frozen selection succeeded. RawAUC0.9945072625313349; grid0/.1/.25/.5 gives0.9948884917510937/0.9949209266933028/0.9949355929850593/0.9948926897265898. Selected25%XGB against original joint anchor;2022pending, no replacement or transfer claim. LockSHA256 `315c420fa43a6a046fe97facafd473069de1d6761ad5d0f372617893bf3db040`. Exact-byte compressed private notebook transfer cell prepared and compile-checked at `artifacts/cuda-xgb-lock-transfer-2021-001/`; not executed by this agent. All manifests/checkpoints/probabilities/logs preserved. Separate independent-review observation records exact approved wrapper. Current recipe:`docs/xgb_return_verification.md`. No local training, finalfit, Kaggle submission, leaderboard score or rank.

## current-blend-residuals-001 and nonlinear-pq preparation — 2026-11-02T14:45:48.520288+00:00

Current frozen four-source blend error analysis completed in0.979s with exact pair-loss/Brier identities. Haukeli/Lio/Hogga explain40.53%/39.91% of2021/2022 pooled error; these are reuseddevelopmentdiagnostics, no candidate retuning or fitting. Source/script/environment/rawdata/arrayhashes and fullcaseIDs: `artifacts/current-blend-residuals-001/manifest.json`.

Independent reviewer approved distinct nonlinearPQ on-profit/DP614feature three-specialist ablation, with sameweight saved608control gate and2021lock before2022. Seven numerical/physical/leakage/lock tests pass0.300s. One firstfixture used2rows where canonicalprojection requires168; failure retained and fixture corrected, with no model/formula change. Separate systemPython metadata command lackedNumPy and was rerun throughuv. Failed and successful proof: `artifacts/nonlinear-pq-preflight-001/manifest.json` and `successful_review.json`.

Input-only cache `nonlinear-pq-features-001` completed3647cases×168×6 in16.5276s;onecase.01527s;1506distinct exact unit/headcurves. Sources/inputs/allcaseIDdates/rawpercase LPsolutionhashes, allcurvegrid/formula settings/environment and outputhashes: `artifacts/nonlinear-pq-features-001/manifest.json`. FeatureSHA`cf2c8661972c2cf62f9ef51b0f4109846c44b50d279cf537ff712645cede40c7`. Rawon-rewards retained. NoLP re-solve or model fit in cachegeneration; noKaggle submission. Prepared experiment frozen rationale/protocol:`docs/nonlinear_pq_protocol.md`. Trainer will log its separate singleyear run. No0.998/transfergate or0.999/top3 claim.

## cuda-xgb608-package-001 — 2026-11-02T14:47:50.353823+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/build_cuda_xgb608_package.py --run-id cuda-xgb608-package-001`. New fixed608GPU-XGB feature ablation package only; no featurematrix/model/GPU run or upload. Same600/depth6 hyperparameters and exact40%row samples as395branch; fixed current608-blend anchor. Original inputbundle preserved; separate minimal private jointcache/source supplement. Protocol/code/data/source/ZIP readback hashes and environment:`artifacts/cuda-xgb608-package-001/manifest.json`. Runtime1.858s. Independent audit required before use. NoKaggle submission/score.

## cuda-xgb608-tests-001 — 2026-11-02T14:48:59.856350+00:00

Status:passed. Five small protocol/staging/sample tests in0.092s. Exactcommand:`UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python -m unittest tests.test_cuda_xgb608`. Modelhyperparameters match395 exactly; unioncollision/tamper/extrafilechecks, restrictedportablepaths, improving2021argmax/sourcehashgate and both originalsampleSHA checks pass. Onlycase metadata read; no largefeaturematrix/model/GPU launch. Environment/source/provenance:`artifacts/cuda-xgb608-tests-001/manifest.json`. NoKaggle score or submission.

## nonlinear-pq-specialists-2021-001 — 2026-11-02T14:46:15.947453+00:00

Status:complete;year2021;614features,three350-tree specialists,exact40%row samples,six-daypurge,seed4173,two threads,noearlystopping. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train_nonlinear_pq_specialists.py --year 2021 --run-id nonlinear-pq-specialists-2021-001`. Frozen global weight0.5;selectedAUC0.9949778297213558,anchor0.9949377186691472,matchedcontrol0.9949257714128659. Beatsboth:True. Runtime232.940s. Fullrawdata/source/cache/model/environment/split/sample/protocol hashes and exactcheckpointreplay:`artifacts/nonlinear-pq-specialists-2021-001/manifest.json`. NoKaggle submission;reuseddevelopmentfold.

## XGBoost2021 selection and locked2022 launch — 2026-11-02T14:50:34.084261+00:00

2021 raw .9945072625313349; verified frozen25%XGB blend .9949355929850593 vsjointanchor .9948884917510937. Exact source/checkpoint/nativeGPU/sample/OOF proof verify-cuda-xgb-2021-return-001. Selection lock315c420fa43a6a046fe97facafd473069de1d6761ad5d0f372617893bf3db040 transferred and byte-verified before2022PID231 launch; one failed compressed transport preserved, recovered through exact canonicalJSON reconstruction. Full observation cuda-xgb-hosted-observation-002/manifest.json. 2022 acceptance pending; no leaderboard or migration claim. Notebook quicksaved version1 without rerun.

## cuda-xgb608-private-dataset-001 — 2026-11-02T14:50:46.919269+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/build_cuda_xgb608_dataset.py`. Derived singlePRIVATE dataset archive with separatecode/supplement namespaces and exact-hash discovery launcher; original608package/original395bundle unchanged. No upload/launch. ZIPreadback everymemberverified. ArchiveSHA`480cacd6a83aef078104fd35ddfeb5c0569f0dd7d84158f63f82a1511d766a9a`, bytes51053983; protocolSHA`789c2a6419c526079e18ea83ac5dd1143ef0b2c590e522b163ddf4e890a72401`. Fullsources/environment/hashes:`artifacts/cuda-xgb608-private-dataset-001/manifest.json`. Two small discoverytests pass0.002s; nofeaturematrix/model. NoKaggle score.

## cuda-xgb608-independent-audit-001 — 2026-11-02T14:51:04.234750+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_cuda_xgb608_package.py`. Independent608GPU package/source/ZIP/schema/cache/sample/anchor audit; five supplied small fixtures plus poisoned future-target prefix check. No featurematrix, training, model import, GPU or upload. Exact hashes, source, environment, runtime and scope:`artifacts/cuda-xgb608-independent-audit-001/manifest.json`. Frozen code/protocol/input archives unchanged. NoKaggle result.

Discovery-test timing correction for `cuda-xgb608-private-dataset-001`: the retained test log reports **0.003s**, not the builder ledger text's hardcoded0.002s. Both tests passed; no data/code-in-archive/protocol/checksum changed. Future builder log text now refers to the test log instead of a hardcoded time.

## cuda-xgb608-dataset-independent-audit-001 — 2026-11-02T14:51:57.339012+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_cuda_xgb608_dataset.py`. Independent derived608PRIVATE ZIP/helper audit. Every inherited code/supplement payload byte matches previously auditedpackage001; unique exact-hash discovery helper and two fixtures verified. No featurematrix/training/remote action. Proof:`artifacts/cuda-xgb608-dataset-independent-audit-001/manifest.json`. NoKaggle score.

## verify-nonlinear-pq-selection-2021-001 — 2026-11-02T14:52:35.162141+00:00

Status:passed. Independentsavedartifact/protocol/source/data/cache/sample/chronologicalID/checkpoint-hash verification; nofit orfeaturematrix. Recomputed fullfixedsplicegrid and sameweightmatched608control; exactselectedarraybytes verified. SelectedAUC0.9949778297213558,anchor0.9949377186691472,control0.9949257714128659,eligibleTrue. Proof/source/command:`artifacts/verify-nonlinear-pq-selection-2021-001/manifest.json`. Runtime1.040s. Unitcounterexamples retained; noKaggle submission orscore.

## cuda-xgb608-return-preflight-001 — 2026-11-02T14:54:30.003545+00:00

Status:awaitingindependentreview. Three small archive/nativeconfig tests pass; no realGPUartifact selection ormodel/featurework. New608-return wrapper pins protocol789c…, validates originalbundle+supplement+608reference, bothsource/loaders, synthetic608smoke, all14UBJ/config/replay/sample/probability/checks and originalfixedselector. No395filechanged. Exactcommand,sourcehashes/snapshots/environment/testlog:`artifacts/cuda-xgb608-return-preflight-001/manifest.json`. NoKaggle submission.

## cuda-xgb-select-2022-001 — 2026-11-02T14:54:58.348734+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python /Users/heap/programming/tdt4173/artifacts/verify-cuda-xgb-2022-return-001/frozen_source/scripts/select_cuda_xgb_blend.py --year 2022 --gpu-run /Users/heap/programming/tdt4173/artifacts/verify-cuda-xgb-2022-return-001/download/cuda-xgb-2022-001 --protocol /Users/heap/programming/tdt4173/docs/cuda_xgb_protocol.json --run-id cuda-xgb-select-2022-001 --selection-lock artifacts/cuda-xgb-select-2021-001/selection_2021.json --selection-lock-sha256 315c420fa43a6a046fe97facafd473069de1d6761ad5d0f372617893bf3db040`. Local GPU-XGB selection/evaluation only, no fit. Fixed global weights0/.1/.25/.5 on2021, immutable lock before separately authorized2022 fit, strict both-year improvement against frozen joint anchor; no leaderboard retuning. Selected0.25, microAUC0.9937259135724086, anchor0.9936419640150564. Full hashes/schema/checkpoint-replay provenance:`artifacts/cuda-xgb-select-2022-001/manifest.json`. No Kaggle submission.

## verify-cuda-xgb-2022-return-001 — 2026-11-02T14:54:57.960364+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python artifacts/xgb_return_preflight_20261102T144214Z_d148bf91/verify_cuda_xgb_return.py --workspace /Users/heap/programming/tdt4173 --archive artifacts/cuda-xgb-2022-001-download.zip --archive-sha256 ae2b12a3f0a8cfc9cd11acfebbcc2153760a9757c8cfecf9adcbe168c7234e4f --manifest-sha256 e8077c744039ecf3b1d702e86ea670dddbfd79b8e93f3fba0f33b0d61d5562df --smoke-manifest-sha256 f31b812df3cdea770b5f04861608d884faa8bbcaccebbe078a6e95c6736e7e4f --year 2022 --gpu-run-name cuda-xgb-2022-001 --run-id verify-cuda-xgb-2022-return-001 --selection-run-id cuda-xgb-select-2022-001 --selection-lock artifacts/cuda-xgb-select-2021-001/selection_2021.json --selection-lock-sha256 315c420fa43a6a046fe97facafd473069de1d6761ad5d0f372617893bf3db040`. Strict returned GPU-XGB archive/manifest/smoke/checkpoint/config/sample/source/OOF verifier, then exact protocol-pinned selector; no model fit. Frozen earlier joint anchor. Scores:{'0.0': 0.9936419640150564, '0.25': 0.9937259135724086}; selected:0.25. Full source snapshots, hashes, environment and verification: `artifacts/verify-cuda-xgb-2022-return-001/manifest.json`. No Kaggle submission.

Independent review completed for `cuda-xgb608-return-preflight-001`: tcn_audit approved exact wrapperSHA`8b192955b39c412cc924472c753c96a4413b252469ee807448d4d77d8bc5060b` after full delta review from approved395returnwrapper. Proof:`artifacts/cuda-xgb608-return-preflight-001/independent_approval.json`. No608selection orfit launched.

## compare-locked-cuda-xgb-current608-001 — 2026-11-02T14:55:59.899562+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/compare_locked_cuda_xgb.py`. Descriptive comparison only, no weight tuning or training. Fixed25%GPU-XGB versus current608blend deltas2021/2022:[-2.1256840878614014e-06, 3.442561875688366e-05]. Neither candidate dominates both years. Both arrays/labels/IDs/unit order verified; exact float64 lockedXGB reconstruction passed. The original395branch is accepted against its older declared jointanchor, but transfergate remains unmet. Exact source/inputs/hashes/environment:`artifacts/compare-locked-cuda-xgb-current608-001/manifest.json`. NoKaggle score.

## nonlinear-pq-specialists-2022-001 — 2026-11-02T14:52:53.128454+00:00

Status:complete;year2022;614features,three350-tree specialists,exact40%row samples,six-daypurge,seed4173,two threads,noearlystopping. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train_nonlinear_pq_specialists.py --year 2022 --run-id nonlinear-pq-specialists-2022-001 --selection-lock artifacts/nonlinear-pq-specialists-2021-001/selection_2021.json --selection-lock-sha256 2853a50245e85e2fbf615f85b5ba93736018b6dd9c1cf6420411d0aa24b30a43`. Frozen global weight0.5;selectedAUC0.993683153449926,anchor0.9936914879536517,matchedcontrol0.9936498343743312. Beatsboth:False. Runtime278.091s. Fullrawdata/source/cache/model/environment/split/sample/protocol hashes and exactcheckpointreplay:`artifacts/nonlinear-pq-specialists-2022-001/manifest.json`. NoKaggle submission;reuseddevelopmentfold.

## verify-nonlinear-pq-selection-2022-001 — 2026-11-02T14:58:07.007087+00:00

Status:passed. Independentsavedartifact/protocol/source/data/cache/sample/chronologicalID/checkpoint-hash verification; nofit orfeaturematrix. Recomputed fullfixedsplicegrid and sameweightmatched608control; exactselectedarraybytes verified. SelectedAUC0.993683153449926,anchor0.9936914879536517,control0.9936498343743312,eligibleFalse. Proof/source/command:`artifacts/verify-nonlinear-pq-selection-2022-001/manifest.json`. Runtime0.647s. Unitcounterexamples retained; noKaggle submission orscore.

## cuda-xgb-final-package-001 — 2026-11-02T15:06:40.468964+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/build_cuda_xgb_final_package.py --run-id cuda-xgb-final-package-001`. Prepare accepted original395 GPU full fit and exact nested local export only; no model fit, upload or submission. Exact baseline full sample SHA `4e38d45e87950d4d84849224ddb64fa9b5bf7b21d20a7d9b998223ce5e0fa83d`, first seed4173 40% draw over all training cases. Both purged development folds accepted at frozen25% GPU weight. All anchor source checkpoints/raw arrays and replayed arrays rechecked; projected anchor reconstructed bitwise. Code/protocol ZIP readback checks and all provenance: `artifacts/cuda-xgb-final-package-001/manifest.json`. Main repository transfer gate remains unmet.


## cuda-xgb608-hosted-observation-001

2026-11-02T15:08:04.056136+00:00: Root uploaded reviewed 51,053,983-byte private608 dataset archive (SHA 480cacd6a83aef078104fd35ddfeb5c0569f0dd7d84158f63f82a1511d766a9a). Dataset https://www.kaggle.com/datasets/ravinav/tdt4173-private-xgboost608-experiment-inputs visibly Private, attached to existing private T4x2 notebook and ready to use. Explicit synthetic smoke001 launched; no real608 fold launched yet. Code/protocol/environment and exact commands in docs/cuda_xgb608_private_upload.md; observation JSON preserved. Original395 folds are complete; no heavy local job active.

## cuda-xgb-final-independent-audit-001 — 2026-11-02T15:09:25.047336+00:00

Status: passed. Independent review of frozen original395 final package001/protocol5033b657…; checked every source/package/ZIP/input/acceptance hash, exact first seed4173 full40% sample, recomputed both selected fold scores and bitwise nested blends, reconstructed saved final projected anchor from checkpoint-verified raw source arrays, and ran four targeted sample/projection/submission tests. No model fit, feature construction, upload or submission. Full source, commands, environment, hashes and findings: `artifacts/cuda-xgb-final-independent-audit-001/manifest.json`. Reused development validation only; transfer gate remains unmet.

## cuda-xgb-final-execution-001 — 2026-11-02T15:09:44.355095+00:00

Prepared explicit private-notebook code transfer, final-fit launch and complete-artifact download cells. All three compile; archive/protocol/smoke hashes are pinned. No upload, model fit or Kaggle submission performed. Root owns queue checks and launch after independent package review. Cell hashes and package identity: `artifacts/cuda-xgb-final-execution-001/manifest.json`.

## cuda-xgb-final-dataset-preflight-failed-001 — 2026-11-02T15:13:39.516908+00:00

Initial dataset builder rejected the independent audit because it expected status `complete`, while the audit correctly records `passed`. Failure happened before any output dataset directory, upload or model work. Failed source and record preserved at `artifacts/cuda-xgb-final-dataset-preflight-failed-001/`. Correct the status schema check without changing audited source artifacts.

## cuda-xgb-final-private-dataset-001 — 2026-11-02T15:13:39.654602+00:00

Prepared separate PRIVATE code-only dataset for the already audited original395 final fit; no upload or training. Original final protocol/package bytes retained, code under disjoint code/ namespace; helper discovers only unique exact protocol/bundle hashes and verifies every package member. ZIP 54183 bytes, SHA `672dee9648558ee328dd0bfa36d9abb3f8e54ca29dbc11bb8b5b98420b7d49ea`. Original608 input dataset untouched. Full sources/readback proof: `artifacts/cuda-xgb-final-private-dataset-001/manifest.json`.

## cuda-xgb-final-dataset-independent-audit-001 — 2026-11-02T15:25:06.175170+00:00

Passed independent derived private dataset audit. ZIP SHA `672dee9648558ee328dd0bfa36d9abb3f8e54ca29dbc11bb8b5b98420b7d49ea`, launcher SHA `f27e6ac809d16b54ccc1275335489cf17bfed73bee2048f96707847a9b7b2170`. All inherited final package bytes unchanged; exact-hash unique discovery, smoke pin and offline uv explicit final command reviewed. Two discovery fixtures passed; proof and command: `artifacts/cuda-xgb-final-dataset-independent-audit-001/manifest.json`. No upload, training or submission by this audit.

## cuda-xgb608-select-2021-001 — 2026-11-02T15:25:44.870192+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python /Users/heap/programming/tdt4173/artifacts/verify-cuda-xgb608-2021-return-001/frozen_source/scripts/select_cuda_xgb608_blend.py --year 2021 --gpu-run /Users/heap/programming/tdt4173/artifacts/verify-cuda-xgb608-2021-return-001/download/cuda-xgb608-2021-001 --protocol /Users/heap/programming/tdt4173/docs/cuda_xgb608_protocol.json --run-id cuda-xgb608-select-2021-001`. Local GPU-XGB selection/evaluation only, no fit. Fixed global weights0/.1/.25/.5 on2021, immutable lock before separately authorized2022 fit, strict both-year improvement against frozen context-joint608 anchor; no leaderboard retuning. Selected0.5, microAUC0.9950359172801482, anchor0.9949377186691472. Full hashes/schema/checkpoint-replay provenance:`artifacts/cuda-xgb608-select-2021-001/manifest.json`. No Kaggle submission.

## verify-cuda-xgb608-2021-return-001 — 2026-11-02T15:25:44.493465+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python artifacts/cuda-xgb608-return-preflight-001/verify_cuda_xgb608_return.py --workspace /Users/heap/programming/tdt4173 --archive artifacts/cuda-xgb608-2021-001-download.zip --archive-sha256 e204180dedaea34b81a1a6665c3bee4083301551b745dfb0066ab7919c935dde --manifest-sha256 61f1526cb1ace06e819467e66394ddb2abbfde8462865f5bbfa14b08b479011f --smoke-manifest-sha256 56baf65e855f47c0c57aca4b14c0cbd774fffff59b4ee9ecb1748b720880d2e8 --year 2021 --gpu-run-name cuda-xgb608-2021-001 --run-id verify-cuda-xgb608-2021-return-001 --selection-run-id cuda-xgb608-select-2021-001`. Strict returned GPU-XGB archive/manifest/smoke/checkpoint/config/sample/source/OOF verifier, then exact protocol-pinned selector; no model fit. Frozen current608 context-joint blend anchor. Scores:{'0.0': 0.9949377186691472, '0.1': 0.994984845100365, '0.25': 0.9950250716671081, '0.5': 0.9950359172801482}; selected:0.5. Full source snapshots, hashes, environment and verification: `artifacts/verify-cuda-xgb608-2021-return-001/manifest.json`. No Kaggle submission.

## latent-shadow-reachability-001 — 2026-11-02T15:25:51.709743+00:00

Status: complete. Read-only finite-grid water-shadow reachability diagnostic:36 metadata-selected pre-2021 cases,3 units,17 additive displacements; no fitted model, LP solve, held-out labels or Kaggle action. Exact command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/diagnose_latent_shadow_reachability.py --run-id latent-shadow-reachability-001`. Declared gates passed: True. Complete data/source/cache/IDs/formulas/control/oracle evidence: `artifacts/latent-shadow-reachability-001/manifest.json`. Runtime 2.196s, peak RSS 183369728 bytes. Label-informed diagnostic scores are not deployable validation or a bound over all correction families.


## cuda-xgb-final-hosted-observation-001

2026-11-02T15:31:56.673652+00:00: Root created and verified PRIVATE final-code dataset https://www.kaggle.com/datasets/ravinav/tdt4173-private-xgboost-final-code (ZIP SHA672dee9648558ee328dd0bfa36d9abb3f8e54ca29dbc11bb8b5b98420b7d49ea), attached to the existing private notebook, and launched independently approved original395 full fit PID291 using exact-hash launcher f27e6ac809d16b54ccc1275335489cf17bfed73bee2048f96707847a9b7b2170 and original successful smoke. Current CUDA1 queue serial;6082021 wascomplete beforelaunch. Exact prepared command in artifacts/cuda-xgb-final-dataset-execution-001/launch_cell.py, notebook live variable xgb_final_process. No export/submission yet.

## Pair-calibration research and latent-shadow preflight — 2026-11-02T15:38:58.574948+00:00

Read-only pair diagnostic `artifacts/pair-calibration-diagnostic-001/manifest.json` completed in1.113s; exact14x14 error counts and unit AUC identities passed. Cross-unit erroneous-pair shares90.25%/91.23% accompany cross-unit pair shares93.37%/93.30%; no calibration fit. Synthetic preflight001 passed3groups; preflight002 additionally ran the unchanged nonlinear-PQ physical tests after boundary-error reporting was added,8tests passed in.781s including imports. Exact source snapshots/commands/environment are in `artifacts/latent-shadow-preflight-002/manifest.json`. The36-case real reachability run is separately logged by its runner. Pre-result prose is preserved in `artifacts/latent-shadow-research-summary-001/` before docs/next_model_hypotheses.md is updated; original manifests unchanged. No new predictions submitted and no remote transfer.

## submission-cuda-xgb-joint-blend-001 — 2026-11-02T15:39:46.325356+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python artifacts/cuda-xgb-final-package-001/source/scripts/export_cuda_xgb_final.py --protocol docs/cuda_xgb_final_protocol.json --protocol-sha256 5033b6577b0142c8d6f38b290ac8119c45b65f0eb8b59a2d1b31b69debbf7a11 --archive artifacts/cuda-xgb-final-001-download.zip --archive-sha256 fd7f30ed90e28008ec58e8c1e0a1f73f9b81965a445f42e7f07fd0bb72c81c5b --manifest-sha256 4658da3585df5c125a8bd3749f1a5d22084966f0c3dc433a4491669f6fa126a4 --gpu-run-name cuda-xgb-final-001 --run-id submission-cuda-xgb-joint-blend-001`. Strict finalGPU artifact/sample/config/checkpoint/fulltestreplay verification and exact nested75%projectedjointanchor/25%GPU blend export. No training or weightretuning. SubmissionSHA:895b957f2c2fb19d1228b4cc6b960a24477ff977a06d2d790e68dd5b3d50cf92. Full provenance:`artifacts/submission-cuda-xgb-joint-blend-001/manifest.json`. NoKaggle result at export.


## cuda-xgb608-hosted-observation-002

2026-11-02T15:40:22.808364+00:00: Downloaded2021 archive SHAe204180dedaea34b81a1a6665c3bee4083301551b745dfb0066ab7919c935dde and verified exact local bytes. Independent frozen selection chose.5 with AUC.9950359172801482 against current608anchor. Root transferred lock through canonicalJSON literal+range2193..2557; remoteSHA5ab61a227270502d0f3a1387ea856a4cd0c10e2b50fec6b3864bad12629ee11f matchedexact. Explicit2022 command launchedPID310 only after original395finalfit exited0. Frozen helper8297e491…,same original608smoke, no2022 retuning. Complete commands/source/data provenance in originalprotocol and artifacts/cuda-xgb608-2022-execution-001; observationJSON addsactualPID/launchrecord.

## verify-cuda-xgb-final-export-001 — 2026-11-02T15:41:05.597558+00:00

Independent final395 GPU export PASSED. Exact75% saved projected joint anchor plus25% rawGPU, projected again, reproduces float64 export bytes and CSV SHA `895b957f2c2fb19d1228b4cc6b960a24477ff977a06d2d790e68dd5b3d50cf92`. All export artifact hashes, downloaded hosted manifest/archive pins and saved hosted full-test replay arrays rechecked; 725 rows,2353 columns, exact IDs, `[0,1]` probabilities, maximum CSV readback error 4.999994862586732e-10. No local model inference, fitting or Kaggle submission. Reused validation: {'2021': 0.9949355929850593, '2022': 0.9937259135724086}. Command: `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python artifacts/verify-cuda-xgb-final-export-001/verify.py`. Proof, source and environment: `artifacts/verify-cuda-xgb-final-export-001/manifest.json`.


## sintef-primary-research-001: primary SINTEF research

2026-11-02T15:41:48.320751+00:00. Read the official SHOP docs, public formulation report and paper abstracts; compared against supplied v5_2026_6/SHOP17.17 provenance and current feature code. Preserved 17-page report SHA `6eb3129ca86449809a5a049062cd1f24ce2ba4afae6649b62add688f2718ca0b` and text/rendered equation checks. Newer public generation source uses SHOP18.4i, so its NoMIP/linear-startup/.5/full-stage extraction settings are hypotheses, not established competition commands. `docs/sintef_web_research.md` maps missing continuous linked-state switching, river marginal costs and tunnels to actual code and rejected experiments. No training, label decoding, external-label import, probability change, validation score or Kaggle action. Exact URLs/file hashes/environment and access limits: `artifacts/sintef-primary-research-001/manifest.json`. Root authorized preparation only of a new fixed-head H1+H2 diagnostic; execution awaits review.

## source-dataset-audit-001 — 2026-11-02T15:44:55.926967+00:00

Completed primary-source generation/topology/water-value research. Official GitLab project20556 exactJune v5 commit `5fca23ebb1fc57b5a29c6f650be0efd21e74713c` and observed main `11b8d7c6e30c81c878e8a80540721d817b47a0c7` fetched read-only via public API; saved `26` source/metadata responses with URL/SHA256. No external output labels/raw data, code execution, feature construction or model fitting. Confirmed water-value timing/mapping and distinction between newerSHOP18.4i full-stage labels and competitionSHOP17.17; exact competition command sequence remains unknown. Existing directional-storage and nonlinearPQ failures consulted to avoid repeats. Report: `docs/source_dataset_audit.md`; provenance: `artifacts/source-dataset-audit-001/manifest.json`. No validation/leaderboard claim.


## submission-cuda-xgb-joint-blend-001 — actual Kaggle observation

2026-11-02T15:46:51.840329+00:00: User reported and root independently UIverified completed publicscore **0.99626** for exactSHA895b957f2c2fb19d1228b4cc6b960a24477ff977a06d2d790e68dd5b3d50cf92. This is0.00005 belowcurrentbest0.99631 and NOTpromoted. No new rankverified/no finalselectionchange. Frozenvalidationgains vsolderjointanchor did nottranslate into an improvement vsbestpubliccandidate. Immutable exportmanifest untouched; separateobservationJSON and docs/kaggle_scores.json updated. Reproduciblefullfit/config/commands/source/environment/checkpoints recordedin docs/cuda_xgb_final.md and exportmanifest. Failedpubliccandidate retained.


## lp-shop-relaxation-preflight-001/002: fixed-head continuous-state diagnostic preparation

2026-11-02T15:52:02.031990+00:00. Root authorized preparation only. New linked state satisfies both power and flow bounds, with joint fractional startup/shutdown costs and free boundaries. Four arms preserve cached baseline and explicitly isolate its float32 RHS using a fresh common-float64 base before zero/supplied-cost states. Seven synthetic tests passed; metadata-only36case IDs match the independent pre2021 date rule. Initial packaging failed because the repository uses a namespace package without __init__.py; tests/source/error preserved under001, corrected packaging002 reuses the unchanged passing test log. No real LP solve, target decode, model fit or Kaggle action. Protocol SHA `583989ba708c304144c5e077e26e46ae772fd846af33fc66a6c5b736c3084734`; exact immutable source/data/environment/test log and launch command in `artifacts/lp-shop-relaxation-preflight-002/manifest.json`. Independent audit required before launch.

## cuda-xgb608-select-2022-001 — 2026-11-02T15:52:04.046439+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python /Users/heap/programming/tdt4173/artifacts/verify-cuda-xgb608-2022-return-001/frozen_source/scripts/select_cuda_xgb608_blend.py --year 2022 --gpu-run /Users/heap/programming/tdt4173/artifacts/verify-cuda-xgb608-2022-return-001/download/cuda-xgb608-2022-001 --protocol /Users/heap/programming/tdt4173/docs/cuda_xgb608_protocol.json --run-id cuda-xgb608-select-2022-001 --selection-lock artifacts/cuda-xgb608-select-2021-001/selection_2021.json --selection-lock-sha256 5ab61a227270502d0f3a1387ea856a4cd0c10e2b50fec6b3864bad12629ee11f`. Local GPU-XGB selection/evaluation only, no fit. Fixed global weights0/.1/.25/.5 on2021, immutable lock before separately authorized2022 fit, strict both-year improvement against frozen context-joint608 anchor; no leaderboard retuning. Selected0.5, microAUC0.9937529477072502, anchor0.9936914879536517. Full hashes/schema/checkpoint-replay provenance:`artifacts/cuda-xgb608-select-2022-001/manifest.json`. No Kaggle submission.

## verify-cuda-xgb608-2022-return-001 — 2026-11-02T15:52:03.680339+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python artifacts/cuda-xgb608-return-preflight-001/verify_cuda_xgb608_return.py --workspace /Users/heap/programming/tdt4173 --archive artifacts/cuda-xgb608-2022-001-download.zip --archive-sha256 d98f71bf6a258c23b5bd4841f00243531bf9383828a302c7ce59755aab8aec49 --manifest-sha256 affc4722227ed1e71d8dee74919ab28eba5f204c75b78b4339ef94e8580391e0 --smoke-manifest-sha256 56baf65e855f47c0c57aca4b14c0cbd774fffff59b4ee9ecb1748b720880d2e8 --year 2022 --gpu-run-name cuda-xgb608-2022-001 --run-id verify-cuda-xgb608-2022-return-001 --selection-run-id cuda-xgb608-select-2022-001 --selection-lock artifacts/cuda-xgb608-select-2021-001/selection_2021.json --selection-lock-sha256 5ab61a227270502d0f3a1387ea856a4cd0c10e2b50fec6b3864bad12629ee11f`. Strict returned GPU-XGB archive/manifest/smoke/checkpoint/config/sample/source/OOF verifier, then exact protocol-pinned selector; no model fit. Frozen current608 context-joint blend anchor. Scores:{'0.0': 0.9936914879536517, '0.5': 0.9937529477072502}; selected:0.5. Full source snapshots, hashes, environment and verification: `artifacts/verify-cuda-xgb608-2022-return-001/manifest.json`. No Kaggle submission.

## latent-shadow-learning-preflight-001 — 2026-11-02T15:50:17.124363+00:00

Passed15 fixture groups in1.300s test runtime,2.061s including imports. New7 checks cover selected-label poisoning, future-storage and label isolation, disconnected inverse-target ambiguity, analytic gradients, synthetic learnability against constant/permuted controls, case isolation and six-daypurge. Existing8 nonlinear-PQ/DP checks also passed. Synthetic parameter fitting only; no real target expansion or learned case mapping. Exact command/environment/source snapshots and hashes: `artifacts/latent-shadow-learning-preflight-001/manifest.json`. Reviewed draft `docs/latent_shadow_learnability_protocol.md`; root approved concept/splits/gates, independent source review pending. No Kaggle action or whole14-unit AUC claim.

## lp-shop-relaxation-independent-audit-001 — 2026-11-02T15:52:59.126236+00:00

Passed independent frozen package002/protocol `583989ba708c304144c5e077e26e46ae772fd846af33fc66a6c5b736c3084734` review. Four-arm numerical control, correct continuous-state power/discharge links and switching objective, metadata-only36case selection, hashes and before-label replay sequencing checked. Seven synthetic fixtures passed; no real solve or target decode. Approved bounded diagnostic only, not full cache/model expansion. Commands/sources/proof: `artifacts/lp-shop-relaxation-independent-audit-001/manifest.json`.

## cuda-xgb608-final-package-001 — 2026-11-02T15:54:48.260992+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/build_cuda_xgb608_final_package.py --run-id cuda-xgb608-final-package-001`. Prepare accepted exact608 GPU full fit and exact nested local export only; no model fit, upload or submission. Exact baseline full sample SHA `4e38d45e87950d4d84849224ddb64fa9b5bf7b21d20a7d9b998223ce5e0fa83d`, first seed4173 40% draw over all training cases. Both purged development folds accepted at frozen50% GPU weight. All anchor source checkpoints/raw arrays and replayed arrays rechecked; projected anchor reconstructed bitwise. Code/protocol ZIP readback checks and all provenance: `artifacts/cuda-xgb608-final-package-001/manifest.json`. Main repository transfer gate remains unmet.

## cuda-xgb608-final-private-dataset-001 — 2026-11-02T15:55:36.946461+00:00

Prepared separate PRIVATE code-only dataset for the prepared original608 final fit; no upload or training. Original final protocol/package bytes retained, code under disjoint code/ namespace; helper discovers only unique exact protocol/bundle hashes and verifies every package member. ZIP 73548 bytes, SHA `56ae4c302ad93fd6b312112e831e78d4b37e157c1d89ddcf0c51b863044aa4b8`. Original608 input dataset untouched. Full sources/readback proof: `artifacts/cuda-xgb608-final-private-dataset-001/manifest.json`.


## latent-shadow-learnability-independent-audit-001: independent latent-shadow learnability review

2026-11-02T15:55:58.618964+00:00. Passed exact protocol `866a2bcc7ac6ff1a9e51633bef4772716cac31c19062dfb5b9d573ce83901d39` with23source pins and prior16tests. Reviewed purged scope, eight input-only features,17fixed physical paths, inverse-target ambiguity, convex map, constant/permuted controls, training-before-screen lock and explicit3-target score scope. Added prefreeze review requirements externally pin LP manifest and bind exact raw metadata/cache target reconstruction; owner implemented and tested. No new target generation or model fit by reviewer. Proof `artifacts/latent-shadow-learnability-independent-audit-001/manifest.json`; root owns launch.

## latent-shadow learnability freeze — 2026-11-02T15:56:50.913392+00:00

Root approved unchanged early-year protocol; independent tcn_audit review corrected external LP-manifest pinning and exact raw-metadata partition checks before freeze. Final preflight `artifacts/latent-shadow-learning-preflight-002/manifest.json` passed16tests in1.283s (1.882s wall), with all source snapshots preserved. Frozen JSON `docs/latent_shadow_learnability_protocol.json` SHA866a2bcc7ac6ff1a9e51633bef4772716cac31c19062dfb5b9d573ce83901d39 covers23source files, raw inputs, original cache and reachability evidence. Independent closure proof `artifacts/latent-shadow-learnability-independent-audit-001/manifest.json` SHAccd7e8ea51d3c072e87416f986750dd017a3c546ae92e45bdad2ef8086b4fca1. Authorized sequence:1455cases2015–2018targetgeneration→real/constant/permutedmaps and hashedlock→725cases2019/2020screen. No2021/2022target parsing or whole14-unit production-AUC claim. Local slot1 assigned. Exact execution commands and measured resources are recorded separately by each phase runner.

## latent-shadow-targets-train-001 — 2026-11-02T15:56:15.429482+00:00

Status complete; latent-shadow train target generation only, no fitted model. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/build_latent_shadow_targets.py --partition train --run-id latent-shadow-targets-train-001 --protocol-sha256 866a2bcc7ac6ff1a9e51633bef4772716cac31c19062dfb5b9d573ce83901d39`. Fixed17-grid/threeunits/eightfeatures; exact legal inputs, source/cache/sample/IDs and all per-case rewards/paths/soft-target hashes in `artifacts/latent-shadow-targets-train-001/manifest.json`. Max target year 2018. Runtime 65.670s, peak RSS 259260416bytes. Kaggle and whole14-unit validation absent.

## latent-shadow-map-fit-001 — 2026-11-02T15:57:41.064552+00:00

Status complete; latent-shadow fit. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/fit_latent_shadow.py --mode fit --run-id latent-shadow-map-fit-001 --train-cache artifacts/latent-shadow-targets-train-001 --protocol-sha256 866a2bcc7ac6ff1a9e51633bef4772716cac31c19062dfb5b9d573ce83901d39`. Three17-state convex linear maps, eight features, fixedsofttargets/prior/ridge and chronological2015–2018training/2019–2020screen. Source/data/cache/splits/coefficients/normalization/permutation/control hashes in `artifacts/latent-shadow-map-fit-001/manifest.json`. Screen gate None; scores cover three targets only, no whole14-unit production-anchor claim. Runtime 1.415s; peakRSS 350027776bytes. No Kaggle action.


## cuda-xgb608-final-independent-audit-001 and dataset audit

2026-11-02T15:58:06.587722+00:00. Independent audit passed exact608final protocol `a5939edb8ec7f46dbdb4c6edd0c06909ea09ed0ac029709b06945a5c6434683b` and separate PRIVATE dataset ZIP `56ae4c302ad93fd6b312112e831e78d4b37e157c1d89ddcf0c51b863044aa4b8`. Diff reviewed from approved395 branch: exact608 prefix/supplement/core loader, accepted50% nested projected anchor, complete fullsample and14saved checkpoint replays. Both frozen fold scores recomputed ({'2021': 0.9950359172801482, '2022': 0.9937529477072502}); both selected arrays and4source final anchor reconstructed bitwise. All source/package/ZIP/input/acceptance hashes verified; seven targeted tests passed. No fit/upload/submission by reviewer. Proof `artifacts/cuda-xgb608-final-independent-audit-001/manifest.json` and `artifacts/cuda-xgb608-final-dataset-independent-audit-001/manifest.json`; original artifacts remain immutable.

## latent-shadow-targets-screen-001 — 2026-11-02T15:58:21.810224+00:00

Status complete; latent-shadow screen target generation only, no fitted model. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/build_latent_shadow_targets.py --partition screen --run-id latent-shadow-targets-screen-001 --protocol-sha256 866a2bcc7ac6ff1a9e51633bef4772716cac31c19062dfb5b9d573ce83901d39 --training-lock artifacts/latent-shadow-map-fit-001/training_lock.json --training-lock-sha256 bca9f556f79e266e4bdac2673252edf529e425fb5f83bd822645913b7d4c4f12`. Fixed17-grid/threeunits/eightfeatures; exact legal inputs, source/cache/sample/IDs and all per-case rewards/paths/soft-target hashes in `artifacts/latent-shadow-targets-screen-001/manifest.json`. Max target year 2020. Runtime 33.417s, peak RSS 249266176bytes. Kaggle and whole14-unit validation absent.

## latent-shadow-map-screen-001 — 2026-11-02T15:59:32.628633+00:00

Status complete; latent-shadow evaluate. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/fit_latent_shadow.py --mode evaluate --run-id latent-shadow-map-screen-001 --train-cache artifacts/latent-shadow-targets-train-001 --screen-cache artifacts/latent-shadow-targets-screen-001 --protocol-sha256 866a2bcc7ac6ff1a9e51633bef4772716cac31c19062dfb5b9d573ce83901d39 --training-lock artifacts/latent-shadow-map-fit-001/training_lock.json --training-lock-sha256 bca9f556f79e266e4bdac2673252edf529e425fb5f83bd822645913b7d4c4f12`. Three17-state convex linear maps, eight features, fixedsofttargets/prior/ridge and chronological2015–2018training/2019–2020screen. Source/data/cache/splits/coefficients/normalization/permutation/control hashes in `artifacts/latent-shadow-map-screen-001/manifest.json`. Screen gate True; scores cover three targets only, no whole14-unit production-anchor claim. Runtime 0.999s; peakRSS 292651008bytes. No Kaggle action.


## latent-shadow-screen-independent-audit-001

2026-11-02T16:01:32.052286+00:00. Independent frozen latent-shadow screen verification passed: exact cache/source/data/lock scope, training-only normalizers, coefficients→distributions→marginalprobabilities bitwise replay and all yearlymetrics/gates recomputed.1455purgedtraining,725screen; lock before screenlabels. Only3target physical-family reuseddevelopment evidence; threshold gains concentrated in Lio, Haukeli/Hogga unchanged. No model fit by reviewer and no2021/2022orleaderboard claim. Proof `artifacts/latent-shadow-screen-independent-audit-001/manifest.json`.

## verify-lp-shop-relaxation-diagnostic-001 — independent saved-array review

- Timestamp: 2026-11-02T16:02:55.511686+00:00; complete in 2.34s, no solver runs or model fits, no seed, no Kaggle submission. Command: `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python artifacts/verify-lp-shop-relaxation-diagnostic-001/verify.py`. Python/macOS/package versions and exact reviewer source hash are in the proof manifest.
- Verified all 225 diagnostic artifacts, frozen protocol `583989ba…`, source/data/cache pins, all36 metadata-selected pre2021 inputs and original labels, four-arm proxies, all unit/block metrics, saved supplied-cost replays, objective/cost formulas and declared gates. Diagnostic manifest SHA `95ca40b0ff3b29f6cb36bb55b5539dcdcd3839fd05ca4f9a222d447a7821800e`.
- Q-proxy AUC cached/fresh/zero/supplied: .925327975264/.925327976396/.925744293690/.944652174548. All declared screens pass. All four slack types exactly zero for every arm/case. Independent float64 aggregate residual fresh/zero/supplied <=1.137e-11 Mm3; the fresh base's stored1.283e-4 diagnostic is a float32 reduction artifact. Historical cached RHS rounding remains preserved.
- Caveats: on-fraction MAE worsens .0802154→.0877504, Tokke_G3 AUC decreases, six units and the2015–18 block have more cutoff errors. Descriptive36-case results with year-end overlap are not chronological model acceptance. Separate fullcache/learning protocol review remains required.
- Proof `artifacts/verify-lp-shop-relaxation-diagnostic-001/manifest.json` SHA `e73ec5f0bb587451670437dcb1d2b49170b88acb3d72088098a29cfaef9d7bdd`; details `docs/lp_shop_relaxation_result_audit.md`.


## lp-shop-relaxation-diagnostic-001:36-case input-only solve then pre2021 proxy screen

Completed at 2026-11-02T16:12:52.758250+00:00, exact frozen protocol583989ba708c304144c5e077e26e46ae772fd846af33fc66a6c5b736c3084734. Runtime24.614s, peakRSS536739840bytes, oneHiGHS thread, no fitted learner. ManifestSHA `95ca40b0ff3b29f6cb36bb55b5539dcdcd3839fd05ca4f9a222d447a7821800e`. All36fresh supplied-cost replays bitwise; labels decoded only after input-only controls. All four arms/rawarrays/replays/inputcases preserved. Q-proxy AUC cached/base64/zero/supplied .925327975264/.925327976396/.925744293690/.944652174548; weak3 .881962261526→.905475628876. All prescribed proxy gates passed, not model/leaderboard acceptance. Independent proof `artifacts/verify-lp-shop-relaxation-diagnostic-001/manifest.json`:225artifact checks, allsoftslacks zero, true float64 aggregate residual≤1.137e-11Mm3. Stored base balance diagnostic up1.283e-4 is float32 initial.sum artifact, preserved unmodified. Onfraction MAE worsened .0802154→.0877504; TokkeG3 AUC and six units' cutoff errors worsened;2015–18 cutoff errors+22. Full per-unit/precision/negative outcomes in `docs/lp_shop_relaxation_result_audit.md`. No full cache or training automatically launched.


## cuda-xgb608-2022-001 — completed hosted return

2026-11-02T16:13:20.241929+00:00: Raw2022 AUC.9935514201255842,291.8813548088074s. Download archiveSHA d98f71bf6a258c23b5bd4841f00243531bf9383828a302c7ce59755aab8aec49,10862783bytes; manifestSHAaffc4722227ed1e71d8dee74919ab28eba5f204c75b78b4339ef94e8580391e0; smoke56baf65e855f47c0c57aca4b14c0cbd774fffff59b4ee9ecb1748b720880d2e8. Exact frozenw.5 accepted2022 .9937529477072502 vsanchor.9936914879536517. No2022tuning; independentproof verify-cuda-xgb608-2022-return-001. Final608package independentlyapprovedbutunlaunched becausebrowsertransportclosed. NotebookQuickSavedprivatelyversion2withoutRunAll; inputcodeandoutputsretained.


## lp-shop-full-cache-preflight-001: compact label-free cache preparation

2026-11-02T16:17:42.912047+00:00. Prepared exact supplied-cost solver expansion only: all3647inputcases,36auditedcases copied,3611newcases planned with fresh bitwise replay. Frozen56dimensionless columns preserve exact608prefix; no target decoding or models. Three synthetic tests passed(.089s); all36saved physical outputs/replays produce identical compact features without re-solving or labels. Metadata guard observed onlyRunNo/starttime. ProtocolSHA `6501db4dd0c2f292f5373310b44e1133e58ff77578c8a031ada1c1c64ac38325`, source/data/36feature replay and command in `artifacts/lp-shop-full-cache-preflight-001/manifest.json`. Estimate20–35minutes1thread/1localslot; not yet executed. Root and independent review required before fullcache launch.


## regression-post-research-001

2026-11-02T16:29:25.840248+00:00: Full unittest discovery passed183tests in53.286s (55.124s including uv/startup). Exact command/environment path and source/log SHA256 are preserved in artifacts/regression-post-research-001/manifest.json; test files remained unchanged during execution. Covers the new CUDA final packaging, latent-shadow learning/splice helper and relaxed-state LP/cache tests alongside the existing suite. Existing pandas copy deprecation warnings remain nonfatal. No model fit, submission, or probability change in this check.

## latent-shadow-splice-2021-001 — 2026-11-02T17:12:28.386249+00:00

Status complete; strictly2021-only direct three-unit latent-map probability splice. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/latent_shadow_splice_2021.py --run-id latent-shadow-splice-2021-001 --protocol-sha256 ca249685a4cf56d635f3a253c91276b8cf0e7744d94f1cf4b95eef40fb2ef2ef`. Reuses2180oldtargets and generates6gapcases; fixed maps fit through2020-12-25, no same-rowfeaturestacking. Whole14-unit weights [0.0, 0.025, 0.05, 0.1, 0.25]; selected 0.05, AUC 0.9950284053858753, same-weightcontrols {'constant': 0.9950173633431421, 'permuted': 0.9950167285562123}; eligiblefor2022 True. Source/data/cache/sample/fit-lock/prediction/selection hashes: `artifacts/latent-shadow-splice-2021-001/manifest.json`. Runtime 26.064s, peakRSS 623591424bytes. No2022evaluation orKaggleaction.


## evening-continuation-audits-001 — 2026-11-02T17:23Z

Authenticated browser confirmed publicbest0.99631/rank23, first0.99820/third0.99805. Private608finalcode dataset uploaded and attached; exact archive56ae4c302ad93fd6b312112e831e78d4b37e157c1d89ddcf0c51b863044aa4b8. Original runtime had stopped. Genuine recovery smoke passed at17:19:28 (newmanifest2d42f7eacb0d88e340cdf75a0d1dc54b20851106405250c46c43aaf61154f49c, exact CUDA/package/checkpoint checks) but Kaggle stopped that runtime too and discarded workingfiles before final launch. Final001 PID110 failed with missing historical smoke before fitting; no score or model artifact claimed. Preparing recovery+foreground final together under newIDs, same frozen protocol. Docs/continuation_20261102_evening.md records exact dataset and consultation.

Independent audits passed: artifacts/cuda-xgb608-final-continuation-audit-001/manifest.json (.415s), artifacts/latent-shadow-splice-2021-result-audit-001/manifest.json (3.071s), artifacts/latent-shadow-splice-2022-result-audit-001/manifest.json (2.949s). Exact commands/source/environment/hashes are in those manifests. No fitting by reviewers; latent2022 rejected despite small anchor gain because permutation control wins. No new Kaggle submission yet. Goal remains unmet.


## Continuation independent audits — detailed provenance supplement

Recorded 2026-11-02T17:24:27.853945+00:00. No reviewer fit, label-derived model tuning, GPU action or Kaggle submission occurred in these audits; artifact manifests preserve exact source/input/checkpoint hashes and environments. All chronological scores remain reused-development evidence.

- `cuda-xgb608-final-continuation-audit-001`: passed in0.415s. Exact command `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python artifacts/cuda-xgb608-final-continuation-audit-001/audit.py`. Validated frozen sources, both private ZIPs, payloads, local acceptance evidence and exact121800×14 anchor reconstruction from all4 checkpoints/replays. ManifestSHA `4cd65baede856aff6acd8cb3e69459aff7bf7a8d6bed6deee50f7a7ba02f3e84`; command/hardware supplement `execution_record.json`. Candidate remains the locked50% GPU608 blend .9950359172801482/.9937529477072502; no final608 score yet.
- `latent-shadow-splice-2021-result-audit-001`: passed in3.071s. Exact command `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python artifacts/latent-shadow-splice-2021-result-audit-001/audit.py`. Verified all pinned source/data/array bytes,2186traincases through2020-12-25 and3652021cases, six-daypurge, fit/prediction seals, train-only normalizers, within-year-season permutation seed4173, model objective/gradient, and coefficient→distribution→schedule replay bitwise. Recomputed all5 declaredweights; .05 selected .9950284053858753 vsanchor .9949377186691472, constant .9950173633431421/permuted .9950167285562123. Selectionlock `7cea46df8d37773ed568dca0b588cec1f20bbb6a06f6d42a6719924f68c67232`; auditSHA `af1dba1873e7bad6635497364a64f17d8dffc6396fe2121815bc6a44a5dc7d7e`. Eligible only for locked2022 confirmation.
- `latent-shadow-splice-2022-result-audit-001`: passed verification, candidate REJECTED. Exact command `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python artifacts/latent-shadow-splice-2022-result-audit-001/audit.py`;2.949s. Verified2551trainingcases through2021-12-25,3652022cases, same purge/units/fixed.05, all targets/modelnormalizers/permutation/objective/gradient/replays and source/artifact seals. Real .9937102130345562 beatsanchor .9936914879536517 andconstant .9937034478257145 but trailspermuted .9937230545979968. Frozenstrictcontrols therefore fail; no finalfit/export/submission. Complete hashes/environment/command in its manifest.
- `cuda-xgb664-source-independent-audit-001`: source preflight passed; fullcache/package verification remains pending. Independently reran5 targetedtests (`UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python -m unittest tests.test_cuda_xgb664 -v`),0.010s test runtime. Reviewed immutable608prefix+56inputs, disjoint3bundle staging, fixed600treeGPUparams/exact40%samples,2021argmax with same-weight608control, gated2022, and strict returnedmanifest/smoke/source/checkpoint/nativeconfig/sample verification. Source hashes/logSHA and limitations in manifestSHA `f5b3b827adf03246b23a328064380c71dd9bba5ec11c1c18ed65f9fb8517513b`. No664fit or score yet.

## cuda-xgb608-smoke-recovery-001 and cuda-xgb608-final-001 — session-loss failure

Recorded 2026-11-02T17:24:27.853945+00:00 from root's authenticated Kaggle UI observations. The restarted session required restoring the original historical smoke bytes from the independently verified local608return; the historical timestamp/manifest were not modified. A new genuine synthetic2048×608,8tree/depth3 smoke ran separately as `cuda-xgb608-smoke-recovery-001`. UI reportedpass at2026-11-02T17:19:28Z and manifestSHA `2d42f7eacb0d88e340cdf75a0d1dc54b20851106405250c46c43aaf61154f49c`; runtimeCUDAenvironment, packageversions and syntheticcheckpoint were each exactly equal to historical. Reviewed source/exactcommands/historical artifact hashes: `artifacts/cuda-xgb608-session-recovery-001/`.

The notebook session then ended unexpectedly. Its next boot removed `/kaggle/working` recovery files. Separately launched `cuda-xgb608-final-001`,PID110, exited1 before the final trainer because the historical smoke manifest was absent. Zero models trained and no candidate predictions or submission produced. Transient hosted files were not downloaded; the distinction between UIobserved and locallyverified proof is preserved in `hosted_observation.json`.

Recovery002 uses fresh runIDs `cuda-xgb608-smoke-recovery-002` and `cuda-xgb608-final-002`, combines restore+freshsmoke+foregroundfinalfit+archivecreation in one notebook cell, and preserves the original frozentrainer/protocol/parameters. Source `artifacts/cuda-xgb608-session-recovery-002/recovery_final_download.py`, SHA `aa0935d7af8a34e0c90fc2e3085f0cef7b14aa245874de3792f8136df7c52adc`; exact-source singlelinewrapper SHA `ffa0de997a8b79e9f38db4840a533b60a330a3734c0d653244b05c2aedf157c3`. Root reportedforegroundlaunch around17:24UTC; completion and archive download remain unverified at this entry. No new leaderboard score.


## joint-residual-crf-preflight-independent-audit-001

2026-11-02T17:27:23.524379+00:00. Source/protocol/numericalpreflight passed; no actualmodel fit or targetvalues read by reviewer. Exact3test command and source/data/input pins in `artifacts/joint-residual-crf-preflight-independent-audit-001/manifest.json`; testtime .050s. Verified full8stateCRF enumeration/analyticgradient,zeroidentity/caseisolation, matched12parameterunarycontrol, ninepurgedbasefit design,724genuineOOF residualtrainingcases,360case2020screen,train-onlynormalizers and labelseals. Frozen protocolSHA `aca67e40296d115cbccba113c5c55229bca74a3b57f2968be66be3e0202fdbed`. Approves boundedpre2021screen only under rootexistingheavy-slot authorization. No2021/2022acceptance or leaderboardclaim.


## cuda-xgb608-standalone-predeclaration-001

2026-11-02T17:29:19.878112+00:00. Root requested two distinct public evaluations under the user's frequent-submission instruction: primary frozen50%blend, then standalone608GPU plus previously establishedVinjeprojection. Protocol `docs/cuda_xgb608_standalone_protocol.json` SHA `d808593cff7033d80b4482ede212b82509ba78385a249f446e7fd57cbd915f3d` was frozen before anyfinal608Kagglescore. No weight sweep or new training. Exactexistingfoldarrays/manifest hashes and exporter/projection source pins in protocol. Recomputed projectedstandalone microAUC2021 **.99483023883311**,2022 **.9935545164712399**; raw .9948276701112405/.9935514201255842. These are weaker than the accepted50%blend and are disclosed as an architecture diagnostic, with no automaticbest-result promotion. `scripts/export_cuda_xgb608_standalone.py` requires the completedstrictprimaryGPUreturn/exportmanifest and its exactSHA, rechecks allreturnedartifacts andrawreplay, then exports a separateCSV/manifest using exact725×2353template and[0,1] checks. Preparation artifact `artifacts/cuda-xgb608-standalone-predeclaration-001/manifest.json`. No finalexport orpublicresult yet.


## cuda-xgb608-standalone-independent-audit-001

2026-11-02T17:30:42.957366+00:00. Independent reviewer shadow approved prospective standalone608GPU diagnostic export after strictprimaryexport. Verified allsource/input pins and recomputedprojectedCV .99483023883311/.9935545164712399; checkedprimarymanifest→allartifacthashes→hostedfinalmanifest→float32bitwisereplay→ID/schema/range/CSV closure. Proof `artifacts/cuda-xgb608-standalone-independent-audit-001/manifest.json` SHA `a1b15d43b1ef251235b41c5590ee9e1a0b193aba3cc711e3102bab01abd07ad1`. No newfit/finalexport/publicscore. Diagnostic remainsdevelopment-weaker andnotautomaticallypromoted.

## latent-shadow-splice-2022-preflight-001 — synthetic preparation

2026-11-02. Preserved original2021 source and implemented a separately pinned2022 runner. Five tests passed, including exact365 eligible-label decoding with poisonous2022 values, rejection of wrong/reversed/later IDs and missing selection locks, and existing exactscope/identity/grid checks. One initial synthetic test attempt failed before anylabels because thefixture mixed timestampformats; fixtureonly corrected to uniform UTC format. No realfit orscore in either test attempt. Exact command, environment, source hashes and log: `artifacts/latent-shadow-splice-2022-preflight-001/manifest.json` SHA513591845a1bf5a399e04d51fcfd32c8584e2dfe4cb0c065d9096ce04ebfc07b. Frozen2022 protocol SHA f0c4fcef45565679280e6685b340e3929d6264333fe68957c3ff0757a99b9cae. Independent agentxgb reviewed finalsource andallpins before launch.

## latent-shadow-splice-2022-001 — 2026-11-02T17:16:28.374296+00:00

Status complete; candidate rejected by the frozencontrolgate. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/latent_shadow_splice_2022.py --run-id latent-shadow-splice-2022-001 --protocol-sha256 f0c4fcef45565679280e6685b340e3929d6264333fe68957c3ff0757a99b9cae --selection-audit artifacts/latent-shadow-splice-2021-result-audit-001/manifest.json --selection-audit-sha256 af1dba1873e7bad6635497364a64f17d8dffc6396fe2121815bc6a44a5dc7d7e`. Exacttraining2551starts2015-01-01through2021-12-25, validation365starts2022-01-01through2022-12-31,6daypurge,full168hours. Reused2186targets plus359savedinputgrids and generated6additionalgrids;365newtargets. Same17-state/eight-feature maps,priorstd.25,softtargetscale2hours,ridge.01,L-BFGS-Bmaxiter500/ftol1e-12/gtol1e-8,seed4173,oneBLASthread. Train-only normalizationclip[-5,5], noearlystop. Fixed2021weight.05 onall3units vsfrozenfour-source608anchor, noalternate2022grid. RealAUC .9937102130345562 vsanchor.9936914879536517,constant.9937034478257145,permuted.9937230545979968. The realmapdoesnotbeatpermutedcontrol; nofinalfit/export/submission. Runtime21.203420s,peakRSS592068608bytes. Fullsource/data/case/model/predictionchecksums andenvironment in `artifacts/latent-shadow-splice-2022-001/manifest.json` SHA832e64f15c53d5a7f4d9b52e11ed89e6c726924367029470c863026604cf6c72. Fitlockb39e490bb77743102db360516e4c4591f8371a61fe46f8d54c74fae94230028a precedes2022grids; predictionlock17b6b020e95af8fc2788833bcf4cbe19fb9632de65cb930d5d69939fe983d7ca precedes2022labels. Fullresults `docs/latent_shadow_splice_results.md`. KaggleID/score/rank absent.

## latent-shadow-residual-diagnostic-001 — 2026-11-02T17:18:09.345798+00:00

Read-only descriptive analysis of bothreuseddevelopmentyears; nomodelfit,seed,hyperparametersearch orsubmission. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/analyze_latent_shadow_residuals.py`. Exactpositive-negative pairendpoint allocation checks sumto1-microAUC within1e-13.2022>=5transitionweeks are37.16%hours/66.37%error;within3hours oftrueswitches15.12%hours/50.69%error;episodes<=12h10.60%hours/39.25%error. Constantweeks39.04%hours/8.68%error. Truth-definedgroups arediagnostics only andcannotbeusedasinferencefeatures. Reportbyall14units andallcandidatecontrols preserved in `artifacts/latent-shadow-residual-diagnostic-001/manifest.json` SHA02c595c543872dcf7cdd98cdf10036f53f32fb306add2a06604f347272da2b88;source/inputhashes,environment,per-grouptable recorded. Runtime3.129626s;peakRSS510279680bytes.

Independent 2022 result audit subsequently passed and confirmed the rejection: `artifacts/latent-shadow-splice-2022-result-audit-001/manifest.json`, SHA256 `12a8e215946f0ab1256ccc8e4c71b45ccf954eef67a613d70d1352d0dfcf738c`. No final model or submission was produced.


## regression-evening-001 — 2026-11-02 evening

Full unittest discovery passed193 tests in11.133s using Python3.11.15/macOS27.0.1 arm64 and existing uv environment. Exact command in artifacts/regression-evening-001/record.json. Log SHA25626c91d118fa25fd8ec2ef6a57aeecc062f6dc1086c0d5329e0eff763b2691494. Covers CRF numerical gradient/enumeration/identity and SHOP664 additions as available at invocation; future capacity branch is separate. No modeltraining, targetscore or Kaggle action in this check.

## joint-residual-crf-screen-001 — 2026-11-02T17:25:47.977890+00:00

Completed; fixed earlier-year gate rejected. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/screen_joint_residual_crf.py --run-id joint-residual-crf-screen-001 --protocol-sha256 aca67e40296d115cbccba113c5c55229bca74a3b57f2968be66be3e0202fdbed`. Nine new Vinje-only base models: exact608 features, 350 trees/31 leaves/lr.06/minchild100/L2=1/columnfraction.9/maxbin255, deterministic seed4173,2threads,40%fixed row samples with savedindices, no early stopping. Each2018/19/20 prefix purges six starts beforeits prediction year. Residual training uses724 forward-OOF starts2018-01-01through2019-12-25; screen360 starts2020-01-01through2020-12-25. All nine checkpoint replays exact. Residual36parameter jointstate model and12parameter unarycontrol share fourfixed legalcovariates, training-only mean/std clipping[-5,5], ridge.01,L-BFGS-B maxiter150/ftol1e-10/gtol1e-6, zeroinitialization. Screenmodels andpredictions sealedbefore2020labelaccess. Projected Vinje-pooled AUC identity/unary/joint .9790386678233353/.9787562121876519/.978395767602622; proper sequenceNLL .6752701456549443/.6812207829740786/.5793405834628612. JointNLL improves but AUC andinteriorAUC worsen againstboth controls. No2021/22 continuation, finalfit, submission or full14-unit claim. Runtime464.426809s,peakRSS3642425344bytes. Manifest `artifacts/joint-residual-crf-screen-001/manifest.json` SHA49a17b57f25d0a976a1cbfa06b2b15a036bf5155c0ecac3be9619500b38fdf4a; exactsource/data/cache/samples/OS/Python/packages/checkpoints and probabilities recorded there. Results `docs/joint_residual_crf_results.md`.

## submission-cuda-xgb608-context-joint-blend-001 — 2026-11-02T17:35:56.087135+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python artifacts/cuda-xgb608-final-package-001/source/scripts/export_cuda_xgb608_final.py --protocol docs/cuda_xgb608_final_protocol.json --protocol-sha256 a5939edb8ec7f46dbdb4c6edd0c06909ea09ed0ac029709b06945a5c6434683b --archive artifacts/cuda-xgb608-final-003-download.zip --archive-sha256 83a0d528cfcf3817e35c060f7446532f017ec2a92a0138db920f58fa9ed0bf3b --manifest-sha256 0177350df6a91f1d0d1eb5eb84f10bc27e01bfe6db3b7c8140a3dc6a63a286db --gpu-run-name cuda-xgb608-final-003 --run-id submission-cuda-xgb608-context-joint-blend-001`. Strict finalGPU artifact/sample/config/checkpoint/fulltestreplay verification and exact nested50%projectedcontextjointanchor/50%GPU blend export. No training or weightretuning. SubmissionSHA:729ce3e406560b30ab867ad6e0f448650a304168e56293d3d681c5fd7d638c76. Full provenance:`artifacts/submission-cuda-xgb608-context-joint-blend-001/manifest.json`. NoKaggle result at export.


## joint-residual-crf-result-independent-audit-001

2026-11-02T17:37:12.662781+00:00. Independent saved-array verification passed in2.198s and confirms candidate rejection. Exact command/source/protocol/run hashes and checks: `artifacts/joint-residual-crf-result-independent-audit-001/manifest.json`. Verified all9 base checkpoint/source hash records, exact chronological sampled rows,724OOFtraining/360screen cases, training-only normalization, residual modelobjective/gradient, serializedmodel→screenmarginals bitwise replay, original2020labels, allmetrics and gate. Reviewer didnot repeat9basecheckpoint inference; the original frozen runner performed those replays. Joint propersequenceNLL improvesidentity .675270145655→.579340583463, but projected VinjeAUC falls .979038667823→.978395767603 and trails unary .978756212188. No2021/2022 continuation or submission.


## cuda-xgb608-capacity-builder-syntax-failure-001 and cuda-xgb608-capacity-package-001

2026-11-02T17:37:12.662855+00:00. A malformed quoted dictionary key caused a pre-execution Python SyntaxError in the new package builder; no package, model or remote action started. Failed source and exact command retained in `artifacts/cuda-xgb608-capacity-builder-syntax-failure-001/`. Corrected only that typo and compiled the builder. Three focused tests passed: exact four parameter overrides, native GPU0/tree configuration and strict matched-control lock.

Successful package command: `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/build_cuda_xgb608_capacity_package.py --run-id cuda-xgb608-capacity-package-001`. Separate PRIVATE code-only ZIP is58,445bytes, SHA `6106c89288057f605fb9389162e5fd23d78aadca3cd05090ae8be8b16e68b99d`; protocolSHA `e9172028088440d8a46a5966946c63cb723606ea916833d3c81e5f67ccfae026`. Exact unchanged608features and40%samples; overrides1200trees/depth8/lr.03/devicecuda:0; all otherparameters fixed.2021 weights0/.1/.25/.5 against accepted50%GPU608blend plus same-weight raw600/depth6control, lock before2022. No early stopping. This tests a combined capacity/learning-rate recipe, not depth in isolation. Independent source/package review pending; no fit or score. Estimated20–40minutes per14-unitfold is unmeasured extrapolation from291.88s baseline, to refine from actual unit runtime. Full source snapshots, input/control hashes, environment, command and runtime in package manifest.

## submission-cuda-xgb608-standalone-001 — 2026-11-02T17:36:16.991665+00:00

Status complete. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/export_cuda_xgb608_standalone.py --protocol docs/cuda_xgb608_standalone_protocol.json --protocol-sha256 d808593cff7033d80b4482ede212b82509ba78385a249f446e7fd57cbd915f3d --verified-blend-run artifacts/submission-cuda-xgb608-context-joint-blend-001 --verified-blend-manifest-sha256 6d13425abe03f5aa7ffe36baffd9b6c06e197f28a4b1afec90c79a8313053555 --run-id submission-cuda-xgb608-standalone-001`. Predeclared raw608GPU plus fixedVinjeprojection; no weight search/training. Validation {'2021': 0.99483023883311, '2022': 0.9935545164712399}. CSV SHA 80c41cbdb328905edc400793e282de8b6411a6cddd3f1ebdbd9915541ac0e428. Full provenance in manifest; runtime 1.708s. No Kaggle score at export.


## lp-shop-full-001 — 2026-11-02T17:12:16.654880+00:00

Status complete; label-free compact56feature cache only, no fitted model. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python artifacts/lp-shop-full-cache-preflight-001/source/scripts/build_lp_shop_features.py --protocol artifacts/lp-shop-full-cache-preflight-001/source/docs/lp_shop_full_cache_protocol.json --protocol-sha256 6501db4dd0c2f292f5373310b44e1133e58ff77578c8a031ada1c1c64ac38325 --run-id lp-shop-full-001`. Exact source/data/protocol/cases/rawinputs/replays/features and failed-run reuse provenance in `artifacts/lp-shop-full-001/manifest.json`. Original608 prefix preserved for later addition; same supplied-cost solver,36verified cases reused, no target values decoded. Runtime 1495.582s, peakRSS 1078575104bytes. No validation, submission or transfer claim.

## cuda-xgb608-submissions-independent-audit-001 — 2026-11-02T17:37:14.764436+00:00

Independent primary and standalone CSV verification: passed in 2.703 seconds. No model fit or Kaggle result. Exact audit source, Python/platform and input hashes are retained in `artifacts/cuda-xgb608-submissions-independent-audit-001/manifest.json` (SHA256 `4d8bce81e8748dd274968a0a53bf6d5fc274e60b056451369eefd3c41210f19d`). Reconstructed both primary fixed50%blend and standalone GPU608 arrays exactly from the saved projected anchor and raw hosted predictions. Independent isotonic min-max check, exact CSV byte reconstruction, all725IDs/2353columns, finite[0,1]range and serialization precision passed. Strict primary artifact closure and hosted raw/replay equality rechecked. Both ready for the separately authorized submissions; standalone remains a diagnostic with weaker validation.

## cuda-xgb608-capacity-independent-audit-001 — 2026-11-02T17:38:21.900832+00:00

Independent capacity package verification: passed in 0.963 seconds. No model fit or Kaggle result. Exact audit source, Python/platform and input hashes are retained in `artifacts/cuda-xgb608-capacity-independent-audit-001/manifest.json` (SHA256 `59eedac3d0d7f8b1f61f08457f631c3112d7d0a0e528373aed638a951ea6093f`). Preserved exact608 feature/source/loader/input closure, independently reconstructed both40% chronological sample arrays, and verified the only overrides are1200trees/depth8/lr.03/deviceGPU0. Code-only archive memberbytes/hashes and safe allowlist pass; all664-specific assumptions removed. Three native-config/locked-gate tests passed. Approve private code upload plusGPU0smoke/2021, with2022 only after an independently verified passing2021lock. Remote PRIVATE visibility remains a UI check.


## cuda-xgb608-final-003 — successful persistent background run and exports

Recorded 2026-11-02T17:40:36.787352+00:00. Private Kaggle notebook version4, scriptVersionId364965805, completed the unchanged frozen final recipe in332.170866s. All14 models and121800×14 raw test probabilities replayed bitwise. Final archive SHA `83a0d528cfcf3817e35c060f7446532f017ec2a92a0138db920f58fa9ed0bf3b`,19,473,846bytes; hosted manifestSHA `0177350df6a91f1d0d1eb5eb84f10bc27e01bfe6db3b7c8140a3dc6a63a286db`. Exact commands, original source, full196358-row sample, data/feature/source/model/config/environment hashes are in the downloaded manifest. Strict original exporter completed primaryCSV SHA `729ce3e406560b30ab867ad6e0f448650a304168e56293d3d681c5fd7d638c76`; separately predeclared diagnosticCSV SHA `80c41cbdb328905edc400793e282de8b6411a6cddd3f1ebdbd9915541ac0e428`. Both independently reconstructed arrays and CSV bytes exactly; proof `artifacts/cuda-xgb608-submissions-independent-audit-001/manifest.json`. Root is submitting them; no new public score known here.

Recovery archiveSHA `edb0d572820d61f68968eed232748d6985fd37ef247ecf69089b516d31b2be1b` passed every member hash and manifest/source/native-smoke check. New actual smoke003SHA `7458da0d71dd52803e8592589443fe11668bdf9fefd8f036bf4fb7fd9eb2bfad`; original historical smokeSHA `56baf65e855f47c0c57aca4b14c0cbd774fffff59b4ee9ecb1748b720880d2e8`. Runtime, package versions and synthetic model bytes match each other and the final run. Exact audit command `python3 artifacts/cuda-xgb608-recovery-return-audit-003/audit.py`, runtime.00268s; proof in that artifact directory. Earlier session002 also ended after an observed successful smoke002 (`7692b53211b7bf37ee9459ba5a7ad3f5e08522ce8237ea279a0c92b0f468e228`), with no retrieved completed final002. Interruption cause remains unproven; user reported interacting with/stopping the notebook. No failed attempt is assigned a model score.

## GPU608 public submission observations — 2026-11-02T17:41:33.893926+00:00

Primary frozen50%blend public0.99640; standalone architecture diagnostic0.99627. Root independently observed both Complete rows with exact descriptions/checksums and leaderboard teamrank20,best0.99640,32entries; leaders0.99820/0.99808/0.99805. Priorbest0.99631. Primary promoted; no leaderboard weightsearch, no finalselection checkbox change, no transfer. Numeric submission IDs/private scores not exposed. Separate immutable-manifest-safe observation files in each export directory and docs/kaggle_scores.json. Full training/runtime/data/source/hyperparameter/replay evidence in the referenced final003/export records; no retraining for these submissions.

## cuda-xgb664-source-preflight-001 — 2026-11-02T17:21:22.648154+00:00

Status complete; five targeted664 protocol tests pass, covering same parameters/grid, immutable2021 lock, matched608control, nativeCUDA664feature/600treeconfiguration, byte/label alignment and smoke input discovery. No model or solver run. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python -m unittest discover -s tests -p test_cuda_xgb664.py -v`. Source snapshots, packages, log hash and runtime in `artifacts/cuda-xgb664-source-preflight-001/manifest.json`. Independent source review requested.

## verify-lp-shop-full-001 — 2026-11-02T17:37:14.367049+00:00

Status complete. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/verify_lp_shop_cache.py --cache artifacts/lp-shop-full-001 --manifest-sha256 e3d9f53ec7ac67a0afcc48c9630a215ad0a2be4abbfc07aa03cdc3a4345d8163 --run-id verify-lp-shop-full-001`. Label-free raw-cache verification; no solver/model runs. Full hash/replay/feature/input/physical-balance checks in `artifacts/verify-lp-shop-full-001/manifest.json`; runtime 57.327s. No validation or Kaggle claim.

## cuda-xgb664-package-001 — 2026-11-02T17:38:12.133171+00:00

Status complete. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/build_cuda_xgb664_package.py --run-id cuda-xgb664-package-001`. Separate PRIVATE664 code/cache dataset, unchanged600treeGPU parameters/sample, disjoint56feature supplement; no training/upload. Full source/data/cache/archive/environment/protocol pins in `artifacts/cuda-xgb664-package-001/manifest.json`; runtime 0.462s. Independent audit required before hosted use.


## cuda-xgb664-package-feature-independent-audit-001

2026-11-02T17:43:42.287651+00:00. Complete package and real feature-loader audit passed in8.4395s, peakRSS5,421,940,736bytes. Exact command: `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python artifacts/cuda-xgb664-package-feature-independent-audit-001/audit.py`. Verified source/package/archive byte closure, disjoint original+608supplement+SHOP bundle union, and actual fresh staged construction of612696×664 inputs under2021target cutoff. Original608 values/dtypes/formulas are preserved; all appended56columns match the completed cache bitwise. No model fit,2022label decoding or remote action. ProofSHA `126f72e716b59fa52e6f8fc5d4a416314f03b99a057bfd0d04d1a720d4e40953`. Approves private upload/synthetic smoke/2021 experiment under root scheduling;2022 still requires locked2021 acceptance.

## cuda-xgb608-capacity-notebook-001 — 2026-11-02T17:42:06.254712+00:00

Prepared clean saved notebook; no hosted execution or training in this run. Source SHA256 `0288d555f20075aacb2ed1a6f9c7e3ad1b7cf10544d33bdaba5b9a50dceb9dcd`, notebook SHA256 `3713a06b480fcce1d3acaf30b5534509f69834e3323f2e0da64a65cf15bd9218`; exact source/runtime/protocol references and local environment in `artifacts/cuda-xgb608-capacity-notebook-001/manifest.json` (SHA256 `3440d4d8a7d207e8b70fca25d01b77fe202bf65cd74351d61208b08ce7e921d4`). All cells unexecuted with empty outputs; Python AST parses and payload source/protocol pins match. Capacity-only alternative executes fresh GPU0 smoke then its single2021fit in foreground with20second heartbeats and returns the safe smoke/fold ZIP. Exact verified final003 Python/package/uv/two-T4 runtime is required; there is no install/download fallback. Rejects existing runIDs/logs/archives. Independent xgb review approved the wrapper.

## cuda-two-gpu-2021-notebook-001 — 2026-11-02T17:44:53.406756+00:00

Prepared clean saved notebook; no hosted execution or training in this run. Source SHA256 `e6074a1c16be49cfe088f1acc48febff9ab46a6fb88aedb44f632286175c364e`, notebook SHA256 `052fa317edeb55a76db2cc3b08c06525bd71c5683de9297b65e7547d78bb2555`; exact source/runtime/protocol references and local environment in `artifacts/cuda-two-gpu-2021-notebook-001/manifest.json` (SHA256 `aabf81e9e20156c929f59797c4e24c6c3b4e4393ebac5aa21123dfbbdbaf0745`). All cells unexecuted with empty outputs; Python AST parses and payload source/protocol pins match. This is the recommended alternative to the capacity-only notebook. Sequential fresh smokes avoid shared offline-uv environment races. GPU1 SHOP664 starts first; GPU0 capacity starts after the first completed model, at least20seconds and12GiB available host memory. Otherwise execution falls back to sequential. Twenty-second heartbeats keep the saved execution foregrounded. Each experiment archives its own smoke and2021fold namespaces plus fitlog immediately when complete, preserving the other job if one fails. Both original protocols remain unchanged; no2022, selection, final fit or submission is executed.

## Private two-GPU input upload — 2026-11-02T17:46:36.029560+00:00

Created and visibly verified both new PRIVATE datasets and attached them to the existing private notebook, preserving original608/baseinputs. Exact archive hashes/URLs/settings are in `artifacts/cuda-two-gpu-input-upload-001/manifest.json`. No training started during upload. Native UI actions under user authorization.


## cuda-two-gpu-2021-notebook-independent-audit-002

2026-11-02T17:49:52.070996+00:00. Independent combined-notebook review passed; no remote launch by reviewer. Exact command `python3 artifacts/cuda-two-gpu-2021-notebook-independent-audit-002/audit.py`. Notebook SHA `2eef12f6de3a7b8c66792695d79bf5d40792515a3cb699a3e960e7d134aabb6d`, sourceSHA `ff52ce56947e9fccc2e11104333c84ac1f8a2018830062c61cdde33827499e5f`. Verified exact code cell, both frozen protocols/runtime/source pins, distinct GPU0/GPU1 stage/output namespaces, sequential smokes to avoid uv environment creation races,2021-only commands, independent completion/failure ZIPs. Seven memory guard fixtures passed: effective free memory is minimum of host availability and finite cgroup allowance minus usage; unknown telemetry runs serially. Capacity starts only after664 first completed model,20seconds and12GiB effective free memory; otherwise serial. ProofSHA `d7bc2fb085a589b391a2effcaa3e71ef5bb0480a58b80bb952494fed08b67c23`. Initial reviewer fixture incorrectly expected one total notebook cell and a compressed wrapper; actual artifact has harmless markdown plus one exact plaintext code cell. Corrected reviewer only; preserved `failed_assumption_source.py`; candidate unchanged. Root authorized background Save&Run after this approval.

## Two-GPU2021 hosted launch — 2026-11-02T17:50:35.198956+00:00

Private savedversion5, scriptVersionId364971894, Queued asobserved. Exact revision002 hash, audit andsettings are in `artifacts/cuda-two-gpu-input-upload-001/launch_observation.json`. No score or successfulruntime yet; startupguards will reject unpinned runtime.


## cuda-two-gpu-2021-notebook-002 — 2026-11-02T17:47:10.178359+00:00

Prepared and independently approved; no training or remote execution by this preparation run. Preserved notebook001 and changed only its runtime memory guard: use the minimum of host MemAvailable and finite cgroup v2/v1 memory allowance minus current usage; unknown telemetry retains serial execution. Seven synthetic memory fixtures passed, covering host-limited, container-limited, unlimited, v1, missing, malformed and exhausted memory. Exact source, both protocol hashes, prerequisite audit hashes, pinned final003 runtime and local Python3.14.6/macOS27.0.1 environment are recorded in `artifacts/cuda-two-gpu-2021-notebook-002/manifest.json` (SHA256 `5fc77375ada65775f1d6df5d2a5d85917cb449e2a29036dc3f5eb9f62a262c26`). Source SHA256 `ff52ce56947e9fccc2e11104333c84ac1f8a2018830062c61cdde33827499e5f`; clean notebook SHA256 `2eef12f6de3a7b8c66792695d79bf5d40792515a3cb699a3e960e7d134aabb6d`. All cells remain unexecuted with empty outputs. Independent proof `artifacts/cuda-two-gpu-2021-notebook-independent-audit-002/manifest.json` SHA256 `d7bc2fb085a589b391a2effcaa3e71ef5bb0480a58b80bb952494fed08b67c23`.

Execution uses the exact commands generated by the pinned source: each discovered dataset launcher receives its frozen `--protocol-sha256`, first `--mode smoke --run-id <fresh smoke ID>`, then `--mode fit --run-id <fresh 2021 ID> --year 2021 --smoke-manifest <its generated smoke manifest>`. Fresh IDs are `cuda-xgb664-smoke-001`, `cuda-xgb664-2021-001`, `cuda-xgb608-capacity-smoke-001`, and `cuda-xgb608-capacity-2021-001`. Both genuine smokes run sequentially. SHOP664 uses GPU1 with600trees/664features; capacity608 uses GPU0 with1200trees/608features and starts after the first SHOP664 model,20seconds and12GiB effective memory, otherwise serially. Fixed seeds, samples, chronological split, features, model parameters and acceptance gates remain exactly as recorded in their original protocols; the wrapper performs no model selection. Each experiment independently archives its smoke and2021 output plus fitlog at completion or failure. No2022, final fit, export, submission or measured model score in this preparation. Execution instructions: `docs/cuda_two_gpu_2021_execution.md`.


## gpu608-pair-error-diagnostic-001 — 2026-11-02T17:53:10.352013+00:00

Read-only exact decomposition of the existing accepted50% GPU608 blend; no fitting, weight selection, input features or submission. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/analyze_gpu608_pair_errors.py`. The two365-case chronological arrays and their source manifests match the frozen capacity-protocol anchor pins. A tie-aware14×14 directed unit-pair count is partitioned into same/different forecast case, with a separate half-endpoint allocation by true transition/episode context. Synthetic exhaustive pairs pass; full matrices sum to1−micro AUC within1e−13, all14 diagonal AUCs match sklearn, and per-unit endpoint sums match matrix marginals. AUC2021/2022 .9950359172801482/.9937529477072502. Cross-case error shares99.6902%/99.6254%; cross-unit90.2581%/91.1655%; cross-case and cross-unit89.9828%/90.8169%. True-switch proximity accounts for40.8596%/50.8206% of endpoint error, with truth-derived masks used only descriptively. Runtime2.294975s, peakRSS282902528bytes. Exact OS/Python/packages/source/data-array hashes and all artifact hashes in `artifacts/gpu608-pair-error-diagnostic-001/manifest.json` SHA256 `46acfb65fcd85bb2c9ff51ded375051575c5e2da5081ea48a2259a3eee9e7a9f`. Human-readable findings: `docs/gpu608_pair_error_findings.md`. Both years are reused development evidence; a loose monotone-unit ceiling is not an attainable forecast.

### Two-GPU2021 actual runtime and smoke checks

2026-11-02T17:56:37.633850+00:00: both exact protocol/runtime guards and genuine GPU smoke checks passed; initial effective memory29.60GiB. SHOP6642021 PID88 started17:56:09.39UTC. Exact observed smoke hashes in `artifacts/cuda-two-gpu-input-upload-001/runtime_observation.json`. Viewer LatestContainerImage label does not override measured exact runtime checks. No completed fold score yet.

## full14-early-oof-preflight-001 — 2026-11-02T17:58:15.526272+00:00

Three targeted tests passed in0.851s; no training or label decoding. Source/protocol setup for33 missing early-year single608LGB models, preserving9 existing Vinje checkpoints/probabilities and all samples. Exact test command, source snapshots, environment and log in `artifacts/full14-early-oof-preflight-001/manifest.json` SHA `4874d7ac29e8d76c8cdfbeb7edaa3d6356c6cf67ddb7c6398fb849a5c7bcc471`. Frozen protocol SHA `58d98fe4d3f7ec3d7b0a102b6a0ca6a538790baaf11ddb0ef2d3677cefc96028`. Independent audit requested before the root-authorized2-thread run.

## cuda-dual-2021-return-preflight-001 — 2026-11-02T17:59:11.538858+00:00

Prepared and independently reviewed2021-only return importer; no training, model scores or remote actions. Source SHA `e02dd43f684f35e9541d6869e0417f15a3f694d3ffdfad5162feed03adf9b8bc`; compile and protocol/source/API/namespace checks passed. Commands and externally observed fresh smoke hashes are in `docs/cuda_dual_2021_return.md`; successful hosted archive and fold-manifest hashes are still required. Frozen selectors retain grid[0,.1,.25,.5], accepted608GPU anchor and same-weight600-tree control. Source snapshot and review in `artifacts/cuda-dual-2021-return-preflight-001/`.


## full14-early-oof-independent-audit-001 — 2026-11-02T17:59:23.968898+00:00

Independent preflight passed in1.0173s. All frozen source/data/input hashes and every prior CRF artifact remain exact. Real date-only CSV metadata independently reproduces1090/1455/1820 training starts,365/359/360 prediction starts, six-day horizon purge, and all three sequential seed4173 40% sample arrays bitwise. Nine saved Vinje models/probabilities are reused with unchanged case order;2020 file has no labels. Three split/reuse/schema tests pass. Runner manually reviewed to fit only33 missing unit models, replay all42 checkpoints, and keep2020+ targets masked. No model fitting or target decoding by reviewer. Exact command/environment/source/test hashes in `artifacts/full14-early-oof-independent-audit-001/manifest.json`, SHA256 `3b617c0f53a9881bdde381404019f3848621038cc339d1fbd53137520414f493`. Root authorized the subsequent bounded2-thread OOF build; this audit itself creates no predictions or scores.

## full14-early-oof-001 launch — 2026-11-02T18:01:47.164940+00:00

Root-authorized2-thread run launched after independent audit SHA `3b617c0f53a9881bdde381404019f3848621038cc339d1fbd53137520414f493`. Command `.venv-ml/bin/python scripts/build_full14_early_oof.py --run-id full14-early-oof-001 --protocol-sha256 58d98fe4d3f7ec3d7b0a102b6a0ca6a538790baaf11ddb0ef2d3677cefc96028`; output redirected to `artifacts/full14-early-oof-001.log`. First new2018Hogga model completed37.20s with exact checkpoint replay; full run remains in progress. This is33 missing models plus9 exact reused models for a single608LGB earlyOOF screen anchor, not the accepted nestedblend. No new validation or Kaggle score. Final runner record will contain full runtime/environment/hashes.

## integer-water-budget-completion-independent-audit-001 — 2026-11-02T18:05:05.359228+00:00

Passed in2.066s, no optimization or model fitting. Revalidated all source/data/cache/old/new artifact hashes; metadata-only12case choice, original targets, four timeout-only retries and16 reused result/proof files; rebuilt physical LP matrices and checked both raw MILP incumbents and saved fixed-state outputs, all bounds/equalities/inequalities, integrality, objective/bound/gap arithmetic, zero slacks, all proxies/metrics and six gates. Weak3 errors844→701; all14 errors2805→2643; all gates pass. Full proof SHA `f428c3633d3fe7f5a83dffd06f0c2edf1324443a2cee465dc01052ab7c76b7d2` in `artifacts/integer-water-budget-completion-independent-audit-001/manifest.json`. The runner asserted two exact fixed-state LP replays, but only one solution is stored; reviewer verified this source/proof closure without independently repeating optimization. Twelve-case mean runtime extrapolates7.77hours for3647cases; full62second caps permit62.81hours before overhead. These are uncertain cost estimates, not full-cache measurements. No full-cache launch or model acceptance.

### Two-GPU2021 hosted completion and downloads

2026-11-02T18:09:38.860396+00:00: privateversion5 successfully ran601.5s. SHOP664fit300.980828s andcapacity608fit497.324278s; bothcomplete with all14 checkpointreplays. Bothseparatelyarchivedreturnfiles downloaded via nativeUI and exactbytecounts/SHA matched externally observed hostedvalues. Full hashes andlocalpaths: `artifacts/cuda-two-gpu-input-upload-001/completion_observation.json`. Independent strictimport/selection remains separate; no2022 orpublicscore inferred.

## verify-cuda-xgb664-2021-return-001 — 2026-11-02T18:09:07.707964+00:00

Status complete; strict2021 GPU return import and frozen selector. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/import_cuda_dual_2021_return.py --experiment shop664 --archive artifacts/cuda-xgb664-2021-001-download.zip --archive-sha256 cb9581e0b4a867a8f8ef679a1cc1d343773f75aa55c2c0b8da9ffb06ed2ca511 --manifest-sha256 2981b28011c35efa69d3c24f416f6b178c7b69fd0a385166adb477de6a4de8ec --smoke-manifest-sha256 dcce36270b687a5a51cb8ae41f0b71ab83a6b51e50c2199fd2909990f3f7fe78 --run-id verify-cuda-xgb664-2021-return-001 --selection-run-id cuda-xgb664-select-2021-001`. Selected weight 0.25, AUC 0.9951018232145258, anchor 0.9950359172801482, matchedcontrol 0.9950177387934181, eligible2022 True. Runtime 3.982s. All hashes/environment/sources in manifest; no training or Kaggle submission.

Externally observed archive/fold/smoke hashes verified; frozen nativeGPU/model/sample/source/feature/replay checks passed. Return audit SHA `c6e0f7f6e3b0bfaaf2dfa110ba8320630033ca4be756da4d5d67a9bf19557479`, selector manifest SHA `879657d952518498a9b8c2a02c74f2629f4ddb8b3e1e96b2f7a8e1e9850b7d77`, lock SHA `1783327cd4ae46182f19ee902dac606f106115d66676518e8f3004e159b6cce5`. Full2021 grid: `{"0.0": 0.9950359172801482, "0.1": 0.9950724016705688, "0.25": 0.9951018232145258, "0.5": 0.9950990294407883}`; raw candidate AUC 0.9948234939791991. Explicit2022 eligibility is provisional pending independent result/lock audit. No new public submission or2022 result.

## verify-cuda-xgb608-capacity-2021-return-001 — 2026-11-02T18:09:07.707933+00:00

Status complete; strict2021 GPU return import and frozen selector. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/import_cuda_dual_2021_return.py --experiment capacity608 --archive artifacts/cuda-xgb608-capacity-2021-001-download.zip --archive-sha256 3ebc6cb5763ca2a713a99706e139c2b46d3a4fa9df0da8a5efb459948b3e4fb5 --manifest-sha256 a0bbf82b98f5a71549901a1006aee54d08277c7dea0f930ea1842be764adb935 --smoke-manifest-sha256 eb431523977bbc4c883780f3b4d593e9f111a64257ed7ce7c995013590250a5f --run-id verify-cuda-xgb608-capacity-2021-return-001 --selection-run-id cuda-xgb608-capacity-select-2021-001`. Selected weight 0.25, AUC 0.9950519105980167, anchor 0.9950359172801482, matchedcontrol 0.9950177387934181, eligible2022 True. Runtime 4.034s. All hashes/environment/sources in manifest; no training or Kaggle submission.

Externally observed archive/fold/smoke hashes verified; frozen nativeGPU/model/sample/source/feature/replay checks passed. Return audit SHA `229fcd30439ed7a50763c09a747f8698fd5c397e909e858fb8583b6e69d6f18d`, selector manifest SHA `bd3131564204028fe6b75781e44e4228fc743e5e48e6facf09bfaadfc25b0636`, lock SHA `f418296446f832b408ccc2e7ca3aee6825408e11420767252f92785f193665be`. Full2021 grid: `{"0.0": 0.9950359172801482, "0.1": 0.9950458693030391, "0.25": 0.9950519105980167, "0.5": 0.9950393675693765}`; raw candidate AUC 0.9948450478035217. Explicit2022 eligibility is provisional pending independent result/lock audit. No new public submission or2022 result.


## cuda-xgb608-capacity-2021-independent-audit-001 — 2026-11-02T18:11:32.065381+00:00

Independent capacity6082021 result audit passed in0.9715s. All import, hosted and selector artifact hashes, nativeGPU0/depth8/1200trees/.03 configuration, exact samples/feature schema/case IDs and original2021 labels checked. Independent min-max isotonic projection reproduces all fixed0/.1/.25/.5 scores, selected.25, matched600-tree control and strict eligibility. Selected AUC .9950519105980167 versus anchor .9950359172801482 and control .9950177387934181. Exact lockSHA f418296446f832b408ccc2e7ca3aee6825408e11420767252f92785f193665be; eligible for fixed2022 confirmation. Reviewer did not repeat14 hosted model predictions; original exact checkpoint replay evidence and artifact closure were verified. No2022 label decoding, training or remote action. Exact command, environment, sources and evidence are recorded in `artifacts/cuda-xgb608-capacity-2021-independent-audit-001/manifest.json`, SHA256 `9f85a649b3d5e82c63fa90d480762ddb55c6a552507d55f309bf9e1c8b72b4da`.


## pooled-pairwise-residual-preflight-001 — 2026-11-02T18:10:17.905285+00:00

Eight targeted numerical and integration tests passed. Exact pair gradient/full Hessian/PSD2×diagonal majorizer, uniform pooled sampling, balancedBCE normalization under unbalanced prevalence, common-score-shift invariance, one dataset reused by all3 objectives, initial-score and saved-model replay, byte-exact zero identity,623-column row alignment and poisonousfuturelabel exclusion checked. Only tiny synthetic3-round fits; no competition residual fit. Ordinary and globally class-balancedBCE controls have identical80-tree/depth4/lr.03 capacity. Frozen protocolSHA a6e75e1665863312b29d64f74dd5ae9bbd54eee99c8d59dcff35e2e1c1220c15. Root fixed advancement before screen results: projected full-year gain overidentity+bothBCE and gain overidentity in bothhalves;5% relative1-AUC reduction is a separate substantial-gain flag. Completed independent full14OOF audit remains required before fitting. Exact command, environment, sources and evidence are recorded in `artifacts/pooled-pairwise-residual-preflight-001/manifest.json`, SHA256 `1f5147749f6a8ed3a3f3eda5e600089584505ac0d539af053d931c62dc6ab7ae`.

## Pooled residual offset numerical preflight

2026-11-02T18:12:59.902650+00:00: existing2018 OOF probabilities span2.082169e-7 to0.999999821; epsilon1e-6 wouldclip3,312low+15,222highvalues. Before residualprotocolfreeze and without2020labels, recommendfloat64logitepsilon1e-12 plus exactidentitybranch. FullinputSHA/environment/arithmetic in `artifacts/pooled-residual-offset-preflight-001/manifest.json`. No model, label-basedchoice orscore.

## cuda-two-gpu-2022-notebook-independent-audit-001 — 2026-11-02T18:13:58.973462+00:00

Passed final-byte/source audit of real independently accepted25%2021 locks for both SHOP664 andcapacity608. Notebook SHA `21844a0ef5da6027e7aefb9a7315cf4aabd1edf187ccc3788e8454aeb3f5efe4`, source SHA `d023bdfe38af784ac360a63956e910ee84bc8987bb836cdf3a75df647d424bdc`. Freshsmoke002/fold2022-001 only, original protocols, exact lock/audit payloads, trainer lock arguments, returned lockhash checks, unchanged sequentialsmokes/cgroup memoryguard/archive/heartbeat and distinctGPU namespaces. No fitting or remote action by reviewer. Audit manifest SHA `cf19c20525be31b6e88548ee83f30ec969d2aa5cc6ae81669aee4acace62443f`. Approved root backgroundlaunch;2022 result unknown.


## cuda-two-gpu-2022-notebook-001 — 2026-11-02T18:13:27.546036+00:00

Prepared clean2022 notebook after both independent2021 result audits passed. Exact builder command, original002template pin and embeddedlock/auditbytes are in `artifacts/cuda-two-gpu-2022-notebook-001/manifest.json` SHA256 `a2a66268a0f1edd933144a14037ad1d77f01d582de8d67fe93d9c25904ac9c67`. Both branches independently retain fixedweight.25. Freshsmoke002/fold2022-001 IDs, strict originalprotocol/runtime/source guards, cgroup-aware stagedconcurrency and separate returnarchives are preserved. NotebookSHA `21844a0ef5da6027e7aefb9a7315cf4aabd1edf187ccc3788e8454aeb3f5efe4`; sourceSHA `d023bdfe38af784ac360a63956e910ee84bc8987bb836cdf3a75df647d424bdc`. Independent wrapper auditSHA `cf19c20525be31b6e88548ee83f30ec969d2aa5cc6ae81669aee4acace62443f` passed. No hosted execution, model fit or result in this preparation. See `docs/cuda_two_gpu_2022_execution.md`.


## pooled-pairwise-residual-preflight-002 — 2026-11-02T18:15:16.945082+00:00

Nine targeted tests passed after a pre-fit numerical correction. Root inspected only already available2018OOF predictions and found18,534 probabilities clipped by1e-6. Revision2 preserves every interior probability in float64 logit computation and handles exact0/1 with1e-12; explicit all-zero residual still returns original bytes. Added near-endpoint order test, including an interior value below1e-12. Original unexecuted protocolSHA `a6e75e1665863312b29d64f74dd5ae9bbd54eee99c8d59dcff35e2e1c1220c15` and preflight snapshots are preserved. No competition residual fit or new2020 label access occurred before this correction. New frozen protocol `docs/pooled_pairwise_residual_protocol_v2.json` SHA `dc01041ec5c239d683c9f7f2d9d5efe36ba19b6fa47e9311bf695442c85405a1`; preflight manifest `artifacts/pooled-pairwise-residual-preflight-002/manifest.json` SHA `b4a52dcbfd38ef2f44df9f618a866aea16b3523bed4d61980dbdc5106c50ec16`. All other objectives, controls, fixed capacity and predeclared incremental/substantial advancement rules remain unchanged.


## head-flow-bound-screen-001 — 2026-11-02T17:47:40.967244+00:00

Status complete. Label-free head-dependent turbine-bound diagnostic on36metadata-selected pre2021cases, all14units. No LPsolve/fit/targetvalues. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/diagnose_head_flow_bounds.py --run-id head-flow-bound-screen-001`. Screen True; original capacityratio>1.01 in 0.0 ofcase-unitpairs, dispatchedviolationfractions baseline 0.12020502645502645, SHOP 0.1134141156462585. Full source/data/cache/formula/gate/environment/CSVproof `artifacts/head-flow-bound-screen-001/manifest.json`; runtime 1.952s. No validation/leaderboard claim.



## integer-water-budget-screen-001 — 2026-11-02T17:55:52.683626+00:00

Status complete. Fixed12-case pre2021 integer water-budget diagnostic, three prescribed binary unit states, same physical/economic LP control. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/diagnose_integer_water_budget.py --protocol-sha256 9a4f29a1ad78204f391d1a4d4b100e1ea6d67519bdb0d0fa50c00a69d984b60b --run-id integer-water-budget-screen-001`. Per-case8-second MILP + two1-second fixed-state LP caps; timeouts and unproven incumbents retained. No model fitting. Screen False; weak3 errors {'control': 844, 'integer': 688}. Full protocol/source/data/inputs/results/solver bounds/statuses/labels/environment/runtime proof `artifacts/integer-water-budget-screen-001/manifest.json`; runtime 59.008s. No validation or Kaggle claim.



## integer-water-budget-completion-preflight-001 — 2026-11-02T18:01:10.725010+00:00

Status passed_preflight_only. Separate solver-completion continuation of the failed original 12-case screen. Four original timeout cases receive 60-second MILP limits; eight proven result/proof files are reused byte-for-byte. Identical formulation, inputs, targets and gates. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/complete_integer_water_budget.py --preflight --protocol-sha256 3a71920074eafc262d7842b37fca1c6ea03d8a3cebd0c7821fb7345656b74a47 --run-id integer-water-budget-completion-preflight-001`. Screen passed: None. Runtime 0.003 s. Every hash, status, bound, runtime and metric is recorded in `artifacts/integer-water-budget-completion-preflight-001/manifest.json`. No model fit, new validation year, promotion, full cache or submission.



## integer-water-budget-completion-001 — 2026-11-02T18:01:23.021235+00:00

Status complete. Separate solver-completion continuation of the failed original 12-case screen. Four original timeout cases receive 60-second MILP limits; eight proven result/proof files are reused byte-for-byte. Identical formulation, inputs, targets and gates. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/complete_integer_water_budget.py --protocol-sha256 3a71920074eafc262d7842b37fca1c6ea03d8a3cebd0c7821fb7345656b74a47 --run-id integer-water-budget-completion-001`. Screen passed: True. Runtime 69.473 s. Every hash, status, bound, runtime and metric is recorded in `artifacts/integer-water-budget-completion-001/manifest.json`. No model fit, new validation year, promotion, full cache or submission.



## gpu608-pair-error-independent-audit-001

Independent pair-error audit passed all input, source and output hashes. Independent average-rank calculations verified pooled and unit-diagonal global and same-case counts, all partitions, endpoint totals and sklearn AUC. No model fitting or target-based changes. Command: UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python artifacts/gpu608-pair-error-independent-audit-001/audit.py.

Full evidence, input/output/source checksums and exact arithmetic: `artifacts/gpu608-pair-error-independent-audit-001/manifest.json`, SHA `9ffd361a2fa06b4ab08ab4995d051ba5d92b74c43386c5625b0fa2f71d4c9c97`.



## cuda-xgb664-select-2021-independent-audit-001

Independent audit assertion failed: strict hosted/local raw-AUC float equality differed by -1.1102230246251565e-16. The existing frozen verification contract permits 1e-12. Preserved failure and source unchanged; no candidate/model/protocol artifact changed. Command: UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python artifacts/cuda-xgb664-select-2021-independent-audit-001/audit.py.

Full evidence, input/output/source checksums and exact arithmetic: `artifacts/cuda-xgb664-select-2021-independent-audit-001/manifest.json`, SHA `2f79c26532bcee77c0376d40c07c0852b7f5e1616602584f9868bdbb1b904e92`.



## cuda-xgb664-select-2021-independent-audit-002

Independent audit passed with the existing frozen 1e-12 hosted/local AUC contract. All hashes, native model configurations and saved replay proofs match; independent generic PAVA exactly reproduced the four fixed-weight scores and selected prediction array. Weight 0.25, AUC 0.9951018232145258, anchor 0.9950359172801482, same-weight control 0.9950177387934181. Eligible for separately locked 2022 evaluation. Exact lock SHA 1783327cd4ae46182f19ee902dac606f106115d66676518e8f3004e159b6cce5. No training or submission. Command: UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python artifacts/cuda-xgb664-select-2021-independent-audit-002/audit.py.

Full evidence, input/output/source checksums and exact arithmetic: `artifacts/cuda-xgb664-select-2021-independent-audit-002/manifest.json`, SHA `b52056d8ed6f9619f80ee242a2c88dd72ef2bb6f4f433b1c42e8775489ffa28c`.


## Frozen2022 dual-GPU hosted launch

2026-11-02T18:15:57.489877+00:00: version6 scriptVersionId364978490 isPrivate andRunning. Both actual independently audited2021 .25 locks embedded, freshsmokes beforeonly2022fits, unchanged runtime/memoryguard. Exact notebook/lock/audit hashes andURL in `artifacts/cuda-two-gpu-2022-launch-observation-001/manifest.json`. No2022 result orpublicscore yet.

## cuda-xgb608-fullsample-package-001 — 2026-11-02T18:16:41.100197+00:00

Status complete; root-authorized preparation only, no fitting or upload. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/build_cuda_xgb608_fullsample_package.py --run-id cuda-xgb608-fullsample-package-001`. Original600/depth6/lr.05/GPU1 recipe and608 features exactly unchanged; external row inclusion1.0 versus.4, while per-tree subsample.9 remains fixed. Full purged row counts `{'2021': 367248, '2022': 428568}` and NPY hashes `{'2021': '33c3e63a934030522c1fe0f14211bdf4b55fa32ef5db87e687982161b0e742a7', '2022': '29d7498c158e5be8846cbdaa5290b9a290c6ca25de8c0661c0d4f17b94eda328'}` reconstructed from metadata; original40% rows verified nested. Four targeted tests passed(.072s), five source files compile. Fixed2021 grid[0,.1,.25,.5], accepted public0.99640 anchor, same-weight original raw600-tree control, immutable eligiblelock before2022. Protocol SHA `d5add5941279d980f156015b4b5ca70da2af4b6d8b99c4841050569b02b7255e`; PRIVATEcodeZIP58,844bytes SHA `de8852a280ca1b8140d680648c9b2e35833ff3552f9b6a619858a7fbbbeaeff9`. All source snapshots, environment, data/bundle references and exact parameters are in the package/protocol. Rough12.5min/fold extrapolation is unmeasured; actual GPU cost pending. Independent review requested.


## cuda-xgb608-fullsample-independent-audit-001 — 2026-11-02T18:18:10.415319+00:00

Independent code/package audit passed in1.1300s. Every original608GPU model parameter remains identical:600trees/depth6/lr.05/GPU1, per-tree row/column fractions.9, seed4173 and2threads. Only external row inclusion changes.4→1.0. Independently rebuilt all367248/428568 purged case-hour indices, exactNPYhashes, no duplicates/omissions and nested original40% samples. Source/input/ZIPbyteclosure, safe private code-only members, actual hash-based launcher discovery and4 targeted tests pass. Inherited folds.sampled_training_rows is historicalbaseline metadata; authoritative new counts are sample_rows and both trainer/verifier enforce them. Full proof `artifacts/cuda-xgb608-fullsample-independent-audit-001/manifest.json` SHA256 `81a4c20bbb8ebff6289eeab59e4ed1c6444d857cfe96d0d8cd4592b1112e2c07`. Approves PRIVATE upload/smoke/2021 under root scheduling;2022 still requires independently reviewed passing2021lock. No fit, GPU execution or new score by reviewer.

## integer-water-budget-pilot-independent-audit-001 — 2026-11-02T18:18:47.409804+00:00

Passed in3.650s without solving or decoding labels. Verified all hashes and exact24 metadata-spaced inputs,48 distinct fresh worker PIDs, all24 independent result replays, and all96 complete conditionalLP output/dual records bitwise. Rebuilt original matrices and checked saved integer primal feasibility, fixed-state LP residuals, dual signs, stationarity, complementary slackness and primal-dual objective agreement. Maximum stationarity error1.16e-11, complementarity0, primalerror2.20e-12. Proof SHA `dd8f7372ab8a347f98270b7687489dedd001be6b46ec1aaa838f14fdd9a4e391` in `artifacts/integer-water-budget-pilot-independent-audit-001/manifest.json`. Timing arithmetic agrees:4.71hours from summed tasktime and5.80hours including observed process startup/scheduling for two full passes on3647cases with two workers; only24cases sampled so tails remain uncertain. No cache launch, model fit or new score.


## pooled-pairwise-objective-benchmark-001 — 2026-11-02T18:19:15.170637+00:00

Synthetic full-size objective check passed for1702848 outputs and1000000 uniformly drawn pairs. Exact frozen revision2 code; finite gradients/Hessians and nearzero gradient sum confirmed. One derivative round took0.08134266700017179s; peakRSS200507392bytes. No competition targets, features or model fitting. Source/protocol/seed/environment and draw hash in `artifacts/pooled-pairwise-objective-benchmark-001/manifest.json`, SHA256 `ba88971df717fafdac0b1ca1797a4916264b1cf7543b6b22a1813996ff248329`. This measures objective arithmetic only, not total boosted-model cost.

## pooled-pairwise-residual-independent-audit-002 — 2026-11-02T18:20:37.062054+00:00

Passed source/protocol closure and9 numerical/integration tests in1.134s; no competition fit. Frozenv2 protocol `dc01041ec5c239d683c9f7f2d9d5efe36ba19b6fa47e9311bf695442c85405a1` preserves exact interior logits, replaces only endpoints by1e-12, and retains byte-exact zero identity. Full14 pooled uniform1Mpairs/round with exactgradient and2xdiagonalPSDmajorizer; matchedordinary andbalancedBCE controls,80trees/depth4/.03,2threads. SharedDataset/offset/replay/featurealignment/futurelabelpoison tests pass. All3models andscreenpredictions sealedbefore2020labels. Root-frozen advancement is strict pooledgainoveridentity+bothBCE andeachhalfoveridentity;5%relativeerrorreduction is a separateflag. Only launch after completed independentlyverified full14OOF cache; no2021/22 fit. Proof SHA `92afd09b8844763811a1e3a016387d63786c996c4920c9fee6b0b89e7a88f8bd`; exact command/source snapshots/log/environment in artifactdirectory.

## Private full-sample608 code upload

2026-11-02T18:21:12.793799+00:00: created and visibly verifiedPRIVATE dataset, attached onlytoexistingdraft. No full-sampletraininglaunch. Exact archivehash/URL/audit: `artifacts/cuda-xgb608-fullsample-upload-001/manifest.json`. Savedversion6 continues itsunchangedlocked2022work.

## cuda-xgb608-fullsample-notebook-independent-audit-001 — 2026-11-02T18:23:34.809572+00:00

Final-byte/source audit passed. One unexecuted exact codecell, freshsmoke001/fold2021-001, original600/depth6/.05/GPU1 recipe,367248 fullpurged rows, exact protocol/source/runtime guards, checkpoint replay and archive hashes. Foregroundheartbeat and archiveallowlist match previously reviewed capacitywrapper. Notebook SHA `29636b2bd88d5b9928ef52a2f3c9e923ab4c8ac3229beee086dddf36dd066e95`; audit SHA `f184e05dca73db609a327a2fc37eee861c3d625501cf9481e112e251cc0305ee`. Prepared only, no remote launch; root schedules after current2022/acceptedfinal priority.


## cuda-xgb608-fullsample-notebook-001 — 2026-11-02T18:22:37.542494+00:00

Prepared a clean standalone fullsample6082021 notebook from the already reviewed capacity-only wrapper. Exactprotocol/source/runtime pins updated to the separate600-tree/depth6/lr.05/GPU1 full-row recipe; explicit completion check requires367248 training rows. Fresh smoke001 andfit2021-001, foreground20second heartbeats and compatible smoke+fold returnZIP. AST parses; code cell unexecuted with emptyoutputs. SourceSHA `ac4f0901fadc47653f53d19063f0a82a35210381c113d0a5b831bf25127bbd6b`; notebookSHA `29636b2bd88d5b9928ef52a2f3c9e923ab4c8ac3229beee086dddf36dd066e95`; preparationmanifestSHA `de792c4a0e2c20a0e52868870e059068d2acf15feda5fcea26ed0502a913124b`. No GPU execution, labels or score in this preparation. Independent final notebook review requested; schedule only after current2022 confirmations and any acceptedfinal work.

## gpu-dual-equal-2021-screen-001 — 2026-11-02T18:24:27.146962+00:00

One fixed50/50average of the independently selected2021 blends; original nestedprojections preserved. Hypothesis: complementaryfeatures/capacity may improve overeachindividual. Gate fixed before computing score; no2022 access. IndividualAUCs [0.9951018232145258, 0.9950519105980167]; equal blend 0.9950866874156993. Reject; no2022 evaluation or submission. Exact command/protocol/source/input/environment/split IDs/artifactSHA/runtime in `artifacts/gpu-dual-equal-2021-screen-001/manifest.json`. No training, hyperparameter search or Kaggle result.


### GPU dual2022 hosted completion — 2026-11-02T18:29:11.809091+00:00

Private saved notebook v6 (scriptVersionId364978490) completed successfully in691.7s. Both archive sizes/SHA256 match the hosted logs. Full completion observation and exact manifest/smoke hashes: `artifacts/cuda-two-gpu-2022-completion-observation-001/manifest.json`. Frozen weights remain.25; strict import and independent acceptance are pending. No new leaderboard score claimed.

## verify-cuda-xgb664-2022-return-001 — 2026-11-02T18:29:30.433540+00:00

Status complete. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/import_cuda_dual_2022_return.py --experiment shop664 --archive artifacts/cuda-xgb664-2022-001-download.zip --archive-sha256 af11ca81bc994c9d28a0a02a03d9107b68f0915d36bf47c3c4f9c62403692d10 --manifest-sha256 2aac2147bed02081437a208d9580f8e1b48a7ef41f6c0b1570c72277eafa78b6 --smoke-manifest-sha256 24f460dcec7014ac696bd8da924cd61ba9a938817a2ce36e791023bb0a55cdce --selection-lock artifacts/cuda-xgb664-select-2021-001/selection_2021.json --selection-lock-sha256 1783327cd4ae46182f19ee902dac606f106115d66676518e8f3004e159b6cce5 --lock-audit artifacts/cuda-xgb664-select-2021-independent-audit-002/manifest.json --lock-audit-sha256 b52056d8ed6f9619f80ee242a2c88dd72ef2bb6f4f433b1c42e8775489ffa28c --run-id verify-cuda-xgb664-2022-return-001 --selection-run-id cuda-xgb664-select-2022-001`. Strict2022 GPU import and unchanged2021 weight evaluation; weight 0.25, AUC 0.9938546734526688, anchor 0.9937529477072502, control 0.9937356307875711, accepted True. Runtime 3.790s; all hashes, environment and source snapshots in the manifest. No fit or Kaggle action.

## verify-cuda-xgb608-capacity-2022-return-001 — 2026-11-02T18:29:30.433536+00:00

Status complete. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/import_cuda_dual_2022_return.py --experiment capacity608 --archive artifacts/cuda-xgb608-capacity-2022-001-download.zip --archive-sha256 d74b761f4a7a73048ed04f33cedd37f47531021bd998591136eb1cd3e86fc2e1 --manifest-sha256 e44201619e800ed092c55e0b2cb99dd53cc7bacac39d9195f723d51c0bfbc534 --smoke-manifest-sha256 8cda83c2e0753d8e6c796bca1698b96f4074ece445e3354eb6d659666160805e --selection-lock artifacts/cuda-xgb608-capacity-select-2021-001/selection_2021.json --selection-lock-sha256 f418296446f832b408ccc2e7ca3aee6825408e11420767252f92785f193665be --lock-audit artifacts/cuda-xgb608-capacity-2021-independent-audit-001/manifest.json --lock-audit-sha256 9f85a649b3d5e82c63fa90d480762ddb55c6a552507d55f309bf9e1c8b72b4da --run-id verify-cuda-xgb608-capacity-2022-return-001 --selection-run-id cuda-xgb608-capacity-select-2022-001`. Strict2022 GPU import and unchanged2021 weight evaluation; weight 0.25, AUC 0.9937720670635095, anchor 0.9937529477072502, control 0.9937356307875711, accepted True. Runtime 3.882s; all hashes, environment and source snapshots in the manifest. No fit or Kaggle action.

## full14-early-oof-001 — 2026-11-02T18:00:22.126446+00:00

Status complete. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/build_full14_early_oof.py --run-id full14-early-oof-001 --protocol-sha256 58d98fe4d3f7ec3d7b0a102b6a0ca6a538790baaf11ddb0ef2d3677cefc96028`. Full14 single608LGB forward-OOF extension: 33 missing350-tree models, 9 frozen Vinje models and probabilities preserved bitwise, matched40% samples, six-day purge; 2018/2019 residual-training OOF and2020 screen predictions. No2020 labels decoded, residual fit, 2021/22 training or Kaggle action. Runtime 1755.276s; peakRSS 3946643456; all source/data/sample/model/features/probability hashes and environment in `artifacts/full14-early-oof-001/manifest.json`.


### Fullsample608 saved GPU2021 launch — 2026-11-02T18:31:59.374738+00:00

Private version7/scriptVersionId364982212. Fresh GPU1 smoke passed, hash`da97e314eb85a138941a7b92031523fc238793bba6162434f96dfd79f6718b15`; fit started18:31:33UTC. Frozen600/depth6 model uses all purged training rows; original40% sample control unchanged. Approved notebook/hash/launch and scheduling rationale: `artifacts/cuda-xgb608-fullsample-launch-observation-001/manifest.json`. No result yet;2022 remains locked until2021 selection.

## cuda-shop664-uncertainty-2021-001 — 2026-11-02T18:35:07.047846+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/paired_auc_uncertainty.py --run-id cuda-shop664-uncertainty-2021-001 --baseline artifacts/cuda-xgb608-select-2021-001/selected_validation_2021.npz --candidate artifacts/cuda-xgb664-select-2021-001/selected_validation_2021.npz --replicates 200`. Runtime 5.44s. Fixed-model paired28-day circular case bootstrap, seed4173, 200replicates; no fitting, selection, early stopping or Kaggle submission. Difference 6.590593437738601e-05; descriptive95%percentile interval [2.4141718692338723e-05, 0.00010613920195099158]. Adaptive research reuse, training uncertainty and distribution shift are not accounted for. Full source/input/artifact hashes, IDs, dates, versions and bootstrap draws: `artifacts/cuda-shop664-uncertainty-2021-001/manifest.json`.

## cuda-shop664-uncertainty-2022-001 — 2026-11-02T18:35:13.401870+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/paired_auc_uncertainty.py --run-id cuda-shop664-uncertainty-2022-001 --baseline artifacts/cuda-xgb608-select-2022-001/selected_validation_2022.npz --candidate artifacts/cuda-xgb664-select-2022-001/selected_validation_2022.npz --replicates 200`. Runtime 5.47s. Fixed-model paired28-day circular case bootstrap, seed4173, 200replicates; no fitting, selection, early stopping or Kaggle submission. Difference 0.00010172574541877921; descriptive95%percentile interval [3.160866770953453e-05, 0.00019341612259969054]. Adaptive research reuse, training uncertainty and distribution shift are not accounted for. Full source/input/artifact hashes, IDs, dates, versions and bootstrap draws: `artifacts/cuda-shop664-uncertainty-2022-001/manifest.json`.

## cuda-capacity608-uncertainty-2021-001 — 2026-11-02T18:35:49.472332+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/paired_auc_uncertainty.py --run-id cuda-capacity608-uncertainty-2021-001 --baseline artifacts/cuda-xgb608-select-2021-001/selected_validation_2021.npz --candidate artifacts/cuda-xgb608-capacity-select-2021-001/selected_validation_2021.npz --replicates 200`. Runtime 5.47s. Fixed-model paired28-day circular case bootstrap, seed4173, 200replicates; no fitting, selection, early stopping or Kaggle submission. Difference 1.59933178683902e-05; descriptive95%percentile interval [-8.808004144988946e-06, 5.451938657307921e-05]. Adaptive research reuse, training uncertainty and distribution shift are not accounted for. Full source/input/artifact hashes, IDs, dates, versions and bootstrap draws: `artifacts/cuda-capacity608-uncertainty-2021-001/manifest.json`.

## cuda-capacity608-uncertainty-2022-001 — 2026-11-02T18:35:55.817881+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/paired_auc_uncertainty.py --run-id cuda-capacity608-uncertainty-2022-001 --baseline artifacts/cuda-xgb608-select-2022-001/selected_validation_2022.npz --candidate artifacts/cuda-xgb608-capacity-select-2022-001/selected_validation_2022.npz --replicates 200`. Runtime 5.41s. Fixed-model paired28-day circular case bootstrap, seed4173, 200replicates; no fitting, selection, early stopping or Kaggle submission. Difference 1.9119356259422737e-05; descriptive95%percentile interval [-2.8297356931886174e-05, 6.10455564531015e-05]. Adaptive research reuse, training uncertainty and distribution shift are not accounted for. Full source/input/artifact hashes, IDs, dates, versions and bootstrap draws: `artifacts/cuda-capacity608-uncertainty-2022-001/manifest.json`.

## cuda-xgb608-capacity-2022-independent-audit-001 — 2026-11-02T18:30:19.814538+00:00

Independent fixed-weight confirmation passed. The immutable2021 weight0.25 gives2022 microAUC0.9937720670635095 versus accepted anchor0.9937529477072502 and matched same-weight original600-tree control0.9937356307875711. Recomputed original2022 target alignment, independent generic min/max isotonic projection, both allowed0/.25 scores and chosen arrays; all strict-import, source, sample and native GPU config hashes match. No model fit or local checkpoint inference repeated; hosted exact full validation checkpoint replay remains the model inference evidence. Eligible for final fit, with no .998 transfer claim.

Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_cuda_capacity_2022.py --run-id cuda-xgb608-capacity-2022-independent-audit-001 --import-manifest-sha256 b32e45d03db70c4c38271e19c87e7f434d0fdcaf134e06200a57153a1584c56c --selection-manifest-sha256 597844a7a4889a0409ecc90a6f37ba8d8799e72cf8799663e7d225e43cd1296d`. Status: `passed`. Manifest: `artifacts/cuda-xgb608-capacity-2022-independent-audit-001/manifest.json`, SHA256 `250d01fb2c65f80d1a9b8c0bd2fd7758cfd8abb14ff44f49fad9805135ef1f8b`. Environment, immutable sources, data hashes and detailed checks are recorded in this manifest and its artifact closure. No Kaggle submission, score or rank was produced by this run.

## full14-early-oof-result-independent-audit-001 — 2026-11-02T18:31:12.373375+00:00

Independent full14 forward OOF audit passed. Rebuilt exact608 feature arrays from original inputs and verified purged train/evaluation case order,40% samples and2018/2019 labels. Independently loaded and replayed all42 saved350-tree LightGBM checkpoints (33 new and9 exact reused Vinje checkpoints), comparing float32 predictions bitwise for2018=365,2019=359,2020=360 cases. No2020 target rows were decoded. This is a single608LGB mechanism anchor, not the accepted multi-component GPU ensemble. Runtime24.44065s, two threads; source/input/checkpoint hashes and full command preserved in manifest.

Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/verify_full14_early_oof.py --run-id full14-early-oof-result-independent-audit-001 --oof-manifest-sha256 1016522707ec517f9e86dbabfe75686cf9b8d963aade721483d2f49bf0586146`. Status: `passed`. Manifest: `artifacts/full14-early-oof-result-independent-audit-001/manifest.json`, SHA256 `6e8bd061d8e00361f3593f32b5ca666b168242becbc479ca044f3a66bb645ed4`. Environment, immutable sources, data hashes and detailed checks are recorded in this manifest and its artifact closure. No Kaggle submission, score or rank was produced by this run.

## integer-water-budget-full-cache-independent-audit-001 — 2026-11-02T18:30:04.587088+00:00

Preserved reviewer-only failed audit: seven intended feature/cache tests passed, but the audit assertion incorrectly expected ten tests from a broader earlier suite. Candidate source/protocol/cache bytes were unchanged. The incorrect audit source and test log are retained; no solve or model fit occurred.

Exact command: `See retained audit.py and tests.log in artifacts/integer-water-budget-full-cache-independent-audit-001/; reviewer-only audit invocation.`. Status: `failed_reviewer_test_inventory`. Manifest: `artifacts/integer-water-budget-full-cache-independent-audit-001/manifest.json`, SHA256 `d3ef3890b6ad04714a9c3ae1d37eaeeb9779e23f376cb8de258fe6ea9aa371e1`. Environment, immutable sources, data hashes and detailed checks are recorded in this manifest and its artifact closure. No Kaggle submission, score or rank was produced by this run.

## integer-water-budget-full-cache-independent-audit-002 — 2026-11-02T18:31:12.758760+00:00

Preserved reviewer-only failed audit: adding the legacy pilot test module through its package namespace triggered an unqualified helper import failure; all seven intended feature/cache tests still passed. Candidate source/protocol/cache bytes were unchanged. The failing invocation and log are retained; corrected scope is audit003.

Exact command: `See retained audit.py and tests.log in artifacts/integer-water-budget-full-cache-independent-audit-002/; reviewer-only audit invocation.`. Status: `failed_reviewer_test_invocation`. Manifest: `artifacts/integer-water-budget-full-cache-independent-audit-002/manifest.json`, SHA256 `0d3bc95f7c90aeb11a72390314bcd912e3544e0dba5b95630a86cc6aad6143da`. Environment, immutable sources, data hashes and detailed checks are recorded in this manifest and its artifact closure. No Kaggle submission, score or rank was produced by this run.

## integer-water-budget-full-cache-independent-audit-003 — 2026-11-02T18:31:14.715169+00:00

Independent frozen full-cache closure audit passed. Protocolf8aaa986c7afdaa802101f84c9cc33ce89309339efcfd321930431b035e8df9a and3647-input no-solve preflightac4847a25277c36d19513068d42641da0bfacfef825e77be5f9d4e69b18d78e2 pinned exactly. All source/data/control/prefix/pilot hashes and24 pilot completed-case input/replay records verified, with seven targeted tests passing. Reviewed metadata-only labels policy, two isolated workers, exact repeated MILP+fixed-LP physical checks, retry only for native timeout, immutable per-attempt provenance, resumed complete-case validation and complete-cache-only export. No fresh3647-input rebuild or MILP solve repeated by reviewer. Strict compact loader uses the externally pinned aggregate proof index; later model package must require independently audited complete-cache proof. Root-authorized long cache run may proceed under resource coordination.

Exact command: `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python artifacts/integer-water-budget-full-cache-independent-audit-003/audit.py`. Status: `passed`. Manifest: `artifacts/integer-water-budget-full-cache-independent-audit-003/manifest.json`, SHA256 `a01f4acfe7565594fc9fadcfdc889f25709fed2ab5acdc3c94f286d3b83e7fbe`. Environment, immutable sources, data hashes and detailed checks are recorded in this manifest and its artifact closure. No Kaggle submission, score or rank was produced by this run.

## cuda-xgb608-capacity-final-independent-audit-001 — 2026-11-02T18:36:34.692994+00:00

Independent capacity final package audit passed. Frozen protocol840879d339732b14349d5355a886ef2e1b8e31101aa482ec8472df56a7be20d3; private code ZIP2b76efff4319f0798c20c93f3b88942ce54233af48bda10c88d1c8a69f709cd2 and all package/source/acceptance/data pins exact. Independently reproduced original196358-row seed4173 sorted40% final sample, all2922 train/725 test IDs and nested accepted anchor with generic isotonic projection. Reviewed fresh externally pinned GPU0 smoke, native/runtime/source closure, all14×1200-tree fits and121800-row bitwise checkpoint replay, strict artifact allowlists and final formula project_vinje(.75*saved_projected_acceptedGPU608anchor+.25*rawGPUcapacity). Four targeted tests and launcher smoke dry-run passed. No training or submission occurred in this audit.

Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_cuda_capacity_final_package.py`. Status: `passed`. Manifest: `artifacts/cuda-xgb608-capacity-final-independent-audit-001/manifest.json`, SHA256 `c509f81241eee3234ef053030903f1b15808ea090259c1b0cdb907758471d5a0`. Environment, immutable sources, data hashes and detailed checks are recorded in this manifest and its artifact closure. No Kaggle submission, score or rank was produced by this run.

## pooled-pairwise-residual-screen-001 — launch 2026-11-02

Status: running, one local two-thread heavy job (session16383), after independent42-checkpoint OOF audit passed. Fixed residual v2 protocol `dc01041ec5c239d683c9f7f2d9d5efe36ba19b6fa47e9311bf695442c85405a1`; independent preflight `92afd09b8844763811a1e3a016387d63786c996c4920c9fee6b0b89e7a88f8bd`. Train724 forward-OOF cases in2018/2019,1,702,848 case-hour-unit outputs;2020screen360cases remains label-sealed until all three80-tree residual models and screen prediction replay artifacts are sealed. Pooled1M-pair RankNet with deterministic seed4173 and2×diagonal Hessian majorizer versus identical feature/capacity ordinaryBCE and class-balancedBCE controls. Exact interior float64 base logits, endpoint-only1e-12 safeguard, byte-identical zero residual, fixed sigmoid and one Vinje projection. Gate: strict full projected AUC gain versus identity and both controls, each calendar half above identity;5%relative error reduction reported separately as substantial gain. No weights/calibration search or2021/2022 continuation is included.

Exact command: `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/screen_pooled_pairwise_residual.py --run-id pooled-pairwise-residual-screen-001 --protocol-sha256 dc01041ec5c239d683c9f7f2d9d5efe36ba19b6fa47e9311bf695442c85405a1 --oof-manifest-sha256 1016522707ec517f9e86dbabfe75686cf9b8d963aade721483d2f49bf0586146 --oof-audit artifacts/full14-early-oof-result-independent-audit-001/manifest.json --oof-audit-sha256 6e8bd061d8e00361f3593f32b5ca666b168242becbc479ca044f3a66bb645ed4 > artifacts/pooled-pairwise-residual-screen-001.log 2>&1`. Artifacts `artifacts/pooled-pairwise-residual-screen-001`; protocol and OOF proof pin all input/source/split/environment/model details. No result or Kaggle score at launch.

## cuda-xgb608-capacity-final-package-001 — 2026-11-02T18:31:45.188988+00:00

Prepared the independently accepted capacity final recipe, without fitting or submission. Frozen .25 capacity / .75 already projected accepted GPU608 anchor, with one final Vinje projection, passed both chronological folds: 2021 .9950519105980167 and 2022 .9937720670635095. Final training uses the original seed4173 sorted40% sample of all2922×168 rows (196358 rows), exact608 features,14 independent GPU0 XGBoost models,1200 trees/depth8/lr.03 and all remaining accepted parameters unchanged; no early stopping. Fresh genuine synthetic smoke and exact accepted runtime are required. Every121800-row test checkpoint replay must be bitwise exact before export.

Exact command and complete source/data/model/environment/acceptance hashes are in `artifacts/cuda-xgb608-capacity-final-package-001/manifest.json`. Protocol SHA `840879d339732b14349d5355a886ef2e1b8e31101aa482ec8472df56a7be20d3`; PRIVATE code ZIP SHA `2b76efff4319f0798c20c93f3b88942ce54233af48bda10c88d1c8a69f709cd2` (78219 bytes). Nested accepted anchor independently reconstructs exactly. Four targeted tests pass. An initial test-only two-row fixture was incompatible with the168-hour projection shape; its failed source/result is preserved in `artifacts/cuda-xgb608-capacity-final-tests-001-failure/manifest.json`, then only the fixture was corrected to168 rows. No model or protocol was changed by that test correction. The independent package audit is recorded separately.

## cuda-xgb664-final-independent-audit-001 — 2026-11-02

Independent source/package audit passed in2.16s, without fitting. Verified all source/input/package hashes, code-only ZIP exact contents, accepted .25 lock and both fold gates, original196358-row sample, three-stage608+56 feature/source closure, fresh GPU1 smoke/native configuration checks, all14×600 full-test checkpoint replay and strict final exporter allowlists. The accepted nested GPU608 anchor reconstructs exactly. Recomputed standalone projected664 diagnostic CV .9948267633302161/.9936462172415264, versus stronger selected primary .9951018232145258/.9938546734526688; this second predeclared candidate has no automatic promotion. Four targeted tests passed.

Command `.venv-ml/bin/python artifacts/cuda-xgb664-final-independent-audit-001/audit.py`. Proof `artifacts/cuda-xgb664-final-independent-audit-001/manifest.json` SHA `bafb124da77c9bd7436985387dacba0e414faa6c3e5457f4477287856440fffc`; protocol SHA `e571c6607bd797f65c76db5d7617dec7b031774a41af44e65283ccf1ad964ab7`; PRIVATE ZIP SHA `c6cee8e27fa12632c3ea9726b8305f504d3d72fa1f36915cab730838869b54d4`. Full source and test logs retained. No Kaggle score produced.

## cuda-dual-final-package-001 / cuda-two-gpu-final-notebook-001 — 2026-11-02T18:37:59.326748+00:00

Prepared one PRIVATE code dataset containing both independently approved final packages under separate directories, preserving every child file byte-for-byte. Parent manifest SHA `585c6ae68a62e6f118c8f676f232b452a4821a8e8ab3d8f91bef6119acdfdbc2`; ZIP SHA `e3d6878960c983c96e76daa7c3150b3dcff2bba61a75ec923b3f0f08ea3eecd3`,161393 bytes. Command `.venv-ml/bin/python scripts/prepare_cuda_dual_final_notebook.py`. Both child acceptance proofs, package hashes and exact immutable source are recorded in `artifacts/cuda-dual-final-package-001/manifest.json`.

Notebook `artifacts/cuda-two-gpu-final-notebook-001/tdt4173-accepted-dual-final.ipynb` SHA `5ca58debfe2967ad8c92b8b6c8dba8b7fdc83272b65ba87477cdc4d7ec3bc37b`; source SHA `67681a6464f00b16e091d0a88c39595abce89ed90b5ab54d4c54929fc8dfac59`. Preserves the reviewed Python3.13.15/T4×2/package/driver guards, sequential fresh smoke003 runs using one offline environment, separate stage/output directories, GPU1 SHOP first then GPU0 capacity only after first model completion and>=12GiB effective host/cgroup memory. Unknown memory telemetry runs serially. Each final001 result receives its own checksum and archive even if the companion fails. Preparation only; independent wrapper audit and parent scheduling precede execution. No new fit or public score at preparation.

## cuda-xgb664-select-2022-independent-audit-001

Independent locked 2022 audit passed in 0.703 seconds. Only weights zero and the unchanged 2021-selected 0.25 were evaluated, with an independent generic PAVA projection. Selected probabilities are bitwise identical to the saved selected artifact; AUC 0.9938546734526688 exceeds anchor 0.9937529477072502 and matched control 0.9937356307875711. Archive/source/configuration/checkpoint replay proofs, IDs/labels/schema and exact lock were verified. No fitting, weight search or submission.

Command: `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python artifacts/cuda-xgb664-select-2022-independent-audit-001/audit.py`. Full artifact/source/acceptance evidence: `artifacts/cuda-xgb664-select-2022-independent-audit-001/manifest.json`, SHA `f79267d4e8477a4d2d88ee9b1c5fd99240e56bf4837a7a62f3a7f9cf8d0e0df4`.

## cuda-xgb664-final-package-001

Accepted locked development scores: 0.9951018232145258 in 2021 and 0.9938546734526688 in 2022, fixed 25% SHOP664 plus 75% accepted projected GPU608 anchor. Both exceed the declared anchor and same-weight control. The final protocol retains 600 trees, unchanged accepted parameters/features and exact seed-4173 40% full-training sample SHA `4e38d45e87950d4d84849224ddb64fa9b5bf7b21d20a7d9b998223ce5e0fa83d`. Fresh synthetic CUDA1 smoke and all-checkpoint/full-test replay are required. No model training or upload occurred in packaging.

The user-requested standalone projected SHOP664 diagnostic is predeclared from the same checkpoints, without additional weights or automatic promotion. Its development AUC is 0.9948267633302161 / 0.9936462172415264, below the primary in both years. Export filenames are `submission.csv` and `submission_standalone_projected.csv` with separate array/CSV checksums and readback checks.

Four tests passed in 0.643 seconds; exact nested primary anchor reconstruction passed during packaging. Package runtime was 0.719 seconds. Exact command, environment, model/source/data/acceptance closure, archive readback and checksums: `artifacts/cuda-xgb664-final-package-001/manifest.json`, SHA `d443bb51d3f718aa727b5bf365f65604f3fa4b63343e13de728f127560678bc1`. Protocol SHA `e571c6607bd797f65c76db5d7617dec7b031774a41af44e65283ccf1ad964ab7`; PRIVATE archive SHA `c6cee8e27fa12632c3ea9726b8305f504d3d72fa1f36915cab730838869b54d4`, 81,011 bytes. No submission, public score or rank is asserted by this package.

## integer-water-budget-feature-preflight-002

Refreshed source/feature preflight after adding strict complete-cache loading and finite price/dual shape guards. All 24 × 168 × 84 proposed float32 feature values from paired proven pilot attempts agree in dtype and literal bytes. No labels, model fits or solver calls. Runtime 1.241 seconds. Command: `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python artifacts/integer-water-budget-feature-preflight-002/preflight.py`.

Ten related integer, recorder, feature and full-cache helper tests passed in 0.071 seconds using `python -m unittest discover -s tests -p 'test_integer_water_budget*.py'` in the pinned project environment. These include explicit byte sensitivity to dtype and signed zero, bounded retry eligibility and preservation of the 664-column prefix.

Full source snapshots/formulas/input identities and output checksum: `artifacts/integer-water-budget-feature-preflight-002/manifest.json`, SHA `2c73b44a2132aa2754d724072e648ea6f6a67fdbadac945e7fe2ed406f1f52c8`. No learned validation or leaderboard result.

## integer-water-budget-full-cache-preflight-001 — 2026-11-02T18:27:56.400198+00:00

Status passed_preflight_only; 24 of 3647 complete cases. Label-free exact frozen integer water-budget cache, two workers and fresh MILP/conditional-LP replays, 60-second attempts with status-1-only 600-second retries. No relaxed fallback. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/build_integer_water_budget_cache.py --preflight --protocol-sha256 f8aaa986c7afdaa802101f84c9cc33ce89309339efcfd321930431b035e8df9a --run-id integer-water-budget-full-cache-preflight-001`. Source/data/environment/protocol/case/output/failure proof: `artifacts/integer-water-budget-full-cache-preflight-001/manifest.json`. Runtime 45.872 s. No model fit or submission.

## cuda-two-gpu-final-notebook-independent-audit-001 — 2026-11-02T18:39:03.307750+00:00

Independent combined final notebook/package audit passed. Exact immutable notebook and plain source match; the combined PRIVATE ZIP preserves both previously audited child code packages byte-for-byte. Checked accepted0.25 recipes,600-tree GPU1 SHOP664 versus1200-tree GPU0 capacity608, freshsmoke003/final001 IDs, child source/package/runtime pins,196358sample and all14/full-test checkpoint completion checks. Sequential shared-venv smokes, first FINAL progress/20second/effective12GiB stagger and serial fallback retained. Memory/runtime helpers match reviewed002 AST exactly. Isolated helper fixtures verify explicit final smokeSHA commands and separate smoke/final archives even on synthetic per-job failure. Fixture failure records are deliberate temporary tests, not failed GPU runs. Root owns launch; no training/upload/submission in this audit.

Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_cuda_dual_final_notebook.py`. Runtime0.035483s. Full environment, source/input/artifact pins and checks: `artifacts/cuda-two-gpu-final-notebook-independent-audit-001/manifest.json`, SHA256 `992ffa1ec4e642b8cf6bbb11b9b6fbe3622a1f79be90546e6bf031473071fee8`. No Kaggle result.

## cuda-xgb608-capacity-standalone-independent-audit-001 — 2026-11-02T18:39:40.186216+00:00

Independent standalone capacity diagnostic preflight passed. Separate exporter/protocol leave the accepted primary final package unchanged. All source/input hashes and primary acceptance pins verified. Independent generic isotonic projection and average-rank Mann–Whitney AUC reproduce2021 0.9948493578713653 and2022 0.993617994398529 within1e-12, both below primary. Strict completed primary export, unique hosted manifest, all primary artifact hashes, fullrawfloat32 replay, fixed float64 projection, exactCSVschema/IDs and separate immutable outputs are required. No new fit or calibration; actual final arrays and CSV must still be independently audited after models exist. No automatic promotion.

Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python artifacts/cuda-xgb608-capacity-standalone-independent-audit-001/audit.py`. Runtime0.213006s. Full environment, source/input/artifact pins and checks: `artifacts/cuda-xgb608-capacity-standalone-independent-audit-001/manifest.json`, SHA256 `1e86e74392a8b56b1b159c01e7c9f01fffac74490b05f8308fd9cf551b81cf29`. No Kaggle result.

## cuda-xgb608-capacity-standalone-preparation-001 — 2026-11-02

Frozen a separate predeclared standalone projected capacity diagnostic from the same final checkpoints, preserving the sealed primary final package unchanged. Exact formula `project_vinje(rawCapacity608GPU.astype(float64))`; no fit, calibration or weight selection. Recomputed projected development CV .9948493578713653/.993617994398529, below primary .9950519105980167/.9937720670635095. Export requires an explicitly hashed completed strict primary verification, all source/input/artifact closure and raw full-test bitwise replay, then original725×2353 CSV schema/ID/range/readback checks. Separate filenames `submission_standalone_projected.csv` and `standalone_test_predictions.npy`. No automatic promotion or public score at declaration.

Preparation source/snapshot/manifest: `artifacts/cuda-xgb608-capacity-standalone-preparation-001/manifest.json`, SHA `3301f0bb5bcfe8f59441328a40a90677c3a471a09fbf6a4eb400b514b919fd4e`. Protocol `docs/cuda_xgb608_capacity_standalone_protocol.json`, SHA `d76722de6506dbb98ab67daecac47bfb2a771267590768e428f0ddf7f8265033`; exporter SHA `f6a6b0c4a61dfd0461ad91f83298ad106b7529924911d79866d9b2c6ec77a590`. Shadow independently recomputed generic isotonic projection and Mann–Whitney AUC and approved prospective export: `artifacts/cuda-xgb608-capacity-standalone-independent-audit-001/manifest.json`, SHA `1e86e74392a8b56b1b159c01e7c9f01fffac74490b05f8308fd9cf551b81cf29`. Actual final arrays/CSV audit remains required when models exist.


### Accepted dualfinal private upload — 2026-11-02T18:40:37.023395+00:00

Combined161393-byte ZIP preserves both approved childpackages byte-for-byte. Private dataset created and verified in UI; archiveSHA`e3d6878960c983c96e76daa7c3150b3dcff2bba61a75ec923b3f0f08ea3eecd3`. Parent/notebook independently approved, proofSHA`992ffa1ec4e642b8cf6bbb11b9b6fbe3622a1f79be90546e6bf031473071fee8`. Observation `artifacts/cuda-dual-final-upload-observation-001/manifest.json`. Attach/import pending; no finalfit or new publicscore yet.

## pooled-pairwise-residual-result-independent-audit-001 — 2026-11-02

Independent completed-result audit passed in7.786s and confirms the frozen residual screen is rejected. Identity projected pooled AUC .9857960937267796; pairwise .9856508874282199; ordinaryBCE .984827902524293; balancedBCE .9851462602107768. Pairwise beats its matched trained controls but loses to zero-residual identity in the full year and both calendar halves; relative error reduction is negative (about−1.0223%). No2021/2022 continuation or final fit is authorized by this failed gate.

Command `.venv-ml/bin/python scripts/audit_pooled_pairwise_residual_result.py --manifest-sha256 9bca57e1d51c9501a023825235b3075ecd922734118156a9ed962ec7eb95cd43 --run-id pooled-pairwise-residual-result-independent-audit-001`. Independently verified every frozen source/data/input/output hash, original selected360-case2020 targets and724-case2018/19 training IDs, all80 deterministic1M global pair-draw hashes, exact endpoint-only logit arithmetic, byte-exact identity, generic min-max isotonic projection, average-rank Mann–Whitney full/half AUC, strict gates and fit→prediction→target-opening lock order. Independently loaded all three saved80-tree623-feature models and replayed every846720 screen output bitwise, without training. Proof `artifacts/pooled-pairwise-residual-result-independent-audit-001/manifest.json` SHA `6365be7f04a36062f84fc2e462f18bac5039cb0681cc8fcf595c5db6d53d55fa`; exact audit source and environment retained. No public score produced.

## pooled-pairwise-residual-screen-001 — completed 2026-11-02T18:43:28.901377+00:00

Frozenv2 mechanism screen completed and rejected: identity projected2020microAUC0.9857960937267796; pairwise0.9856508874282199; ordinaryBCE0.984827902524293; class-balancedBCE0.9851462602107768. Pairwise loses identity in both calendar halves, relative1−AUC reduction−1.022298%; all advancement checks false. All learned arms improve rawBCE while worseningranking; pairwise within-unitAUC worsens13/14units. No posthoc clipping/temperature/offset/weight/unit search and no2021/2022/final continuation. This single608LGB earlyOOF mechanism screen is not the acceptedGPUensemble. Runtime647.213581s, peakRSS7,194,230,784bytes,2threads. Environment macOS27.0.1arm64/Python3.11.15/numpy2.4.6/pandas3.0.6/scipy1.17.1/sklearn1.9.1/lightgbm4.7.0.

Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/screen_pooled_pairwise_residual.py --run-id pooled-pairwise-residual-screen-001 --protocol-sha256 dc01041ec5c239d683c9f7f2d9d5efe36ba19b6fa47e9311bf695442c85405a1 --oof-manifest-sha256 1016522707ec517f9e86dbabfe75686cf9b8d963aade721483d2f49bf0586146 --oof-audit artifacts/full14-early-oof-result-independent-audit-001/manifest.json --oof-audit-sha256 6e8bd061d8e00361f3593f32b5ca666b168242becbc479ca044f3a66bb645ed4`. Full fixed features/splits/hyperparameters, all source/input/checkpoint/prediction hashes and sealing order in `artifacts/pooled-pairwise-residual-screen-001/manifest.json`, SHA256 `9bca57e1d51c9501a023825235b3075ecd922734118156a9ed962ec7eb95cd43`. Independent all3checkpoint replay/80pairhash/target/projection/AUC/gate audit passed: `artifacts/pooled-pairwise-residual-result-independent-audit-001/manifest.json`, SHA256 `6365be7f04a36062f84fc2e462f18bac5039cb0681cc8fcf595c5db6d53d55fa`. Readable result and next distinct hypothesis: `docs/pooled_pairwise_residual_results.md`. No Kaggle submission or score. Heavy local model slot released.


### Accepted dualfinal GPU launch — 2026-11-02T18:45:34.169229+00:00

Private savedversion8/scriptVersionId364985464; both freshsmoke003 checks passed and runtime/protocol pins verified. SHOP600/664features started18:45:19UTC GPU1; capacity1200/608features GPU0 starts under the reviewed memory/model-progress guard. Original40% samples, seed4173, all training rows eligible by originalfinalprotocol. Nestedfixed.25primary plus separatelypredeclaredstandalone diagnostics; no finalresults yet. Full launch identities: `artifacts/cuda-dual-final-launch-observation-001/manifest.json`.


### Fullsample608 GPU2021 completion — 2026-11-02T18:46:25.859530+00:00

Private savedv7 completed in550.8s. Download size and SHA256 match hosted; strictimport/selection pending. Hostedidentity evidence: `artifacts/cuda-xgb608-fullsample-completion-observation-001/manifest.json`. No leaderboard score or2022 continuation claimed.

## verify-cuda-xgb608-fullsample-2021-return-001 — 2026-11-02T18:46:34.467912+00:00

Status complete; strict2021 GPU return import and frozen selector. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/import_cuda_fullsample_2021_return.py --experiment fullsample608 --archive artifacts/cuda-xgb608-fullsample-2021-001-download.zip --archive-sha256 73069d8816060fcf5d5b9e06f0e010907ce3c3034d5e3c2c017282dec74fd139 --manifest-sha256 270e0b9a76d1d56caa33eeb0746ffce8384fef1f11a9a422eb4a01053c8f564b --smoke-manifest-sha256 da97e314eb85a138941a7b92031523fc238793bba6162434f96dfd79f6718b15 --run-id verify-cuda-xgb608-fullsample-2021-return-001 --selection-run-id cuda-xgb608-fullsample-select-2021-001`. Selected weight 0.25, AUC 0.9950754228706409, anchor 0.9950359172801482, matchedcontrol 0.9950177387934181, eligible2022 True. Runtime 4.049s. All hashes/environment/sources in manifest; no training or Kaggle submission.

## cuda-xgb608-fullsample-select-2021-001 — 2026-11-02T18:46:36.562548+00:00

Status complete; local608-fullsample selection, no model fit or submission. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python /Users/heap/programming/tdt4173/scripts/select_cuda_xgb608_fullsample_blend.py --year 2021 --gpu-run /Users/heap/programming/tdt4173/artifacts/verify-cuda-xgb608-fullsample-2021-return-001/download/cuda-xgb608-fullsample-2021-001 --gpu-manifest-sha256 270e0b9a76d1d56caa33eeb0746ffce8384fef1f11a9a422eb4a01053c8f564b --smoke-manifest /Users/heap/programming/tdt4173/artifacts/verify-cuda-xgb608-fullsample-2021-return-001/download/cuda-xgb608-fullsample-smoke-001/manifest.json --smoke-manifest-sha256 da97e314eb85a138941a7b92031523fc238793bba6162434f96dfd79f6718b15 --protocol /Users/heap/programming/tdt4173/docs/cuda_xgb608_fullsample_protocol.json --run-id cuda-xgb608-fullsample-select-2021-001`. Fixed global weights0/.1/.25/.5 on2021; lock before2022. Selected 0.25; microAUC 0.9950754228706409, anchor 0.9950359172801482, matched608 control 0.9950177387934181. Full provenance `artifacts/cuda-xgb608-fullsample-select-2021-001/manifest.json`; runtime 1.868s.

## cuda-xgb608-fullsample-2022-notebook-001 — 2026-11-02T18:47:44.109795+00:00

Prepared the conditional2022 confirmation notebook from the actual independently audited eligible2021 .25 lock. Lock SHA `d5fe5b3c2a6c64b0f9a35405411b629aff1698906c876e2a7f99fb73fc4d5e83`; independent selection proof SHA `ff945b0e24ff0710ba2d651a60ff5524046386d7538f5df8d1be0ff4de8ebbda`. Exact reviewed2021 wrapper adapted only for freshsmoke002,2022 scope/all428568 purged rows, embedded immutable lock/audit bytes and explicit locked selection CLI plus returned-lock identity. Original600/depth6/.05/GPU1 recipe unchanged, no early stopping or weight search. Notebook `artifacts/cuda-xgb608-fullsample-2022-notebook-001/tdt4173-fullsample608-frozen-2022.ipynb` SHA `c27c57e5c6b33e87bf0aa8c02d4bcab072ac5ca1852236bd33dea4244e385d67`; plaintext SHA `8e03c0b6375ad643d9201bce95f4adf4cd0a4393463aa56f049e6b9e72a6f1d4`. Complete command/source/inputs in preparation manifest. Independent wrapper review and parent scheduling precede execution; no2022 result known.

## cuda-xgb608-fullsample-final-template-preparation-001 — 2026-11-02

Prepared unsealed final trainer/exporter/launcher/packager templates while the locked2022 confirmation is pending. No protocol, package, fit or submission was created. Fullsample final would use every2922×168=490896 canonical training row in order, original600/depth6/.05/GPU1 parameters and unchanged per-tree subsample.9; all-row NPY SHA `faba3c7f9b70486171bd0f3704fc6c558d05f359d4f8aa0160a0dfbc3d89ad01`. Original40% final sample is a verified subset. Builder requires actual accepted2022 independent proof, externally pinned return/selector and reference smoke identities before sealing. Four targeted tests pass in.617s: all-row sample, nested fixed blend, real fresh smoke/tampering and duplicate code discovery. Complete preparation source, template snapshots and test observation are preserved under `artifacts/cuda-xgb608-fullsample-final-template-preparation-001`. No future acceptance or public performance is implied.

## cuda-fullsample-2021-return-preflight-001 / cuda-fullsample-2022-return-preflight-001 — 2026-11-02

Prepared strict fullsample return orchestration by changing only the experiment recipe and source snapshot filename in the already reviewed dual2021/2022 importers;2021 additionally checks selection-output nonexistence before import. Frozen fullsample trainer, return verifier, selector, protocol and projection hashes are checked before safe externally hashed archive import. The2022 path additionally requires the genuine independently eligible2021 lock/audit, invokes the original trainer lock validator, and checks that only zero and the locked weight were evaluated. No training or remote action occurs. Source snapshots and exact template/source hashes are in the two preflight manifests.2021 importer SHA `a2ca66daaf94e500560c5e80dff687df5dc4e9841e0461ae6ace7507dfa6670c`;2022 importer SHA `c02a39cee726bce735117f56bb7fa25f2f38c89a1a9a3bdac702db44fe2c80cc`. The actual2021 verification/selection is recorded separately;2022 awaits hosted results.


### Fullsample608 locked2022 concurrent GPU launch — 2026-11-02T18:53:31.587396+00:00

Private savedv9/scriptVersionId364987203 runs besidev8, bothvisiblyRunning; no priorrun interrupted. Official Kaggle efficientGPUdocs supportconcurrent savedbatchsessions. Frozen.25lock, original600/depth6/.05, all428568purgedrows, freshsmoke002 verified. SHA/lock/independentproofandactualstart in`artifacts/cuda-xgb608-fullsample-2022-launch-observation-001/manifest.json`. No result yet.

## linear-leaf-residual-independent-audit-001 — 2026-11-02

Independent frozen preflight passed in1.841s, without competition fitting or2020 label decoding. Verified exact protocol `ad6197db7076990ff111d6bba68bbb596687eed93d028713f782aaec8da5e4db`, preflight003 `12887922d06ae709262191b6c270ea2cd0103f055a2ce0702c7ed7a8631a3a4c`, all source/data/input/OOF/Pro-consultation pins and preserved earlier failure provenance. Independently reconstructed chronological2018/2019/2020 case IDs, all14 pre2018 gain-ranked32-feature selections with raw/LightGBM-name mapping, original shared48652-row seed4173 sample, and all14 sample-only mean/population-std normalizers. Six focused numerical/scope/resource tests passed, including actual linear leaf slopes, init-score offset correctness and saved checkpoint replay.

Reviewed28 matched40-round constant/linear unit fits with exact BCE offsets, all model/normalizer seals before screen inference and all28 checkpoint replays/raw prediction locks before selected2020 target decoding. Strict whole-screen and both-half gains against identity and constant controls required;5%relative error reduction is a separate flag. Root's600s/4GiB operational guard preserves inconclusive runs; no adaptive retry or later-year fit. Approved for the already authorized bounded screen. Command `.venv-ml/bin/python artifacts/linear-leaf-residual-independent-audit-001/audit.py`. Proof `artifacts/linear-leaf-residual-independent-audit-001/manifest.json` SHA `ae4ad2d7b8cb67de22df2bcd255dfb269fba076aaa77b4224de02b6fd4c16473`; exact audit source/environment/test log retained. No Kaggle score produced.

## cuda-xgb608-fullsample-2021-independent-audit-001 — 2026-11-02T18:47:21.007491+00:00

Independent fullsample2021 selection passed: fixed grid0/.1/.25/.5 =0.9950359172801482/0.9950579688726214/0.9950754228706409/0.9950691414617967, selected0.25 beats anchor and same-weight original600-tree control0.9950177387934181. Exact367248 purged training row NPY bytes, original2021 labels, all artifact/source/config hashes, independent min-max isotonic arrays/grid/gate checked. Eligible2022 lock d5fe5b3c2a6c64b0f9a35405411b629aff1698906c876e2a7f99fb73fc4d5e83; no2022 evaluation or fit in this audit.

Exact command: `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python artifacts/cuda-xgb608-fullsample-2021-independent-audit-001/audit.py`. Full environment/source/data/artifact closure in `artifacts/cuda-xgb608-fullsample-2021-independent-audit-001/manifest.json`, SHA256 `ff945b0e24ff0710ba2d651a60ff5524046386d7538f5df8d1be0ff4de8ebbda`. No Kaggle submission or score.

## linear-leaf-residual-preflight-001 — 2026-11-02T18:50:04.606636+00:00

Preserved preflight failure before any competition fit: LightGBM deepcopies callable objective instances, so the original BCE state trace remained empty. Corrected by a function closure preserving the state, as in the previous audited runner. Three synthetic tests ran, one failed; source and error record retained.

Exact command: `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python -m unittest tests.test_linear_leaf_residual`. Full environment/source/data/artifact closure in `artifacts/linear-leaf-residual-preflight-001/manifest.json`, SHA256 `da7e3421014b6a84905bf9d5a638939f0f5c746bfb7f8be4ae3f6d6cd00057ec`. No Kaggle submission or score.

## linear-leaf-residual-preflight-002 — 2026-11-02T18:54:28.561336+00:00

Preserved preflight failure before any competition fit: raw608 feature names contain spaces for Vesle Kjela while LightGBM serializes underscores. Exact608-column mapping was independently confirmed; corrected guard verifies space-to-underscore mapping and records both names, preserving feature order/bytes. Source and error record retained.

Exact command: `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/prepare_linear_leaf_screen.py`. Full environment/source/data/artifact closure in `artifacts/linear-leaf-residual-preflight-002/manifest.json`, SHA256 `03834d2261ae46ad8dbb6d1146a7a6f0a078d3ab2c8daba60c84f6e445a58ef9`. No Kaggle submission or score.

## linear-leaf-residual-preflight-003 — 2026-11-02T18:54:30.592088+00:00

Frozen linear-leaf screen prepared without competition fitting. Pro consultation94b4557e… pinned; perunit32gain features from strictly pre2018 checkpoints and shared48652-hour40% residualsample fixed. Six meaningful tests passed: offset/actuallinearcoefficients/replay, gain ties/sample, train-onlyNaN scaling, operationalresourcecheck and targetpoison/alignment. Protocol ad6197db7076990ff111d6bba68bbb596687eed93d028713f782aaec8da5e4db; independent preflight ae4ad2d7b8cb67de22df2bcd255dfb269fba076aaa77b4224de02b6fd4c16473 subsequently approved bounded training.

Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/prepare_linear_leaf_screen.py`. Full environment/source/data/artifact closure in `artifacts/linear-leaf-residual-preflight-003/manifest.json`, SHA256 `12887922d06ae709262191b6c270ea2cd0103f055a2ce0702c7ed7a8631a3a4c`. No Kaggle submission or score.

## linear-leaf-residual-screen-001 — 2026-11-02T18:56:14.529257+00:00

Completed and rejected,13.530815s,peakRSS760414208bytes,two threads. All28 fixed40-round models and full saved-checkpoint replays completed within600s/4GiB limits. Projected2020 identityAUC0.9857960937267796; matchedconstant0.9846910396792928; linear0.9329727403913785. Linear loses bothcontrols overall andbothcalendarhalves; rawBCE0.70793245 and residualmax80.3847 show extreme outputs. All14 perunitAUCs worsen; Kjela−0.36045 is largest. No adaptive retry, clipping/weight/unit selection,2021continuation or finalexport. Independent result review requested.

Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/screen_linear_leaf_residual.py --run-id linear-leaf-residual-screen-001 --protocol-sha256 ad6197db7076990ff111d6bba68bbb596687eed93d028713f782aaec8da5e4db --preflight-audit artifacts/linear-leaf-residual-independent-audit-001/manifest.json --preflight-audit-sha256 ae4ad2d7b8cb67de22df2bcd255dfb269fba076aaa77b4224de02b6fd4c16473`. Protocol ad6197db7076990ff111d6bba68bbb596687eed93d028713f782aaec8da5e4db; independent preflight ae4ad2d7b8cb67de22df2bcd255dfb269fba076aaa77b4224de02b6fd4c16473. Completed manifest `artifacts/linear-leaf-residual-screen-001/manifest.json`, SHA256 `141204f193e03bd36fa36126aed4eb98d12d5f7a596b4d223cf5f8439cf59719`, retains exact allsource/input/splits/features/normalizers/models/traces/environment and prelabel model/prediction locks. Readable recipe/results: `docs/linear_leaf_residual_results.md`. Single608LGB earlyOOF mechanism screen on already reused2020 development labels, not acceptedGPUensemble; no Kaggle score.

## linear-leaf-residual-result-independent-audit-001 — 2026-11-02

Independent result audit passed in2.374s and confirms the fixed linear-leaf hypothesis is rejected. Projected2020 microAUC identity .9857960937267796, matchedconstant .9846910396792928, linear .9329727403913785; linear loses the full screen and both halves against both controls. No rescue parameter search or later-year continuation. Independently verified complete source/input/output closure, chronological fit/prediction/label-opening locks, original2020 targets, all48652 sampled rows, all14 actual normalizers and feature indices. Replayed all28 saved40-tree models over every2020 output bitwise, checked actual serialized slopes, reconstructed `sigmoid(base_logit + saved_raw_residual)` with the offset added exactly once, generic isotonic projection and average-rank AUC/halves. The actual fit source asserts first objective scores equal frozen init_score in every model; synthetic offset/serialization tests already passed. No numerical or schema mismatch found.

Command `.venv-ml/bin/python scripts/audit_linear_leaf_residual_result.py --manifest-sha256 141204f193e03bd36fa36126aed4eb98d12d5f7a596b4d223cf5f8439cf59719 --run-id linear-leaf-residual-result-independent-audit-001`. Proof `artifacts/linear-leaf-residual-result-independent-audit-001/manifest.json` SHA `c1cd3fb0f0e0904ec6cfab2b07725aed4f2a94bdcec0c48b2c5c1659ba6524bb`. Complete audit source/helper, environment and results retained. Saved residual magnitudes suggest instability; this descriptive observation is not a retuning decision. No Kaggle result produced.

## submission-cuda-xgb608-capacity-anchor-blend-001 — 2026-11-02T18:59:32.371036+00:00

Status:failed. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/export_cuda_xgb608_capacity_final.py --protocol docs/cuda_xgb608_capacity_final_protocol.json --protocol-sha256 840879d339732b14349d5355a886ef2e1b8e31101aa482ec8472df56a7be20d3 --archive artifacts/cuda-xgb608-capacity-final-001-download.zip --archive-sha256 f21f3016326f22c2d9b6878c619909821cce68ed4e3e6840f05ecb439385c616 --manifest-sha256 80c3586e474ac0e87054c06e73f3dfe965f9e9a9aefd4faa4f050c8e9e9483972 --gpu-run-name cuda-xgb608-capacity-final-001 --smoke-run-name cuda-xgb608-capacity-smoke-003 --smoke-manifest-sha256 0d138cc7d7d1ae67b548e7cef768101ee4f0f97ca304492328c6b1baa741a279 --run-id submission-cuda-xgb608-capacity-anchor-blend-001`. Strict finalGPU artifact/sample/config/checkpoint/fulltestreplay verification and exact nested75%acceptedGPU608anchor/25%capacityGPU blend export. No training or weightretuning. SubmissionSHA:None. Full provenance:`artifacts/submission-cuda-xgb608-capacity-anchor-blend-001/manifest.json`. NoKaggle result at export.


### Accepted dualfinal hosted completion and downloads — 2026-11-02T18:59:46.585668+00:00

Privatev8 completed730.1s; bothfinal archives downloadedwith exact hostedbytecounts/SHA256. Allhostedfinalmanifest/freshsmoke/archivenames/hashes are in`artifacts/cuda-dual-final-download-verification-001/manifest.json`; originalhosted observation remainsimmutable. Strictprimary+predeclaredstandaloneexports andindependentCSVchecks pending. No newpublicscore yet.

## submission-cuda-xgb608-capacity-anchor-blend-002 — 2026-11-02T19:00:12.699691+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/export_cuda_xgb608_capacity_final.py --protocol docs/cuda_xgb608_capacity_final_protocol.json --protocol-sha256 840879d339732b14349d5355a886ef2e1b8e31101aa482ec8472df56a7be20d3 --archive artifacts/cuda-xgb608-capacity-final-001-download.zip --archive-sha256 f21f3016326f22c2d9b6878c619909821cce68ed4e3e6840f05ecb439385c616 --manifest-sha256 80c3586e474ac0e87054c06e73f3dfe965f9e9a9efd4faa4f050c8e9e9483972 --gpu-run-name cuda-xgb608-capacity-final-001 --smoke-run-name cuda-xgb608-capacity-smoke-003 --smoke-manifest-sha256 0d138cc7d7d1ae67b548e7cef768101ee4f0f97ca304492328c6b1baa741a279 --run-id submission-cuda-xgb608-capacity-anchor-blend-002`. Strict finalGPU artifact/sample/config/checkpoint/fulltestreplay verification and exact nested75%acceptedGPU608anchor/25%capacityGPU blend export. No training or weightretuning. SubmissionSHA:b7d42b0752ed0de639337b93ac57dc05498f478bea2f3217e86d16917a34cc47. Full provenance:`artifacts/submission-cuda-xgb608-capacity-anchor-blend-002/manifest.json`. NoKaggle result at export.

## submission-cuda-xgb664-anchor-blend-001 — 2026-11-02T19:00:12.700917+00:00

Status complete. Strict SHOP664 final artifact/sample/config/checkpoint/replay verification and fixed nested 75% accepted GPU608 anchor / 25% SHOP664 export. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/export_cuda_xgb664_final.py --protocol docs/cuda_xgb664_final_protocol.json --protocol-sha256 e571c6607bd797f65c76db5d7617dec7b031774a41af44e65283ccf1ad964ab7 --archive artifacts/cuda-xgb664-final-001-download.zip --archive-sha256 ea8fdec6dad3baaba01b034e41ef12b8a296f98b939aa490784d9119c2be15e7 --manifest-sha256 c914fc7f3b2c739e162eea8e9809ee0399713a378e036d8a175bc58e3e9cdec3 --gpu-run-name cuda-xgb664-final-001 --smoke-run-name cuda-xgb664-smoke-003 --smoke-manifest-sha256 0d405c8f3dd3554053f0208b50ad60840f082119c467d2e3b40ff84f0a471aab --run-id submission-cuda-xgb664-anchor-blend-001`. Submission SHA 898d29699cda7aa93cc5b22e410d63eb7fa371ce4afc9f0e9b5a176993e9e96a. No training, weight retuning or Kaggle claim; full evidence in `artifacts/submission-cuda-xgb664-anchor-blend-001/manifest.json`.

## submission-cuda-xgb608-capacity-standalone-001 — 2026-11-02T19:00:26.752136+00:00

Status complete. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/export_cuda_xgb608_capacity_standalone.py --protocol docs/cuda_xgb608_capacity_standalone_protocol.json --protocol-sha256 d76722de6506dbb98ab67daecac47bfb2a771267590768e428f0ddf7f8265033 --verified-blend-run artifacts/submission-cuda-xgb608-capacity-anchor-blend-002 --verified-blend-manifest-sha256 51758f020ddd0faa483909277da327a8f928f6b647d6741351b84f3130f3c4f0 --run-id submission-cuda-xgb608-capacity-standalone-001`. Predeclared rawCapacity608GPU plus fixedVinjeprojection; no weight search/training. Validation {'2021': 0.9948493578713653, '2022': 0.993617994398529}. CSV SHA 85ac35d8d5c8ba460e034b3f43b0d5bfc3c461906e97daa579a5a290c167acd7. Full provenance in manifest; runtime 1.817s. No Kaggle score at export.

Capacity export001 failure clarification: its caller supplied a65-character final-manifest hash with one duplicated `a`, so the immutable strict verifier rejected before producing any CSV. The downloaded archive and actual embedded final manifest exactly matched the independently observed hosted hashes. Preserved failed001 artifacts unchanged; retried the same frozen exporter under002 with the original correct64-character hosted manifest SHA, which passed all checks. This was a CLI transcription error, with no code/model/data/recipe change.

Successful final exports: capacity primary manifest `51758f020ddd0faa483909277da327a8f928f6b647d6741351b84f3130f3c4f0`, CSV `b7d42b0752ed0de639337b93ac57dc05498f478bea2f3217e86d16917a34cc47`; capacity standalone manifest `533c705727c954b6f288d2cbe9ebf0a1b261e11e6813387e936967e55314cd04`, CSV `85ac35d8d5c8ba460e034b3f43b0d5bfc3c461906e97daa579a5a290c167acd7`; SHOP primary+diagnostic manifest `dcd1617cb60aa29dce9c9a64bff16fbc25f00ef569912ea1b80c21eea5aa973c`, primary CSV `898d29699cda7aa93cc5b22e410d63eb7fa371ce4afc9f0e9b5a176993e9e96a`, diagnostic CSV `4c96e9f69459417fd138af7d33f5339275bb5ef29ffdd59f0a4590ee8f60ae42`. Native configuration, exact original final sample, source/bundle closure, all28 checkpoints' complete121800-row hosted replays, nested saved anchor arithmetic, schema/IDs/probability range and CSV readback passed. Independent actual CSV audit requested before parent submits; no public score at this point.


### SHOP664 primary public improvement — 2026-11-02T19:03:21.197613+00:00

FrozenSHOP.25primary scored **0.99651,rank18**, versus previous0.99640/rank20. Both Complete submission description/SHA and leaderboard independently observed; user separatelyconfirmed. CSV SHA`898d29699cda7aa93cc5b22e410d63eb7fa371ce4afc9f0e9b5a176993e9e96a`; exactreplayandall4CSVindependentauditpassed. Observation saved separately in`artifacts/submission-cuda-xgb664-anchor-blend-001/kaggle_observation.json`; immutableexportmanifest unchanged. Primary promoted aspublicbest; no finalselectionchange; top3/above.998andvalidationtransfergates stillunmet.


### Capacity608 primary public result — 2026-11-02T19:05:57.040353+00:00

Frozen.25capacityprimary scored0.99644 (Complete exactchecksumrowobserved), belowcurrentSHOPbest0.99651. Notpromoted. CSV`b7d42b0752ed0de639337b93ac57dc05498f478bea2f3217e86d16917a34cc47`; separateobservation`artifacts/submission-cuda-xgb608-capacity-anchor-blend-002/kaggle_observation.json`; immutableexportmanifest unchanged. Nofinalselectionchange.

## cuda-dual-final-submissions-independent-audit-001 — 2026-11-02T19:00:47.295431+00:00

Allfour actual final submissions independently passed. Verified external strict export manifests, full artifact/hosted checkpoint replay closure and saved acceptedGPU608 anchor. Reconstructed both fixed75%projectedanchor/25%raw candidates and both predeclared rawprojected diagnostics; independent generic min-max isotonic result equals production arrays with maxdifference0. All savedfloat64array bytes and CSV bytes exact,725IDs/2353columns/range/serializationerror≤5.000001e-10. Capacity primary is immutable002 because001 preserved a CLI hash transcription failure; reviewed exporter unchanged. SHOP primary SHA898d29699cda7aa93cc5b22e410d63eb7fa371ce4afc9f0e9b5a176993e9e96a, standalone4c96e9f69459417fd138af7d33f5339275bb5ef29ffdd59f0a4590ee8f60ae42. Capacity primarySHAb7d42b0752ed0de639337b93ac57dc05498f478bea2f3217e86d16917a34cc47, standalone85ac35d8d5c8ba460e034b3f43b0d5bfc3c461906e97daa579a5a290c167acd7. Ready for root-authorized submissions; no public score measured by reviewer.

Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_cuda_dual_final_submissions.py --run-id cuda-dual-final-submissions-independent-audit-001 --shop-manifest-sha256 dcd1617cb60aa29dce9c9a64bff16fbc25f00ef569912ea1b80c21eea5aa973c --capacity-manifest-sha256 51758f020ddd0faa483909277da327a8f928f6b647d6741351b84f3130f3c4f0 --capacity-diagnostic-manifest-sha256 533c705727c954b6f288d2cbe9ebf0a1b261e11e6813387e936967e55314cd04`. Runtime5.718327s. Full environment/sources/input/checks in `artifacts/cuda-dual-final-submissions-independent-audit-001/manifest.json`, SHA256 `ff88843e5eb8aa1d351ceb36b0640afb08dfc82d2490ecba294e3dacf272aa7d`. No training or Kaggle score produced by this independent audit.

## generator-offset-independent-audit-001 — 2026-11-02T19:03:15.614223+00:00

Independent preflight approved frozen12-group boundedoffset experiment, retaining prior same-family Round9 negative history. Allsource/data/input pins, actual359purged2018→2019 and7242018/19→2020 scopes, interiorOOFprobabilities and five derivative/fullHessian/replay/rank/purge tests verified. Endpoints preserved by analyticodds transform, Vinje3 sharebias afterfixedprojection, allwithinunit ranks muststayexact. Fixed1M uniformpooledpairs, bounds±.5,lambda.01mean-square ridge. Meanpairloss1/M and balancedmeanBCE do not have identical gradient scales/effective regularization; compare frozenrecipes without claiming purelossisolation. Only externallypinned passed2019 permits unchanged2020.

Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python artifacts/generator-offset-independent-audit-001/audit.py`. Runtime1.028809s. Full environment/sources/input/checks in `artifacts/generator-offset-independent-audit-001/manifest.json`, SHA256 `fde8531dc7fed16f777c317e62fb616076a03894f03dcbef0f457f90c4328254`. No training or Kaggle score produced by this independent audit.

## generator-offset-2019-independent-audit-001 — 2026-11-02T19:05:33.461950+00:00

Independent result audit passed confirming REJECTION. Pairwise full2019AUC0.9930746746206733 beats identity0.992950521413564 and balancedBCE0.9927988094401643, but calendarhalf gains+0.0004877533/−0.0001821353 fail the fixedsecondhalfgate. Allsource/input/outputhashes, original359purged2018train/3592019screen labels, exact1Mpairdraw, independent convexobjective/boxKKT stationarity for bothmodels, genericprojection/logistictransform, exactwithinunitrank/ties and rank-sum full/halfAUC verified. Fit/predictionlock precedes targetdecode. No2020read or optimizer rerun by reviewer; no2020continuation orsubmission.

Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python artifacts/generator-offset-2019-independent-audit-001/audit.py`. Runtime1.144605s. Full environment/sources/input/checks in `artifacts/generator-offset-2019-independent-audit-001/manifest.json`, SHA256 `ad1c4a99c23070a8de6e3b62fdea7fc349ed4d9c464a074c36a4ad311ed95445`. No training or Kaggle score produced by this independent audit.

## cuda-xgb664-fullsample-template-preparation-001-failure / -002 — 2026-11-02

Prepared the root-authorized unsealed SHOP664 full-row interaction hypothesis. Original664600/depth6/.05/GPU1 model and all three frozen input bundles unchanged; only external row inclusion changes.4→1.0. Anchor is acceptedSHOP.25 blend, matched control original40% rawSHOP, fixed2021 grid0/.1/.25/.5 with lower-weight ties and strict anchor/control gate;2022 only locked. No protocol/package/fit created: builder requires actual independently accepted fullsample6082022 proof and selector closure first.

An initial preparation generator failed while looking for a nonexistent comment insertion anchor after producing three unsealed source templates. Its source, partial outputs and explicit failure record are preserved under `artifacts/cuda-xgb664-fullsample-template-preparation-001-failure`; no training or protocol was affected. Corrected the generator's insertion anchor and module imports, completed five isolated sources, documented the hypothesis and consistently updated sample count metadata. Four tests pass in.073s for exact original params/grid, full367248/428568 purged rows and original subset, nativeGPU1/664feature/600tree configuration and matchedcontrol lock rejection. Complete new source snapshots/environment/test observation in `artifacts/cuda-xgb664-fullsample-template-preparation-002/manifest.json`. Readable pending recipe: `docs/cuda_xgb664_fullsample.md`. No leaderboard score or future acceptance claimed.


### SHOP664 standalone public diagnostic — 2026-11-02T19:07:08.881645+00:00

Predeclared projectedstandalone scored0.99647 (Complete exactchecksumrowobserved), belowprimary.99651; notpromoted. Noextraweightsearchortraining. CSV SHA`4c96e9f69459417fd138af7d33f5339275bb5ef29ffdd59f0a4590ee8f60ae42`; separateobservation`artifacts/submission-cuda-xgb664-anchor-blend-001/kaggle_observation_standalone.json`; exportmanifest unchanged.


### Capacity608 standalone public diagnostic — 2026-11-02T19:07:37.083736+00:00

Predeclared projectedstandalone scored0.99637 (Complete exactchecksumrowobserved), belowprimarycapacity.99644andbestSHOP.99651; notpromoted. Allfourpredeclareddualfinalsubmissionscomplete,86submissionquotaremainingfromprior90. Noextraweightsearch/finalselectionchange. CSV SHA`85ac35d8d5c8ba460e034b3f43b0d5bfc3c461906e97daa579a5a290c167acd7`; separateobservation`artifacts/submission-cuda-xgb608-capacity-standalone-001/kaggle_observation.json`; exportmanifest unchanged.

## linear-leaf-extrapolation-diagnostic-001 — 2026-11-02

Read-only descriptive diagnosis after independent rejection, without decoding labels or fitting. Verified saved result/proof, frozen feature arrays and actual normalizers, then compared2020 inputs to the exact sampled2018/2019 feature ranges. Kjela's maximum residual+80.384697 occurs at normalizedweeklyprice .5427576 versus sampled training range [.9617066,.9847315], about93.99 training standard deviations beyond that interval; maximum screen absolute z100.35 versus training20.27. All28604 Kjela outputs with|residual|>10 have at least one feature outside its empirical training range. VesleKjela44 and TokkeG1 one such extreme residual also occur outside ranges. This supports exposure to severe linear extrapolation, not causal proof, and follows verified correct init-score/once-only inference offset. No clipping, reweighting, rescue fit or continuation was performed.

Command `.venv-ml/bin/python artifacts/linear-leaf-extrapolation-diagnostic-001/analyze.py`. Complete perunit diagnostics/source/environment in `artifacts/linear-leaf-extrapolation-diagnostic-001/manifest.json`, SHA `03f20919eabeff2e427e8b17d2ac8660efbc63b7a0c7a453b2bbfb2488043f20`. Original fixed screen rejection unchanged.

## cuda-xgb664-fullsample-template-independent-audit-001 — 2026-11-02T19:08:34.550454+00:00

Unsealed conditional SHOP664 fullsample source templates independently reviewed,4 targeted tests passed in1.01s. Exact original664model parameters/feature loaders/threeinputbundles preserved; only external row fraction.4→1.0 and fresh namespaces change. Actual367248/428568 purged row construction and originalsubset tested, copiedfoldcountmetadata made consistent, fixed2021globalgrid and locked2022/matched40%rawSHOP control preserved. Builder requires independently accepted fullsample6082022 and exactselectorclosure before any protocol/package seal. Actual future protocol/package/anchor hashclosure still needs review; no training, upload or candidate approval yet.

Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python artifacts/cuda-xgb664-fullsample-template-independent-audit-001/audit.py`. Proof `artifacts/cuda-xgb664-fullsample-template-independent-audit-001/manifest.json`, SHA256 `47b759509a7d20bc9148a0ca89759d6afb385e0aa311d2d14635eaa5c505d5ae`. Full source snapshot, environment, checks and test log retained. No Kaggle result.


### Fullsample608 frozen2022 GPU completion and suite verification — 2026-11-02T19:10:34.854833+00:00

Privatev9completed683.1s; downloadexact12788044bytesandhostedSHA256. Trustedhashesin`artifacts/cuda-xgb608-fullsample-2022-completion-observation-001/manifest.json`; strictimport/acceptancepending. Independently,thelocalfullunittestsuitepassed249tests/11.577s; exactcommand/log/sourcehashes/environmentin`artifacts/verification-suite-20261102-1907/manifest.json`. No weights or data changed for either observation.

## verify-cuda-xgb608-fullsample-2022-return-001 — 2026-11-02T19:10:56.888179+00:00

Status complete. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/import_cuda_fullsample_2022_return.py --experiment fullsample608 --archive artifacts/cuda-xgb608-fullsample-2022-001-download.zip --archive-sha256 ecfe1c8a3c28120d3044c07a51c545ae31cf5bb827924a6469005b038c86b212 --manifest-sha256 1ad301a4bc9d7a987979ed08bc3f45c5e6962e676d815dba60cbf7f5e61a9315 --smoke-manifest-sha256 78d64b9bbd3b63285c3e0c1527636403d859e988f36f5139c9b70edbd3395ec7 --selection-lock artifacts/cuda-xgb608-fullsample-select-2021-001/selection_2021.json --selection-lock-sha256 d5fe5b3c2a6c64b0f9a35405411b629aff1698906c876e2a7f99fb73fc4d5e83 --lock-audit artifacts/cuda-xgb608-fullsample-2021-independent-audit-001/manifest.json --lock-audit-sha256 ff945b0e24ff0710ba2d651a60ff5524046386d7538f5df8d1be0ff4de8ebbda --run-id verify-cuda-xgb608-fullsample-2022-return-001 --selection-run-id cuda-xgb608-fullsample-select-2022-001`. Strict2022 GPU import and unchanged2021 weight evaluation; weight 0.25, AUC 0.9937685726056958, anchor 0.9937529477072502, control 0.9937356307875711, accepted True. Runtime 3.661s; all hashes, environment and source snapshots in the manifest. No fit or Kaggle action.

## verify-cuda-xgb608-fullsample-2022-return-001 — 2026-11-02T19:10:56.888179+00:00

Status complete. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/import_cuda_fullsample_2022_return.py --experiment fullsample608 --archive artifacts/cuda-xgb608-fullsample-2022-001-download.zip --archive-sha256 ecfe1c8a3c28120d3044c07a51c545ae31cf5bb827924a6469005b038c86b212 --manifest-sha256 1ad301a4bc9d7a987979ed08bc3f45c5e6962e676d815dba60cbf7f5e61a9315 --smoke-manifest-sha256 78d64b9bbd3b63285c3e0c1527636403d859e988f36f5139c9b70edbd3395ec7 --selection-lock artifacts/cuda-xgb608-fullsample-select-2021-001/selection_2021.json --selection-lock-sha256 d5fe5b3c2a6c64b0f9a35405411b629aff1698906c876e2a7f99fb73fc4d5e83 --lock-audit artifacts/cuda-xgb608-fullsample-2021-independent-audit-001/manifest.json --lock-audit-sha256 ff945b0e24ff0710ba2d651a60ff5524046386d7538f5df8d1be0ff4de8ebbda --run-id verify-cuda-xgb608-fullsample-2022-return-001 --selection-run-id cuda-xgb608-fullsample-select-2022-001`. Strict2022 GPU import and unchanged2021 weight evaluation; weight 0.25, AUC 0.9937685726056958, anchor 0.9937529477072502, control 0.9937356307875711, accepted True. Runtime 3.661s; all hashes, environment and source snapshots in the manifest. No fit or Kaggle action.

## cuda-xgb608-fullsample-select-2022-001 — 2026-11-02T19:10:58.822339+00:00

Status complete; local608-fullsample selection, no model fit or submission. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python /Users/heap/programming/tdt4173/scripts/select_cuda_xgb608_fullsample_blend.py --year 2022 --gpu-run /Users/heap/programming/tdt4173/artifacts/verify-cuda-xgb608-fullsample-2022-return-001/download/cuda-xgb608-fullsample-2022-001 --gpu-manifest-sha256 1ad301a4bc9d7a987979ed08bc3f45c5e6962e676d815dba60cbf7f5e61a9315 --smoke-manifest /Users/heap/programming/tdt4173/artifacts/verify-cuda-xgb608-fullsample-2022-return-001/download/cuda-xgb608-fullsample-smoke-002/manifest.json --smoke-manifest-sha256 78d64b9bbd3b63285c3e0c1527636403d859e988f36f5139c9b70edbd3395ec7 --protocol /Users/heap/programming/tdt4173/docs/cuda_xgb608_fullsample_protocol.json --run-id cuda-xgb608-fullsample-select-2022-001 --selection-lock /Users/heap/programming/tdt4173/artifacts/cuda-xgb608-fullsample-select-2021-001/selection_2021.json --selection-lock-sha256 d5fe5b3c2a6c64b0f9a35405411b629aff1698906c876e2a7f99fb73fc4d5e83`. Fixed global weights0/.1/.25/.5 on2021; lock before2022. Selected 0.25; microAUC 0.9937685726056958, anchor 0.9937529477072502, matched608 control 0.9937356307875711. Full provenance `artifacts/cuda-xgb608-fullsample-select-2022-001/manifest.json`; runtime 1.645s.

## cuda-xgb608-fullsample-final-package-001 — 2026-11-02

Sealed only after actual locked2022 fullsample confirmation and independent proof `a018f7431bfc7673595d68db52fbaddcf2c937de67a57567158961b782a7f4d5`. Fixed.25 selected blend scores2021 .9950754228706409 and2022 .9937685726056958, both above frozen acceptedGPU608 anchor and matched original40%rawGPU608 control; below the new acceptedSHOP blend. No .998 transfer claim. Final uses all490896 training rows, original600/depth6/.05/GPU1, exact608 features, seed4173, no early stopping and full checkpoint replay. Fresh genuine smoke required, actual2022 smoke78d64b9b… pins runtime. Complete exact command/source/data/sample/acceptance evidence: `artifacts/cuda-xgb608-fullsample-final-package-001/manifest.json`.

Protocol SHA `4bf91a4ceb200a884802f40af82e3f1f186f50d647c4d44f7bd4d6f601b6714e`; package manifest `8a36c25efa019b02bed4a4ee1ac7084cd2a2056eb5012ac41ab21909a92da036`; PRIVATE ZIP `c529a4de89117e640f22a9b2e19c905b1068a15bfc863be02792357a33331e24` (78824 bytes). Actual independent package audit passed: `artifacts/cuda-xgb608-fullsample-final-package-independent-audit-001/manifest.json`, SHA `bc246235ddb216c8ad47738dde97d6ff083b59b6cf978989a8882b3b55863349`. No final fit/public score at preparation.

## cuda-xgb664-fullsample-package-001 — 2026-11-02

Sealed the conditional SHOP664 row-coverage interaction only after the same independently accepted fullsample6082022 prerequisite. OriginalSHOP664600/depth6/.05/GPU1 model, features and three bundles unchanged; all367248/428568 purged rows replace original40% external sample, which remains a verified subset. Anchor is the acceptedSHOP.25 blend; matched control original40%rawSHOP. Fixed2021 global0/.1/.25/.5 grid, lower-weight ties, strict anchor/control improvement;2022 only an independently audited actual positive lock. No competition fit yet.

Protocol SHA `2a5a114ae75d9a34d4a1f580ff7c3a781332fa92120605501b5b76bc72f31c5d`; PRIVATE ZIP `9d1acb04b973aa7cc7891b76b7188ed0c408916a9e436745a5477212a97e6a34` (61577 bytes). Complete source/input/parameter/sample/anchor/control/prerequisite/ZIP pins and exact command in `artifacts/cuda-xgb664-fullsample-package-001/manifest.json`. Independent actual package audit passed with nested anchors bitwise reconstructed: `artifacts/cuda-xgb664-fullsample-package-independent-audit-002/manifest.json`, SHA `1f8774d033b6e4ec32de8ccbd6fdefcdba2164207d863d302cfc54b808b948cd`. Its reviewer-only001 syntax failure is preserved separately; candidate sources unchanged.

## cuda-fullsample-parallel-package-001 / cuda-fullsample-parallel-notebooks-001 — 2026-11-02T19:15:02.150475+00:00

Prepared one byte-preserving PRIVATE parent dataset from both independently audited child packages plus two separate saved-notebook wrappers. Both originalCUDA1 choices remain unchanged, requiring separate VMs; no shared-session dualGPU reinterpretation. Parent manifest `ae1a7b7af811e919ab0af6233e6456536a8f2eb5f964ea8f9af715247bf89617`; ZIP `3931c540ea611b28c4913f817f545a07228761247bb6685dc2a8aacad65e66dc` (142881 bytes). The fullsample608 final notebook `tdt4173-fullsample608-final.ipynb` SHA `97058cea23ac76d5e9e8c6757881fdf8ad39eb01f05458c6c5563521f265be96` runs only freshsmoke003 and final001/all490896rows. The SHOP664 interaction notebook `tdt4173-shop664-fullsample-2021.ipynb` SHA `19f3740ad7f9e73c818057f7bc822f2a3d9b182d0a7fd871c4f0a9ef8c805d17` runs only freshsmoke001 and2021/all367248rows. Both check exact parent/child/runtime/source hashes, produce independent archived outputs and retain immutable IDs. Full builder source and child/audit/notebook hashes in both preparation manifests. Parent upload/launch follows independent wrapper review; no fit at preparation.

## cuda-xgb608-fullsample-standalone-preparation-001 — 2026-11-02

Root explicitly predeclared a second standalone fixedVinjeprojection diagnostic from the same future fullsample608 final checkpoints, with no added fit or weights, after accepted2022 and before final execution. Frozen diagnostic protocol `ffae7413c80ae91f5bec45de0c28100fba712c982ea1e1c59b1900c429e51b37`; preparation manifest `eb0fd05d37fc58da4ffcb3ffa964b0a04b2b64b126d23feadf2917bfc890733d`. Projected standalone CV .9949084950375574/.9935439554765468, weaker than both primary blends. The standalone2022 score is also slightly below original40%rawGPU performance: the locked blend's gain is complementary, not uniform standalone improvement. Separate exporter requires exact completed strict primary closure and full raw checkpoint replay; primary package remains immutable. Source/input/CV hashes retained, no public result or automatic promotion.


### Fullsample continuation private code upload — 2026-11-02T19:19:49.468132+00:00

Two independentlyapproved childpackages combinedin142881byteZIP, SHA`3931c540ea611b28c4913f817f545a07228761247bb6685dc2a8aacad65e66dc`. Private datasetcreationverified; no runyet. Observation`artifacts/cuda-fullsample-continuation-upload-001/manifest.json`. Fullsample608 final usesacceptedlocked.25recipe; SHOP664 fullsample starts2021only withacceptedSHOPanchorandmatched40%control. Bothfrozencuda1jobs willuseindependentconcurrentsavedVMs, max2.

## cuda-xgb608-fullsample-2022-independent-audit-001

- Timestamp: 2026-11-02T19:11:36.576342+00:00; status: passed; manifest SHA256 `a018f7431bfc7673595d68db52fbaddcf2c937de67a57567158961b782a7f4d5`.
- Independent fixed-weight fullsample608 confirmation: all428568 purged rows, source/native GPU1/600-tree closure, targets, generic projection, exact fixed0.25 arrays and gates. Accepted AUC0.9937685726056958 exceeds anchor0.9937529477072502 and matchedcontrol0.9937356307875711; no new fit or weight search.
- Reproduction/source snapshot, exact inputs/checksums, environment/hardware and checks: `artifacts/cuda-xgb608-fullsample-2022-independent-audit-001/manifest.json` and its preserved reviewer source. Command: `['/Users/heap/programming/tdt4173/.venv-ml/bin/python', 'scripts/audit_cuda_fullsample_2022.py', '--run-id', 'cuda-xgb608-fullsample-2022-independent-audit-001', '--import-manifest-sha256', '2f3c595048010109594e06130808243e2e10c7698e9a4cedce303ca7d0188dc2', '--selection-manifest-sha256', '93d753ac56ad377b9797c0833b1ece2582e677c46d371ed0ba2e38f6632cafd3']`. Runtime: 0.7323763340000369 seconds. No model training, validation split changes, submission or new leaderboard observation from this audit.

## cuda-xgb664-fullsample-package-independent-audit-001

- Timestamp: 2026-11-02T19:14:42.431007+00:00; status: failed_reviewer_syntax; manifest SHA256 `da47606b41232c9b2c4e67f981588e4528bcc6c5d11cc2827a0513df61236e9f`.
- Reviewer-only SyntaxError (missing space in conditional expression) occurred before candidate inspection or execution. Frozen candidate files unchanged; failure/source preserved; corrected reviewer audit002 follows.
- Reproduction/source snapshot, exact inputs/checksums, environment/hardware and checks: `artifacts/cuda-xgb664-fullsample-package-independent-audit-001/manifest.json` and its preserved reviewer source. Command: `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python artifacts/cuda-xgb664-fullsample-package-independent-audit-001/audit.py`. Runtime: not measured before syntax failure seconds. No model training, validation split changes, submission or new leaderboard observation from this audit.

## cuda-xgb664-fullsample-package-independent-audit-002

- Timestamp: 2026-11-02T19:14:43.663708+00:00; status: passed; manifest SHA256 `1f8774d033b6e4ec32de8ccbd6fdefcdba2164207d863d302cfc54b808b948cd`.
- Independent actual SHOP664 fullsample package closure passed: exact ZIP/source/prerequisite proofs, full purged row arrays, unchanged600/depth6/.05/GPU1 parameters and664 features/three bundles. Independently reconstructed acceptedSHOP anchors bitwise, AUC0.9951018232145258/0.9938546734526688, with original40%rawSHOP matched control. Only row inclusion changes; fresh2021 selection required before2022. No fit.
- Reproduction/source snapshot, exact inputs/checksums, environment/hardware and checks: `artifacts/cuda-xgb664-fullsample-package-independent-audit-002/manifest.json` and its preserved reviewer source. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python artifacts/cuda-xgb664-fullsample-package-independent-audit-002/audit.py`. Runtime: 0.42031425000004674 seconds. No model training, validation split changes, submission or new leaderboard observation from this audit.

## integer-water-budget-four-worker-independent-audit-001

- Timestamp: 2026-11-02T19:12:58.557728+00:00; status: passed; manifest SHA256 `675f9f9a3971c0d2f952d74a642a7287cd09e8cc057f9d4a9ecb7cb37bdcb1c0`.
- Source/no-solve preflight review passed: exact original mathematical closure, compatible originalproof import,24pilot byte preservation,2focused tests, conditional resource/drain gate. No live signal/allocation/solve. Subsequent native-session v1 operational drain gate failed; original healthy2-worker cache remains running. Revised interrupted-partial contract requires a separately auditedv2 harness; this source approval did not authorize bypassing failed checks.
- Reproduction/source snapshot, exact inputs/checksums, environment/hardware and checks: `artifacts/integer-water-budget-four-worker-independent-audit-001/manifest.json` and its preserved reviewer source. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python artifacts/integer-water-budget-four-worker-independent-audit-001/audit.py`. Runtime: 1.0536369169994941 seconds. No model training, validation split changes, submission or new leaderboard observation from this audit.

## cuda-fullsample-parallel-final-notebook-independent-audit-001

- Timestamp: 2026-11-02T19:20:00.980629+00:00; status: passed; manifest SHA256 `823b242aefb1d01913715085d3046ab88249e410444300cd3d33ba5925b196ad`.
- Combined PRIVATE parent and fullsample608 final notebook passed: exact complete child datasets/ZIPbytes, passed child proof pins, oneGPU1 final job/all490896rows/600trees, freshsmoke003, fixedaccepted0.25, inherited runtime/source/completion/replay/archive guards. Every executable AST statement matches reviewed dual-final template after explicitly enumerated one-job/sourceID/rowcount substitutions. Synthetic helper command and failurearchive tests pass; no native execution. SHOP2021 notebook separately reviewed by shop; both must run in separate saved VMs.
- Reproduction/source snapshot, exact inputs/checksums, environment/hardware and checks: `artifacts/cuda-fullsample-parallel-final-notebook-independent-audit-001/manifest.json` and its preserved reviewer source. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_cuda_fullsample_parallel_final_notebook.py`. Runtime: 0.04870124999979453 seconds. No model training, validation split changes, submission or new leaderboard observation from this audit.

## cuda-xgb664-fullsample-return-preflight-001 — 2026-11-02

Prepared2021 and conditional2022 strict return entrypoints by changing only the experiment recipe and immutable snapshot filename in the independently reviewed fullsample608 orchestrators. New sources `scripts/import_cuda_xgb664_fullsample_2021_return.py` and `scripts/import_cuda_xgb664_fullsample_2022_return.py` pin actual sealedSHOP fullsample protocol `2a5a114ae75d9a34d4a1f580ff7c3a781332fa92120605501b5b76bc72f31c5d`, frozen trainer/verifier/selector/projection and exact remote namespaces.2022 requires actual independently eligible2021 lock and limits selection to zero plus locked weight. No execution, fit or future acceptance implied. Exact source/template hashes and copies in `artifacts/cuda-xgb664-fullsample-return-preflight-001/manifest.json`, SHA `0f74abf3a7da901a4bfc9042cc43bf9af364dbcc024170a682ea4d6b6875c690`.

## all14-integer-water-budget-preflight-001 — 2026-11-02T19:23:55.990465+00:00

Status passed_preflight_only. Two fixed cases1/1826; all14 binary states, unchanged network economics. One fresh single-thread process percase;60second MILP and two1second fixedLP caps, no retry/fallback. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/probe_all14_integer_water_budget.py --preflight --protocol-sha256 71988ecdbf020a8d44cecb1cd9cc298c1a98f7eb6f920ef5040306cfcc890bef --run-id all14-integer-water-budget-preflight-001`. Runtime 1.232s; feasibility gate None. Full source/data/input/environment/status/bounds/replay/RSS proof: `artifacts/all14-integer-water-budget-preflight-001/manifest.json`. No target values, model fit, cache modification, submission, or automatic continuation.

Three synthetic gate/byte-identity/failure-preservation tests passed in0.026s before protocolfreeze; no competition solve in tests. Frozen protocol SHA `71988ecdbf020a8d44cecb1cd9cc298c1a98f7eb6f920ef5040306cfcc890bef`; no-solve preflight manifest SHA `89fe11543d3d575ee6b0b66c840e3d1d71acc0a7b627ea8299ddc3ecb0d5ddce`. Independent audit required before the already authorized bounded probe.

## all14 integer probe unrun v1 review correction

Before any competition solve, independent source review found the signed-maximum zero-slack check could miss negative nonzero values. Preserve v1 source/protocol and successful no-solve preflight001; no model/solver result was affected. Separate v2 runner/test/doc/protocol uses maximum absolute slack;4synthetic tests pass0.027s including ±1e-14 rejection. Mathematical solver, frozen inputs/cases, budgets and all remaining gates are unchanged. V2 protocolSHA `d391f3c0cb7803e3946348d9f25e0c18502e1b914e94f1eb50ea0a20b0bc5af7`.

## all14-integer-water-budget-v2-preflight-001 — 2026-11-02T19:26:08.011525+00:00

Status passed_preflight_only. Two fixed cases1/1826; all14 binary states, unchanged network economics. One fresh single-thread process percase;60second MILP and two1second fixedLP caps, no retry/fallback. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/probe_all14_integer_water_budget_v2.py --preflight --protocol-sha256 d391f3c0cb7803e3946348d9f25e0c18502e1b914e94f1eb50ea0a20b0bc5af7 --run-id all14-integer-water-budget-v2-preflight-001`. Runtime 1.212s; feasibility gate None. Full source/data/input/environment/status/bounds/replay/RSS proof: `artifacts/all14-integer-water-budget-v2-preflight-001/manifest.json`. No target values, model fit, cache modification, submission, or automatic continuation.

No-solve preflight manifestSHA `29bea27b262fa125f8e7a1665b73e7ef46174e4990decc03cfebcc2a53e2befd`; both inputs reconstructed exact shape/dtype/bytes. Independent audit pending before authorized bounded solve.

## integer-water-budget-four-worker-v2-independent-audit-001 — 2026-11-02T19:26:29.471719+00:00

Status passed. Independent source and saved-evidence review of the separately frozen four-worker continuation v2, protocol `7571930990cadc542c7ad4f47e7f1195d7a239885c887ad45b051afd9b9377d9`. Reproduced the exact permitted protocol differences; every original physical source/function/input pin remains unchanged. Verified all 360 no-solve preflight artifacts, 24 complete pilot receipts and all 33 native-session interruption harness artifacts. The original completed case receipt/payload bytes survive the actual compatible importer; two incomplete attempts remain excluded. Two focused compatibility/mutation tests passed. Command `.venv-ml/bin/python scripts/audit_integer_cache_transition_v2.py`; runtime 2.168s; source/environment/input/test closure in `artifacts/integer-water-budget-four-worker-v2-independent-audit-001/manifest.json`, SHA `a8427c51281a9bf7ff661bdf07ddf661d614e5166ddcedd88c179ab3f1efdfdd`. No solves, target decoding, model fit, submission, headroom allocation or live process interruption in this audit. Before the authorized live transition, require the fresh exact-protocol 3 GiB headroom proof, observed coordinator/worker exit and an externally hashed finalized original manifest with all completed receipts verified.


## Fullsample608 accepted final saved launch — 2026-11-02T19:27:33.502098+00:00

Private version10/scriptVersionId364994788 launched with original pinned runtime and fresh smoke003 `318d9f528f1bd24477d7ce6d6ff047b303cb8debd0bb1f05fd334e5f0e672754`. Final001 began19:21:48.718665UTC; protocol `4bf91a4ceb200a884802f40af82e3f1f186f50d647c4d44f7bd4d6f601b6714e` and notebook `97058cea23ac76d5e9e8c6757881fdf8ad39eb01f05458c6c5563521f265be96` unchanged. All490896 rows,600trees/depth6/lr.05/GPU1; frozen.25 primary plus separate predeclared standalone diagnostic only. Full recipe/data/source/hyperparameters are in `docs/cuda_fullsample_parallel_execution.md` and its immutable package. Launch observation `artifacts/cuda-xgb608-fullsample-final-launch-observation-001/manifest.json` SHA `284b0c674b147bf0eed4cdbe60958e3cf9bf64502aa9f93811a4391114046013`. No completion/public score observed yet. CompanionSHOP664 fullsample2021 import remains unconfirmed after native computer access returned cgWindowNotFound. Saved background run is independent of browser access.

## fullsample608-shop-reuse-preflight-001 — 2026-11-02T19:30:33.391093+00:00

Status passed_preflight_only. Separate parent-authorized complementarity test of raw fullsample608 probabilities against the current accepted SHOP664 blend. Freeze2021 grid0/.1/.25/.5 with lower-weight ties; positive winner must strictly beat SHOP identity and original40%608 raw at the same weight.2022 requires the exact eligible2021 lock plus independent result audit and evaluates only zero/locked weight. Both years are reused development; no leaderboard weight search or fit. Three synthetic tie/control/lock tests pass. Protocol `docs/fullsample608_shop_reuse_protocol.json` SHA `56c35e3a5d7d76e5be5631728e0bbd44b45a14b2228d9bbfd7e26e722e7fbba7` pins all six component arrays/owner manifests, prior independent proofs, original targets by byte hash only, fold IDs/units, and source. Command `['/Users/heap/programming/tdt4173/.venv-ml/bin/python', 'scripts/select_fullsample608_shop_reuse.py', '--protocol', 'docs/fullsample608_shop_reuse_protocol.json', '--protocol-sha256', '56c35e3a5d7d76e5be5631728e0bbd44b45a14b2228d9bbfd7e26e722e7fbba7', '--year', '2021', '--run-id', 'fullsample608-shop-reuse-preflight-001', '--preflight']`; runtime 0.035s. Source/preflight manifest `artifacts/fullsample608-shop-reuse-preflight-001/manifest.json`, SHA `2f6348ca6d0c2c3bd9c3b2d81b90137f56aeb2338a4c0b55e4427d8bbca48c7b`. No labels decoded or new scores computed during this preflight. Independent review required before scoring.

## cuda-xgb664-fullsample-2022-builder-preparation-001 — 2026-11-02T19:28:24.512822+00:00

Prepared conditional SHOP664 fullsample2022 notebook builder `scripts/prepare_cuda_shop_fullsample_2022_notebook.py` SHA `b882678fdb347c3a07f6871c9155945d3b337df1eb50ea3a3407bb9a00513ec6` from the already reviewed fullsample6082022 builder. Changes only select the exact approved SHOP2021 template/protocol and distinct output names. The actual eligible2021 lock and independent audit are mandatory; source embeds their exact bytes, fresh smoke002,428568 full purged rows and the frozen600-tree/GPU1 recipe. No lock/result/notebook was fabricated and no fit launched. Independent source/transform proof `artifacts/cuda-xgb664-fullsample-2022-builder-independent-audit-001/manifest.json` SHA `e6159b4e624525f874820cd7022696d45d52365adb48cac61311eeda688c563e` passed; actual future generated notebook requires byte audit.


## Fullsample608 versus currentSHOP reuse independent preflight — 2026-11-02T19:31:44.862019+00:00

Root independently reviewed frozen protocol `56c35e3a5d7d76e5be5631728e0bbd44b45a14b2228d9bbfd7e26e722e7fbba7` and selector, all source/input hashes, component prediction closure and date-only365-case fold IDs. Three focused tests passed; no prediction/target arrays decoded or scores computed. Artifact `artifacts/fullsample608-shop-reuse-independent-preflight-001/manifest.json`, SHA `c7287b6e1ef9e837cb6c536dd534a145c05be4a181ad1fa5769800ad843de0f5`; runtime0.9997s. Fixed2021 grid0/.1/.25/.5, lower-weight ties, strict currentSHOPidentity and matched40%raw608 control gates;2022 requires independently approved actual lock and evaluates only its weight. Both years reused development, no training/public score.

## all14-integer-water-budget-probe-001 — 2026-11-02T19:29:51.635663+00:00

Status complete. Two fixed cases1/1826; all14 binary states, unchanged network economics. One fresh single-thread process percase;60second MILP and two1second fixedLP caps, no retry/fallback. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/probe_all14_integer_water_budget_v2.py --protocol-sha256 d391f3c0cb7803e3946348d9f25e0c18502e1b914e94f1eb50ea0a20b0bc5af7 --independent-audit artifacts/all14-integer-water-budget-v2-independent-audit-001/manifest.json --independent-audit-sha256 1740cb1c4a0a9bb92e58e4535971b3727b0244bd2d3ee6f957aa71e9b5eed8b0 --run-id all14-integer-water-budget-probe-001`. Runtime 15.908s; feasibility gate True. Full source/data/input/environment/status/bounds/replay/RSS proof: `artifacts/all14-integer-water-budget-probe-001/manifest.json`. No target values, model fit, cache modification, submission, or automatic continuation.

Manifest SHA `b29c685f193df82074ac389c5c5f1ebac6502860fb01443f3f3bcf6b6fb990c4`; protocolv2 SHA `d391f3c0cb7803e3946348d9f25e0c18502e1b914e94f1eb50ea0a20b0bc5af7`; independentpreflight SHA `1740cb1c4a0a9bb92e58e4535971b3727b0244bd2d3ee6f957aa71e9b5eed8b0`. Both cases proven optimal with exact full conditional-LP array replay, zero absolute slack and subset/resource gates passed. Case1 runtime1.8683s/318177280bytes; case1826 runtime10.1831s/488062976bytes. Additional objective losses versus proven3unit controls €516.35686/€415.54562. Independent result audit pending; no predictive metric, label read or cache continuation.

## cuda-xgb664-fullsample-2022-builder-independent-audit-001

Timestamp 2026-11-02T19:29:23.184992+00:00; independent unsealed source/template review PASSED in0.01401320900004066s, manifestSHA `e6159b4e624525f874820cd7022696d45d52365adb48cac61311eeda688c563e`. Exact archived audited original builder except enumerated protocol/template/namespace replacements; actualSHOP2021 source transformations unique and compile. Freshsmoke002/locked2022-001/full428568rows/returnedlock check; genuine independent eligible2021lock required. No actual notebook/lock invented, no fit or score. Reproduction command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_cuda_shop_fullsample_2022_builder.py`; exact source/environment/snapshot in immutable manifest. Actual sealed notebook requires separate byte audit.

## fullsample608-shop-reuse-2021-001 — 2026-11-02T19:32:02.270221+00:00

Status complete. Root independent preflight `c7287b6e1ef9e837cb6c536dd534a145c05be4a181ad1fa5769800ad843de0f5` approved before score computation. Fixed2021 grid scores `{'0.0': 0.9951018232145258, '0.1': 0.9951157955026635, '0.25': 0.9951223352879797, '0.5': 0.9950998287443567}`; selected weight 0.25, candidate 0.9951223352879797, acceptedSHOP identity 0.9951018232145258, matched original40%GPU608 control 0.9950673898246423; strict eligibility True. Exact lock SHA `06017186992d72d2507ca64ebbb6fb0a9f30120010b8c0e2dd4a17fd205b473d`. Reused chronological development, no model fit or public score selection. Runtime 1.162s; command/source/input/environment/prediction closure in `artifacts/fullsample608-shop-reuse-2021-001/manifest.json`, SHA `2caa04a4380ab1ff997936500ec196e20addc9b46cfe5766416ffba6479e519c`. Independent actual2021 audit required before2022; original final exports unchanged.

## fullsample608-shop-reuse-2022-001 — 2026-11-02T19:33:30.922051+00:00

Status complete. Independently audited2021 positive .25 lock `06017186992d72d2507ca64ebbb6fb0a9f30120010b8c0e2dd4a17fd205b473d` and audit `c8e843b0f2ec06ace038eb115b7bed970d0a3d80ab50ca2f22b158ffacbcf66e` preceded2022 array loading. Evaluated only zero/locked.25: `{'0.0': 0.9938546734526688, '0.25': 0.9938443216332464}`; candidate 0.9938443216332464, acceptedSHOP identity 0.9938546734526688, matched original40%GPU608 control 0.993812994867662; accepted False, >=.998 both-year transfer gate False. Reused chronological development; no fit, weight expansion or public score selection. Runtime 1.013s; exact command/source/input/environment/prediction closure in `artifacts/fullsample608-shop-reuse-2022-001/manifest.json`, SHA `5ad82da0b5a41107d9797d0603e57423008781a6e50382b924677eb4ed8c9183`. Independent actual2022 audit required before any new final export. Original fullsample final primary/diagnostic declarations remain unchanged.


## Computer-access interruption — 2026-11-02T19:36:46.814726+00:00

Native Brave reports cgWindowNotFound after the verifiedv10 launch; recovery by app/name/bundle selection and session reset failed. Browserprovider unavailable. Presence-only officialAPI fallback check found no CLI/config/tokenfile/tokenenvironment; no credential content read. User restore/unlock request pending; local cache/experiments continue. Observation `artifacts/browser-access-interruption-20261102-1937/manifest.json`. No v11 launch, new hosted completion or public score inferred.

## all14-integer-water-budget-result-independent-audit-001

Independent two-case physical/result review PASSED: manifest SHA `97c1ff331fb7ad5fe87c7fd2ebd8de7e00d06c384bbb4fa5d0ee69e04a666e69`. Incumbent feasibility, full conditionalLP KKT/duals, complete fixedLP array replay, objective/bounds, exact input/source closure independently checked without solves or labels. Full command/source/environment/runtime/check details retained in `artifacts/all14-integer-water-budget-result-independent-audit-001/manifest.json`. Root now permits separately frozen12-case pre2021 descriptive diagnostic after independent preflight/resource check; no fullcache expansion or model is inferred.

## lgb-shop664-preflight-001 — 2026-11-02T19:37:49.432496+00:00

Status passed_preflight_only. Frozen LightGBM664 feature ablation: exact608prefix plus56SHOP, original350trees/31leaves/lr.06/minleaf100/.9featurefraction/L2=1/40%sample,2CPUthreads, no early stopping. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train_lgb_shop664.py --protocol docs/lgb_shop664_protocol.json --protocol-sha256 882e8c3124ba834322c8f062a7544e9ca3e9512b385e8830db387c38f7c97d08 --year 2021 --run-id lgb-shop664-preflight-001 --preflight`. Year2021; selectedweight None, selectedAUC None, SHOPanchor None, matched originalLGB608 None. Runtime 0.291s, peakRSS 184107008bytes. Full sources/data/splits/samples/config/checkpoints/replays/environment/gates: `artifacts/lgb-shop664-preflight-001/manifest.json`. No final training/submission;2022 requires independent eligible2021lock.

Metadata-only preflight passed0.2914s/184107008bytes: exact staged three-bundle path/size/hash closure, same original146899/171427 sampled rows with original across-fold RNG sequence and six-day purge. Four synthetic chronology/tie/strictgate/lock/future-poison tests passed0.017s. Protocol SHA `882e8c3124ba834322c8f062a7544e9ca3e9512b385e8830db387c38f7c97d08`, manifest SHA `3d4daba614b5b00de205568dd15261601db725477dfce8a6809affdbeba464cc`. No feature build/target values/model fit occurred. Await independent source approval and explicit release of shared heavy local slot after745 staging audit; root already authorizes2021 under those conditions. Cache resource observation:688/3647 cases, maximum recent workerRSS0.732GB,68% memory free/no swap, per shop.


## Continuation verification suite — 2026-11-02T19:39:12.824332+00:00

Command `.venv-ml/bin/python -m unittest discover -s tests`; 272 tests passed in11.549s, exit0. Log `artifacts/tests-continuation-20261102-1939.log`; source inventory and execution proof `artifacts/verification-suite-20261102-1939/manifest.json`. Test source inventory unchanged during execution:True. This validates implementation checks, not a new competition score.

## lgb-shop664-2021 launch observation

Launched 2026-11-02T19:41:58.115328+00:00 in localsession75108 after independentpreflight `45af410ce0e04f9bae88d6b7f93f4512a64d14e7cf17fa280438ea4d2230c5f6` passed and xgb released745 staging heavy slot. TwoCPUthreads; memory54%free at observed prelaunch on32GiB. Exact command/recipe references in `artifacts/lgb-shop664-2021-launch-observation-001/manifest.json`, SHA `abf406f842cf7ff7912b9cc4449e4166dcd0dd22399a7d4c5c51730afc852e8b`; run `artifacts/lgb-shop664-2021-001`. No score yet. One heavy local model job alongside separately controlled4-worker physics cache;2021 only.

## lgb-shop664-independent-preflight-001 — 2026-11-02T19:41:04.169201+00:00

Status passed. Independent exact source/input/stage/recipe review for the parent-authorized LGB664 ablation. Original350-tree/31-leaf parameters are unchanged except authorized n_jobs3→2. Independently reconstructed the physical horizon purge,365 validation cases each and byte-identical146899/171427-row40% sequential seed4173 samples. Four tests passed, including future-label poisoning and strict tie/control/lock behavior. No fitting, feature build, label decoding or scores during audit. Command `.venv-ml/bin/python artifacts/lgb-shop664-independent-preflight-001/audit.py`; runtime 1.335s. Full environment/source/inputs/tests/splits in `artifacts/lgb-shop664-independent-preflight-001/manifest.json`, SHA `45af410ce0e04f9bae88d6b7f93f4512a64d14e7cf17fa280438ea4d2230c5f6`.2021 only approved after745 feature preflight exited;2022 requires independent actual positive lock.

## cuda-xgb745-package-001 — 2026-11-02T19:38:25.193831+00:00

Status failed. The first package attempt stopped before protocol creation: Python tuple-valued bypass route metadata compared unequal to its JSON-list serialization. Original source/cache remained exact; normalized only the expected math specification through JSON, preserved failed001 source/report, and retried unique002. Exact command/environment/source/package/protocol/artifact closure `artifacts/cuda-xgb745-package-001/manifest.json`, SHA `0edf3d6e5380ee461aedf780da6aefe77d609e34ecbf56ca84a43bf7f28dbb3c`; runtime 0.031s. Protocol None; PRIVATEdataset SHA None, bytes None. Independent package audit required before hosted use.

## cuda-xgb745-package-002 — 2026-11-02T19:38:37.153724+00:00

Status complete. Sealed fourth disjoint LPv2 supplement and new745 code. Preserves exact original664 prefix, all original model/sample parameters and the three existing bundles. Full3653-cache-artifact/source/data closure passed. Five745 tests plus six originalLPv2 tests passed before seal. No upload, score or fit. Exact command/environment/source/package/protocol/artifact closure `artifacts/cuda-xgb745-package-002/manifest.json`, SHA `9030621c95f1527c420295876e2e46bbe0095b8dd56353a8ae3e27ece63e5a06`; runtime 4.033s. Protocol 4e8aaf0a32c7899cac0b0c95927af6b0bb764d49847069fcf4af5a5c8092b41e; PRIVATEdataset SHA c01101fa9a6756fb78ec5f70c16b47e21a95ea22711a0fb66ab8f26a142a9894, bytes 54218767. Independent package audit required before hosted use.

## cuda-xgb745-feature-preflight-001 — 2026-11-02T19:39:35.667523+00:00

Status passed. Fresh-process actual four-bundle staging/feature construction produced612696×745. Every81 appendedLPv2 value equals the frozen cache;56SHOP values remain exact; pinned ancestor loaders and new explicit equality check preserve all664 original columns/values/dtypes. Full package/archive closure and disjoint bundle union verified.2021 cutoff guard active; no2022 target decoding or model fit. Runtime 12.219s, measured peakRSS 8353513472bytes, higher than the prior rough3–5GiB estimate. Process exited and heavy slot released beforeLGB664 fitting. Source/feature formula/environment/command proof `artifacts/cuda-xgb745-feature-preflight-001/manifest.json`, SHA `9c90dff0df4a232c53dc56410130e200625da9deaa49361a586ae28a95d46413`. Root performs independent package review; this execution preflight was performed by the package author.

## fullsample608 SHOP reuse — independently confirmed rejection

Root independently checked2021 grid/lock with generic min-max isotonic projection, average-rank Mann–Whitney AUC and exact selected bytes: proof `artifacts/fullsample608-shop-reuse-2021-independent-audit-001/manifest.json` SHA `c8e843b0f2ec06ace038eb115b7bed970d0a3d80ab50ca2f22b158ffacbcf66e`. Locked2022 failed the unchanged SHOP identity gate and was independently confirmed: `artifacts/fullsample608-shop-reuse-2022-independent-audit-001/manifest.json` SHA `3d2746a5dc82706b2f8749acc7bbf3dcd0a6cc6898d0e450a10e0313ae49beed`. No combination export, fit, extra weight, public submission or promotion. Original accepted fullsample608 primary/standalone declarations remain intact.


## Additive745 independent package review — 2026-11-02T19:42:59.614428+00:00

Frozen protocol `4e8aaf0a32c7899cac0b0c95927af6b0bb764d49847069fcf4af5a5c8092b41e`, PRIVATE dataset54,218,767bytes SHA `c01101fa9a6756fb78ec5f70c16b47e21a95ea22711a0fb66ab8f26a142a9894`. Root inspected source differences from664, all new ZIP members/manifest closure, original3653LPv2 artifacts/source/data, disjoint fourth bundle, unchanged model/sample/purge and currentSHOPanchor/raw664 control. Root also reviewed the producer's real612696×745 feature-build source and proof `9c90dff0df4a232c53dc56410130e200625da9deaa49361a586ae28a95d46413`, including exact664prefix preservation and exact81/56 appended caches. Actual producer peakRSS8.3535GB,12.22s; no second heavy rebuild was performed by root.

Reviewer run001 failed from a reviewer-only JSON key assumption (`prediction_path` instead of actual `prediction`); source/failure preserved at `artifacts/cuda-xgb745-package-independent-audit-001`. Corrected reviewer run002 passed1.1700s, proof `4b2e34c296e0c6835c97f09242bfd0ef85b6dfd03e14dd1c940fcec301e24540`. Exact command `.venv-ml/bin/python scripts/audit_cuda_xgb745_package.py --run-id cuda-xgb745-package-independent-audit-002`; source snapshot/environment/full recorded checks in that artifact. Candidate package/protocol never changed.272-test suite includes exact frozen745 tests. No model fit, decoded new targets, upload, score or submission by this review. A reviewed saved notebook/fresh actual CUDA smoke is still required.

## cuda-xgb745-return-preparation-001 — 2026-11-02T19:43:09.807686+00:00

Prepared strict2021/conditional2022 transfer orchestration from the reviewed664 fullsample importers by changing only recipe protocol/runner/verifier/selector identifiers. No import, score, fit or remote action. Exactsources/templates/protocol in `artifacts/cuda-xgb745-return-preparation-001/manifest.json`, SHA `286e1a34ceffdcadcdce465489afc8d4d3dc680907df51d944a9e07217c7fdfa`.2021 requires hostedarchive/fold/smoke hashes;2022 additionally requires an actual independently positive2021lock and evaluates only zero plus the locked weight.


## Additive745 saved2021 wrapper independent review — 2026-11-02T19:43:46.423503+00:00

Root reviewed exact notebook/plaintext bytes, parsed only freshsmoke001 and2021-001 commands, pinned runtime/source/input discovery, immutable outputs and fullcheckpoint/sample guards. Notebook SHA `7c4854fb500471436f8b27c7fcd82e8ae45c7564f39d8ae1a3b9958e1e3cf5c5`, sourceSHA `eb02fdab19cd66232639770220c65d298ac8f5154c094da6d1cdb8425d51b5b6`. Proof `artifacts/cuda-xgb745-2021-notebook-independent-audit-001/manifest.json`, SHA `705c5c34c3b2e2ef0ef46126e671a04076b41cbcec982e4d669405030166beac`; code compiled but notexecuted. No new model/GPU/score/public action. Browser recovery remains pending; authorized privateupload and2021savedrun followactualruntime/smoke checks.

## all14-integer-water-budget-twelve-independent-preflight-001

Timestamp 2026-11-02T19:44:05.315969+00:00; independent no-solve preflight PASSED in2.7654622919999383s. Manifest SHA `c991b00c0718198e40e55e340fa6423710fac7f4d5d3f69527115292f682553f`; protocol SHA `14711cf1aa752bcd0932932d59cad48371e1c0c9482d296301e67b2fd0207abc`. Independently rebuilt all12 fixed input dictionaries with exact dtype/shape/bytes, checked originalmetadataIDs/blocks/strictpre2021intervals, exact two-case copied resultproofbytes, full source/control/environment closure. Two producer tests plus4additional negative proof/resource gate fixtures pass. Source ensures all12proven/fullLPbyte/absolutezero/subset/RSS gates and physical lock before selected target read; no adaptive scored subset. Ten new originalall14worker attempts only, one fresh singlethread/60s+2×1s, no retry/fallback. Resource guard30%free/no swapgrowth required before each solve, alongside existing cache/localmodel. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_all14_integer_diagnostic.py`; preserved reviewer source/test output/environment at audit directory. No solves, target values, model fitting or submission in review.

## cuda-xgb745-2021-notebook-001 — 2026-11-02T19:42:43.521261+00:00

Prepared saved2021-only notebook `artifacts/cuda-xgb745-2021-notebook-001/tdt4173-xgb745-2021.ipynb`, SHA `7c4854fb500471436f8b27c7fcd82e8ae45c7564f39d8ae1a3b9958e1e3cf5c5`, exactcode SHA `eb02fdab19cd66232639770220c65d298ac8f5154c094da6d1cdb8425d51b5b6`. Inherits verified Python3.13.15/numpy2.1.3/pandas2.3.3/XGBoost3.4.1/scipy1.16.3/sklearn1.6.1/PyYAML6.0.3/uv0.13.5/T4×2driver580.178.04 runtime,20-second heartbeat and durable success/failure archive helpers. Fresh745smoke001 before2021-001; exact146899sample/745feature/600×14/fullreplay completion guards. Root independent wrapper proof `artifacts/cuda-xgb745-2021-notebook-independent-audit-001/manifest.json` SHA `705c5c34c3b2e2ef0ef46126e671a04076b41cbcec982e4d669405030166beac` passed. No actual hosted launch/result.

## cuda-xgb748-template-preparation-001 — 2026-11-02T19:51:10.418635+00:00

Prepared unsealed portable748 trainer/fourth-bundle loader/strictverifier/selector/launcher/packagebuilder by adapting the reviewed745 paths. Exact664 prefix plus existing84 integer-water-budget features; original600/depth6/.05/GPU1/40%samples, currentSHOP anchor/raw664 matched control, fixed2021grid0/.1/.25/.5 and independentlylocked2022 gates. Six focused tests passed, including rejection of partial/unproven caches or missing compatibility/physical audit claims. No partial cache features read, no solve/fit/upload, no model protocol or dataset sealed. Builder requires explicit complete3647cache manifest and independent physical/replay audit hashes; verifies allraw artifacts/completed receipts and original compatible receipt/payload bytes. Shop agreed required final audit schema. Exact source snapshots/dependencies/testcommand in `artifacts/cuda-xgb748-template-preparation-001/manifest.json`, SHA `41f91c0f0b9e3d4224c6a6188f876b399b1a9e7906d7a23e846b9d99f97d57c4`. Actual package remains deferred untilcompletecache+independentphysicalproof.

## Conditional standalone745/748 diagnostic predeclarations

Root explicitly predeclared one fixed projected raw diagnostic per branch, separate from its primary, before either branch2021 fit. `docs/cuda_xgb745_standalone_declaration.json` SHA `9803b943ad0d2b82e4459ed33dde2275ceddfa112ea77a8aabb95e1905dee511`; `docs/cuda_xgb748_standalone_declaration.json` SHA `5d2e8274e62e52ac6a7088b8373e28d48cdd5b23dad95eff633e1ba46d8070ce`. Formula is fixed `project_vinje(rawGPU)`,float64; no additionalfit/weights/automaticpromotion. Both-year primary acceptance, actual strictfinal model/checkpoint/schema verification, completedprimaryexport and independently audited actualdiagnosticCSV are mandatory before submission. No diagnostic export currently enabled; frozen745 training protocol unchanged.

## all14-integer-water-budget-twelve-physical-independent-audit-001

Independent audit PASSED verifying the original60-second twelve-case physical-screen failure:11nativeoptima and case2186 status1. Manifest SHA `0e5c05d2f6448ec7509162605c2a35ae4b47884a9deaffe1cfa34c642d062b74`; reviewed original resultSHA `f971f8def3cf7259b820402393babd08aafde7ea992441dd5cc4bdb0a3af042f`. All12 physical inputs reconstructed exactly; every stored incumbent feasibility, full fixed-LP primal/dual KKT, byte replay, zero absolute slack, continuous/three-unit subset objective and below2GiB RSS independently checked without solving. Case2186 native relativegap1.125174082e-6 and approximately€86.10 absolutegap independently recomputed. All nonoptimality gates pass; no target values, metrics or diagnostic arrays were opened/created. Runtime 1.8772592500008614s; exact command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_all14_integer_twelve_physical.py --manifest-sha256 f971f8def3cf7259b820402393babd08aafde7ea992441dd5cc4bdb0a3af042f`, preserved source/environment/case evidence in the audit artifact. Root separately authorized a frozen2186-only600s proof completion preserving all11 proven results; this is not automatic rescue or a change to the closed original run.

## cuda-xgb748-template-independent-audit-001 — root independent review

Status `passed_template_only`: TEMPLATE ONLY. Root reviewed the748 trainer/loader/verifier/packagebuilder deltas, exact unchanged664 ancestry, source dependency closure, complete external physical proof schema, raw artifact/receipt/compatibility gates and separate745/748 standalone declarations. Six focused tests passed. Proof `artifacts/cuda-xgb748-template-independent-audit-001/manifest.json` SHA `13c34c736f5e576e4e92626363579ea6b9be833c99ed1b3a2c95d3d74c3382da`. No partial features read, no model protocol/package sealed, no staging/CUDA smoke/fit performed. Actual full3647cache completion and independent physical/replay/compatibility audit remain prerequisites; final source/protocol/package and actual staged features must be reviewed before any hosted fit. Original frozen cache sources and745protocol unchanged.

## all14-integer-water-budget-completion-independent-preflight-001

Timestamp 2026-11-02T19:56:05.087529+00:00; passed in2.8805060000013327s. Manifest SHA `5a58d1ca44f706f86c450078f9adcb364aebae4c8e351c9810c057a0ccf023d2`; frozen completion protocolSHA `63fa814bf8db436194272bf0596ad848e841625deb7358bd081825fbbc2b7b7d`. Independent review verified exact sole new job2186, all11 proven result directories preserved byte-for-byte and all12 physical input dictionaries freshly reconstructed exact. Worker AST identical to auditedv2 except60→600seconds; same two1-second fixedLPs, physical formulas, original target-date/physicalgate/read-label helpers. Two tests and4additional negativegate cases pass. New physical lock/all12gates before any target decoding; resource30%free/no swapgrowth gate retained. Original60s physical failure remains immutable. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_all14_integer_completion.py`; source/environment/full checks in artifact. No solves/targets/model fit/submission in audit. Root explicitly authorized this separate one-case proof-completion; no further retry inferred.

## lgb-shop664-2021-001 — 2026-11-02T19:41:19.029762+00:00

Status complete. Frozen LightGBM664 feature ablation: exact608prefix plus56SHOP, original350trees/31leaves/lr.06/minleaf100/.9featurefraction/L2=1/40%sample,2CPUthreads, no early stopping. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train_lgb_shop664.py --protocol docs/lgb_shop664_protocol.json --protocol-sha256 882e8c3124ba834322c8f062a7544e9ca3e9512b385e8830db387c38f7c97d08 --year 2021 --run-id lgb-shop664-2021-001 --independent-audit artifacts/lgb-shop664-independent-preflight-001/manifest.json --independent-audit-sha256 45af410ce0e04f9bae88d6b7f93f4512a64d14e7cf17fa280438ea4d2230c5f6`. Year2021; selectedweight 0.1, selectedAUC 0.9951042081969695, SHOPanchor 0.9951018232145258, matched originalLGB608 0.9950939207833579. Runtime 1137.028s, peakRSS 6873006080bytes. Full sources/data/splits/samples/config/checkpoints/replays/environment/gates: `artifacts/lgb-shop664-2021-001/manifest.json`. No final training/submission;2022 requires independent eligible2021lock.

Completed2021 manifest SHA `88faf9b155271b5ab46488031b9510d5aecbb6a9bbd1abe0c19554808b5bf413`; selected lock SHA `cc4c7e21967d61bbb6c9139ff32a6da2586a8686156d520f3bccf3eaab28a1b9`. All14 complete350tree models, every validation prediction replayed exactly in float64; storedrawfloat32 matches priorrecipe. Fixedgrid scores {"0.0": 0.9951018232145258, "0.1": 0.9951042081969695, "0.25": 0.9950904829409322, "0.5": 0.995022209739604}. Positive0.1 wins: improvement overcurrentSHOPanchor 2.384982443759398e-06 AUC, versus matchedrawLGB608control 1.0287413611642648e-05. This is a small reused-development gain, not a public score. Independent fresh-feature/checkpoint/grid audit required before2022; process exited0 and heavy slot handed to xgb for that review.

## lgb-shop664-2021-independent-audit-001 — 2026-11-02T20:01:36.152841+00:00

Status passed. Original2021target values, exact training sample bytes, source/input/artifact/seal closure and native350tree/664feature/CPU2thread parameters independently checked. Fresh664feature reconstruction replayed all14 checkpoints and reproduced every saved rawfloat32 validation probability exactly. Independent generic min-max isotonic and average-rank Mann–Whitney AUC reproduced fullfixedgrid, selected.1, same-weight rawLGB608control and strict eligibility. Candidate .9951042081969695 >SHOP .9951018232145258/control .9950939207833579; gain is only2.38498e-6. Eligible2021lock SHA `cc4c7e21967d61bbb6c9139ff32a6da2586a8686156d520f3bccf3eaab28a1b9`. Command `['/Users/heap/programming/tdt4173/.venv-ml/bin/python', 'scripts/audit_lgb_shop664_result.py', '--year', '2021', '--candidate-run', 'artifacts/lgb-shop664-2021-001', '--candidate-manifest-sha256', '88faf9b155271b5ab46488031b9510d5aecbb6a9bbd1abe0c19554808b5bf413', '--run-id', 'lgb-shop664-2021-independent-audit-001']`. Runtime 13.876s; peakRSS 5055332352bytes. Full proof `artifacts/lgb-shop664-2021-independent-audit-001/manifest.json` SHA `31e11af7408082765541ba8cf45d7c4c7bcfc2884d80a664a7d5969961d0e005`. No refit, extra weight or remoteaction. Audit process exited and heavy slot released; only locked.1 is eligible for2022 under parent authorization.

## lgb-shop664-uncertainty-2021-001 — 2026-11-02T20:02:41.143222+00:00

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/paired_auc_uncertainty.py --run-id lgb-shop664-uncertainty-2021-001 --baseline artifacts/cuda-xgb664-select-2021-001/selected_validation_2021.npz --candidate artifacts/lgb-shop664-2021-001/selected_validation_2021.npz --replicates 200`. Runtime 5.34s. Fixed-model paired28-day circular case bootstrap, seed4173, 200replicates; no fitting, selection, early stopping or Kaggle submission. Difference 2.3849824438704204e-06; descriptive95%percentile interval [-1.8745576275833176e-05, 2.037688064081833e-05]. Adaptive research reuse, training uncertainty and distribution shift are not accounted for. Full source/input/artifact hashes, IDs, dates, versions and bootstrap draws: `artifacts/lgb-shop664-uncertainty-2021-001/manifest.json`.

## lgb-shop664-2021 independent result and locked2022 launch

Independent2021 result audit PASSED: proof `31e11af7408082765541ba8cf45d7c4c7bcfc2884d80a664a7d5969961d0e005`, source/input/sample/nativeparams/originaltargets/full664feature rebuild/all14 checkpoint replays and independent projection/rank-AUC grid/control verified in13.88s/5.06GB. Exact selected0.1 lock `cc4c7e21967d61bbb6c9139ff32a6da2586a8686156d520f3bccf3eaab28a1b9` is eligible. After reviewer process exit/heavy-slot release and60%memoryfree, launched authorized locked2022 session39478. Launch observation `artifacts/lgb-shop664-2022-launch-observation-001/manifest.json`, SHA `42d31ff82a64a056001fec1a68b05eaef51b033b59569d1e60667c56ada49396`, retains exact command and all proof pins. Same original171427-row sample/350trees/31leaves/2CPUthreads; only0 and0.1 scored. No2022score yet, final training/export/submission not performed.

## all14-integer-water-budget-completion-result-independent-audit-001

Independent completion result audit PASSED in2.046544000000722s; manifestSHA `d4bc01a1b79425bebed753d5b4b5441f9d0aa39d7d81be2c25f6c475897cd458`. Actual completed resultSHA `7c3ea0d07408b0f91d7c0d34b3b877fb68c3357eae186790bdba9c555300f26d`. All11 originalproven directories preserved byte-for-byte, sole2186 new600s-cap run reached nativeoptimum in72.652s. All12 frozeninputs/incumbentphysicalfeasibility/fullfixedLPKKT/replay/absolutezeroslack/subsetobjective/RSS checks independently passed without solving. Physicalseal preceded selectedpre2021 target decode. Originalselected12labels, every diagnostic qratio/pratio/state array, full14/weak3/unit/block/oncount/mismatch metrics independently reconstructed, with separate average-rank Mann–Whitney AUC arithmetic. Negative proxy effect confirmed: all14Q-AUC0.938095018405212 versus threebinary0.9421982777182167, powercutofferrors2646 versus2643, weak3errors701unchanged. Root declined all14fullcache expansion; no newmodel/publicpromotion. Exact reviewer command/source/environment/pins/case findings are preserved in audit artifact.


## tool-transition-live-observation-20261102-2012 — 2026-11-02T20:11:40.637674+00:00

Execution-tool transition observation, no training/configuration/data mutation. Old session handles 39478/84741 are unavailable in this backend, but read-only `ps -p 11014,16081 -o pid,etime,%cpu,rss,state,command` independently confirms the integer-cache coordinator and LGB664 locked2022 process live; neither was restarted. Detailed process evidence, environment and hash in `artifacts/tool-transition-live-observation-20261102-2012/manifest.json`. LGB completed 5 of14 units at observation. Prior goal turn classified PROGRESS: actual fits/physics diagnostics/preparation completed.

Native Brave still reports `cgWindowNotFound`; current-IAB-context tool also fails because its runtime references missing browser-service.mjs version51404. Listed browser skill61615 is missing and alternate search found no replacement. No tool runtime patching or prohibited Dock retry. User restore request remains pending. No Kaggle observation or score added; last confirmed public0.99651/rank18.


## integer-cache-auditor-pilot-preflight-001 — 2026-11-02T20:16:30.922213+00:00

Integer-cache auditor preflight passed, completed24case pilot only; 24 checked. No current-cache or partial-feature reads, optimizations, labels or fits. Exact command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/preflight_integer_cache_audit.py --run-id integer-cache-auditor-pilot-preflight-001`, source and input pins, physical/KKT/replay/independent84feature reconstruction and corrupted-proof checks retained in `artifacts/integer-cache-auditor-pilot-preflight-001/manifest.json`. Runtime 2.798s. Does not approve an incomplete cache.


## lgb-shop664-final-template-independent-audit-001 — 2026-11-02T20:15:24.270185+00:00

Status `passed_template_only`. Independently reviewed conditional final trainer, protocol builder and exporter; original196358-row sample bytes reproduced exactly from seed4173, original350-tree/31-leaf parameters preserved with authorized2threads, actual currentSHOP projected anchor and full artifact closure verified. Three eligibility/composition/ID tests passed. Actual complete accepted2022 result and independentproof remain mandatory before any finalprotocol is sealed; separate all14 fulltest-checkpoint replay and actualCSV audit remain required. No feature build, target decode, fitting or submission. Command `.venv-ml/bin/python scripts/audit_lgb_shop664_final_template.py`. Runtime 1.096s; peakRSS 208044032 bytes. Exact source/input/environment evidence in `artifacts/lgb-shop664-final-template-independent-audit-001/manifest.json`, SHA `0f952fa4e58dc574ea73e77239a8ebb6be1f567a585f445ae83ef0f4930454da`.


Tool-access observation 2026-11-02T20:17:18.523007+00:00: native ChatGPT desktop fallback denied by computer-use safety review (`com.openai.codex`); denial respected and explained to user. No new Pro consultation or model-generation claim. Separate observation retained alongside tool-transition evidence.


## integer-cache-auditor-pilot-preflight-002 — 2026-11-02T20:18:23.001936+00:00

Integer-cache auditor preflight passed, completed24case pilot only; 24 checked. No current-cache or partial-feature reads, optimizations, labels or fits. Exact command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/preflight_integer_cache_audit.py --run-id integer-cache-auditor-pilot-preflight-002`, source and input pins, physical/KKT/replay/independent84feature reconstruction and corrupted-proof checks retained in `artifacts/integer-cache-auditor-pilot-preflight-002/manifest.json`. Runtime 2.822s. Does not approve an incomplete cache.


## lgb-shop664-2021-pair-decomposition-001 — 2026-11-02T20:18:40.029245+00:00

Descriptive only: exact14×14 positive-unit/negative-unit win-plus-half-tie decomposition of the already-audited2021 SHOPidentity→selected0.1LGB664 blend. GlobalAUC gain 2.38498244381939e-06 consists of cross-unit +2.78650261616551e-06 and within-unit -4.01520172346111e-07. The strongest gaining cell is Haukeli-positive versus Lio-negative(+1.168347e-6); the strongest losing cell is Kjela-positive versus VinjeG1-negative(−6.702672e-7). This is neither causal attribution nor a significance claim; no new weights, predictions, fits,2022 reads or acceptance changes. Both recomputed pooledAUCs match frozen candidate manifest within1e-13. Command `.venv-ml/bin/python scripts/analyze_lgb_shop664_2021_pairs.py`; runtime 0.721s, peakRSS 64585728bytes. Exact source/input/proof/environment/matrix artifact `artifacts/lgb-shop664-2021-pair-decomposition-001/manifest.json`, SHA `e85b7a4079d466424979b87f75d3862a02bd626c9b835e48cebb3314e52069da`.


## integer-cache-auditor-independent-source-review-001 — 2026-11-02T20:20:09.537580+00:00

Status `passed_source_review_only`. Independent review identified and closed copied-reuse-manifest, canonical diagnostic, pilot bypass and retry lineage gaps before sealing the auditor. Reviewed source SHA `78ff7efe84b00129bafa112c9a2a6be8149100f99f456313af8178788f9660e9` now checks complete3647gate,636 compatible original receipts/payloads,12canonical source/result/proof origins,24original pilot cases and native-status1-only60→600retries. Frozen no-solve matrices, native incumbent feasibility, LP primal/dual KKT, all four conditionalLP payloads/two MILP replicas and independent84feature formula checks reviewed; native status0 remains the optimum criterion without an extra literal-zero gap rule. A reviewer fixture constructed solely from the already completed pilot passed audit_case; duplicate replicas and unapprovedpilot identity were rejected. Four focusedtests independently passed; original24case physicalpreflight002 pin retained. No live/incompletecache feature access, optimizer call, target decode or model fit. Command `.venv-ml/bin/python scripts/review_integer_cache_auditor.py`, runtime 1.762s. Full source/input/fixture proof `artifacts/integer-cache-auditor-independent-source-review-001/manifest.json`, SHA `4d8d57e1c051493aadfcc1ca1f498c9b5f124a4e118539905cf55dddd542af9b`. Actualcomplete3647cache audit remains mandatory before748package.


## Full integer-cache auditor preparation — 2026-11-02T20:20:47.762784+00:00

Root implemented independent all-case physical/KKT/replay/84feature reconstruction and original636receipt/pilot/canonical/retry lineage checks in `scripts/audit_integer_water_budget_cache.py`. Real completed24case pilot preflights001/002 passed (2.798/2.822s), all native/conditional outputs verified, corrupted evidence rejected; four focused tests pass. Source review by xgb passed `artifacts/integer-cache-auditor-independent-source-review-001/manifest.json`, SHA4d8d57e1c051493aadfcc1ca1f498c9b5f124a4e118539905cf55dddd542af9b. No live-cache derived features read, optimizer call, target decoding or model fit; preparation is not fullcache approval. Future command and interpretation in `docs/integer_water_budget_full_cache_audit.md`.


## hyljelihyl-river-cost-v2-independent-preflight-001 / 002

Independent v2 source/protocol/preflight review PASSED in23.304547s. Proof `artifacts/hyljelihyl-river-cost-v2-independent-preflight-002/manifest.json`, SHA `6268f5155b40869074ae3e673bef2c10e973c13d0e5d03302fd6c3c8b879e203`. Exact protocol `ba460ccb6085f9b032ea0647e9d19879801d0095ae5eece240ffbbdc33ab91f0`; producer preflight `d34d49f5cf0eeb07fc84fbf8f164c9babf5519cc66729ba2abec7bf1bd31ff23`. All twelve original metadata windows, raw physical inputs, independent float64 Hyljelihyl inventory partitions, equal terminal values and A/B/C matrix signatures verified without real solves or labels. Separate sparse algebra proves every old balance/inequality, original finite bounds including aggregate bypass cap, and no larger projected-A minimization cost for arbitrary B-feasible points; C adds the fixed nonnegative convex segment costs with unbounded tail. Seven synthetic tests passed, including adversarial joint-cap overflow and marginal-dual units/sign. Exact command/source/environment in proof. Preserved reviewer001 failure SHA `5f9332611ade88e867f651b72f200b0cb17bc1a8c04c1d0ece54dea599ed265a`: independent expected-array construction used NumPy scalar promotion instead of declared float64 conversion. Only reviewer corrected; frozen candidate unchanged. Unrun mathematical v1 remains intact; actual diagnostic and later result audit remain separate.

## lgb-shop664-2022-001 — 2026-11-02T20:02:27.535436+00:00

Status complete. Frozen LightGBM664 feature ablation: exact608prefix plus56SHOP, original350trees/31leaves/lr.06/minleaf100/.9featurefraction/L2=1/40%sample,2CPUthreads, no early stopping. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train_lgb_shop664.py --protocol docs/lgb_shop664_protocol.json --protocol-sha256 882e8c3124ba834322c8f062a7544e9ca3e9512b385e8830db387c38f7c97d08 --year 2022 --run-id lgb-shop664-2022-001 --independent-audit artifacts/lgb-shop664-independent-preflight-001/manifest.json --independent-audit-sha256 45af410ce0e04f9bae88d6b7f93f4512a64d14e7cf17fa280438ea4d2230c5f6 --selection-lock artifacts/lgb-shop664-2021-001/selection_2021.json --selection-lock-sha256 cc4c7e21967d61bbb6c9139ff32a6da2586a8686156d520f3bccf3eaab28a1b9 --lock-audit artifacts/lgb-shop664-2021-independent-audit-001/manifest.json --lock-audit-sha256 31e11af7408082765541ba8cf45d7c4c7bcfc2884d80a664a7d5969961d0e005`. Year2022; selectedweight 0.1, selectedAUC 0.9938717710733546, SHOPanchor 0.9938546734526688, matched originalLGB608 0.9938428198540179. Runtime 1297.663s, peakRSS 6454132736bytes. Full sources/data/splits/samples/config/checkpoints/replays/environment/gates: `artifacts/lgb-shop664-2022-001/manifest.json`. No final training/submission;2022 requires independent eligible2021lock.

Completed2022 manifest SHA `fa10113f17fe184559921fd9b248aeb77245aa11ccc89dbd1318855344642adb`; actual locked0.1 improved AUC by0.0000170976206858 versus the acceptedSHOP anchor and0.0000289512193367 versus matched oldLGB608. All14 rawfloat32 arrays derived from exactfloat64 checkpoint replay. No new weight or early stopping. Process exited (read-only pgrep returned no match); heavy slot handed to independent actual result audit. Conditional final remains unsealed until that audit passes.


## lgb-shop664-2022-independent-audit-001 — 2026-11-02T20:24:57.583550+00:00

Status passed; accepted true at the unchanged2021weight0.1. Independent generic projection and rankAUC reproduce candidate0.9938717710733546 >SHOPanchor0.9938546734526688 and matchedrawLGB608control0.9938428198540179. Only0/0.1 evaluated. Original2022labels, exact171427-row purgedsample, source/input/artifact/seal closure and native350-tree/664feature/CPU2thread configuration verified; fresh664feature reconstruction replayed all14 checkpoints and every rawfloat32 probability exactly. Command `['/Users/heap/programming/tdt4173/.venv-ml/bin/python', 'scripts/audit_lgb_shop664_result.py', '--year', '2022', '--candidate-run', 'artifacts/lgb-shop664-2022-001', '--candidate-manifest-sha256', 'fa10113f17fe184559921fd9b248aeb77245aa11ccc89dbd1318855344642adb', '--run-id', 'lgb-shop664-2022-independent-audit-001', '--selection-lock', 'artifacts/lgb-shop664-2021-001/selection_2021.json', '--selection-lock-sha256', 'cc4c7e21967d61bbb6c9139ff32a6da2586a8686156d520f3bccf3eaab28a1b9']`. Runtime 13.949s; peakRSS 5785829376bytes. Exact proof `artifacts/lgb-shop664-2022-independent-audit-001/manifest.json`, SHA `35f65d33e51cd3ee88d3b477f5b7cd3fa0e02c90ad9bfaf0bd318013e8a22d76`. No refit, newweight or remoteaction; producerPID16081 and auditprocess exited. This permits conditionalfinal protocol sealing and review, not an unverified submission.

## lgb-shop664-final-preflight-001 — 2026-11-02T20:25:40.996252+00:00

Status passed_preflight_only; conditional acceptedLGB664 final/replay, original196358-row sample,14×350trees/2CPUthreads. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train_lgb_shop664_final.py --protocol docs/lgb_shop664_final_protocol.json --protocol-sha256 5513e67123a968ded3790ec6bcda898d50d3682685fa7f62020ea25108e322ad --run-id lgb-shop664-final-preflight-001 --preflight`. Runtime 0.315s, peakRSS 189136896. Exact source/input/checkpoint/acceptance/replay evidence in `artifacts/lgb-shop664-final-preflight-001/manifest.json`. No submission performed by trainer.

Actual accepted2022 proof `35f65d33e51cd3ee88d3b477f5b7cd3fa0e02c90ad9bfaf0bd318013e8a22d76` allowed sealing `docs/lgb_shop664_final_protocol.json`, SHA `5513e67123a968ded3790ec6bcda898d50d3682685fa7f62020ea25108e322ad`. Builder command: `.venv-ml/bin/python scripts/prepare_lgb_shop664_final.py --confirmation artifacts/lgb-shop664-2022-001 --confirmation-manifest-sha256 fa10113f17fe184559921fd9b248aeb77245aa11ccc89dbd1318855344642adb --confirmation-audit artifacts/lgb-shop664-2022-independent-audit-001/manifest.json --confirmation-audit-sha256 35f65d33e51cd3ee88d3b477f5b7cd3fa0e02c90ad9bfaf0bd318013e8a22d76`. Source/template bytes unchanged from independently reviewed version; exact original196358-row sample,14×350trees/2CPUthreads, current projectedSHOP anchor and locked0.1. Metadata/sample/source-only preflight passed; actual finalprotocol byteaudit and resourceguard precede fitting.


## lgb-shop664-final-protocol-independent-audit-001 — 2026-11-02T20:26:33.752101+00:00

Actual sealedfinal protocol independently passed: `5513e67123a968ded3790ec6bcda898d50d3682685fa7f62020ea25108e322ad`. Actual2021+2022 complete acceptedmanifest/independentproofs, fixed0.1lock, sourcebytes unchanged from reviewedtemplates, original196358-row sample independently reproduced again, old350treeparameters/2threads, currentSHOPanchor and resourcegates all verified. Metadata-only producerpreflight6403e747…closure checked. Independent source-proof `artifacts/lgb-shop664-final-protocol-independent-audit-001/manifest.json`, SHA `191f57b14b4bf85d4b7f54fb44806dd3131dfe28f472e9d17991fb644272897f`, runtime 0.323s; command `.venv-ml/bin/python scripts/audit_lgb_shop664_final_protocol.py`. No featurebuild, labels, fit or submission. Separate actualfinal full121800-row/all14checkpoint replay and exactCSV audit still required; preparedauditor source `a84739888990f64d99ac1f0057c9b7ac28c75cdedbf0193880b9514f4486fe46` was peer-reviewed byshadow.


## hyljelihyl-river-cost-v2-result-independent-audit-001 — 2026-11-02T20:27:03.699263+00:00

Actual river result audit passed; source/command/input/result/environment and all failure evidence in `artifacts/hyljelihyl-river-cost-v2-result-independent-audit-001/manifest.json`. Exact command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_hyljelihyl_river_results.py --run artifacts/hyljelihyl-river-cost-v2-diagnostic-001 --manifest-sha256 fbb1ac2019fd0a8bf81695d7bc7959dc5d887c6149ecd55f74be60383e3ed401 --run-id hyljelihyl-river-cost-v2-result-independent-audit-001`. No optimization/model fit; only twelve pre2021 target rows after full physical verification. Runtime 3.036s. No leaderboard claim or automatic promotion.


## lgb-shop664-final-001 launch — 2026-11-02T20:27:37.004252+00:00

Actual accepted2021/2022 and final protocol byteaudit `191f57b14b4bf85d4b7f54fb44806dd3131dfe28f472e9d17991fb644272897f` passed. Root explicitly authorized final fit; launched session70549 with exact196358 sampledhours,14×350trees/2CPUthreads and locked0.1 nestedSHOPblend. Command `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/train_lgb_shop664_final.py --protocol docs/lgb_shop664_final_protocol.json --protocol-sha256 5513e67123a968ded3790ec6bcda898d50d3682685fa7f62020ea25108e322ad --run-id lgb-shop664-final-001 --independent-audit artifacts/lgb-shop664-final-protocol-independent-audit-001/manifest.json --independent-audit-sha256 191f57b14b4bf85d4b7f54fb44806dd3131dfe28f472e9d17991fb644272897f > artifacts/lgb-shop664-final-001.log 2>&1`. Source/input/sample/acceptance closure in sealed protocol `5513e67123a968ded3790ec6bcda898d50d3682685fa7f62020ea25108e322ad`; original40% sample/350tree parameters unchanged. Enforced40%free beforebuild/20%perunit. Launch observation `artifacts/lgb-shop664-final-launch-observation-001/manifest.json`, SHA `f63422683824d784eb648dc11713d1defd6370a304d90c21aef41216fad526ee`. Persistent log `artifacts/lgb-shop664-final-001.log`. No final result/export/submission yet; independent actualcheckpoint and CSV gates remain.


## cuda-vector14-smoke-independent-review-001 — 2026-11-02T20:32:20.956057+00:00

Root independent source/wrapper/notebook byte review PASSED; no GPU execution or competitiondata. Pins2048syntheticrows,664features,14labels,8depth3sharedtrees, CUDA1/nativevectorleaf anddistinctprocessUBJ/JSONreplaychecks,120sphasecaps, immutablefailure/successarchive. Exactreview andunmeasurednative-schema assumptions in `artifacts/cuda-vector14-smoke-independent-review-001/manifest.json`, SHA `971cdfb18711dd69b037b79bb57834848d49c8a8891e038ee868282b4f86a006`. Official3.4.1release docs support intended mechanism, but actualsmoke required. Fullfoldprotocol/fit notauthorized.


## cuda-vector14-smoke-notebook-001 — 2026-11-02T20:30:11.304325+00:00

Prepared only; no model executed. Root approved a synthetic-only shared14-output vector-leaf CUDA/serialization feasibility notebook, after read-only inventory found no prior shared-tree experiment. Official XGBoost3.4.1 CPU/CUDA hist and multilabel documentation is linked in `docs/xgb_vector_leaf_proposal.md`. Exact2048×664synthetic inputs/14binaryoutputs,8depth3trees,seed4173,originalotherparams/CUDA1; freshofflinevenv/pinnedhostruntime; distinct fit/replay processes must prove native14vectorleaves/8trees, finiteprobabilities and bitwiseUBJ+JSONreplay.120s phasecaps and durablefailurearchive. No competitioninputs, validationmetrics, fullfoldprotocol, parametersearch or submission. Nativeconfig/JSON fields remain unverified until actualhosted smoke; fail closed on mismatch. Command `.venv-ml/bin/python scripts/prepare_cuda_vector14_smoke_notebook.py`. NotebookSHA `8acac6006529c51b36db2ac7d5b74ba9b1beb70bd5123c728b998077c7ab85f9`; source/embeddedwrapper/allartifact proof `artifacts/cuda-vector14-smoke-notebook-001/manifest.json`, SHA `5a83eb71c8f71767beddac87dcb7bf5d730321497ba03a894b60ff95d3c33c93`. Root owns independentreview and laterhosted scheduling afterUIrecovery; queuedphysics/finalreturns remain ahead.


## hyljelihyl-river-cost-36-independent-preflight-001 — 2026-11-02T20:32:45.163654+00:00

Status passed,7.458294seconds. Proof `artifacts/hyljelihyl-river-cost-36-independent-preflight-001/manifest.json`, SHA `e52b665b57f3e0b4c102ee3924d3760f4c504495b03e3dba10e1c0b855226544`. Exact protocol `380911fa5b18556cf18a734f6f38e24d3aea01b7705bdc95ed6e9e30b03966a8`, preflight `9c644929136e2f4ff04e9e07c97e30783c08b0a7c17093eacf776c3d13222091`, original36metadata IDs/blocks and twelve whole-directory reused byteclosures verified. Independently rebuilt all36 original and split physical inputs and matrix signatures; all windows end by2021-01-01. Exactly24newjobs, fixed v2 math/coefficients/resource/replay/gates, worker and selected-label-loader AST unchanged. Two reuse tests independently pass, including changed/missing/extra payload rejection. No real optimization, labels, fitting or promotion. Command/source/input/environment and allcase evidence in proof; ready for root-authorized one-worker actual diagnostic after fresh headroom check.


## cuda-vector14-smoke-return-preparation-001 — 2026-11-02T20:37:31.443181+00:00

Prepared strictsyntheticreturn verifier only; no actualarchive imported or model fit. Externalarchive/threehostedmanifest hashes, exact15-file namespace, reviewedsource/nativeCUDA/14vectorleaf/8tree/parameters, regeneratedsyntheticinputs/labels, distinctprocessUBJ+JSONbitwisereplay and independentNumPyJSONforest arithmetic are required. LocalXGBoost3.2.0 is intentionally neither imported nor used to load hosted3.4.1 artifacts. Four tiny tree/leafmapping/rejection/archivetests passed0.002s; source compiles. Proof `artifacts/cuda-vector14-smoke-return-preparation-001/manifest.json` SHA `e77e04e1e4861dea7214019f33824483ffc8a47a2d8d77ef25a2850166939851`. Root source-only smoke approval971cdfb… is documented. No fullfoldprotocol exists; before results root proposed possibly freezing600/1800rounds jointly withthe2021blend grid because equalroundsharedtrees canunderfit. Neither realfit is implemented/authorized here.


## hyljelihyl-river-cost-36-result-independent-audit-001 — 2026-11-02T20:37:39.308462+00:00

Actual river result audit passed; source/command/input/result/environment and all failure evidence in `artifacts/hyljelihyl-river-cost-36-result-independent-audit-001/manifest.json`. Exact command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_hyljelihyl_river_36_results.py --run artifacts/hyljelihyl-river-cost-36-diagnostic-001 --manifest-sha256 f403ed698047e6e47e94160fac245b019e6e286f8d9256014187bf6d1d29b4e7 --run-id hyljelihyl-river-cost-36-result-independent-audit-001`. No optimization/model fit; only thirty-six pre2021 target rows after full physical verification. Runtime 8.183s. No leaderboard claim or automatic promotion.


## cuda-vector14-return-independent-source-review-001 / 002 — 2026-11-02T20:39:23.126040+00:00

Return verifier source review passed; four synthetic arithmetic/archive tests independently pass in0.002s. Proof `artifacts/cuda-vector14-return-independent-source-review-002/manifest.json`, SHA `6d2a04ad0046808de55435e3fbc89372860abef91220b43a215ad2aaf1b6b5cc`. Exact source53947eae534b6d1e45ee99ec4d781ce9ba1a31cae5eac70951fa0e95713b38e7; test command and snapshot retained. Checks15flatfiles, external4hashes, runtime/phase/worker/source/native14vector8tree closure, deterministic syntheticdata, hosted fresh-process replay and independentfloat32JSONprediction within3e-7. No real CUDA smoke, competitiondata, fit or fold authorization. Reviewer001 preserved a sourceSHA transcription typo (extra5e), fixed only reviewer metadata; candidate unchanged.


## cuda-vector14-return-source-peer-review-001 — 2026-11-02T20:39:49.647272+00:00

Shadow independently reviewed strictsyntheticreturn source and reran all4fixtures (.002s), passing. Exact namespace/provenance/runtime/parameters, syntheticregeneration, two-processserializedreplay and independentJSONvector arithmetic checked. No actualsmoke, competitiondata, cache featurebuild or fit. Recordedreview `artifacts/cuda-vector14-return-source-peer-review-001/manifest.json`, SHA `9fdbeaa5049913aeed4bb5bd8943529823cc5113363a98f6b3455b26ecc19d0c`.


## River36 decision — 2026-11-02T20:41:07.344449+00:00

Actual36-case river result independently passed, proof `0be8e29f96f08fe25c20a5b69182f1efff2a289e6a60ebafe715c9f13b14b85f`. C/A/B global qAUC .9474329228372067/.9446521745478594/.9463705482645913; errors7269/7894/7628.2019 qAUC counterexample retained. Root authorizes feature/execution-pilot preparation only, keeping62 formulas and exact664prefix; no full cache or learned acceptance. Full outcomes in `docs/hyljelihyl_river_cost_36_results.md`.


## verification-suite-20261102-2040 — 2026-11-02T20:41:57.418351+00:00

Full unittest status passed; exit 0, runtime 12.672s. Exact command, test-source inventory, environment and log hash in `artifacts/verification-suite-20261102-2040/manifest.json`. No fit, target evaluation or Kaggle submission.


## river726-learned-design-001 — 2026-11-02T20:42:05.509083+00:00

Conditional design only, root requested. Exact664prefix+62riverCfeatures for separate matchedLGB350 andGPUXGB600 arms; original40%samplebytes/fullparams, currentpublicbestSHOPanchor and same-architecture raw664controls. Fixed2021weights[0,.1,.25,.5]/lowerties; positivewinnermustbeatidentity+sameweightcontrol, independentlylocked2022 repeatsstrictgates. Full3647physically/replay/feature-auditedcache is mandatory; no partialfeatures or changedprefix. Reuse existinginfrastructure through reviewedextensioncontract, no newtrainer/protocol/fits/cacheaccess/scores. Design `docs/river726_learned_test_design.md` SHA `51cf3e7c0ebe558a948d1812bc4df943768057c75165858174835b1cb9bd3760`; metadata/source reference proof `artifacts/river726-learned-design-001/manifest.json` SHA `0633dfa692880f2b729622a1053f509fa95e46771f171e8c5ccddfacf8c9c529`. CurrentLGB664finalreplay/export stayspriority.


## hyljelihyl-river-persistent-independent-preflight-001 / 002 — 2026-11-02T20:46:01.413679+00:00

Independent59-case persistent-worker/62-feature preparation review PASSED6.469839seconds. Proof `artifacts/hyljelihyl-river-persistent-independent-preflight-002/manifest.json`, SHA `0000e4d1c8c054056a8a30d60c1d12a6ff5ee556c1e4bf203e0647da76986996`; protocol `a1d860345862538c22498ca675d25e8be6730fda6051255e5e450ba08f2e440e`, producer preflight `9b07b76654644fda4a072322d4926c2321c29ee5ac0c60141410625876cab421`. Exact36canonical+24linspace deterministic union59cases, fixed4173permutation versus ascending order, two separatelyspawned persistent1threadworkers/freshLPpercase reviewed. Child protocol/source/environment/preparedinputseal/control hashes checked; all59 original/splitinput bytes and Cmatrixsignatures independently rebuilt; all36 canonical62feature arrays independently reproduced bitwise. Four feature and two population/order tests pass. Future726loader source rejects incompletecache and requires separatelypinned3647case physical/replay/featureproof; no actualcompletecache or supervisedfit authorized. Reviewer001 failure `b1dcd959cb06bfc18e141f1b8149278d5f49121614cc32ec77272411340d6cd4` preserved: independent min/max clamp changed in-range -0.0 to+0.0 (numericerror0); fixed only reviewer conditionalclamp to preserve declared signedzero semantics. No real solves or labels. Root may launch only the input-only pilot after freshresourceguard; actualresult audit remains separate.


## integer-case2012-exact-certificate-001 — 2026-11-02T20:52:15.175685+00:00

Read-only exact-rational numerical certificate status passed_read_only_certificate. No solve, target decoding, retry or eligibility change. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/certify_integer_objective_gap.py --run-id integer-case2012-exact-certificate-001`. Exact operands/input hashes/rational components/source/environment and failure evidence in `artifacts/integer-case2012-exact-certificate-001/manifest.json`. Runtime 3.749s. Original case2012 failures remain failed under v2.

## lgb-shop664-final-001 — 2026-11-02T20:26:57.960210+00:00

Status complete; conditional acceptedLGB664 final/replay, original196358-row sample,14×350trees/2CPUthreads. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train_lgb_shop664_final.py --protocol docs/lgb_shop664_final_protocol.json --protocol-sha256 5513e67123a968ded3790ec6bcda898d50d3682685fa7f62020ea25108e322ad --run-id lgb-shop664-final-001 --independent-audit artifacts/lgb-shop664-final-protocol-independent-audit-001/manifest.json --independent-audit-sha256 191f57b14b4bf85d4b7f54fb44806dd3131dfe28f472e9d17991fb644272897f`. Runtime 1481.774s, peakRSS 5917081600. Exact source/input/checkpoint/acceptance/replay evidence in `artifacts/lgb-shop664-final-001/manifest.json`. No submission performed by trainer.

Final manifest SHA `2d3e2f85e40a7bbf8d849664a063a09a61eb30306746018935689754a6295ada`. All14 fixed350-tree models replayed every121800testhour probability exactly in float64; raw/replayfloat32 arrays identical. Runtime1481.773664seconds, peakRSS5917081600bytes; exact196358sample and locked0.1 retained. Session70549 exited0 and pgrep found no remaining finaltrainer; heavy slot released to independent actualfulltestreplay, then strict primaryexport and exactCSVproof. No submission by trainer.


## hyljelihyl-river-persistent-result-independent-audit-001 / 002 — 2026-11-02T20:52:15.434875+00:00

Actual pilot result audit PASSED6.009528seconds: `artifacts/hyljelihyl-river-persistent-result-independent-audit-002/manifest.json`, SHA `7fdf874bb958062424ffbb4278cacb4f5dc84de820a8a968386c9d8d8bd135f8`, actual pilot `134ad85d2533e33ac218f4be70c4ef5d246cef37fb77a4656f6c4e855538dc49`. All59cases×2workers/order/PID/source/input/environment/resource closures, all118native arrays/fullprimaldualKKT/bound complementarity/objective certificates, every decodedCfield including frozen negative-price averageenergy, originalAprojection feasibility and all independent62feature bytes verified without solves or labels. Full arrays match acrossworkerorders and all36canonical fresh-process cases. Both persistentPIDs23812/23813 absent from scoped process-family observation; actualworker manifests complete and parentjoin gate passed. Pilotwall22.954s, workerpeak330/332MB. No fullcache or learnedfit promotion by this audit. Reviewer001 `0c8c1f0ad3ddfc3edacbbb113c0d17e7407c863e5977aeb5829ce47306c6c08d` preserved nestedpgrep sandbox denial; direct permitted exec process observation `181a46aa3e064b830148876141410fec382e9d36b2a3e8a6bcbcfac660984d7d` used in002. Reviewer decode also completed frozen negative-price energy handling before successfulrun; candidate/source unchanged.


## 2026-11-02 — LGB664 final independent replay, primary export and exact CSV audit

After the producer's complete final manifest `2d3e2f85e40a7bbf8d849664a063a09a61eb30306746018935689754a6295ada` and process exit, ran `lgb-shop664-final-independent-audit-001` (32.729 s, 6237323264-byte peak RSS), `submission-lgb-shop664-anchor-blend-001` (1.788 s), and `submission-lgb-shop664-independent-audit-001` (1.973 s). All passed. Fresh 664-feature reconstruction, all fourteen native 350-tree configurations, exact 196358-row seed4173 sample, full 121800-row checkpoint replays, original IDs/schema/data/source/anchor closure and independent nested projection were verified. Both float64 checkpoint replays were bitwise identical and their float32 outputs matched the saved predictions. The separate CSV audit reconstructed every CSV byte with zero array difference; 725 rows, 2353 columns and [0,1] probabilities were preserved.

Frozen recipe: `P(.9 * saved_projected_public_best_SHOP664 + .1 * raw_LGB664)`. The existing locked validation scores remain 2021 0.9951042081969695 and 2022 0.9938717710733546. No new scores, tuning, fitting, labels or standalone diagnostic were introduced by the verification/export. CSV SHA256 `1c1d173816127997e711d54c8269406830666af66b6f4eb6d24f4f227b70d496`; primary export manifest `be4ec369caf7850c1259aee81b3344a738d62c51b9c547382fed9daa1cb6247c`; raw replay proof `b63b5872411fa59eaea6c7c6cbc132803b6ea3ae445c7f871b3df67875d976b8`; actual CSV proof `529c5b4e7a8bf1106882de2e0594c54de2d016ec979e5e69e84a723dbcfc3c99`. Exact commands, source/environment/data-to-export closure and reproduction notes are in `docs/lgb_shop664_final_execution.md` and each immutable manifest. Root notified that the CSV is ready. Kaggle submission ID, score and rank remain unobserved.


## root-river726-preparation-001 — 2026-11-02T21:03:31.671627+00:00

Prepared training/conditional protocol builder for exact664+62river features, reusing immutable664 closure/sample/reference/selection helpers. Four focused eligibility/lock/cache tests passed0.008s. No actualcache read, feature build, model fit or protocol emission. Full-cache completion+independent physical audit and source review still required. Source/environment/command evidence: `artifacts/root-river726-preparation-001/manifest.json`.


## integer-case2012-exact-certificate-independent-audit-001 — 2026-11-02T21:02:05.629546+00:00

Status passed,4.877032seconds. Exact source/result review proof `artifacts/integer-case2012-exact-certificate-independent-audit-001/manifest.json`, SHA `121b90dcf8394a9967aab609134b19cf29e8c99591d2fae4315a024317e7c926`. Independently reconstructed actualcase2012 physicalinputs/frozenmatrices/nativefeasibility+integrality/conditionalLPKKT/allreplica bytes. Separate sparse-row/transposed-dual Fraction arithmetic matches every rational bound component, identity, objective-reporting discrepancy and EURscale exactly. Bound3.990695556714721e-6EUR on reportedworsening3.8007611491910276e-6EUR; original1e-6scaled guard stillfailed and eligibilityunchanged. Ten focusedtests pass, including128randomexactidentities and nonfinite/wrong-sign/infinitebound cases. Initial9-test fixturefailure retained in `artifacts/integer-exact-bound-test-development-001`: expected4 omitted legitimate wrong-sign upperterm2; corrected testfixture only, certificateunchanged. Full command/source/input/test/environment evidence inproof; explanation `docs/integer_case2012_exact_certificate_review.md`. No solves, labels, retries or eligibilitychanges.


## 2026-11-02 — Case2012 numerical-completion preparation only

Prepared `scripts/complete_integer_case2012.py` SHA256 `5a739e4e78546098fb46751f7f81d84012d5bd71390700cd23ebba081b25638d` and froze `docs/integer_case2012_completion_protocol.json` SHA256 `6499a285d267e8ccab0cb2063e682880c8837423fdd8606d9c83d11063cff989`. Preparation manifest `artifacts/integer-case2012-completion-preflight-001/manifest.json` SHA256 `c231d938940e8d29cef2db0717fa7a7c65ffede725a16cd0450a9c41b0c424cf` preserves all commands, source snapshots, logs, environment and dependency hashes. Four focused rejection tests passed: wrong case/status/time/failure, native status1, malformed scale/dimensions, self-consistent but altered payloads and altered native bound. Source/data/original-failure closure preflight passed. Commands: `.venv-ml/bin/python scripts/complete_integer_case2012.py --freeze-protocol`; `.venv-ml/bin/python -m unittest tests.test_integer_case2012_completion`; `.venv-ml/bin/python scripts/complete_integer_case2012.py --protocol-sha256 6499a285d267e8ccab0cb2063e682880c8837423fdd8606d9c83d11063cff989`.

No optimization, target decoding, model training, fresh retry, feature construction or aggregate eligibility change occurred. The new case2012-only contract requires two future fresh unchanged60s MILP/1s conditional-LP pairs and file/array identity with both preserved failed replicas, plus root's exact rational certificate within1e-5EUR. It pins independent certificate review `121b90dcf8394a9967aab609134b19cf29e8c99591d2fae4315a024317e7c926`. It explicitly preserves original and fresh attempt status failed, unchanged physical/KKT/zero-slack checks and a separate numerical-exception receipt. No ordinary complete.json is emitted. Later aggregate/748 use needs a separately reviewed new namespace/protocol/loader/auditor. Runtime requires independent completion-source/protocol proof and root authorization; neither execution nor acceptance is claimed by this preparation. Root notified for review.


## integer-case2012-completion-source-independent-audit-001 — 2026-11-02T21:06:25.879479+00:00

Status passed. Source/protocol review and no-solve saved-evidence rehearsal, four rejection tests. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_integer_case2012_completion.py`. Runtime 4.884s. No target decoding or eligibility mutation. Full source/data/protocol checks and reasoning in the audit manifest.


## integer-case2012-completion-result-independent-audit-001 — 2026-11-02T21:08:45.214201+00:00

Status passed; actual fresh payload/native/KKT/exact-rational bound and independently decoded result audit. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_integer_case2012_result.py`; runtime 3.897s. No optimizer or target reads. Old failures and aggregate ineligibility preserved; full evidence in manifest.


## 2026-11-02 — Case2012 authorized fresh completion and independent acceptance

Root authorized the actual frozen two-process completion after source/protocol review `d9a797b58bdcf9f71921ef4bc9927bcca0adc6a40a4b2650af0042d6a8041888`. Run `integer-case2012-completion-001` completed in7.030818s and its parent exited0; fresh PIDs26758/26759 were independently observed absent after exit. Two unchanged60s native MILP calls and four unchanged1s conditional-LP calls ran. All six fresh NPZ files and arrays exactly matched BOTH original failed replicas. All four LP calls actually reported6577 iterations. Both recomputed exact rational bounds were3.990695556714721e-6EUR; unchanged physical/KKT/zero-slack checks passed. Original and fresh attempt proof statuses remain failed. No solver/source/parameter/tolerance/cache mutation occurred.

Completion manifest `artifacts/integer-case2012-completion-001/manifest.json` SHA256 `7da83d48119d48a0826a6d3edf01a265ee932edb8ec7f899be6dc2bb9dce8984`; exception result SHA256 `753aeb3b827d10583c999e44e391da2815f186ff7e728f65ab9bb61e4c9921fc`. Root independently audited actual fresh payload identity, preserved original failures, all decoded physical fields, KKT, exact Fraction components using shadow's separate implementation and process exits: `artifacts/integer-case2012-completion-result-independent-audit-001/manifest.json` SHA256 `2708cafb0b2ed2ae7f0b43ee1515d63dc80ebcd3908109277a2a7977fde66b02`, passed. This approves only the explicit new case2012 exception. Original v2 and full aggregate remain ineligible; any aggregate requires a separately frozen/reviewed namespace and lineage. No labels, features, model, score or submission were produced.

Exact command:

```sh
/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/complete_integer_case2012.py --protocol-sha256 6499a285d267e8ccab0cb2063e682880c8837423fdd8606d9c83d11063cff989 --execute --run-id integer-case2012-completion-001 --certificate-audit artifacts/integer-case2012-exact-certificate-independent-audit-001/manifest.json --certificate-audit-sha256 121b90dcf8394a9967aab609134b19cf29e8c99591d2fae4315a024317e7c926 --completion-audit artifacts/integer-case2012-completion-source-independent-audit-001/manifest.json --completion-audit-sha256 d9a797b58bdcf9f71921ef4bc9927bcca0adc6a40a4b2650af0042d6a8041888
```

Full original/fresh source, protocol, environment, data/control/evidence hashes, native bounds, exact certificate components and LP metadata are preserved in the completion manifest and referenced proofs.


## root-river726-preparation-002 — 2026-11-02T21:10:42.080390+00:00

Added actualrawLGB664 independent-control audit binding before any protocol/fitting. Five focused tests pass; exact sources snapshotted under `artifacts/root-river726-preparation-002/source`; command/output/hashes in manifest/tests.txt. No cache/labels/model read. Prior preparation001preserved, superseded for futureexecution.


## 2026-11-02 — LGB726 independent result-auditor preparation only

Prepared `scripts/audit_lgb_river726_result.py` SHA256 `d8b1407ed24b3b6c5a9e8a1845a0ec7c7188532045784457553f9633b57c9081`, reusing original664 auditor hash8603a85b… and independent rank-AUC/min-max-isotonic helper0d191dba…. Preflight `artifacts/lgb-river726-result-auditor-preflight-001/manifest.json` SHA256 `4a1d70ac0a42ef0261bcdd65d87ca22cbd4fa1693ecc987d37115c9d39b81c8a`; command `.venv-ml/bin/python -m unittest tests.test_lgb_river726_result_audit` passed4 synthetic tests. Tests cover lower-weight ties, strict matched-control gain, changed protocol/reference pins, original prefix dtype/value/order changes and selected-blend substitution for raw664 control. Python3.11.15/macOS27.0.1arm64/NumPy2.4.6/pandas3.0.6/LightGBM4.7.0; two numerical threads. No actual726protocol exists, and no physical feature array, label, model, score or final artifact was opened or produced. Complete-cache proof, actualraw664 control audits, source/data/sample closure, fullcheckpoint replays and fixed chronological gate remain runtime requirements; independent source review pending. Root trainer files were untouched. Invocation contract is `docs/lgb_river726_result_audit.md`.


## lgb-river726-result-auditor-source-review-001 — 2026-11-02T21:14:24.842877+00:00

Root source review passed; no cache/input arrays/model opened. Originalsource/independenthelper/tests/preparation pins verified, targetlock ordering, exact664prefix and fullfold checkpoints/metric reconstruction reviewed. Actual runtime proof remains required. See manifest.


## hyljelihyl-river-full-cache-v2-independent-preflight-001 / launch-review-001 — 2026-11-02T21:17:30.444384+00:00

Independent fresh reconstruction of all3,647 original and split physical inputs, metadata dates, full C matrices, all source/data/control pins and59 independent feature reconstructions passed209.740s; seven lifecycle/receipt tests passed. Exact command/environment/artifacts: `artifacts/hyljelihyl-river-full-cache-v2-independent-preflight-001/manifest.json` SHA `68bc2d9fbfa19f390b09e55f2e41e7085270a1ecd3012e7d3f6337738367dc32`. No solves or labels. **Launch approval withheld** after a separate source-review finding: complete replicas with unequal bytes were not retained as failed pairs and could be retried on operational resume. V3 is required. Explicit decision: `artifacts/hyljelihyl-river-full-cache-v2-launch-review-001/manifest.json`, SHA `05bcd5134dcc51a8e4155f3713c72c5044f435d4e921a2e3dcacc501d1c5dd7f`. Unrun v2 is preserved.


## lgb-river726-template-independent-audit-001 — 2026-11-02T21:18:35.901641+00:00

Template-only source/control audit passed; five focused tests. Exact originalLGB664 recipe, auditedraw664controls and currentSHOPanchor; additive62requires independentlyacceptedcomplete3647cache. No cache read, featurebuild, labels, fit or protocolseal. Source snapshots/controlpins/checks: `artifacts/lgb-river726-template-independent-audit-001/manifest.json` SHA `eb34cb9cfbdde95aee2e7fec1276fff606b74467f2867fdfd07731c16cf26850`.


## hyljelihyl-river-full-cache-v3-independent-preflight-001 — 2026-11-02T21:19:33.803033+00:00

Independent minimal v3 repair/source/input-compatibility audit passed6.237s with eight focused tests. Protocol `e4b7bdf6954838739174f767a0a69ab110317fef069e646ac8725ab8cc9e187e`; proof `artifacts/hyljelihyl-river-full-cache-v3-independent-preflight-001/manifest.json`, SHA `f2068d9835f78d5ed1fb88e332b523053d71e9f43ebc2c241ae153255a71f75b`. Exact prior audited3647 prepared inputs are reused under pinned v2 manifest/protocol/inputaudit/blockedreview; full file hashes and unchanged source/data/metadata/formula/solver fields plus identical input/worker ASTs verified. Durable immutable pair-verification failure records now block retries; only clean resource drains may resume. No optimization or labels in this review. Root authorization remains the execution gate; complete-cache physical audit remains the learned-feature gate.


## hyljelihyl-river-full-cache-auditor-preparation-001 — 2026-11-02T21:22:30.406979+00:00

Prepared independent full3647-cache audit source and five focused lineage/path/resource rejection tests, all pass. Exact source/helper pins and tests: `artifacts/hyljelihyl-river-full-cache-auditor-preparation-001/manifest.json` SHA `fe907aa41636b09e6bb03f58386dfdfc3decc8ff3dfeafa8857dbec3aa2c8814`. Actual cache not opened, no solves/labels/training; independent source review pending and actual full result requires complete manifest and process-exit evidence.


## cpu-vector14-smoke-result-independent-audit-001 — 2026-11-02T21:25:58.715785+00:00

Root verified actualCPU8-tree/14-output smoke source/runtime/file lineage, regeneratedsynthetic arrays byteexact, confirmed exitedworkers and independentJSONforest predictions(error=0.0). No competitiondata/model fit, noGPUclaim. Actualproof manifest contains externalphase/environment/helperpins.


## 2026-11-02 — Isolated XGBoost3.4.1 CPU vector14 capability

Official registry confirmed the3.4.1 macOSARM64 wheel requiringPython>=3.12. Created separate `.venv-xgb341-cpu` with existingPython3.14.6 and exact checksum-verified official wheels; original `.venv-ml` remains unchanged. Environment manifest2094900a418de72fa78499a0ad92b2fab2019c24002f60edfa048d17bec0b92e preserves exact package/native/runtime/install provenance. Direct pip DNS failure and initial missing narwhals/cloudpickle dependency check are preserved; pinned offline installation then passed pipcheck. Smoke001 failed on a denied subprocess ps monitor before any completed model, preserved46aa4891f88084b2662075b23463ab1d1b27c7554fff3e3ffb1a8987c92d0a8a. Unchanged synthetic/model/package settings were rerun under a separate wrapper revision with hard120s phase timeout and OSru_maxrss acceptance<=2GiB, without live-RSS-enforcement claim.

Authorized smoke002 PASSED:2048×664 float32 syntheticinputs,14binarylabels,seed4173,8shareddepth3CPUhisttrees,lr.05/minchild20/L2=4/maxbin128/row+column.9/2threads. Actual14-vector leaf structure, two independent serialization processes and NumPy JSON arithmetic verified. Fit0.437176s, total2.398978s, peak381075456bytes, UBJ+JSON predictions bitwise exact and independent JSON error0. Manifestb160fb7ef6784a3e64a7ce6b5a326a6353f49291e46355b32cb799c5faac482f; root actual independentproof6bff707f56cc3deb552fbcf0c21afe9a86f15224da18f43db93c4cc5f1fefb2a passed and PIDs exited. Exact sources/protocol/data-generation/commands/model/prediction hashes are recorded in `docs/cpu_vector14_smoke_result.md` and `artifacts/cpu-vector14-smoke-002`. No competitiondata/labels/fullfold/GPU/score claim. A realistic-depth synthetic timing pilot is authorized; fullfold remains preparation only.

## 2026-11-02 — Complete river-cache actual auditor independent source review

Reviewed `scripts/audit_hyljelihyl_river_full_cache.py` source40947e594d8a89ec3a0396b691956cfe34ec4578547966a69d211740cb3ee955 against frozenv3e4b7bdf6954838739174f767a0a69ab110317fef069e646ac8725ab8cc9e187e. Proof `artifacts/hyljelihyl-river-full-cache-auditor-independent-source-review-001/manifest.json` SHA256c8b4787f5a2880d720cba4aa5889473e08b6a51c0bcd40c645f896152f5e0ac3 passed SOURCE ONLY. Review closed helper-hash, symlink, originalpilot PID ordering, exact inventory and post-exit observation-lineage gaps. V3 retained pair-failure records and non-resource resumes are rejected. Five tests independently passed, plus completed-pilot cases1/2186×bothreplicas verified KKT/all decoded fields/Aprojection/62feature bytes and eight corruptprimal/dual rejections. No optimizer, labels or livecache/partialfeatures were used. Complete3647-case actual audit remains required after coordinator/workers exit.


## continuation-root-2133 — 2026-11-02T21:32:55.374748+00:00

Currentturn PROGRESS. Actual newCPUvector capability independentlyverified, river fullcache002 actively solving alongside originalintegercache, numericalcase2012freshcompletionacceptedseparately, newLGB726training/audittemplates reviewed. Lastconfirmedpublic0.99651/rank18; zero newsubmissions due persistentBravecgWindowNotFound. Latest owner/liveobservation under `artifacts/continuation-live-check-20261102-2128/next_turn_observation.json`. Goalactive/unmet, no transfer.


## integer-water-budget-aggregate-source-independent-audit-001 — 2026-11-02T21:36:00.611204+00:00

Status passed; root review of frozen separate aggregate builder/loader and three actualpreflight receipts. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_integer_aggregate_preparation.py`. Five eligibility/preservation tests, zero solves/target decoding/fullaggregate approval. Runtime 1.044s. Full evidence/source snapshots in manifest directory.


## integer-aggregate-auditor-preparation-001 / exception-golden-001 / receipt-golden-001 — 2026-11-02T21:36:43.353191+00:00

Prepared the separate full3647 integeraggregate actual auditor; originalcache remains ineligible. Five synthetic path/provenance/exit tests passed. No-solve golden exception check independently recomputed both freshfailed2012 replicas against both oldfailedreplicas, native feasibility/integrality, conditionalKKT, exactrationalbound and every decodedphysicalfield:4.087s, proof `artifacts/integer-aggregate-auditor-exception-golden-001/manifest.json` SHA `dabe2299405ec81a2e30a5e721e65fdc553be5bb9ed63b27e9e48881b115d44d`. All three producerpreflight examples1(imported)/1000(ordinary)/2012(exception) then passed the actual receiptcontract, unchanged originalaudit_case/636compatibility or independentexception path, and84featurebytes:4.148s, proof `artifacts/integer-aggregate-auditor-receipt-golden-001/manifest.json` SHA `6494359307a4963b7ade29653e880db903486198f0e476064da6215f06f1f4c7`. No actualaggregate, targetvalues or optimizer opened. Full source/helperpins/test evidence: `artifacts/integer-aggregate-auditor-preparation-001/manifest.json`, SHA `b1cac984cbc8cc8c745e9be37992703cae1ebefe6a17c4fccfac97ad743d9785`. Independent source review pending; originalfinalization/exits and completeaggregate mandatory for actualaudit.


<!-- imported:artifacts/hyljelihyl-river-full-cache-001/experiment_record.md -->

## hyljelihyl-river-full-cache-001 — 2026-11-02T21:20:27.742117+00:00

Status incomplete_operational; C-only river62 all-input cache, no labels/training. Source/data/execution/receipt/feature proof in `artifacts/hyljelihyl-river-full-cache-001/manifest.json`, SHA `40dad570f82858aa3520e36859d49ac93447498232978a5b6ec5be4466e6dff9`.


<!-- imported:artifacts/hyljelihyl-river-full-cache-002/experiment_record.md -->

## hyljelihyl-river-full-cache-002 — 2026-11-02T21:20:58.133242+00:00

Status complete; C-only river62 all-input cache, no labels/training. Source/data/execution/receipt/feature proof in `artifacts/hyljelihyl-river-full-cache-002/manifest.json`, SHA `3a195f72b9c837ab4aa14e06662be078f72699e44122c81697efe32628a3c738`.


<!-- imported:artifacts/integer-water-budget-aggregate-preflight-001/experiment_record.md -->

## integer-water-budget-aggregate-preflight-001 — 2026-11-02T21:32:35.723064+00:00

Status passed_preflight_only; separate input-only integer aggregate, zero solves/labels/training, original failures preserved. Full recipe/command/data/source/receipt provenance: `artifacts/integer-water-budget-aggregate-preflight-001/manifest.json`, SHA `0563e518a505935f26f18dcfa9d49955a375bc1620ae064f2c0e4a5fa809d902`.


<!-- root-aggregate-source-audit-20261102-2140 -->
## Integer aggregate builder and loader independent source approval — 2026-11-02T21:36:00.611204+00:00

Passed five focused tests and all three actual finalized no-solve preparation examples (imported1, ordinary1000, explicit exception2012). Source, protocol, input checksums, environment, exact command and review decisions are recorded in `artifacts/integer-water-budget-aggregate-source-independent-audit-001/manifest.json` (SHA256 `c4c63c6a805969308d3b32791f72b65f6845979806f7059a160406e9578154d3`). Runtime 1.044336334000036 seconds. No labels, solves, full aggregate acceptance, training or submission. Root authorizes assembly when the frozen finalized original-manifest and actual process-exit prerequisites hold; the original cache and its failed receipts remain unchanged. Full independent aggregate audit is mandatory before learned features.


## Integration tests after river and integer-aggregate additions — 2026-11-02T21:41:54.866417+00:00

All377 discovered tests passed in15.251s (exit0). Exact command and Python source fingerprints: `artifacts/integration-tests-20261102-2142/manifest.json`, SHA256 `502baaadd8ddf7c88bfd4ec6fb85a1dcc9562f971300f6933aae773dc32d18c6`. Existing SciPy HiGHS option-forwarding and pandas copy-keyword deprecation warnings only. No training or submission occurred.


<!-- imported:artifacts/cpu-vector14-timing-001/experiment_record.md -->
## cpu-vector14-timing-001

Timestamp: 2026-11-02T21:35:09.991252+00:00. Status: failed. Manifest SHA256 `1e8e99d2edab40236b5a6eaa976fa0abbc3315367e90002fcb3601577a6c99da`.

Fixed seed4173 synthetic10000×664 float32 inputs and14binary labels; CPU2threads, depth6,64rounds then separate16rounds; other original parameters unchanged,128bins.64fit22.11575975s,16fit6.487729791s,prefixcompare0.005627792s; every first16native tree payload, base score and prediction matched exactly. Worker48454 exited0 and subsequent ps returned no process. Total29.651329458s, peak2762899456B (2.573GiB), so the declared2GiB post-exit memory threshold FAILED; fresh replay was skipped. Preserved whole-run failure; no retry or relabeling. The120s combinedwallcap was not reached. Corrected directrow×round extrapolation:2021(146899rows)600≈50.762min/1800≈152.287min;2022(171427rows)600≈59.238min/1800≈177.714min. Earlier message attached600cost to1800 by mistake and was promptly corrected. These timings extrapolate syntheticwork and are not measured real-fold costs. No competitiondata/labels/fullfold, no score/publicsubmission.

Environment: Isolated Python3.14.6,macOS27.0.1 arm64,XGBoost3.4.1,NumPy2.4.6,SciPy1.17.1,scikit-learn1.9.1; full native/wheel closure artifacts/xgb341-cpu-environment-001/manifest.json SHA2094900a418de72fa78499a0ad92b2fab2019c24002f60edfa048d17bec0b92e.

Command: `/Users/heap/programming/tdt4173/.venv-xgb341-cpu/bin/python scripts/pilot_cpu_vector14_timing.py --protocol docs/cpu_vector14_timing_protocol.json --protocol-sha256 d0940b5e4d79137fbccf4183e23efda4f78d43bf62c50ed547075571073dcd95 --run-id cpu-vector14-timing-001`.

Source and data: immutable local snapshots and manifest artifact hashes; synthetic inputs only. No validation split, early stopping, ensemble tuning, Kaggle score or rank.


<!-- imported:artifacts/cpu-vector14-portability-preflight-001/experiment_record.md -->
## cpu-vector14-portability-preflight-001

Timestamp: 2026-11-02T21:40:33.466951+00:00. Status: failed_synthetic_preflight. Manifest SHA256 `ef374863f219401ab811216cc7f3fbfd2ef547b8678305c03016733099d7dbb7`.

CSR.toarray sums stored negative zero with positive zero, losing sign bits; actual native CSR.data preserves the correct original bits. Test correctly rejected bridge comparison. No frozen protocol existed. Synthetic mixed float32/float64/int64 frame only. Two tests: one rejected native-byte comparison and onepassed. The failure is verifier implementation evidence, not a competition/model result. No fit.

Environment: Unchanged .venv-ml Python3.11.15,macOS27.0.1 arm64,XGBoost3.2.0,pandas3.0.6,NumPy2.4.6.

Command: `.venv-ml/bin/python -m unittest tests.test_cpu_vector14_portability`.

Source and data: immutable local snapshots and manifest artifact hashes; synthetic inputs only. No validation split, early stopping, ensemble tuning, Kaggle score or rank.


<!-- imported:artifacts/cpu-vector14-portability-preflight-002/experiment_record.md -->
## cpu-vector14-portability-preflight-002

Timestamp: 2026-11-02T21:41:03.005029+00:00. Status: passed_synthetic_portability_only. Manifest SHA256 `7c140bf88438e893a892cb39645a72d80d1c66b96ae20e0680530ab31024919c`.

Synthetic seed4173 257×664 mixed float32/float64/int64 frame, shuffled exactroworder,signednegativezero,64-rowblocks. Producer .venv-ml compares actual DataFrame DMatrix CSR.data against float32NPY; isolated3.4.1 consumer loads sameNPY and each nativeCSR inputblock has identical bytes. Exact finite dense CSRstructurechecked before reshapingvalues. Two focused tests pass. Firstpreflight preserved separately. No competitionfeatures/labels, no model fit, no fullfoldauthorization. Actual664bridge still requires frozenrecipe/review and original purgedsamples.

Environment: Producer Python3.11.15,XGB3.2.0,pandas3.0.6,NumPy2.4.6; consumer Python3.14.6,XGB3.4.1,NumPy2.4.6. Both nativearm64macOS27.0.1, full consumer environment in manifest.

Command: `Producer .venv-ml/bin/python heredoc invoked write_dataframe_rows(frame,rows,synthetic_input.npy,names,xgb,64); consumer .venv-xgb341-cpu/bin/python heredoc invoked verify_dmatrix_blocks(memmap,names,xgb,64). Exact function sources and all inputs/rowNPY/blockhashes preserved in this directory.`.

Source and data: immutable local snapshots and manifest artifact hashes; synthetic inputs only. No validation split, early stopping, ensemble tuning, Kaggle score or rank.


## lgb-river726-2021-preflight-001 — 2026-11-02T21:46:21.822402+00:00

Status passed_preflight_only; frozen664+62 river feature test. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train_lgb_river726.py --protocol docs/lgb_river726_protocol.json --protocol-sha256 c71d6a026f7ca3e802464df4be7b8874ab8592f54d38d85cd97e2b9040762b7c --year 2021 --run-id lgb-river726-2021-preflight-001 --preflight`. Year 2021, selected weight None, AUC None, anchor None, matched LGB664 None. 350trees/31leaves/lr.06/minleaf100/L2=1/featurefraction.9/seed4173/2threads/exact40% sample; no early stopping. Runtime 0.338s; peakRSS 209682432bytes. Exact source/data/cache/split/config/replay evidence: `artifacts/lgb-river726-2021-preflight-001/manifest.json`. No public score.


## hyljelihyl-river-full-cache-independent-audit-001 — 2026-11-02T21:38:48.479775+00:00

Complete independent physical/feature audit PASSED398.957s, process exited0. All3,647cases and7,294savedcalls checked: nativeoutput identity, fullLPprimal/dual/KKT, every independentlydecodedphysicalfield, exactzero slacks, projection intooriginalAconstraints, bothreplicaarraybytes, independentlyreconstructed62featurebytes andaggregatealignment. Exact59pilotimports/36canonicalfreshprocess lineage, onephase/distinctPIDs/orderedexecution/receipts andworkerexits checked. No solves/labels/training. Cache `artifacts/hyljelihyl-river-full-cache-002/manifest.json` SHA `3a195f72b9c837ab4aa14e06662be078f72699e44122c81697efe32628a3c738`; frozenprotocol `e4b7bdf6954838739174f767a0a69ab110317fef069e646ac8725ab8cc9e187e`;features SHA `0d9b6c20e0054ae3f2380cdd96579c69c22f1eb71a6b4ec411babc717ec39191`. Audit `artifacts/hyljelihyl-river-full-cache-independent-audit-001/manifest.json` SHA `2111ce26a75932f994819781e547e4c9f881acaea9cbfb8e37210ad5737f86fd`; exact command andsource/helper/environment closure retained. External finalmanifest-bound processobservation `artifacts/hyljelihyl-river-full-cache-exit-observation-001/manifest.json` SHA `76687dc8ba1a154f076f0f66c306abbfae27c8c653866d1f1f735918085eb61e`;workers28370/28373 absent andbothjoined0. Input-feature acceptance only; no learned score/publicscore inferred.


## lgb-river726-protocol-independent-audit-001 — 2026-11-02T21:46:51.282769+00:00

Actual frozenprotocol review PASSED0.352s, process exited0. Exact reviewed trainer/builder/physicalfeature source, completedrivercache002+independentfullphysicalaudit2111ce26…, original664prefix+62schema, original350tree31leaf/.06/min100/.9feature/L2=1/2thread/seed4173 recipe, original146899/171427samplebytes and167hourpurge, savedSHOPanchor and both auditedrawLGB664controls. Metadata2021preflight `acce4b3fb419b4846eb42438a95b0031ccc26f7528d092418ddc5cdcabd519c8` verified. No featurebuild, targetdecode or fit. Actualprotocol `docs/lgb_river726_protocol.json` SHA `c71d6a026f7ca3e802464df4be7b8874ab8592f54d38d85cd97e2b9040762b7c`; review `artifacts/lgb-river726-protocol-independent-audit-001/manifest.json` SHA `f0df370c38ac821e3399f4b693f58a3c744b0f5ca0ae212a9d3461d4ce749892`. Frozen2021fit gate clear;2022 still requires independently eligible actual2021lock.


## shop664-pair-error-diagnostic-001 — 2026-11-02T21:49:23.558243+00:00

Status complete; read-only error decomposition of current SHOP664 and already-fixed LGB664 blend. Runtime 2.145s; no fitting, new recipe selection or submission. Exact commands, input/source/environment hashes, case IDs and directed pair matrices: `artifacts/shop664-pair-error-diagnostic-001/manifest.json`. Both years are reused development data.


## shop664-pair-error-independent-audit-001 — 2026-11-02T21:51:46.910989+00:00

Independent directedpair/error attribution audit PASSED3.887s, savedarrays only. Concatenatedaverage-rank Mann–Whitney independently reconstructed each14x14 pairpopulation/error matrix forboth2021/2022 andfixedSHOP/LGB candidates, globalandwithinunitAUC, endpointshares andunitrankings, plusallsource/input/artifacthashes andtie fixture. CurrentSHOP/unchangedLGB topfiveendpointunits:2021Haukeli,Hogga,VinjeG3,Lio,VinjeG2;2022Lio,Haukeli,Hogga,VinjeG3,Kjela. Diagnosticmanifest77d9b016… verified. Exactproof/source/metrics: `artifacts/shop664-pair-error-independent-audit-001/manifest.json` SHA `a9dc28fbad0e51dbd5c84b557bcdec4e58682ec1220191191ef7159737d7bb19`. No fit, newweight, calibration orsplice. Bothyearsreuseddevelopment evidence.


## lgb-river726-final-template-preparation — 2026-11-02

Conditional final trainer, protocol builder, independent raw-model/CSV auditor and strict exporter are prepared without emitting a final protocol or building features. The original 196,358-row sample, 350-tree recipe, two threads, actual audited 664+62 river features and current projected SHOP anchor are fixed; the actual positive 2021 weight and unchanged independently accepted 2022 result are required before sealing. All 14 final checkpoints must replay every 121,800 test row, followed by an independent nested-projection/CSV proof. No fitting, new validation score or submission. Exact six-file source snapshot and three passing focused tests are retained in `artifacts/lgb-river726-final-template-independent-audit-001/manifest.json` (root independent template approval SHA `923ea2d11aa5406436003c332a88fd4936e2b1259fafbb8fd75e29f0bb267269`). Prior 664 sources remain unchanged; actual final-protocol review is still required.


## cpu-vector14-actual-auditor-independent-source-review-001 — 2026-11-02T22:03:28.354038+00:00

Independent source review PASSED in 0.396s. Exact auditor `e5c64878020eb325fc38d7afc8066a2f89a104a8666fb7e63bdc1d0889ea8340` and frozen prospective protocol/source/input closure verified. Three focused tests pass; all 10,000×14 saved synthetic predictions from 16 and 64 native vector trees reproduce byte-for-byte using independent JSON arithmetic; eight changed native recipe configurations rejected. Reviewed original664 bridge finite dense CSR/dtype/shape/sample/target identity, recorded environment/RSS, externally bound process exits, actual bridge/fit protocol/year/lock, distinct pinned-runtime fit/replay phases, trajectory prefix and native-format replay. No competition feature array, target decode or model fit. Previous failed timing pilot is preserved. Exact command, source snapshots and environment are in `artifacts/cpu-vector14-actual-auditor-independent-source-review-001/manifest.json`; manifest SHA `485e9f45073bf9b50c2f0152bac0d6f98b825d740418e07d5f8502e6f86dcfbb`. Source approval only; actual bridge/model audits remain required.


## LGB664-SHOP blend001 actual Kaggle result — 2026-11-02T22:04:37.100265+00:00

Submitted the independently audited primary CSV at22:02:46UTC; authenticated submission detail and leaderboard show **0.99652, rank18**, improving prior0.99651 by0.00001. CSV SHA256 `1c1d173816127997e711d54c8269406830666af66b6f4eb6d24f4f227b70d496`. Observation saved separately in `artifacts/submission-lgb-shop664-anchor-blend-001/kaggle_observation.json`; original manifests unchanged. Submission ID not exposed in observed UI. Leader scores0.99820/0.99820/0.99805. No final-selection checkbox changed. Browser access restored around22:01UTC; canceled stale notebook import before submission. Existing prospective experiments retain their frozenSHOP664 anchor despite the newpublicbest.

## integer-aggregate-auditor-independent-source-review-001 — 2026-11-02T21:36:43.018326+00:00

Status `failed`. Review001 stopped at stale expectedsource hash before checking candidate; exact prior evidence retained. Command: `.venv-ml/bin/python heredoc with stale b631f82a expectedsourcepin; traceback preserved`. Manifest `artifacts/integer-aggregate-auditor-independent-source-review-001/manifest.json`, SHA `fb5b24a2e2cc63a4e387df4cda5e8a56b5136919f716b3da190199080c145c9e`. Runtime 0.0007215409987111343s. Source/protocol/data/sample/input checksums and exact preserved output evidence inmanifest/snapshots. Environment unchangedmacOS27.0.1arm64/Python3.11.15/NumPy2.4.6/pandas3.0.6; no optimization/modeltraining here. No Kaggle submission, score or rank from this run.

## integer-aggregate-auditor-independent-source-review-002 — 2026-11-02T21:37:19.906492+00:00

Status `passed_source_review_only`. Source-only independent review passed: currenta441388a auditor, originalsource/protocol/helper closure,5tests and no-solve finalized preflightcases1/1000/2012 allnative/KKT/replica/Fraction/84feature checks. Actualcomplete3647aggregate remainsrequired. Command: `.venv-ml/bin/python artifacts/integer-aggregate-auditor-independent-source-review-002/review.py`. Manifest `artifacts/integer-aggregate-auditor-independent-source-review-002/manifest.json`, SHA `a3febff61ee79034e1ba9c48d3c3c0dea124ac731d87d1e60aa683a8a3e432d5`. Runtime 5.076157458001035s. Source/protocol/data/sample/input checksums and exact preserved output evidence inmanifest/snapshots. Environment unchangedmacOS27.0.1arm64/Python3.11.15/NumPy2.4.6/pandas3.0.6; no optimization/modeltraining here. No Kaggle submission, score or rank from this run.

## cpu-vector14-fold-preparation-001 — 2026-11-02T21:49:59.757518+00:00

Status `prepared_source_and_metadata_only`. ProspectiveCPU664vector14 recipe only: original664stage/146899and171427samples, currentSHOPanchor,600/1800prefixes, fixed0/.1/.25/.5grid/lowerround+weightties, locked2022; separate6GiB/3hfit proposal. No featurebuild/labeldecode/modeltraining/score. Metadata/source/stage closure passed;5tests. Revision1snapshot preserved before lineage-strengthening review. Command: `.venv-ml/bin/python scripts/prepare_cpu_vector14_fold.py; .venv-ml/bin/python -m unittest tests.test_cpu_vector14_fold tests.test_cpu_vector14_portability`. Manifest `artifacts/cpu-vector14-fold-preparation-001/manifest.json`, SHA `86caac0111eefac002c202bb337258bf64f54ede308debcab86636a3ffdf7e8d`. Runtime 0.926344457999221s. Source/protocol/data/sample/input checksums and exact preserved output evidence inmanifest/snapshots. Environment unchangedmacOS27.0.1arm64/Python3.11.15/NumPy2.4.6/pandas3.0.6; no optimization/modeltraining here. No Kaggle submission, score or rank from this run.

## cpu-vector14-fold-preparation-002 — 2026-11-02T21:51:38.400193+00:00

Status `prepared_source_and_metadata_only`. ProspectiveCPU664vector14 recipe only: original664stage/146899and171427samples, currentSHOPanchor,600/1800prefixes, fixed0/.1/.25/.5grid/lowerround+weightties, locked2022; separate6GiB/3hfit proposal. No featurebuild/labeldecode/modeltraining/score. Metadata/source/stage closure passed;6tests. Revision2 addslock-audit protocol/identityscore and returnedfit/bridge lock+round bindings; source/protocol snapshots and priorpreparationSHA retained. Command: `.venv-ml/bin/python scripts/prepare_cpu_vector14_fold.py; .venv-ml/bin/python -m unittest tests.test_cpu_vector14_fold tests.test_cpu_vector14_portability`. Manifest `artifacts/cpu-vector14-fold-preparation-002/manifest.json`, SHA `8b9ce9a065851233b20c3cc301768871885fd407c8a8cd761ecfa555afa6883e`. Runtime 0.9270240420009941s. Source/protocol/data/sample/input checksums and exact preserved output evidence inmanifest/snapshots. Environment unchangedmacOS27.0.1arm64/Python3.11.15/NumPy2.4.6/pandas3.0.6; no optimization/modeltraining here. No Kaggle submission, score or rank from this run.

## cpu-vector14-actual-auditor-preflight-001 — 2026-11-02T21:58:19.347476+00:00

Status `prepared_synthetic_read_only_audit_only`. ActualCPU bridge/fit auditor preparation only;3synthetic tests and independent nativeJSON16/64tree replay of all10000×14 saved synthetic outputs, maxerror0. No competitionfeatures/targets or modelfit. Neverloaded3.4.1models inoldXGBoost. Timingpilot originalfailed2GiBstatus unchanged. Initial preparation; actualdata audit still required. Command: `.venv-ml/bin/python -m unittest tests.test_cpu_vector14_fold_audit; read-only Python calls forest_prediction(saved_json,synthetic_x,rounds) and native_config over saved timing001 evidence`. Manifest `artifacts/cpu-vector14-actual-auditor-preflight-001/manifest.json`, SHA `d501c9a9e5ed2a64bff21423011dfde9414932ecc0065749560a57bf5378d5e7`. Runtime 0.20531579200178385s. Source/protocol/data/sample/input checksums and exact preserved output evidence inmanifest/snapshots. Environment unchangedmacOS27.0.1arm64/Python3.11.15/NumPy2.4.6/pandas3.0.6; no optimization/modeltraining here. No Kaggle submission, score or rank from this run.

## cpu-vector14-actual-auditor-preflight-002 — 2026-11-02T21:59:16.595411+00:00

Status `prepared_synthetic_read_only_audit_only`. ActualCPU bridge/fit auditor preparation only;3synthetic tests and independent nativeJSON16/64tree replay of all10000×14 saved synthetic outputs, maxerror0. No competitionfeatures/targets or modelfit. Neverloaded3.4.1models inoldXGBoost. Timingpilot originalfailed2GiBstatus unchanged. Additional nativehistogram/default recipe and distinctcoordinator-PID checks before source review. Command: `.venv-ml/bin/python -m unittest tests.test_cpu_vector14_fold_audit; read-only Python calls forest_prediction(saved_json,synthetic_x,rounds) and native_config over saved timing001 evidence`. Manifest `artifacts/cpu-vector14-actual-auditor-preflight-002/manifest.json`, SHA `9cba785b0d3da69e229361c6503189d0458cb0c001a9883e1428c7a3a1c67f1f`. Runtime 0.23862158399788314s. Source/protocol/data/sample/input checksums and exact preserved output evidence inmanifest/snapshots. Environment unchangedmacOS27.0.1arm64/Python3.11.15/NumPy2.4.6/pandas3.0.6; no optimization/modeltraining here. No Kaggle submission, score or rank from this run.

## cpu-vector14-actual-auditor-preflight-003 — 2026-11-02T22:02:32.064126+00:00

Status `prepared_synthetic_read_only_audit_only`. ActualCPU bridge/fit auditor preparation only;3synthetic tests and independent nativeJSON16/64tree replay of all10000×14 saved synthetic outputs, maxerror0. No competitionfeatures/targets or modelfit. Neverloaded3.4.1models inoldXGBoost. Timingpilot originalfailed2GiBstatus unchanged. Independent review requested explicit bridgeenvironment/buildRSS, exact sample/validationlabel dtype+shape, fit bridge status/year/protocol/lock. All added before sourceapproval. Producer sources/protocol unchanged. Command: `.venv-ml/bin/python -m unittest tests.test_cpu_vector14_fold_audit; read-only Python calls forest_prediction(saved_json,synthetic_x,rounds) and native_config over saved timing001 evidence`. Manifest `artifacts/cpu-vector14-actual-auditor-preflight-003/manifest.json`, SHA `0574a6a530bc1e022839319aded3a9355533ed8f31feb482b3b4dc7a7843aa12`. Runtime 0.25866312500147615s. Source/protocol/data/sample/input checksums and exact preserved output evidence inmanifest/snapshots. Environment unchangedmacOS27.0.1arm64/Python3.11.15/NumPy2.4.6/pandas3.0.6; no optimization/modeltraining here. No Kaggle submission, score or rank from this run.

## pre2020-canonical-input-inventory-001 — 2026-11-02T22:03:42.745013+00:00

Status `completed_read_only_inventory`. Read-only exactpre2020 inventory:664stagedloader cutoff2019 masks2020+targets;122304int64sample is thirdsequentialseed4173draw after2018and2019,1820purgedtrainingcases through2019Dec25;360screen casesJan1–Dec25. Reconstructed actualNPY bytes frommetadata only. Savedoldsingle608LGBanchor scope/proofs pinned; no newscore/featurebuild/fit. Command: `.venv-ml/bin/python metadata-only NumPy/pandas sample reconstruction heredoc; actualpaths/checksums/RNGformula inmanifest`. Manifest `artifacts/pre2020-canonical-input-inventory-001/manifest.json`, SHA `eab86ac08864a2f470aa624ca53e71f6bb91fbd23a6d630ddef1e5e6894e1641`. Runtime not separately timeds. Source/protocol/data/sample/input checksums and exact preserved output evidence inmanifest/snapshots. Environment unchangedmacOS27.0.1arm64/Python3.11.15/NumPy2.4.6/pandas3.0.6; no optimization/modeltraining here. No Kaggle submission, score or rank from this run.


## lgb-river726-2021-001 — 2026-11-02T21:47:11.926715+00:00

Status complete; frozen664+62 river feature test. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train_lgb_river726.py --protocol docs/lgb_river726_protocol.json --protocol-sha256 c71d6a026f7ca3e802464df4be7b8874ab8592f54d38d85cd97e2b9040762b7c --independent-audit artifacts/lgb-river726-protocol-independent-audit-001/manifest.json --independent-audit-sha256 f0df370c38ac821e3399f4b693f58a3c744b0f5ca0ae212a9d3461d4ce749892 --year 2021 --run-id lgb-river726-2021-001`. Year 2021, selected weight 0.25, AUC 0.9951306250691139, anchor 0.9951018232145258, matched LGB664 0.9950904829409322. 350trees/31leaves/lr.06/minleaf100/L2=1/featurefraction.9/seed4173/2threads/exact40% sample; no early stopping. Runtime 1190.267s; peakRSS 7542095872bytes. Exact source/data/cache/split/config/replay evidence: `artifacts/lgb-river726-2021-001/manifest.json`. No public score.

## submission-cuda-xgb608-fullsample-anchor-blend-001 — 2026-11-02T22:07:59.864675+00:00

Status:complete. Command:`/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/export_cuda_xgb608_fullsample_final.py --protocol docs/cuda_xgb608_fullsample_final_protocol.json --protocol-sha256 4bf91a4ceb200a884802f40af82e3f1f186f50d647c4d44f7bd4d6f601b6714e --archive artifacts/cuda-xgb608-fullsample-final-001-download.zip --archive-sha256 3e4c366eec70f560a7c41daae2f1278682c9dae083050d34ce81b8b3c526c6ea --manifest-sha256 125e2f220ae27e8782904a279f86e15825e638a41bbafa3ce91493353ea00ac8 --gpu-run-name cuda-xgb608-fullsample-final-001 --smoke-run-name cuda-xgb608-fullsample-smoke-003 --smoke-manifest-sha256 318d9f528f1bd24477d7ce6d6ff047b303cb8debd0bb1f05fd334e5f0e672754 --run-id submission-cuda-xgb608-fullsample-anchor-blend-001`. Strict finalGPU artifact/sample/config/checkpoint/fulltestreplay verification and exact nested75%acceptedGPU608anchor/25%fullsampleGPU blend export. No training or weightretuning. SubmissionSHA:621c8df2470c9962e2e45441e138263cac67fe7fe9a384b1bf6e96803b96c20d. Full provenance:`artifacts/submission-cuda-xgb608-fullsample-anchor-blend-001/manifest.json`. NoKaggle result at export.


## lgb-river726-2022-preflight-001 — 2026-11-02T22:08:47.900172+00:00

Status passed_preflight_only; frozen664+62 river feature test. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train_lgb_river726.py --protocol docs/lgb_river726_protocol.json --protocol-sha256 c71d6a026f7ca3e802464df4be7b8874ab8592f54d38d85cd97e2b9040762b7c --independent-audit artifacts/lgb-river726-protocol-independent-audit-001/manifest.json --independent-audit-sha256 f0df370c38ac821e3399f4b693f58a3c744b0f5ca0ae212a9d3461d4ce749892 --selection-lock artifacts/lgb-river726-2021-001/selection_2021.json --selection-lock-sha256 99b3ea5906626118cf66f89ffa23e2d53f703a28fbeeef390a758d17ef9805b9 --lock-audit artifacts/lgb-river726-2021-independent-audit-001/manifest.json --lock-audit-sha256 571d4f5f9e8d96fabe075fbdbdff6ef21c9ca5e9e5d133afdd6581ad412ae3c0 --year 2022 --run-id lgb-river726-2022-preflight-001 --preflight`. Year 2022, selected weight None, AUC None, anchor None, matched LGB664 None. 350trees/31leaves/lr.06/minleaf100/L2=1/featurefraction.9/seed4173/2threads/exact40% sample; no early stopping. Runtime 0.365s; peakRSS 209453056bytes. Exact source/data/cache/split/config/replay evidence: `artifacts/lgb-river726-2022-preflight-001/manifest.json`. No public score.

## submission-cuda-xgb608-fullsample-standalone-001 — 2026-11-02T22:08:12.418677+00:00

Status complete. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/export_cuda_xgb608_fullsample_standalone.py --protocol docs/cuda_xgb608_fullsample_standalone_protocol.json --protocol-sha256 ffae7413c80ae91f5bec45de0c28100fba712c982ea1e1c59b1900c429e51b37 --verified-blend-run artifacts/submission-cuda-xgb608-fullsample-anchor-blend-001 --verified-blend-manifest-sha256 7e48523b011fe0806e1f88fdd766bb930bde4f811371690abf1eb8b23c388974 --run-id submission-cuda-xgb608-fullsample-standalone-001`. Predeclared rawFullsample608GPU plus fixedVinjeprojection; no weight search/training. Validation {'2021': 0.9949084950375574, '2022': 0.9935439554765468}. CSV SHA bbe1c25b28f5c1be129ae7303f85669e8803e873df3e977e9b23a6d690ec7284. Full provenance in manifest; runtime 2.296s. No Kaggle score at export.


## lgb-river726-2021-independent-audit-001 — 2026-11-02T22:08:03.479218+00:00

Actual2021 independent audit PASSED 28.461s/peakRSS5267079168bytes; process exited0. Full fresh664+62 reconstruction preserves every664 prefix byte, all14 native350-tree checkpoints replay original365-case validation exactly, and original targets/purged146899-row sample plus independent isotonic/rankAUC fullgrid/control/lock agree. Selected .25 AUC .9951306250691139 versus frozenSHOP .9951018232145258 and matchedrawLGB664control .9950904829409322 (independent arithmetic differs at most floatingpoint rounding). Actualfit manifest `ab99480d326a9da3bf3d01aa7c89a7bb43c474c661ec46e9ad1111a821e729b1`; .25lock `99b3ea5906626118cf66f89ffa23e2d53f703a28fbeeef390a758d17ef9805b9`; independentproof `571d4f5f9e8d96fabe075fbdbdff6ef21c9ca5e9e5d133afdd6581ad412ae3c0`. Exact command/source/environment/arrays in `artifacts/lgb-river726-2021-independent-audit-001/manifest.json`. PID52511 exit observation `8e5304409a476e76aa47133715e1e8bf5eaf9bd0ea4ebba4002b76248e637626`. No newfit or publicscore in this audit.

## lgb-river726-2022-launch-observation-002 — 2026-11-02T22:10:22.599967+00:00

Root-authorized fixed2022 continuation launched session64428 after actualeligible2021 audit and metadata2022preflight `cd79c02bad3d146997583c722337056bc5b3c9256aa6fe5345861ddefbf2759d` passed0.365s. Same frozen726protocol/350tree recipe/twoCPUthreads/exact171427-row sample and frozenSHOPanchor; only0 andlocked.25 are scored. No earlystopping or additionalweightsearch. Exact command and authorization provenance in `artifacts/lgb-river726-2022-launch-observation-002/manifest.json` SHA `0b628d0505a315e91ae9ae70ed7fb5b7a432d2d9e33c63d29abde80c671f9b51`; live log `artifacts/lgb-river726-2022-001.log`. Result pending; final fit remains conditional on actual2022 acceptance and exact final-protocol review. Observation001 preserved a command-string parsing failure before any record; training was unaffected and not restarted.


## canonical-pooled-screen-independent-source-review-001 — 2026-11-02T22:11:50.005784+00:00

Source review blocked execution of unrun frozen v1 protocol `c376b822d32bde6ba87340240c68d1c12abb82aa18e541599284d602e781c283` solely for missing final observed resource checks after pooled replays and scoring. Requested pre-label/seal and post-score/pre-complete guards; mathematical recipe, original2019cutoff, exactsample,115mapping,14controls and six fixed gates otherwise coherent. No feature build, fit or target decode. Proof `artifacts/canonical-pooled-screen-independent-source-review-001/manifest.json` SHA `6e8ac7ce1c110160d73905e53ede74c45b6bd4e8193aebe23c92ab06c5b6c608`. V1 remains preserved; v2 review pending.


<!-- root-proof:cpu-vector14-fold-source-independent-audit-001 -->
## cpu-vector14-fold-source-independent-audit-001 — 2026-11-02T21:54:20.391644+00:00

Status passed. Prospective CPU vector14 source and exactprotocol review. No actual bridge or fit. Exact source/config/environment/commands and checks: `artifacts/cpu-vector14-fold-source-independent-audit-001/manifest.json`, SHA256 `c07f97cfbca82e0bf1b224a9db069d28f81882cc4cae02b43a0e2103b07304cd`. Runtime 0.9394596250022005s. No publicscore from thisrun.


<!-- root-proof:lgb-river726-final-template-independent-audit-001 -->
## lgb-river726-final-template-independent-audit-001 — 2026-11-02T21:58:30.441265+00:00

Status passed_template_only. Conditional726 final trainer/export/independentreplay templates reviewed; no finalprotocol or model. Exact source/config/environment/commands and checks: `artifacts/lgb-river726-final-template-independent-audit-001/manifest.json`, SHA256 `923ea2d11aa5406436003c332a88fd4936e2b1259fafbb8fd75e29f0bb267269`. Runtime 0.915104291998432s. No publicscore from thisrun.


<!-- root-proof:canonical-pooled-timing-001 -->
## canonical-pooled-timing-001 — 2026-11-02T22:00:25.230142+00:00

Status complete. Synthetic-only32768x11564round timings: pooled0.941571s,oneunit0.742855s; bothcheckpointreplays exact. Rough full2020 linear estimates18min pooled1400 plus3.5min controls, not measured full-data runtime. Exact source/config/environment/commands and checks: `artifacts/canonical-pooled-timing-001/manifest.json`, SHA256 `dff41a8baf3d58c0c29088ee09f6ed5072271aa763959d2d71e763378867afde`. Runtime 1.9963525829989521s. No publicscore from thisrun.


## canonical-pooled-screen-v2-independent-preflight-001 — 2026-11-02T22:13:54.864379+00:00

Independent source/metadata preflight PASSED 2.487s. Frozen protocol `dc1ed39784e6382b815ec37445ba17220561ddae388c6caa041ba38b5fe51684`, exact preservedv1 mathematical contract, allsource/input/preflight hashes, independent167-hour chronology and third sequentialseed4173 sample (1820cases/122304hours/1712256pooledrows),360screen cases and half-year scopes verified. Target CSV accessed only with RunNo/starttime usecols. Adapter source bound to root's all115fields/all14units scalar oracle; identity anchor bound to actual42checkpoint OOF audit and exact saved projection. Nine tests pass, including time/RSS failures and pre-seal/pre-label/post-score guard ordering. No competitionfeature construction, targetdecode or model fit. Fixed14×350 canonical controls, one1400trajectory700/1400checkpoints, unweightedBCE, fixed0/.1/.25/.5grid, sixstrictfull/half gates and no rescue. Proof `artifacts/canonical-pooled-screen-v2-independent-preflight-001/manifest.json` SHA `33be045efbb45f88de6d29b249d069092a923bf19831236a717ded06b29ec9af`; exact command/source/environment saved. Observed35min/8GiB limits are not hard OS caps. Actualscreen waits root scheduling behind LGB726; no laterfit/submission authorized.

## gpu-shop-fullsample745-launch-recheck-001 — 2026-11-02T22:14:58.773781+00:00

Passed exact local protocol/notebook/source/package ZIP and importer verifier/selector closure recheck. Both notebooks remain single2021 runs with fresh smoke and durable return archives; no feature build, model fit, score or remote action. The completed v10 fullsample608 final remains closed. Two saved GPU VMs maximum. Complete paths/hashes and intended return namespaces are in this manifest.


## Fullsample608 actual primary and standalone Kaggle evaluations — 2026-11-02T22:16:37.435356+00:00

Both predeclared candidates submitted once and observed Complete. Primary `621c8df2470c9962e2e45441e138263cac67fe7fe9a384b1bf6e96803b96c20d` scored **0.99643**; standalone `bbe1c25b28f5c1be129ae7303f85669e8803e873df3e977e9b23a6d690ec7284` scored **0.99631**. Separate authenticatedUI observations are in both exportdirectories as `kaggle_observation.json`. Independent actualpairCSV audit `f199cf2df3f035a3f6f6240820c10df23a99257ef5e70e4d2a8e408928c284e4`. Standalone handoff named the genericfilename incorrectly; root downloaded the actualuploadedfile and confirmed exactexpectedSHA, preserving thisverification at `artifacts/kaggle-uploaded-fullsample608-standalone-001.csv`. Neither exceeds newbestLGB664-SHOP0.99652/rank18. They improve their older608comparators0.99640/.99627. No weights or finalselection changed; submissionIDs not exposed. Bestrecipeunchanged.


## canonical-pooled-result-auditor-preparation-001 — 2026-11-02T22:19:21.725219+00:00

Prepared independent actual canonical-screen auditor; no actualcompetitionfeatures/targets or model fit. Six focused synthetic tests pass0.223s: all115 independent semantic features/signedzero, saved64-tree native synthetic replay plus categorical/replay corruption, independentno-rescue/tie decisions, candidate-bound post-completion exit evidence, directtargetrow/hour/unit ordering and safepath/score checks. Actual checker requires fullcompleteproducer/protocol/source/inputclosure, independently rebuilt training/screen115bytes from original664, original122304-hour sample and labels,14matchedcontrols+700/1400nativecheckpointreplays and exact700prefix, before-label seal, post-score resources, independentisotonic/rankAUC allgrid/full/half sixgates and selectedarray. Source `scripts/audit_canonical_pooled_result.py` SHA `1e9437b522855d44735bf4590c72d5bf04182064b10c31de5cce76ee922a5295`; snapshot `artifacts/canonical-pooled-result-auditor-preparation-001/manifest.json` SHA `73320f80bbd7999e67609b38fb640f4dc7164940be4bc37579654e089928acaf`. Shop independent source review pending. Actual audit waits completedscreen/process exit and large-slot scheduling; no learned result inferred.


## canonical-pooled-result-auditor-preparation-002 — 2026-11-02T22:24:23.839991+00:00

Preserved unrun v1 and reviewer counterexample8fec6a0f… showing genericisotonic floatingpoint differences can break exact ties and changeAUC. Separatev2 independently reconstructs frozen3valuePAV arithmetic for rank inputs and keeps generic min-max as probability-only semantic proof; exactsavedanchor/selectedbytes required, scoretolerance unchanged. Adds nativeordinaryunweightedBCE/defaultbase-score/bin-sample guards. Seven focused tests pass0.183s including random/exacttie/neartie/endpoints and exposedAUCdifference. No competitionfeature/target/modelrun. SourceSHA `43eccdc3e90ab6908c9dd51d5d648f0213fd8c21f727862a92b60b388e20cd84`; snapshot `artifacts/canonical-pooled-result-auditor-preparation-002/manifest.json` SHA `5459896dbb6c1a02eac25e575518d131b16bf716ab038eb043a45a22e1f35380`. Producer/model/protocol unchanged; independent v2 source review pending.

## lgb-river726-result-auditor-v2-preparation-001 — 2026-11-02T22:29:18.554275+00:00

Prepared a separately versioned actual-result auditor after a synthetic counterexample showed generic isotonic arithmetic can move exact ties by1e-16 and change rank AUC. Uses the independently reviewed exact frozen-operation three-value PAV helper, with a separate generic semantic check. Original auditor, producer, protocol and scores remain unchanged; seven synthetic tests passed in0.011s. Strict score tolerance stays1e-13; selected output now requires byte identity. No competition features/labels/results or model fitting. Exact sources, parent reviewer evidence, diff and hashes are in this manifest. Root source approval is required before the actual2022 audit.


## lgb-river726-2022-001 — 2026-11-02T22:09:02.638441+00:00

Status complete; frozen664+62 river feature test. Command `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train_lgb_river726.py --protocol docs/lgb_river726_protocol.json --protocol-sha256 c71d6a026f7ca3e802464df4be7b8874ab8592f54d38d85cd97e2b9040762b7c --independent-audit artifacts/lgb-river726-protocol-independent-audit-001/manifest.json --independent-audit-sha256 f0df370c38ac821e3399f4b693f58a3c744b0f5ca0ae212a9d3461d4ce749892 --selection-lock artifacts/lgb-river726-2021-001/selection_2021.json --selection-lock-sha256 99b3ea5906626118cf66f89ffa23e2d53f703a28fbeeef390a758d17ef9805b9 --lock-audit artifacts/lgb-river726-2021-independent-audit-001/manifest.json --lock-audit-sha256 571d4f5f9e8d96fabe075fbdbdff6ef21c9ca5e9e5d133afdd6581ad412ae3c0 --year 2022 --run-id lgb-river726-2022-001`. Year 2022, selected weight 0.25, AUC 0.9938778756239264, anchor 0.9938546734526688, matched LGB664 0.9938790447855327. 350trees/31leaves/lr.06/minleaf100/L2=1/featurefraction.9/seed4173/2threads/exact40% sample; no early stopping. Runtime 1435.604s; peakRSS 5619335168bytes. Exact source/data/cache/split/config/replay evidence: `artifacts/lgb-river726-2022-001/manifest.json`. No public score.
