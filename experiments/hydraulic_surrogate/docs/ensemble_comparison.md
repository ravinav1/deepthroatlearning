# Ensemble comparison — 2026-11-02T10:27:00.251200+00:00

No models were fitted. Six configurations were declared in the source before running. All receive the same fixed Vinje projection; smoothing is disabled. Selection uses only 2021 and is locked before reading 2022. These are repeatedly reused development years, not untouched holdouts.

| Candidate | 2021 micro AUC | 2022 micro AUC |
|---|---:|---:|
| context | 0.98985442 | 0.98947692 |
| normalized | 0.98955299 | 0.98896450 |
| physics | 0.98475739 | 0.98609080 |
| context75_normalized25 | 0.98993303 | 0.98950560 |
| context50_normalized50 | 0.98990730 | 0.98943056 |
| context75_physics25 | 0.98996608 | 0.98970403 |

2021-selected candidate: **context75_physics25**, weights [0.75, 0.0, 0.25] in order ['lgb-normalized-context-001', 'lgb-normalized-001', 'lgb-physics-001'].

The explicit migration threshold is at least 0.998 on both folds. Results below that threshold remain experimental; no leaderboard score is inferred.

Complete provenance, raw/projected scores, source and artifact checksums: `artifacts/ensemble-001/analysis.json`. The selected fold predictions are saved alongside it. No Kaggle submission was made by this experiment.
