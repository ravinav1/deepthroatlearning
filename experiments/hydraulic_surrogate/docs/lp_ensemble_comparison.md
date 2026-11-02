# Ensemble comparison — 2026-11-02T10:54:29.459298+00:00

No models were fitted. Four configurations were declared in the source before running. All receive the same fixed Vinje projection; smoothing is disabled. Selection uses only 2021 and is locked before reading 2022. These are repeatedly reused development years, not untouched holdouts.

| Candidate | 2021 micro AUC | 2022 micro AUC |
|---|---:|---:|
| lp | 0.99437222 | 0.99335146 |
| prior_ensemble | 0.98996608 | 0.98970403 |
| lp90_prior10 | 0.99446734 | 0.99344637 |
| lp75_prior25 | 0.99427897 | 0.99332642 |

2021-selected candidate: **lp90_prior10**, weights [0.9, 0.075, 0.025] in order ['lgb-normalized-lp-001', 'lgb-normalized-context-001', 'lgb-physics-001'].

The explicit migration threshold is at least 0.998 on both folds. Results below that threshold remain experimental; no leaderboard score is inferred.

Complete provenance, raw/projected scores, source and artifact checksums: `artifacts/lp-ensemble-001/analysis.json`. The selected fold predictions are saved alongside it. No Kaggle submission was made by this experiment.
