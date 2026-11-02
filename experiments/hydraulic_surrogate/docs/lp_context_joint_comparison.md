# Combined LP context and joint-plant feature comparison

The new supervised source `lgb-normalized-lp-context-joint-001` retains all521 existing normalized LP temporal-context features and appends87 dimensionless joint-plant cache features, for608 columns. Model settings remain fixed at350 trees, seed4173,40% sampled training hours and the same six-day-purged chronological folds. No previous feature module or cache was changed.

A metadata-only audit verifies that the first521 column names exactly match `lgb-normalized-lp-context-001`, the final87 names exactly match the joint cache, raw-data fingerprints agree, original context feature/cache specifications are identical, joint feature/cache specifications are identical, and both feature-module dependency closures have unchanged hashes. Trainer order builds and normalizes the original context table before appending the joint columns. This establishes unchanged formulas and source/cache inputs without loading a second large feature matrix; it does not claim a direct numeric full-matrix comparison.

Predeclare new-model global weights `[0, .25, .5, .75, 1]` against the current accepted18.75% LP-base/56.25% LP-context/25% joint-model mixture. Flatten the four source weights to

`[.1875*(1-w), .5625*(1-w), .25*(1-w), w]`.

Source order is `[lgb-normalized-lp-001, lgb-normalized-lp-context-001, lgb-normalized-lp-joint-001, lgb-normalized-lp-context-joint-001]`. Promote raw prediction arrays tofloat64, accumulate in that order into a zero float64 array, then apply the existing Vinje projection once. Weight0 must reproduce the accepted anchor array exactly. This exact arithmetic matches the generic exporter.

Select highest2021 whole-matrix micro AUC; exact ties choose the smaller new-model weight. Persist an exclusively created `selection_2021.json` before decoding any2022 prediction or label array. On2022, evaluate only the selected mixture and fixed diagnostic references, without an alternate weight grid. Accept a nonzero weight only if it strictly improves the fixed anchor on2022. No unit-specific selection, smoothing or posthoc weight changes occur. Both years are repeatedly reused development evidence; neither is a fresh holdout. Transfer still requires at least0.998 on both years.

The comparison checks exact source labels/IDs/unit order, finite bounded probabilities, equality of sampled training rows, unchanged source/data/cache/checkpoint-manifest provenance and exact anchor reconstruction. It records source snapshots, declaration and selection timing, hashes, both selected arrays, and per-unit/month/lead-day AUC. Accepted and rejected choices remain distinguishable. Analysis/lock output follows the generic four-source export contract; no final fit, export or Kaggle submission happens automatically.

Prepared before the combined2021 prediction array existed. Command: `.venv-ml/bin/python scripts/lp_context_joint_compare.py --run-id lp-context-joint-comparison-001`. It uses small arrays only and waits for the separately controlled trainer. Independent reconstruction and provenance verification follow completion.

## Completed result: lp-context-joint-comparison-001

The frozen new-model weight0.25 passes the declared point-estimate gate. Accepted flattened weights are **14.0625% base,42.1875% context,18.75% joint,25% combined**. Both scores remain below0.998.

| Recipe | 2021 micro AUC | 2022 micro AUC |
|---|---:|---:|
| matched_base_raw | 0.994370012291 | 0.993348542164 |
| combined_raw | 0.994678494898 | 0.993404061184 |
| combined_projected | 0.994680317304 | 0.993404731337 |
| current | 0.994888491751 | 0.993641964015 |
| selected | 0.994937718669 | 0.993691487954 |

The gains over the fixed prior are+0.000049226918/+0.000049523939. Standalone combined features improve the older LP-base point estimate, but the selected gain comes from a small complementary mixture. No alternative2022 weight was evaluated.

| New-model weight | 2021 selection AUC |
|---:|---:|
| 0.0 | 0.994888491751 |
| 0.25 | 0.994937718669 |
| 0.5 | 0.994927715075 |
| 0.75 | 0.994857834037 |
| 1.0 | 0.994680317304 |

The lock was written at`2026-11-02T13:15:32.507390+00:00` and precedes creation of the combined2022 NPZ. Independent verification passed49 registered hashes, exact mixtures/labels/IDs/units/AUC, lock/protocol checks and the declared acceptance rule. Analysis SHA256`32f692df612148094d4bfa06e45d4d0f321c35109444de297d4b0513cf2a43d5`; lockSHA256`8f7b79039a21f9aaab122484a02fc4a317808f667cd6a38c070782deafc06acf`. Proof:`artifacts/verify-lp-context-joint-comparison-001/manifest.json`.

| Unit | 2021 AUC delta | 2022 AUC delta |
|---|---:|---:|
| Hogga_G1 | +0.000153856 | -0.000077824 |
| Tokke_G1 | +0.000053815 | +0.000036804 |
| Tokke_G2 | +0.000036787 | +0.000036805 |
| Tokke_G3 | +0.000018227 | +0.000041542 |
| Tokke_G4 | -0.000052375 | +0.000045118 |
| Lio_G1 | +0.000178090 | +0.000119532 |
| Byrte_G1 | +0.000009303 | +0.000000438 |
| Vinje_G1 | +0.000052657 | +0.000095105 |
| Vinje_G2 | +0.000000094 | +0.000066290 |
| Vinje_G3 | +0.000176313 | +0.000139369 |
| Haukeli_G1 | +0.000671587 | +0.000687568 |
| Songa_G1 | -0.000015121 | -0.000126904 |
| Kjela_G1 | -0.000090663 | -0.000129900 |
| Vesle Kjela_G1 | +0.000019668 | -0.000033363 |

2021: month_auc: 10/12 improve; decreases {'7': -3.2049701e-05, '9': -0.000165409455}; lead_day_auc: 7/7 improve; decreases {}. Full scores are in the analysis manifest. No subgroup retuning.


2021 paired28-day circular block bootstrap,200 replicates,seed4173: observed delta0.000049226918; conditional95% percentile interval[0.000005692710,0.000100710929]; positive fraction0.990. Provenance:`artifacts/lp-context-joint-uncertainty-2021-001/manifest.json`.


2022: month_auc: 8/12 improve; decreases {'4': -0.000263476966, '6': -2.8979369e-05, '7': -0.000165194572, '9': -7.2722622e-05}; lead_day_auc: 7/7 improve; decreases {}. Full scores are in the analysis manifest. No subgroup retuning.


2022 paired28-day circular block bootstrap,200 replicates,seed4173: observed delta0.000049523939; conditional95% percentile interval[-0.000022383091,0.000115661765]; positive fraction0.935. Provenance:`artifacts/lp-context-joint-uncertainty-2022-001/manifest.json`.


The2022 interval crosses zero. The improvement is small and uncertain; these descriptive intervals exclude adaptive candidate selection, repeated development use, training variability and year shifts. They do not alter the already frozen acceptance rule. No full-data refit, export or Kaggle submission was performed by this comparison. Root owns those decisions.
