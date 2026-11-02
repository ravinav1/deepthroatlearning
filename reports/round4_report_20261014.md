# Transformer optimizer controls and LightGBM temporal ranks

**Completed 14 October 2026.** One candidate is fitted, exported and independently verified: [submission_round4_transformer_lr_mean.csv](../submissions/submission_round4_transformer_lr_mean.csv). It averages the existing task-weighted Transformer with a new, lower-learning-rate version inside that component's existing ensemble share. The public incumbent remains the user-reported **0.99068** round3 submission. This round has no Kaggle result.

## What changed and what we learned

The new Transformer uses learning rate **0.0005 instead of 0.001**, with the same architecture, 306 input channels, task-weighted loss, seed 42, batch size 32 and weight decay 0.0001. It trains for **39 epochs**, selected on 2021 and then fixed for 2022 and full-data fitting. The selected component is a 50/50 mean of old and new predictions. All other ensemble components and plant coefficients stay unchanged.

| Evaluation | Incumbent AUC | Selected mean AUC | Gain |
|---|---:|---:|---:|
| 2021, trained before 2021 | 0.988984063 | 0.989044618 | +0.000060555 |
| 2022, trained before 2022 | 0.988606658 | 0.988713461 | +0.000106803 |
| Same 2022 fit, Jan 7+ | 0.988512663 | 0.988625291 | +0.000112628 |
| 2022, model frozen before 2021, Jan 7+ | 0.986419019 | 0.986454618 | +0.000035599 |

The matched standard-learning-rate reference also trained for 39 epochs. Its mean with the incumbent improves scores, so part of the gain comes from combining checkpoints. The lower-learning-rate mean exceeds that reference by **0.000025365** on 2021, **0.000084408** on native 2022 Jan 7+, and **0.000018278** on frozen 2022 Jan 7+. This is directional evidence from one seed and one training budget, not proof that this is the optimal learning rate. Every selected checkpoint reached the 39-epoch limit; these runs do not establish convergence.

Direct replacement with the new model loses AUC in frozen 2022, while its mean with the incumbent gains. Its useful contribution therefore depends on the ensemble. Raw all-head AUC from this task-weighted model is not the ensemble AUC: several output heads have zero training-loss weight and contribute nothing to the delivered ensemble.

[Historical comparison](round4_neural_both_comparison.json), [matched references and final decision](round4_neural_final_decision.json), [selection recorded before candidate 2022 evaluation](round4_neural_selection_2021.json).

## Rejected changes

- **LightGBM temporal ranks:** adding only the existing whole-week and remaining-week price ranks improved the 2021 and frozen-2022 ensemble checks. It then reduced native-2022 ensemble AUC by **0.000053013**, despite slightly improving standalone LightGBM AUC from 0.986684 to 0.986768. The original LightGBM is retained. No full-data LightGBM fit was made.
- **Stronger Transformer weight decay (0.01):** its best predeclared 2021 variant improved AUC by 0.000043060, below the selected learning-rate arm. It was not advanced to 2022; this is not evidence that stronger weight decay generally fails.
- **Combined LightGBM and Transformer change:** passed the 2021 and frozen screens but was rejected when LightGBM failed its individual native-2022 gate. No native combined score, full fit or submission was made.

LightGBM comparisons used matched 1,499-tree models for the original 2021/frozen context, plus a matched 800-tree 2021 check before accessing candidate 2022 results. Native-2022 models used 800 trees, matching the incumbent and intended deployment count. Thus the negative native result is not a mismatched-tree-count comparison. Haukeli regressed, while error similarity to the fixed remainder increased slightly; this is consistent with reduced ensemble complementarity, not causal proof.

[LightGBM decision](round4_lgb_ranks_final_decision.json), [native diagnostics](round4_lgb_ranks_native_2022_diagnostics.json), [combined selection and gates](round4_combined_selection_2021.json).

## Remaining weaknesses and uncertainty

The selected neural mean improves aggregate AUC within two hours of true ON/OFF transitions in each scenario. Tokke improves consistently. **Lio remains mixed:** its within-generator AUC falls by about 0.000584 in 2021 and 0.000639 in frozen 2022, but rises by 0.000269 in native 2022 Jan 7+. Kjela also regresses in both native years. This change does not solve the difficult sustained-ON cases identified in the previous round.

Paired 14-day whole-case block bootstrap intervals (200 draws) for total AUC gain are:

| Evaluation | Approximate 95% interval |
|---|---:|
| Native 2021 | [−0.00001193, +0.00011831] |
| Native 2022, Jan 7+ | [−0.00000063, +0.00023198] |
| Frozen 2022, Jan 7+ | [−0.00010849, +0.00018555] |

All include zero. Both years are reused development data; the intervals do not correct for repeated selection. Frozen and native 2022 share labels and are two deployment scenarios, not independent replications. Validation training excludes the six case starts whose weekly horizons overlap validation. The Jan 7+ check additionally excludes 2022 cases overlapping the final 2021 selection horizons. No competition labels are used.

[Native neural diagnostics](round4_neural_lr_native_diagnostics.json), [frozen diagnostics](round4_neural_lr_frozen_diagnostics.json).

## Implementation and delivery

The only edits to existing source files wire the default-off `use_temporal_ranks` option into `train_baseline.py` and add an optional component-specific validation monitor in `train_cnn.py`. The monitor keeps every other ensemble contribution fixed and preserves the original monitoring mode. Its cached predictions affect checkpoint selection only, never training inputs or targets. Identity and legacy checks passed. The new optimizer controls are configurations; model architecture and loss code are unchanged.

Heavy fits run sequentially on four CPU threads. Diagnostics and inference use one thread. Torch and XGBoost run in separate processes. The [round4 ledger](autonomous_round4_20261014.json) records runs, selections and artifact paths.

The final model trained on all **2,922 supplied labelled cases**, using the fixed 39 epochs, in 398 seconds. It produced **725 rows and 2,352 probabilities per row**, plus `Run No`, in template order. Competition IDs have zero overlap with training IDs. Scalers were fitted on training inputs; competition inputs were used for prediction only. No Kaggle upload was performed.

Independent verification reconstructed every one of the **1,705,200** final blend probabilities and mean-slot probabilities with zero difference. Replaying the saved checkpoint and scaler on 17 fixed competition cases differed from CSV values by at most **2.98e-8**. This is sampled model replay, not an independent rerun of every case. All **414 protected originals** remain unchanged, including raw data, prior models and submissions, the round3 record and `kaggle_metric.py`. The old prediction cache also matches its incumbent export hash. A verifier-only snapshot-membership assumption was corrected before the successful check; predictions were not changed.

[Final verification](round4_final_verification.json), [export provenance](submission_round4_transformer_lr_mean_run.json), [full-fit provenance](round4_transformer_lr_full_run.json), [component mean provenance](round4_transformer_lr_slot_full_run.json), [submission recipe](../configs/round4_submission_transformer_lr_mean.json).

Submission SHA-256: `0f3e3286f27ee6f4e15670b7c7b31a2b35995fdad9d26381663659c409828396`.

For a future bounded round, the unresolved question is whether longer controlled neural training adds more than another checkpoint average: all three 2021 arms reached the fixed epoch limit. Any such test should retain matched-duration references and the same chronological gates. Generator-specific Lio investigation remains separate; this experiment did not repair it.
