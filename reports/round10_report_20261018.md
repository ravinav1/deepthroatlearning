# Round10: price ranks, temporal correction, and water/constraint timing

All three requested directions were tested. Price ranks produced the strongest supported candidate. A small temporal correction also improved the historical checks. Neither hydraulic timing variant qualified for promotion.

## Submissions

1. **Try `submissions/submission_round10_ranks.csv` first.** It keeps the existing 50/50 blend of Round3 and separate-generator LightGBM, adding only two price-rank features to the latter. Seeds 299 and 673 are averaged uniformly; tree counts, regularization, row sampling and blend weights remain fixed.
2. **`submissions/submission_round10_temporal.csv` is a smaller, separate experiment.** It applies the frozen 2021 temporal correction to the original Round5 LightGBM branch. It does not include the new rank models.

Both contain 725 cases and 2,352 probabilities per case, in the supplied template order. Both were independently reconstructed from their saved models. Neither has been uploaded, and Kaggle performance is unknown. The existing `submission_round5_lgb_generators.csv` is preserved.

## Historical results

Changes below are absolute **micro ROC-AUC deltas against the actual Round5 ensemble**, not against a weaker single-model baseline.

| Experiment | 2021 | Native 2022, Jan 7+ | Frozen 2022, Jan 7+ | Decision |
|---|---:|---:|---:|---|
| Price ranks, both seeds | +0.000105519 | +0.000155991 | +0.000159853 | Exported |
| Temporal correction | Not scored on its fitting year | +0.000022315 | +0.000028996 | Exported as a small experiment |
| Shared river constraint timing, seed299 replacement | -0.000053826 | +0.000007930 | -0.000038835 | Rejected |
| Distinct inflow timing, both seeds | -0.000046334 | +0.000024575 | +0.000036946 | Rejected |

Reference AUCs are 0.990374846, 0.989411166 and 0.987045304 respectively. Native models train before their evaluation year, with overlapping training weeks purged. Frozen2022 reuses the exact pre2021 models without fitting. January 1–6 are excluded from the primary 2022 comparison because their target weeks overlap the earlier development period. The 2021 comparison contains 365 cases; the primary 2022 comparisons contain 359.

These are **reused development years**, and the two 2022 scenarios share labels. They are useful robustness checks, not three independent holdouts. Bootstrap intervals are conditional on these predictions and do not remove model-selection bias.

### Price ranks

The two features describe the current price's midpoint percentile within the complete week and within the remaining week. They are computed from all 168 supplied prices before training-hour sampling. The existing 113-column representation becomes 115 columns.

Seed299 passed all three initial checks, then seed673 was run with the same settings. The paired 14-day block-bootstrap 95% intervals for the final two-seed gains were:

- 2021: [+0.000008827, +0.000205953].
- Native2022 Jan7+: [+0.000084419, +0.000244854].
- Frozen2022 Jan7+: [+0.000072315, +0.000292832].

The gain is uneven: 2021 Q1/Q2 and frozen2022 Q1 declined, along with some generators. Quarter diagnostics cover complete calendar quarters, whereas the primary 2022 score excludes January 1–6. No plant weights or settings were changed to hide these regressions.

Evidence: [paired gate and diagnostics](round10_ranks_pair_gate.json), [independent historical reconstruction](round10_review_ranks_results.json), and [full submission verification](round10_ranks_full_verification.json).

### Temporal correction and controls

A ridge-logistic residual model learns from historical out-of-time 2021 predictions. Its inputs are the mean LightGBM logit, generator indicators, neighbouring logit differences at ±1, ±2, ±6 and ±24 hours, and case-boundary indicators. Neighbours remain within each case and generator. The learned correction is applied with a fixed 0.10 shrinkage.

It beat identity, an ordinary same-hour logistic residual, and a same-hour control with matching boundary indicators. The boundary-matched control was documented before inspecting the 2022 scores. Its purpose is to distinguish neighbour information from simply recognizing week boundaries. The temporal gain over that strongest control was +0.000015378/native and +0.000019950/frozen; all tested block intervals were positive.

Improvement was concentrated near ON/OFF transitions. This is evidence for this particular linear correction. It does not exhaust richer multi-model stacking: the controls do not receive Round3 and the two individual seed predictions as separate inputs. The exported correction keeps its 2021 coefficients and scaler unchanged; it was not refitted on 2022.

Evidence: [original comparison](round10_temporal_results.json), [matched control](round10_temporal_matched_control_results.json), and [independent export verification](round10_review_temporal_export.json).

### Water and constraint timing

The two arms were tested separately, with default-off switches:

- `use_shared_river_constraint_timing`: remaining required release volume for r_Vest, time to its next known within-week constraint change, and the signed change size. Three added columns.
- `use_distinct_inflow_timing`: hourly, cumulative and remaining natural inflow for Staavatn and Venemo. Six added columns. Cumulative and remaining volumes include the current hour.

These are summaries of supplied inputs, not inferred optimizer flows or hard feasibility rules. All complete-week features precede row sampling.

Constraint timing failed the first-seed gate. Water timing initially passed narrowly, including only +0.000001469 on native2022 Jan7+, but the two-seed result lost on 2021. All three water confirmation bootstrap intervals include zero. Neither was included in a submission.

The water features also failed to repair the inspected late2022 Vesle Kjela collapse: for case2872, expected ON hours moved from approximately 0.523 to 0.530 against 100 labelled ON hours. Small aggregate gains therefore did not establish that the diagnosed rare-regime problem was solved.

Evidence: [feature checks and first-seed runs](round10_water_constraint_summary.json), [water confirmation summary](round10_water_pair_summary.json), [water paired2021](round10_water_pair_native_2021_comparison.json), [water paired native2022](round10_water_pair_native_2022_comparison.json), and [water paired frozen2022](round10_water_pair_frozen_2022_comparison.json).

## Implementation and preservation

The only existing source files changed were `src/round5_lightgbm.py` and `src/round6_evaluate.py`: optional timing features, Round10 output names, single-seed quarter diagnostics, and allowing saved-model replay after an old fitting deadline has expired. The existing feature flags remain off unless a new experiment config enables them. New temporal code and `src/scheduling_context.py` contain the additional methods.

Full rank fits used all 2,922 supplied labelled cases, with zero competition-case ID overlap. Independent replay reproduced all 28 saved models on all 725 competition cases, their seed mean, and the final blend exactly. Temporal export also replayed exactly. Training used CPUs, with two threads per tree fit and at most three such fits concurrently.

Raw data, the supplied `evaluation/kaggle_metric.py`, and the old submission remain unchanged. Of 356 protected original files, only the two intended source edits changed. See [preservation check](round10_preservation_check.json) and the [initial protocol](round10_protocol_20261018.json).
