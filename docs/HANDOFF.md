# Project handoff

Updated **2 November 2026** for a teammate continuing the modelling work.
The confirmed public reference is **0.99172**. The latest work produced two
verified local candidates, with no Kaggle result recorded for either. Begin
with [REPRODUCING.md](REPRODUCING.md) to run a new experiment without depending
on this computer's cached models.

## Problem and available information

Each case has a start time and a seven-day horizon. Predict an ON probability
for each of 14 generators at each of 168 hours. There are 2,922 supplied labelled
cases and 725 competition cases. The submission contains `Run No` followed by
2,352 prediction columns in the exact order of `sample_submission.csv`.

The teacher is a scheduling optimizer. Inputs include the supplied hourly
price and inflow paths, reservoir volumes at the case start, terminal water
values at start +168 hours, equipment topology and minimum-volume/flow
constraints. The whole supplied price week is available to the prediction
model; this is not a separate task of forecasting next week's prices. Later
observed reservoir levels and true future commitment labels are not valid
replacement inputs.

Daily case starts mean neighbouring target weeks overlap. Predictions for the
same calendar hour in different cases may have different labels because they
belong to different optimization problems with different starting conditions
and horizon boundaries. Do not collapse them into one chronological label.

Micro ROC-AUC pools all generator/hour predictions. Within-generator AUC,
transition errors and expected ON hours are useful diagnostics, but the pooled
metric is the selection target. Thresholding at 0.5 is not required for a
submission, and changing per-generator probability scales can affect pooled AUC.

## Current model and entry points

The reference recipe is
[`round5_submission_lgb_generators.json`](../configs/round5_submission_lgb_generators.json):

```text
0.50 × Round3 ensemble
+ 0.25 × separate-generator LightGBM seed 299
+ 0.25 × separate-generator LightGBM seed 673
```

The Round3 branch combines XGBoost, pooled LightGBM, Transformers and a Haukeli
tree specialist with plant-specific weights. Its six sources and exact weights
are in [`round3_submission_temporal_ranks.json`](../configs/round3_submission_temporal_ranks.json).
The current best is not simply the older XGBoost/CNN blend.

| File | Role |
| --- | --- |
| [`features.py`](../src/features.py) | Load data and build hourly tabular inputs |
| [`price_features.py`](../src/price_features.py), [`temporal_features.py`](../src/temporal_features.py) | Neighbouring-price summaries and within-week ranks |
| [`topology_features.py`](../src/topology_features.py), [`tree_preprocessing.py`](../src/tree_preprocessing.py) | Reservoir connections and economic rescaling |
| [`round5_lightgbm.py`](../src/round5_lightgbm.py) | Current separate-generator tree training and frozen replay |
| [`round6_evaluate.py`](../src/round6_evaluate.py) | Matched ensemble comparisons from cached predictions |
| [`round10_temporal.py`](../src/round10_temporal.py) | Small correction using neighbouring predicted logits |
| [`scheduling_context.py`](../src/scheduling_context.py) | Optional water/constraint timing experiments |
| [`make_goal_submission.py`](../src/make_goal_submission.py) | Export a declared blend from saved component predictions |

The separate-generator reference has 113 features, including local and
system-wide reservoir information, topology and price context. Monetary inputs
are divided by `max(mean(abs(weekly_price)), 10)`; the logarithm of the scale is
retained. The rank candidate has 115 features.

Tree settings are 63 leaves, learning rate 0.04, minimum 100 rows per leaf,
L1 0.2, L2 4, `max_bin=63`, and 85% row/feature sampling. Per-generator tree
counts were fixed from 2019/2020 validation. Full-week features are built first;
training then uses `hour % 4 == (Run No + seed) % 4`, or 42 hours per case and
generator. All 168 hours are predicted. Seeds are uniformly averaged.

## Results worth retaining

| Work | Outcome | Evidence |
| --- | --- | --- |
| Separate-generator LightGBM and blend | Public score 0.99172; fixed reference | [Round5](../reports/round5_report_20261014.md) |
| Capacity, regularization and training-length search | No robust improvement; experimental L2-free model did not become the reference | [Round6](../reports/round6_tuning_report_20261017.md) |
| Input removal, case thinning, randomized thresholds, calibration and revised plant weights | Failed promotion criteria | [Round9](../reports/round9_report_20261017.md) |
| Two price ranks on current separate models | Improved all three checks with two seeds; exported candidate | [Round10](../reports/round10_report_20261018.md) |
| Temporal residual correction | Small improvement over identity and same-hour controls; exported separately | [Matched control](../reports/round10_temporal_matched_control_results.json) |
| Shared river constraint timing and distinct inflow timing | Failed confirmation; leave disabled | [Water confirmation](../reports/round10_water_pair_summary.json) |

Latest candidate names are `submission_round10_ranks.csv` and
`submission_round10_temporal.csv`; their generated CSVs remain outside Git.
The rank candidate changes only the separate-generator branch. The temporal
candidate instead corrects the original Round5 branch, using a frozen 2021
linear model with 0.10 shrinkage. It does not include the new rank models.
Public scores are recorded in [kaggle_scores.json](../reports/kaggle_scores.json),
separately from historical validation scores.

## Validation and remaining uncertainty

Training weeks must finish by the validation start. At a January 1 boundary,
the last eligible daily start is December 25; December 26–31 are purged.
Native2022 trains before 2022. Frozen2022 reuses pre2021 models without fitting.
The additional January 7 onward evaluation removes overlap with late2021
selection horizons. Native and frozen checks share 2022 labels.

These years have been reused extensively. Some inherited native components
also selected checkpoints using their evaluation year. The comparisons and
bootstrap intervals do not create an untouched holdout. Predeclare the split,
model change and promotion rule before inspecting more results, and compare
against the actual ensemble with the same seeds and blend weights.

The [October audit](../reports/audit_20261017_report.md) found no current
submission-order or feature-alignment defect. Remaining errors include
switching times and unusual operating regimes, with varying generator-level
performance. A severe Vesle Kjela case remains around 0.53 expected ON hours
against 100 labelled ON hours after the latest inflow experiment. That narrow
feature change did not explain or repair the failure. It also does not prove
that the same rare condition explains the leaderboard gap.

The useful questions are now whether a proposed change fixes a measured error
slice, improves pooled AUC across historical checks, and contributes something
the existing ensemble lacks. Do not assume a larger model, more seeds or another
feature group will help without a matched comparison. Keep failed experiments
in the record so subsequent agents do not repeat them unknowingly.

## Working conventions and archived tools

Use new config/run names. Keep code simple, retain the best artifacts, and
record the exact recipe, training cohort, validation scores and source paths.
Use `reports/` for small results and explanations; use ignored directories for
large data, predictions and checkpoints. The source files and reports contain
many historical experiments; the date and recipe matter more than the largest
round number in a filename.

Some audit tools are tied to the original machine: `audit_project.py` refers to
the original Downloads ZIP, and several older `round3_` through `round6_`
verification scripts require temporary preservation manifests. Reports retain
those paths and hashes as historical provenance. They are not prerequisites
for a new training run; do not rewrite recorded hashes to make an old audit pass.
The supplied `evaluation/kaggle_metric.py` and raw materials must remain unchanged.

For another coding agent, use this starting brief:

> Read README.md, docs/HANDOFF.md, docs/REPRODUCING.md and the Round10 report.
> Establish which raw data, caches and checkpoints are present before running.
> Keep the 0.99172 reference and both unscored candidates intact. Propose and
> run a bounded, named experiment with fixed chronological comparisons; report
> both successes and failures. Fit only labelled cases, preserve the supplied
> metric, and do not use competition labels or tune repeatedly to public scores.
> Keep the implementation understandable for teammates and record enough
> provenance for another computer to reproduce the work.
