# Round 9: bounded experiment search

None of the seven tested ideas passed its promotion criteria. Keep `submissions/submission_round5_lgb_generators.csv`, whose user-reported Kaggle score is **0.99172**. No new submission was generated. The search started at 18:43:16 UTC on 17 October 2026 and finished within the one-hour limit; exact timing is recorded in [the run ledger](autonomous_round9_20261017.json).

The reference combines 50% Round 3 with 25% each of the original Round 5 LightGBM seeds 299 and 673. Its historical 2021 micro-AUC is **0.9903748458953463**. The experimental Round 6 models were not used as the reference.

## Training changes and input reductions

Four variants were fitted with seed 299, one classifier per generator: **56 new LightGBM models**. Tree counts, capacity and other settings stayed fixed, apart from each stated change. Counts came from the existing 2019/2020 selection. Training cases ended before the 2021 validation horizon; six overlapping case starts were excluded.

Each variant had two fixed ensemble comparisons: replace the old seed-299 component, or average old and new components equally. Both retained 50% Round 3. Their matched single-seed reference AUC is **0.9902748710982663**, which differs from the two-seed reference above.

| Change | Replacement ΔAUC, 2021 | Old/new mean ΔAUC, 2021 |
|---|---:|---:|
| Remove only absolute weekly price-scale feature | −0.000205952 | −0.000044973 |
| Remove 51 detailed system-wide reservoir features | −0.004266279 | −0.001322195 |
| Train on every fourth complete case, using all its hours | −0.001161218 | −0.000291150 |
| Randomize tree split thresholds with `extra_trees=True` | −0.001014059 | −0.000204928 |

The case-sampling variant used 546 complete cases and 91,728 rows per generator. The reference used selected hours from all 2,186 eligible cases, totaling 91,812 rows per generator. Four-day case spacing still leaves overlapping weeks; this experiment changes coverage and correlation, and does not give LightGBM a joint sequence objective.

The eight fixed comparisons all lost to their matched reference. They therefore stopped before second-seed confirmation, candidate 2022 evaluation or full-data fitting. See [the selection record](round9_tree_selection_2021.json). The sampling and ensemble extensions were documented as adaptive additions after initial 2021 results, before any tree-candidate 2022 evaluation.

## Other experiments

| Idea | Evidence | Decision |
|---|---|---|
| Plant-specific blend weights | +0.000122687 on 2021; −0.000007267 on native 2022; −0.000207364 on frozen 2022 | Reject: gain did not transfer |
| Enforce Vinje generator probability ordering | Best 2021 gain +0.000000500 | Reject: negligible gain |
| Shared logit slope plus generator-specific offsets | Fit on purged first-half 2021; second-half AUC fell from 0.985693455 to 0.985210171 | Reject before 2022 |

Native 2022 uses models fitted on the preceding eligible years. Frozen 2022 keeps models fitted before 2021, testing a longer extrapolation. These evaluations share 2022 labels. Plant blending also failed when the first six 2022 case starts were excluded to remove overlap with the 2021 selection windows.

## What this establishes

Detailed reservoir context is useful in the current model: removing it caused the largest deterioration. Removing absolute scale also failed, although the old/new mean was close to the baseline. These experiments support retaining the current inputs.

Vinje ordering looked inconsistent in 14,486 case-hours, but 14,443 of those had identical labels for all three generators. Sorting probabilities among observations with the same label preserves the pooled AUC score distribution. This explains why correcting many apparent ordering violations barely changed the metric.

The failed transfer of plant weights shows that a 2021 improvement alone is insufficient. These results apply to the tested configurations; fixed boosting counts may limit the randomized-threshold variant. Historical years remain reused development data, and this search does not establish a performance ceiling or predict a private leaderboard score.

## Verification and artifacts

Independent reviews reproduced the comparison scores with the supplied Kaggle metric, checked case/column alignment, source hashes, averaging, purging, and the sampling/randomized model checkpoints. Competition test IDs had zero overlap with fitted cases. The original submission and metric remained unchanged. All Round 9 training processes finished; no Kaggle upload or Git push was performed.

- [Plant blend results](round9_plant_blend_results.json)
- [Unit ordering results](round9_unit_order_results.json) and [explanation](round9_unit_order_explanation.json)
- [Calibration results](round9_group_calibration_results.json)
- [Tree selection and all eight comparisons](round9_tree_selection_2021.json)
- [Isolated experimental trainer](../src/round9_lightgbm.py) and [evaluator](../src/round9_tree_evaluate.py)

The randomized-threshold setting follows the [LightGBM parameter documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html#extra_trees): it remains a boosted model, with randomized threshold selection.
