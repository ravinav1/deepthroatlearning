# Separate generator LightGBM experiment

**Completed 14 October 2026 within the two-hour budget.** One verified candidate is ready: [submission_round5_lgb_generators.csv](../submissions/submission_round5_lgb_generators.csv). It combines **50% of the existing round3 ensemble with 50% of new generator-specific LightGBM predictions**, averaged equally across seeds 299 and 673. The new models use relative monetary inputs. After delivery, the user reported a Kaggle public score of **0.99172**, up **0.00104** from the previous user-reported best of **0.99068**. This result is attributed to the delivered candidate from the conversation context; it was not independently retrieved from Kaggle. The historical improvement also appeared on the public leaderboard, though the gain cannot be assigned to a single change in this combined experiment.

## Historical results

| Evaluation | Existing ensemble AUC | Candidate AUC | Gain |
|---|---:|---:|---:|
| 2021, trained before 2021 | 0.988984063 | 0.990374846 | +0.001390782 |
| 2022, trained before 2022 | 0.988606658 | 0.989497911 | +0.000891254 |
| Same 2022 fit, Jan 7 onward | 0.988512663 | 0.989411166 | +0.000898503 |
| 2022, pre-2021 training and no refit, Jan 7 onward | 0.986419019 | 0.987045304 | +0.000626285 |

The family and blend were [selected on 2021](round5_selection_2021.json) before evaluating candidate predictions on 2022. The same choice passed the frozen-model check and the subsequent refit check. No alternative was selected after seeing 2022. [Historical comparison](round5_selected_historical_comparison.json), [frozen comparison](round5_selected_frozen_2022_comparison.json).

## Which ideas helped

- **Separate classifiers were useful.** With the original parameters and 800 trees per model, standalone 2021 AUC rose from **0.983881** for the pooled 800-tree model to **0.987614** for 14 separate models. This also increases total capacity: it does not isolate generator specialization from having more trees overall. [Matched comparison](round5_structure_matched800.json).
- **Every-fourth-hour sampling preserved quality and reduced runtime.** The controlled separate-model run fell from **681 to 240 seconds**. Its fixed 50/50 blend with the incumbent changed from **0.989975 to 0.990070** on 2021. The small score difference alone does not establish an accuracy improvement. Complete weekly inputs are built first; only training rows are thinned. All 168 hours are predicted and evaluated.
- **The peer parameter bundle with relative prices improved the selected blend.** Using seed 299 gave **0.990275** on 2021; averaging seeds 299 and 673 gave **0.990375**. Seed 673 was used through uniform averaging, not selected as a preferred seed. The seeds also select different hourly sampling offsets.
- **Absolute prices were useful at a smaller contribution, but did not win this comparison.** Their standalone seed-299 AUC was **0.965145**, yet averaging that model into the existing LightGBM contribution improved ensemble AUC. The peer-style combination of 50% incumbent, 25% absolute and 25% relative models, each new component averaging both seeds, scored **0.989450** on 2021—below the selected relative-only addition.

All eight screened families and their three predefined blend variants are recorded in the selection report. The existing ensemble includes XGBoost and neural components; it is not the other group's unspecified legacy LightGBM ensemble.

## Training setup

Each seed has one classifier per generator, trained on hourly rows with the existing features. The selected representation has **113 features**. Monetary inputs are divided by `max(mean(abs(weekly_price)), 10)` and retain `log_week_price_scale`; this is the existing normalization, not an exact reconstruction of the unspecified V5 features. The absolute variant retains unscaled inputs, including price minus the weekly mean.

The new models use 63 leaves, learning rate 0.04, minimum 100 rows per leaf, L1 0.2, L2 4, and 85% row/feature sampling, with row sampling activated every iteration. `max_bin=63` is retained. The sampling rule is `hour % 4 == (Run No + seed) % 4`, giving 42 training hours per generator and case.

Separate 2019 and 2020 fits selected tree counts with a ceiling of 1,800 and patience 100. Their validation case starts end on December 25 so target weeks finish within that year. Each generator uses the median of its two raw best counts, rounded half up and then clipped to 100–1,800. The selected relative counts range from **129 to 1,250**. Counts are fixed for 2021, 2022 and final fitting; both seeds share that schedule. [Count provenance](round5_tree_count_selection.json), [example final configuration](../configs/round5_lgb_peer_relative_seed299_full.json).

Validation training requires each target week to finish by the validation start: the last training start is at least seven days earlier, excluding six overlapping daily starts. The additional Jan 7 check excludes 2022 cases overlapping late-2021 selection horizons. Both evaluation years are reused development data; frozen and refitted 2022 checks share labels and are not independent replications.

## Stability and remaining weaknesses

Paired circular 14-day whole-case bootstrap intervals, using 200 draws, give these approximate 95% intervals for the overall gain:

| Evaluation | Gain interval |
|---|---:|
| 2021 | [+0.000749, +0.002328] |
| Refitted 2022, Jan 7 onward | [+0.000560, +0.001211] |
| Frozen 2022, Jan 7 onward | [−0.000157, +0.001366] |

All four case-start quarters improve in 2021 and refitted 2022. All 14 generators improve in the refitted 2022 Jan 7 check. However, **Lio's gain varies across evaluation setups**: its within-generator AUC gains 0.01342 in 2021 and 0.00201 in refitted 2022, but loses 0.01098 in frozen 2022. Haukeli loses 0.00203 in 2021 before improving in both 2022 checks. These comparisons do not identify the cause. The bootstrap intervals condition on the chosen models and do not correct repeated model selection.

[2021 diagnostics](round5_selected_2021_diagnostics.json), [refitted 2022 diagnostics](round5_selected_native_2022_diagnostics.json), [frozen 2022 diagnostics](round5_selected_frozen_2022_diagnostics.json).

The parameter bundle was tested together; the individual effects of leaf count, penalties and row/feature sampling remain unresolved. Exact peer replication also requires their V5 feature definitions and legacy ensemble. No new hydraulic features or further parameter search were added in this run.

## Delivery and verification

Both final seeds trained on all **2,922 supplied labelled cases**, producing 28 saved models. Competition case IDs have zero overlap with training IDs. Heavy fits ran sequentially on four CPU threads and have finished. Raw data, earlier models/submissions and `kaggle_metric.py` remain unchanged; all **432 protected originals** passed checksum checks.

The CSV has **725 rows, 2,352 probabilities per row, and `Run No`**, in template order. Independent reconstruction matched all **1,705,200** final blend probabilities and every seed-mean probability exactly. Replaying all 28 saved models on five fixed test cases per seed also matched exactly; this is sampled model replay, not an independent replay of every test case. [Final verification](round5_final_verification.json), [submission recipe](../configs/round5_submission_lgb_generators.json), [export provenance](submission_round5_lgb_generators_run.json), [run ledger](autonomous_round5_20261014.json).

The implementation is isolated in new `round5_` scripts and configurations. For a later rerun, copy a configuration to a new filename and update or remove its `deadline_utc`; the trainer deliberately refuses to overwrite an existing run. The automated run performed no Kaggle upload or git push; the user subsequently reported the public result above.
