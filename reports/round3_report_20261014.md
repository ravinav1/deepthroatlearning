# Temporal features and generalization investigation

**Completed 14 October 2026.** One candidate is fitted, exported and independently verified: [submission_round3_temporal_ranks.csv](../submissions/submission_round3_temporal_ranks.csv). The user subsequently reported a **public Kaggle score of 0.99068**, improving on 0.99053 by **0.00015**. No Kaggle upload was performed by the agent.

Two price-rank features are the only change advanced from this round. They improve the current ensemble on both historical years and in the second-future-year frozen-model check. The gains are small, with weaker evidence in 2021 and the frozen check. The **0.99068 submission is now the user-reported public best**; the previous 0.99053 submission remains preserved. This public-2023 result supports the direction of the historical improvement, but does not establish private-2024 performance or identify which generators improved on Kaggle.

The [experiment ledger](autonomous_round3_20261014.json) records selection, rejected experiments, artifacts and verification. Heavy fitting was sequential with four CPU threads; independent analysis and inference used one thread.

## First check: do the previous improvements survive another year?

The saved pre-2021 models were evaluated on 2022 without fitting or updating scalers. The latest recipe scores **0.986403634**, versus **0.985721661** for the older frozen recipe: **+0.000681973**. Excluding January 1–6, the gain is **+0.000658124**. Thirteen of fourteen generators improve, including Lio and Haukeli.

This supports transfer of the previous combined recipe; it does not establish a universal generalization failure. It also shows that Lio's earlier regression with separately refitted models was not universal. Changes between years still mix model age, economic conditions and case difficulty. [Frozen comparison](round3_frozen_ensemble_comparison.json), [independent verification](round3_frozen_verification.json).

This check resembles the deployment gap described in the course slides: training through 2022, public evaluation in 2023, private evaluation in 2024. Different historical conditions prevent treating it as a guarantee about 2024.

## What the EDA found

- **No consistent early/late switching bias.** At threshold 0.5, roughly 72–73% of true ON transitions match a predicted transition within six hours, with 44–46% exactly timed. Lio's early/late balance reverses between years. A fixed shift is not justified.
- **Lio is not only a transition problem.** After separating missed, extra and boundary-censored runs, roughly 46–48% of its threshold-0.5 errors remain more than six hours from true transitions. Some difficult weeks are actually ON for long periods while the model predicts low probabilities.
- **Short opportunities are difficult.** Short positive-margin blocks near break-even have worse ranking than longer blocks in both years. Higher BCE and threshold-0.5 error also persist in matched case/generator/narrow-margin comparisons. These are descriptive findings, not proof of a causal effect.
- **Input conditions change.** For calendar-matched starts, median weekly price SD divided by the existing economic scale rises from 0.186/0.219 in 2021/2022 to 0.288/0.399 in 2023/2024. Hourly changes in price rank are much steadier. Lio's median longest positive-margin block is 5/3/14/42 hours across those years. Monthly comparisons account for seasonality; no competition labels are used.

Input comparisons use January 1–December 25, excluding February 29: 359 starts per year. Rank marginals are mechanically constrained by construction; their apparent stability alone is not predictive evidence. Positive price minus local terminal water value is an economic proxy, not a rule forcing a generator ON.

Useful figures:

- [Seasonal input comparisons](figures/round3_inputs_seasonality.png)
- [Month-matched distribution shifts](figures/round3_inputs_shift.png)
- [Timing and error decomposition](figures/round3_errors_timing.png)
- [Illustrative difficult weeks](figures/round3_errors_examples.png)

Full evidence: [input EDA](round3_inputs_analysis.json), [error EDA](round3_errors.json).

## Controlled experiments

Each experiment retained the baseline's 748 trees, depth 8, learning rate 0.05, seed 42, other parameters, data and plant blend weights. Replacement versus a 50/50 old/new component mean was selected on 2021 and recorded before examining that candidate's 2022 results.

1. **Ranks:** two midpoint ranks for the current price, among the full week and among current-to-end hours. Ties share a rank; a flat or single-hour suffix has rank 0.5. Existing price, water-value and economic-scale features remain unchanged. EDA used endpoint-normalized whole-week ranks; the engineered midpoint convention is an affine transformation of those ranks with the same ordering.
2. **Favourable periods:** seven features describing duration, elapsed/remaining time, previous/next distance and boundary censoring for positive price–water-value blocks. All calculations stay within the supplied case. These differ from the fixed 6/12/24-hour economic-window features tested previously.
3. **Regularization control:** increase XGBoost's minimum child Hessian weight from 25 to 100. This discourages small leaf splits; it is not a minimum count of 100 observations. [XGBoost parameter documentation](https://xgboost.readthedocs.io/en/stable/parameter.html). The control tests regularization at the fixed budget, not its best possible tuned configuration.

| Change and 2021-selected variant | 2021 ensemble gain | Frozen 2022 Jan 7+ gain | Decision |
|---|---:|---:|---|
| Child weight 100, component mean | +0.000000522 | −0.000047573 | Reject |
| Favourable periods, replacement | +0.000064676 | −0.000062524 | Reject |
| Price ranks, replacement | +0.000065613 | +0.000048635 | Confirm with 2022 refit |

The failed frozen checks stopped the first two experiments before a native-2022 or full-data fit. The favourable-period mean was nearly flat on frozen 2022, but was not substituted after seeing those results. The two feature families were not combined because the period family failed its individual gate.

## Selected rank candidate

Only the existing depth-8 XGBoost contribution is replaced. Every plant coefficient, LightGBM, Transformer and Haukeli component remains unchanged.

| Evaluation | Current recipe | Rank candidate | Gain |
|---|---:|---:|---:|
| 2021, trained before 2021 | 0.988918450 | 0.988984063 | +0.000065613 |
| 2022, trained before 2022 | 0.988375535 | 0.988606658 | +0.000231123 |
| Same 2022 fit, Jan 7+ | 0.988277942 | 0.988512663 | +0.000234722 |
| 2022, frozen before 2021 | 0.986403634 | 0.986460892 | +0.000057258 |
| Same frozen models, Jan 7+ | 0.986370384 | 0.986419019 | +0.000048635 |

Paired 14-day whole-case bootstrap intervals, with 200 draws, give these approximate 95% intervals for the AUC gain:

- 2021: **[−0.00003195, +0.00017090]**.
- Native 2022: **[+0.00012462, +0.00035306]**; Jan 7+ **[+0.00012754, +0.00037041]**.
- Frozen 2022 Jan 7+: **[−0.00006167, +0.00017723]**.

The native-2022 evidence is stronger. In that fit, AUC within two hours of true transitions improves by 0.001707 and short positive-block AUC by 0.001494. Overall Lio AUC improves by 0.000479, but within-Lio transition AUC falls by 0.002393. Lio also regresses in 2021 and frozen 2022. This is a modest overall improvement with an unresolved generator-specific tradeoff.

[Comparison](round3_xgb_ranks_both_comparison.json), [2021 diagnostics](round3_ranks_2021_diagnostics.json), [2022 diagnostics](round3_ranks_2022_diagnostics.json), [frozen uncertainty](round3_ranks_frozen_uncertainty.json).

## Validation limits and implementation

Historical validation fits exclude the six case starts whose weekly targets overlap validation. Final fitting uses all supplied labelled cases. The additional Jan 7+ check removes 2022 starts overlapping the final 2021 selection horizons. **Both years are reused development data.** Native and frozen 2022 are two deployment scenarios on the same labels, not independent replications. Bootstrap intervals are conditional on the selected predictions and do not correct repeated experiment selection. Neither positive historical scores nor the public leaderboard establish private-2024 performance.

The new options are `use_temporal_ranks` and `use_favourable_periods`, both off by default. They are wired into the shared feature builder, XGBoost trainer and full-submission builder. Existing defaults reproduce the original features and saved-model predictions exactly. Independent checks cover ties, case boundaries, monetary scaling, row order and seven real cases spanning 2015–2024. No neighbouring-case target features are used. [Math checks](round3_temporal_feature_checks.json), [integration checks](round3_feature_integration_checks.json).

## Delivery and deferred work

The [candidate submission](../submissions/submission_round3_temporal_ranks.csv) contains **725 rows and 2,352 probabilities per row**, plus `Run No`, in the exact template order. The model uses all 2,922 supplied labelled cases, with zero overlap with competition IDs, and the same fixed 748 trees. The existing best submission and models are preserved. Kaggle upload remains manual.

Independent verification reconstructed all **1,705,200** blended probabilities with zero difference. Saved-checkpoint replay on 17 fixed competition cases agreed exactly at float32 precision; the maximum CSV difference was 2.98×10⁻⁸. This is a sampled checkpoint replay, not a claim that every case was independently rerun. All **402 protected originals** remain unchanged, including raw data, original models, the previous best submission and `kaggle_metric.py`. The supplied metric independently reproduced the selected historical scores. [Final verification](round3_final_verification.json), [export provenance](submission_round3_temporal_ranks_run.json), [full-fit provenance](round3_xgb_ranks_full_run.json).

Submission SHA-256: `6a19fc33a14e492a25914d678785162a010ec78d42e3e4db3ac06b7379aac4d9`.

Deferred: combining rejected period features, further blend-weight searches, applying these features to other model families, and new architecture searches. A subsequent bounded investigation should prioritize the already-prepared neural learning-rate/weight-decay controls and additional chronological checks, rather than assume more features will resolve Lio. Those controls were not trained in this round.
