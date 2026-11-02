# Read-only comparison with the user's main repository

Reviewed 2026-11-02 from `artifacts/reference_repo/deepthroatlearning-main/`. No reference files were edited. Sources: `docs/HANDOFF.md`, `docs/REPRODUCING.md`, `src/features.py`, `src/topology_features.py`, `src/tree_preprocessing.py`, `src/price_features.py`, `src/temporal_features.py`, `src/round5_lightgbm.py`, the Round5/Round6/Round9/Round10 reports, and `configs/round10_ranks_seed299_2021.json`.

## What the archived scores mean

The main repository documents a **0.99172 public Kaggle score** for an ensemble: 50% Round3, 25% separate-generator LightGBM seed299 and 25% seed673. This is an inherited recorded result, not a new submission from this workspace. The 113-feature separate-generator model alone is not that ensemble; Round6 reports its seed299 2021 AUC as **0.988261**. Round3 includes pooled trees, neural models, and a Haukeli specialist.

The documented two-rank candidate improves the full historical ensemble from 0.990374846 by +0.000105519 on 2021, and from 0.989411166 by +0.000155991 on native2022 January7 onward. Its public score is unknown. The latter check still falls below 0.99. Our complete2022 folds use 365 starts rather than the reference's primary359-start January7+ subset, so direct numbers are not perfectly matched. Neither historical results nor reused local holdouts establish an unbiased future score, and none establish the active Kaggle target of 0.999.

## Feature differences and actionable changes

| Feature/mechanism | Canonical113/115 design | Current local304 normalized design | Assessment |
|---|---|---|---|
| Whole-week price rank | Present in115 candidate, midpoint ties | Present, pandas average rank divided by168 | Essentially the same ordering; the tie convention differs only by constant1/(2×168). Do not duplicate it. |
| Remaining-week price rank | Present in115 candidate, current hour through end | Missing | **Highest-priority incremental input.** It expresses hourly dispatch priority within remaining opportunities. The archived matched rank experiment supports the pair, but cannot isolate the remaining-rank contribution. |
| Neighbor price summaries | Previous/next means over6/12/24h excluding current, plus price-minus-mean contrasts and one-hour changes | Point offsets ±1/3/6/12/24, daily means, week/remaining summaries | Add six local-window means and six contrasts as a bounded ablation. Startup/stop tradeoffs depend on blocks of future/past profit, not only a price at one offset. Physics DP already captures some of this. |
| System reservoir context |17 initial volumes,17 terminal values,17 weekly inflow volumes | All those inputs are present, plus extra hourly/cumulative/remaining flow terms | Already covered; do not remove indiscriminately. ReferenceRound9 removal lost0.004266/0.001322 on its native/frozen2022 checks. |
| Upstream aggregate topology | Initialvolume/fill, arithmetic terminal-value mean, weekly natural inflow | Catchment hourly inflow fraction; pooled local storage/supply; individual reservoir values | Explicit upstream state aggregates are missing. Tree models otherwise must learn reservoir combinations from individual columns. |
| Downstream topology | First receiving reservoir(s) fill, freecapacity, mean terminalWV; upstreamWVminusdownstreamWV proxy | Downstream head contributes to physical head; individual downstream inputs present | Add the simple aggregate state as empirical interactions. Arithmetic WV differences must not be described as physically converted water-volume costs. |
| Per-hour storage margin | Initialvolume minus current minimum-volume constraint | Initialvolume and minimum separately; physics subtracts max weekly minimum | Three cheap direct subtraction features are missing. Avoid conflating this initial margin with predicted evolving storage. |
| Monetary denominator |max(mean(abs(weekprice)),10) |mean(abs(weekprice))+1 | A meaningful difference for low-price training regimes. Test only after current normalization results are complete; use a matched ablation. |
| Scale-retention feature |log(denominator) |1/denominator | Both monotonically encode the same scalar given the same denominator; trees do not need both. Denominator changes matter more than log versus inverse. |
| Hydraulic approximation |Disabled in canonical113/115 configuration |Initialhead, pool supply, catchment inflow, spill pressure, startup/stop DP | Current local additions are new relative to this canonical branch, but their full-ensemble value is unknown. Avoid treating feature count as progress. |

The canonical feature builder is generator-specific, but its models already train separately by generator. GeneratorID is constant inside each such fit; it does not itself provide extra predictive information. Most meaningful differences are deliberate feature combinations and regularization, not the presence of the ID.

## Normalization evidence

Read-only measurement of supplied training input weeks found mean absolute weekly price below10 EUR/MWh in **61.75% of2020 cases** and **9.32% of2015 cases**, with no such cases in2016–2019 or2021–2022. Median2020 mean-absolute weekly price is7.145 EUR/MWh. Therefore canonical floor10 versus local+1 changes a substantial block of training features, not merely numerical stabilization. It does not follow that floor10 is better; compare fixed configurations before selecting.

Measurement used all168 supplied prices per case at exact UTC timestamps. No labels were used in this calculation. Python3.11.15, pandas3.0.6, NumPy2.4.6.

## Model configuration and experiments to avoid repeating

Canonical separate models use63 leaves, learningrate.04, minleaf100, maxbin63, L1.2, L2=4, featurefraction.85, rowsample.85 activated everyiteration, and rotating stride4 training rows. Our local recipe uses a different sample fraction, maxbin255, no L1, L2=1 and featurefraction.9. All are potential confounders in a direct feature comparison.

Canonical boosting rounds were fixed using2019/2020, separately by generator: Hogga278; Tokke712/464/421/207; Lio324; Byrte177; Vinje664/1250/355; Haukeli129; Songa246; Kjela261; Vesle573. One uniform large tree count may especially overfit Haukeli. ReferenceRound6 already found near-perfect training AUC and much lower heldout Haukeli/Lio/Hogga AUC; larger capacity and longer fitting did not yield a robust promoted model. Repeating an undirected capacity search is lower priority than a matched missing-feature ablation or complementary model.

Do not re-run the following as though untested:

- Shared r_Vest constraint timing: failed the reference confirmation gate.
- Staavatn/Venemo distinct inflow timing: failed two-seed confirmation and did not repair the severe late2022 Vesle case.
- Removing detailed system-wide state: strongly harmful in the reference.
- Removing absolute scale: did not pass promotion.
- Free plant-weight tuning or generic calibration: several archived variants failed transfer.
- Plain probability smoothing: our context and physics2021 selection prefers no smoothing. The reference's small learned logit residual is a different method, fitted on2021 and checked against same-hour and boundary controls.

## Suggested bounded next step

Finish the current normalized baseline. Then add **remaining-week midpoint price rank** with all other local settings fixed, followed by the previous/next price-window means as a separate change if resources permit. Keep the current full-week rank unchanged. Record complete2021/2022 and January7+2022 scores so the main-repository comparison is aligned. If either fails the explicit migration requirement of at least0.998 on both historical folds, keep it as research output and preserve the main recipe.

A separate canonical-configuration reproduction would help establish whether normalization/regularization alone explains the branch gap; it must be compared as a reproduction, not credited to a single new feature. The main ensemble's missing checkpoints/predictions prevent direct reconstruction from source files alone without retraining its components. Reused holdouts remain development checks.
