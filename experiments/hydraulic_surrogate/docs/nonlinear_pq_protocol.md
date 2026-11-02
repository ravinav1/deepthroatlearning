# Nonlinear PQ profit and switching specialists

Prepared 2026-11-02 after independent review by the combined-export agent and before any specialist fitting. Root authorized one local heavy slot for the 2021 run only. The 2022 run requires the immutable 2021 selection lock; no 2022 model may run before it. Frozen historical feature modules, caches, models and GPU protocols remain unchanged.

## Motivation and scope

The accepted four-component blend still attributes 40.53% / 39.91% of pooled AUC error to Haukeli, Lio and Hogga on the reused 2021 / 2022 folds. Their Brier case-mean contributions are 18–44%, so within-week scheduling remains material. The exact pair-error and Brier identities were checked in `artifacts/current-blend-residuals-001/manifest.json`; these are descriptive diagnostics, not new holdouts or deployable label-derived features.

The rejected `lp_switching.py` experiment used LP opportunity cost at constant best-point conversion multiplied by nominal Pmin/Pmax. This distinct ablation evaluates actual nonlinear, physically feasible on-state production before applying switching costs. It does not repeat the rejected whole-case mean calibration, reservoir history, locality restriction, head re-solve or large per-subset Tokke/Vinje switching LP.

SHOP describes head/discharge-dependent turbine efficiency, power-dependent generator efficiency and original operating points before convexification. That supports testing the original physical production curve instead of reusing its relaxed origin-to-best-efficiency extension. [Official PQ curves](https://docs.shop.sintef.energy/examples/pq_curves/pq_curves.html).

SHOP's non-MIP commitment cutoff defaults to half the minimum flow, while binary and relaxed modes differ. The original competition commands are unavailable. This feature experiment does not assert that the supplied labels use either particular mode or threshold. [Generator commitment and fractional starts](https://docs.shop.sintef.energy/objects/generator/generator.html), [turn-off command](https://docs.shop.sintef.energy/commands/set_gen_turn_off_limit.html).

## Frozen feature equations

Targets, in order: Haukeli_G1, Lio_G1, Hogga_G1. Keep the exact existing 608-column context/joint prefix and append two fields per target, totaling 614 columns: `nlpq_profit_<unit>` and `nlpq_switch_<unit>`.

At the same initial gross head and own hydraulic-loss convention as LP v1, compute original P(q), with eight generator-efficiency fixed-point iterations and no convex hull. Use 257 initial discharge scan points plus all supplied turbine-efficiency breakpoints. Add numerical roots of effective-head positivity, head-dependent minimum/maximum turbine discharge and nominal Pmin/Pmax boundaries. Fill the physical on-domain with 257 points between its endpoints, retain feasible knots/roots, and recheck all constraints at every point. Include feasible Qmin and Qmax found to numerical root tolerance; retain no zero-discharge off point. Fail with a recorded error if the on-domain is empty. Initial heads are cached by exact float values without rounding. Maximization is over this fixed finite feasible grid, not the exact continuous optimum; the sign-change root scan could miss an extremely narrow feasible interval. No grid size is tuned using labels.

For each hourly known price, calculate `r_t=max_q[price_t*P(q)-0.0036*q*(lambda_up,t-lambda_down,t)]`, where raw frozen LP v1 duals are in EUR/Mm3 and q is m3/s. The `.0036` conversion occurs once. Preserve negative best-on rewards; off is a separate state with zero operating reward. Use the existing tested full-168-hour max-sum DP with the supplied asymmetric startup/shutdown costs and free initial/final state. Normalize both on-reward and DP on-minus-off advantage by `(1+mean(abs(weekly_price)))*nominal_Pmax`, once. Outputs are float32; raw on-rewards are retained in float64.

Only case metadata, storage at case start, full-horizon supplied prices, static topology and already verified input-only LP duals enter these six fields. Future realized storage, commitments and solver runtime are excluded. Frozen LP duals do not respond to changed dispatch, and the DP does not enforce a new water balance or environmental constraints. This is an auxiliary scheduling prior, not a feasible hydraulic schedule or an exact SHOP reconstruction. Monetary cost conventions and fixed initial head remain inherited limitations.

## Fixed supervised and selection protocol

Fit only three specialists. Reuse exact 608-feature source model settings except two threads: 350 trees, 31 leaves, learning rate .06, min child samples 100, L2=1, column fraction .9, max bin255, deterministic column-wise LightGBM, seed4173, no early stopping. Reuse the exact saved 40% sampled rows and six-day purge from `lgb-normalized-lp-context-joint-001`; load no post-2021 target values for the first run. Hash all source/data/cache/sample references before training and save every model plus exact checkpoint replay.

The fixed anchor is `lp-context-joint-comparison-001` with weights `[.140625,.421875,.1875,.25]` in base/context/joint/combined order. For a single global target-unit replacement weight w from `[0,.1,.25,.5]`, blend each of the three target columns as `(1-w)*anchor+w*new_specialist`; all other columns remain unchanged. Apply the existing Vinje projection. Exact ties prefer the lower weight. Choose w using only 2021 pooled micro AUC, then exclusively create the selection lock before any 2022 fit/evaluation.

At the selected w, also compute a matched control by splicing saved raw 608-feature source predictions into the same anchor with the same weight. No control retraining or new control grid is needed. A nonzero candidate must beat both the anchor and same-weight control in 2021 and 2022. The 2022 command verifies the explicit lock SHA256 and evaluates only the locked weight and its references. Rejection preserves all arrays/checkpoints; it does not trigger another weight choice. Neither development year is an untouched test. Main-repository transfer still requires .998 in both years, and leaderboard success requires actual Kaggle evidence.

## Checks before fitting

Five small tests cover exact physical power/flow endpoints and unhulled efficiency knots, one-time water conversion, negative on-profit, effective-head feasibility and empty-domain failure, brute-force switching optimality with free boundaries, metadata-only target access and invariance to poisoned post-start storage. Run a one-case input-only timing first. The initial separate metadata diagnostic invoked system Python without NumPy and failed before reading inputs; the successful retry used the existing uv environment. No package was installed.

```sh
UV_CACHE_DIR=/private/tmp/tdt4173-xgb-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python -m unittest tests.test_nonlinear_pq -v
UV_CACHE_DIR=/private/tmp/tdt4173-xgb-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/build_nonlinear_pq_features.py
```

Existing run IDs must not be overwritten or rerun. Source/configuration snapshots and exact commands belong to each artifact manifest. The best-profit tutorial was also consulted, but its full page exceeded the web tool's retrieval limit; no claim relies on unread content.
