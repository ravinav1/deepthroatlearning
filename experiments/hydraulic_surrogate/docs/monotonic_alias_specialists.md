# Monotonic LP specialists

Run `lgb-monotonic-alias-specialists-001`; status `complete`.

Positive monotonic priors on each weak unit’s own LP dispatch/output/economic margin may reduce regime-sensitive corrections while preserving nonlinear unconstrained context.

Direction audit before fitting:

- `lp_qmin_ratio`: q/Qmin from input-only LP: greater modeled discharge relative to minimum operating flow supports greater commitment; positive shape prior retained.
- `lp_power_fraction`: LP power/Pmax: greater modeled production supports greater commitment; positive shape prior retained.
- `lp_economic_margin`: (price-(lambda_up-lambda_down)/Ebest)/(1+mean(abs(price))): a larger relative operating margin supports commitment; positive shape prior retained.
- `limitations`: These are conditional modeling priors, not physical necessities. LPv1 omits directional storage, integer minimum flow and startup costs; output features are correlated and independently changing one while fixing all others may be infeasible. Forced generation can occur at negative margins. No negative direction follows from that fact, and no supplied definition supports reversing a sign. Retain all three requested +1 directions and constrain no other column.
- `method`: Explicit basic method (LightGBM default), monotone_penalty0; no method or penalty search.

All other feature directions are0. LightGBM requires the full feature-order vector; every vector and named index is saved before fitting. [Official parameter documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html#monotone_constraints).

Predeclare joint specialist weights[0,.5,1] on Haukeli/Lio/Hogga only. Choose highest2021 pooled microAUC; exact ties prefer smaller weight. Freeze choice before reading2022 validation predictions/label slice, then evaluate only that weight on2022. All other units remain the original baseline probabilities.

|Fold|Baseline micro AUC|Frozen specialist blend|Delta|
|---|---:|---:|---:|
|2021|0.994370012291|0.994468404240|+0.000098391950|
|2022|0.993348542164|0.993310287283|-0.000038254881|

Frozen weight: `0.5`. Independent2022 improvement: `False`. Both folds≥0.998: `False`.

Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/monotonic_specialists.py --run-id lgb-monotonic-alias-specialists-001 --include-affine-aliases --gate-run artifacts/lgb-monotonic-specialists-001/attempt-002 --gate-run artifacts/lgb-normalized-lp-context-final-001`. Runtime 316.08s. Complete source snapshots, all data/input/model/sample/artifact hashes, environment, split dates/IDs, model settings, direction audit and saved predictions: `artifacts/lgb-monotonic-alias-specialists-001/manifest.json`.

The baseline’s exact395 features, saved sampled rows and six-day purge are reused. Only the three specified units are refit, using350 trees,31 leaves, learning rate0.06, minimum100 rows/leaf,L2=1,column fraction0.9,maxbin255,seed4173 and2 threads. No early stopping, extra feature, projection, smoothing or calibration. Both years are reused project development folds;2022 is held out only from this experiment’s weight selection. No Kaggle action or transfer.

Corrected alias policy: all exact positive affine aliases of each original target coordinate are also constrained +1. Aliases are detected on every row of the complete legal395-feature table without labels or scores; no approximate aliases or score-based exceptions. Full slopes/intercepts and expanded vectors were frozen before fitting.

The corrected candidate is rejected. Its frozen 50% blend gains 0.000098391950 on 2021 but loses 0.000038254881 on 2022. The full legal-table audit found exactly the expected own-plant power aliases (slope 1, intercept 0), with no additional exact positive affine aliases; each saved model has four positive constraints. All artifact hashes, exact baseline sample files, saved constraint vectors, case/unit/label alignment, finite bounded predictions, mixing and fold AUCs passed the post-run audit. No final model fit or transfer is planned for either monotonic variant.
