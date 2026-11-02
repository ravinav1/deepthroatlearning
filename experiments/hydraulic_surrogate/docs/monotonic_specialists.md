# Monotonic LP specialists

Run `lgb-monotonic-specialists-001/attempt-002`; status `complete`.

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
|2021|0.994370012291|0.994431002189|+0.000060989899|
|2022|0.993348542164|0.993297920824|-0.000050621340|

Frozen weight: `0.5`. Independent2022 improvement: `False`. Both folds≥0.998: `False`.

Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/monotonic_specialists.py --run-id lgb-monotonic-specialists-001/attempt-002 --reuse-failed-run artifacts/lgb-monotonic-specialists-001`. Runtime 276.19s. Complete source snapshots, all data/input/model/sample/artifact hashes, environment, split dates/IDs, model settings, direction audit and saved predictions: `artifacts/lgb-monotonic-specialists-001/attempt-002/manifest.json`.

The baseline’s exact395 features, saved sampled rows and six-day purge are reused. Only the three specified units are refit, using350 trees,31 leaves, learning rate0.06, minimum100 rows/leaf,L2=1,column fraction0.9,maxbin255,seed4173 and2 threads. No early stopping, extra feature, projection, smoothing or calibration. Both years are reused project development folds;2022 is held out only from this experiment’s weight selection. No Kaggle action or transfer.

The literal three-column experiment is rejected: its frozen 2022 score decreases. This result must not be described as enforcing monotonicity along the coupled physical feature path. For each target single-unit plant, `lp_power_fraction_<unit>` is exactly duplicated by the unconstrained `lp_plant_power_fraction_<plant>` column across the complete cached feature table. The model can compensate through that alias. The requested vectors were left unchanged throughout this run.

The initial attempt at `artifacts/lgb-monotonic-specialists-001/` failed after 66.90 seconds because the raw-unit diagnostic print lacked a `roc_auc_score` import. Its completed 2021 Haukeli checkpoint, source snapshot and failed manifest remain intact. The corrected `attempt-002` run verified that checkpoint's hash, exact raw data, feature source, parameters and sampled rows before reusing it; it finished in 276.19 seconds. No candidate score or selected blend existed when the original attempt failed.

Post-run audit verified every recorded artifact hash, exact baseline sample files and blending, unchanged case/unit/label order, finite bounded probabilities, reproduced fold AUCs, and all six checkpoint constraint vectors: exactly three positive entries and 350 rounds per model. No final fit or main-repository transfer is justified by this result.
