# Case-fraction calibration experiment

Run `case-calibration-001`, 2026-11-02T09:54:49.737363+00:00. Status: complete.

A case-level commitment-fraction model can correct level errors in hourly forecasts while retaining within-case ranking.

Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/case_calibration.py --run-id case-calibration-001 --base-run artifacts/lgb-context-001`

Runtime: 69.01s. Exact source/data/artifact hashes, environment, all model parameters, input feature definitions, row IDs and base prediction provenance: `artifacts/case-calibration-001/manifest.json`.

The model regresses each generator's weekly on-fraction from 231 initial-row base features, the full 168-hour price vector, and 36 inflow maximum/standard-deviation features. It has fixed 200 rounds, 15 leaves, minimum 20 cases/leaf, learning rate 0.04, seed 4173 and 2 threads. No early stopping or validation fitting. Chronological case starts use a six-day purge.

Predeclared selection: choose one global blend or a blend limited to Lio/Haukeli, weights 0, 0.1, 0.25, 0.5, by 2021 micro AUC. Freeze before loading 2022 predictions. The adjustment shifts each case-unit's hourly logits to match the independent regressor's fraction, then blends probabilities.

- 2021: baseline 0.980653166; frozen-selection result 0.982402139; delta +0.001748973. Train ['2015-01-01 00:00:00+00:00', '2020-12-25 00:00:00+00:00']; validation ['2021-01-01 00:00:00+00:00', '2021-12-31 00:00:00+00:00'].

|2021 scope|Weight|Micro AUC|
|---|---:|---:|
|global|0.0|0.980653166|
|global|0.1|0.980935408|
|global|0.25|0.980061163|
|global|0.5|0.977277990|
|weak_units|0.1|0.981722548|
|weak_units|0.25|0.982165706|
|weak_units|0.5|0.982402139|

- 2022: baseline 0.984358034; frozen-selection result 0.983619831; delta -0.000738204. Train ['2015-01-01 00:00:00+00:00', '2021-12-25 00:00:00+00:00']; validation ['2022-01-01 00:00:00+00:00', '2022-12-31 00:00:00+00:00'].

Frozen choice: `{'scope': 'weak_units', 'weight': 0.5}`. Positive independent 2022 improvement: `False`.

Per-unit AUC/fraction errors and frozen adjusted validation predictions are saved in the artifact directory. Models trained on all provided labels and test fraction targets are saved only if the selected adjustment improves 2022. This adjustment has only been tested against the specified base model; applying it to a stronger hourly model requires separate validation.

Kaggle submission ID, leaderboard score, rank and observation time: none; this is local validation only.

The frozen strategy fails the independent 2022 holdout and should not be integrated into the final prediction recipe. For diagnosis only, Lio's unit AUC deteriorates in both years (0.959224 to 0.948579 in 2021; 0.943692 to 0.913999 in 2022), while Haukeli's improves (0.912147 to 0.933344; 0.930624 to 0.934196). A Haukeli-only strategy was not among the predeclared candidates; testing it now would be exploratory and would need new independent validation.

A separate numerical check confirmed that bisection reaches the specified case means, preserves within-case probability ranks, and returns unchanged probabilities at zero blend weight. No final case-fraction models were fitted because the holdout test failed. The fold models and target fractions remain available for error analysis.
