# CatBoost specialists

Run `catboost-lp-specialists-001`; training status `complete`; decision **reject**. The 2021-selected blend worsens 2022, so no final fit, submission or transfer is warranted.

A fixed CatBoost learner may provide complementary commitment errors to the LP-v1 LightGBM baseline on the three highest-priority weak generators.

Predeclared joint replacement weights were `[0, .25, .5, 1]` for Haukeli, Lio and Hogga only. Select the highest 2021 pooled micro AUC; exact ties prefer the smaller weight. The choice was persisted before reading the 2022 validation predictions and label slice. Other 11 units retain their exact baseline probabilities.

|Fold|LP-v1 baseline|Frozen CatBoost blend|Delta|
|---|---:|---:|---:|
|2021|0.994370012291|0.994525542721|+0.000155530430|
|2022|0.993348542164|0.993245083821|-0.000103458343|

Frozen joint weight: `0.5`. The candidate does not pass the required 0.998 threshold on either year. The 2021 grid scores were 0.994370012291, 0.994504231894, 0.994525542721 and 0.994301821433, respectively.

The three units use the same 395 normalized-LP-v1 features, exact stored baseline sampled rows, and six-day purged chronological splits. CatBoost settings were fixed before fitting: 600 iterations, depth 6, learning rate 0.08, L2 leaf regularization 5, `border_count=128`, `random_strength=1`, Bayesian bootstrap with `bagging_temperature=1`, Plain boosting, Logloss, seed 4173 and 2 CPU threads. There were no categorical features, evaluation set, early stopping, extra calibration, feature changes or parameter search. Every fitted model’s complete expanded defaults are saved through `get_all_params()`.

Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/catboost_specialists.py --run-id catboost-lp-specialists-001`. Runtime: 633.43s. Full source snapshots, raw input/cache/model/sample hashes, environment, expanded parameters, exact split dates/IDs, selected probabilities and checkpoint replay checks: `artifacts/catboost-lp-specialists-001/manifest.json`.

Training started at 2026-11-02 12:07:49.440481 UTC. The blend was frozen at 12:12:45.661566 UTC. Manifest SHA-256: `32b6e5155450ad56e71f8486c6622ecdd1e0d36e45f0a4678dee3483e4c65a53`.

The frozen 2022 blend improves Haukeli unit AUC from 0.963653518078 to 0.966569466630 but reduces Lio from 0.973547589897 to 0.969686199810 and Hogga from 0.986279537479 to 0.985879218971. These post-selection diagnostics do not justify another weight search or choosing individual units from the observed 2022 results.

Independent artifact verification passed: all 48 recorded hashes, exact sampled rows, label/ID/unit alignment, frozen probability blending, untouched other units, finite bounded probabilities, pooled AUC, and all six 600-tree checkpoint structures and expanded parameters. During training, loading every checkpoint reproduced its first 64 validation-row probabilities exactly. The verification record is `artifacts/catboost-lp-specialists-001/verification.json`.

The first artifact verification failed because reloaded CatBoost parameter metadata rounds two floats by at most 9.89e-12. The failed verification script and failure record are retained. The corrected verifier permits absolute error below 1e-9 only for floating-point parameter metadata; prediction, sample and mixing comparisons remain exact. No model or prediction was changed.

Both years are reused project development folds. Only 2021 selects this experiment’s joint blend; 2022 is checked with that weight fixed. No GPU dependency, installation, network request, Kaggle action or main-repository transfer occurs in this script.
