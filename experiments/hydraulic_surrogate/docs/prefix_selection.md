# Generator-specific prefix selection

Run `prefix-selection-001/attempt-002`; status `complete`.

Early overfitting differs by generator; frozen per-generator prefixes may improve pooled micro AUC without changing fitted trees.

Exact command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/prefix_selection.py --run-id prefix-selection-001/attempt-002 --base-run artifacts/lgb-normalized-context-001`

Predeclared prefixes [100,150,200,250,350]. Compare baseline350, per-unit maximum AUC, and per-unit minimum log loss on2021. Per-unit exact ties prefer larger prefixes. Choose one strategy by highest2021 pooled microAUC; exact ties prefer baseline350 then unitAUC then unitLogLoss. Persist all counts and the primary choice before opening2022. Evaluate only the primary frozen strategy on2022. No2022 feedback into settings.

2021 has already been reused for model-family and feature development; its selection score is optimistic. 2022 is an untouched holdout for this prefix choice, but it has also informed earlier project-level choices and is not a pristine final test. Local AUC is not a leaderboard score.

Complete hashes, model parameters, feature definitions, raw input paths, split IDs, source snapshot, environment and runtime are in `artifacts/prefix-selection-001/attempt-002/manifest.json`.

Frozen primary strategy: `unit_log_loss`, frozen at `2026-11-02T10:27:19.461885+00:00` before opening 2022 predictions.

|Generator|AUC rule|Log-loss rule|Primary|
|---|---:|---:|---:|
|Hogga_G1|100|250|250|
|Tokke_G1|250|250|250|
|Tokke_G2|350|200|200|
|Tokke_G3|350|200|200|
|Tokke_G4|250|200|200|
|Lio_G1|350|150|150|
|Byrte_G1|250|150|150|
|Vinje_G1|350|350|350|
|Vinje_G2|250|250|250|
|Vinje_G3|350|350|350|
|Haukeli_G1|150|100|100|
|Songa_G1|100|250|250|
|Kjela_G1|100|100|100|
|Vesle Kjela_G1|350|350|350|

|2021 strategy|Micro AUC|Log loss|
|---|---:|---:|
|baseline_350|0.989849818|0.132524863|
|unit_auc|0.989919349|0.129441261|
|unit_log_loss|0.990343518|0.126997665|

2021: baseline micro AUC 0.989849818435; selected 0.990343518048; delta +0.000493699614.

2022: baseline micro AUC 0.989469422412; selected 0.989269099257; delta -0.000200323155.

Both folds ≥0.99: **False**. Independent 2022 improvement: **False**.

No refitting, early stopping, learned preprocessing, calibration or ensembling. The fixed pretrained model seed and sampling settings are inherited verbatim. Predictions retain `[0,1]` probabilities and original case-major/hour-major order. Prefix raw scores are accumulated over nonoverlapping tree ranges; 350-tree output must match the original saved predictions within 1e-7.

Kaggle submission ID, leaderboard score, rank and observation time: none; no Kaggle action performed.

The initial `prefix-selection-001` attempt failed after 3.16 seconds, before selecting or scoring prefixes: the strict checkpoint feature-name comparison did not account for LightGBM replacing spaces with underscores. Its failed manifest and original source snapshot remain at `artifacts/prefix-selection-001/manifest.json`; no results were overwritten. The corrected completed attempt is nested at `artifacts/prefix-selection-001/attempt-002/` and took 19.22 seconds.

The completed attempt is rejected for promotion. It improves2021 enough to cross0.99 but lowers the frozen2022 pooled AUC and leaves that fold below0.99. Although2022 log loss improves from0.127497748 to0.124706469, that is not the competition objective. Unit-level tradeoffs include improved2022 Lio AUC (0.960947716→0.965525063) alongside lower Haukeli (0.962089129→0.956900650) and Kjela (0.990749881→0.988660956). These2022 diagnostics do not alter any tree count or select a second candidate.

Post-run verification checked every recorded artifact SHA256, identical case/unit/label alignment against each base fold, finite bounded `(61320,14)` predictions, exactly reproduced both recorded AUCs, exact assembly of selected2021 counts from cached prefixes, and exact equality of every full350-tree reconstructed prediction to the saved base prediction. `python -m py_compile scripts/prefix_selection.py` also passed. The final script matches the completed attempt's source snapshot. No main-repository transfer or Kaggle action was performed.
