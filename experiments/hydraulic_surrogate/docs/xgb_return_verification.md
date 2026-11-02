# Returned CUDA XGBoost verification

The2021 GPU fold completed and passed independent local verification. Raw micro AUC is **0.9945072625313349**. The frozen scalar grid `[0, .1, .25, .5]` against the already projected earlier joint anchor gives:

| XGBoost weight | 2021 micro AUC |
| --- | ---: |
|0|0.9948884917510937|
|.1|0.9949209266933028|
|.25|0.9949355929850593|
|.5|0.9948926897265898|

The selected weight is **.25**. Its gain over the declared anchor is 0.0000471012339656. The separately locked 2022 evaluation also improved its declared anchor, so the original395 GPU branch was accepted. These are reused chronological development folds; no new public Kaggle result is established.

Immutable local lock: `artifacts/cuda-xgb-select-2021-001/selection_2021.json`, SHA256 `315c420fa43a6a046fe97facafd473069de1d6761ad5d0f372617893bf3db040`. Its exact bytes, compressed for the notebook UI, are embedded in `artifacts/cuda-xgb-lock-transfer-2021-001/transfer_lock_cell.py`; that cell writes `/kaggle/working/cuda-xgb-selection-2021-001.json` exclusively or checks an existing identical file, then verifies the SHA. Root alone owns browser transfer and2022launch.

## Provenance

The downloaded ZIP `artifacts/cuda-xgb-2021-001-download.zip` has SHA256 `4f8109ef9062feb41350dfd9874d6b6d93c95182345306f8438c173d65bae1d8`, matching the directly observed hosted hash. The downloaded2021 manifest is `a20279e00a5fbdff4c8c6f9c096f58440fbb259ab950b4d85b22bd3f103a5d2a`; original successful smoke manifest is `f31b812df3cdea770b5f04861608d884faa8bbcaccebbe078a6e95c6736e7e4f`. The complete extracted data/checkpoints/configs/log/smoke are under `artifacts/verify-cuda-xgb-2021-return-001/download/`. Original downloaded/verified manifests remain unchanged.

The frozen protocol7317e… and selector2182b0… were not modified. A separate verifier wrapper checks the exact archive hash and safe allowlist, hosted manifest/smoke identity, all14checkpoint/configs, native cuda:1/GPU histogram updater,600trees,395features, all declared hyperparameters, exact purged IDs/dates and40%sample bytes, feature formulas/source/cache, bundle lineage, raw probability bounds/schema, and independent micro/per-unit AUC. It verifies hosted checkpoint-replay reports for all61320validation rows per model. No local XGBoost model was loaded and no cross-platform inference equivalence is claimed.

Only after all verification did it import an exact selector snapshot with its pinned source dependencies. The immutable wrapper is `artifacts/xgb_return_preflight_20261102T144214Z_d148bf91/verify_cuda_xgb_return.py`, SHA256 `4bae40e926be270a750b57eec39b35642315110c500780be6b427727a6bc8de4`. Independent combined_export review approved these exact bytes before execution. Three synthetic transfer/native-config test groups passed0.007s. Original selectors remain untouched. Full source/environment/commands/hashes: `artifacts/verify-cuda-xgb-2021-return-001/manifest.json` and `artifacts/cuda-xgb-select-2021-001/manifest.json`.

## 2022 evaluation

The original compressed lock transport failed without creating a lock file. Root then reconstructed the exact JSON and verified SHA315c before launching 2022. The 600-tree/depth6/.05/two-thread/cuda:1 recipe and original feature/source bundle were preserved. Hosted 2022 completed in 192.341470 seconds; raw GPU micro AUC is 0.9934912458703047. Only zero and the already selected .25 weight were evaluated locally.

| Recipe | 2021 micro AUC | 2022 micro AUC |
| --- | ---: | ---: |
| Declared projected joint anchor | 0.9948884917510937 | 0.9936419640150564 |
| Locked25% original395 GPU blend | 0.9949355929850593 | 0.9937259135724086 |
| Newer608-feature current blend, descriptive only | 0.9949377186691472 | 0.9936914879536517 |

The fixed GPU blend improves both declared anchor folds. Relative to the newer608 candidate it is lower by 0.0000021256840879 in 2021 and higher by 0.0000344256187569 in 2022; neither dominates both. No anchor substitution or retuning was performed. This comparison is recorded in `artifacts/compare-locked-cuda-xgb-current608-001/manifest.json`.

The 2022 archive SHA is `ae2b12a3f0a8cfc9cd11acfebbcc2153760a9757c8cfecf9adcbe168c7234e4f`; its hosted manifest SHA is `e8077c744039ecf3b1d702e86ea670dddbfd79b8e93f3fba0f33b0d61d5562df`. The original successful smoke is unchanged. The exact approved verifier completed all artifact, model, input scope, raw-array, AUC and replay checks before selection. Evidence is preserved in `artifacts/verify-cuda-xgb-2022-return-001/` and `artifacts/cuda-xgb-select-2022-001/`. Both years remain below .998, so main-repository transfer is not permitted.

The accepted final-fit recipe is being prepared separately in `docs/cuda_xgb_final.md`. Original validation runners, protocols and verified manifests remain immutable.
