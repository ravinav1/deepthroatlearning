# Private608-feature GPU XGBoost ablation

This is a new feature ablation after the395-feature branch, with the exact same XGBoost hyperparameters: all14 units,600trees,depth6,learning_rate.05,min_child_weight20,L2=4,L1=0,max_bin128,subsample.9,colsample_bytree.9,seed4173,twoCPUthreads,cuda:1,noearlystopping. Fixed395 runner/protocol/bundle remain unchanged. No GPU run or upload occurs when building this package.

Inputs reproduce the exact521 normalizedLP temporal-context features then append the same87 joint-plant features as `lgb-normalized-lp-context-joint-001`. The portable loader checks complete ordered608 names and full feature specification, allowing only the two absolute cache-directory paths to differ. It uses the identical original40% sampled training rows with exactNPY SHA checks, six-day purge, full168-hour validation, and target rows after the requested year redacted. Cross-platform bitwise feature equality is not asserted without measurement. No feature matrix is built locally during package preparation.

Keep the existing private base dataset and `kaggle-neural-inputs.zip`. Upload the new code and supplement archives to a PRIVATE dataset only. The supplement contains exactly the missing52MB compressed joint cache, cache/source verification files and frozen608 reference manifest; it contains no anchor probabilities or additional raw labels. The original bundle is independently hash-checked and staged separately; the new merged stage rejects every overlapping path, so supplement files cannot replace an original input. All labels needed for training remain in the already authorized private base bundle.

The anchor is fixed to `lp-context-joint-comparison-001`, weights `[.140625,.421875,.1875,.25]` for base/context/joint/combined. Select a single global weight in `[0,.1,.25,.5]` on2021 only, against that exact anchor: float64`project_vinje((1-w)*anchor_projected+w*rawXGB608)`. Ties choose lower weight. Retain a nonzero candidate only with2021 improvement, save its exclusive lock, then allow a separate2022 command with exact lock SHA. The same weight must improve2022; no2022 grid search. Both years are reused development evidence. This does not change the original395 branch's older anchor. All checkpoints/configs/raw probabilities and failed runs are preserved, and every model must replay all validation rows exactly from its saved UBJ.

The package requires independent review before use. Root owns browser uploads, private-state verification and GPU scheduling. The commands below use observed paths supplied after upload; placeholders are intentionally not executable.

First run synthetic608 smoke only, through the existing offline uv launcher and preinstalled system CUDA packages:

```sh
python launch_cuda_xgb.py --runner train_cuda_xgb608.py -- --mode smoke --run-id cuda-xgb608-smoke-001 --protocol cuda_xgb608_protocol.json --shared-loader cuda_xgb608_loader.py
```

After successful smoke, explicitly run2021 only:

```sh
python launch_cuda_xgb.py --runner train_cuda_xgb608.py -- --mode fit --year 2021 --run-id cuda-xgb608-2021-001 --protocol cuda_xgb608_protocol.json --shared-loader cuda_xgb608_loader.py --bundle OBSERVED_PRIVATE_BASE/kaggle-neural-inputs.zip --supplement OBSERVED_PRIVATE608/cuda-xgb608-supplement.zip --smoke-manifest /kaggle/working/xgb608-runs/cuda-xgb608-smoke-001/manifest.json
```

Download complete2021 models/configs/OOF and original smoke. Verify external archive/manifest hashes and frozen native configuration, then run the independently reviewed local608 selector to create `selection_2021.json`. Only an improving nonzero lock permits2022 with a new run ID, `--year 2022 --selection-lock OBSERVED_LOCK_PATH --selection-lock-sha256 VERIFIED_LOCK_SHA`. No command runs both years, a final fit or a submission. No repository transfer until both validation years reach.998; leaderboard success still requires actual.999/top-three evidence.
