# GPU execution queue

Checkpoint: 2026-11-02 22:16 UTC public observations. Browser access is restored. Root owns the authenticated UI, uploads, saved launches, downloads and submissions. At most two saved GPU VMs may run concurrently. Prepared notebooks are not launch evidence; record each actual saved version and runtime separately.

Public best is the LGB664–SHOP blend at **0.99652, rank18**, observed22:04:37UTC in `artifacts/submission-lgb-shop664-anchor-blend-001/kaggle_observation.json`, SHA `9a74feb9be25f21a45ca92af75451d9206ef62c7924840d0660c8acc8b334f95`. The exact recipe remains in `docs/best_result.md`. Existing frozen experiment anchors are unchanged.

## Completed; do not rerun or resubmit

Private savedv10/scriptVersionId364994788 completed all14 fullsample608 final models in667.537s. Its archive, models, primary and predeclared standalone exports passed strict verification and independent actualCSV audit `f199cf2df3f035a3f6f6240820c10df23a99257ef5e70e4d2a8e408928c284e4`.

- Primary: `artifacts/submission-cuda-xgb608-fullsample-anchor-blend-001/submission.csv`, SHA `621c8df2470c9962e2e45441e138263cac67fe7fe9a384b1bf6e96803b96c20d`; actual public **0.99643**, observed22:14:09UTC. Observation SHA `8c33c3422576d885cccb0babf1f7b8de42e554576b82327cb9c02e674878d5bf`.
- Standalone: `artifacts/submission-cuda-xgb608-fullsample-standalone-001/submission_standalone_projected.csv`, SHA `bbe1c25b28f5c1be129ae7303f85669e8803e873df3e977e9b23a6d690ec7284`; actual public **0.99631**, observed22:16:37UTC. Observation SHA `26482ded6e84aaf367e50b58dff4726a16d91121ee5aeefe3a6a929f8b01a4cf`. Root downloaded the actual submittedCSV to `artifacts/kaggle-uploaded-fullsample608-standalone-001.csv`; it matches the export exactly. Earlier handoff text incorrectly named `submission.csv`, which does not exist in the standalone directory.

Both submissions are Complete and below the current best. Submission IDs were not exposed in the observed UI. No final-selection checkbox changed. These results do not reopen the separately rejected fullsample608-versus-currentSHOP combination.

## Next saved GPU launches

1. **SHOP664 fullsample2021.** Import `artifacts/cuda-fullsample-parallel-notebooks-001/tdt4173-shop664-fullsample-2021.ipynb`, SHA `19f3740ad7f9e73c818057f7bc822f2a3d9b182d0a7fd871c4f0a9ef8c805d17`. PRIVATE code dataset already attached: https://www.kaggle.com/datasets/ravinav/tdt4173-private-fullsample-continuation-code . Keep original base,608 supplement andSHOP56 supplement. Use pinned Python3.13.15/T4×2/driver580.178.04. Fresh `cuda-xgb664-fullsample-smoke-001`, then only2021/all367248 purged rows. Output `cuda-xgb664-fullsample-2021-001-download.zip`. No actual launch yet observed at this checkpoint. Exact command/protocol pins: `docs/cuda_fullsample_parallel_execution.md`.
2. **Additive745 on the second VM.** Upload PRIVATE `artifacts/cuda-xgb745-package-002/cuda-xgb745-private-dataset.zip`,54,218,767bytes, SHA `c01101fa9a6756fb78ec5f70c16b47e21a95ea22711a0fb66ab8f26a142a9894`. Retain the original three bundles; this dataset adds the disjoint LPv2 supplement and745code. Import `artifacts/cuda-xgb745-2021-notebook-001/tdt4173-xgb745-2021.ipynb`, SHA `7c4854fb500471436f8b27c7fcd82e8ae45c7564f39d8ae1a3b9958e1e3cf5c5`. Fresh `cuda-xgb745-smoke-001`, then only2021/original600/depth6/.05/40%. Output `cuda-xgb745-2021-001-download.zip`. Runtime, approvals and return command: `docs/cuda_xgb745_execution.md`.

Both packages were byte-rechecked after browser recovery: `artifacts/gpu-shop-fullsample745-launch-recheck-001/manifest.json`, SHA `6588cb6203134628c4b7c3b8c5516cd4360f84ec2c21b2e8dce04980989582a5`. Notebook/plaintext/protocol/ZIP/verifier/selector bytes are exact. Both retain CUDA1 and require separate VMs. Root records externally observed archive size/SHA, fold manifest SHA and fresh smoke SHA before xgb imports. Each2021selection requires independent lock audit before2022.

The bounded synthetic CUDA vector14 smoke remains behind these model experiments: `artifacts/cuda-vector14-smoke-notebook-001/tdt4173-vector14-synthetic-smoke.ipynb`, SHA `8acac6006529c51b36db2ac7d5b74ba9b1beb70bd5123c728b998077c7ab85f9`; source-only review `971cdfb18711dd69b037b79bb57834848d49c8a8891e038ee868282b4f86a006`. It uses2048synthetic rows/8shared trees, requires no dataset and establishes no full-fold eligibility. No actual CUDA vector capability result exists. The separate CPU vector preparation has no authorized full fit.
