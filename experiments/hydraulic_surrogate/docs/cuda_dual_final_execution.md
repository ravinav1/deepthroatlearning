# Accepted dual final execution

## Completed export and independent review

All four CSVs passed actual array/ID/schema/range/byte verification: `artifacts/cuda-dual-final-submissions-independent-audit-001/manifest.json`, SHA256 `ff88843e5eb8aa1d351ceb36b0640afb08dfc82d2490ecba294e3dacf272aa7d`.

The actual successful capacity primary directory is **`artifacts/submission-cuda-xgb608-capacity-anchor-blend-002`**. Attempt001 is preserved as a CLI hash transcription failure; its unchanged exporter succeeded with the correctly recorded hosted pins under002. The command examples below preserve the preparation-time IDs; use the actual successful export manifest for current artifacts and fresh IDs for any reproduction.

| Candidate | CSV path | SHA256 |
|---|---|---|
| SHOP primary | `artifacts/submission-cuda-xgb664-anchor-blend-001/submission.csv` | `898d29699cda7aa93cc5b22e410d63eb7fa371ce4afc9f0e9b5a176993e9e96a` |
| SHOP standalone diagnostic | `artifacts/submission-cuda-xgb664-anchor-blend-001/submission_standalone_projected.csv` | `4c96e9f69459417fd138af7d33f5339275bb5ef29ffdd59f0a4590ee8f60ae42` |
| Capacity primary | `artifacts/submission-cuda-xgb608-capacity-anchor-blend-002/submission.csv` | `b7d42b0752ed0de639337b93ac57dc05498f478bea2f3217e86d16917a34cc47` |
| Capacity standalone diagnostic | `artifacts/submission-cuda-xgb608-capacity-standalone-001/submission_standalone_projected.csv` | `85ac35d8d5c8ba460e034b3f43b0d5bfc3c461906e97daa579a5a290c167acd7` |

Public observations are recorded separately by root. This export audit does not assert a new leaderboard score.

Both independent chronological confirmations passed the frozen0.25 weight. This execution does not retune either recipe. Root owns PRIVATE upload, saved notebook execution, download and public submissions.

The PRIVATE combined code archive is `artifacts/cuda-dual-final-package-001/cuda-dual-final-private-dataset.zip`,161393 bytes, SHA256 `e3d6878960c983c96e76daa7c3150b3dcff2bba61a75ec923b3f0f08ea3eecd3`. It preserves both independently reviewed child code directories exactly. Parent manifest SHA256 `585c6ae68a62e6f118c8f676f232b452a4821a8e8ab3d8f91bef6119acdfdbc2`; independent combined wrapper audit `artifacts/cuda-two-gpu-final-notebook-independent-audit-001/manifest.json`, SHA256 `992ffa1ec4e642b8cf6bbb11b9b6fbe3622a1f79be90546e6bf031473071fee8`.

Import `artifacts/cuda-two-gpu-final-notebook-001/tdt4173-accepted-dual-final.ipynb`, SHA256 `5ca58debfe2967ad8c92b8b6c8dba8b7fdc83272b65ba87477cdc4d7ec3bc37b`, into the existing PRIVATE notebook with original base input,608 supplement and SHOP56 supplement datasets attached. Use saved background Run All with original pinned two-T4 runtime and Files only persistence. Source `run_dual_final.py` SHA256 `67681a6464f00b16e091d0a88c39595abce89ed90b5ab54d4c54929fc8dfac59`.

Fresh smoke003 runs execute sequentially through the same offline environment. SHOP600/depth6/lr.05 uses GPU1 and664 features. Capacity1200/depth8/lr.03 uses GPU0 and608 features. Both use the identical original196358-row sorted40% final sample, seed4173, no early stopping. Capacity starts after SHOP's first final model and at least20 seconds, if effective host/cgroup available memory is at least12GiB; otherwise serial. Separate stage/output roots prevent collisions. Each of14 models must replay all121800 test rows bitwise. Each final result is independently archived and hashed even if the other job fails.

Expected hosted archives:

- `cuda-xgb664-final-001-download.zip`, containing `cuda-xgb664-final-001/` and `cuda-xgb664-smoke-003/` plus final log.
- `cuda-xgb608-capacity-final-001-download.zip`, containing `cuda-xgb608-capacity-final-001/` and `cuda-xgb608-capacity-smoke-003/` plus final log.

The following exact successful export commands are copied from completed immutable manifests, using independently observed hosted hashes. Existing run IDs are immutable; use a new output ID for a deliberate replay:

```sh
.venv-ml/bin/python scripts/export_cuda_xgb664_final.py --protocol docs/cuda_xgb664_final_protocol.json --protocol-sha256 e571c6607bd797f65c76db5d7617dec7b031774a41af44e65283ccf1ad964ab7 --archive artifacts/cuda-xgb664-final-001-download.zip --archive-sha256 ea8fdec6dad3baaba01b034e41ef12b8a296f98b939aa490784d9119c2be15e7 --manifest-sha256 c914fc7f3b2c739e162eea8e9809ee0399713a378e036d8a175bc58e3e9cdec3 --gpu-run-name cuda-xgb664-final-001 --smoke-run-name cuda-xgb664-smoke-003 --smoke-manifest-sha256 0d405c8f3dd3554053f0208b50ad60840f082119c467d2e3b40ff84f0a471aab --run-id submission-cuda-xgb664-anchor-blend-001

.venv-ml/bin/python scripts/export_cuda_xgb608_capacity_final.py --protocol docs/cuda_xgb608_capacity_final_protocol.json --protocol-sha256 840879d339732b14349d5355a886ef2e1b8e31101aa482ec8472df56a7be20d3 --archive artifacts/cuda-xgb608-capacity-final-001-download.zip --archive-sha256 f21f3016326f22c2d9b6878c619909821cce68ed4e3e6840f05ecb439385c616 --manifest-sha256 80c3586e474ac0e87054c06e73f3dfe965f9e9a9efd4faa4f050c8e9e9483972 --gpu-run-name cuda-xgb608-capacity-final-001 --smoke-run-name cuda-xgb608-capacity-smoke-003 --smoke-manifest-sha256 0d138cc7d7d1ae67b548e7cef768101ee4f0f97ca304492328c6b1baa741a279 --run-id submission-cuda-xgb608-capacity-anchor-blend-002
```

Each primary formula is `project_vinje(.75*saved_projected_acceptedGPU608anchor + .25*new_raw_GPU)`, in float64, preserving the intermediate anchor projection. SHOP's primary exporter also emits its predeclared standalone projected diagnostic. Capacity's separate frozen diagnostic follows strict primary verification:

```sh
.venv-ml/bin/python scripts/export_cuda_xgb608_capacity_standalone.py --protocol docs/cuda_xgb608_capacity_standalone_protocol.json --protocol-sha256 d76722de6506dbb98ab67daecac47bfb2a771267590768e428f0ddf7f8265033 --verified-blend-run artifacts/submission-cuda-xgb608-capacity-anchor-blend-002 --verified-blend-manifest-sha256 51758f020ddd0faa483909277da327a8f928f6b647d6741351b84f3130f3c4f0 --run-id submission-cuda-xgb608-capacity-standalone-001
```

The two standalone diagnostics use the same final checkpoints and one fixed Vinje projection, no new training or public-score-driven weight search. They have weaker chronological validation than the corresponding primary blends and no automatic best-result promotion. Both CSVs need independent actual-array/ID/schema/range/byte audits before parent submits. Preserve observations separately from immutable export manifests. The SHOP primary was subsequently submitted and independently observed at public AUC0.99651, rank18. Observation and other submission results are maintained separately in docs/kaggle_scores.json and docs/best_result.md.


# Reproduce the public0.99651 SHOP primary

The exact submitted file is `artifacts/submission-cuda-xgb664-anchor-blend-001/submission.csv`, SHA256 `898d29699cda7aa93cc5b22e410d63eb7fa371ce4afc9f0e9b5a176993e9e96a`. Strict export manifest SHA256 `dcd1617cb60aa29dce9c9a64bff16fbc25f00ef569912ea1b80c21eea5aa973c`; independent actual-four-CSV proof SHA256 `ff88843e5eb8aa1d351ceb36b0640afb08dfc82d2490ecba294e3dacf272aa7d`. Public score0.99651/rank18 is a parent-observed Kaggle result; chronological2021/2022 validation is separately .9951018232145258/.9938546734526688. The .998/both-validation-years transfer threshold and top-three goal remain unmet.

Let `P` denote the fixed float64 Vinje isotonic projection, applied to each case/hour while preserving all14 unit columns. Preserve all three projection stages exactly:

```text
L = P(.140625*LGB_base + .421875*LGB_context + .1875*LGB_joint + .25*LGB_context_joint)
A = P(.5*L + .5*raw_GPU608)
submission = P(.75*A + .25*raw_GPU664)
```

The four original LightGBM components use14 unit-specific350-tree binary models with31 leaves, learning_rate.06, min_child_samples100, reg_lambda1, colsample_bytree.9, max_bin255, seed4173 and no early stopping. Their exact parameters, data/source/cache hashes and saved checkpoint paths are recorded in each run manifest. Their raw array locations and inner weights are:

| Raw final run | Features | Inner weight | Raw array SHA256 |
|---|---:|---:|---|
| `artifacts/lgb-normalized-lp-final-001` | 395 | 0.140625 | `92d2173a18bdad5ef9847755aae1bc279141c64c6279c2a424a1af8f0ca149c3` |
| `artifacts/lgb-normalized-lp-context-final-001` | 521 | 0.421875 | `2867660286ab1a241f35afe98255d674c096c05bdc94231a1e552151b042fb79` |
| `artifacts/lgb-normalized-lp-joint-final-001` | 404 | 0.1875 | `f97dc6b63aa52bd37133e006bd7efa8901a1039863118bb156f1e6f0602e4d21` |
| `artifacts/lgb-normalized-lp-context-joint-final-001` | 608 | 0.25 | `0d334556a8828b37ceb930a76b3d38b4c46ee7c26a525b4faa70dcfc32afc479` |

Each row's raw array is `test_predictions.npy`, with checkpoints `final_<unit>.txt`. The sealed inner blend is `artifacts/submission-lp-context-joint-blend-001/test_predictions.npy`, SHA256 `dd0bc40c2ca7a4fe84b57430b08f31a227f66ff4f8004f4f86b7284da8dbd580`; its manifest/source order and full checkpoint closure are verified by `artifacts/verify-lp-context-joint-export-001/manifest.json` (SHA256 `3a6686b5be0157cdae94e0cc8370c39d27ff51dcd07798f3eb130944ecab274d`). The original inner export command is retained in that blend manifest.

`raw_GPU608` is the float32 `predictions` array in `artifacts/submission-cuda-xgb608-context-joint-blend-001/download/cuda-xgb608-final-003/test_predictions.npz`; its sibling `replay_test_predictions.npy` must match bitwise. Original final protocol `docs/cuda_xgb608_final_protocol.json`, SHA256 `a5939edb8ec7f46dbdb4c6edd0c06909ea09ed0ac029709b06945a5c6434683b`. The resulting accepted anchor `A` is `artifacts/submission-cuda-xgb608-context-joint-blend-001/test_predictions.npy`; anchor export manifest SHA256 `6d13425abe03f5aa7ffe36baffd9b6c06e197f28a4b1afec90c79a8313053555`, independent proof SHA256 `4d8bce81e8748dd274968a0a53bf6d5fc274e60b056451369eefd3c41210f19d`. The current SHOP exporter reconstructs every inner source and this anchor exactly before using it.

`raw_GPU664` is `predictions` in `artifacts/submission-cuda-xgb664-anchor-blend-001/download/cuda-xgb664-final-001/test_predictions.npz`, accompanied by its bitwise replay and14 saved `final_<unit>.ubj` checkpoints plus native configuration JSON files. Both GPU recipes use original600/depth6/learning_rate.05 XGBoost, binary logistic objective, min_child_weight20, reg_lambda4, max_bin128, per-tree subsample.9 and colsample_bytree.9, seed4173, no early stopping. SHOP adds the frozen56 input-only linked-storage/PQ physical features to the exact608-feature prefix; formulas, source/data/cache/bundle hashes, full native parameters and column order are pinned in `docs/cuda_xgb664_protocol.json` (SHA256 `4e09f0fe59973d7a9c2c5e5792feb30c29b8af04a67c27bafbe1386ae9da8182`). Final protocol `docs/cuda_xgb664_final_protocol.json`, SHA256 `e571c6607bd797f65c76db5d7617dec7b031774a41af44e65283ccf1ad964ab7`.

All final models use2922 historical training cases through2022, the original seed4173 sorted40% sample of490896 case-hour rows (196358 rows, NPY SHA256 `4e38d45e87950d4d84849224ddb64fa9b5bf7b21d20a7d9b998223ce5e0fa83d`), and predict725 fixed test cases×168 hours×14 units. Original labels/input files and prediction_mapping IDs remain unchanged and are hashed in the frozen manifests. Validation uses unchanged2021/2022 chronological partitions with six-day start-time purge; the .25 SHOP weight was selected from0/.1/.25/.5 on2021 before2022 and never tuned from public scores.

For hosted refitting, the reviewed combined notebook and exact child package/data bundles above implement the full final training command and check Python3.13.15, NumPy2.1.3, pandas2.3.3, XGBoost3.4.1, SciPy1.16.3, scikit-learn1.6.1, PyYAML6.0.3, uv0.13.5, two TeslaT4 GPUs and driver580.178.04. All14 SHOP models replayed every121800 test row bitwise. The actual saved run is private version8, scriptVersionId364985464. For export from saved predictions, use the exact SHOP strict export command above; it verifies saved model/replay evidence and reconstructs the complete nested anchor, then writes the original725×2353 ID/column order at `%.9g`, requiring [0,1] probabilities and CSV readback error<=5.1e-10.
