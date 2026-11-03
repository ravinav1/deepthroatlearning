# Provided-only canonical115 capacity ablation

The fixed63-leaf variant passed both reused chronological development gates by small margins. Its completed Kaggle submission scored **0.99650** publicly and does not replace the measured provided-only0.99653 best.

| Fold | Original canonical115 blend | New fixed.1 blend | Gain |
| --- | --- | --- | --- |
| 2021 | 0.9952596524075921 | 0.9952676572007169 | 0.0000080047931248 |
| 2022 | 0.9940509264719491 | 0.9940771953548430 | 0.0000262688828939 |

Only the pooled model's leaf limit changed:31→63. All115 features, original40% hour samples, case/hour/unit ordering and categorical columns111/112 were retained. Other parameters:1400 trees,learning rate.06,min-child100,min-child-weight.001,L2=1,L1=0,colsample.9,subsample1,max_bin255,deterministic=true,force_col_wise=true,seed4173,four threads,no early stopping. Original canonical115 cached arrays and labels were verified against the immutable source manifests before fitting. No external numerical data, downloaded solver outputs, pseudo-labels or teachers enter this recipe.

The2021 fixed weight grid `[0,.1,.25,.5,1]` selected.1 against the accepted31-leaf canonical115 ensemble;2022 used only0 and that same.1. Independent saved-array/selection audits passed in `artifacts/provided-only-capacity-2021-audit-001` and `artifacts/provided-only-capacity-2022-audit-001`. Source,input,split,purge,exact sample,model configuration and all scores remain in `artifacts/metric-trick-provided-capacity-{2021,2022}-001`. The2021 lock SHA-256 is `2eb9a85d8d5b20d1d7138743f8aabed4f3bbf702afaddc8c5e53dd7a2d7f00b9`. The original supplied-only lineage proof is `artifacts/provided-only-lineage-audit-002/manifest.json`, SHA-256 `6c017919db2e6879e718398bb0aec7fc59f79b73b68a439a73a9750885b61f92`.

Final export completed in913.893s with2223144960-byte peakRSS; process exited0. It fits196358 original training hours×14units =2749012 rows from all2922 supplied training cases. All1705200 flattened test predictions were replayed bitwise from the saved checkpoint before export. Recipe: `project_vinje(0.9 * saved_clean_canonical115_blend + 0.1 * raw_63leaf_canonical115)` in float64, preserving all nested original anchor projections.

- Final directory: `artifacts/metric-trick-provided-capacity-final-001`.
- Final manifest SHA-256: `ec9478639df76a6b8a179382388471b18d914f065db609b79d9748325d1f4c10`.
- Checkpoint `canonical115_63leaf_1400.txt`: `85d40adce926f8b1ff074b51033d7497c5c0afc5f090d486bdcacfd96784de51`.
- `raw_predictions.npy`: `31f5b02e3e484c9628975bc3504afb6a76473f949368d1ff2c83b6927b1c0e16`.
- `test_predictions.npy`: `677f2aaec28b65e581b823a9206b9d1b574576e4588b124067b7fd2e16012285`.
- `submission.csv`: `efc9166cab6a5809625eabe65b958a0f13b8630c8a0a6a2315be9af434128f8d`.

The saved clean anchor is `artifacts/physics-trick-canonical115-final-001/test_predictions.npy`, SHA-256 `e41c498dce0a5910509a290c78ff5a0a47357601e13c90ccce7b770635a324f8`. Its complete data-to-model ancestry and source/data hashes are documented in `docs/best_provided_only_result.md` and the lineage audit. Submission schema remains725×2353 with original IDs/order and probabilities in[0,1], serialized to nine significant digits. The independent actual-final audit passed in `artifacts/provided-only-capacity-final-audit-001/manifest.json`, SHA-256 `028c006fca92df7d3630df404c41c940bb651e8dff46ab2b9e60ecb9f01177b7`. It reconstructed the full selected array and actual CSV bytes, checked original IDs/schema/range/readback and both prior lock audits, and independently replayed98 checkpoint rows. It did not rebuild all features or repeat the complete checkpoint replay. Parent verified Complete/public0.99650 at2026-11-03T04:16:30Z; submitted timestamp04:15:15Z. The separate `kaggle_observation.json` records this result without altering the final manifest. This is below the measured supplied-only best0.99653.

Exact final fitting/inference/export command (use a fresh run ID to reproduce):

```sh
.venv-ml/bin/python scripts/metric_trick_provided_capacity.py --year final --run-id metric-trick-provided-capacity-final-001 --lineage-audit artifacts/provided-only-lineage-audit-002/manifest.json --lineage-audit-sha256 6c017919db2e6879e718398bb0aec7fc59f79b73b68a439a73a9750885b61f92 --selection-lock /Users/heap/programming/tdt4173/artifacts/metric-trick-provided-capacity-2022-001/selection_2022.json
```

Runtime environment is macOS27.0.1 arm64,Python3.11.15,NumPy2.4.6,pandas3.0.6,LightGBM4.7.0,scikit-learn1.9.1 and12 logical CPUs. The final fit overlapped separately authorized provided-only experiments; it did not change its frozen statistical recipe. Further fits from this branch are paused pending parent prioritization, with all accepted locks retained.
