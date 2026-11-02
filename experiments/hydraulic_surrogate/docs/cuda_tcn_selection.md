# Local verification and frozen CUDA TCN blend

The fixed scalar grid is `w ∈ [0, .1, .25, .5]`, declared before returned neural OOF predictions exist. The anchor is the accepted `lp-joint-comparison-001` OOF array, identified by its immutable analysis and prediction hashes in `docs/cuda_tcn_protocol.json`. The blend is:

`project_vinje((1-w) * accepted_joint_anchor_already_projected + w * raw_TCN)`

Arithmetic is float64, across the complete 14-unit matrix. There is no smoothing or per-unit weight selection. The anchor already includes its original projection; this nested composition must be preserved in any future final exporter. It is not interchangeable with flattening all underlying raw component weights and projecting once. Weight zero reproduces the accepted anchor exactly.

Run the local selector only after downloading the complete requested remote run directory unchanged:

```sh
.venv-ml/bin/python scripts/select_cuda_tcn_blend.py --year 2021 \
  --gpu-run artifacts/downloads/cuda-tcn-2021-001 \
  --run-id cuda-tcn-selection-2021-001
```

The script requires `manifest.json` and precisely the seven hashed runner artifacts: trainer source, protocol, float64 normalizer, 24 epoch-case orders, checkpoint, OOF NPZ and fresh-model replay NPY. It checks frozen trainer/protocol/bundle/baseline lineage, legal purged case IDs, unit order, actual CUDA settings, all 24 scheduled learning rates and every training-case permutation. Only the absolute LP cache path is removed when comparing portable feature specifications. Original local data checksums are verified without constructing a feature table. Checkpoints are hashed but never deserialized locally. The saved fresh-model replay must match the original OOF probabilities byte for byte; this is evidence recorded by the hosted trainer plus an independent comparison of the downloaded arrays, not another local neural inference run.

The saved statistics must use the exact training IDs and feature order, finite float64 means/scales and no heldout cases. They are not numerically recomputed here. Cross-platform Mac-versus-Linux feature bitwise equality is not claimed. Raw pooled and per-unit AUC are recomputed locally with a 2e-12 tolerance for scorer version differences. Prediction/target shapes, IDs, labels and unit order must agree exactly with the local anchor.

Only 2021 predictions are decoded for selection. Ties favor the smaller neural weight. The resulting `selection_2021.json` includes the exact `cuda_tcn_2021_blend_lock_v1` contract expected by the frozen hosted trainer, 2021 model-manifest/OOF hashes, protocol/anchor hashes, fixed grid, selected weight, formula and scores. Root must review that file and its SHA before explicitly invoking the separate hosted 2022 command. There is no automatic remote training here.

After that separate 2022 run finishes and is downloaded:

```sh
.venv-ml/bin/python scripts/select_cuda_tcn_blend.py --year 2022 \
  --gpu-run artifacts/downloads/cuda-tcn-2022-001 \
  --run-id cuda-tcn-evaluation-2022-001 \
  --selection-lock artifacts/cuda-tcn-selection-2021-001/selection_2021.json \
  --selection-lock-sha256 REVIEWED_SHA256
```

The 2022 manifest must contain the same lock and checksum. Only the anchor and previously selected weight are scored. Promotion requires a positive neural weight and strict improvement on both 2021 and 2022; otherwise the prior accepted joint recipe remains accepted. No other weight is selected using 2022. The transfer gate remains AUC ≥.998 on both years. These years have been repeatedly reused for development and are not an untouched holdout. Kaggle public scores never select this weight.

Each invocation creates a unique artifact directory and appends success/failure, environment, command, source snapshots, hashes, results and runtime to `docs/experiments.md`. Existing runs are never overwritten. The code caps numerical thread pools at two and runs no fit or feature rebuild.

Validation before returned GPU predictions: seven small synthetic tests passed, covering a valid portable-cache fixture, checkpoint tampering before numeric decoding, mismatched replay despite updated hashes, heldout normalizer scope, missing/duplicate epoch cases, changed feature formulas and evaluation of only requested weights. These tests do not establish GPU execution or neural quality.
