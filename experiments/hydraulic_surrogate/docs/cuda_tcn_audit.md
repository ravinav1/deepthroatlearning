# Independent CUDA TCN review

The reviewed runner and notebooks pass static review and nine small-fixture groups. They are ready for the already-authorized hosted synthetic CUDA smoke. This audit performs no Torch import, GPU execution, feature-matrix construction or model fit. Actual CUDA compatibility, timing, memory and checkpoint replay remain runtime measurements; a successful smoke is mandatory before real fitting.

Reviewed files:

| File | SHA256 |
| --- | --- |
| `scripts/train_cuda_tcn.py` | `1afae39884da9d210a8ef695fb0c99636536bc688ce79961d502bf4c4b743e57` |
| `docs/cuda_tcn_protocol.json` | `d55bee5f39fc50eb5dc9ed48b99f244b7cad75aff1c9bc91c10c3a29de8cdd88` |
| `notebooks/cuda_tcn_smoke.ipynb` | `aefc237b0fab18c9a03145949bfa89866eb77b34647adecdca201e8a5533a63f` |
| `notebooks/cuda_tcn_review.ipynb` | `859594ff4b2a487157596ce5edff1036098fbfb7a8c894500f152da01f59d3c5` |

`scripts/audit_cuda_tcn.py --run-id cuda-tcn-audit-002` completed its fixtures in 0.052 seconds. Exact reviewed source bytes, audit-source hash, protocol/notebook hashes and results are in `artifacts/cuda-tcn-audit-002/manifest.json`. The earlier preliminary review is preserved as `cuda-tcn-audit-001`; it predates the replay and uv-launcher additions. No implementation file owned by the environment agent was edited by this audit.

## Information boundaries

The frozen feature builder normally decodes all commitment columns while separating labels from predictors. The portable wrapper restricts that full-target read to a chronological prefix through the requested case year and supplies zero sentinels for later rows. Metadata-only reads remain permitted. A fixture containing an invalid 2022 target confirmed that the 2021 path requests only all-row metadata plus a two-row permitted prefix and never decodes the future target. This concerns case-year targets: a 2021 validation case still legitimately has its own full 168-hour output horizon.

For 2021, training uses IDs 1–2186, ending 2020-12-25; validation uses 2193–2557. For 2022, training uses IDs 1–2551, ending 2021-12-25; validation uses 2558–2922. The code checks exact baseline IDs and verifies the last training hour precedes the validation year. Training examples remain complete cases; adjacent cases are never concatenated into the temporal axis.

Feature means and population standard deviations use only retained training-case hours and are accumulated in float64. Heldout feature poisoning leaves these statistics unchanged in the independent fixture. Small standard deviations become one; transformed inputs are clipped to [-10,10], converted to float32 and arranged case/channel/hour. Commitment labels do not enter these statistics or the input features.

Validation labels for the selected case year are resident in memory, but the training loop indexes only retained training cases. It has exactly 24 epochs, a fixed learning-rate schedule and no validation evaluation, early stopping or checkpoint selection inside the loop. Only the final epoch is saved/evaluated. The 2022 command requires a separately supplied 2021 selection lock and its explicit SHA256 before feature building. Wrong-year, wrong-protocol, out-of-grid weight and incomplete-2021 locks fail the fixture. Anchor arrays remain local and never enter neural inputs.

## Frozen features and staging

The approved bundle manifest and archive hashes are pinned in the new protocol. Staging requires its exact allowlist, canonical relative paths, no symlinks, expected sizes and every file hash before importing staged feature code. Exact existing stages may be reused only after the same file-list/byte checks. Bytecode writes are disabled. Fixtures cover a valid round trip, exact reuse, altered content, unlisted files and traversal attempts; none writes outside the designated temporary stage.

The runner imports only the staged feature modules and rejects an already imported foreign `sintef_project` implementation. The original strict LP loader still verifies source/data hashes and the feature archive. All 395 feature names and their order must match the protocol. The complete reconstructed feature specification is compared to the baseline after removing only `lp.cache_directory` from both copies. That is the sole absolute path in this baseline specification; formulas, mathematics, column schema, raw hashes, source closure and LP manifest/feature digests remain compared.

This proves source/input/formula identity, not bitwise equality of recomputed Mac and hosted numerical features. Python, numpy, pandas, scipy, sklearn, PyYAML, Torch and CUDA environment versions are recorded. Different platforms can have numerical differences, so cross-platform equality is not claimed without a numerical comparison.

## Model, CUDA and replay

The architecture matches the declared 395-to-64 projection, four residual blocks, two width-three convolutions per block at dilations 1,2,4,8, per-hour channel LayerNorm, ReLU/dropout .1 and 14-output head. Arithmetic gives 126,094 parameters and a 61-hour convolutional receptive field. Existing input features also summarize the known full horizon; centered convolutions therefore do not imply forbidden future-label access. Padding stays inside each case and normalization mixes no cases.

The smoke checks actual CUDA availability/build, output device and shape, parameter count, finite backward/update results, synchronized runtime and GPU memory. There is no CPU model fallback. Deterministic algorithms, CuDNN flags, TF32 settings and `CUBLAS_WORKSPACE_CONFIG` are set explicitly, with the environment variable set before importing Torch. Current PyTorch documentation describes deterministic CUDA convolution/replication-padding behavior, but the installed hosted version must still pass the smoke. Reproducibility claims apply to the measured software/hardware, not arbitrary platforms. [PyTorch deterministic algorithms](https://docs.pytorch.org/docs/2.15/generated/torch.use_deterministic_algorithms.html), [PyTorch reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html)

After a real fit, the runner reloads its own hashed checkpoint with `weights_only=True` into a fresh model. It checks protocol/trainer/bundle hashes, ordered features/units, epoch and seed, then reloads saved float64 normalization statistics and their training IDs. In evaluation/inference mode, it reproduces OOF batches and requires byte-identical probabilities. It saves the replay output and checksum. This is a sound replay path by inspection; the audit does not claim it has run successfully on CUDA.

The checkpoint includes model, optimizer, scheduler and numpy/Torch RNG states. Fold output includes exact sampled case orders, fixed-epoch losses/rates, normalizer, raw aligned OOF probabilities/labels/IDs/units, source/protocol snapshots and all artifact hashes. No smoothing or projection is applied remotely; the frozen local blend workflow handles that separately.

## Notebook and private-runtime controls

Both notebooks embed trainer/protocol bytes that exactly match the reviewed files; all cells compile and have no saved outputs or execution counts. The smoke notebook has no real-fit cells. The review notebook initializes both real-fit controls to false, uses separate year-specific stage roots and requires the external lock for 2022. A normal Run All therefore performs only metadata inspection and synthetic smoke.

The launcher requires `uv`, creates an offline `--system-site-packages` environment using the hosted interpreter, and performs no package installation or download. If uv is absent it reports the condition and stops. The runner independently verifies the environment configuration and package versions. No rejected local package-download attempt is retried.

The notebook discovers exactly one input mount by the pinned bundle-manifest hash. It cannot enforce Kaggle's privacy setting through notebook metadata. The root coordinator owns the already-approved private upload/session actions and must retain the verified private dataset and notebook settings when importing the reviewed file. No UI or remote action was performed by this audit.
