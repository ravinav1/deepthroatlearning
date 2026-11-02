# Proposed LP temporal convolution branch

The local MPS branch has not trained a real model. The isolated `.venv-mps` now contains PyTorch2.11.0 and numpy2.4.6 after a successful normal root-session uv install. Child-session probe003 reports MPS support built but no available MPS device; local GPU speed and memory remain **unmeasured**. Hosted CUDA work is a separate branch. Earlier failed dependency checks below remain preserved as historical evidence.

## Hardware and dependency findings

Read-only `system_profiler SPHardwareDataType SPDisplaysDataType -detailLevel mini` reports Apple M2 Max, 12 CPU cores, 38 GPU cores, 32 GB memory, and Metal support. Python reports macOS 27.0.1 arm64. These facts make MPS a plausible backend, but hardware compatibility alone does not establish that a particular PyTorch installation can execute this model.

The official [PyTorch MPS backend documentation](https://docs.pytorch.org/docs/2.15/notes/mps.html), consulted 2026-11-02, distinguishes `torch.backends.mps.is_built()` from `is_available()` and demonstrates moving tensors and models onto `mps`. That version's example names macOS 15+. Apple's [PyTorch setup page](https://developer.apple.com/metal/pytorch/) names PyTorch 2.11.0 with Apple silicon, macOS 14+, Python 3.10+, and Xcode command-line tools. The pages describe different release versions; this note does not infer a current wheel version from either page.

At the initial feasibility check, there was no installed `torch` in `.venv-ml`, no matching `torch-*.dist-info` in `/Users/heap/.cache/uv/archive-v0`, and no compatible torch wheel in the checked temporary wheel/cache directories. Nearby project `.venv*` site-packages searches also found none. The isolated environment was created with uv:

```sh
uv venv --offline --cache-dir /private/tmp/tdt4173-uv .venv-mps --python .venv-ml/bin/python
uv pip install --offline --cache-dir /private/tmp/tdt4173-uv --python .venv-mps/bin/python torch
```

Environment creation succeeded with Python 3.11.15. That initial install failed because `torch was not found in the cache`; no package was installed and the active ML environment was untouched. A later successful normal-network root-session install is recorded below; the earlier unpopulated environment was not treated as ready.

The attempted metadata command was:

```sh
curl -L --fail --silent --show-error https://pypi.org/pypi/torch/json -o /private/tmp/tdt4173-torch-pypi.json
```

Its ordinary sandbox invocation failed with `curl: (6) Could not resolve host: pypi.org`. The identical command requested with `sandbox_permissions="require_escalated"` and approved-prefix hint `["curl"]` was rejected before execution: `you cannot ask for escalated permissions` under the granular policy with `sandbox_approval: false`. No retry or alternative download route was attempted after this rejection. This is an environment restriction, not evidence that PyTorch or MPS is incompatible.

## Synthetic smoke and resource measurement

`scripts/smoke_mps_tcn.py` is ready to run when an authorized environment contains PyTorch. It loads only the LP-v1 manifest, never a real feature table or label array. It checks MPS explicitly, disables acceptance of CPU fallback, sets two CPU threads and one interop thread, and requests deterministic algorithms. Unsupported operations fail visibly instead of being silently rerouted.

```sh
.venv-mps/bin/python scripts/smoke_mps_tcn.py --run-id mps-tcn-smoke-002
```

Run `mps-tcn-smoke-001` was executed now and recorded `blocked` before any tensor allocation. Its source snapshot and manifest are in `artifacts/mps-tcn-smoke-001/`. Both `measured_gpu_runtime` and `measured_gpu_memory` are null. It performed no synthetic optimizer steps, no real fit, and no competition inference.

The planned measurement uses 16 complete synthetic 168-hour cases, 395 features, 14 outputs, two warmup updates, and ten timed AdamW updates. Each timed update includes host-to-device copy, forward/backward, clipping, optimizer step, and synchronization. It records every step duration, median/mean duration, and memory after each synchronized update. These are sampled allocations, not true peaks. The [official MPS API](https://docs.pytorch.org/docs/2.15/mps.html) provides `synchronize`, `current_allocated_memory`, `driver_allocated_memory`, and `recommended_max_memory` for these checks.

Only tensor-size arithmetic can currently be stated: one input batch is 4,247,040 bytes and its labels are 150,528 bytes in float32. The proposed network has 126,094 parameters, or 504,376 bytes of float32 weights. These exclude gradients, Adam state, intermediate activations, workspaces, allocator caching, and framework overhead; they are **not** an estimate of actual process/GPU memory. No seconds-per-epoch or full-fit runtime estimate is justified without the smoke. Once measured, `ceil(training_cases / 16) × median_step_time × 24` is only a compute extrapolation, excluding feature construction, loading, evaluation, warmup, and interference from other jobs.

## Predeclared model and data protocol

The exact 395 input column names, frozen feature source hashes, data hashes, purged case IDs, and proposed settings are saved in `docs/neural_tcn_protocol.json`. The source is `artifacts/lgb-normalized-lp-001/manifest.json`. Use legal normalized LP-v1 inputs only: the existing input-derived price, reservoir, horizon, physics, continuous LP discharge/storage/dual features. No supervised model output, true commitment, learned residual, validation label, or in-sample prediction enters the network. The existing complete known 168-hour input horizon permits centered convolutions and existing future-price context.

Use the same case split and six-day purge as the baseline: 2,186 training cases through 2020-12-25 for validation on 365 cases starting in 2021; 2,551 training cases through 2021-12-25 for validation on 365 cases starting in 2022. Match saved IDs exactly. Each training example is one complete case and retains all 168 hours. This differs from the tree baseline's 40% sampled hours; it is a new model comparison, not a strictly matched row-density ablation. Never concatenate adjacent cases along the temporal axis.

Compute one mean and population standard deviation for each feature using training-case hours only. Replace standard deviations below `1e-6` with 1; transform and clip to [-10, 10]; save immutable float64 statistics and feed float32 tensors. Compute no validation/test normalization statistics. Shuffling occurs at case level with seed 4173; keep all cases, including the final smaller batch. Use no data-loader workers initially.

The model implemented by `build_model` is a 1×1 projection from 395 channels to 64, then four residual blocks with dilations 1, 2, 4, 8. Each block has two centered width-three convolutions, replication padding confined to its case, per-hour channel LayerNorm, ReLU, and dropout 0.1; the residual sum has ReLU. A 1×1 head emits 14 logits for each hour. The convolutional receptive field is 61 hours, with both previous and future known inputs. LayerNorm uses channels at the current hour, never a batch or another case.

Fix 24 epochs, batch 16 cases, seed 4173, unweighted mean BCEWithLogitsLoss, AdamW learning rate 0.001, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.0001, `foreach=False`, and gradient norm clipping at 1. Use epoch-wise cosine decay over 24 epochs to 0.0001. Use float32 throughout, no AMP, no compilation, no pretrained weights, no early stopping, no validation-guided epoch selection, and no additional architecture search within this experiment. Save the final checkpoint and optimizer/scheduler/RNG state, ordered features, normalization, case IDs, and all code/environment fingerprints. Gate real fitting on a successful synthetic smoke, source review, and parent scheduling; no full fit is authorized by this note.

## Frozen blend and acceptance rule

The comparison anchor is the accepted `context75_base25` candidate from `lp-candidate-comparison-001`, using its already projected `validation_YYYY.npz` probabilities and fixed unit/ID order. Compare whole-matrix blends `(1 - w) * anchor + w * neural_probability` with `w` in `[0, 0.1, 0.25, 0.5]`, then apply the same fixed Vinje projection and no smoothing. These supervised probabilities are used only for the final evaluation blend, never as network inputs.

Fit the 2021 network first, choose the maximum 2021 pooled micro AUC, and break exact ties toward the smaller neural weight. Write an immutable selection manifest before the 2022 fit/evaluation reads labels. Fit the 2022 network with identical frozen settings on its own historical training split, evaluate the locked weight, and retain the prior accepted anchor if 2022 fails to improve; do not pick another weight using 2022. Record raw neural and blended per-unit/micro scores, log loss, source/checkpoint hashes, case IDs, and full probability arrays. Both chronological folds have already been reused in research, so this is not a fresh holdout estimate. No main-repository migration is allowed unless both years reach 0.998; the Kaggle objective remains 0.999 and verified top-three rank.

## Hosted-runtime availability check and installed CPU alternative

2026-11-02 approximately12:36UTC: root attempted to inspect Kaggle's notebook/GPU availability through the already authorized browser while local fits continued. Native Brave getAXState and fresh getApp both returned `cgWindowNotFound`; inventory still listed running Brave, and browser host remained unavailable. No remote notebook, upload, account setting or compute session was created. This is a UI availability limitation, not evidence that the account lacks GPU quota. Kaggle's official [notebook documentation](https://www.kaggle.com/docs/notebooks) and [GPU usage documentation](https://www.kaggle.com/docs/efficient-gpu-usage) were consulted; account-specific availability remains unverified.

An independent installed-library CPU MLP feasibility check succeeded (`artifacts/cpu-mlp-smoke-001/manifest.json`). It uses scikit-learn1.9.1, not PyTorch or MPS. Synthetic4096×395 inputs,14binary outputs, hidden128/64ReLU, batch512, three fixedAdam epochs, twoBLASthreads: measured fit0.04152s,59854float32parameters and14logisticoutputs. This synthetic timing does not establish real-data convergence or performance. No package installation or network workaround was involved. A separate fixed24epoch multilabel validation protocol is being prepared. The [official MLP documentation](https://scikit-learn.org/stable/modules/neural_networks_supervised.html) describes multilabel support and CPU-only execution.

## Local retry after the reported permission change

On2026-11-02 at14:06UTC, root reported a user-authorized full-access/network change and requested a new normal uv attempt, without any sandbox override. `mps-environment-002` attempted the exact pinned `torch==2.11.0` and `numpy==2.4.6` install from official PyPI in the existing `.venv-mps`. The child command environment still failed DNS resolution for pypi.org after three retries (8.087s, exit2). This does not show that either version is unavailable or incompatible. No alternate version or download route was selected, and no escalation was requested.

The isolated environment remains empty according to its own Python3.11.15 metadata inventory. `mps-tcn-smoke-002` recorded a missing-Torch preflight blocker before tensor allocation; runtime and memory remain unmeasured. Logs, exact arguments, versions and hashes are preserved in `artifacts/mps-environment-002/`; smoke source/manifest are preserved in `artifacts/mps-tcn-smoke-002/`. Blocked001 remains unchanged. Root was notified of the possible stale child-session network state so it can distinguish this observation from the newly authorized policy and from the separate successful hosted CUDA smoke.

## Root-session install succeeds; child MPS availability remains unproven

The root agent's normal command subsequently succeeded under the changed permission/network state:

```sh
uv pip install --python .venv-mps/bin/python --index-url https://pypi.org/simple torch==2.11.0 numpy==2.4.6
```

The official PyPI resolver installed the exact requested versions plus filelock4.2.6, fsspec2026.9.0, Jinja2 3.1.6, MarkupSafe3.0.4, mpmath1.3.0, networkx3.6.1, setuptools81.0.0, sympy1.14.0 and typing_extensions4.16.0. The target remains the isolated Python3.11.15 `.venv-mps`; `.venv-ml` was unchanged. Root log `artifacts/mps-environment-root-003.log` has SHA256 `469fb1114e80ab3021c665e194c6c7095b58d0530bc84ba71f8478b75b784150`. Exact command, inventory, uv/Python/OS versions and log metadata are in `artifacts/mps-environment-root-003/manifest.json`. The inventory observation time is recorded separately rather than invented as the installation start time.

The child then ran only `.venv-mps/bin/python scripts/smoke_mps_tcn.py --run-id mps-tcn-smoke-003`. PyTorch2.11.0 imported successfully and reported `mps_built=True`, `mps_available=False`. The script recorded blocked after5.105s, before any tensor allocation or synthetic optimizer step. No GPU runtime/memory, real-data training or CPU fallback is claimed. The differing root-versus-child network observations show that session propagation matters; this child MPS result alone does not prove that the root command environment cannot access Metal. Root was notified to distinguish the environments before further conclusions. Blocked001 and002 remain immutable.
