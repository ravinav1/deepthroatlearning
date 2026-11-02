# Two-GPU chronological execution notebook

Use `artifacts/cuda-two-gpu-2021-notebook-002/tdt4173-two-gpu-2021-cgroup.ipynb` as a clean PRIVATE saved notebook with two T4 GPUs. Attach the already reviewed original data, original608 supplement, SHOP664 package and capacity608 code package. Do not also launch the capacity-only alternative or preserved notebook001. The source is readable in the adjacent `run_two_gpu_2021.py`; all notebook outputs are empty.

The wrapper preserves both frozen trainers and protocols. It pins the verified final003 runtime (Python3.13.15, XGBoost3.4.1, NumPy2.1.3, pandas2.3.3, scipy1.16.3, sklearn1.6.1, PyYAML6.0.3, uv0.13.5, two T4 GPUs and driver580.178.04), with no environment installation or network fallback. Genuine smokes run sequentially to avoid a shared-venv creation race.

SHOP664 uses GPU1 and starts first. Capacity608 uses GPU0 and starts after the first completed664 model, at least20seconds and12GiB effective available memory. This avoids simultaneous feature construction. The independently measured664 feature-build peak was5.42GB. Runtime memory headroom is the minimum of host MemAvailable and the finite cgroup v2 or v1 allowance minus current usage. If telemetry is missing or memory insufficient, the second job waits for the first to finish.

Both fits run in separately monitored child processes while the notebook remains active and emits20second heartbeats. A failed smoke skips only its own experiment. Each fit, whether successful or failed, creates its own return ZIP as soon as it finishes; the companion continues. Successful output requires14completed models, fixed tree count, requested GPU, matching smoke/protocol and exact checkpoint replay. Each archive contains only its smoke/fit namespaces and fitlog, matching the local return-verifier allowlist. No2022 fit, selection, final fit, upload or Kaggle submission occurs in the notebook.

Notebook SHA256: `2eef12f6de3a7b8c66792695d79bf5d40792515a3cb699a3e960e7d134aabb6d`.
Source SHA256: `ff52ce56947e9fccc2e11104333c84ac1f8a2018830062c61cdde33827499e5f`.
Preparation manifest: `artifacts/cuda-two-gpu-2021-notebook-002/manifest.json`, SHA256 `5fc77375ada65775f1d6df5d2a5d85917cb449e2a29036dc3f5eb9f62a262c26`.

Independent review approved notebook002 before hosted execution. Proof: `artifacts/cuda-two-gpu-2021-notebook-independent-audit-002/manifest.json`, SHA256 `d7bc2fb085a589b391a2effcaa3e71ef5bb0480a58b80bb952494fed08b67c23`. The review checked the exact plaintext code cell, both frozen protocols and source/runtime pins, namespace separation, archive compatibility, and seven mocked memory cases. Notebook001 remains an unexecuted preparation artifact; its host-only memory guard is superseded by002.


## Hosted execution observation

Root launched the reviewed notebook as PRIVATE version5, scriptVersionId364971894, **Frozen 2021 SHOP664 and capacity608**, at2026-11-02T17:49:34UTC. The viewer was **Queued** at17:50:35UTC, with T4x2 confirmed; no runtime, smoke, training or validation result is claimed yet. Do not start another copy. [Saved run](https://www.kaggle.com/code/ravinav/tdt4173-private-neural-experiment?scriptVersionId=364971894). Observation: `artifacts/cuda-two-gpu-input-upload-001/launch_observation.json` (SHA256 `2a15af22895223f773c03c7aafe48908e5fd5fb4d36d517fab21de0b751c0c4a`).

The editor showed original environment2026-10-22 before launch; the queued viewer displayed LatestContainerImage. The wrapper will verify the actual runtime and fail before any fit if it differs. Root monitors this saved run and downloads each separate return archive for strict local verification and the locked2021 decision.


At2026-11-02T18:00:29UTC root observed version5 **RUNNING**, both smokes and the pinned runtime guard passed, SHOP664 approximately11/14 models and capacity608 approximately5/14 models. Reported RAM headroom was24.5GiB. Neither completed fold score was known at this observation. This supersedes the earlier queued status; preserve and inspect the existing run.
