# Frozen2022 GPU confirmation

Import `artifacts/cuda-two-gpu-2022-notebook-001/tdt4173-frozen-2022.ipynb` into the existing PRIVATE two-T4 notebook. Both experiments passed their independently audited fixed2021 gates. SHOP664 and capacity608 each retain their own locked0.25 weight; this notebook does not choose a combined blend or tune weights.

The notebook embeds the exact2021 lock and audit bytes. SHOP lockSHA `1783327cd4ae46182f19ee902dac606f106115d66676518e8f3004e159b6cce5`; capacity lockSHA `f418296446f832b408ccc2e7ca3aee6825408e11420767252f92785f193665be`. Original source/protocol/runtime pins and the cgroup-aware memory guard are unchanged. Fresh genuine smokes use `cuda-xgb664-smoke-002` and `cuda-xgb608-capacity-smoke-002`; fits use `cuda-xgb664-2022-001` and `cuda-xgb608-capacity-2022-001`. Smokes run sequentially, then fits use separateGPUs with the same first-model/20second/12GiB guard. Each completion or failure independently emits its return ZIP. No final fit, export or submission occurs here.

Notebook SHA256 `21844a0ef5da6027e7aefb9a7315cf4aabd1edf187ccc3788e8454aeb3f5efe4`; plaintext source SHA256 `d023bdfe38af784ac360a63956e910ee84bc8987bb836cdf3a75df647d424bdc`. Preparation manifestSHA `a2a66268a0f1edd933144a14037ad1d77f01d582de8d67fe93d9c25904ac9c67`. Independent final notebook audit passed: `artifacts/cuda-two-gpu-2022-notebook-independent-audit-001/manifest.json`, SHA256 `cf19c20525be31b6e88548ee83f30ec969d2aa5cc6ae81669aee4acace62443f`.

Prepared and independently approved at2026-11-02T18:13UTC. Root owns hosted execution and downloads; no2022 runtime or score is claimed in this preparation record. Exact builder command and all input paths/hashes are in `artifacts/cuda-two-gpu-2022-notebook-001/manifest.json`.


Root subsequently launched the reviewed notebook as PRIVATE version6, scriptVersionId364978490, **Frozen 2022 SHOP664 and capacity608**. Viewer status is **RUNNING**; runtime/smoke results are pending. Evidence: `artifacts/cuda-two-gpu-2022-launch-observation-001`. This supersedes preparation-only status. No2022 score is known.
