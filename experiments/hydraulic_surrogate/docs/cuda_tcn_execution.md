# Portable CUDA TCN code package

Import notebooks/cuda_tcn_smoke.ipynb for metadata and synthetic-only CUDA smoke. Import notebooks/cuda_tcn_review.ipynb for the same default smoke plus separately disabled real-fold cells. No real fit is enabled by default.

The notebook embeds the exact trainer/protocol and writes them under /kaggle/working. It creates an offline uv --system-site-packages environment using the hosted interpreter; preinstalled packages are reused and no install/download occurs. The private dataset mount is discovered by exact bundle-manifest SHA256, not an assumed path. Before a real feature build every listed payload size/hash and exact allowlist are verified during staging.

The sealed input data bundle remains unchanged: artifacts/kaggle-neural-bundle-001/kaggle-neural-inputs.zip SHA256 2dd110c37ae3bd81cda27104e09386a016bd88a8972e9c8708ee4518d410db44. Its dataset is private under root's verified UI workflow.

Only one explicit year is accepted by the trainer. The2021 run decodes targets through2021 only and writes OOF/checkpoint/normalizer/source/protocol/environment files. Download these for local selection against the accepted joint anchor, weights[0,.1,.25,.5]; anchor arrays stay local. The2022 run requires the externally created2021 lock and its explicit SHA256. The lock contract is documented in the review notebook and validate_selection_lock(). It is not generated remotely.

Each fit uses exactly24 epochs, batch16 complete cases, all purged historical cases, fixed seed4173 and the declared395-feature64-channel TCN. No early stopping or public-score tuning. It saves float64 training-only statistics, final model/optimizer/RNG state and all per-epoch case orders, then reloads a hash-verified own checkpoint with weights_only=True into a fresh model and reproduces OOF with saved stats under eval/inference_mode. Any probability-byte mismatch fails. No implicit CPU fallback, AMP, compilation, batch-size or epoch adjustment.

Use fresh output run IDs if repeating a smoke. Existing result directories are not overwritten. When importing the review notebook into a session with an earlier smoke, either use a fresh smoke ID or explicitly point SMOKE_MANIFEST to the already verified matching smoke. Separate stage roots are supplied for2021 and2022.

Independent code review: artifacts/cuda-tcn-audit-002 records9 passing fixture groups. Notebook cells compile without execution. Actual CUDA forward/backward/model replay and speed/memory remain unverified until the hosted smoke/fit execute. Root's runtime probe reports Python3.13.15, torch2.11.0+cu128/CUDA12.8 with2devices, numpy2.1.3,pandas2.3.3,scipy1.16.3,sklearn1.6.1,PyYAML6.0.3,uv0.13.5. No package versions are changed. Do not claim bitwise Mac-versus-Linux full-feature equality; the frozen LP cache bytes and source/data fingerprints are checked, while actual hosted versions are recorded.

Official references consulted: https://docs.pytorch.org/docs/2.15/generated/torch.use_deterministic_algorithms.html and https://docs.pytorch.org/docs/main/notes/randomness.html. Deterministic CUDA algorithms are requested and unsupported behavior fails rather than falling back. Compatibility with the actual installed version must be demonstrated by the synthetic gate.
