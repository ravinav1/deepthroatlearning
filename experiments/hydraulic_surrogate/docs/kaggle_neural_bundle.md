# Local private-neural input bundle

`artifacts/kaggle-neural-bundle-001/kaggle-neural-inputs.zip` is ready for review. This task performed no upload, notebook execution, GPU smoke test or model training. The portable trainer is a separate workstream.

- ZIP size: 62,925,128 bytes.
- ZIP SHA256: `2dd110c37ae3bd81cda27104e09386a016bd88a8972e9c8708ee4518d410db44`.
- Payload: 24 allowlisted files, 153,925,168 uncompressed bytes, plus `bundle_manifest.json`.
- Build and streamed readback: 2.694 seconds.
- Full report and builder snapshot: `artifacts/kaggle-neural-bundle-001/manifest.json` and `source/scripts/build_kaggle_neural_bundle.py`.

The unzipped review copy is `artifacts/kaggle-neural-bundle-001/payload/`. The ZIP root has the same repository-relative paths, so a later staging step can preserve the existing strict LP loader's structure.

## Contents and exact verification

The ten raw/input-schema files are exactly those required by the frozen LP manifest: initial volumes, prices, inflows, terminal water values, commitment training labels, minimum-flow/minimum-volume constraints, topology YAML, test mapping and sample submission. Their bytes and paths are unchanged.

The LP cache contains its original `features.npz`, original `manifest.json` and cached schema-repair source. The root source closure includes `features.py`, `physics.py`, `lp_features.py`, `scripts/build_lp_features.py` and `scripts/repair_lp_cache_schema.py`. Additional required modules are `context_features.py`, `normalized.py`, `postprocess.py` and the original `kaggle_metric.py`. The frozen proposed neural protocol and its referenced baseline feature manifest are included as provenance; no model checkpoints or prediction arrays are included.

The original LP manifest records historical artifacts beyond those needed by the feature loader. Per-case solutions and unrelated cache artifacts are intentionally absent from this bundle; their presence is not claimed. The strict loader checks the feature archive and root data/source hashes, all of which are included. The Unicode schema repair is already applied and independently recorded; it must not be rerun on this frozen cache.

The inner `bundle_manifest.json` lists every payload path, SHA256, size, role and source of the expected checksum. Each ZIP member was read back through the decompressor, checked against its expected size and hash, and checked for duplicate or unexpected paths. All source files parse as Python. Only the NPZ header and small case-ID/unit/column metadata arrays were decoded; its 3,647×168×78 numeric feature array was not loaded and the 395-column full feature matrix was not constructed.

The bundle excludes credentials, environment directories, model checkpoints, prediction arrays, per-case LP solutions, other caches, unrelated scripts and private repository snapshots. The build uses an explicit allowlist rather than copying the repository recursively.

## Later staging contract

Extract preserving relative paths into a writable repository root, verify every `payload_files` entry from `bundle_manifest.json`, then import feature builders from that root. Preserve the original LP manifest and source bytes. The existing 395-column proposed protocol has an old blend anchor and local path metadata; any authorized portable-training changes need their own versioned protocol and explicit path mapping. Do not silently rewrite frozen provenance or weaken hash checks.

The prepared notebook has not been run by this task. An actual GPU smoke test, portable trainer validation, upload review and explicit authorization for remote execution remain separate requirements owned by the root coordinator.

```sh
.venv-ml/bin/python scripts/build_kaggle_neural_bundle.py --run-id NEW_LOCAL_BUNDLE
```
