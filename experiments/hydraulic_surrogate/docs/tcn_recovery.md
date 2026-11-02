# CUDA TCN recovery — 2026-11-02

## Final recovered result: reject the fixed24-epoch TCN

The downloaded2021 fold and its original smoke were fully verified. Raw TCN micro AUC is **0.9851992438242643**. The predeclared neural weights0/.1/.25/.5 score **0.9948884917510937 / 0.9947939462246643 / 0.9944392967058471 / 0.9932144346578071** after blending with the projected joint anchor and fixed Vinje projection. Every nonzero weight is worse. The immutable2021 lock selected0; no2022 fit is needed to reject this branch, and root explicitly chose not to launch it. Neural2022 performance is unmeasured; no submission or leaderboard score is assigned.

Lock: `artifacts/cuda-tcn-blend-2021-001/selection_2021.json`, SHA256 `9c59e49b3954c9c4e11d668316fe9c76fa9732f848cfbff166001e82162cd9d3`. Complete selection manifest remains immutable; the later decision is separate `rejection_decision.json`. Hosted fit102.114478s; local full verification/selection1.457389s. Original downloaded archive/run/smoke/checkpoints/OOF remain preserved.

Concurrent source writing produced two incompatible live selector/test variants during recovery. No real selection used an unaudited mutable copy. Authoritative code is the unique snapshot `artifacts/tcn_audit_reconciled_20261102T142849Z_a892a747/`, selector SHA256 `b14edb9f2e609038020202fd57c7c279b33835ba7c330b8608c8476bee51ad5f`. The combined_export agent independently approved those exact bytes before execution. Nine targeted snapshot tests passed in0.173s, including extra-file/symlink rejection. The unmatched original tests are preserved under that snapshot's `historical_unmatched_tests/`; they were removed from active discovery. Earlier preflight source paths are historical and should not drive inference.

Exact successful selection command (do not rerun this existing run ID):

```sh
TDT4173_WORKSPACE_ROOT=/Users/heap/programming/tdt4173 .venv-ml/bin/python artifacts/tcn_audit_reconciled_20261102T142849Z_a892a747/scripts/select_cuda_tcn_blend.py --year 2021 --gpu-run artifacts/cuda-tcn-download-001/cuda-tcn-2021-001 --manifest-sha256 73e9df6ccd4ab7b7d94f39c0dda471b53f4509a2443ce51e30c6e8b968ffc49e --smoke-manifest artifacts/cuda-tcn-download-001/cuda-tcn-smoke-001/manifest.json --run-id cuda-tcn-blend-2021-001
```

The execution/download guidance below is historical operational documentation. **Do not launch its2022 command for this rejected branch.**

Root owns all Kaggle/browser actions. The TCN audit agent performed only file/hash inspection and six small selection tests. No real feature matrix, Torch import, training, package install or browser action was performed by this agent.

## Verified state

- Current trainer, protocol and notebooks exactly match `cuda-tcn-audit-002` and `cuda-tcn-code-001`. The audit passed nine fixture groups. Trainer SHA256 `1afae39884da9d210a8ef695fb0c99636536bc688ce79961d502bf4c4b743e57`; protocol SHA256 `d55bee5f39fc50eb5dc9ed48b99f244b7cad75aff1c9bc91c10c3a29de8cdd88`; code ZIP SHA256 `e5c4f066fcf75a3733cc377bb0cd29b87ea70044e22afe00796fd8e48be8f0d6`.
- Parent recovery reported that the existing notebook had already launched `cuda-tcn-2021-001` as PID142. Parent subsequently observed its manifest `status:complete`, runtime102.114 seconds. This agent has not downloaded that run or independently confirmed its numerical score. Do not launch a duplicate.
- Parent recovered the original successful `cuda-tcn-smoke-001`; reuse its saved manifest. The later `FileExistsError` did not overwrite the successful run.
- Local `mps-tcn-smoke-003` is a preserved blocked run: Torch2.11.0 imported, MPS built but unavailable in that child tool context. `mps-tcn-smoke-004` completed at2026-11-02T14:13:47.759393Z with MPS available,126094parameters,22.5702ms median synthetic update,119472128 sampled driver bytes and436191232 process peak RSS. Its8.12534s overall runtime includes setup. Neither run loaded competition inputs or trained a real model. Historical absent-Torch claims are obsolete.

The fixed TCN uses395features,64channels, four residual blocks with dilations1/2/4/8,61-hour receptive field,14outputs, batch16 complete cases,24epochs, AdamW.001/weight-decay.0001, cosine schedule to.0001, seed4173, two CPU threads, cuda:0, float32, no AMP, early stopping or automatic fallback. Training-only float64 population statistics are clipped to[-10,10] after standardization. Six starting days are purged. No network/protocol change was made.

## Package the completed hosted run

Run this explicit cell only after reviewing the existing completed manifest. It preserves the model directory and packages its files under a separate output name. Adjust the optional log only to a path actually observed in the notebook. No anchor probabilities are uploaded.

```python
from pathlib import Path
import hashlib, json, zipfile
from IPython.display import FileLink, display

run = Path('/kaggle/working/tcn-runs/cuda-tcn-2021-001')
smoke = Path('/kaggle/working/tcn-runs/cuda-tcn-smoke-001/manifest.json')
m = json.loads((run / 'manifest.json').read_text())
assert m['status'] == 'complete' and m['year'] == 2021
package = Path('/kaggle/working/cuda-tcn-2021-001-transfer.zip')
with zipfile.ZipFile(package, 'x', compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(run.rglob('*')):
        if path.is_file():
            z.write(path, 'cuda-tcn-2021-001/' + path.relative_to(run).as_posix())
    z.write(smoke, 'cuda-tcn-smoke-001/manifest.json')
for path in [package, run / 'manifest.json', smoke]:
    print(path.name, hashlib.sha256(path.read_bytes()).hexdigest())
print('raw_micro_auc', m['raw_micro_auc'])
print('checkpoint_replay', m['checkpoint_replay'])
display(FileLink(str(package)))
```

Use the authenticated notebook's file download control/FileLink; keep notebook/dataset private. Observe and record remote hashes before download. If the transfer ZIP already exists, inspect/hash it rather than rerunning the exclusive creation cell. Download to an ignored artifact directory and verify its ZIP hash against the observed hosted hash, reject unsafe members/symlinks, then extract into a fresh directory. Preserve ZIP, manifests, checkpoints, probabilities, normalizer, case orders, source snapshots and log.

## Local selection and 2022 lock

The new `scripts/select_cuda_tcn_blend.py` performs only small-array verification/selection. It requires the exact hosted manifest hash, verifies every declared file and an exact file allowlist, source/protocol/smoke hashes,24epochs, purged IDs, saved normalization scope and dtype, all epoch case orders, and byte equality of raw probabilities and fresh-checkpoint replay. It independently recalculates raw AUC and verifies local anchor bytes, labels, IDs and unit order. It does not deserialize the Torch checkpoint or load features. The frozen numerical feature specification must match after removing only the machine-specific cache directory.

Use a fresh selection run ID. Replace paths/hash only with the actually downloaded files and observed hosted manifest hash:

```sh
.venv-ml/bin/python scripts/select_cuda_tcn_blend.py --year 2021 --gpu-run artifacts/cuda-tcn-transfer-001/cuda-tcn-2021-001 --manifest-sha256 ACTUAL_HOSTED_MANIFEST_SHA256 --smoke-manifest artifacts/cuda-tcn-transfer-001/cuda-tcn-smoke-001/manifest.json --run-id cuda-tcn-blend-2021-001
```

The anchor remains `lp-joint-comparison-001`, its already projected18.75%base/56.25%context/25%joint probabilities. This matches the established neural blend convention in `docs/neural_tcn_feasibility.md` and `scripts/cpu_mlp.py`. Select among neural weights[0,.1,.25,.5] on2021, exact ties to smaller weight; blend in float64 and apply Vinje projection. Do not substitute the608-feature candidate, tune unit weights or read2022 arrays during selection. These are reused development years.

The selector writes `artifacts/cuda-tcn-blend-2021-001/selection_2021.json`, validates it with the exact frozen remote validator, and prints its hash. Its schema includes `version:cuda_tcn_2021_blend_lock_v1`, selection year, training protocol, frozen anchor, full grid and selected weight, completed2021 flag, neural manifest/OOF hashes and local selector hash. It records source/package versions, exact command, hashes, scores and probabilities in its own manifest and appends the experiment ledger. Preserve this lock unchanged. An independent review should precede first actual selection.

Only after successful local selection, put this exact lock into the private hosted working directory through the authorized UI, verify its hash there, then execute2022 explicitly. No training-setting overrides:

```python
subprocess.run([
    '/kaggle/working/tdt4173-cuda-system-env/bin/python',
    '/kaggle/working/tdt4173-tcn-code-d55bee5f39fc/train_cuda_tcn.py',
    '--mode', 'fit', '--year', '2022',
    '--protocol', '/kaggle/working/tdt4173-tcn-code-d55bee5f39fc/protocol.json',
    '--run-id', 'cuda-tcn-2022-001',
    '--bundle', '/kaggle/input/datasets/ravinav/tdt4173-private-hydraulic-training-inputs',
    '--stage-root', '/kaggle/working/tdt4173-tcn-inputs-2022',
    '--smoke-manifest', '/kaggle/working/tcn-runs/cuda-tcn-smoke-001/manifest.json',
    '--selection-lock', ACTUAL_VERIFIED_LOCK_PATH,
    '--selection-lock-sha256', ACTUAL_VERIFIED_LOCK_SHA256,
], check=True)
```

After2022 completes, download and verify it in the same way. Run the local selector with `--year 2022`, its new hosted manifest hash, the same smoke, and `--selection-lock artifacts/cuda-tcn-blend-2021-001/selection_2021.json --selection-lock-sha256 ACTUAL_LOCK_SHA256`. It scores only zero and the locked weight and accepts a nonzero candidate only if both years strictly improve over this anchor. The transfer gate is separately reported only if both selected scores reach.998. Nothing here establishes public Kaggle success; the actual objective remains.999 and an observed top-three rank.

## Local selector verification

Command `.venv-ml/bin/python -m unittest discover -s tests -p 'test_cuda_tcn_selection.py'` passed six tests in0.023s. The tests cover required/hash-verified artifacts, unexpected files/symlinks, wrong IDs/units/labels, invalid probabilities, a one-ULP replay change, smaller-weight ties, restricted second-year scoring and exact lock/provenance validation. No fit or real array was used. Proof/source snapshots are in `artifacts/cuda-tcn-selection-preflight-001/`.
