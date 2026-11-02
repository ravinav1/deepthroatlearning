# Reviewed private608 GPU upload

Upload `artifacts/cuda-xgb608-private-dataset-001/cuda-xgb608-private-dataset.zip` to a **PRIVATE** Kaggle dataset. SHA256: `480cacd6a83aef078104fd35ddfeb5c0569f0dd7d84158f63f82a1511d766a9a`; size 51,053,983 bytes. Keep the existing private base dataset attached. The new archive has separate `code/` and `supplement/` namespaces; Kaggle may extract it automatically.

Independent approvals are `cuda-xgb608-independent-audit-001` and `cuda-xgb608-dataset-independent-audit-001`. Protocol SHA256 is `789c2a6419c526079e18ea83ac5dd1143ef0b2c590e522b163ddf4e890a72401`. The original395 experiment, original base bundle and original608 package remain unchanged. Root owns browser work and GPU scheduling; finish the current395 job on cuda:1 before starting the608 branch.

After attaching the new private dataset, this explicit smoke cell locates the launcher by its exact reviewed bytes. It handles Kaggle's extracted directory paths; no guessed dataset path is used.

```python
from pathlib import Path
import hashlib, subprocess, sys
expected = "8297e491bfe94fe7d78a8b171653a689c1494db201f59add13088f71f18b8891"
matches = [p for p in Path("/kaggle/input").rglob("launch_cuda_xgb608_dataset.py")
           if p.is_file() and not p.is_symlink()
           and hashlib.sha256(p.read_bytes()).hexdigest() == expected]
assert len(matches) == 1, matches
subprocess.run([sys.executable, str(matches[0]), "--mode", "smoke",
                "--run-id", "cuda-xgb608-smoke-001"], check=True)
```

Verify the successful original smoke manifest before a separate explicit2021 invocation. Do not rerun an existing run ID, and do not run all notebook cells together.

```python
subprocess.run([sys.executable, str(matches[0]), "--mode", "fit", "--year", "2021",
                "--run-id", "cuda-xgb608-2021-001", "--smoke-manifest",
                "/kaggle/working/xgb608-runs/cuda-xgb608-smoke-001/manifest.json"], check=True)
```

The launcher verifies the unique original input manifest and supplement manifest by frozen SHA, validates every code dependency, then delegates to the offline uv launcher. Smoke locates only code. Actual fitting verifies all staged bytes, exact521+87 feature names/formulas and the stored sample fingerprints. Anchor predictions stay local.

Download complete fold artifacts plus the original smoke manifest and fit log; capture archive/fold/smoke SHA256 values in the notebook before download. Use the independently reviewed return verifier `artifacts/cuda-xgb608-return-preflight-001/verify_cuda_xgb608_return.py`, SHA256 `8b192955b39c412cc924472c753c96a4413b252469ee807448d4d77d8bc5060b`, with explicit workspace, archive and observed hashes. It invokes the exact protocol-pinned local608 selector only after provenance/model/sample/schema/replay checks pass. A nonzero improving2021 lock must be independently verified before2022; no2022 weight retuning is allowed.
