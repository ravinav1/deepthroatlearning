# LGB664 final export and independent verification

Completed on 2026-11-02. This is a prepared, independently audited primary submission; no Kaggle submission ID or public score has been observed yet.

The frozen final protocol is `docs/lgb_shop664_final_protocol.json`, SHA256 `5513e67123a968ded3790ec6bcda898d50d3682685fa7f62020ea25108e322ad`. Its complete source/data/schema/feature/anchor closure remains unchanged. The final candidate is `artifacts/lgb-shop664-final-001/manifest.json`, SHA256 `2d3e2f85e40a7bbf8d849664a063a09a61eb30306746018935689754a6295ada`. Fourteen models use 350 trees, the original 664 features and 196358 fixed sampled training hours. The independent audit rebuilt features and replayed all 121800 test hours per model twice in float64, with exact raw float32 agreement.

The submission is `artifacts/submission-lgb-shop664-anchor-blend-001/submission.csv`, SHA256 `1c1d173816127997e711d54c8269406830666af66b6f4eb6d24f4f227b70d496`. It preserves the saved public-best SHOP664 anchor and applies `project_vinje(0.9 * saved_projected_SHOP_anchor + 0.1 * raw_LGB664)` in float64. Preserve all earlier nested projection stages; do not flatten the recipe. The weight was fixed on 2021 and confirmed unchanged on 2022. Scores are 0.9951042081969695 / 0.9938717710733546 versus SHOP anchors 0.9951018232145258 / 0.9938546734526688 and matched raw LGB608 controls 0.9950939207833579 / 0.9938428198540179. These are reused development results, not leaderboard scores.

Independent generic isotonic reconstruction matched the exported prediction array exactly and reproduced every CSV byte. Schema is 725 rows by 2353 columns with original IDs/order and valid [0,1] probabilities. CSV readback error is at most 5.000000413701855e-10 under the frozen 9-significant-digit format.

| Artifact | SHA256 | Runtime |
| --- | --- | --- |
| `artifacts/lgb-shop664-final-independent-audit-001/manifest.json` | `b63b5872411fa59eaea6c7c6cbc132803b6ea3ae445c7f871b3df67875d976b8` | 32.729 s |
| `artifacts/submission-lgb-shop664-anchor-blend-001/manifest.json` | `be4ec369caf7850c1259aee81b3344a738d62c51b9c547382fed9daa1cb6247c` | 1.788 s |
| `artifacts/submission-lgb-shop664-independent-audit-001/manifest.json` | `529c5b4e7a8bf1106882de2e0594c54de2d016ec979e5e69e84a723dbcfc3c99` | 1.973 s |

Replay used macOS 27.0.1 arm64, Python 3.11.15, NumPy 2.4.6, pandas 3.0.6, LightGBM 4.7.0, scikit-learn 1.9.1 and SciPy 1.17.1; two numerical threads. Replay peak RSS was 6237323264 bytes; its process exited before export. The verifier source is `a84739888990f64d99ac1f0057c9b7ac28c75cdedbf0193880b9514f4486fe46`; independent projection helper `0d191dba63e8c199baac2828bf36adc2e5eecef7a31dd6724f11da29f100ab2a`; exporter `812dcdadd0adb4e013816f19ef0db5cddcca6628c7a78614d2e1dd8326c092c4`. Each run preserves its source snapshot and exact inputs in the referenced manifests. No training, alternate weight or standalone diagnostic occurred during replay/export.

Exact completed commands follow. Run IDs are immutable; a deliberate reproduction needs fresh IDs and the newly generated audit hash passed to subsequent stages.

```sh
/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_lgb_shop664_final.py --protocol docs/lgb_shop664_final_protocol.json --protocol-sha256 5513e67123a968ded3790ec6bcda898d50d3682685fa7f62020ea25108e322ad --final-run artifacts/lgb-shop664-final-001 --final-manifest-sha256 2d3e2f85e40a7bbf8d849664a063a09a61eb30306746018935689754a6295ada --run-id lgb-shop664-final-independent-audit-001
```

```sh
/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/export_lgb_shop664_final.py --protocol docs/lgb_shop664_final_protocol.json --protocol-sha256 5513e67123a968ded3790ec6bcda898d50d3682685fa7f62020ea25108e322ad --final-run artifacts/lgb-shop664-final-001 --final-manifest-sha256 2d3e2f85e40a7bbf8d849664a063a09a61eb30306746018935689754a6295ada --final-audit artifacts/lgb-shop664-final-independent-audit-001/manifest.json --final-audit-sha256 b63b5872411fa59eaea6c7c6cbc132803b6ea3ae445c7f871b3df67875d976b8 --run-id submission-lgb-shop664-anchor-blend-001
```

```sh
/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/audit_lgb_shop664_final.py --protocol docs/lgb_shop664_final_protocol.json --protocol-sha256 5513e67123a968ded3790ec6bcda898d50d3682685fa7f62020ea25108e322ad --final-run artifacts/lgb-shop664-final-001 --final-manifest-sha256 2d3e2f85e40a7bbf8d849664a063a09a61eb30306746018935689754a6295ada --export-run artifacts/submission-lgb-shop664-anchor-blend-001 --export-manifest-sha256 be4ec369caf7850c1259aee81b3344a738d62c51b9c547382fed9daa1cb6247c --raw-audit artifacts/lgb-shop664-final-independent-audit-001/manifest.json --raw-audit-sha256 b63b5872411fa59eaea6c7c6cbc132803b6ea3ae445c7f871b3df67875d976b8 --run-id submission-lgb-shop664-independent-audit-001
```

