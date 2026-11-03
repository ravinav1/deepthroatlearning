The original 0.99652 anchor and the canonical115 candidate pass a provided-data-only lineage audit. The new export `artifacts/provided-only-submission-001/submission.csv` is exactly the original canonical115 submission, with SHA256 `b422d6ca990b8d18e7d397060254af3a95fb3e5a0e1cdc191a88baa10c6f5783`. The historical external-data result is a different submission and is not attributed to this file.

The immutable lineage receipt is `artifacts/provided-only-lineage-audit-002/manifest.json`, SHA256 `6c017919db2e6879e718398bb0aec7fc59f79b73b68a439a73a9750885b61f92`. It verifies 528 files and 253 source snapshots. Its numerical leaves are exactly the original eight kernel/extended files plus `prediction_mapping.csv` and `sample_submission.csv`. The kernel label table supplies the original training labels only. No downloaded SINTEF dataset, public solver output, external constraint or externally trained teacher is an ancestor of the audited candidate.

The full prediction ancestry is:

1. Four locally trained LightGBM components use the base LP features, temporal LP context, joint-plant LP features and their combined representation. Their blend weights are `[0.140625, 0.421875, 0.1875, 0.25]`, followed by the original Vinje projection.
2. A GPU608 model contributes 50% to that projected blend.
3. A GPU664 model contributes 25% to the preceding projected blend.
4. A LightGBM664 model contributes 10% to the preceding projected blend, yielding the original 0.99652 anchor.
5. A pooled generator-relative canonical115 model contributes 25% to that anchor, followed by the original projection.

Every nested projection and dtype conversion is preserved, and the audit reconstructs all saved prediction bytes exactly at each step. The first audit attempt failed because the checker multiplied raw float32 LightGBM outputs before converting them to float64; the original exporter performs the conversion first. The largest difference was 4.47e−9. Audit002 corrected that checker defect without loosening equality or modifying any candidate artifact. Audit001 remains recorded as failed.

Three caches have potentially confusing names: `lp-full-001`, `lp-joint-full-001` and `lp-shop-full-001`. They are our own input-only SciPy optimizations of the supplied topology, initial storage, prices, inflows, water values and constraints. They are not the later downloaded SHOP simulation archives. The audit verifies their raw input hashes, generation source hashes and consumed feature bytes. Every numerical payload in the hosted GPU bundles is checked against those same original leaves or closed local caches. Archived training-source snapshots are used where working-tree scripts changed later.

The final canonical model uses original labels through 2022. Its 2021 and 2022 comparisons retain the original six-day purge; their latest training target hours precede each validation year. The 25% choice selected on 2021 remains unchanged for 2022. Independent rank AUC and array checks reproduce:

| Development year | Original anchor | Canonical115 blend |
| --- | ---: | ---: |
| 2021 | 0.9951042081969695 | 0.9952596524075921 |
| 2022 | 0.9938717710733546 | 0.9940509264719491 |

These are reused development folds, not fresh untouched estimates. This audit does not retrain every historical model or repeat every LP solve. It verifies their saved source/data/checkpoint lineage, exact arrays, existing full-replay receipts and the final composition; small canonical checkpoint replays are also checked independently. General SHOP documentation appears in historical feature specifications as algorithm references. Those references do not introduce external numerical data.

The new export independently passed in `artifacts/provided-only-export-audit-001/manifest.json`: all source, input, lineage and artifact hashes; original-label alignment; both rank AUCs; exact prediction composition; 725 case IDs and 2,353 columns; probabilities in [0,1]; and actual CSV readback error below 5e−10. Export manifest SHA256 is `667f9229e0d521a47dfafce84e9a313797ec9f5225197f0f775ad8dafdfc5d1b`. No public score is inferred by this report.

Reproduction of the audits:

```sh
.venv-ml/bin/python scripts/provided_only_lineage_audit.py --run-id provided-only-lineage-audit-NEW
.venv-ml/bin/python scripts/provided_only_export_audit.py --run-id provided-only-export-audit-NEW --export artifacts/provided-only-submission-001
```
