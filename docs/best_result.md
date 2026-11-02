# Best recorded result

**Public micro ROC-AUC 0.99652, rank 18**, observed on 2026-11-02 at
22:04:37 UTC. This is the hydraulic LGB664–SHOP nested blend. It was submitted
at 22:02:46 UTC; the UI did not expose a numeric submission ID. Private score
is unknown. This migration did not make a new Kaggle submission.

- Recipe: [`configs/hydraulic_surrogate_best.json`](../configs/hydraulic_surrogate_best.json).
- Restored CSV: `models/hydraulic_surrogate/submission-lgb-shop664-anchor-blend-001/submission.csv`.
- SHA256: `1c1d173816127997e711d54c8269406830666af66b6f4eb6d24f4f227b70d496`.
- Dated observation: the same run's `kaggle_observation.json` in the artifact archive.
- Full model/source/selection evidence: [original best-result history](../experiments/hydraulic_surrogate/docs/best_result.md)
  and [LGB664 execution record](../experiments/hydraulic_surrogate/docs/lgb_shop664_final_execution.md).

Let P be the fixed Vinje probability projection. Compute in float64, preserving
all intermediate projections:

```text
L = P(0.140625 LP + 0.421875 context + 0.1875 joint + 0.25 context_joint)
A = P(0.5 L + 0.5 GPU608)
B = P(0.75 A + 0.25 GPU664)
best = P(0.9 B + 0.1 LGB664)
```

The seven components each have 14 saved classifiers: **98 checkpoints**. The
four original LightGBM models use 395, 521, 404, and 608 features. GPU608/GPU664
use 600 trees, depth 6, learning rate 0.05 and seed 4173. LGB664 uses 350 trees,
31 leaves, learning rate 0.06, minimum leaf size 100, L2 1, feature fraction 0.9,
and seed 4173. The final sample has 196,358 training hours. Expanded parameters,
sample indices, input/source hashes and feature formulas remain in each original
manifest and protocol; the adapter does not refit or flatten this ensemble.

Purged, reused development AUC is **2021: 0.9951042081969695; 2022:
0.9938717710733546**. The 10% LGB664 weight was selected on 2021 and confirmed
unchanged on 2022 against the SHOP anchor and matched LGB608 control. The later
public result improved the preceding 0.99651 by 0.00001. These small gains do not
establish a generalization guarantee. Both validation targets in GOALS.md remain
unmet. The older Round5 public reference (0.99172) is retained separately.

## Reproduce from the migrated repository

Follow [environment and artifact setup](REPRODUCING.md), then:

```sh
.venv-hydraulic/bin/python src/hydraulic_surrogate.py export-best --run-id teammate-export-001
.venv-hydraulic/bin/python src/hydraulic_surrogate.py replay-best --run-id teammate-replay-001
```

`export-best` verifies and composes saved component predictions. `replay-best`
rebuilds the frozen 664-feature table from supplied inputs and saved LP caches,
replays all 98 checkpoints on CPU and compares every component against its saved
probabilities. All 70 LightGBM predictions must be bitwise identical. The 28 XGBoost
models were originally replayed on CUDA; their separate CPU portability check
requires maximum absolute error at most 1e-5. Exact export then uses the hash-verified
original CUDA probabilities and requires the original CSV checksum. CPU/GPU bitwise
equivalence is not claimed. Both refuse existing run IDs and record a new manifest under
`models/hydraulic_surrogate/reproductions/`. A replay mismatch is a failure, not
permission to modify historical checksums.

Original machine-specific audit scripts are preserved unchanged. The new adapter
records relocation separately; it does not pretend that a different OS or Python
installation matches an old whole-environment assertion. See the migration
verification report for what was actually tested in this checkout.
