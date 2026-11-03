# Results and numerical provenance

Updated 3 November 2026 from preserved receipts, not a fresh leaderboard refresh.
The highest recorded public result is **0.99952**, observed at rank 1 around
03:09 UTC. It directly uses external solver outputs. The best measured
supplied-numerical-data result is **0.99653**; the latest submitted model in that
track scored **0.99650**. Private scores are unknown.

The external dataset was discovered during autonomous experiments and the Kaggle
submission sequence. Subsequent reconstruction research is externally informed;
prior exposure to external test-date outputs cannot be undone by removing those
files from inference. The [reconstruction plan](RECONSTRUCTION_PLAN.md) is recorded
for later work and is **not started by this incorporation**.

## Historical 0.99952 recipe

Let A0 be the preserved 0.99652 ensemble described below, P its empirical Vinje
ordering projection, and clip bound solver numerical excursions to [0,1]. Keep
all nested projections in float64:

```text
A1 = P(0.90 A0 + 0.10 clip(SHOP17.15.1_NoMIP_solver_committed))
     public 0.99690
A2 = P(0.50 A1 + 0.50 clip(SHOP17.17.0_envC_NoMIP_solver_committed))
     public 0.99795
Final = P(0.50 A2 + 0.50 clip(SHOP17.17.0_envC_AllMIP_solver_committed))
        public 0.99952
```

The external source is the [SINTEF data product](https://data.sintef.no/product/dp-107e1305-9802-4d21-9599-4554a1a2dd70),
especially its [SHOP 17.17.0 release](https://data.sintef.no/feature/fe-973ce2e0-064f-4136-a4cc-d7d3eab679c4).
The envC inputs matched all 3,647 exposed cases and supplied environmental
constraints. The older 17.15.1 source has a recorded initial-storage discrepancy.
NoMIP states are continuous relaxed optimization variables, not calibrated
probabilities; AllMIP states are essentially binary simulator predictions.
The external blending stages fit no classifier. They use `solver_committed`
directly, excluding `result_committed` and official hidden labels.

2021 selected the final AllMIP family/0.5 weight; 2022 evaluated that locked
choice. Reused development AUC: **0.999295078891271 / 0.9996107099972343**.
0.99795 is a separate measured public AUC, not the 2022 development AUC of
0.9979688210864383. AUC is a pooled ranking metric, not scheduling accuracy.

Winning CSV SHA256:
`5d9b58d4fa9153ed69d7a77ffb1f8ec2ad3e04c8894fda9329b6120d8b96f93e`.
See the byte-preserved [complete historical recipe](../reports/reconstruction_20261103/research/best_result.md),
[winning receipt](../reports/reconstruction_20261103/receipts/metric-trick-public-mip17170-submission-001.json),
and [evidence index](../reports/reconstruction_20261103/evidence_index.json).
Numerical artifacts remain at the original workspace paths recorded there; this
documentation update does not publish or regenerate them.

## Supplied numerical data track

The [best measured recipe](../reports/reconstruction_20261103/research/best_provided_only_result.md)
scored **0.99653**. It blends 10% retrospective971 predictions into A0 for Hogga,
Lio and Haukeli, leaving the other eleven units unchanged. Its future historical
storage/water-value covariates are supplied, but retrospective; this does not
establish prospective availability or organizer approval. The recorded gap to
0.99952 is **0.00299 AUC**.

The [latest submitted capacity model](../reports/reconstruction_20261103/research/provided_capacity_result.md)
scored **0.99650**: P(0.9 clean canonical115 blend + 0.1 raw 63-leaf model).
It uses 1,400 trees, learning rate 0.06, seed 4173 and 115 supplied-input/local
physics features. Development AUC: 0.9952676572007169 / 0.9940771953548430.
Its CSV SHA256 is `efc9166cab6a5809625eabe65b958a0f13b8630c8a0a6a2315be9af434128f8d`.
Startup142 and busbar144 are completed, audited, **unsubmitted** candidates with
no public score. Do not attribute the external winner's first-place rank to them.

See the [information boundary](reconstruction_information_boundary.md) and
[mechanism evidence](reconstruction_mechanism_evidence.md). General external
research inspiration and numerical ancestry are separate claims.

## Preserved 0.99652 baseline and existing replay commands

The following recipe and adapter remain unchanged. Commands named `export-best`
and `replay-best` still reproduce this baseline, not the external winner.

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
