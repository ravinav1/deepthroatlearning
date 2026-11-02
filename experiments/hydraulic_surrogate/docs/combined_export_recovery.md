# Combined context and joint LP final export

Recovered 2026-11-02 after interruption. The existing full-data fit `lgb-normalized-lp-context-joint-final-001` had completed successfully in 992.53 seconds; it was not retrained. Its 608 inputs retain the exact 521 context inputs and append 87 joint-plant inputs. The fixed fit uses 350 trees, 31 leaves, learning rate 0.06, minimum leaf size 100, L2 1, feature fraction 0.9, deterministic 40% training-hour sample, seed 4173, no early stopping. Complete data, source, cache, feature and checkpoint hashes, environment, and sample IDs remain in its immutable manifest.

Checkpoint replay `replay-lp-context-joint-final-001` completed in 19.40 seconds. Its full raw float32 probability array and CSV match the fit exactly. Raw CSV SHA256: `ea03802cc1630a8c8e9a3c86b8e83a37916bbaff6ebed32297ad58543d760d29`.

The frozen selected recipe is 14.0625% LP base, 42.1875% LP context, 18.75% joint LP, and 25% combined LP, accumulated in that order after float64 promotion and followed by one Vinje projection. Its repeatedly reused purged chronological development scores are 0.9949377186691472 for 2021 and 0.9936914879536517 for 2022. Selection and acceptance are unchanged from `lp-context-joint-comparison-001`; the 2021 lock predates creation of the combined 2022 predictions. The small gains do not meet the 0.998/both-years transfer gate. The descriptive 2022 bootstrap interval crosses zero; these scores provide no leaderboard guarantee.

Export `submission-lp-context-joint-blend-001` completed in 3.51 seconds. Independent verifier `verify-lp-context-joint-export-001` completed in 5.51 seconds and checks all four raw checkpoint replays, model/data/source/cache hashes, frozen selection proof, earlier base/context proofs, exact schema and ID order, finite probabilities in [0,1], byte-exact float64 blend reconstruction, and CSV readback. The maximum serialization error is 4.999994862586732e-10. Four numerical-integrity helper tests passed. Source is `scripts/verify_lp_context_joint_export.py`, with a frozen snapshot and complete proof in the verifier artifact directory.

Submission: `artifacts/submission-lp-context-joint-blend-001/submission.csv`.

SHA256: `f7907cce9d7348a1c77959e64d5e6099303123846c0648b37c816bb495411fcf`.

No submission was made by the export agent. Root owns browser submission and the leaderboard observation. Preserve verified export manifests; record later observations separately in `kaggle_observation.json` and `docs/kaggle_scores.json`.

Reproduction commands use the existing uv-managed environment. Use fresh run IDs if repeating a check; do not overwrite these completed runs.

```sh
.venv-ml/bin/python scripts/train.py --run-id replay-lp-context-joint-final-001 --years --feature-set normalized-lp-context-joint --inference-from artifacts/lgb-normalized-lp-context-joint-final-001 --threads 2
.venv-ml/bin/python scripts/export_selected_blend.py --run-id submission-lp-context-joint-blend-001 --selection-run lp-context-joint-comparison-001 --final-runs lgb-normalized-lp-final-001 lgb-normalized-lp-context-final-001 lgb-normalized-lp-joint-final-001 lgb-normalized-lp-context-joint-final-001
.venv-ml/bin/python scripts/verify_lp_context_joint_export.py --run-id verify-lp-context-joint-export-001
.venv-ml/bin/python -m unittest tests.test_verify_lp_joint_export
```
