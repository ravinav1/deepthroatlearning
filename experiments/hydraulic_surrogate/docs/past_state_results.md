# Past-state specialist ablation result

The fixed 50% specialist blend improved 2021 slightly and regressed in 2022. Do not promote or fit these specialists on all training data. This result does not establish that reservoir history identifies initial operating state.

| Fold | Matched base micro AUC | Locked blend micro AUC | Difference |
|---|---:|---:|---:|
| 2021 | 0.9943700123 | 0.9944142705 | +0.0000442582 |
| 2022 | 0.9933485422 | 0.9932906222 | -0.0000579199 |

| Unit | 2021 base | 2021 history | 2022 base | 2022 history |
|---|---:|---:|---:|---:|
| Haukeli_G1 | 0.96235039 | 0.96316710 | 0.96365352 | 0.95836184 |
| Lio_G1 | 0.98926163 | 0.98842045 | 0.97354759 | 0.97011105 |
| Hogga_G1 | 0.97915337 | 0.97982898 | 0.98627954 | 0.98474854 |

The 2021 joint-weight grid was 0.0: 0.9943700123, 0.25: 0.9944141648, 0.5: 0.9944142705, 1.0: 0.9942699412. The maximum selected 0.5; 0.25 was almost tied but was not substituted after seeing 2022. Only the frozen weight and baseline were evaluated as whole-matrix candidates in 2022. All three raw specialist unit AUCs regressed in that fold.

The run took 482.0 seconds with two threads, six 350-tree models, all 395 baseline features plus 225 past-only fields, exact saved sample rows and unchanged baseline parameters. No full-data fit or Kaggle submission occurred. These chronological years have been reused for development; neither score meets the 0.998 transfer threshold.

Source/data/model/sample/environment provenance and checkpoints: `artifacts/lp-past-state-specialists-001/manifest.json`. Selection lock: `selection_2021.json`. Saved predictions preserve raw three-unit outputs and the selected full 14-unit blend. Independent checks in `artifacts/lp-past-state-specialists-001-verification/manifest.json` confirm artifact hashes, exact row samples, aligned labels/IDs/units, bitwise blend reconstruction, and recomputed scores. This verification is not a fresh checkpoint-inference replay.

Input legality, measured coverage, formulas and tests: `docs/past_state_features.md`. Frozen feature code was not modified.
