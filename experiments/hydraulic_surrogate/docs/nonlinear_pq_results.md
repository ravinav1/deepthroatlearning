# Nonlinear PQ specialist results

The predeclared 2021 gate passed, but the unchanged 2022 candidate failed the required anchor comparison and is rejected. No full-data fit, export, Kaggle submission or repository transfer has occurred for this branch. The independent 2022 review passed artifact integrity and confirmed that the candidate is ineligible.

`nonlinear-pq-specialists-2021-001` completed in 232.940 seconds. It rebuilt the exact 608-feature context/joint prefix, verified case IDs/dates against the six-field cache, and fitted the three specified specialists with the frozen 614-column recipe. Each of the three saved checkpoints reproduced every 2021 raw validation probability exactly. Training used the original 146,899 sampled rows, six-day purge, 350 trees, 31 leaves, learning rate .06, min leaf 100, L2=1, column fraction .9, seed4173 and two threads; no early stopping or later-year target decoding.

| 2021 global replacement weight | Micro AUC |
| --- | ---: |
| 0 | 0.9949377186691472 |
| .1 | 0.9949583567124495 |
| .25 | 0.9949757402943934 |
| .5 | **0.9949778297213558** |

The selected .5 splice exceeds the fixed current anchor by 0.0000401110522086 and the same-weight saved 608-feature control (0.9949257714128659) by 0.0000520583084899. The .25 and .5 scores are close, but the maximum and original tie rule select .5; no alternate weight will be chosen after 2022. These are reused development scores, not leaderboard evidence.

The pooled improvement is not universal within each generator:

| Raw specialist 2021 AUC | Saved 608 control | New 614 model |
| --- | ---: | ---: |
| Haukeli_G1 | 0.969680806320 | 0.966003335746 |
| Lio_G1 | 0.990803706679 | 0.990842809297 |
| Hogga_G1 | 0.979552267777 | 0.980427259196 |

Haukeli's within-unit ranking deteriorates. Preserve this counterexample and the joint global selection; do not introduce a post-hoc unit-specific weight.

The immutable selection lock was written at 2026-11-02T14:50:08.884373+00:00, before a new 2022 specialist run existed. SHA256: `2853a50245e85e2fbf615f85b5ba93736018b6dd9c1cf6420411d0aa24b30a43`. Training protocol SHA256: `dc8b89cae457340adc7d517648e207aa8bcb0c1abdce6150a8625fbac1b15693`. Complete source/data/cache/samples/fold IDs/environment/model/probability/checkpoint hashes are in `artifacts/nonlinear-pq-specialists-2021-001/manifest.json`; its SHA256 is `d53ed1d16ed3366cc50525c70316abd0cdd9206bc02fd9b5d6351976b21be147`.

Independent lock/artifact review passed in `artifacts/verify-nonlinear-pq-selection-2021-001/manifest.json`. Root then authorized and this branch launched the unchanged 2022 command:

```sh
UV_CACHE_DIR=/private/tmp/tdt4173-xgb-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/train_nonlinear_pq_specialists.py --year 2022 --run-id nonlinear-pq-specialists-2022-001 --selection-lock artifacts/nonlinear-pq-specialists-2021-001/selection_2021.json --selection-lock-sha256 2853a50245e85e2fbf615f85b5ba93736018b6dd9c1cf6420411d0aa24b30a43
```

The 2022 run completed in 278.091 seconds, with log `artifacts/nonlinear-pq-specialists-2022-001.log`. It retained the frozen .5 weight and all three checkpoint replays matched exactly.

| Frozen 2022 comparison | Micro AUC |
| --- | ---: |
| Current four-source anchor | **0.9936914879536517** |
| New nonlinear-PQ splice, .5 | 0.9936831534499260 |
| Saved 608-feature same-weight control | 0.9936498343743312 |

The candidate improves over the matched control by 0.0000333190755948 but falls below the actual anchor by 0.0000083345037257. This fails the predeclared requirement to beat both. The 2021 lock, weight, feature formula and target-unit selection remain unchanged; no alternate weight or unit-specific correction is evaluated using this 2022 result.

The completed 2022 manifest is `artifacts/nonlinear-pq-specialists-2022-001/manifest.json`, SHA256 `3f711877c467c7ece42db421b2c7c421d62fce5ecf7b6f76cb13335531450739`. Independent verification is recorded in `artifacts/verify-nonlinear-pq-selection-2022-001/manifest.json`: integrity passed, eligibility false. All raw and selected probabilities, three checkpoints, source snapshots, samples, rejected recipe and failed preflight evidence are preserved.

The separate `scripts/finalize_nonlinear_pq.py` draft was prepared while 2022 ran, but was never launched. It requires both independent fold reviews to accept the candidate, so this rejection blocks final fitting and export. Its static review found no urgent operational issue; no additional tests or model runs were spent on this rejected branch. The .998 two-year transfer threshold and .999 actual-leaderboard/top-three objective remain unmet.
