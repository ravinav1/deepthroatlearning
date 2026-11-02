# LP locality matched ablation

Only the approved per-unit column lists changed. Model hyperparameters, seed and exact sampled training rows match the full-feature LPv1 LightGBM baseline; CPUthreads were reduced to2. No smoothing/projection, newfeaturecomputation or LPsource changes.

| Year | Baseline microAUC | Frozen-weight microAUC |
|---|---:|---:|
| 2021 | 0.99437001 | 0.99444321 |
| 2022 | 0.99334854 | 0.99341376 |

2021-selected joint weight:0.25. Improves both matchedfolds:True. Meets>=.998both:False.

These are reused chronological developmentfolds. No fullfit or Kaggle submission was made. A positive matched result still needs comparison with the prioraccepted combinedcandidate before deciding any fullfit.

Exactcode/data/model/sample provenance, perunitAUC, allgridresults and selected/rawlocalitypredictions:`artifacts/lp-locality-001/manifest.json`.

## Unit diagnostics

| Year | Unit | Full-feature baseline AUC | Locality-only AUC |
|---|---|---:|---:|
| 2021 | Haukeli_G1 | 0.96235039 | 0.95008793 |
| 2021 | Lio_G1 | 0.98926163 | 0.98350590 |
| 2021 | Hogga_G1 | 0.97915337 | 0.97700661 |
| 2022 | Haukeli_G1 | 0.96365352 | 0.90911281 |
| 2022 | Lio_G1 | 0.97354759 | 0.95869241 |
| 2022 | Hogga_G1 | 0.98627954 | 0.98347169 |

The local-only model loses within-unit AUC for all three units on both folds. Haukeli2022 is particularly worse. A25%mixture still improves the matched pooledmetric modestly, indicating complementary predictions; this does not establish that calendar or distantreservoir information was a harmful shortcut. Removing those inputs wholesale is not supported by this test.

The frozen localblend remains below the accepted75%LPcontext/25%LPbase candidate on both years (0.99478111/0.99347025), and below the existing specialist/count composition. Recommendation:retain the stronger acceptedcandidate and do not spend a fullfit on this localitybranch at present. No fullfit was run.
