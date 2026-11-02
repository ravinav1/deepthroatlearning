The fixed 36-case SHOP-inspired relaxation diagnostic passes its predeclared descriptive screen. This supports reviewing a label-free feature cache; it establishes no trained-model, chronological-validation, or leaderboard improvement.

Independent verification is preserved in `artifacts/verify-lp-shop-relaxation-diagnostic-001/`. Its manifest SHA256 is `e73ec5f0bb587451670437dcb1d2b49170b88acb3d72088098a29cfaef9d7bdd`. The audited diagnostic manifest is `95ca40b0ff3b29f6cb36bb55b5539dcdcd3839fd05ca4f9a222d447a7821800e`, with protocol `583989ba708c304144c5e077e26e46ae772fd846af33fc66a6c5b736c3084734`. All 225 saved artifact hashes, source/data/cache pins, 36 original input reconstructions, saved solver replays, original selected labels, metric tables, and twelve declared screen gates passed. The audit took 2.34 seconds and ran no solvers or model fits. Exactly 36 pre-2021 target rows were decoded; the final selected window ends 2020-12-31 23:00 UTC.

| Arm | Q proxy AUC | P proxy AUC | Weak-three Q AUC | Cutoff errors | On-fraction MAE | Switch-count MAE |
|---|---:|---:|---:|---:|---:|---:|
| Cached base | 0.925327975264 | 0.924209935648 | 0.881962261526 | 8,224 | 0.080215420 | 4.186507937 |
| Fresh base, float64 RHS | 0.925327976396 | 0.924209935648 | 0.881962261526 | 8,224 | 0.080215420 | 4.186507937 |
| Linked state, zero cost | 0.925744293690 | 0.924652834236 | 0.881962261526 | 8,172 | 0.079246977 | 4.230158730 |
| Linked state, supplied cost | 0.944652174548 | 0.943782610203 | 0.905475628876 | 7,894 | 0.087750378 | 0.918650794 |

The supplied-cost improvement over the fresh base is approximately 0.0193242 Q-proxy AUC. Promoting natural inflows to float64 does not explain it. Every base solution admits a feasible fractional state lift; the zero-cost and fresh-base objectives agree within numerical tolerance. Some unit allocation changes remain because the LP can have multiple optima.

The independent audit reconstructed each pool's hourly balance and the entire system's water balance from saved flow, bypass, spill, storage, float64 initial volume, and natural inflow. Maximum absolute residuals, in Mm3, were:

| Arm | Hourly pool residual | Aggregate residual | Stored aggregate diagnostic |
|---|---:|---:|---:|
| Cached base | 2.936e-5 | 4.787e-5 | 1.397e-4 |
| Fresh base, float64 RHS | 5.204e-14 | 1.023e-11 | 1.283e-4 |
| Linked state, zero cost | 5.204e-14 | 1.023e-11 | 1.023e-11 |
| Linked state, supplied cost | 3.663e-13 | 1.137e-11 | 1.137e-11 |

The fresh base's stored aggregate diagnostic uses float32 `initial.sum()` and is numerically misleading. Its actual saved flows conserve the float64 inputs. The historical cached arm genuinely incorporates the old float32 initial-hour RHS rounding. The original artifacts remain unchanged. All minimum-storage, maximum-storage, bypass, and Vest slacks are exactly zero in all four arms on all 36 cases. Reconstructed inequality violations are at most 1.95e-13, and objectives and transition costs match their saved values.

The mechanism is not uniformly better. On-fraction MAE increases from 0.0802154 to 0.0877504, even though pooled cutoff errors decline. The 2015–2018 block has 2,461 cutoff errors versus 2,439 in the cached base (its AUC improves). Tokke_G3 Q-proxy AUC declines by 0.00175822. Six units have more cutoff errors:

| Unit | Q-proxy AUC change | Cutoff-error change |
|---|---:|---:|
| Hogga_G1 | +0.041380347 | -55 |
| Tokke_G1 | +0.038459540 | -283 |
| Tokke_G2 | +0.008051196 | +174 |
| Tokke_G3 | -0.001758219 | +38 |
| Tokke_G4 | +0.003424806 | +125 |
| Lio_G1 | +0.003187106 | -22 |
| Byrte_G1 | +0.008438733 | -56 |
| Vinje_G1 | +0.044785551 | -255 |
| Vinje_G2 | +0.034091226 | -127 |
| Vinje_G3 | +0.021351627 | +97 |
| Haukeli_G1 | +0.008605999 | +100 |
| Songa_G1 | +0.007750193 | -35 |
| Kjela_G1 | +0.008377468 | +96 |
| Vesle Kjela_G1 | +0.024388672 | -127 |

These are descriptive samples selected by metadata, with overlapping year-end windows between blocks. A full-cache proposal must retain the fixed physics, preserve the zero-cost control for attribution, keep all supplied inputs label-free, and predeclare any learned experiment and 2021 selection lock before 2022 fitting. The screen does not justify replacing probabilities with the raw physical cutoff.

The completed audit was run from the repository root with:

```sh
UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python artifacts/verify-lp-shop-relaxation-diagnostic-001/verify.py
```

Preserve the completed proof. For a new replay, copy the script into a fresh `artifacts/<new-audit-id>/` directory and run that copy; its parent directory determines the new output path.
