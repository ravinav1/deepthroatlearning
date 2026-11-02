# Tokke G1/G2: physical symmetry does not imply symmetric labels

The training evidence rejects unconditional pair shrinkage toward the mean as a well-motivated next experiment. The units have exactly identical supplied generator attributes, but their commitment labels have a strong, persistent ordering. This conclusion uses only 2015–2020 rows; no 2021/2022 validation predictions or label values were parsed for this track.

`scripts/analyze_tokke_exchangeability.py --run-id tokke-exchangeability-001` explicitly reads the first 2,192 rows, verifies IDs 1–2,192 and dates 2015-01-01 through 2020-12-31, and extracts only the two units' 168-hour label vectors. It completed in 0.204 seconds. Source/data/environment hashes, exact dates/IDs, month/horizon summaries and saved pair arrays are in `artifacts/tokke-exchangeability-001/manifest.json`.

| Training pair state | Case-hours | Share |
| --- | ---: | ---: |
| Both off | 103,217 | 28.0286% |
| G1 off, G2 on | 120 | 0.0326% |
| G1 on, G2 off | 78,819 | 21.4033% |
| Both on | 186,100 | 50.5355% |

G1 is on in 71.9388% of case-hours; G2 is on in 50.5681%. Among the 78,939 discordant case-hours, G1 is the active unit in **99.8480%**. The weekly cases overlap, so these are descriptive counts, not independent samples or a significance test.

| Year | G1-only hours | G2-only hours | G1 share of discordant states |
| --- | ---: | ---: | ---: |
| 2015 | 11,672 | 6 | 99.9486% |
| 2016 | 15,536 | 4 | 99.9743% |
| 2017 | 15,414 | 9 | 99.9416% |
| 2018 | 9,910 | 0 | 100.0000% |
| 2019 | 13,322 | 8 | 99.9400% |
| 2020 | 12,965 | 93 | 99.2878% |

The direction is stable across all seven horizon days. Exceptions exist: 14 of the 2,192 cases contain at least one G2-only hour; no case is G2-only for all 168 hours. Therefore the evidence supports a strong ordering tendency, not a physical impossibility or a hard label rule.

The supplied topology has G1/G2 on the same penstock with identical efficiencies, capacity limits and switching costs. Their objective symmetry permits alternative physical optima. The observed label asymmetry is consistent with a solver/unit priority or deterministic allocation convention, but the unavailable SHOP commands prevent identifying its cause conclusively. The poor unit-level score of the per-subset no-cost relaxation, despite equal financial objectives, is consistent with this distinction; it does not establish the exact labeling convention.

The optional transformation `p'_j=(1-alpha)*p_j + alpha*(p_G1+p_G2)/2` for alpha in {0, .25, .5, 1} was **not evaluated**: it would erase a systematic learned difference in the dominant one-unit state, so the training evidence does not support its stated exchangeable-noise hypothesis. No accepted prediction, frozen postprocessing function or cache changed.

A separately predeclared diagnostic can consider only cases where predictions reverse the observed tendency, using a soft correction and checking the exceptions chronologically. That is different from unconditional mean shrinkage. The root coordinator subsequently authorized that separate experiment; its results follow. The empirical direction must not become a hard physical assumption merely because the training rates are asymmetric.

## Separately authorized violation-only correction: no change accepted

`sintef_project/pair_order.py` copies the probability matrix and changes only rows where `p_TokkeG1 < p_TokkeG2`. For those rows, each probability becomes `(1-alpha)*p + alpha*pair_mean`. Existing ordered pairs, all other units and the already applied Vinje projection remain bitwise unchanged. Three tests cover scope, input immutability, named-column alignment, alpha endpoints, preserved pair totals, probability bounds and invalid inputs. The frozen `postprocess.py` is untouched.

The fixed alpha grid was 0, .25, .5 and 1. `scripts/compare_tokke_order.py --run-id tokke-order-comparison-001` uses the accepted joint blend from `lp-joint-comparison-001`: raw weights .1875 baseline LP, .5625 LP context and .25 joint LP, followed by its existing Vinje projection. Input hashes match that comparison and its independent verification. A protocol snapshot is written before prediction arrays are decoded. Selection uses pooled 2021 micro AUC, with smaller alpha winning exact ties; an immutable selection is saved before 2022 decoding. Only that choice and the baseline are evaluated on 2022, with strict improvement required in both years.

| Alpha | 2021 pooled micro AUC | Tokke G1 AUC | Tokke G2 AUC |
| --- | ---: | ---: | ---: |
| **0** | **0.994888491751** | 0.997901368042 | 0.997347714722 |
| .25 | 0.994888416887 | 0.997901460722 | 0.997348412489 |
| .5 | 0.994888381653 | 0.997901343768 | 0.997349087538 |
| 1 | 0.994888373739 | 0.997900921188 | 0.997350453862 |

The rule selects **alpha 0**, leaving predictions unchanged. The locked 2022 pooled micro AUC is consequently the unchanged **0.993641964015**. No nonzero alpha is evaluated on 2022. The differences in 2021 pooled AUC are tiny and are not claimed statistically meaningful; they nonetheless fail the predeclared strict-improvement criterion. A small gain in G2's individual AUC does not override the competition's pooled metric. The accepted joint submission remains unchanged.

Full source snapshots, baseline/proof/raw-data hashes, exact IDs, timestamps proving selection precedes 2022 decoding, four 2021 scores, selected arrays and artifact digests are in `artifacts/tokke-order-comparison-001/manifest.json`, with immutable `protocol.json` and `selection_2021.json`. These chronological folds have been reused in adaptive research and are not a newly untouched evaluation set. This run adds no Kaggle score or migration eligibility.
