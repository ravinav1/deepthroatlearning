# Data audit

Inspected 2026-11-02 with Python stdlib CSV parsing and Poppler `pdftotext -layout`. Raw files are unchanged. This is a dataset audit, not a model experiment. No validation or leaderboard score is asserted.

## Task and input semantics

The local assignment describes SHOP-generated optimization decisions for the Tokke-Vinje system, 17 reservoirs, 9 plants, 14 generators. A daily case predicts 168 hourly decisions per generator, totaling 2,352 labels. Labels come from separate seven-day optimizations rather than one continuous realized schedule.

Source documents: `sintef_project/ML_Task_TDT4173.pdf` and `sintef_project/Dataset_definitions_and_explanation.pdf`. The latter explicitly specifies initial reservoir volume at case start and terminal water value at case start plus seven days. For a January 1 start, this means January 1 volume and January 8 water value. Hourly price and inflow throughout the seven-day horizon are supplied optimization inputs. Future historical reservoir volumes are not documented case inputs. Exclude `total_calculation_time`, which is a post-solve outcome and absent from prediction mapping.

## File inventory

| File | Data rows | Columns | First key | Last key |
|---|---:|---:|---|---|
| `sintef_project/data/extended/Constraint_min_flow.csv` | 96,432 | 7 | 2015-01-01 00:00:00+00:00 | 2025-12-31 23:00:00+00:00 |
| `sintef_project/data/extended/Constraint_min_volume.csv` | 96,432 | 4 | 2015-01-01 00:00:00+00:00 | 2025-12-31 23:00:00+00:00 |
| `sintef_project/data/kernel/Historical_day_ahead_price_2015_2025.csv` | 103,062 | 2 | 2015-01-01 00:00:00+00:00 | 2025-12-31 23:45:00+00:00 |
| `sintef_project/data/kernel/Historical_inflow_1958_2025.csv` | 594,224 | 19 | 1958-01-01 00:00:00+00:00 | 2025-10-15 07:00:00+00:00 |
| `sintef_project/data/kernel/Historical_volume_2015_2024.csv` | 3,654 | 18 | 2015-01-01 00:00:00+00:00 | 2025-01-01 00:00:00+00:00 |
| `sintef_project/data/kernel/Synthetic_water_value_2015_2024.csv` | 3,654 | 18 | 2015-01-01 00:00:00+00:00 | 2025-01-01 00:00:00+00:00 |
| `sintef_project/data/kernel/Unit_commitment_decisions.csv` | 2,922 | 2,355 | 1 | 2922 |
| `sintef_project/prediction_mapping.csv` | 725 | 2 | 2923 | 3647 |
| `sintef_project/sample_submission.csv` | 725 | 2,353 | 2923 | 3647 |

All inspected CSVs have consistent row widths, no empty/NA/NaN/null tokens, and no duplicate first-column keys. Inflows and constraints have contiguous hourly timestamps. Volume and water-value tables have contiguous daily timestamps. Price is hourly throughout the modeling interval but changes to 15-minute records beginning 2025-09-30 22:15 UTC; join by timestamps and do not assume the entire CSV is hourly. All timestamps are UTC.

Volumes for Hyljelihyl and Vatjern are constant 0.13 and 0.15 million m3 respectively, matching the documented assumption of half-full small reservoirs. Some recorded volumes exceed YAML maximum storage slightly; retain raw volumes and permit fill ratios above 1 instead of silently clipping them.

## Target schema and split

Training has IDs 1 through 2922, dates 2015-01-01 through 2022-12-31. Test has IDs 2923 through 3647, dates 2023-01-01 through 2024-12-25. The case date is exactly 2015-01-01 plus ID minus one days for every supplied ID. Training and test IDs are disjoint. Their time horizons overlap in the first six days of 2023 because training includes December 26-31 starts; these are legitimately supplied training cases, but should not be imitated by random split validation.

Submission IDs exactly match prediction mapping order. All 2,352 submission label names and their order exactly match the training labels after the first three metadata columns. Preserve the sample schema, including the space in `Vesle Kjela`. There are no constant label columns and all labels are exactly 0 or 1. Fifteen cases have identical decisions in every label: 13 all-on and two all-off. They are not necessarily corrupt; no evidence justifies dropping them. All-off IDs are 726 and 2742. All-on IDs are 1026, 1027, 1028, 1107, 1108, 1109, 1110, 1111, 1117, 1118, 1129, 1347, and 1596.

Generator order: Hogga_G1, Tokke_G1, Tokke_G2, Tokke_G3, Tokke_G4, Lio_G1, Byrte_G1, Vinje_G1, Vinje_G2, Vinje_G3, Haukeli_G1, Songa_G1, Kjela_G1, Vesle Kjela_G1. Each has hours t0 through t167 in numeric order.

## Chronological validation protocol

Use whole-case chronological folds, never randomly split hourly rows. For strict separation of target timestamps, keep training starts at most validation_start minus 7 days. Purge six immediately preceding case starts. Fit all transforms and models on the retained training cases. Predictions in validation must not use any validation commitment label, including from previous validation days. Use exact micro-average AUC over all unit-hour entries, plus per-unit and temporal diagnostics.

| Validation case starts | Validation IDs | Last retained training start | Last train ID | Purged IDs |
|---|---|---|---:|---|
| 2020-01-01 through 2020-12-31 | 1827-2192 | 2019-12-25 | 1820 | 1821-1826 |
| 2021-01-01 through 2021-12-31 | 2193-2557 | 2020-12-25 | 2186 | 2187-2192 |
| 2022-01-01 through 2022-12-31 | 2558-2922 | 2021-12-25 | 2551 | 2552-2557 |

A two-year final holdout starting 2021-01-01 better mimics the two-year test duration. Later validation should include 2022 because it is a large price-distribution shift. Final submission fitting may use all supplied training labels; the strict purge is a validation discipline rather than a requirement to discard authorized train data.

Adjacent cases share 144 timestamps, but their optimized decisions differ. Measured mismatch over those shared timestamps: Hogga 15.67%; Tokke G1/G2/G3/G4 15.13/18.23/17.31/17.87%; Lio 4.89%; Byrte 3.58%; Vinje G1/G2/G3 17.44/18.01/16.53%; Haukeli 12.46%; Songa 4.54%; Kjela 5.65%; Vesle Kjela 3.93%. Hence same-time copying is neither exact nor a trustworthy random-validation method.

## Modeling implications

- Price scale shifts sharply: annual mean EUR/MWh is 9.29 (2020), 75.11 (2021), 211.28 (2022), 79.44 (2023), 50.12 (2024). Use price-water-value margins/ratios, within-week rank, and relative price features alongside absolute values.
- Seven-day price ranking, neighboring-hour price, horizon position, and remaining inflow can model scheduling trade-offs. The whole weekly optimization horizon is known input, so this is not conventional one-step price forecasting.
- Storage fractions and headroom relative to expected weekly inflow can represent spill/availability effects. Reservoir links and production capacities offer further physical feature opportunities.
- Unit frequencies vary substantially across years. Haukeli's on-rate falls from 87.41% in 2015 to 14.15% in 2022; Songa is 70.68% in 2020 and 11.71% in 2022. Calendar-only extrapolation and untested historical means are weak.
- Multiple units at a plant have asymmetric on-rates; do not force all Tokke or Vinje units to have identical probabilities.
- Water-value columns share exact groups: {Kjelavatn, Staavatn}; {Langesae, Foersvatn, Bordalsvatn}; {Venemo, Vaamarvatn, Totak, Hyljelihyl}; {Bitdalsvatn, Songavatn}; {Langeidvatn, Vatjern}. Other water-value columns are distinct. Redundant columns can be removed in later ablations.
- Micro AUC compares predictions across units as well as within units. Per-unit model probability calibration and unit prior shifts can affect pooled score even when within-unit ranking is unchanged.

## Source SHA-256

### Water-value interpretation and physics features

The CSV units are EUR/MWh according to the local dataset definition. The official [SHOP water-value descriptions](https://docs.shop.sintef.energy/examples/water_values/water_values.html), consulted 2026-11-02, distinguish local `energy_value_input` in currency/MWh from global `water_value_input` in currency/Mm3. Local energy values are relative to the first downstream plant; SHOP converts them using initial-head best-point energy equivalents and then adds downstream global values. Thus subtracting the raw EUR/MWh values of upstream and downstream reservoirs would be dimensionally and semantically inappropriate. Mapping the provided CSV to `energy_value_input` is consistent with these units, but remains an inference because case-generation code is absent. The [official individual-water-value example](https://docs.shop.sintef.energy/examples/water_values/individual_water_values.html) also shows that reservoirs draining through the same plant share their local conversion factor.

`sintef_project/physics.py` is an optional feature ablation. It adds 72 features, eight per plant: initial gross head, grouped storage filling, remaining available-water/full-capacity ratio, upstream natural inflow/full-capacity ratio, required dispatch to avoid overflow, remaining profitable-hour fraction, and start/stop-cost-aware on-versus-off profit advantages at minimum and maximum representative-unit output. Its head-based energy equivalent is an explicit approximation that omits hydraulic losses and joint-unit efficiency. Storage availability omits inferred upstream production, and switching dynamic programming omits water budgets and cascade constraints. These features must earn inclusion through chronological validation.

Smoke verification on 2026-11-02: base feature shape is 612,696 by 231 float32 columns, approximately 566 MB; enriched shape is 612,696 by 303. All features are finite and the original input frame remains unchanged. Independent brute-force enumeration of every binary path for eight random seven-hour profit sequences matches switching DP advantages with asymmetric transition costs. An initial feature smoke failed because pandas 3 uses microsecond timestamp storage in this environment; explicitly converting start timestamps to nanoseconds fixed the issue. The subsequent base build completed in 1.81 s, and base plus physics completed in 2.20 s. These timings are checks, not model results.

### File digests

- `sintef_project/Dataset_definitions_and_explanation.pdf`: `256adf9a662446d391650dc1e8e764b819b38ddd435126b8c4ea93a613488cb8`
- `sintef_project/ML_Task_TDT4173.pdf`: `90a84ac45e16e1bc89f963d7846ca2a4e17c047247154e1f7f3947096e056804`
- `sintef_project/data/extended/Constraint_min_flow.csv`: `146062702f58920440fc064d461d544e9cd7c503c8df74ce96e721bee04629ea`
- `sintef_project/data/extended/Constraint_min_volume.csv`: `8b639b8339b84c5ff7db34ba4b4618e99d05a03819bf99fa5032a54490278806`
- `sintef_project/data/extended/Tokke_Vinje_topology.yaml`: `29cfa233cec7fc4b5401d0d571a566e6e17b7bb7b46a175abcd37a53fe175497`
- `sintef_project/data/extended/Tokke_topology.pdf`: `ea6e62174fd037f42d95bcfb16e0e5408d7866802d7a196e529aafcbc42bb54c`
- `sintef_project/data/kernel/Historical_day_ahead_price_2015_2025.csv`: `e5edd141b34a61139f3ee3022305175446d4b4f74618df85552c4f2b5e556ee4`
- `sintef_project/data/kernel/Historical_inflow_1958_2025.csv`: `ff4f87d2768133b4a717c0ad366969244f7a86901695e1d4fef3c2ed8963ad66`
- `sintef_project/data/kernel/Historical_volume_2015_2024.csv`: `ef3d9c69699eef0d953b495ba69bf6676858e6b9c536087498135074912dd3b6`
- `sintef_project/data/kernel/Synthetic_water_value_2015_2024.csv`: `a5f09ff96b551d3c297d90ccf3a0bbe92d8814b537ad63a46579cb0ff44aa4cb`
- `sintef_project/data/kernel/Unit_commitment_decisions.csv`: `6d51f8cc6869cb82b44708168808c305c1e007cf79878ca1c1d316d07aae3ab3`
- `sintef_project/prediction_mapping.csv`: `c8b0e5bc500a5f1dec2337f5ac680847f6a7794a7c2b7a8fb19384910a029519`
- `sintef_project/sample_submission.csv`: `83e72b9e7299aff61065911af67437c81e8be2b0491ff012761d897c334f1cba`
