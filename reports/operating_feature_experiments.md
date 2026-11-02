# Operating feature experiments

Three changes tested separately on the existing 2021 and 2022 splits. Water and economics start from the 112-feature XGBoost price baseline; LightGBM price starts from the 98-feature topology model. Model hyperparameters are held fixed within each comparison. Six overlapping starts are purged before validation. No competition test cases enter fitting or validation.

## Overall micro ROC-AUC

| Model | 2021 | 2022 |
|---|---:|---:|
| XGBoost: price baseline | 0.965719 | 0.971633 |
| XGBoost: price + water | 0.967406 | 0.970877 |
| XGBoost: price + economics | 0.966477 | 0.971143 |
| LightGBM: original | 0.964473 | 0.968316 |
| LightGBM: price | 0.963674 | 0.970617 |
| Current 75/25 blend | 0.966169 | 0.971874 |
| 75/25 blend: both with price | 0.965816 | 0.972115 |

## Decision

No change improves overall AUC in both years. Keep the current blend as the baseline. Leave the new water/economic switches off in selected submission configs. LightGBM with price is promising on 2022 but worse on 2021, including in the 75/25 blend; it has not been promoted to replace the existing LightGBM component. No new submission was created. With no consistently better candidate, the conditional third-year confirmation has not been started.

## Difficult generators

| Model | Lio 2021 | Lio 2022 | Haukeli 2021 | Haukeli 2022 |
|---|---:|---:|---:|---:|
| XGBoost: price baseline | 0.906129 | 0.918593 | 0.922151 | 0.916646 |
| XGBoost: price + water | 0.917708 | 0.912948 | 0.930749 | 0.916000 |
| XGBoost: price + economics | 0.904911 | 0.930437 | 0.931110 | 0.916978 |
| LightGBM: original | 0.907843 | 0.887032 | 0.921031 | 0.907750 |
| LightGBM: price | 0.915322 | 0.901104 | 0.921207 | 0.905717 |

Water helps both difficult generators in 2021 but not 2022. Economics helps Haukeli in 2021 and Lio in 2022, but does not provide a repeatable overall improvement. Aggregate switch-neighbourhood AUC also varies by year; see the JSON comparison.

## Feature definitions

- Water (15 features): local natural inflow over the next 6/12/24 hours, relative to feeding-reservoir capacity, starting free capacity minus that inflow; upstream hourly, cumulative, remaining and next-window natural inflow. Upstream reservoirs include nearest feeders. These are natural-inflow summaries, not actual releases or arrival forecasts. The free-capacity reference does not track changing storage.
- Economics (8 features): start+stop cost divided by minimum or nominal MW, plus hypothetical continuous-run margin for each reference power over the next 6/12/24 hours. `reference MW * sum(price - local terminal water value) - start cost - stop cost`, in EUR. Actual initial unit state, dispatch, hydraulic losses, efficiency and cascade valuation are not reconstructed.
- Windows include the current hour and truncate at the supplied week boundary. They never cross into another case.
- Fixed costs are provided in YAML, not latent inputs. The economics feature always charges a hypothetical complete start/stop cycle, so it is an approximate reference rather than a hard decision rule.

## Verification and reproducibility

All six saved models were reloaded and checked against saved validation predictions. Feature-window calculations, economic units and preservation of original feature values were checked. Original models and submissions were preserved.

Run configs: `configs/xgboost_water_2021.json`, `configs/xgboost_water_2022.json`, `configs/xgboost_economics_2021.json`, `configs/xgboost_economics_2022.json`, `configs/lightgbm_price_2021.json`, `configs/lightgbm_price_2022.json`. Reproduce the comparison with `python src/compare_operating_experiments.py`.

All results are development scores: validation selects the best tree count, and these years have been consulted repeatedly. Within-generator AUC is diagnostic and does not average to overall micro AUC. Switch diagnostics use true labels within ±2 hours only for analysis, never as inputs.
