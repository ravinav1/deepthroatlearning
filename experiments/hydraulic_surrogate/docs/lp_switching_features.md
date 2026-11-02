# LP dual switching features

`sintef_project/lp_switching.py` adds 28 dimensionless predictors, two for each of the 14 units, through `add_lp_switching_features(x, root, lp_prefix=None)`. It accepts LP v1 `lp_*` or directed Haukeli v2 `lp2_*`; explicitly select a prefix if both are present. It can run before normalization using raw price or afterward using `monetary_scale_inverse`. No new LP solves, training, labels or fitted statistics are involved.

For each unit, recover the hourly economic margin from the LP export:

`margin_EUR/MWh = lp_economic_margin * (1 + mean(abs(case price)))`.

For nominal YAML power Pmin and Pmax separately, hourly operating profit is this margin times power. The existing independently tested binary switching DP computes the best horizon objective conditional on being on at each hour minus the best objective conditional on being off. Initial and terminal status are free. Asymmetric startup/shutdown costs are the unit's supplied constants. Divide the resulting objective difference by nominal power and the case price scale to obtain `lpdp_switch_margin_min_<unit>` and `lpdp_switch_margin_max_<unit>`.

These prefixes are already dimensionless and are preserved by the existing normalizer. A full v2 physical feature table has 385 columns after monetary normalization and 413 after these additions. The complete-case smoke took 1.596 s to add all28 columns; provenance and exact column names are in `artifacts/lp-v2-full-001-loader-verification.json`.

The LP opportunity cost uses best-point energy conversion, `(lambda_up-lambda_down)/best_energy`. Applying it at nominal Pmin and Pmax is a constant-conversion approximation: actual efficiency, case-feasible maximum power and shared hydraulic losses differ. The original water shadow prices are held fixed when the DP changes state; water budgets, changing reservoir levels and downstream dispatch are not reoptimized. Accordingly, these are auxiliary temporal predictors, not feasible hydraulic schedules or a reconstruction of SHOP's unknown command settings. YAML monetary switching costs are assumed consistent with the dataset's EUR price units; original case-generation currency-conversion commands are unavailable. SHOP's linear-startup setting can also alter whether such costs apply. [SHOP startup command](https://docs.shop.sintef.energy/commands/set_linear_startup.html)

Four tests pass. A hand-computed isolated profitable hour verifies the correct sign before/after scaling while startup costs stay fixed; zero transition cost recovers the input hourly margin; labels and runtime metadata leave the features unchanged; normalized/raw price interfaces agree; and coexisting LP versions require explicit selection. The transform leaves its input frame unchanged. The returned specification records topology, source and DP-source SHA256, all unit costs/powers, formulas and limitations.

```sh
.venv-ml/bin/python -m unittest discover -s tests -p 'test_lp_switching.py' -v
```

No supervised model or Kaggle score has been measured for these additional features. Keep the matched v1/v2 comparison separate before combining this hypothesis with the selected LP version.
