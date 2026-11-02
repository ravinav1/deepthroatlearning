# Optional physical features

`sintef_project/physics.py` adds 72 input-derived columns to the case-hour table. It is a feature-engineering approximation, not an exact SHOP simulator. Inclusion in the final model requires measured chronological validation improvement.

## Water values

The local `Dataset_definitions_and_explanation.pdf` states EUR/MWh and specifies the value at case start plus seven days. [SHOP's official water-value documentation](https://docs.shop.sintef.energy/examples/water_values/water_values.html), consulted 2026-11-02, defines `energy_value_input` as a local energy value relative to the downstream plant. Conversion to a global water value uses the plant energy equivalent and downstream global values. The [official example](https://docs.shop.sintef.energy/examples/water_values/individual_water_values.html) illustrates the distinction.

We interpret the provided EUR/MWh series as local energy values. The assignment does not provide the generating script, so this remains an explicit inference consistent with its units. We do not subtract EUR/MWh values across reservoirs or invent a cumulative value conversion. The economic margin for a plant is hourly electricity price minus the terminal local energy value of its feeding reservoir.

## Feature definitions

Each of the nine plants receives these eight columns:

| Feature | Calculation and limitation |
|---|---|
| Initial gross head | Interpolate initial storage on the YAML storage/head curve and subtract the greater of plant outlet elevation and downstream reservoir head. Average the two supply heads for Songa. |
| Pool filling | Sum initial storage divided by total maximum storage over upstream reservoirs connected without generation. |
| Remaining supply fraction | Initial usable pool storage plus remaining natural inflow, converted to energy, divided by remaining hours times total plant output capacity. Does not subtract unknown preceding dispatch or add unknown upstream production. |
| Catchment inflow fraction | Sum current natural inflows over the upstream reservoir graph divided by estimated full-plant discharge. Ignores river travel times and storage buffering. |
| Dispatch required to avoid overflow | Positive initial-storage-plus-weekly-inflow excess above total pool capacity, converted to energy, divided by a full week of plant output. |
| Profitable remaining fraction | Fraction of remaining hours with positive local price-water margin. |
| Switching advantage at minimum output | Optimal net profit forcing on at this hour minus forcing off, divided by representative-unit minimum output. |
| Switching advantage at maximum output | Same calculation at representative-unit maximum output. |

Energy equivalent is approximated by `2.725 * gross_head_m * peak_turbine_efficiency * peak_generator_efficiency` MWh/Mm3. Turbine efficiency is interpolated across the YAML reference heads; efficiencies are fractions. The constant follows from water density, gravitational acceleration and joule-to-MWh conversion. Hydraulic losses, unit interactions and optimal discharge-point differences are omitted.

The switching dynamic program uses hourly margin times fixed operating output and the YAML start/stop costs. Initial and terminal states are free because no initial commitment is supplied. It has two states per hour and includes both sides of each hour's optimal schedule. It omits storage constraints, cascades, minimum runtime and ramping. The result is a scheduling prior that a fitted model can adjust, not a feasible dispatch claim.

## Verification and reproducibility

`tests/test_physics.py` compares switching advantages against exhaustive binary-path optimization, including asymmetric and zero switch costs. It verifies case isolation, finite outputs, unchanged input frames, unused-label/metadata independence, input-order validation, and a known storage/head interpolation. Synthetic fixtures use schema and YAML constants without training targets.

Run `.venv-ml/bin/python -m unittest discover -s tests -p 'test_physics.py'`. `add_physics_features` returns the exact feature list, pool/catchment mapping, all representative-unit constants, source links and formulas as JSON-serializable metadata. The training manifest records that metadata, code snapshots and topology checksum. Measured model results belong in `docs/experiments.md` rather than this methods document.
