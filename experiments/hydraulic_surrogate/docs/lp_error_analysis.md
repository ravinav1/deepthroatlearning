# LP error analysis and remaining model errors

The strongest next step beyond the directional Haukeli correction is **hourly LP temporal context, including switching costs applied to LP water opportunity costs**. The residuals do not support a general whole-week state classifier as the first next experiment. This recommendation is exploratory: both chronological years have already been reused for development.

The current normalized-LP model scores0.994370012 on2021 and0.993348542 on2022. Neither reaches the revised transfer requirement of at least0.998 on both years. These are local validation scores; no leaderboard submission or rank is reported here.

## Exact competition-error concentration

For every positive score, count higher negative scores plus half of equal scores; for every negative, count lower positives plus half ties. Assign half of each pair's error to each endpoint and divide by the total positive-negative pair count. The generator contributions then sum exactly to `1−micro AUC`. This measures impact on the pooled competition objective, unlike ranking generators by their individual AUC alone.

|Year|Unit|Share of all micro-AUC error|Share of its error from all-on/off cases|Case-mean share of Brier error|Within-week AUC|
|---|---|---:|---:|---:|---:|
|2021|Haukeli|17.45%|21.74%|39.13%|0.84717|
|2021|Hogga|13.51%|11.88%|19.25%|0.93895|
|2021|Lio|10.10%|10.87%|28.34%|0.97836|
|2021|Vinje G3|10.11%|3.68%|10.53%|0.93601|
|2022|Lio|16.20%|27.10%|45.54%|0.96268|
|2022|Haukeli|11.98%|13.81%|34.77%|0.90346|
|2022|Hogga|10.87%|4.28%|17.72%|0.95100|
|2022|Vinje G3|9.40%|1.02%|10.65%|0.94753|

Haukeli, Hogga and Lio together explain41.07% of2021 and39.06% of2022 pair error under this symmetric attribution. Within-week AUC compares only positive-negative pairs in the same case and generator, weighted by their pair counts; it is not the competition score. Case-mean Brier attribution uses the exact identity `mean((y−p)²)=mean(y−p)²+mean(((y−p)−mean(y−p))²)`.

Most errors lie in mixed weeks. That fact alone would not prove a switching problem, because a mixed week can still have a large mean bias. The independent within-week ranking and Brier decomposition support the same direction: hourly allocation contributes more than mean-level error for all three units in both years. Lio2022 is the strongest case-level exception, with four all-on and three all-off cases assigned the wrong majority state; its case-mean error still accounts for less than half of Brier. Hogga has no such constant-case majority failures in either year.

A bounded followup is to add a dynamic-programming switching feature on LP duals. For a generator and hour, use an on-state reward `max over legal q≥Qmin of [price×P(q)−(lambda_up−lambda_down)×0.0036q]`, an off-state reward0, and the supplied start/stop costs. Compute the best objective with the hour forced on minus the best objective with it forced off. Preserve the original nonlinear minimum-flow point when creating the legal operating grid. Multiplying `(price−dual/E_best)` byPmin assumes best efficiency at minimum output; that approximation is particularly poor for Haukeli. This DP omits joint water and environmental constraints and should be an input feature, not a replacement schedule or fitted calibration. SHOP's documented production curve and startup treatment support this distinction. [SHOP PQ curves](https://docs.shop.sintef.energy/examples/pq_curves/pq_curves.html), [SHOP plant attributes](https://docs.shop.sintef.energy/objects/plant/plant.html).

The environment agent's LP temporal-context experiment is already the appropriate direction. Avoid simultaneously enlarging every model or fitting a global case-state correction. The separate Vinje count probe is also motivated by the very small constant-case contribution and substantial within-week G3 errors.

## What the input-only LP gets wrong

The LP discharge proxy is `p=(q/Qmin)/(1+q/Qmin)`. Its0.5 cutoff corresponds toq=Qmin and is only diagnostic; it is not calibrated commitment probability.

|Unit|2021 LP AUC|2022 LP AUC|2021 false-off hours withq=0 / total|2022 false-off hours withq=0 / total|
|---|---:|---:|---:|---:|
|Haukeli|0.857379|0.711749|6205/6206|4966/4966|
|Lio|0.950702|0.917091|356/11332|816/14615|
|Hogga|0.932122|0.944159|2310/2326|1516/1539|

Lio's missed hours are predominantly positive but subminimum LP discharge:10976 hours in2021 and13799 in2022. The LP meets continuous water/flow constraints without the discrete minimum operating flow and startup decisions represented by commitment labels. This is a separate mechanism from Haukeli's zero-discharge failures.

Hogga false-off hours atq=0 have mean price versus LP opportunity cost of64.49 versus67.91 EUR/MWh in2021, and153.74 versus163.39 in2022. Short negative-margin periods are a reasonable target for switching-aware features, but their benefit has not been measured by this analysis.

## Haukeli directional storage and counterexamples

The YAML has a one-way `Langeidvatn → r_Langeidvatn → Vatjern → Haukeli` path. Langeidvatn has32.1 Mm³ capacity; Vatjern only0.3. Freely combining them in a32.4 Mm³ pool permits natural inflow arriving at Vatjern to be retained as if it could move uphill. The supplied initial Vatjern volume is0.15 Mm³ throughout these folds. A conservative legal-input lower bound on water that must leave Vatjern during the week is `max(0, .0036×sum(hourly Vatjern inflow)−.15)`; this is required outgoing water, not necessarily turbine generation.

44.87% of2021 Haukeli zero-flow false-off hours and67.20% in2022 occur in cases with a positive direct-intake release lower bound. Example2022 run2690 has0.326136 Mm³ excess natural intake inflow,103 true on-hours and only7 LP hours aboveQmin. Run2698 has0.632172 Mm³ excess,94 true on-hours and4 LP on-hours. The independent audit agent's isolated directional-pool benchmark reduced false-off hours96→53 and90→8 respectively, while adding11 and2 false-on hours. The benchmark was diagnostic and selected using known errors; it is not a general validation improvement. Its complete record is `artifacts/lp-v2-benchmark-001/manifest.json`.

The correction is incomplete. Dry2022 runs2653,2655 and2656 are all-on for168 hours despite zero LP discharge. Run2653 has direct Vatjern inflow0.01 m³/s, initial Langeidvatn volume2.614 Mm³, prices162.61–275.91 EUR/MWh, terminal local water value330.13 and mean LP opportunity cost303.64. There is no direct-intake overflow lower bound. The directional benchmark leaves these dry cases unchanged. Likewise2021 run2404 has151 true on-hours, both LP and new LP-feature model predict no on-hours, and the conservative direct-intake release bound is zero.

Haukeli's initial head is exactly266.5 m and its reconstructed best-point factor604.8359369 MWh/Mm³ in both years. Their drift therefore cannot explain the2022 decline. The LP already includes physical energy conversion and downstream terminal opportunities; no missing scalar energy multiplier was established. Official SHOP documentation states that an unspecified production factor is derived from best-point operation at initial head. The absent case-generation commands leave the exact input-attribute mapping and commitment conventions uncertain; these contradictions do not justify silently changing local/global water-value semantics. [SHOP production factor](https://docs.shop.sintef.energy/objects/plant/plant.html#prod-factor).

The directional river also has a supplied flow-cost curve omitted by the free pool. SHOP documents it as a piecewise constant marginal flow-cost schedule, so it should not be treated as a total-cost interpolation. [SHOP river flow costs](https://docs.shop.sintef.energy/objects/river/river.html#flow-cost-curve).

## Reproduction and limits

All paths below are under `artifacts/lp-error-analysis-001/`:

- Initial literal-input and frozen-LP audit: `manifest.json`, `all_case_summary.csv`, `selected_cases.csv`, `selected_hourly.csv`;13.99s. Exact command: `.venv-ml/bin/python scripts/lp_error_analysis.py --run-id lp-error-analysis-001`.
- New LP-feature model2021 case inspection: `lp2021/`;13.31s. Command: `.venv-ml/bin/python scripts/lp_error_analysis.py --run-id lp-error-analysis-001/lp2021 --ml-run artifacts/lgb-normalized-lp-001 --years 2021`.
- Exact pooled residual decomposition2021: `residual-2021/`;10.85s. Command: `.venv-ml/bin/python scripts/lp_error_analysis.py --residual-only --run-id lp-error-analysis-001/residual-2021 --years 2021`.
- Exact pooled residual decomposition2022: `residual-2022/`;0.68s. Command: `.venv-ml/bin/python scripts/lp_error_analysis.py --residual-only --run-id lp-error-analysis-001/residual-2022 --years 2022`.

The full manifests preserve source snapshots/hashes, raw input hashes, model/LP provenance, every inspected case archive hash, exact chronological split IDs and dates, original training settings, OS/Python/package versions, hardware, commands and runtime. The raw-input audit covers730 validation cases and three generators. It selects up to three cases per model/year/unit/error direction by incorrect probability mass, with seven-day separation, and exports literal hourly evidence. Separate residual tables cover all14 generators.

Checks reconstruct every saved LP q/Qmin and dual-margin feature, preserve identical case/unit/label alignment, and verify all cache archive hashes. Exact micro-pair error sums match `1−micro AUC` within1e−13; the Brier decomposition matches within1e−12. Script syntax compilation passed. No model fit, LP solve, future observed volume, hidden test label or Kaggle call was used by these analyses. The requested GPT-6 Pro consultation remains unavailable because its model identity could not be verified; see `docs/consultation.md`. No result is attributed to Pro.

A read-only exploratory preflight failed before calculating diagnostics because pandas3 returned microsecond timestamps while the probe assumed nanoseconds; the corrected script explicitly uses `DatetimeIndex.as_unit('ns')`. The failure is recorded in the initial manifest; it changed no inputs or model artifacts. Its original wall-clock start timestamp was not captured and is not invented.
