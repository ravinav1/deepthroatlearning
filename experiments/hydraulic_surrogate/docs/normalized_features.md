# Case-normalized economic features

`sintef_project/normalized.py` transforms the complete physical feature table without fitting any statistics or reading labels. The hypothesis is that relative economic values transfer better across the large electricity-price regime changes than raw EUR/MWh predictors. This hypothesis requires chronological validation; no improvement is claimed by the transform itself.

For each 168-hour case, define `s = 1 + mean(abs(hourly electricity price))`, in EUR/MWh. Replace each absolute monetary feature with `norm_<name> = original / s`. Replaced features are current and neighboring prices, daily and weekly price summaries, price-minus-water-value margins, terminal water values, and the physical model's switching-cost-adjusted economic margins. Their original unnormalized columns are removed.

Preserve storage, inflow, calendar and hydraulic features. Also preserve existing price/water ratios, within-week price rank and the existing price z-score. Add `monetary_scale_inverse = 1/s` so the model can represent fixed costs relative to price levels. This adds one net column: the current 303-column physical table becomes 304 columns.

The transform is deliberately not described as exact economic scale invariance. Startup and shutdown costs remain fixed in the upstream dynamic program; its resulting economic margin is normalized afterward. The denominator's additive 1 EUR/MWh stabilizes near-zero price cases. The retained price/water ratio and z-score similarly contain their existing additive stabilizers. Negative prices are handled by using absolute price to define a positive scale while preserving each predictor's sign.

The scale uses only the known full price horizon of the same optimization case. No cross-case statistic, historical target, validation label or post-solve metadata enters it. The input table remains unchanged. Exact formulas and replaced/retained column lists are returned as JSON-serializable metadata for the experiment manifest.

Verification: `.venv-ml/bin/python -m unittest discover -s tests -p 'test_normalized.py' -v`. Five tests pass, covering independent per-case scales, monetary replacement without duplication, preserved relative/nonmonetary predictors, label/metadata independence of derived features, case isolation, zero-price stability and invalid input rejection. Model comparison results belong in `docs/experiments.md`.
