# Historical solver research for supplied-only hypotheses

This is descriptive external research authorized by the user after requesting a genuinely supplied-only model. It is separate from the clean numerical lineage: no external solver output, fitted teacher, target, or feature is transferred into training or inference. Findings may motivate mechanisms evaluated separately using original competition labels.

Run `metric-trick-historical-mechanisms-001` compares only original cases1–2186, whose starts run from2015-01-01 through2020-12-25, preserving the six-day purge before2021. Each public file is read with `nrows=2186`; original target decoding has the same bound. Exact IDs, ordered columns, source CSV hashes, environment, formulas, command, and all diagnostics are in `artifacts/metric-trick-historical-mechanisms-001/manifest.json`, SHA256 `3bb1cf53e89d80dbcaea396e726a0ae8b37f9ccd35e77b96527d333be00c4410`. Runtime6.263s. No2021/2022 target rows, later public solver rows, or hidden targets were decoded. No rule was selected and no model was fitted.

The external source is the public CC BY4.0 [SINTEF solver data product](https://data.sintef.no/product/dp-107e1305-9802-4d21-9599-4554a1a2dd70), downloaded earlier into the recorded SHOP17.15.1 and17.17.0 archive directories. This analysis reads only `solver_committed` and `solver_production` plus ID/date/horizon metadata. A solver state is converted to a binary diagnostic at0.5. Original `result_committed` columns are the comparison labels.

| Public historical simulator | All-unit threshold disagreement | Lio disagreement | Entire cases matching all original labels |
|---|---:|---:|---:|
|17.17 envC All_MIP|1.5717%|0.8049%|47.3010%|
|17.17 envC No_MIP|3.5702%|8.2936%|7.8683%|
|17.17 standard All_MIP|3.7541%|17.9947%|5.0320%|
|17.15.1 All_MIP|4.6550%|18.6294%|3.4309%|

These are historical binary disagreements, not validation AUCs or leaderboard results. The fixed solver/version comparison shows environmental constraints are critical to the teacher’s Lio behavior. It does **not** establish that our current local model omits a constraint. Physics’s separate fixed36 comparison found only27/6,048 Lio threshold differences between our local linked-state model and matching envC No_MIP, with normalized-power MAE2.18%. That direct local-model comparison is stronger evidence about our implementation. Together, these observations support investigating integer commitment effects after matching the relaxed environmental model.

For envC All_MIP, the first-hour disagreement is2.101%, versus1.255% at the last hour. Only14.694% of all errors occur within one hour of an original switching event; errors are more frequent near switches, but most wrong states persist away from transitions. Fixed shifts from−3 to+3 hours show zero shift is best for Lio, Haukeli and Hogga. A global time-offset repair is unsupported by these historical diagnostics.

An oracle that chooses the best within-case permutation of identical-plant generator labels would remove2,418 of20,379 Tokke errors and150 of32,866 Vinje errors. It helps79/2,186 Tokke cases and13/2,186 Vinje cases. This oracle is explicitly non-deployable: it uses original labels only to bound the explanatory value of permutation. It gives no hidden-label-derived mapping and no clean predictor. In particular, Vinje label permutation explains very little of the residual gap.

The public relaxed variable is not a universal production-threshold proxy. For Lio No_MIP, hours with production between0.51 and0.75 times static minimum power are officially on82.0% of the time, while mean relaxed commitment is0.431. For Haukeli, the same production band is officially on only5.5% of the time, while mean relaxed commitment is0.243. These differences are consistent with unit-specific integer/startup/hydraulic economics. They do not justify fitting or transferring a threshold from external outputs. A corresponding clean experiment must derive its own features from supplied topology, inputs and local optimization, and select any predictor only against original chronological labels.

Reproduction uses a fresh artifact run ID:

```sh
.venv-ml/bin/python scripts/metric_trick_historical_mechanisms.py --run-id metric-trick-historical-mechanisms-001
```

The complete diagnostic includes per-unit and horizon-bin errors, false positives/negatives, case error quantiles, switching rates, all seven temporal offsets, within-plant permutation bounds, and fixed production/static-Pmin bins. No cached external numerical arrays are produced for downstream model consumption.
