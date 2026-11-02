# Past reservoir state audit and bounded feature family

The supplied histories support strictly past-only covariates for every 2021/2022 validation case and every test case. This is a plausible way to expose recent reservoir behavior, **not evidence that SHOP used historical operation to initialize generator status**. No target history, solver runtime, learned prediction, or future realized storage is used.

## Supplied semantics and legitimacy

The local [dataset definitions](../sintef_project/Dataset_definitions_and_explanation.pdf) describe reservoir storage at the start of each UTC day, in million m3, and inflow into each reservoir at each UTC hour, in m3/s. They explicitly identify V(t) as the initial reservoir volume for a scheduling case starting at t. The [assignment](../sintef_project/ML_Task_TDT4173.pdf) permits storage/inflow time series as model inputs and supplies their history; it does not state that only the start-date row may be used as an ML covariate. Treating supplied observations before t as available covariates is consistent with those semantics. Past-history use is an inference from this input description, not an explicit organizer statement about lagged features or SHOP initialization.

The documents contain no measurement publication-delay specification. This audit uses the dataset's timestamp semantics. It never uses V after t or an inflow hour starting at/after t. The YAML contains generator startup/shutdown costs but no initial on/off or initial production field was found. The independent rolling SHOP cases are not a continuous realized commitment schedule, and historical storage does not reveal their omitted initial on/off state.

Inflow is called reservoir inflow in the supplied definition. The balance proxy treats that series as the supplied external inflow term; it does not claim that every upstream transfer/release is included or that the resulting balance is a physical generation measurement.

## Measured availability

Run `past-state-audit-001` read the two raw histories and case metadata only, finishing in 0.91 seconds. It did not load commitment labels or construct the 612,696-row model matrix. Manifest, source snapshots, data/environment hashes, and case-level availability are in `artifacts/past-state-audit-001/`. The audit retained only the 87,672 inflow hours relevant to the requested past windows while scanning the supplied CSV in bounded chunks.

Volume spans 2015-01-01 through 2025-01-01 (3,654 daily rows); inflow spans 1958-01-01 through 2025-10-15 07:00. At each lookback, completeness requires both exact volume endpoints and every expected hourly inflow value, separately for every reservoir.

| Cases | Count | Volume: 1 day | Volume: 3 days | Volume: 7 days | Inflow: all three windows |
|---|---:|---:|---:|---:|---:|
| Training, 2015–2022 | 2,922 | 2,921 | 2,919 | 2,915 | 2,922 |
| Validation, 2021 | 365 | 365 | 365 | 365 | 365 |
| Validation, 2022 | 365 | 365 | 365 | 365 | 365 |
| Test, 2023–2024 | 725 | 725 | 725 | 725 | 725 |
| All supplied cases | 3,647 | 3,646 | 3,644 | 3,640 | 3,647 |

The only missing volume histories are IDs 1, 1–3, and 1–7 for the one-, three-, and seven-day windows, respectively. Every reservoir shares these same coverage counts in the actual files. No missing timestamp is filled from a later date.

Hyljelihyl and Vatjern storage is constant at 0.13 and 0.15 million m3. The definition explicitly says these values were assumed half-full because historical records were unavailable. Exclude these two reservoirs from the proposed initial-state ablation. Their apparent balance residual would merely re-express inflow while pretending to observe constant storage.

## Frozen proposed family

Use these 15 reservoirs in the supplied volume-column order: Kjelavatn, Staavatn, Langesae, Foersvatn, Venemo, Botnedalsvatn, Bitdalsvatn, Vaamarvatn, Langeidvatn, Totak, Songavatn, Vinjevatn, Bandak, Bordalsvatn, Byrtevatn. Use only d = 1, 3, 7 days and the supplied YAML `max_vol` capacity C. There are five fields per reservoir/lookback, or **225 added columns**:

| Field suffix | Formula at case start t |
|---|---|
| `delta_fill` | [V(t) − V(t−d)] / C |
| `inflow_capacity` | 0.0036 × sum of q(s) for hourly s in [t−d, t), divided by C |
| `withdrawal_proxy` | `inflow_capacity` − `delta_fill` |
| `volume_missing` | 1 if either exact storage endpoint is absent/nonfinite; otherwise 0 |
| `inflow_missing` | 1 if any of the expected 24d hourly inflows is absent/nonfinite; otherwise 0 |

Names are `past_state_<field>_<d>d_<reservoir>`. All are dimensionless finite float32. The volume delta and its missing flag are computed independently of inflow completeness. Missing volume makes its delta zero; missing inflow makes its cumulative term zero; either makes the withdrawal proxy zero. The corresponding flags explicitly distinguish these fallbacks from measured zero changes. No partial-window sum, interpolation, as-of match, forward fill, or backward fill is used. Signed values are retained, with no clipping or conversion to binary operating states.

The pure API is:

```python
case_features, spec = build_past_state_features(
    case_starts,                 # timezone-aware DatetimeIndex, UTC midnight
    volume,                     # daily indexed frame, million m3
    inflow,                     # hourly indexed frame, m3/s
    capacities,                 # reservoir -> supplied YAML max_vol
    reservoirs=the_15_names,
    lookback_days=(1, 3, 7),
)
```

Implementation: `sintef_project/past_state_features.py`. It has no filesystem access or label/model arguments, preserves case order, and emits one row per case. The caller must archive source/data/capacity hashes and repeat each case's vector only across that case's 168 hours before taking the saved training-row sample. It must not re-evaluate history relative to each forecast hour, which would introduce realized observations after case start. No frozen baseline feature module was edited.

## Interpretation limits demonstrated by input data

The residual is signed net withdrawal unexplained by supplied inflow/storage change. It can also reflect upstream controlled inflow, inter-reservoir transfers, bypasses, spill, storage measurement errors, or simplifying assumptions. It is neither generator discharge nor production, and it cannot uniquely identify on/off state.

Among 3,646 cases with valid one-day history, the residual is negative for 40.78% at Vinjevatn, 39.17% at Bandak, and 35.38% at Kjelavatn. These are input-only measurements from the audit, not performance evidence. They directly discourage a hard conversion from this residual to commitment or a claim of reconstructed generation.

## Boundary verification

Seven `unittest` tests passed in 0.028 seconds:

```sh
.venv-ml/bin/python -m unittest discover -s tests -p 'test_past_state_features.py'
```

They cover formula units and input case order; invariance when storage after t and inflow at/after t are changed or removed; early missing storage with flags and zero fallback; missing hourly inflow rejecting a partial sum; absent current storage not being replaced with a future value; nonfinite histories and source immutability; and invalid timezone/duplicate timestamps/capacities/windows.

## Next matched specialist ablation

The accepted next experiment retains all **395 normalized LP-v1 baseline features** and appends exactly the above **225 history fields**, giving **620 input columns** for Haukeli_G1, Lio_G1, and Hogga_G1. It changes feature availability, unlike the previous locality experiment which removed baseline columns. Retain the exact baseline purged case IDs, exact saved sampled training rows, 350-tree LightGBM settings and seed; use two threads. No full-data specialist refit is selected by this protocol.

Compare simultaneous replacement weights `[0, 0.25, 0.5, 1]` on these three units only in the full 14-unit baseline matrix. Select by 2021 pooled micro AUC, with ties favoring lower replacement weight; write the lock before reading 2022 labels. Then fit/evaluate the second fold with the same frozen choice; the runner evaluates only baseline and locked-weight matrices in 2022. Report matched baseline, raw specialists, and selected replacement; do not retune using 2022. Both folds have been reused for research. The features are a hypothesis about additional input information and do not assert how SHOP set initial operation. The parent authorized fitting once CatBoost completed; the runner verifies that gate and uses two threads alongside the joint-LP branch.

The runner's local read wrapper supplies the unchanged shared feature builder with all case metadata, target rows only through 2021, and discarded zero sentinels for 2022. Therefore it computes every input row without decoding 2022 target values. A dedicated test checks the two permitted CSV read calls and confirms a synthetic non-label marker in the held-out row is never decoded. That test initially exposed a temporary-directory symlink path comparison; resolving both compared paths fixed the guard, and the test then passed. No training run occurred before that fix.

```sh
.venv-ml/bin/python scripts/lp_past_state_specialists.py --run-id lp-past-state-specialists-001
```

The runner archives all source files, frozen baseline manifest, exact sampled row indices, raw specialist probabilities, selected full-matrix probabilities, checkpoints, feature specifications, and environment/data hashes. It checks the original 395 column names/specifications before appending history. Added history is computed in float64 and emitted as float32; no new normalization fitting or baseline changes occur.
