# Provided-input busbar144 candidate

The fixed0.1 busbar144 blend passed both reused chronological development folds:2021 microAUC0.9952804039903538 and2022 microAUC0.9940585327322639. The corresponding clean canonical115 anchor scores are0.9952596524075921 and0.9940509264719491. These are small validation gains; no Kaggle improvement is claimed before an actual submission result. The final fit `artifacts/provided-busbar144-final-001` completed in823.88s, after the independent2022 result audit passed. Its CSV is independently audited and ready for submission.

All numerical inputs and training labels are competition-provided. External publications and historical simulator research suggested the aggregate conduit-loss formulation, but no external simulator outputs, labels, teachers or numerical covariates enter the runtime cache, training or inference. The original clean115 anchor lineage is covered by `artifacts/provided-only-lineage-audit-002/manifest.json`, SHA256 `6c017919db2e6879e718398bb0aec7fc59f79b73b68a439a73a9750885b61f92`. Its complete data-to-anchor recipe remains in `docs/best_provided_only_result.md`; the account's external0.99952 result remains separate.

The local physical cache is `artifacts/physics-provided-busbar-cache-001`, manifest SHA256 `9c266f71bad082e8a69986f4eac7a05e68be260266444f7c352a596afbc68fde`. It solves all3647 cases and has zero fallback. Recipe `docs/physics_provided_busbar_cache_recipe.json` has SHA256 `1f202848bd62297d4c4a305e8daf81c63172779cd560c089655dfcf96fb44de0`. It replaces the Tokke/Vinje individual net curves with supplied-derived gross curves and eight aggregate convex cubic conduit-loss intervals, using capacity-average loss slopes for negative prices and conservative gross-unit ratings. Initial heads and terminal water values retain the original provided-input calculation. This is an approximate linear model, not an exact reconstruction of a proprietary simulator: constant group efficiency, finite upper chords, conservative gross capacity and negative-price approximation remain explicit limitations. Net fractional minimum-power deficits are recorded.

The complete cache audit is `artifacts/provided-busbar-full-cache-audit-001/manifest.json`, SHA256 `1f8a253ac2cb092792cd39e757c9d1f8239ab97c42f7795f0e82174d8911c005`. It verifies every case/source/input/artifact hash, all15 feature formulas and aggregate bytes, original case/date/unit ordering, water balances and conduit loss allocation. It relies on the separately recorded producer solver replay, and does not claim a second physical solver implementation.

The learner preserves the exact115-column canonical feature prefix and appends15 own-unit fields and all14 contemporaneous physical commitment states. The own fields are state, state change from the original LP, discharge divided by original minimum and gross maximum discharge, net power divided by nominal minimum and gross maximum power, gross power and allocated conduit loss divided by gross maximum power, discharge change divided by original maximum discharge, normalized gross-curve water margin, upstream/downstream final filling, Tokke/Vinje state sums and availability. All fields are computed from supplied inputs/local solves. Failed physical cases would restore the exact clean anchor after blending and projection; no case failed in this completed cache.

The model is pooled LightGBM binary GBDT with1400 trees,31 leaves, learning rate0.06, minimum100 child samples, minimum child weight0.001, L2=1, L1=0, column fraction0.9, max_bin255, subsample1, deterministic column-wise construction, seed4173 and four threads. It has no early stopping. Unit/plant identities are categorical columns111/112. The original40% hourly sample and original pooled row order are unchanged. Train/validation IDs are1–2186/2193–2557 for2021, then1–2551/2558–2922 for2022, with six case-start dates purged. Final training uses caseIDs1–2922,196358 sampled hours×14units=2749012 pooled rows; prediction caseIDs2923–3647.

The2021 grid `[0,.1,.25,.5,1]` selected0.1, and2022 used it unchanged. Full grid, descriptive per-unit results and feature importances are in `docs/provided_physics144_results.md`. Independent result audits passed at `provided-busbar144-2021-result-audit-001` (SHA256 `44ebde3483814b1e28d62fe47d02fcf80824eba5c74ea26cc78aae41d78b3041`) and `provided-busbar144-2022-result-audit-001` (SHA256 `f7bdac212aa5493777177ec5261f30096fee5f75d395ea471effa8ff7944004b`). They independently recompute rank AUC and projection, check original scoped targets and exact selected bytes, and bind the source/cache/anchor/locks. Per-unit diagnostics never changed the shared weight.

The selected recipe is `project_vinje(0.9 * clean_canonical115_anchor + 0.1 * raw_busbar144)`, followed by exact anchor restoration on unavailable cases. Preserve the canonical anchor's complete historical nested composition. The final native checkpoint is `canonical144_31leaf_1400.txt`; raw and selected arrays are `raw_predictions.npy` and `test_predictions.npy`. The final CSV uses the original725×2353 schema and nine-significant-digit probabilities in[0,1]. The inference check requires the reproduced selected arrays and actual CSV to match the sealed final bytes exactly, with CSV readback error at most5.000001e-10.

The exact final fitting command is:

```sh
.venv-ml/bin/python scripts/train_provided_busbar144.py \
  --year final --run-id provided-busbar144-final-001 \
  --selection-lock artifacts/provided-busbar144-2022-001/selection_2022.json \
  --lineage-audit artifacts/provided-only-lineage-audit-002/manifest.json \
  --lineage-audit-sha256 6c017919db2e6879e718398bb0aec7fc59f79b73b68a439a73a9750885b61f92 \
  --physics-cache artifacts/physics-provided-busbar-cache-001 \
  --physics-cache-sha256 9c266f71bad082e8a69986f4eac7a05e68be260266444f7c352a596afbc68fde \
  --physics-audit artifacts/provided-busbar-full-cache-audit-001/manifest.json \
  --physics-audit-sha256 1f8a253ac2cb092792cd39e757c9d1f8239ab97c42f7795f0e82174d8911c005
```

Use a fresh run ID for any reproduction. The final manifest SHA256 is `b33d298ca36c6df1cdfbd4e0dfec6f1f2ecd33e391b9273374871fa62cf37433`; checkpoint SHA256 `d6a18ee017605166ced74f81031fe2eaa98281c0e8160b5cdb0492fbcfd69819`; raw prediction SHA256 `59812fac1458f1278b95bddeabf395a60c6aa3828e982f4a383242148415af5b`; selected prediction SHA256 `9d0baba4008a700ce84990c005b9439295f997149c9f04c19e2a92497b2915ca`. The actual `submission.csv` SHA256 is **`b989424b21b5deaf13ffab4e969484b40bb8e464678d6cbb43b644c29f87fac1`**.

Independent final result/CSV audit `artifacts/provided-busbar144-final-result-audit-001/manifest.json` passed, SHA256 `d40027161310b16956f8c4c8b48218d37daf5346d635264fd5e530c6c382b0fa`. It checks original schema/IDs/range/readback, full selected bytes, exact actual CSV, original source/cache/anchor/lock bindings and native model dimensions. The producer also passed complete native checkpoint replay.

`scripts/metric_trick_busbar144_infer.py` provides standalone inference from the sealed native checkpoint, pinned canonical115 prefix and fully verified physical cache; it does not refit or decode targets. The frozen trainer and cache helpers are unchanged. Physics independently reviewed the new inference script's mapping, flattening, fallback and lock semantics and found no blocker. The following completed inference rebuilt all physical feature additions, compared every final144 feature block against the sealed matrix, replayed every native prediction bit-exact and reproduced the same selected array and actual CSV bytes:

```sh
.venv-ml/bin/python scripts/metric_trick_busbar144_infer.py \
  --final-run artifacts/provided-busbar144-final-001 \
  --final-manifest-sha256 b33d298ca36c6df1cdfbd4e0dfec6f1f2ecd33e391b9273374871fa62cf37433 \
  --run-id provided-busbar144-final-inference-001
```

The inference receipt passed, SHA256 `35f1ec6d1986f44dd3eeba5bb08c17463bcdc648572836c5c95298438b8b9ad9`. Its source snapshot and independently reproduced CSV are preserved in that run folder. It does not claim to rebuild the upstream115-feature prefix from raw inputs or to solve the physical network again; those pinned upstream caches and their audited provenance remain part of the recipe. No numerical external data enters this pipeline. A separate Kaggle observation will be recorded only after root verifies the actual uploaded result.
