# LP temporal context — predeclared bounded ablation

Prepared2026-11-02. Hypothesis: the frozen hydraulicLP omits start/stop and integer unit-commitment costs; neighboringLP discharge can expose whether current-hour activity is an isolated spike or a persistent operating block. A supervised model may learn a correction more easily from explicit neighborhood summaries. No improvement is claimed before chronological validation.

The module `sintef_project/lp_context.py` appends **126 dimensionless float32 inputs**: nine for each of the14 generators. It does not modify the frozenLP implementation, solve anLP, fit a model, read labels, access external files, or change existing columns.

For each unit, let q[h] be the existing `lp_qmin_ratio_<unit>` feature, for h=0..167:

1. q[h-1],q[h+1],q[h-3],q[h+3], clamping the index inside the same case.
2. Mean of up to6 preceding hours, excludingcurrent; mean of up to6 following hours, excludingcurrent. Divide by the actual number of available hours. At an empty boundary use currentq.
3. Whole168-hour meanq, fraction(q>=1), fraction(q>0), repeated across that case.

The strict comparisons are part of the declared recipe: q==1 counts toward the first fraction; arbitrarily small positiveq counts toward the second. No tolerance, clipping, binarized modelprediction, or ground-truthcommitment is introduced. These fractions are LP input summaries rather than commitment-label estimates or asserted feasibility rules.

Build the features on completecase-major168-hour blocks **after `add_lp_features` verifies its cache** and before any training-row sampling. Columns start `lp_context_` and therefore survive the existing monetary normalization unchanged. The function accepts root only for interface compatibility and performs no filesystem reads. The trainer should retain the upstreamLP specification and archive/hash this newmodule as usual.

Initialstorage/end-minus-current storage features are deliberately deferred: currentLP storagefill already exists, and adding another featurefamily would obscure the contribution of discharge context. This experiment is exactly126 columns, with no additionalstorage columns.

## Integration

```python
x, lp_spec = add_lp_features(x, root)
x, lp_context_spec = add_lp_context_features(x, root)
x, normalized_spec = add_normalized_features(x, root)
```

Runner integration and model fitting are owned by the rootagent. No training was launched for this module. Compare against the existing normalized-LP model with exactly the same seeds, sampledtrainingrows, treecount and chronologicalpurge before changing modelcapacity. The current migration requirement is at least0.998 on both chronologicalyears; the Kaggle objective remains at least0.999 with a verifiedtop3rank.

## Verification and provenance

Five tests pass via `.venv-ml/bin/python -m unittest tests.test_lp_context -v`: lag/lead and mean boundarysemantics; no crosscase leakage or inputmutation; exactthresholds using all168hours; completeorderedfiniteinput and duplicatefeatureguards; preservation through monetarynormalization. The existing completedLP cache manifest contains all14 requiredinputcolumns.

- Newmodule SHA256:`44b329a78ec9271d00dbd7e6c57469c31d01bc307d9b45a176768ea8583fd291`.
- Testmodule SHA256:`821b2297f8d0561a7ea8cb9541e78c845296f893aa6efc4e165811acdaa74a72`.
- FrozenLP module SHA256:`bd16e0b4e9e32976b7eb50d5334e05b94f2b074bc0d8eb83a787f26f95268720` (read-only; unchangedbythiswork).
- Current basecache manifest:`artifacts/lp-full-001/manifest.json`, SHA256:`0c52829ff84f464cbf3ed00bcd7746e3e5e8b373ea21134dea0f50e61814b9d1`.
- Basecache features SHA256 asrecorded:`0ebba7d10ffb29294e36e5e8742c73165e202ac1d96e6d15fbef7ea9d7cca7b9`.

The cache/source/data integrity checks remain the responsibility of `add_lp_features`; this temporaltransform deliberately consumes the alreadyverified inputframe. Any newcache version must be documented as a separate upstreamchange.
