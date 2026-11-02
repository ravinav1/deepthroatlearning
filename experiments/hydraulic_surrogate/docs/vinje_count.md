# Vinje count classifier

Run `vinje-count-001`; status `complete`.

Joint prediction of Vinje's active-unit count shares hydraulic demand information and regularizes the weakest third generator while preserving soft probabilities.

Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/vinje_count.py --run-id vinje-count-001`

Predeclared weights[0,.25,.5,1] apply jointly to only3 Vinje units. Choose largest2021 pooled microAUC; exact ties prefer smaller weight. Persist choice before accessing2022 validation predictions/label slice. Apply that one weight unchanged on2022.

The target is the sum of3 true unit commitments. Cumulative count probabilities assume the usual G1>=G2>=G3 ordering, which rare observed labels violate. Report exact nonnested fractions per training sample and validation fold; do not relabel originals, force hard0/1 decisions, or assert exact nesting. Soft blending can preserve nonnested baseline probabilities.

|Fold|Baseline micro AUC|Frozen blend micro AUC|Delta|
|---|---:|---:|---:|
|2021|0.994370012291|0.994408340132|+0.000038327841|
|2022|0.993348542164|0.993382225812|+0.000033683647|

Frozen count-model weight: `0.5`. Both folds≥0.998: `False`. Independent2022 improvement: `True`.

The features, sampled row indices and purged splits are exactly those from normalized-LP v1. The model is multiclass LightGBM with count0..3,350 rounds,31 leaves, learning rate0.06, minimum100 rows/leaf,L2=1,column fraction0.9,maxbin255,seed4173 and2 threads. No early stopping. Other parameters are copied explicitly from the baseline and recorded in the manifest.

Runtime: 565.35s. Full source/data/model/sample/artifact hashes, complete row IDs, feature definitions, configuration and environment: `artifacts/vinje-count-001/manifest.json`.

2021 and2022 have both been reused for broader project development. Only2021 chooses this experiment's blend; the frozen2022 result is a transfer check, not a pristine final test. No Kaggle action or main-repository transfer is performed by this script.

## Fixed specialist/count composition

Run `vinje-count-combined-001`, status `complete`. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/vinje_count.py --combine --run-id vinje-count-combined-001`. No extra tuning; both original component weights remain0.5, followed by the fixed Vinje projection. The composition was proposed after both years' component results were known and is development evidence.

2021: baseline 0.994370012291; XGB standalone 0.994539323758; fixed combined+projection 0.994577404717.

2022: baseline 0.993348542164; XGB standalone 0.993423834939; fixed combined+projection 0.993458196827.

Full provenance and saved validation predictions: `artifacts/vinje-count-combined-001/manifest.json`. Runtime 13.77s. No Kaggle action or transfer.

## vinje-count-final-001

Status: complete. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/vinje_count.py --final --run-id vinje-count-final-001`. Runtime: 236.21s. Complete provenance: `artifacts/vinje-count-final-001/manifest.json`. Checkpoint prediction replay error: 0.0. Submission: `{'shape': [725, 2353], 'sha256': '3f80c8a42d41434f129e9b9bbfc1cf628c93039022673f56dd4ed72a328bc3ec', 'readback_max_abs_error': 5.000000413701855e-10, 'probability_range': [6.192407795424515e-07, 0.9999997615814209]}`. No Kaggle action or transfer.

## Combined final export and checkpoint replay

Run `vinje-count-combined-final-001`, status `complete`. Command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/vinje_count.py --combined-final --run-id vinje-count-combined-final-001`. No fitting. Reconstructed the shared feature table once, replayed final count and all three XGB checkpoints, verified equality with saved component probabilities, then merged the frozen weights and applied the fixed Vinje projection. Count replay maximum error 0.0; XGB replay maximum error 0.0.

Submission `{'shape': [725, 2353], 'sha256': 'e769f2c77336ad958f14c6c38b18248707c55b248453d9012cf76fbc07ea4039', 'readback_max_abs_error': 5.000000413701855e-10, 'probability_range': [6.192407795424515e-07, 0.9999997615814209]}`. Runtime 9.71s. Complete recipe, checkpoints and hashes: `artifacts/vinje-count-combined-final-001/manifest.json`. No Kaggle action or transfer.
