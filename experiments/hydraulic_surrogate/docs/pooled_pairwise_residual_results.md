# Pooled ranking residual: rejected on the frozen early screen

The fixed pairwise residual did not improve the single608 LightGBM anchor on the2020 forward screen. All three advancement checks failed. No2021/2022 continuation, final training or submission is warranted for this recipe.

Run `pooled-pairwise-residual-screen-001` completed in647.214s with peakRSS7,194,230,784bytes (6.70GiB), using two local threads. This is a mechanism screen on a single350-tree608-feature LightGBM anchor; these scores are not those of the accepted GPU ensemble and are not Kaggle scores.

| Fixed arm | Projected microAUC | Delta vs identity | First half | Second half | Raw BCE |
|---|---:|---:|---:|---:|---:|
| Identity | 0.985796093727 | 0 | 0.988060653540 | 0.982866422044 | 0.181718901700 |
| Pooled pairwise | 0.985650887428 | −0.000145206299 | 0.987825747318 | 0.982804210097 | 0.174438327138 |
| Ordinary BCE | 0.984827902524 | −0.000968191202 | 0.987319833209 | 0.981569709373 | 0.167513411180 |
| Class-balanced BCE | 0.985146260211 | −0.000649833516 | 0.987446778603 | 0.982286834595 | 0.171284275932 |

The screen contains360 cases dated2020-01-01 through2020-12-25:182 in January–June and178 in July–December. Training uses724 genuine forward OOF cases in2018/2019, with no same-case unary labels used to fit their base predictor. The full14 OOF cache consists of33 new models plus9 preserved Vinje models; all42 checkpoints and all three input feature arrays were independently replayed/rebuilt exactly.

The pairwise arm beats both learned controls but loses the identity in both calendar halves. Its relative reduction of `1−AUC` is−1.022298%, so ranking error increased. The separate5% substantial-gain flag is false; even the less restrictive positive-gain advancement gate fails.

## Interpretation and limits

All learned arms reduce raw BCE while worsening pooled ranking. The pairwise arm also worsens13 of14 within-unit AUCs, with the largest losses at Haukeli (−0.003431668), Lio (−0.000666832) and Hogga (−0.000644406). This is broader than an isolated cross-unit offset problem. Vesle Kjela is the sole within-unit pairwise improvement; it does not justify a posthoc unit splice.

Pairwise residuals have mean0.272875 and standard deviation0.379789 on the screen. Ordinary and balanced BCE residuals are much more extreme (standard deviations8.313 and43.917; maxima50.107 and369.532). The ordinary BCE optimizer uses its exact diagonal curvature, whereas pairwise training uses the fixed2×diagonal PSD majorizer. Thus this experiment compares complete fixed optimization recipes, not purely the effect of the loss function under identical Newton dynamics. The class-balanced control handles the training-class weighting difference but does not remove that optimizer difference.

These findings do not rule out ranking objectives generally. They reject this frozen80-tree residual trained on the early single-model OOF cache. No shift, temperature, clipping, tree-count, weight or unit-selection sweep was run after seeing the screen. The accepted GPU models were not altered.

The next distinct evidence should come from the already frozen sample-size and physical-feature experiments: fullsample608 and the integer water-budget cache. This result supports prioritizing new input information and stronger base fits over repeating generic residual correction on the same representation. Any renewed residual hypothesis needs a new, explicit protocol and an honest account of this reused screen.

## Frozen recipe and provenance

Features are608 existing case-hour inputs,14 generator indicators and the matching raw base OOF logit. All interior probabilities retain exact float64 logits; only exact0/1 use1e-12 endpoint substitutes. Zero residual returns the original bytes. The same final sigmoid and Vinje projection are used in every arm; there is no calibration or blend search.

Each arm uses80 LightGBM rounds, maximum depth4,16 leaves, learning rate0.03, L2=1, L1=0, minimum leaf100, minimum Hessian0.001, column fraction0.9, maxbin255, seed4173 and two threads. Pairwise training draws1,000,000 independent uniform positive-negative endpoint pairs per round using `SeedSequence([4173, round_index])`, scale `N/(2M)` and the prespecified2×diagonal Hessian majorizer. The balanced BCE weights are0.8710264503 for positives and1.1738064087 for negatives, based only on training counts977495/725353.

All three model/trace files were sealed before screen inference. All screen predictions and exact saved-checkpoint replays were sealed before target decoding at2026-11-02T18:43:28.901377Z. Independent result review replayed all three checkpoints, verified every one of the80 pair draw hashes, original2020 target alignment, generic isotonic projection, rank-sum AUCs, both half-year gates and sealing order.

- Frozen protocol: `docs/pooled_pairwise_residual_protocol_v2.json`, SHA256 `dc01041ec5c239d683c9f7f2d9d5efe36ba19b6fa47e9311bf695442c85405a1`.
- Completed run: `artifacts/pooled-pairwise-residual-screen-001/manifest.json`, SHA256 `9bca57e1d51c9501a023825235b3075ecd922734118156a9ed962ec7eb95cd43`.
- Independent result audit: `artifacts/pooled-pairwise-residual-result-independent-audit-001/manifest.json`, SHA256 `6365be7f04a36062f84fc2e462f18bac5039cb0681cc8fcf595c5db6d53d55fa`.
- OOF cache: `artifacts/full14-early-oof-001/manifest.json`, SHA256 `1016522707ec517f9e86dbabfe75686cf9b8d963aade721483d2f49bf0586146`.
- Independent42-model OOF audit: `artifacts/full14-early-oof-result-independent-audit-001/manifest.json`, SHA256 `6e8bd061d8e00361f3593f32b5ca666b168242becbc479ca044f3a66bb645ed4`.
- Model lock SHA256 `773c5bfd90cd2754c57505ee987cdc0e3fef6f3b37a12bbfdeba21504104bf69`; prediction lock SHA256 `d4f6ea248606b1cfd752e46814b0e6cb58ef76902048b844b6d0bcfda23d6453`.

Environment: macOS27.0.1 arm64, Python3.11.15, NumPy2.4.6, pandas3.0.6, SciPy1.17.1, scikit-learn1.9.1 and LightGBM4.7.0. Full source snapshots, package versions, case IDs, input hashes, objective traces, model files and predictions are retained with the run.

```sh
UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/screen_pooled_pairwise_residual.py \
  --run-id pooled-pairwise-residual-screen-001 \
  --protocol-sha256 dc01041ec5c239d683c9f7f2d9d5efe36ba19b6fa47e9311bf695442c85405a1 \
  --oof-manifest-sha256 1016522707ec517f9e86dbabfe75686cf9b8d963aade721483d2f49bf0586146 \
  --oof-audit artifacts/full14-early-oof-result-independent-audit-001/manifest.json \
  --oof-audit-sha256 6e8bd061d8e00361f3593f32b5ca666b168242becbc479ca044f3a66bb645ed4
```

The command records the completed immutable run. Reproduction requires a fresh run ID; do not overwrite its artifacts.
