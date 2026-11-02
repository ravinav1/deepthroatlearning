# Linear-leaf residual screen: rejected

The fixed Pro-inspired linear-leaf experiment completed all28 models and checkpoint replays in13.531s, with peakRSS760,414,208bytes (0.71GiB). It failed every advancement gate. No2021/2022 continuation, final fit or submission will use this recipe.

| Fixed arm | Projected2020 microAUC | First half | Second half | Raw BCE |
|---|---:|---:|---:|---:|
| Identity | 0.985796093727 | 0.988060653540 | 0.982866422044 | 0.181718901700 |
| Constant leaves | 0.984691039679 | 0.985717337010 | 0.983086021326 | 0.188183012680 |
| Linear leaves | 0.932972740391 | 0.937236581565 | 0.925729827998 | 0.707932452687 |

The linear residual's standard deviation is6.0999 logit units, with a maximum80.3847. The largest within-unit deterioration is Kjela, whose AUC falls by0.36045. All14 within-unit AUCs worsen for both learned arms. Extreme linear extrapolation is a plausible explanation for the linear arm's failure; the saved output distributions establish the instability, not its unique cause. The experiment was not stopped by its operational resource limits.

This is a reused2020 development screen on the single350-tree608-feature LightGBM anchor, not a score for the accepted GPU ensemble. The previous pairwise experiment already exposed these screen labels. The new model recipe was frozen before fitting, with no feedback-driven feature, weight or hyperparameter search afterward.

## Fixed design

Each generator receives32 numeric features selected by descending gain from its frozen2018 base checkpoint. That checkpoint was trained only through2017-12-25. Ties use original feature-column order. Both the raw feature name and LightGBM's serialized space-to-underscore name are recorded; numerical column order is unchanged.

Residual fitting uses2018/2019 forward OOF probabilities and labels,724 cases and121632 case-hour rows. One sorted seed4173 40% sample selects48652 hours, shared across every generator and both arms. Means and population standard deviations are fitted only on those sampled rows. Real zeros remain numeric; missing values remainNaN; all-missing columns and infinities are rejected.

For every unit, the matched constant and linear arms use40 rounds, depth3,8 leaves, learning rate0.05, minimum leaf512, L2=10, linear_lambda=10, maxbin255, all selected features and the same rows. The CPU serial tree learner uses two threads and deterministic seeds. The binary BCE gradient uses the fixed raw OOF logit as `init_score`, with an exact-diagonal Hessian plus the same negligible1e-12 floor in both arms. The first gradient is checked against that offset. Only `linear_tree` differs between the arms; actual serialized linear coefficients are verified.

Saved models predict residual logits. Inference adds the frozen base logit once, takes sigmoid and applies the existing Vinje projection once. Zero residual returns the raw base probabilities byte-for-byte. There is no calibration, clipping, blend grid or unit selection.

All28 model files, normalizers and traces were sealed before2020 input inference. All predictions and all28 exact checkpoint replays were sealed before decoding the selected2020 targets. The360 screen cases run throughDecember25; the two calendar halves contain182/178 cases. Advancement requires linear leaves to beat both identity and constant leaves overall and in each half. A5% reduction in `1−AUC` is a separate descriptive flag, not a veto. All three checks are false.

The operational budget is600wall seconds and4GiB process peakRSS, checked at stages and every boosting callback. Any limit failure is operationally inconclusive with partial evidence preserved, not a negative score. A failed pre-label budget check prevents target decoding; no adaptive reduced-budget retry is allowed. Native calls can return after crossing a limit, which is recorded at the next check.

## Consultation, preflight and artifacts

Root consulted the browser's visible Pro model and preserved the exact prompt, recommendation and decision. The exact generation is unverified in that UI record. Consultation manifest: `artifacts/pro-swarm-consultation-001/manifest.json`, SHA256 `94b4557edae207df2f189def950c624fe7e8a35b0351a1109e4e43408afe2478`.

The implementation follows LightGBM's documented branch-feature linear leaves, constant first tree, numeric scaling and missing-value requirements. The documented memory increase motivated the bounded screen. [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html#linear_tree), [piecewise linear tree paper](https://arxiv.org/abs/1802.05640).

- Preflight001 preserved a state-trace failure caused by LightGBM deep-copying a callable objective object. A function closure fixes trace ownership, matching the earlier audited residual runner. No competition fit occurred.
- Preflight002 preserved an incorrect assumption that raw feature names exactly equal LightGBM's serialized names. The exact space-to-underscore mapping is now checked and both names retained. No data or feature order changed, and no competition fit occurred.
- Preflight003 passed six targeted tests, including first-gradient offset integration, actual linear coefficients, checkpoint replay, sample/tie behavior, missing-value scaling, resource limits and future-target poisoning. Manifest SHA256 `12887922d06ae709262191b6c270ea2cd0103f055a2ce0702c7ed7a8631a3a4c`.
- Independent preflight passed after separately reconstructing all14 pre2018 feature selections and training normalizers. Proof SHA256 `ae4ad2d7b8cb67de22df2bcd255dfb269fba076aaa77b4224de02b6fd4c16473`.
- Frozen protocol `docs/linear_leaf_residual_protocol.json`, SHA256 `ad6197db7076990ff111d6bba68bbb596687eed93d028713f782aaec8da5e4db`.
- Completed run `artifacts/linear-leaf-residual-screen-001/manifest.json`, SHA256 `141204f193e03bd36fa36126aed4eb98d12d5f7a596b4d223cf5f8439cf59719`.
- Independent result audit replayed all28 models and verified normalizers, offsets, original target alignment, projected AUCs and rejection gates in2.37s: `artifacts/linear-leaf-residual-result-independent-audit-001/manifest.json`, SHA256 `c1cd3fb0f0e0904ec6cfab2b07725aed4f2a94bdcec0c48b2c5c1659ba6524bb`.
- Fit-lock SHA256 `fb35056aca7e79522b239ebd38a619b47509827ff5d9f1c8281f04728d48db63`; prediction-lock SHA256 `ae25120601b404f80d1586350b82d4aeb52ae83cebfa51b9673afda5a4d0e581`.

All source snapshots, package versions, input hashes, split IDs, selected columns, normalizers, model traces and arrays are in those immutable artifacts. The exact completed command is:

```sh
UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/screen_linear_leaf_residual.py \
  --run-id linear-leaf-residual-screen-001 \
  --protocol-sha256 ad6197db7076990ff111d6bba68bbb596687eed93d028713f782aaec8da5e4db \
  --preflight-audit artifacts/linear-leaf-residual-independent-audit-001/manifest.json \
  --preflight-audit-sha256 ae4ad2d7b8cb67de22df2bcd255dfb269fba076aaa77b4224de02b6fd4c16473
```

Reproduction needs a fresh run ID. Preserve the failed candidate without posthoc parameter rescue. The accepted GPU fits, fullsample confirmation and bounded rank-preserving offset hypothesis are separate branches.
