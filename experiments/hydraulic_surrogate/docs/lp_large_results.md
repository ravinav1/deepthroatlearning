# Normalized LP capacity and coverage ablation

The larger normalized-LP model does not justify a full-data fit. Its frozen 25% mixture improves the context/base prior that was current when this experiment was declared, but remains below the newer accepted joint blend on both chronological development years.

| Recipe | 2021 micro AUC | 2022 micro AUC |
|---|---:|---:|
| Matched LP350 raw | .9943700123 | .9933485422 |
| LP700 raw | .9941794369 | .9933340358 |
| Original context75/base25 prior | .9947811119 | .9934702459 |
| Locked 25% LP700 mixture | .9948175945 | .9935429174 |
| Newer accepted joint blend | .9948884918 | .9936419640 |

The single predeclared capacity/coverage configuration used all 395 frozen normalized-LP features, 700 trees, 63 leaves, learning rate .04, minimum child samples 100, training-row fraction .7, seed 4173 and two threads. It trained the same 14 generators across the same purged years, with no early stopping or final fit. `lgb-normalized-lp-large-001` completed in 6,632.98 seconds.

The original scalar grid was [0,.25,.5,.75,1] against context75/base25. Selection on 2021 locked .25 before decoding the 2022 predictions. Final raw source weights were base .1875, context .5625 and large .25, accumulated in float64 in that order, followed by fixed Vinje projection. The locked gain was +.0000364826 in 2021 and +.0000726714 in 2022 against its original prior. The comparison manifest records `accepted:true` relative to that original prior; this does not promote it above the newer joint recipe. No alternate 2022 weight or posthoc unit subset was chosen.

`artifacts/lp-large-blend-verification-001/manifest.json` records independent source/data/artifact checksum checks, exact reconstruction of both saved prediction arrays and independent AUC calculation. No checkpoint inference was repeated. Both years have been reused during development, and none of these scores reaches the .998 transfer threshold. No Kaggle submission or full-data large-model training was performed.

Reproduce training with the command in `docs/lp_large_protocol.md`; run the fixed comparator with `scripts/compare_lp_large.py` under a fresh run ID. Independent verification command: `.venv-ml/bin/python scripts/verify_lp_large_comparison.py` (the recorded verification ID already exists and is protected against overwrite).
