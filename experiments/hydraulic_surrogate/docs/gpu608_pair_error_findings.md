# Accepted GPU608 blend: exact ranking-error decomposition

The accepted fixed 50% GPU608 blend has chronological micro AUC **0.9950359172801482 in2021** and **0.9937529477072502 in2022**. This read-only analysis uses those unchanged saved predictions. Almost all erroneous positive–negative comparisons cross forecast cases, and roughly90–91% also cross generators. This supports testing a global ranking objective; it does not show that a simple calibration map will improve unseen data.

## What the micro metric compares

For every positive output, the metric compares its score to every negative output across all cases, hours and units. An error costs1 for reversed scores and0.5 for ties. The directed14×14 matrix has positive unit on rows and negative unit on columns. Counts are exact integer or half-integer sums. Each matrix sum reproduces1−micro AUC within1e−13; diagonal elements reproduce all14 within-unit AUCs. A small brute-force fixture separately verifies ties, empty within-case classes and same-case partitions.

“Same case” means the same forecast start and its168-hour schedule. Different forecast cases can have overlapping calendar dates. These are diagnostic partitions, not independence assumptions.

| Pair category |2021 share of errors|2022 share of errors|2021 share of all pairs|2022 share of all pairs|
|---|---:|---:|---:|---:|
|Same case, same unit|0.034%|0.026%|0.005%|0.007%|
|Same case, different unit|0.275%|0.349%|0.187%|0.186%|
|Different case, same unit|9.707%|8.808%|6.628%|6.693%|
|Different case, different unit|89.983%|90.817%|93.180%|93.114%|

Cross-case comparisons account for **99.690% / 99.625%** of errors. Their dominance partly reflects their **99.808% /99.807%** share of the pair population. Within-case pooled AUC is0.9919982441 /0.9878812287, while cross-case pooled AUC is0.9950417666 /0.9937643086. Improving a sequence model can still improve global comparisons; the observed pair mass is not a ceiling on sequence-feature gains.

Cross-unit comparisons contribute **90.258% / 91.165%** of errors. Strictly increasing per-unit maps cannot fix same-unit ordering, giving a loose necessary AUC ceiling of **0.9995164048 / 0.9994481026**. This assumes every cross-unit error can be corrected simultaneously and is not an attainable-score estimate.

## Which outputs contribute the errors

Each pair error is split equally between its positive and negative endpoints. The following percentages sum to100% within each year. Switch context is truth-defined and must never be used as an inference feature.

|Unit|2021 endpoint error share|2022 endpoint error share|2021 within-unit AUC|2022 within-unit AUC|2022 error near true switches|
|---|---:|---:|---:|---:|---:|
|Lio_G1|10.092%|16.820%|0.991304|0.973717|33.666%|
|Haukeli_G1|16.433%|12.607%|0.968574|0.962502|26.072%|
|Hogga_G1|13.793%|11.044%|0.980718|0.986199|45.045%|
|Vinje_G3|10.361%|9.104%|0.987611|0.985635|65.737%|
|Kjela_G1|5.574%|8.682%|0.994036|0.992761|42.722%|
|Vinje_G2|8.802%|7.321%|0.992124|0.992611|68.506%|
|Vinje_G1|6.674%|6.524%|0.995511|0.994868|70.911%|
|Vesle Kjela_G1|2.386%|5.502%|0.999559|0.996551|38.361%|
|Tokke_G2|5.262%|4.890%|0.997376|0.997303|65.442%|
|Tokke_G4|6.119%|4.614%|0.996416|0.997564|69.404%|
|Tokke_G1|4.649%|3.939%|0.997961|0.998325|79.732%|
|Tokke_G3|5.269%|3.799%|0.997351|0.998391|76.264%|
|Songa_G1|2.608%|3.152%|0.999433|0.996057|55.909%|
|Byrte_G1|1.977%|2.001%|0.999652|0.999563|62.328%|

In2022, Lio, Haukeli and Hogga together account for40.47% of endpoint error. Lio negatives alone occur in21.24% of all erroneous pairs. The largest directed2022 pair is Haukeli-positive versus Lio-negative at3.12%, followed by Kjela-positive versus Lio-negative at2.58%. In2021, Haukeli is the largest endpoint source at16.43%; its self-pair is the largest directed cell at3.04%. All196 cells for both years are retained in `directed_unit_pairs.csv`.

Mean prediction minus observed on-rate is stable and positive for Lio (+0.0311/+0.0444) and Hogga (+0.0178/+0.0179), but Haukeli changes sign (+0.0128/−0.0154). These observed shifts are descriptive evidence against assuming one universal calibration correction. They are not fitted offsets.

## Schedule context of erroneous endpoints

|Truth-defined context|2021 hours|2021 endpoint errors|2022 hours|2022 endpoint errors|
|---|---:|---:|---:|---:|
|Within3h of a true switch|10.489%|40.860%|15.123%|50.821%|
|Case-unit schedule with≥5 switches|25.577%|49.073%|37.162%|66.235%|
|Entire schedule off|25.049%|4.969%|29.765%|3.262%|
|Entire schedule on|21.683%|4.582%|9.276%|5.511%|

Episodes lasting≤12h account for6.52% of2021 hours and29.23% of errors; in2022 they account for10.60% of hours and39.27% of errors. Vinje G1/G2/G3 have70.91%/68.51%/65.74% of their2022 endpoint error within3h of a true switch. Haukeli and Lio have only26.07% and33.67% there. A transition-only correction would therefore address different portions of these units’ failures. Group masks overlap and should not be summed except within their explicit partitions.

## Implication for the next experiment

The rejected CRF improved proper sequence likelihood while reducing AUC. A new residual objective can target globally comparable scores across units and cases, using genuine earlier-year out-of-fold anchor predictions. It should retain a matched contextual unary control, include same-unit and cross-unit comparisons, and screen before2021. The diagnostic cannot select training weights, unit offsets or a leaderboard recipe. Both2021 and2022 have already been reused for development; no new fitted model, changed prediction, submission or untouched-test claim results from this analysis.

## Reproduction and provenance

Command: `UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/analyze_gpu608_pair_errors.py`. The run ID is immutable; use a distinct ID for any future rerun. Runtime2.294975s, peak RSS282,902,528bytes. Local Python/package/platform versions, exact source and accepted-array/manifest checksums are in the manifest. The original pair-calibration audit and latent residual diagnostic are recorded as methodological sources, not changed.

Manifest: `artifacts/gpu608-pair-error-diagnostic-001/manifest.json`, SHA256 `46acfb65fcd85bb2c9ff51ded375051575c5e2da5081ea48a2259a3eee9e7a9f`. Matrices and endpoint arrays are in `pair_errors_2021.npz` and `pair_errors_2022.npz`; CSVs preserve all unit pairs, all14 endpoint summaries and all context groups. No RNG or model training was used.

Independent result audit passed using average-rank Mann–Whitney arithmetic rather than the producer's searchsorted implementation. Global and same-case pooled/diagonal counts, all four partitions, endpoint totals, sklearn AUC and every source/input/artifact hash were verified. Proof: `artifacts/gpu608-pair-error-independent-audit-001/manifest.json`, SHA256 `9ffd361a2fa06b4ab08ab4995d051ba5d92b74c43386c5625b0fa2f71d4c9c97`.
