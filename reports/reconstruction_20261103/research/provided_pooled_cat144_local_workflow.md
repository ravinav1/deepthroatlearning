# Supplied-input pooled CatBoost144 verification

This is a distinct learner experiment using the audited supplied-input gradient144 bundle. The numerical inputs, original targets, clean canonical115 anchor and same-feature LightGBM control contain no external numerical data. External historical research informed the upstream physical mechanism only. The feature family was chosen using the frozen2021 LightGBM blend score: gradient0.9953035361075371 exceeded busbar0.9952804039903538. That choice remains fixed despite the later gradient LightGBM2022 rejection.

The stage2021 protocol is `docs/provided_pooled_cat144_stage2021_protocol.json`, SHA256 `58678bb12af9c05014998ddabf93678dc54f11267aeacf385f55f73f4f637f33`. It pins the worker, typed categorical ingestion, exact bundle, runtime, source closures, clean anchor and matched control. The model is one1800-tree scalar CatBoost trajectory, depth8, learning rate0.05, Logloss, L2=5, border count128, Bayesian bootstrap temperature1, Plain boosting, random strength1, seed4173, GPU0 and one-hot maximum16. There is no early stopping or validation evaluation set. The unit and plant ID columns are integer categorical fields; the remaining142 columns retain their float32 bytes. CTR splits are prohibited.

The fixed2021 candidate grid is600/1200/1800 trees crossed with weights0/0.1/0.25/0.5/1. Ties choose fewer trees, then smaller weight. The winner must have positive weight and beat both the clean115 anchor and the same-weight raw gradient LightGBM blend. Both candidate and control receive the same fixed Vinje projection and exact unavailable-case anchor restoration after projection. A winning candidate that fails either gate is rejected; a different grid candidate is not substituted. The unchanged round count and weight then face the same two gates in2022. A separate immutable2022 stage protocol must bind the original2021 protocol, lock and independent selection audit.

The2021 training bundle has2,056,586 pooled rows and144 columns; prediction has858,480 rows. Its manifest SHA256 is `f370007c03a3600aeea6a31872ce7738dc2cacca863bb2b08bf803e54f8ad2c1`. The original40% hourly sample, RNG4173 sequence, six-start-date purge, case/hour/unit order and original scoped training targets are independently checked locally. The hosted bundle contains no validation labels. The checker reads only the original requested validation year after model verification passes.

The local preparation receipt is `artifacts/provided-pooled-cat144-local-preparation-001/manifest.json`, SHA256 `0ffbc9d9e5ac6887cb43d6432c9e1939825f6219d055cc298a99e61a87947de6`. It verifies the full actual bundle and five focused tests: categorical integer preservation/rejection, independent scalar one-hot JSON predictions and prefix shrink behavior, tied rank AUC, deterministic selection ties, and exact projection/fallback identity. One test fits a tiny synthetic CPU model. No competition-data fit or GPU fit occurs in this preparation; the receipt records that distinction explicitly. Physics independently reviewed all four local checker/selector/auditor files and found no blocker. Their exact source hashes are in the preparation receipt and remain frozen.

After downloading the original complete hosted return, use the original top-manifest SHA256 supplied by its download receipt. Substitute the actual return folder and hashes below; do not guess them.

```sh
.venv-ml/bin/python scripts/provided_pooled_cat144_verify.py \
  --protocol docs/provided_pooled_cat144_stage2021_protocol.json \
  --protocol-sha256 58678bb12af9c05014998ddabf93678dc54f11267aeacf385f55f73f4f637f33 \
  --fit-run RETURN_FOLDER --fit-manifest-sha256 RETURN_MANIFEST_SHA256 \
  --run-id provided-pooled-cat144-2021-return-audit-001

.venv-ml/bin/python scripts/provided_pooled_cat144_select.py \
  --protocol docs/provided_pooled_cat144_stage2021_protocol.json \
  --protocol-sha256 58678bb12af9c05014998ddabf93678dc54f11267aeacf385f55f73f4f637f33 \
  --fit-run RETURN_FOLDER --fit-manifest-sha256 RETURN_MANIFEST_SHA256 \
  --return-audit artifacts/provided-pooled-cat144-2021-return-audit-001/manifest.json \
  --return-audit-sha256 RETURN_AUDIT_SHA256 \
  --run-id provided-pooled-cat144-selection-2021-001

.venv-ml/bin/python scripts/provided_pooled_cat144_audit.py \
  --protocol docs/provided_pooled_cat144_stage2021_protocol.json \
  --protocol-sha256 58678bb12af9c05014998ddabf93678dc54f11267aeacf385f55f73f4f637f33 \
  --selection-run artifacts/provided-pooled-cat144-selection-2021-001 \
  --selection-manifest-sha256 SELECTION_MANIFEST_SHA256 \
  --run-id provided-pooled-cat144-selection-2021-audit-001
```

The return checker verifies all source/artifact/phase hashes, fresh ingest/fit/replay processes, actual model settings and categorical splits, semantic checkpoint prefixes, exact input bytes and original training target/sample chronology. It independently replays every native CBM over the complete retained local prediction bundle with2CPU threads. Full hosted CBM replay must be bit-exact; full hosted native JSON replay and local native replay must be within1e-12. Independent one-hot JSON probes include all14 unit identities and evenly spaced rows and must also be within1e-12. The local check does not claim a second physical solver implementation or a new fit. Native categorical Pool does not expose a float-only `get_features()` array; the checker validates exact numeric DataFrame bytes, exact integer categorical conversion, Pool dimensions/category indices/names and complete prediction replay instead.

The selection audit separately reconstructs every declared candidate and matched control with an explicit three-unit isotonic calculation, computes Mann–Whitney rank AUC, verifies exact selected bytes and original IDs/units/targets, and confirms both acceptance gates and frozen lock. No leaderboard score or final eligibility is inferred from a preparation receipt or incomplete hosted run.

The actual private GPU stage2021 completed in207.1s. Its original downloaded return is `artifacts/provided-pooled-cat144-2021-return-001`, manifest SHA256 `5f4ba487b469541e5c6182bae87326c702797df6e94c205184d2c2d731342b45`; archive SHA256 `3f4179a72aa3370626cb32a59f86da0f19aefd2015c140b018de71120e3e47d5`. The full return audit passed at `provided-pooled-cat144-2021-return-audit-001`, SHA256 `a179ff5161d3c64374cdba8c0711b5ab6e4f507dfbfabcde62d5012cd9e1607e`. The complete local CBM prediction replay differed by at most2.22e-16; independent JSON probes also passed the frozen1e-12 tolerance.

The fixed2021 grid produced:

| Trees | Weight0 | Weight0.1 | Weight0.25 | Weight0.5 | Weight1 |
|---:|---:|---:|---:|---:|---:|
|600|0.9952596524|0.9952815286|0.9952637536|0.9951286924|0.9943248682|
|1200|0.9952596524|0.9953104541|0.9953274074|0.9952478058|0.9945887548|
|1800|0.9952596524|0.9953201205|0.9953531266|0.9953039255|0.9947393234|

The selected1800-tree,0.25 blend scores0.9953531265542916, exceeding the clean115 anchor0.9952596524075921 and same-weight gradient LightGBM control0.9953035361075371. The selection manifest is `provided-pooled-cat144-selection-2021-001/manifest.json`, SHA256 `26355a8cf6fb5037bda1c10d35e5a9136a46ddc6aea98c55967b1f6de8d5e239`. Its `selection_2021.json` lock SHA256 is `fce9cc8fa2cf09f04a91d0e9d0d2778677387c60c8c9a6211162b45f9aff9807`. The independent rank/projection/full-grid/selected-byte audit passed at `provided-pooled-cat144-selection-2021-audit-001`, SHA256 `a6ed16b3792a05109fc39df45800416cad0337cd886e0f7fee8783a4dc80257e`. It is eligible only for the unchanged1800-tree/0.25 confirmation on2022 at this update.

Post-lock diagnostics are in `provided-pooled-cat144-2021-diagnostics-001`, SHA256 `17986e19adc01932e5a6d97a4a830fc5c70dfc9fcb08ea0bc3779d23dce3c82f`. Selected-minus-anchor per-unit AUC was+0.00188277 for Haukeli,+0.00031833 for Vinje_G3,+0.00014839 for Songa and+0.00007303 for Hogga. Lio declined0.00000448; Tokke_G1/G2/G3 and Vinje_G1/G2 also declined slightly. All14 global states and14/15 own fields appeared in splits. Native PredictionValuesChange totals were87.74% for original115 fields,8.08% for own15 and4.18% for global14. These descriptive findings did not alter the frozen shared weight or transfer gate.

The follow-on protocol is `docs/provided_pooled_cat144_stage2022_protocol.json`, SHA256 `b8e890b4f6e85c51e42ce1a4e04b0b61a5fa5d1dd87d0b4bfe7af3f75423567d`. Its independent preparation review passed at `provided-pooled-cat144-stage2022-preflight-001`, SHA256 `66dc3d9cd022229e09af0c59e1ee515bcce3b918447651f2270d4d71c2c34782`. It verifies the embedded original2021 protocol/lock/audit, unchanged source/model/feature/runtime settings, exact2022 bundle manifest `86b5e5626d1b24e89e81f56b560e061d232b12e122ca76381b122890ba54f7a6`, original RNG sample, all training target bytes through2021 and the purge. No2022 targets were decoded by that preflight. The stage trains only the locked1800 trees and uses the unchanged0.25 mixture.
