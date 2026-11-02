# Independent LGB726 result audit

Preparation only: no actual model protocol, candidate fit, feature cache or result is consumed by the preparation tests. Source `scripts/audit_lgb_river726_result.py` SHA256 `d8b1407ed24b3b6c5a9e8a1845a0ec7c7188532045784457553f9633b57c9081`; source-preparation manifest `artifacts/lgb-river726-result-auditor-preflight-001/manifest.json` SHA256 `4a1d70ac0a42ef0261bcdd65d87ca22cbd4fa1693ecc987d37115c9d39b81c8a`. Four synthetic lock/tie/control/prefix rejection tests passed. Independent source review is still required.

The auditor accepts only externally pinned complete candidate, prospective model protocol and actual training preflight audit. It requires the complete independently audited 3647-case physical cache before decoding feature arrays or labels. It independently checks the raw LGB664 controls against their actual checkpoint/target audit proofs; selected blends cannot substitute for those controls. The public-best SHOP anchor and original350-tree parameters remain unchanged.

It recreates the exact six-day-purged seed4173 samples, verifies original case/hour/unit identities and labels, rebuilds664 then appends62, and hashes every original column's name/dtype/bytes before and after addition. It replays all fourteen native726-column/350-tree checkpoints over all61320 validation hours twice in float64, with byte-exact saved float32 agreement. Generic min-max isotonic projection and average-rank AUC reuse the byte-pinned independent664 audit helpers. It independently reconstructs all four2021 weights or only identity plus the audited fixed2022 weight. Eligibility requires strict gain over both saved SHOP identity and same-weight rawLGB664; lower-weight ties remain fixed. Both years are reused development data.

Run only after the candidate training process exits and releases the single large local slot. The fresh feature build also requires at least40% measured free headroom. No fitting, alternate weight, unit splice or public selection occurs in this audit. Every failure is retained.

Future2021 invocation, using actual externally supplied SHA256 values:

```sh
.venv-ml/bin/python scripts/audit_lgb_river726_result.py \
  --protocol docs/lgb_river726_protocol.json --protocol-sha256 ACTUAL_PROTOCOL_SHA \
  --year 2021 --candidate-run artifacts/lgb-river726-2021-001 \
  --candidate-manifest-sha256 ACTUAL_COMPLETE_CANDIDATE_SHA \
  --training-audit ACTUAL_TRAINING_PREFLIGHT_AUDIT --training-audit-sha256 ACTUAL_PREFLIGHT_SHA \
  --run-id lgb-river726-2021-independent-audit-001
```

The2022 invocation additionally requires the exact2021 `--selection-lock/--selection-lock-sha256` and `--lock-audit/--lock-audit-sha256`; these gates precede2022 target reads. The result exposes `status`, `protocol_sha256`, `candidate_manifest_sha256`, `selection_lock_sha256`, `eligible_for_2022` or `accepted`, plus checkpoint/target/prefix/cache/control-proof fields. There is no automatic final training, export or submission.
