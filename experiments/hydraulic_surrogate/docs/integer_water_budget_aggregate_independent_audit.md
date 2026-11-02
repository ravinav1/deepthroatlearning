# Independent integer aggregate audit

The new `integer_water_budget_aggregate_v1` namespace accepts exactly 3,647 cases: 3,010 ordinary strict cases, 636 preserved compatible imports, and the sole separately reviewed case-2012 numerical exception. The original four-worker cache remains incomplete and ineligible. Its failed proofs, complete receipts, source and data are preserved unchanged.

`scripts/audit_integer_water_budget_aggregate.py` audits only a completed aggregate after the original coordinator has finalized and all its worker processes have exited. The externally pinned observation must identify the exact original path, final manifest and protocol, coordinator PID 11014, all worker PIDs derived from its attempt inventory, the declared scoped process query and its output, and a timestamp after final runtime. The auditor rejects a partial aggregate before reading its feature archive. It never optimizes or decodes targets.

The auditor verifies exact source/data/protocol and all artifact hashes, including complete copies of each original case directory. Every separate `aggregate_receipt.json` must identify its type, accepted attempt protocol, original complete receipt or explicit absence, physical input/control hashes, selected result/features and exception/import lineage. Its 3,010/636/1 groups are disjoint and exhaust the original metadata population.

For ordinary and imported cases, the existing `audit_integer_water_budget_cache.audit_case` is used unchanged. It reconstructs the frozen physical matrices, verifies both native incumbents and their solver bounds, all four conditional LP certificates, exact replica replay, the original objective guard, zero absolute slacks and independently computed 84 feature columns. The preserved 636 receipts must equal their original base-protocol receipts plus precisely the previously authorized compatibility transformation; their old original bytes and canonical diagnostic origins remain checked.

Case 2012 follows its declared exception only. The two newly completed attempts must retain failed status and exactly match both old failed replicas. The auditor independently rebuilds native feasibility and integrality checks, conditional LP KKT conditions and every component of the exact-rational one-sided objective bound. It requires the actual accepted completion and independent result audit, the unchanged €0.00001 certificate budget, and the continued violation of the original scaled `1e-6` guard. It independently decodes every physical result field and all 84 features. It does not rewrite a failed proof or claim that this case passed the old native strict gate.

All original inputs are reconstructed through metadata-only access. Case/date order, 168-hour alignment, unit/column order, float32 dtype and literal bytes of the full `(3647,168,84)` array are verified against the separate aggregate archive. Full acceptance exposes the new loader's required cache/protocol/features hashes, case counts, compatibility flag and explicit exception identity. It implies no learned-model or leaderboard improvement.

The frozen aggregate protocol is `aacf3ee1492fbb48f1d62827cfc7306bad20841c48088cdb1b68842b94fe0c48`. Actual invocation requires the completed aggregate manifest SHA and independent producer source audit:

```sh
UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/audit_integer_water_budget_aggregate.py \
  --cache artifacts/ACTUAL_AGGREGATE_RUN \
  --manifest-sha256 ACTUAL_FINAL_MANIFEST_SHA \
  --protocol-sha256 aacf3ee1492fbb48f1d62827cfc7306bad20841c48088cdb1b68842b94fe0c48 \
  --source-audit artifacts/ACTUAL_SOURCE_AUDIT/manifest.json \
  --source-audit-sha256 ACTUAL_SOURCE_AUDIT_SHA \
  --run-id integer-water-budget-aggregate-independent-audit-001
```

Preparation includes five synthetic provenance/path/exit rejection tests and no-solve golden checks against the already completed examples 1, 1000 and 2012. No actual full aggregate audit has yet run.

Independent source review passed: `artifacts/integer-aggregate-auditor-independent-source-review-002/manifest.json`, SHA `a3febff61ee79034e1ba9c48d3c3c0dea124ac731d87d1e60aa683a8a3e432d5`. The reviewer independently ran all five tests and completed examples 1, 1000 and 2012, including native/KKT/exact-rational/84-feature checks. Review 001 preserved a stale source-pin stop before any checks while preparation was still being strengthened; no producer or actual cache was changed.
