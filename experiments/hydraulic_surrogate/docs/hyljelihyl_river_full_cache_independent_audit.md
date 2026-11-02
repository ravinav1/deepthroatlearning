# Independent audit of the complete river cache

`scripts/audit_hyljelihyl_river_full_cache.py` is prepared for the frozen v3 cache. It performs no solves, target decoding or training. It must run against an externally hashed **complete** cache manifest, after the two worker processes have exited. It rejects partial cache status before loading aggregate features.

The source pins its independent arithmetic helpers and the exact producer protocol. It checks the full source/data/input closure, immutable receipt paths and payload inventories, the imported 59 pilot cases, their pinned worker identities and the 36 canonical fresh-process cases. The v2 all-input preflight is an explicit v3 import: the earlier complete independent 3,647-case reconstruction, all prepared input bytes, and the blocked v2 launch review remain pinned. The producer's physical formulas and solver calls did not change in v3.

Each new phase must preserve the two declared case orders, distinct worker PIDs and exact input/protocol seals. Only prior resource-drain manifests can be imported. Their complete receipts remain byte-identical; prior stop markers are checked at their archived names. Any retained pair-failure record or failed case blocks acceptance. A resumed phase with missing worker evidence is rejected for separate review. Unpaired outputs from prior clean resource drains are retained in the artifact closure but are not used as features; the final selected two calls per case receive the complete arithmetic audit.

For all 7,294 selected saved calls, the auditor checks native status, saved native/primal-dual identity, primal feasibility, residual arrays, dual signs, stationarity, complementarity, infinite-bound marginals and objective consistency under the frozen `1e-5` numerical contract. It independently reconstructs every decoded physical field, including the average-energy treatment of negative prices, verifies absolute-zero slacks and projects the split solution back into the original continuous model's constraints. It independently computes all 62 feature formulas and compares their literal float32 bytes against both replicas and the corresponding aggregate row. Canonical physical outputs also match their original saved arrays exactly.

The actual worker-exit observation must contain `status="passed"`, `returncode` 0 or 1, exact `expected_worker_pids` for every phase, `scoped_process_output`, `timestamp_utc`, `cache_directory`, `cache_manifest_sha256` and `cache_protocol_sha256`. It must be captured after coordinator completion using a scoped process query. Its exact final-manifest binding and timestamp prevent reuse of an observation from another run. The frozen producer's joined-worker evidence is checked separately. No process signal is needed.

Example invocation after completion, substituting the actual manifest and observation hashes:

```sh
UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/audit_hyljelihyl_river_full_cache.py \
  --cache artifacts/ACTUAL_CACHE_RUN \
  --manifest-sha256 ACTUAL_COMPLETED_MANIFEST_SHA \
  --source-audit artifacts/hyljelihyl-river-full-cache-v3-independent-preflight-001/manifest.json \
  --source-audit-sha256 f2068d9835f78d5ed1fb88e332b523053d71e9f43ebc2c241ae153255a71f75b \
  --exit-observation artifacts/ACTUAL_EXIT_OBSERVATION/manifest.json \
  --exit-observation-sha256 ACTUAL_OBSERVATION_SHA \
  --run-id hyljelihyl-river-full-cache-independent-audit-001
```

A passing proof exposes the strict loader's required fields: `status`, `cache_manifest_sha256`, `cache_protocol_sha256`, `features_sha256`, `case_count=3647`, `all_cases_physical_and_replay_verified` and `canonical_reuse_verified`. It also records phase lineage, process exits and independent feature reconstruction. It establishes input-feature correctness under the frozen approximation; it does not establish a learned-model improvement or leaderboard result.

Independent source review passed before actual execution: `artifacts/hyljelihyl-river-full-cache-auditor-independent-source-review-001/manifest.json`, SHA `c8b4787f5a2880d720cba4aa5889473e08b6a51c0bcd40c645f896152f5e0ac3`. The reviewed auditor source SHA is `40947e594d8a89ec3a0396b691956cfe34ec4578547966a69d211740cb3ee955`. The reviewer reran five synthetic tests and four saved calls from completed pilot cases 1 and 2186, including full arithmetic and eight corruption rejections. No in-progress full-cache output or solver was used for this review.

The actual cache002 audit passed: `artifacts/hyljelihyl-river-full-cache-independent-audit-001/manifest.json`, SHA `2111ce26a75932f994819781e547e4c9f881acaea9cbfb8e37210ad5737f86fd`. All3,647 cases and7,294 saved calls passed in398.957seconds, with the full feature archive byte-exact. The auditor process exited0. This is the full-cache proof required by the strict726 loader.
