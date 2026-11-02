# Case2012 numerical completion contract

Preparation only. No solve is authorized by creating this protocol or running the default preflight. Root must authorize execution after independent source/protocol review. The original four-worker v2 cache remains ineligible under its original contract; both case2012 attempt proofs remain failed and all original bytes remain untouched.

The exception applies only to case2012. Its native HiGHS status is zero, both native/conditional replicas agree, and all physical/KKT/zero-slack checks pass; the original fixed-state LP objective guard fails by approximately 1.68593e-6 in scaled units. The independently reviewed exact-rational certificate accounts for native feasibility, conditional residuals, marginal signs, stationarity and both reported-objective summation discrepancies. Its bound is approximately 3.990695556714721e-6 EUR, below the existing absolute objective-reconstruction budget of 1e-5 EUR. This describes the stored binary64 problem; it is not exact original decimal physics or an independent global MILP proof.

`scripts/complete_integer_case2012.py` reuses the unchanged original attempt runner, integer solver and conditional-LP capture. Two newly spawned processes each perform one 60-second MILP followed by the original two 1-second LP calls, using unchanged model, inputs, options and environment. A small observer records LP iteration counts separately and returns the identical SciPy result. No retry, alternative-optimum selection, tolerance adjustment or target decoding is allowed.

The fresh attempt proofs must also remain failed with the same objective guard and native optimum/bound. Every fresh native/LP NPZ file must have exactly the same SHA256 and array bytes as both original failed replicas. Fresh PIDs, timestamps, failed-proof hashes and observed iteration counts are retained. The exact certificate is recomputed from fresh saved operands, after explicit scale/dimension checks, and must equal the pinned original certificate and satisfy the exact rational 1/100000 EUR cap.

Physical result reconstruction makes no optimization call. It uses the saved LP solution and actual separately observed iteration count, then calls the unchanged original physical auditor. Only its specific objective-guard exception is permitted, after every preceding physical/KKT/native check; the original remaining objective/subset checks are applied explicitly. Native feasibility under the state-fixed bounds is also checked. Source/protocol hashes pin this ordering. Original and fresh failed proofs are never edited or recast as ordinary successful attempts.

Successful execution writes a separate `complete_case2012_exception` manifest and `exception_case2012_result.npz`. Its acceptance basis is `case2012_dual_feasibility_certificate`; it names the original failure lineage, new completion protocol/source/reviews and fresh evidence. It deliberately emits no ordinary `complete.json`, cache features or aggregate approval. A later independently reviewed new aggregate namespace/protocol/loader/auditor must explicitly account for this exception and preserve all ordinary/imported/canonical provenance. Existing v2 and frozen748 eligibility are not changed.

Prepare and check without solves:

```sh
.venv-ml/bin/python scripts/complete_integer_case2012.py --freeze-protocol
.venv-ml/bin/python scripts/complete_integer_case2012.py --protocol-sha256 ACTUAL_PREPARED_PROTOCOL_SHA
.venv-ml/bin/python -m unittest tests.test_integer_case2012_completion
```

An authorized future execution additionally requires `--execute --run-id UNIQUE_ID`, the external protocol SHA, `--certificate-audit/--certificate-audit-sha256`, and `--completion-audit/--completion-audit-sha256`. Both independent review manifests must pass and name the exact reviewed sources/protocol. No actual execution command with completed review pins exists at preparation time.
