# Canonical screen execution revision

Unrun v1 source, protocol `c376b822d32bde6ba87340240c68d1c12abb82aa18e541599284d602e781c283`, and metadata preflight `5778317b178fd5e3afeed2c942cb5f600b2d09237fe7bbd5517c4e5d84731730` remain unchanged. Independent source review found that the interval covering final pooled prediction replay and scoring was outside the observed resource checks.

V2 adds a recorded resource check immediately before the complete prediction seal and first 2020 label read, and another after scoring before marking the result complete. Failure of either check preserves the failed run and prevents acceptance; failure of the first prevents screen-label decoding. This is an execution-only revision. The 115 features, sample, chronological populations, ordinary BCE models, 700/1,400 checkpoints, matched 350-round controls, blend grid, tie order and six comparison gates are unchanged. Manifest hashing and final evidence I/O occur after the last eligibility check and are reported separately through the total runtime; the checks do not claim operating-system hard limits.

Use `scripts/prepare_canonical_pooled_screen_v2.py` and `scripts/screen_canonical_pooled_v2.py` with `docs/canonical_pooled_screen_protocol_v2.json`. The full source/protocol/preflight must receive independent approval before any real feature build or fit.
