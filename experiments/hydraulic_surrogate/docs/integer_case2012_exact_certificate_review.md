# Independent review of the case2012 objective discrepancy

The saved fixed-state LP's reported objective exceeds the native MILP incumbent by **€0.0000038007611491910276**. An independent exact-rational calculation reproduces a one-sided upper bound of **€0.000003990695556714721**. This is below the certificate's €0.00001 comparison budget. It does **not** satisfy or replace the original frozen `1e-6` scaled non-worsening guard: both original attempts remain failed and ineligible.

The actual certificate manifest is `artifacts/integer-case2012-exact-certificate-001/manifest.json`, SHA `a6892f16ca3677f40f21ac24fd896b7202f5f5212fbd5b83488961c15c26916c`. Its exact source SHA is `57374b3bb6568d56ed5089a7483780ee23c9c86f893d9eb277709bcb6ef30fe9`. Independent review: `artifacts/integer-case2012-exact-certificate-independent-audit-001/manifest.json`, SHA `121b90dcf8394a9967aab609134b19cf29e8c99591d2fae4315a024317e7c926`, runtime4.877seconds. No optimization, target decoding, retry or eligibility change occurred.

For a minimization problem, let `n` be the native incumbent, `l` the fixed-state LP solution, and `d=l-n`. Let `y,z,a,b` denote equality, upper-inequality, lower-bound and upper-bound marginals. With the exact stationarity residual

```
r = c - Aeq.T*y - Aub.T*z - a - b,
```

the identity

```
c.T*d = y.T*(Aeq*l-Aeq*n) + z.T*(Aub*l-Aub*n)
        + a.T*d + b.T*d + r.T*d
```

holds exactly for the stored binary64 numbers interpreted as rational numbers. The reviewer computes sparse row residuals and separate transposed dual products, then checks this identity and every reported rational component independently.

Each equality contribution is bounded by the dual magnitude times the sum of the two residual magnitudes. Each inequality contributes the fixed solution's absolute complementarity term plus the positive part of the negative native residual–dual product. The latter includes either native infeasibility or an incorrectly signed marginal; the calculation does not assume that solver signs are exact. Lower and finite-upper bounds use the same decomposition. For an infinite upper bound, the whole `abs(b*d)` contribution is retained, without multiplying infinity by a nominally zero dual. The stationarity contribution is bounded by `sum(abs(r*d))`.

Finally, the absolute discrepancies between each reported objective and its exact rational dot product are added. Scaling uses the exact rational representation of the stored positive scale,2.2544047556529265. The resulting claim is `reported_LP - reported_native <= bound`; it is not a bound on the absolute distance between objectives. A genuine improvement may have a large negative difference.

The largest component comes from a native Lio maximum-storage inequality violation at horizon hour97: approximately1.418783313e-9 in volume units, paired with a scaled marginal near−1215.909141. The conditional LP remains feasible to the existing numerical tolerances, has zero physical slack and passes the full saved primal/dual certificate. The independent exact stationarity residual maximum is4.554294719e-10; a conventional floating dot-product calculation reports a smaller value because its summation cancels terms differently. The exact calculation includes this difference rather than assuming the floating residual is exact.

Both native incumbents and both conditional-LP replays are byte-identical across the two original replicas. Their native integrality discrepancy is2.220446049e-16; original primal residuals are below1e-5. Native status0/gap0 is preserved as solver evidence. The reported native dual bound exceeds its reported incumbent by2.328306437e-10 scaled units, another numerical reporting detail; this review does not convert that solver report into an independent exact global MILP proof.

Ten focused tests cover equality toys, lower and upper violations, incorrect marginal signs, nonzero marginals on infinite upper bounds, imperfect complementarity, both reported-objective errors, binary64 cancellation, the one-sided interpretation,128 randomized exact identities, and nonfinite operands. Run:

```sh
UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python -m unittest discover -s tests -p test_exact_integer_objective_bound.py
UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/audit_integer_exact_certificate.py
```

The audit command uses an immutable output ID and must be assigned a fresh output directory for another run. The original test-development artifact preserves one corrected fixture expectation: a positive upper marginal also incurred a wrong-sign term, so the implementation's bound6 was correct. The final complementarity fixture uses a correctly signed upper marginal and expects4. Certificate source and original case evidence were not modified.

The reusable `exact_bound` API assumes finite, correctly shaped operands and a positive scale; it does not explicitly validate every dimension or reject nonpositive scales. Those conditions are checked independently for this actual case and should be asserted by any new caller. The certificate concerns stored binary64 coefficients, not exact decimal physical inputs, exact primal/integer feasibility, or teacher-label fidelity. A separately authorized, frozen and reviewed completion would still be required before any new case artifact could become eligible.
