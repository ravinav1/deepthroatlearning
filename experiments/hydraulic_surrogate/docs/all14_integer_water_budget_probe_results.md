# All-unit binary feasibility results

The frozen two-case probe passed its feasibility gates in15.91seconds total. No commitment targets were read and no model was trained. Only the chosen binary unit-state index set changed from three units to all fourteen in the existing supplied-cost network model.

| Case/date | All14 solve+fixedLP seconds | Worker peak bytes | Proven native gap | Extra objective loss versus3unit, EUR |
|---|---:|---:|---:|---:|
|1 /2015-01-01|1.8683|318177280|0|516.35685894|
|1826 /2019-12-31|10.1831|488062976|0|415.54561591|

Both native MILPs returned optimal status. Every retained fixed-state LP field (solution, residuals, duals and status) replayed with exact dtype/shape/bytes. All physical slacks were exactly zero, all fourteen unit states were binary, and each objective stayed below the relaxed and proven three-unit optima within the existing1e-4EUR subset tolerance. PeakRSS was below the2GiB eligibility threshold. The fixed-LP objective can differ from the native bound by a few micro-EUR within solver tolerances; the reported native MILP gap is zero.

The original three-unit per-case runtimes were0.6680s and9.8830s, with8s and60s MILP caps respectively. Timing context differs and these two cases do not establish full-cache cost. Objective reductions show the additional constraints change the physical solution; they are not evidence of a predictive gain.

Evidence: `artifacts/all14-integer-water-budget-probe-001/manifest.json`, SHA `b29c685f193df82074ac389c5c5f1ebac6502860fb01443f3f3bcf6b6fb990c4`. Exact command, all source/data/input hashes, solver version, per-case native bounds/arrays and result artifacts are in that manifest. Frozen protocolv2 SHA `d391f3c0cb7803e3946348d9f25e0c18502e1b914e94f1eb50ea0a20b0bc5af7`; independent preflight proofSHA `1740cb1c4a0a9bb92e58e4535971b3727b0244bd2d3ee6f957aa71e9b5eed8b0`.

Unrun v1 remains preserved: independent review corrected its signed-maximum slack predicate to maximum absolute slack before any solve. Four focusedv2 tests include negative numerical-slack rejection. Independent physical result review passed: `artifacts/all14-integer-water-budget-result-independent-audit-001/manifest.json`, SHA `97c1ff331fb7ad5fe87c7fd2ebd8de7e00d06c384bbb4fa5d0ee69e04a666e69`. It independently checked incumbent feasibility, full conditional-LP KKT/duals, exact replay, objective bounds and input closure without new solves or target reads. Root subsequently authorized preparation of a separate frozen12-case descriptive pre2021 diagnostic; shop owns its producer and shadow will independently review it before launch. The original three-unit cache remains separate.
