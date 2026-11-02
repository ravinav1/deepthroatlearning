# Integer water-budget solver completion

This is a computational proof continuation of `integer-water-budget-screen-001`, authorized after that frozen screen finished. Its original 8-second MILP limit produced eight proven optima and four finite, physically feasible incumbents with time-limit status. The original screen remains failed. Its five descriptive metric gates passed; its proof gate failed.

Only the four cases selected by solver status 1 are rerun: 1826, 1827, 1958 and 2055. Each receives a 60-second MILP limit, then the unchanged two 1-second conditional LP replay limits. The eight original proven result files and their original proof records are copied byte-for-byte. No case is selected by target errors, and no labels beyond the existing 12-case pre-2021 diagnostic are loaded.

The mathematical formulation, three binary unit choices, supplied inputs, original relaxed control, solver gap target, number of threads, conditional replay requirements and all metric gates remain unchanged. Status 1 remains unproven even if the solver reports a rounded zero gap. Every failure, incumbent, bound and measured runtime is retained. The actual wall time can exceed the solver time limits by Python and solver shutdown overhead.

New physical results are saved before loading the original diagnostic's labels. The combined 12-case result is recorded separately, with an explicit original-versus-new case map and a separate proof gate. It cannot revise the original failed manifest and does not authorize a full cache, model fit, 2021/2022 evaluation or submission.

The frozen protocol pins this document, the new continuation script, the original screen manifest and diagnostic, its complete frozen source/data closure, and every reused result/proof file. The preflight validates this closure and the deterministic 8/4 partition before any solve.
