# Full14 early forward OOF cache

Extend the existing nine Vinje models to all14 generators using the single608-feature LightGBM recipe. This cache is an input to a separately frozen residual-objective screen; it does not recreate the accepted five-component blend and does not select or fit a residual model.

Keep all nine original checkpoint files and their probability columns byte-exact. Fit the other11 generators independently for each prediction year2018,2019,2020:350 trees,31 leaves, learning rate.06, min child100, L2=1, column fraction.9, max bin255, deterministic seed4173,2 threads, no early stopping. Reuse the exact saved40% case-hour row samples and independently reconstruct their sequential RNG draws. All sources, data, cache references, schemas, settings and reuse artifacts are pinned in the separate JSON protocol.

Training ends2017-12-25/2018-12-25/2019-12-25 respectively, maintaining a six-day start purge and no overlapping168-hour horizon with the prediction year. Store365 OOF cases for2018,359 through2019-12-25 and360 through2020-12-25. The combined2018/19 residual-training set contains724 starts,121632 case-hour rows and1702848 binary outputs;2020 contains60480 rows and846720 outputs. Existing historical labels are read only through2019. The2020 cache contains predictions, inputs, IDs and unit names but no labels.

Every new checkpoint and each reused checkpoint must reproduce its saved float32 probabilities exactly. Save608-column float32 features once per case-hour, avoiding14-fold expansion at this stage. Prediction shape is(case,168,14), feature shape is(case×168,608), and units use the original complete generator order. Manifest records exact rows, model times, source/data/environment hashes and replay evidence. No2021/22 labels, new score, submission or public leaderboard evidence is produced by this run.

Root authorized one local heavy job with2 threads. Estimate about30 minutes from the earlier nine-model run; actual timings are logged per generator. A separate independent review must approve the frozen source and protocol before launch.
