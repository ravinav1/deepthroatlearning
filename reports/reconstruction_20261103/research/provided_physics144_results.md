# Provided-input physical proxy results

These are validation experiments, not new Kaggle scores. The training recipe, input restrictions, cache gates and commands are documented in `docs/provided_physics144_workflow.md`. Each branch retains the same original clean canonical115 anchor. External research informed the physics methods; all numerical features, training labels and inference inputs are competition-provided or locally computed from them.

The conditional shared-flow cache completed all3647 cases in1220.04s, with zero fallbacks. Its manifest SHA256 is `daca717e605f5c20ee82643ac64d4d914890fb7795e5022d17f04f215de5f681`; the complete independent audit is `artifacts/provided-conditional-full-cache-audit-001/manifest.json`, SHA256 `374720f3ab21a80117b8057d5471946082da90ee5817fb6e675b467823e45aba`. Every15-field case block and the aggregate array were reconstructed byte-exact.

The conditional1442021 run selected the declared shared weight0.1:

| Weight | MicroAUC |
|---:|---:|
|0|0.9952596524075921|
|0.1|0.9952709694343881|
|0.25|0.9952423777967205|
|0.5|0.9950862466867311|
|1|0.9942006496743026|

This is a small gain of0.0000113170. The independent full-grid/byte/layout audit passed: `artifacts/provided-conditional144-2021-result-audit-001/manifest.json`, SHA256 `dd0894a1826c09556c1a7e2c22de6648b79bc84509b37efd6ff83b88ce164c65`. The1400-tree fit finished at698.99s, used about2.89GB peak RSS, and passed full checkpoint replay. Its lock remains unchanged for2022.

Post-lock diagnostics are descriptive only and did not change the global weight. Selected-minus-anchor per-unit AUC was +0.00068137 for Haukeli,+0.00007793 for Vinje_G3,+0.00005057 for Hogga and+0.00000284 for Lio; Vinje_G1/G2,Kjela,Songa and Tokke_G2/G4 declined. All14 new contemporaneous states and14/15 own fields appeared in splits; the unused availability field was constant because every solve succeeded. The original115 fields accounted for20.58million gain, own15 for1.77million and global14 for0.13million. Correlated tree importance is not a causal contribution or an ablation. Full per-unit raw/anchor/selected scores and field-level split/gain values are in `artifacts/provided-conditional144-2021-diagnostics-001/manifest.json`, SHA256 `12869ffe3272c450836570fef51341941856158dec6128c24e3b503ccba97375`.

The independent bounded cross-gradient cache also completed3647 cases with zero fallback. Its manifest SHA256 is `14fe1ed5ee1d271e1f083938d3d764d3f3390fc341300d0548e26e5f4bfd8230`; full-cache audit `artifacts/provided-crossflow-full-cache-audit-001/manifest.json`, SHA256 `42ee6f95cbeb1629c46178d25115af98a2ac79f4e6d058ea720515e722ba600a`, verified all formulas/aggregate bytes and final mixed-state reconstruction. It is a distinct2021 model experiment; the conditional model's selected weight is not transferred to it.

The original conditional global0.1 blend **failed** its frozen2022 gate:0.9940497484705896 versus anchor0.9940509264719491, a decline of0.0000011780. The independent result audit passed (`provided-conditional144-2022-result-audit-001`, SHA256 `1508d3382a98dedd9458b230009d44e775445eb7371c302f49308ac9de2a7499`); this is a model result, not a pipeline failure. No global final fit is authorized by that rejected lock. Per-unit diagnostics (`provided-conditional144-2022-diagnostics-001`, SHA256 `ddb04496854be1e6fa8857400e04080184f795775785d5424e471572aaf258e1`) show Lio+0.00019247 and VesleKjela+0.00006975, but Hogga−0.00012701,Kjela−0.00013175,Haukeli−0.00005855. The Haukeli raw model AUC was0.95110781 versus anchor0.96447783, so its2021 diversity gain did not transfer. All14 global and14/15 own fields remained used. The separately declared parent-owned plant-weight experiment also failed its unchanged2022 confirmation: microAUC0.9936592129, below both the anchor and global0.1 blend. The2021-selected plant vector was `[.1,0,0,0,0,.5,0,1,0]` for Hogga,Tokke,Lio,Byrte,Vinje,Haukeli,Songa,Kjela,VesleKjela and had scored0.9953574082 in2021; its lock SHA256 `58ad7472afb9e404bceb8a9b372f6b6a5436bfc05634d02dec7f19e17ce9d641` was sealed before2022 predictions existed. Independent2021 trace audit SHA256 `1dfcc35f22808731acca4398678084e6adcea958a9f004e4914012d802a18ce4` reproduced all90 grid scores and18 choices. Independent2022 audit `provided-conditional144-plant-confirmation-2022-audit-001` SHA256 `08808ee890abad8dc4285a797e089bdf497e4cb5804e4fa8d3f84903535d4fd3` confirmed the rejected result and exact selected bytes. Both conditional branches are closed without final training, submission or post2022 adjustment.

The gradient1442021 branch selected0.25, scoring0.9953035361075371 versus0.9952596524075921 (+0.0000438837). The declared grid scored0.9952977451984579 at0.1,0.9952085467171866 at0.5 and0.9944855997742378 at1. Independent audit `provided-crossflow144-2021-result-audit-001` passed, SHA256 `a067d39d5e40282c61712c41f5aaf8b517951f2c9bbcb13f249cbd36e1332005`. Its post-lock diagnostics (`provided-crossflow144-2021-diagnostics-001`, SHA256 `c450a9632f62c84389b99d5b39e43ae9d59dc59c832b3e57a519b8a049a1ed17`) show Haukeli+0.00242147,Vinje_G3+0.00015051,Hogga+0.00010508,Lio+0.00010009; all Tokke units,Vinje_G1/G2 and Kjela declined. All14 global states and14/15 own fields were used. Native gain totals were20.676million original,1.603million own and0.232million global. The fixed0.25 gradient2022 confirmation is reported below.

The busbar cache also completed3647 cases with zero fallback. Full manifest SHA256 `9c266f71bad082e8a69986f4eac7a05e68be260266444f7c352a596afbc68fde`, independent cache audit `provided-busbar-full-cache-audit-001` SHA256 `1f8a253ac2cb092792cd39e757c9d1f8239ab97c42f7795f0e82174d8911c005`. Its separate frozen144-feature fit results are reported below.

The gradient144 unchanged2022 confirmation subsequently **failed**: locked0.25 scored0.9940156938979107 versus anchor0.9940509264719491. Independent result audit `provided-crossflow144-2022-result-audit-001` passed, SHA256 `c1086854eebac16571b76717fd2043eb762a44f033ab0f2f7a8f5d3dee3e83d9`; no gradient LGB final is permitted by that lock. Descriptive diagnostics `provided-crossflow144-2022-diagnostics-001`, SHA256 `8c7e00da3e5ba4ab50e41eb862d413632f64784b829e0c54720d2c6c36ae28d6`, show Lio+0.00037836 and Vinje_G3+0.00004746, but Hogga−0.00025361, Kjela−0.00030292 and Haukeli−0.00018123. The raw Haukeli AUC was0.94765325 versus anchor0.96447783. No weights were changed after2022.

The busbar1442021 run completed and selected0.1:

| Weight | MicroAUC |
|---:|---:|
|0|0.9952596524075921|
|0.1|0.9952804039903538|
|0.25|0.9952762557200010|
|0.5|0.9951857168685516|
|1|0.9945636631489904|

Its independent result audit passed (`provided-busbar144-2021-result-audit-001`, SHA256 `44ebde3483814b1e28d62fe47d02fcf80824eba5c74ea26cc78aae41d78b3041`). The complete run took619.14s. Descriptive diagnostics (`provided-busbar144-2021-diagnostics-001`, SHA256 `569e32807e279555818e14fec151467fabb2aacae173c8ee1e7d5bc01ed42513`) show Haukeli+0.00029696, Lio+0.00014006 and Vinje_G3+0.00009649; Kjela−0.00029665 and Vinje_G2−0.00004534. All14 global and14/15 own fields were used.

The frozen busbar0.1 blend **passed2022** at0.9940585327322639 versus anchor0.9940509264719491 (+0.0000076063), in675.35s. Independent result audit `provided-busbar144-2022-result-audit-001` passed, SHA256 `f7bdac212aa5493777177ec5261f30096fee5f75d395ea471effa8ff7944004b`. Diagnostics `provided-busbar144-2022-diagnostics-001`, SHA256 `78c85c40b940da07bf5e4d70c67899d3369ad6ad7d2f62e13c0dffad559a193a`, show Haukeli+0.00047531, Lio+0.00008508, Vinje_G3+0.00009290 and VesleKjela+0.00006688, with Hogga−0.00011614 and Kjela−0.00008399. Importance totals were24.366million original115 gain,1.691million own15 and0.204million global14. These diagnostics did not alter the locked shared weight. The authorized final fit `provided-busbar144-final-001` is running with the same1400-tree recipe and0.1 blend.

A separate scalar pooled CatBoost144 experiment chose gradient features using only the already frozen2021 global result:0.9953035361075371 exceeded busbar0.9952804039903538. That feature choice preceded the gradient2022 rejection and remains fixed. CatBoost must independently beat both the clean115 anchor and the same-weight gradient LGB control on2021, then with unchanged choice on2022. Its protocol is `docs/provided_pooled_cat144_stage2021_protocol.json`, SHA256 `58678bb12af9c05014998ddabf93678dc54f11267aeacf385f55f73f4f637f33`. This does not reopen the rejected gradient LGB gate.

No new Kaggle result is inferred from these validation experiments. The best measured provided-only score remains the separately documented historical0.99653; the account's0.99952 rank-one entry uses external simulator predictions.

The busbar final subsequently completed in823.88s. CSV `artifacts/provided-busbar144-final-001/submission.csv`, SHA256 `b989424b21b5deaf13ffab4e969484b40bb8e464678d6cbb43b644c29f87fac1`, passed the independent final result/actual-CSV audit (`d40027161310b16956f8c4c8b48218d37daf5346d635264fd5e530c6c382b0fa`) and an additional complete feature/checkpoint/CSV inference replay (`35f1ec6d1986f44dd3eeba5bb08c17463bcdc648572836c5c95298438b8b9ad9`). Full final recipe, native checkpoint/array checksums and standalone inference command: `docs/provided_busbar144_result.md`. It is ready for root to submit; no public score is asserted yet.
