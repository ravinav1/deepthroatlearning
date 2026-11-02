# Latent water-shadow learnability results

The frozen early-year screen passed every declared gate. This establishes a useful predictive signal within the three-unit physical schedule family. It does not identify true water values, demonstrate improvement over the strong fourteen-unit anchor, or supply leaderboard evidence.

Training used 1,455 starts from 2015-01-01 through 2018-12-25. Six intervening starts were purged. The unchanged models were screened on 365 starts in 2019 and 360 starts from 2020-01-01 through December25. These are reused development years; the original reachability diagnostic had included a small date-selected sample from both.

| Three-target result | 2019 real map | Constant control | Permuted-target control |
| --- | ---: | ---: | ---: |
| Pooled AUC | .964439828960 | .947558169829 | .950888723411 |
| Thresholded hour errors | 20,230 | 29,862 | 29,755 |
| Expected hour mismatch | 38,246.59 | 42,991.20 | 41,814.60 |
| Soft-target cross entropy | 1.971043 | 2.018709 | 2.005338 |

| Three-target result | 2020 real map | Constant control | Permuted-target control |
| --- | ---: | ---: | ---: |
| Pooled AUC | .960536099115 | .954200344987 | .953766621753 |
| Thresholded hour errors | 16,607 | 21,447 | 21,404 |
| Expected hour mismatch | 27,725.51 | 30,472.69 | 30,337.58 |
| Soft-target cross entropy | 2.124997 | 2.139729 | 2.135974 |

The real map reduces threshold errors by 32.25% and 22.57% against the constant control. The unshifted path has the same threshold counts as that control. All threshold-error gains come from Lio: 17,590→7,958 in 2019 and 12,184→7,344 in 2020. Haukeli remains at 7,060/6,951 errors, and Hogga at 5,212/2,312. Their continuous probabilities change, but their `.5` decisions do not. No retrospective target-unit selection is justified.

Counterevidence remains material. Mean expected-displacement distance to the label-defined acceptable set is worse than both controls: real .056607 versus constant .054336/permuted .052259 in 2019; real .062940 versus .060091/.056781 in 2020. Central90% intervals cover approximately98–100% of cases, so coverage does not establish informative water-value identification. Expected mismatch under the real distribution is worse than the deterministic unshifted path's 29,862/21,447 errors; the declared expected-mismatch gate compared matched uncertainty-mixture controls, and it is not changed retrospectively. Haukeli's individual AUC is .955276 in 2019 but only .850857 in 2020; its .9406 reachability reduction has not become an equivalent predictive dispatch correction.

The fixed model is three independent 17-state multinomial linear maps over eight legal case features, 153 coefficients each. It learns a regularized distribution over the unchanged additive displacement grid `[-1,-.875,...,1]`, then averages the corresponding physical MAP schedules. Soft-target mismatch scale2hours, prior std .25, ridge .01, training-only normalization clipped to `[-5,5]`, one thread and deterministic L-BFGS-B settings were all frozen. Real/constant/permuted fits converged in 18–34 iterations; JSON coefficient replay reproduced all saved training distributions exactly. No 2021/2022 targets were decoded.

Protocol: `docs/latent_shadow_learnability_protocol.json`, SHA256 `866a2bcc7ac6ff1a9e51633bef4772716cac31c19062dfb5b9d573ce83901d39`. Source approval: `artifacts/latent-shadow-learnability-independent-audit-001/manifest.json`, SHA256 `ccd7e8ea51d3c072e87416f986750dd017a3c546ae92e45bdad2ef8086b4fca1`. Sixteen tests passed before the first expanded target run.

| Artifact | Runtime | Peak RSS | Manifest SHA256 |
| --- | ---: | ---: | --- |
| `latent-shadow-targets-train-001` | 65.670s | 259,260,416 bytes | `2fdb1506ac613735d163bec6a443e84f57fc7bd6cf8e695f8c9e210980c8309d` |
| `latent-shadow-map-fit-001` | 1.415s | 350,027,776 bytes | `7b83afc1ef44c7b19eb96c3f06dc5b2eb549535805a7acf2727cc808b44df07f` |
| `latent-shadow-targets-screen-001` | 33.417s | 249,266,176 bytes | `57b8adb7e425e778a27bac95aedd6b99b8fe701f717003d0e32cb12842866189` |
| `latent-shadow-map-screen-001` | .999s | 292,651,008 bytes | `62598be07f07065f69b280306f3394e30d65fb3f29ed82cb7141667b0caeeaa0` |

The training lock `artifacts/latent-shadow-map-fit-001/training_lock.json` was sealed at 15:57:42.483UTC, SHA256 `bca9f556f79e266e4bdac2673252edf529e425fb5f83bd822645913b7d4c4f12`. Screen labels were first decoded at 15:58:22.842UTC, after its explicit hash check. All2,180 case targets replayed the frozen `d=0` rewards/features exactly. No completed target generation was restarted.

Independent result verification passed in `artifacts/latent-shadow-screen-independent-audit-001/manifest.json`, SHA256 `aec70debbfa98949ad4ffd49e6bc46700192abb7392afb5db313a6fe24e9b988`. It verified exact partitions, source/data/cache/lock closure, training-only normalization, coefficient-based distributions and marginal probabilities bitwise, and all yearly metrics and gates. It fitted no model and read no 2021/2022 targets.

Exact uv commands, data and source hashes, environment, case IDs/dates, every grid reward/path, targets, coefficients, controls and probabilities are retained in the phase manifests. Root authorized preparation of a separate 2021-only direct probability splice against the fixed strong608-feature anchor. That experiment must use new IDs and a new protocol. Feeding same-row label-trained latent features into another classifier would require chronological cross-fitting; that unperformed feature-stacking approach is not part of the probability-splice plan.
