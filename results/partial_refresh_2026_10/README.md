# Partial-refresh experiment index

All source is under `scripts/partial_refresh_2026_10/`; all new outputs and logs stay here. Existing models, data and earlier results are read-only.

**Latest completed exploration:** [second-round report](round2/report.md), [index](round2/README.md). Batch-local item-term reuse saves35.10% refresh/maintenance FLOPs with no persistent item table; a static4k table saves49.94% (4deploymentreplicas). Both retain≈100% full-code gain on all3360users/69,361requests, passing the requested>90% interval lower bound. Length/sparsity/priority negative paths and the61.13%/96.04% partial combination are retained. These gains mainly come from equivalent encoder computation reuse, while the first two points still refresh all rows.

**First-round entry:** [research report](report.md), [3360-user quality](benchmark3360/quality/summary.json), [complete fixed30k FLOPs](population_cost_raw/summary.json), [frozen strategy](frozen_raw_candidates.json), [source and reproduction](../../scripts/partial_refresh_2026_10/README.md).

Frozen Medium V2→V3→V4→V5, context1024, rank8, original local32/joint16 weights; no foundation training or refit. The strict raw-storage policy is rho0/.875/.75. It retains **96.0448% [93.3510,97.4663]%** of full-code gain on all3360 eligible benchmark users /69,361 requests. Partial−Reuse mean AUC gain is **0.0438681 [0.0301162,0.0554255]**; partial−full is **−0.0018065 [−0.0027639,−0.0010660]**. The overall95% preservation-margin interval crosses zero; do not claim a guarantee of at least95% preservation.

Refresh encoding saves9.4053%; complete added FLOPs save0.51314% against the original masked full-refresh method, including new publication validation and raw read masking. Complete cost is18.34294% of once-per-user/release Rebuild, under the retained-native-input assumption. New publication input reconstruction raises it to20.10753%; all-source/no-prefix-reuse sensitivity is23.60814%.

| Run | Role / conclusion | Evidence |
| --- | --- | --- |
| `probe_original` | 32-fit preliminary validation; provisional V5 rho0 was too aggressive | `summary.json` |
| `canary_all_refresh` | GPU numerical equality to saved full-code method; exact cost match | `summary.json`, `cost_verification.json` |
| `validation_original` | 256 fit / 16 validation; frozen masked policy 0/.75/.5 | `summary.json`, `cost_audit.json` |
| `development512_probe` | Provisional policy retained 86.17% gain; not selected | `summary.json` |
| `development512_selected` | Small check retained 99.87%; insufficient for final conclusion | `summary.json` |
| `development4096_selected` | Full original policy retained 93.36%, below95% preservation target | `report.md`, `summary.json`, `cost_verification.json` |
| `population_cost_original` | Fixed30k complete FLOPs18.0030% vs18.4375%; refresh encoding saves18.33% | `summary_complete.json` |
| `fresh1024/panel` | Superseded preliminary role exclusion; retained for numerical canary provenance only | `README.md` |
| `fresh_canary_all_refresh` | 8-user numerical check only, not held-out quality evidence | `summary.json` |
| `fresh1024_v2/panel` | Frozen1024 users excluding89 prior roles; no outcome/activity filter | `selection.json`, `excluded_roles.json`, `cohorts.json` |
| `fresh1024_v2/quality` | Full code and Rebuild themselves regress vsReuse; gain-retention ratio undefined | `report.md`, `summary.json` |
| `probe_raw` | Retain unmasked codes, gate only at read; preliminary calibration | `summary.json` |
| `canary_raw_all_refresh` | Raw full-refresh remains numerically identical; raw mask costs exact | `summary.json`, `cost_verification.json` |
| `validation_raw` | Frozen raw 1% tolerance policy0/.75/.5 | `summary.json`, `cost_audit.json` |
| `validation_raw_strict` | Frozen raw .5% tolerance with extra rho.875: policy0/.875/.75 | `summary.json`, `cost_audit.json` |
| `development512_raw` | Small development check retained101.03%; larger population later worsened | `summary.json` |
| `development512_raw_strict` | Small strict development check retained100.15%; not final evidence | `summary.json` |
| `development4096_raw` | Raw 1% policy retained89.91%, worse than masked; not selected | `report.md`, `summary.json`, `cost_verification.json` |
| `development4096_raw_strict` | Strict raw retained96.25% [91.57,101.16]%;95% margin not established | `report.md`, `summary.json`, `cohorts_summary.json`, `cost_verification.json` |
| `benchmark1024/panel` | Frozen hash-ordered alternative population; excludes80 prior roles/current development | `binding.json`, `selection.json`, `cohorts.json` |
| `benchmark1024/quality` | Frozen strict policy retained97.24%; then unchanged eligible-population expansion | `report.md`, `summary.json`, `cost_verification.json` |
| `benchmark3360/panel` | All3360 eligible benchmark users, no outcome/activity filter; exact1024 prefix | `binding.json`, `selection.json`, `extension_check.json`, `cohorts.json` |
| `benchmark3360/quality` | Final overall96.04%; long-history97.46%, short-history71.87%; V5 weaker | `report.md`, `summary.json`, `cohorts_summary.json`, `cost_verification.json` |
| `population_cost_raw` | Both raw policies, raw/full/original controls, all shared preparation and service FLOPs | `summary.json` |

Quality gains use equal-release AUC differences, never averaged recovery percentages. All preselected releases remain in the results. Existing development4096 is not independent confirmation. The benchmark population existed before this task, and expanding1024 to3360 happened after seeing1024; settings stayed frozen and every eligible user was retained. This is not a new blind full-method confirmation. New-user negative evidence is retained.

Predeclared long-history (>768;3023 users) retains97.4561% [96.4639,98.3588]%; history≤256 (126 users) retains71.8730% [40.6216,97.1242]%. These are exploratory boundary results; all five history/activity cohorts remain in the aggregate. The pure-new1024 population regresses even with full codes and once-Rebuild, so no positive full-method transfer claim follows.

Publication references and all candidate-query forwards are charged. `cost_audit.json` adds required previous-code preparation; native-input reconstruction sensitivities are separate. Code versions add4bytes/row (0.0434% nativeKV). No wall-time savings or loaded-service claim.

`logs/` contains one log per owned run. Each run's `configuration.json`, `summary.json` and `users.json` index settings, essential source/weight/input hashes and progress outcomes. Shards and512prefixes are merged with exact user/request coverage checks; completed prefixes are reused. First-round final3360 merges the old1024 shards plus four584-user remainders after exact UID/request/history/model equivalence checks. Cost replay matches252,000 executed scalar counters. All first-round GPU jobs completed; these directories do not represent a pending launch queue. New second-round jobs are indexed separately above.

Global `cohorts.json` is fixed from pre-release history/activity, before evaluating quality. `frozen_policy.json` records the initial masked freeze; `frozen_raw_candidates.json` records raw variants and strict selection before benchmark scoring. Raw storage/grid/tolerance choices used development evidence; per-publication rho selection used only fit/validation forward scores. CPU regression evidence is in `logs/cpu_suite.log` (184 passed,7 CUDA skips).
