# V4→V5 addition and four-release preview

历史开发记录；以下设置和结果仅指本次实验。旧执行入口及预览图已退役，
原始评分、配置、失败和封存来源继续保留。当前三规模 Motivation 见
[实验索引](../../../docs/motivation_observations.md)。

The V4→V5 diagnostic completed all 37 paths in **459.96 s (7 min 40 s)**. This is a separate preview of V0→V1, V2→V3, V3→V4 and V4→V5; the then-current paper and four-edge results were unchanged by this preview. All original V1→V2 results remain in the [completed teacher-budget record](../unified_auc_10k_teachers7144_20260920/README.md) and the original summaries retain all five measured edges.

The [configuration](../../../configs/insight/unified_auc_v4_v5_preview_20260920.json) was fixed before V4→V5 scores (SHA-256 `a3e65543ae1bb551343de784c417d78de221d5f9194c6055c66df4eecd0a2f96`). It uses the admitted, frozen six-layer Medium V4/V5 checkpoints, seed 17, the same 10,000 evaluation users and the same nested teacher order as the preceding experiment.

## Fixed protocol

- Every user has a fresh day-287 snapshot containing the latest 1,024 real listen events strictly before cutover. That snapshot is fixed throughout evaluation; each real feedback item is read at its actual timestamp. All methods share the same users, requests and retained histories.
- The existing frozen request manifest ends at day 300. The nominal V5 window is [287,301), but this diagnostic uses all available requests in **[287,300), 13 complete days**: 50,672 known-target requests from 5,444 users, plus 13,132 OOV-target requests excluded from AUC. The other three displayed edges each retain their original 14-day window.
- The five method families and all 37 paths are retained: Parent/Reuse/Current Exact anchors; DroidSpeak intervals of 1–6 layers; tail lengths 32–1,024; KV translation using 1–4 source layers; and shared read correction with or without native user-history response at teacher budgets **32, 64, 128, 256, 512, 1,024, 2,048, 4,096 and 7,144**.
- The 7,144 teachers remain disjoint from the evaluation and reserved groups: 3,585 are from the Medium training population and 3,559 are other existing Yambda users. The original 256 teachers remain the prefix. All budgets share a day-287 popularity bank constructed from those 256 histories and use the same 16 selected candidates per teacher. Each budget fits its own normalization and coefficients, with ridge 0.01 including the intercept, FP32 reads and FP64 solves.
- The two read corrections are `Δr = N b` and `Δr = N b + T r`, with `N = 1024`; the second adds actual native history-response content. Evaluation users supply no individual teacher responses.
- The primary metric is pooled ROC AUC on real feedback. Recovery is `(AUC_method − AUC_Reuse) / (AUC_Exact − AUC_Reuse)`, displayed when the denominator exceeds `1e-4`, without clipping stored values. The horizontal axis charges calibration and actual extra reads relative to one Exact cache rebuild per evaluation snapshot; it is arithmetic accounting, not a runtime ratio.

## V4→V5 results

The 50,672 requests include 45,763 positive and 4,909 negative labels. Parent Full, Reuse and Current Exact have AUC **0.644319, 0.648037 and 0.679783**, respectively. The Exact–Reuse gap is 0.031746, or 89.52% of the Current–Parent improvement.

Adding native user-history response improves over the shared offset at every teacher budget and recovers 29.08–41.47% of the Exact–Reuse gap. The shared offset alone remains below Reuse at every budget. Increasing teachers beyond 512 does not improve recovery monotonically: the response branch reaches 41.47% at 512 and 38.56% at 7,144. DroidSpeak with one refreshed layer recovers 66.84% at 8.63% compute; four layers recover 97.73% at 65.20% compute. These are fixed-state, single-seed development results.

Compute percentages include calibration and actual extra reads, relative to one Exact rebuild per snapshot.

| Teachers | Shared AUC | + response AUC | Shared recovery (%) | + response recovery (%) | Shared compute (%) | + response compute (%) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.643639 | 0.657269 | -13.85 | 29.08 | 0.3784 | 0.4356 |
| 64 | 0.643592 | 0.659492 | -14.00 | 36.08 | 0.7565 | 0.8168 |
| 128 | 0.644888 | 0.659805 | -9.92 | 37.07 | 1.5129 | 1.5791 |
| 256 | 0.644824 | 0.660635 | -10.12 | 39.68 | 3.0256 | 3.1038 |
| 512 | 0.645139 | 0.661201 | -9.13 | 41.47 | 6.0511 | 6.1532 |
| 1,024 | 0.645275 | 0.660767 | -8.70 | 40.10 | 12.1021 | 12.2520 |
| 2,048 | 0.645290 | 0.660846 | -8.65 | 40.35 | 24.2040 | 24.4496 |
| 4,096 | 0.645340 | 0.660561 | -8.49 | 39.45 | 48.4078 | 48.8447 |
| 7,144 | 0.645382 | 0.660277 | -8.36 | 38.56 | 84.4299 | 85.1516 |

All 34 method-budget curve points, including absolute AUC, recovery, compute, log loss and Brier score, are available in [analysis/curve_metrics.csv](analysis/curve_metrics.csv). The three anchors and all 37 quality records are retained in [diagnostic/summary.json](diagnostic/summary.json); raw logits and joined labels remain in `diagnostic/v4_to_v5/`.

## Execution and artifacts

CPU preparation used 48 threads and completed in 20.82 s; [prepared metadata](prepared/metadata.json) records the day-287 arrays and hashes. The [canary](canary/summary.json) used 96 separate pilot users, 375 requests and 25 paths, completing in 38.20 s. Single-GPU and four-GPU logits matched exactly for every path; native-reader and Exact-replacement differences were at most `2.3842e-7`. Full-interval and full-tail KV checks passed. No labels were read by the canary.

The full run used GPUs 0–3 and retains its [runtime log](diagnostic.runtime.log), configuration and raw scores under `diagnostic/`. It completed within the prospective 5–10 minute estimate. Initial calibration and baseline construction took 43.02 s with a peak GPU allocation of 16.43 GiB. The expanded teacher stage took 279.63 s, stored 125.58 GiB of paired cache tensors in CPU memory, and peaked at 3.17 GiB per GPU; evaluation workers peaked at 5.28 GiB. Full-interval and full-tail checks also passed in the completed run.

The old preview renderer, PDF/PNG and derived cost ledger have been removed. The following records the original display convention; all measured quality remains in the retained diagnostic summary. Translation displayed the original four 256-teacher configurations, converting every user's cache; the region below its minimum 27.07% cost was left empty. The [completed 64-teacher translation supplement](../translation64_auc_preview_20260920/README.md) remains in the numerical evidence but was omitted from that preview. Each panel showed the four release curves in four distinct gray shades with distinct markers and a red equal-weight mean: recovery is clipped to 0–100% separately for each edge before averaging, while mean x uses the four corresponding compute percentages. The axes were linear 0–100%, with no shaded region. The last two panels showed five teacher budgets: 64, 256, 1,024, 4,096 and 7,144. All nine measured budgets and signed outcomes remain in the diagnostic summary; the retired ledger also recorded translation points above the display limit. The preview labeled V4→V5 as 13d and the other displayed edges as 14d.
