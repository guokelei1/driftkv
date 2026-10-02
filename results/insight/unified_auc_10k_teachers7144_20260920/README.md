# Shared-read teacher budgets through 7,144 users

历史开发记录；以下设置和结果仅指本次实验。旧执行入口及预览图已退役，
原始评分、配置、失败和封存来源继续保留。当前三规模 Motivation 见
[实验索引](../../../docs/motivation_observations.md)。

The [four-edge extension](diagnostic/summary.json) completed successfully in 969.436 seconds (16 minutes 9 seconds); its [runtime log](diagnostic.runtime.log) is retained. The [prospective configuration](../../../configs/insight/unified_auc_10k_teachers7144_20260920.json) fixes nine nested teacher budgets: 32, 64, 128, 256, 512, 1,024, 2,048, 4,096 and 7,144. All execution-source hashes remained unchanged. The [base experiment](../unified_auc_10k_20260920/README.md) and its 27 score paths remain unchanged; the extension adds ten paths, for 37 per edge. Recomputed Reuse scores match the base exactly on all four edges.

The table reports equal-weight means over all four edges. Compute is additional compatibility arithmetic, including calibration and actual E14 reads, as a percentage of one Exact rebuild for each of the 10,000 snapshots.

| Teachers | Shared recovery (%) | + user response recovery (%) | Shared compute (%) | + user response compute (%) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 29.96 | 56.24 | 0.3784 | 0.4382 |
| 64 | 30.94 | 64.37 | 0.7565 | 0.8194 |
| 128 | 26.45 | 54.77 | 1.5129 | 1.5818 |
| 256 | 26.90 | 52.59 | 3.0257 | 3.1065 |
| 512 | 25.98 | 49.39 | 6.0511 | 6.1559 |
| 1024 | 27.09 | 45.14 | 12.1021 | 12.2547 |
| 2048 | 27.26 | 47.99 | 24.2040 | 24.4522 |
| 4096 | 27.31 | 45.94 | 48.4078 | 48.8474 |
| 7144 | 27.27 | 45.61 | 84.4299 | 85.1543 |

Adding user response improves over the shared offset in all 20 new edge/budget comparisons. Increasing teachers does not produce monotonic recovery gains: at 256, 1,024 and 7,144 teachers, the response rule's mean recovery is 52.59%, 45.14% and 45.61%, while compute rises from 3.1065% to 12.2547% and 85.1543%. Both rules remain below Reuse on V1→V2 at every new budget. The full [72-row metric CSV](analysis/shared_budget_metrics.csv) contains every budget, edge, arm, absolute AUC, recovery and cost. Other comparison methods remain in the [base experiment summary](../unified_auc_10k_20260920/diagnostic/summary.json).

At the maximum 7,144-teacher budget:

| Edge | Shared AUC | + user response AUC | Shared recovery (%) | + user response recovery (%) |
| --- | ---: | ---: | ---: | ---: |
| V0→V1 | 0.602095 | 0.603048 | 80.38 | 92.13 |
| V1→V2 | 0.624414 | 0.624424 | -27.14 | -27.06 |
| V2→V3 | 0.662605 | 0.664243 | 30.47 | 40.50 |
| V3→V4 | 0.638269 | 0.660014 | 25.34 | 76.85 |

The original 256 teachers retain their exact order. The extended pool appends 3,329 unused eligible Medium users and 3,559 other existing Yambda-500M users selected by frozen selector rank and UID. Thus 3,585 teachers belong to the Medium training population and 3,559 are outside it. All have at least 1,024 listens before day 217 and remain disjoint from the unchanged 10,000 evaluation users, all 128 pilot users, both reserved confirmation groups and the other 256 historical fitting users. The Medium model, item mapping and stable OOV buckets remain fixed; no model training occurs.

Every budget uses the first corresponding teacher UIDs, with its own fitted normalization and coefficients. Each teacher contributes 16 fixed candidates. The popularity bank comes only from the original 256 pre-cutover histories; the existing recent/old/novel panel rule is applied to every additional teacher. All four edges' first-256 histories and complete candidate panels exactly match the original experiment. The [prepared metadata](prepared/metadata.json) records eligibility, isolation, old-loader references, input/output hashes and the 28.73-second CPU preparation.

Only five new budgets were fitted for each of the two shared rules, `N*b` and `N*b+T*r`: ten new score paths per edge. The original 27 paths, including all comparison methods and the four previous teacher budgets, are preserved unchanged. Evaluation reuses the original four release snapshots, 10,000 UID order and 212,773 known feedback requests; pooled ROC AUC and the compute denominator remain fixed. Every budget and outcome is retained.

The 256-user calibration canary reproduced the original parameters and logits with maximum difference zero. The [evaluation canary](calibration_canary/evaluation_canary.json) compared the new 512-user rules with the reference reader on 96 pilot users and 473 requests: Reuse and both new rule paths had zero logit difference, without reading labels. The [prospective resource estimate](resource_estimate.json) was 20–35 minutes on GPUs 0–3, with 48-thread input preparation, four CPU threads per GPU worker and serial release edges. The completed run used 125.58 GiB for paired CPU teacher-cache tensors; the maximum allocation across calibration and evaluation workers was 3.19 GiB per GPU. Runtime measurements remain separate from the theoretical compute plotted in the paper.

The former five-panel figure is retired; [figure provenance](figure_record.json) records its historical evidence and paper-copy bindings.
