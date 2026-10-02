# Translation with 64 teachers: four-release preview

历史开发记录；以下设置和结果仅指本次实验。旧执行入口及预览图已退役，
原始评分、配置、失败和封存来源继续保留。当前三规模 Motivation 见
[实验索引](../../../docs/motivation_observations.md)。

The supplemental run completed in **176.56 s (2 min 57 s)**. It adds independent 64-teacher KV translation fits with 1–4 source layers on V0→V1, V2→V3, V3→V4 and V4→V5, alongside the existing 256-teacher results. The paper and its assets remain unchanged.

The [configuration](../../../configs/insight/translation64_auc_preview_20260920.json) was fixed before these scores (SHA-256 `1867fa127c6501a5b47f4b5804d51e60733770da2b8a3e8c759b4984c8f5cc74`). The frozen six-layer Medium models, seed 17, 10,000 evaluation users, release snapshots and real-feedback requests are unchanged. Histories contain 1,024 events strictly before cutover; V0→V1, V2→V3 and V3→V4 use their original 14-day windows, and V4→V5 uses the original manifest's 13 complete days [287,300).

## Calibration and evaluation

The 64 teachers are the first 64 users in the original 256-teacher order, disjoint from evaluation and reserved groups. Each edge independently selects source layers and fits translation using **64 × 1,024 aligned Parent/Current K/V rows**. Selection ranks source layers using calibration-only, same-head OLS mean R² across heads and K/V; the final ridge maps use all heads from each of the selected 1–4 source layers. Ridge is 0.01, reads are FP32, and fitting is FP64. This KV calibration uses no candidate bank or 16-query Insight panel.

All four source-layer counts are retained. Raw logits are saved before joining the unchanged real-feedback labels. The primary metric is pooled ROC AUC; recovery is `(AUC_method − AUC_Reuse) / (AUC_Exact − AUC_Reuse)`. Existing anchors and all 256-teacher scores remain unchanged. Evaluation uses 54,826, 52,309, 51,764 and 50,672 known-target requests for the four edges, respectively (209,571 total); OOV targets remain outside the AUC denominator.

The horizontal axis includes Current teacher cache construction, source selection, the standalone ridge fit for that configuration, and translation of 10,000 snapshots. It is divided by one Current Exact rebuild per snapshot; common native reads are excluded. This arithmetic ratio is separate from measured runtime.

## Results

Across all 16 matched edge/source-layer pairs, the 64-teacher fit has higher AUC than the retained 256-teacher fit and uses less theoretical compute. With 64 teachers, mean recovery rises from 54.74% at one source layer to 67.13% at three, then is 66.89% at four. These are fixed-state, single-seed development results.

The table reports equal-weight means over the four displayed edges. All translation recoveries fall within 0–100%, so their raw and display-clipped means coincide.

| Source layers | 64-teacher compute (%) | 64-teacher mean recovery (%) | 256-teacher compute (%) | 256-teacher mean recovery (%) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 23.0005 | 54.7431 | 27.0747 | 49.9004 |
| 2 | 45.1416 | 64.6329 | 50.8757 | 62.8367 |
| 3 | 67.5614 | 67.1261 | 75.7844 | 64.4939 |
| 4 | 90.2607 | 66.8854 | 101.8014 | 64.4652 |

Absolute AUC and raw recovery for every measured pair are retained below; [analysis/translation_comparison.csv](analysis/translation_comparison.csv) contains all 32 edge/configuration rows with exact AUC, raw and display recovery, and compute. The 256-teacher, four-layer point remains in the evidence at 101.80% compute, beyond the figure's 100% x limit.

| Edge | Source layers | 64-teacher AUC | 256-teacher AUC | 64-teacher recovery (%) | 256-teacher recovery (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| V0→V1 | 1 | 0.599651 | 0.598500 | 50.23 | 36.04 |
| V0→V1 | 2 | 0.601742 | 0.601414 | 76.02 | 71.98 |
| V0→V1 | 3 | 0.602128 | 0.601735 | 80.79 | 75.94 |
| V0→V1 | 4 | 0.602195 | 0.601802 | 81.61 | 76.76 |
| V2→V3 | 1 | 0.664875 | 0.664437 | 44.37 | 41.69 |
| V2→V3 | 2 | 0.665540 | 0.665446 | 48.44 | 47.87 |
| V2→V3 | 3 | 0.666187 | 0.665735 | 52.41 | 49.64 |
| V2→V3 | 4 | 0.665874 | 0.665567 | 50.49 | 48.61 |
| V3→V4 | 1 | 0.659030 | 0.658409 | 74.52 | 73.05 |
| V3→V4 | 2 | 0.662132 | 0.661418 | 81.87 | 80.18 |
| V3→V4 | 3 | 0.662957 | 0.661764 | 83.82 | 80.99 |
| V3→V4 | 4 | 0.662975 | 0.661783 | 83.86 | 81.04 |
| V4→V5 | 1 | 0.663862 | 0.663535 | 49.85 | 48.82 |
| V4→V5 | 2 | 0.664608 | 0.664329 | 52.20 | 51.32 |
| V4→V5 | 3 | 0.664382 | 0.664355 | 51.49 | 51.40 |
| V4→V5 | 4 | 0.664411 | 0.664368 | 51.58 | 51.44 |

## Preview display

The retired preview renderer combined this supplement with the existing [teacher-budget results](../unified_auc_10k_teachers7144_20260920/README.md) and [V4→V5 record](../unified_auc_v4_v5_preview_20260920/README.md). Each panel used release curves in four distinct gray shades with distinct markers and a red equal-weight mean. For each matched configuration, the red y value is **the mean of the four individually clipped recovery percentages**, `mean(clip(recovery_edge, 0, 100))`; red x is the mean of the four corresponding compute percentages. This displayed mean is not a pooled AUC or the clipped mean of raw recoveries.

The last preview translation panel displayed the original 256-teacher configurations with source-layer counts 1–4; all users were translated, and the region below the minimum 27.07% cost was left empty. The measured 64-teacher configurations remain in this record and its comparison CSV; the last preview omitted them following the user's plotting preference. The final two read-correction panels showed teacher counts 64, 256, 1,024, 4,096 and 7,144. The display used linear 0–100% axes without shading; raw numerical outcomes remain unclipped in the retained evidence.

The combined PDF/PNG and derived preview ledger have been removed. All 32 measured translation rows remain in [the comparison CSV](analysis/translation_comparison.csv). The red read-correction means at 64 teachers are 33.3039% for shared offset and 66.0449% with user response; at 256 they are 33.1003% and 64.7947%, respectively.

## Execution and sources

The [canary](canary/summary.json) completed in 41.01 s on 96 separate pilot users and 473 requests. Calibration took 11.29 s with peak GPU allocation 5.60 GiB. Reuse and all four translation paths matched the unchanged reference evaluator exactly on raw logits; no labels were read.

The full run used GPUs 0–3 with serial edges and batch size 64. The prospective estimate was 3–6 minutes; measured completion was 176.56 s. Individual edges took 43.05, 40.16, 40.06 and 42.78 s. Calibration took 10.35–11.36 s per edge, peaking at 5.60 GiB on its GPU; evaluation workers peaked at 3.45 GiB per GPU. Reuse logits matched the retained scores exactly on all four full cohorts. The [runtime log](diagnostic.runtime.log), [summary](diagnostic/summary.json), saved rules and raw scores preserve the complete run.

The configuration records exact source hashes and prepared-array locations. The first three edges reuse `results/insight/unified_auc_10k_20260920/prepared`; V4→V5 reuses `results/insight/unified_auc_v4_v5_preview_20260920/prepared`. Existing 256-teacher scores come from the two linked result records above.
