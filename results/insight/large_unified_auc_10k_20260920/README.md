# Ten-layer, five-edge AUC comparison

历史开发记录；以下设置和结果仅指本次实验。旧执行入口及预览图已退役，
原始评分、配置、失败和封存来源继续保留。当前三规模 Motivation 见
[实验索引](../../../docs/motivation_observations.md)。

This user-authorized development diagnostic repeats the five Medium methods on
the frozen Large 10-layer, 320-wide, 10-head models. The main settings are fixed
in [the prospective configuration](../../../configs/insight/large_unified_auc_10k_20260920.json).
The existing six-layer results and paper assets are preserved.

The 10,000 evaluation users, 7,144 ordered teachers, 1,024-event release
snapshots, and exact real-feedback identities match the six-layer comparison.
Large item indices and Large request IDs are used with the original Large label
source. Five release windows contain 54,826 / 53,874 / 52,309 / 51,764 / 50,672
evaluation requests; the final window spans 13 complete days.

The main comparison measures DroidSpeak widths 1–10, tail lengths
32/64/128/256/512/1024, KV translation with 256 teachers and 1–4 source layers,
and both shared-read rules with 32/64/128/256/512/1024/2048/4096/7144 teachers.
The read rules are `delta_r = N*b` and `delta_r = N*b + T*r`, where `r` is the
native read of that user's retained history. Their common ridge is 0.01.
The separate 96-user pilot explores ridge 0.001/0.01/0.1 with 256 teachers and
retains all results. The main comparison keeps the same default as six layers.

The five selected edges include the previously selected V5@1epoch endpoint.
Its original failed bootstrap admission gate remains in the records; the
user's all-five-edge request authorizes this development diagnostic without
changing the admission seal.

The new figure uses five gray release curves plus a red equal-weight mean.
Each edge's AUC-gap recovery is clipped to 0–100 before averaging for display.
Raw AUC, signed recovery and theoretical FLOPs remain in the ledger. FLOPs are
relative to one Current Exact rebuild per evaluation user; timing is used only
for execution planning. The last two panels display 64/256/1024/4096/7144
teachers. Both axes are linear and bounded by 0–100.

Execution uses GPU 0–3 and 48 CPU threads for preparation. The input preparation
took 59.53 seconds; all reference histories, UID ordering, prefix causality and
teacher-panel checks passed. The main canary completed in 135.56 seconds:
29 score paths agreed exactly between serial and four-GPU execution, and
streamed Translator parameters and source-layer selections matched the original
implementation exactly. The expanded calibration check took 98.75 seconds;
both 256-teacher arms matched the original fitter in parameters and logits.
The 512-teacher extraction used all four GPUs, peaking at 8.56 GiB per worker;
the complete in-GPU 256-teacher reference peaked at 24.10 GiB.

The task completed successfully (exit status 0), after running in detached tmux `evokv_large_auc_10k`, via
the now-retired `scripts/design/launch_large_auc.sh`. `running_phase.txt`, the per-phase logs,
and `exit_status.txt` retain its state. It started at 2026-09-20 11:47 UTC.
`run_large_auc_parallel.py` distributes the first four independent baseline
calibrations across GPU 0–3, then queues the fifth. Each subsequent edge uses
all four GPUs for expanded teacher calibration and evaluation. Per-edge
baseline logs are in `initial_parallel/<edge>/runtime.log`; later stages use
`diagnostic.log` and per-worker evaluation logs. The GPU1 scheduling check took
82.16 seconds and matched the original GPU0 rules, Translator parameters and
selected intervals exactly. The stopped serial-baseline attempt is retained
in `serial_baseline_attempt/`; it produced no evaluation scores. The post-canary
[resource estimate](resource_estimate.json) is 60–120 minutes for the main
five-edge run. This timing estimate does not enter figure coordinates.

The retired renderer `figures/src/large_unified_auc.py` checked every raw-score
AUC and source identity before writing the former five-panel preview.
Its derived images are no longer present. All five edges
completed with 10,000 snapshots and 41 score paths each, totaling 263,445 real
feedback requests. Initial calibration took 645.61 seconds and the main
calibration/evaluation took 7,158.08 seconds: 130.06 minutes combined. Maximum
allocated GPU memory was 39.66 GiB during initial calibration and 14.61 GiB
during evaluation. The renderer independently verified every raw-score AUC.

At 256 teachers, shared correction costs 3.20% of Exact and shared plus user
history costs 3.30%. Their five-edge display means are 14.54% and 37.02%; the
corresponding **unclipped** recovery means are 2.13% and 23.61%. History raises
AUC on the first three edges and lowers it on the last two. Per-edge unclipped
recoveries are shared→history: 26.12→81.21%, 13.64→62.68%, −15.56→41.19%,
32.93→−11.05%, and −46.48→−55.98%. Raising the history branch to 7,144 teachers
costs 90.31% and gives 37.23% display recovery (21.31% unclipped).

DroidSpeak reaches 55.31% display recovery at 10.87% cost and 80.70% at 64.21%
cost. Tail recomputation reaches 4.03% at 16.65%, 37.59% at 59.48%, and 100%
at full recomputation. KV translation's display mean stays near 39.7% as its
cost rises from 31.76% to 120.86%; both last edges have negative raw recovery.
Its first point has 22.74% unclipped mean recovery. These outcomes retain all
five edges and all settings; the plotted mean follows the requested per-edge
clipping rule.

The ridge pilot completed in 95.08 seconds. All 30 ridge/arm/edge observations
are in [the sensitivity CSV](analysis/ridge_sensitivity.csv), with a
[short analysis](analysis/ridge_sensitivity.md). Neither alternative improves
both arms consistently. Under the default 0.01, adding history raises pilot
AUC on four of five edges, with an equal-edge mean increase of 1.10895
percentage points. The default256 rules on V0→V1 also match the original
fitter exactly, recorded in `analysis/ridge_default_reference.json`.

The background handoff and subsequent completion review are complete.
`exit_status.txt` is 0, `diagnostic/summary.json` is completed, and `render.log`
records successful verification. The combined PNG/PDF and five individual
panels were generated and inspected at completion, then retired. Their recorded
display used five gray release curves and a red mean; the numerical results
and original completion logs remain.
