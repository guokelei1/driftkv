# Max 16-layer AUC comparison and three-scale preview

The user authorized this frozen-model diagnostic on 2026-09-21. The Max chain
is V0 → V1 epoch1 → V2 epoch1. Both parent/current Full assessments passed their
original four gates. The experiment evaluates five methods on both completed
edges: DroidSpeak, tail recomputation, KV translation, shared correction, and
shared correction with the user's native history response.

Each edge uses 10,000 evaluation snapshots with 1,024 pre-release events and
the real known-target feedback in its following 14-day window. Max has its own
200,000-user population and vocabulary; its evaluation and calibration cohorts
are selected independently, using pre-day217 history eligibility and a fixed
label-free hash order. Calibration, numerical-check and evaluation users are
disjoint. The exact selection, bindings and request counts are recorded in
`configs/insight/max_unified_auc_10k_20260921.json` and preparation metadata.

The method settings follow the completed 6L/10L diagnostic: DroidSpeak profiles
all contiguous layer intervals on 32 teachers and retains one interval per
layer-count budget; tail lengths are 32/64/128/256/512/1024; translation uses
256 teachers and 1–4 source layers, converting every evaluation user's cache.
The two read rules use 32/64/128/256/512/1024/2048/4096/7144 nested teachers,
16 calibration queries per user, and ridge 0.01. Their corrections are
`delta_r = N*b` and `delta_r = N*b + T*r`. Normalization and fitting use only
the corresponding teacher subset.

Raw logits are saved and hashed before labels are joined. The primary metric
is pooled ROC-AUC. Recovery is `(AUC_method - AUC_reuse) / (AUC_exact - AUC_reuse)`;
the display clips it to 0–100%, retaining signed recovery and absolute AUC in
the output ledger. A gap at or below 0.0001 has no displayed recovery.
The horizontal axis is theoretical incremental FLOPs relative to one Current
Exact cache rebuild per evaluation snapshot, including teachers, fitting,
profiling and extra reads. Execution time is used only for resource planning.

The requested preview contains 15 individual panels, a 3×5 combined figure,
and one five-panel strip per scale. The 6L and 10L rows reuse all five completed
version edges; the 16L row uses its two completed edges. Every panel labels
each version curve, uses different gray shades and markers, and has no mean
curve. Both axes run linearly from 0 to 100. The read-rule panels show only
64/256/1024/4096/7144 teachers. Translation preserves its measured left boundary.
Figure outputs are independent of the paper source and earlier previews.

Execution uses GPUs 0–3, 48 CPU threads for preparation, and bounded GPU
batches. The two edges run serially so their full teacher caches do not overlap.
At 7,144 teachers, the paired 16L FP32 CPU caches require 558.125 GiB in total;
initial inspection found about 790 GiB available RAM and four idle A40 GPUs.
The initial planning estimate is 80–150 minutes, extrapolated from the
completed 130-minute five-edge 10L run with greater depth and interval-profile
work. A focused numerical/resource check will refine this before detached
execution. This diagnostic and its launch are covered by the user's request.

Preparation completed in 19.43 seconds with 48 CPU threads. Four reference
users across both edges matched the existing loader exactly, including all
five arrays and dtypes. Both edges have full causal 1,024-event snapshots;
the first 256 expanded arrays and candidate panels equal the initial fitting
group. The two evaluation windows contain 50,753 and 50,981 known requests,
from 5,600 and 5,578 feedback-active users. All 10,000 snapshots enter the
Exact compute denominator on each edge.

The focused check completed in 310.90 seconds: 35 score paths on 334 requests
matched exactly between serial and four-GPU scoring. Streamed translation
matched the original fitter in selected layers and every parameter. Both
128-teacher read rules matched the original fitter in parameters and logits.
The batch64 check ran all method families in 5.82 seconds on 205 requests,
peaking at 23.35 GiB GPU memory; the original-fit comparison peaked at
23.49 GiB. The [updated resource estimate](resource_estimate.json) is
100–140 minutes for the complete two-edge run.

Status: completed, exit code 0, at 2026-09-20 21:27:23 UTC
(2026-09-21 05:27:23 Asia/Shanghai). The main run took 92.11 minutes;
including the reused initial calibration, successful computation took
102.62 minutes. Both edges completed all 47 score paths on 101,734 requests
in total. The renderer checked all 94 AUC values against retained raw scores.
All 15 PNG/PDF panel pairs, three scale strips and the combined figure exist;
the combined figure was visually inspected. No mean curves are included.

At 256 teachers, shared correction costs 3.528% of Exact FLOPs and recovers
-95.67% / 11.15% on V0→V1e1 / V1e1→V2e1. Adding the native user history
response costs 3.645% and recovers 53.05% / 20.40%. At 7,144 teachers the
personal branch recovers 54.48% / 23.37%, at approximately 99.995% cost.
One-source-layer translation costs 32.277% and recovers 75.39% / 89.14%.
Tail512 costs 59.636% and recovers 57.52% / 52.53%; full tail recomputation
recovers 100% at 100% cost. Negative recoveries are retained in the ledger
and displayed at zero, following the requested plotting convention.

Execution history: repaired and restarted in detached tmux `evokv_max_auc_10k` at
2026-09-20 19:55 UTC. Both complete initial calibrations passed and are reused
by the four-GPU main run. At 20:13 UTC, the user permitted ending interactive
supervision while the detached job continues. The tmux session was verified
alive, with all four calibration workers active and the first edge at the
4,096-teacher user-history fit. All five methods remain scheduled on both
edges; the launcher automatically verifies and renders all 15 panels after
scoring. The subsequent completion is recorded above.

The first run stopped on GPU memory exhaustion. The failed attempt
is retained in `failed_initial_attempt_20260921_032712/`, including its original
source snapshot, logs and exit status. Both edges completed the small shared
fits, then failed at the first 32-user DroidSpeak profiling read: eight
candidates per user expanded to 256 complete caches. The initial canary used
eight profiling users and did not cover this allocation. No full evaluation
scores were produced. The repair reduces candidate chunks while retaining all
32 profiling users, 16 candidates and 136 intervals.

The repaired `canary_v2` completed in 323.54 seconds. Its eight-user comparison
covered all 136 intervals under candidate chunks 8 and 2: selections were
identical and the largest MSE difference was 2.28e-8. Serial/four-GPU scores,
Translator parameters and both 128-teacher fitted rules agreed exactly.
The subsequent `initial_preflight` ran the actual 256-teacher, batch64,
32-user profiling path on both edges, including all four full-width
Translators. It completed in 631.09 seconds with each GPU peaking at 19.50 GiB.
The main run verifies and reuses these artifacts through `--initial-record`.
The preflight time is recorded separately from the restarted main timer and
must be added when reporting total successful execution time.

The first attempt used detached tmux `evokv_max_auc_10k`, launched at
2026-09-20 19:27 UTC (2026-09-21 03:27 Asia/Shanghai). The launcher is
`scripts/design/launch_max_auc.sh`: initial edge calibrations run in parallel
on GPUs 0/1, followed by four-GPU expanded calibration and scoring, one edge
at a time. It then runs `figures/src/three_scale_auc_preview.py`, which checks
all 94 Max raw-score AUC values and writes the 15 individual panels, three
scale strips and `combined_15panels.png/pdf` under
`figures/out/three_scale_auc_preview/`. Every panel identifies its version
curves and has no mean. `phase.txt`, `main.runtime.log`, `render.runtime.log`
and `exit_status.txt` record execution and completion. The `existing_layout/`
figure subdirectory contains only the already-completed 6L/10L layout check.
