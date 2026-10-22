# Repository agent notes

## Active work — 2026-10-19 (read first)

Latest Yambda-Large completion — 2026-10-22: all five independent adjacent
baseline edges V0→V1 through V4→V5 are complete (24 points/edge, 75,807 eligible
quality/cost users per edge). The GPU 1 four-edge queue finished at 08:07
Shanghai, taking 12h27m14s; no Yambda-Large job remains running. All 95 committed
score blocks, input/config hashes and formula/row-count consistency passed.
Full 120-point report, CSV, TeX and derived summary are in
experiments/01_quality_cost/runs/yambda/large/baselines_all_edges/seed_17/attempt_1/.
Original edge summaries/weights/scores were not rewritten. V4→V5 remains
E14_partial; its wide recovery intervals do not establish superiority over
Rebuild. Layer2 recovery is 84.88/85.48/55.92/47.28/60.46% across the five
edges at about 21.04–21.05% cost. All 15 Query-only point estimates are below
Reuse; History256 falls to 4.57% and −40.42% on the last two edges. Preserve
these adverse outcomes and diagnose independently rather than tuning on test
outcomes. No new experiment or CacheTide quality run was launched.
Latest RecFlow metric clarification — 2026-10-22: the user wants common,
task-natural metrics (AUC/NDCG/Recall/MRR), roughly 1–3% relative update gains
with at least three of five releases showing a clear phenomenon, plus scores
clearly above a matched random baseline. RecFlow should emphasize a
generative/retrieval task rather than only like/dislike discrimination; this
new task preference supersedes the older RecFlow feedback-only planning
direction, without rewriting existing results. Current Large/Max checkpoints
are still effective_view BCE models with a scalar candidate head. Their
sampled positive-item retrieval results are not autoregressive or full-catalog
generation evidence. No change to the running training recipe or new
foundation training is authorized by this metric search.
The user explicitly instructed: stop ONLY this new metric-search inference on
GPUs 0/1; DO NOT stop the existing foundation training. Metric inference was
migrated to physical GPUs 2/3, retaining 177 committed blocks and original
hardware/hash provenance. Foundation tmux session recflow-foundation and
training PIDs remain running on 0/1. Read experiments/STATUS.md and
experiments/foundation_recflow/runs/chain_2gpu/attempt_1/large/
sampled_retrieval_full/plan.json. Full inference completed in tmux sessions
recflow-metric-full-2/3 (socket recflow202), covering all 94,809 eligible D20–24
requests, includes all-negative/OOV zero scores, and retains NDCG/Recall
20/50/100 and MRR, including regressions. Inspect gpu2/gpu3 outputs, not the
stopped gpu0/gpu1 attempts. NDCG@50 and Recall@100 were selected as focus
after a 1024-request/day pilot; full-population verification includes that
pilot and is exploratory, not independent confirmation. The user's three-of-
five rule does not permit averaging only favorable releases or describing
positive point estimates as statistically confirmed gains. Full-population
results corrected the pilot: NDCG@50 has only one positive edge and mean
-1.56585%; MRR also regresses on average. Recall@100 has five positive point
estimates (three above1%), mean +0.94667%, paired UID95% [0.32784,1.58641]%,
and current/random ratios 2.107–2.272. Only V3→V4's per-edge Recall interval
excludes zero; do not call three or five releases statistically significant.
Recommend Recall@100 for current candidate coverage, retain adverse NDCG@50
as ordering evidence. See sampled_retrieval_full/report.md and
paired_uid_analysis/summary.json. Both inference jobs exited0; original
training continues unchanged on0/1. This does not launch further jobs.

Latest execution authorization — 2026-10-21 evening: after the completed
Yambda-Large V0→V1 report, the user requested all remaining independent adjacent
edges V1→V2, V2→V3, V3→V4 and V4→V5 on physical GPU 1. The four-edge sequential
queue started at 19:39 Shanghai on g201. Resume from experiments/STATUS.md and
experiments/01_quality_cost/runs/yambda/large/baselines_v1_v5_gpu1_201/attempt_1/.
Keep the same 24 natural points, all 75,807 eligible quality/cost users and
4096-user checkpoints. Each edge fits its own six learned points and profiles
its own layer intervals; reuse UID roles only, not correction weights or state.
V4→V5 [287,301) includes partial source day 300 and must remain E14_partial,
not complete 14-day evidence. Do not rerun V0→V1 or launch CacheTide quality.
The queue freezes source/config hashes; do not edit its execution Python files
while queued edges remain pending. Runtime estimate: 3–4h/edge, 12–16h total,
progress only, not controlled timing evidence. This supersedes GPU 0 and the
first-edge-only queue restriction in the historical paragraph below.

Latest RecFlow execution clarification — 2026-10-21: prepare a single sequential
job on machine 202 using TWO GPUs (NVLink pair 0/1): finish Large V0-V5, then
Max V0-V5. This supersedes the earlier four-GPU and stop-after-V1 scope.
Every version uses two complete epochs, preserving all twelve permanent full
model/AdamW/RNG/progress checkpoints per scale, fresh -> v01 -> v02 -> v11 ->
v12 -> ... -> v51 -> v52. V0 fits D1-18; V1-V5 fit only D19 through D23,
one new day per version. Inherit optimizer state within each scale; Max starts
fresh independently. Hourly rolling recovery retains two generations and
permanent epoch checkpoints are also resumable. Preserve frozen roles and
paired multi-metric future-day evaluation on all eligible final-role targets.
The user has NOW explicitly authorized starting training and requires a durable
launch that survives network/client disconnection. The formal sequence is
running in detached tmux socket recflow202, session recflow-foundation, on 202
GPUs 0/1. Entry: scripts/recflow/run_foundation_chain.py; configuration:
experiments/foundation_recflow/configs/chain_large_max_2gpu_202.json now has
training_authorized=true. Run: experiments/foundation_recflow/runs/chain_2gpu/
attempt_1; launch.json records commands/PIDs/logs. Do not change frozen numeric
source, design/run configuration or data during execution. Resume only after
the original process has exited; never start a duplicate sequence. Source data
is already on 202; earlier copying to 201 did not start training there.
The user additionally requires explicit GPU-memory risk assessment and long-run
reliability. Large remains global96/micro48 per GPU. Max now uses global80,
micro20 per GPU with two accumulated microbatches, replacing micro40 to leave
headroom. Cover all 15 bounded history/candidate shapes, repeated mixed replays,
full checkpoint saves and fresh-process recovery before reporting readiness.
Do not equate these bounded probes with an already completed long training run.

Latest RecFlow planning clarification — 2026-10-21: the user designates the old
4096-user 6L/H192 generative chain RecFlow-Medium and requests a design only for
Large/Max matching Yambda parameter counts and structure, with different video
table sizes. See experiments/docs/recflow_model_design_2026_10_21.md and its
config/statistics. Proposed Large/Max have 2.2M/3.5M independent videos,
10/16 layers, width320 and 65,536 OOV hash rows each: 730,211,201/1,149,285,121
parameters. Both use the existing development population and preserve
admission/final roles; original Medium NDCG is not feedback AUC. This design
does not authorize new training, GPU probes or a launch queue.

Latest user clarification — 2026-10-21: experiment 01 compares independent
adjacent-version transitions. All methods start each edge from the same old
native state; correction/student state is not carried between edges. Normal
causal reads/appends within an edge remain required. Experiment 02 alone tests
our method's continuous multi-release state; do not require baselines to support
continuous releases or reuse experiment 01 edges as a continuous trajectory.
On machine 201 the user initially allocated physical GPU 1, then changed
subsequent experiments to physical GPU 0. The user authorized fetching needed
data and existing V0–V5 models from g202 using the sync workflow. Current work
prioritizes baselines only, first Yambda-Large V0→V1, roughly 3–5 natural
operating points per adjustable method, with method-specific counts/settings.
Latest clarification: choose natural controls, then report actual FLOPs; do not
invert costs to force exact 15/50/80% points. Layer uses 2/4/6/8 layers; Tail
uses 10/20/40/80% tokens; Deviation/Query use 10/20/40/60% selected positions.
The original roughly 15–80% cost range is guidance, not an exact grid.
Our method's quality runs and
other version edges wait until this first comparison has been reported.
The user subsequently required full eligible test-population quality, rather
than the initially proposed 512-user pilot. Preserve fit/validation and earlier
calibration/development reservations; do not exclude ordinary historical test
users based on their outcomes. The current Large V0→V1 test pool is 75,807.
Before a population run, optimize execution and report a measured-input-based
30k-user runtime estimate. This is done: the 24-point plan is 70–100 minutes for
30k resident users and about 3–3.5 hours for all 75,807 eligible users.
The full run completed on GPU 0 in 3h06m on 2026-10-21: 75,807 eligible users,
34,190 feedback users, 345,630 feedback events, all 24 points and UID intervals.
Resume from experiments/STATUS.md; this task has no active GPU job.
Selected Layer intervals all start at zero and require no persistent old hidden
boundary cache. The earlier generic 45% claim was corrected in report metadata;
all measured metrics/FLOPs/scores/weights are unchanged. Layer2 is 84.88% recovery
at 21.04% added FLOPs / once-Rebuild, not the earlier partial-preview ~95–98%.
Full deployment costs still include events
omitted solely because they cannot affect an independent edge's scored quality.
Latest reliability instruction: save results promptly and support bounded-loss
resume. The runner commits each 4096-user score block (normally about 8–10 minutes)
and reuses completed fitting; resume validates frozen settings, inputs, code,
weights and block hashes. Never discard an entire long run after interruption.

Latest organization — 2026-10-21: formal experiment plans, configurations and
new results are centralized under `experiments/`. Resume from
`experiments/README.md` and `experiments/STATUS.md`; readiness and scheduling
are in `experiments/docs/experiment_readiness_2026_10_21.md`. Quality/FLOPs
runs may share GPUs; formal timing requires controlled resource conditions.
Legacy system-study logs and results retain their historical paths. This
directory organization does not launch or queue experiments.

Latest evaluation-scope clarification: the user explicitly excludes end-to-end
serving experiments, including the briefly proposed limited arrival replay.
Validate the design using real-model feedback quality, continuous cache/code
evolution, complete FLOPs, total auxiliary memory, and controlled measurements
of preparation/read/append/handover paths. Keep calibration, shared preparation,
all per-event work and transfers accounted for; measured path time is not online
throughput, p99 or complete-system wall time. Optional scale-time estimates must
state their assumptions and remain distinct from measurements. This supersedes
older service-platform and full-cycle-time requirements below. The active task
is evaluation planning, not launching GPU work. See paper/new_eval.md and
docs/system_study_2026_10/evaluation_rationale.md for the claim/evidence audit.

Latest user clarification during evaluation figure review: final evaluation covers
Yambda and RecFlow, each at Large and Max scale (four conditions). Medium is
development-only and is not a final-result configuration. This supersedes the
earlier Medium/30k final-evaluation priorities below; existing evidence remains
unchanged. See paper/new_eval.md and eval_figure/README.md for the latest plan
and explicitly synthetic figure previews. RecFlow Large/Max model/task/population
settings still need to be established; its existing six-layer NDCG evidence must
not be relabeled as feedback-AUC results. This figure task launches no training.

Later evaluation-planning clarification: foundation models are fixed backbones;
do not retrain their chains for multiple seeds. Proposed repeats concern only
the paper's correction-training modules and key learned baselines. The current
new_eval.md separates training seeds, fitting-user samples and timing repeats,
with single-run exploration and shared main trajectories to limit costs. This
is an evaluation plan, not authorization to launch a new training queue.

The user authorized organizing and beginning the parallel system study. Resume
from `docs/system_study_2026_10/README.md` and its `STATUS.md`; these point to
the actual commands, outputs and pending work. Do not rely on chat memory.
The current reference is `A_itemgate6_rms_roll4` (query_relay_2026_10), with
shared12/private6 persistent codes, shared3/private3 online reads, item gates,
corrected behavior writes and composable anchor-code lifecycle. The rank8 and
support-flag candidate below is historical, not the active default.
Quality exploration, strong baselines, service replay and cross-condition
preparation may proceed in parallel. 80% recovery is a historical reference,
not a pass/fail requirement; retain the compact state and approximately20%
complete added-cost design target and report the quality/cost frontier.
New small development implementations and probes are authorized. Record
concrete settings before expensive runs; do not restart completed experiments
or launch old configurations merely because they are retained. No foundation
model retraining is part of the initial work. Current workload roles, task
separation and complete cost rules are in the active plan. Later dated user
instructions take precedence over the archived settings below.
The user's latest clarification on 2026-10-19 restores the original real
like/dislike feedback task as primary, matching Motivation. Next-item results
remain separate historical/exploratory evidence; do not reuse their quality,
request counts or cost/time projections as feedback results. Existing next-item
run configurations are not the default continuation queue. This supersedes the
earlier same-day decision to prioritize next-item. Small-panel timing, including
4096 users, is diagnostic rather than a final deployment-scale verdict. First
derive fixed/shared versus per-user costs and the scale break-even; then test
whether implementation converts that saving into complete preparation/service
time. Common foundation training must be distinguished from correction fitting.
Sample multiple model chains, version edges and pre-release history strata;
unadapted H10/RecFlow cases are not evidence of correction-quality transfer.
The user clarified that every foundation-model group requires its own correction
calibration and fitting users; cross-model correction transfer is not a goal.
Experiments must support the paper's central quality/cost claim or determine its
final settings. Prioritize the actual 30k Medium lifecycle time/cost accounting
and fitting-population choice; broader inventories and numerical probes are
supporting evidence, not reasons to expand the experiment matrix on their own.

## Research workflow — applies to code and experiments

EvoKV is a paper research repository. Optimize for fast, credible idea
validation and simple, readable implementations. Production readiness,
exhaustive edge-case handling and test coverage are not project goals.

- Start with the smallest implementation and experiment that can answer the
  current research question. Reuse existing code and inspect the signal before
  building a general framework or scaling up.
- Implement the inputs and execution modes the experiment actually uses.
  Add boundary checks only for a concrete risk to numerical correctness,
  evidence, data causality or expensive execution. Do not build speculative
  compatibility layers, generic validators, fallback stacks or exhaustive
  configuration support.
- Defer serialization, concurrent lifecycle handling, fault recovery and
  deployment machinery until the experiment or paper claim needs them.
  A synchronous in-memory prototype is a valid first implementation.
- Keep exploratory configuration and notes lightweight. Reuse an existing
  experiment description; do not create a new contract, audit document or
  approval step for every small code/configuration change. Small development
  probes within an authorized task can proceed directly. Long training and
  formal population runs retain the authorization requirements below.
- Preserve scientific validity: causal data, honest controls and metrics,
  held-out separation, reproducible essential settings and retained evidence.
  These checks serve the research question; they do not require industrial
  hardening or an exhaustive engineering checklist before exploration.

## Archived direction and evidence — 2026-10-05

The user fixed the three-part framework on 2026-10-05. Select clear, technically
justified internal choices; do not change the three problems or add modules just
to improve a development endpoint. A fixed-model mechanism case is useful; a
majority-of-cases or transfer claim needs evidence for its stated scope.

- D1 keeps one native K/V cache and rank8 codes. Head-local native K/V and the
  complete Current item embedding feed a small encoder. Shared Key/Value
  projections adjust scores and the aggregated read in one history scan.
  Native writes remain unchanged; no corrected full K/V copy is retained.
- D2 fits shared parameters before release: local read alignment followed by
  joint fitting through the frozen complete query. Current-teacher outputs
  supply targets; no future request labels or online teacher enter serving.
- D3 binds releases, initializes on the first read, follows appends/evictions,
  and retires codes after reconstruction or all-layer causal clearance.
  Existing per-layer/row inherited-support flags zero codes of cleared rows.
  Current-produced upper-layer rows may still depend on inherited lower state.
  Clearance means independence from inherited state, not equality to Fresh.

The current development candidate is
`fullitem8_local_kv32_joint16_support_lazy`: 84.80659% recovery at 18.43754713%
complete cost, with release recovery 92.86093/69.32950/100.33430%.
Recovery UID95% is [51.635,102.364]%. The paired mean AUC gain over the unmasked
76.13645%/18.42950495% control is 0.00173493, UID95%
[-0.00099248,0.00481246]; V5 regresses. This is not independent confirmation or
proof of stable 80% recovery. Masking reuses original weights and flags, adds
0.016216TF and 337,840,368 transfer bytes, and adds no persistent tensor/refit.

The balanced mean/relative-score objective completed at 72.3082%/18.5421% and
is not selected. Key-auxiliary and Value-capacity branches also failed to improve.
Historical key outcomes remain in final summary/analysis files and
docs/design/plan.md; redundant per-run reports have been retired.
Better teacher MSE does not imply better AUC; the V5 mechanism case found a
large common-offset contribution and concentrated gains, not all-layer synergy.
Do not rerun completed controls, probes, candidates or the old stopped learned
strength run. No model training or population experiment is queued.

## Scientific and resource boundaries

- Quality: fixed4096 users, 51,731 requests, V2→V3→V4→V5 continuous development.
  Equal-release mean AUC differences define recovery; do not average recovery
  percentages or pick favorable subsets/releases. Freeze before confirmation.
- Complete costs: separately fixed30k real event ledger, including shared
  preparation and every service operation. Sum actual FLOPs before taking the
  ratio. Quality/calibration/deployment populations are separate quantities.
- Full/Rebuild: once per user/release at the first request, then normal native
  incremental service. Per-request Fresh is a historical diagnostic. Report
  eager all-resident rebuilding separately, including idle users.
- Keep the20% complete added-cost ceiling and compact state boundary. The
  historical80% target is a reference, not sufficient acceptance. Codes/flags
  occupy12.5%/0.0651% of native K/V. If source-KV reconstruction is required for
  calibration, the selected point costs21.9382%; retain that sensitivity.
- Query work must also be reported against native query/attention itself.
  Isolated kernels/graphs are not loaded throughput or p99. No loaded serving
  advantage or negligible-query claim is established.
- Current fit settings: local32/joint16,256fit/16validation users,16queries,
  rank8,seed17. Retained real histories and native trajectories are causal.
  Calibration roles and previously reserved validation/fit users remain
  excluded from later evaluation; do not tune on old confirmation outcomes.
- Paper Design explains problem–mechanism connections. Concrete numerical
  settings and outcomes belong in Implementation/Evaluation. Paper source and
  current diagrams describe the selected development candidate, with limits.
  Motivation and its measured points remain unchanged. Do not compile the paper
  without a user request.

## Repository cleanup and retained assets

On2026-10-05 the user explicitly authorized deleting branches proven not to
work, completed experiment code/intermediates while keeping important results,
and assets known to be unnecessary. This supersedes earlier blanket retention
instructions for every historical source snapshot, weight and per-user score.

Retain the24 selected foundation models, current original/processed datasets,
latest method's original calibration and local checkpoints, frozen calibration
and quality panels, matched merged controls, and primary Motivation evidence.
Keep historical key summaries, settings, costs, confidence intervals and negative
results at original paths. Their recorded hashes/paths describe past execution;
files outside the retained asset list may now be absent. Do not promise complete
historical reruns or manufacture replacement raw evidence.

On2026-10-07 the user requested a narrower result inventory. Result JSONs were
reduced from12,036 to411, including199 summary/analysis/report files instead
of1006. Retain final branch aggregates, unique adverse outcomes and essential
current dependencies; do not restore redundant shard, per-user, epoch, canary,
pilot, status or source-manifest records merely for historical completeness.
Keep frozen user-role membership even when its experiment has ended. The
remaining JSON contents, weights and scores were not rewritten. Current panel,
calibration, matched-control, cost and90-point Motivation reads passed.

Later on2026-10-07 the user also authorized removing obsolete raw score/request
Parquet files and exit/status text. Result Parquet files dropped from5868 to98;
5770 Parquet and296 .exit/.txt files were deleted. The98 retained files are90
Motivation Q/H merged scores,7 current calibration/quality/control inputs and
one unique historical Max20k-user membership list. Full/Reuse now retains all15
final edge summaries instead of raw score/request shards. Historical raw paths
are provenance only; do not promise per-request reanalysis or resume from them.
Current datasets, weights, histories, JSON conclusions and retained scores were
not rewritten; current reads passed. This stage left785 result files; the later
source/doc cleanup below reduced that count to680.

Retired experiment-specific code/tests, failed candidate weights, old history
packs/raw outputs, per-user checkpoints, logs, repeated source copies and old
previews have been removed. Shared older modules remain only where a retained
entry actually imports them. Capture/signature helpers were extracted without
changing their numerical logic; live source hashes can differ from original
execution snapshots after cleanup. Original results and frozen weights were not
rewritten.

The2026-10-07 source/doc pass removed30 retired code files,3 launchers,5
branch-specific test files,22 obsolete/duplicate documents,85 execution-source
copies and107 Python caches. The repository now contains1342 regular files
excluding .git, of which680 are results. Legacy calibration CLIs were trimmed
to the shared functions still used by current dependencies; retained function
ASTs are unchanged. Lifecycle tests now exercise the current full-item KV
adapter. CPU checks passed176 tests with7 CUDA skips; all411 result JSON hashes,
current inputs, weights and retained scores remain unchanged. Historical source
hashes describe past execution; they do not imply exact source copies remain.

Read README.md, scripts/README.md and results/README.md for current navigation.
Details and retained history are in docs/design/plan.md and
 docs/design/exploration_history.md. Cleanup does not authorize new training,
change sample roles, or make completed experiments a launch queue.
