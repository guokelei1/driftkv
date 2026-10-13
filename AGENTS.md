# Repository agent notes

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

## Current direction

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
