# CohortKV single-configuration full-chain development v1

## Status

`cohortkv_single_config_full_chain_development_v1` completed and froze Stages 0–4 plus the
Stage-4.5 source-plan and Stage-4.6 lifecycle amendments on 2026-07-27. The workload contract, baseline frontier, deployed
compiler artifacts, common-layout resident operator, normal-path full-cohort HBM/DRAM engine, and
direct-old-K/V hot-HBM source plan are closed. Stage 4.6 now covers recursive migration of actual
approximate outputs through the fixed one-A40 `theta0 -> theta11` chain. Its frozen lifecycle uses
balanced age/deadline scheduling, 15%–25% edge-severity exact budgets, and maximum depth four.
Stage 5 is the next open stage for executable fallback, guard, transactional rework, and failure
visibility. These are adaptive seed-0 development results, not replicated generality evidence.

The frozen artifacts are:

- `configs/cohortkv_single_config_v1/blueprint.json`;
- `configs/cohortkv_single_config_v1/workload_manifest.json`;
- `configs/cohortkv_single_config_v1/result.schema.json`;
- `configs/cohortkv_single_config_v1/stage1_frontier_summary.json`;
- `configs/cohortkv_single_config_v1/stage2_compiler_summary.json`;
- `configs/cohortkv_single_config_v1/stage3_operator_summary.json`;
- `configs/cohortkv_single_config_v1/stage4_system_summary.json`;
- `configs/cohortkv_single_config_v1/stage4_5_source_plan_summary.json`;
- `configs/cohortkv_single_config_v1/stage4_6_lifecycle_policy.json`;
- `configs/cohortkv_single_config_v1/stage4_6_lifecycle_summary.json`;
- `scripts/freeze_cohortkv_stage4_6.py`;
- `scripts/freeze_cohortkv_single_config_v1.py`.

Regenerate the files from the current source artifacts with:

```bash
python scripts/freeze_cohortkv_single_config_v1.py
```

Verify that the prepared data, checkpoints, verified plans, selected programs, workload records,
and generated contracts have not changed with:

```bash
python scripts/freeze_cohortkv_single_config_v1.py --check
```

Any mismatch stops the run. A deliberate change requires a new protocol or an explicit revision
of this development protocol before downstream experiments continue.

## Frozen vertical slice

| Field | Value |
|---|---|
| Data | KuaiRand-1K standard log, 4+12 split |
| Base / updates | D1-D4 / D5-D16 |
| Evaluation endpoint | theta11 on D16 (`20220423`) |
| Source cohorts | theta0, theta4, theta10 |
| Model | 16 layers, hidden/K/V width 512, maximum history 2,048 |
| Training seed | 0 |
| Eligible records | 682 |
| Valid prefix tokens | 1,087,785 |
| Logical FP16 capsule | 17,822,269,440 bytes (16.60 GiB) |
| Logical FP16 target K/V | 35,644,538,880 bytes (33.20 GiB) |
| Primary destinations | HBM and pinned DRAM |
| GPU counts | 1, 2, 4 NVIDIA A40 |

The workload manifest hashes the prepared data and fixes every record ID, prepared one-based model
user index, source-log raw user ID, evaluation role, source version, target version, observed
history length, prefix-token count, and pre-cap history length. It contains no target item or
recommendation label. All migrated source tensors and target K/V cover `history[:-1]`;
`history[-1]` remains the current-model latest token used by stale-serving evaluation.

## Record roles and permissions

The verified compiler's existing split is retained exactly:

| Role | Records | May affect | Must not affect |
|---|---:|---|---|
| Fit | 40 | Affine parameter fitting | Hyperparameter/action selection, certificate, final report |
| Program selection | 60 | Candidate/profile selection and per-method runtime tuning | Affine fit, primary contract, final report |
| Certificate | 60 | Apply the frozen label-free contract to a predeclared library | Candidate grids, recommendation labels |
| Final test | 522 | Final semantic and task reporting | Any fit, tuning, selection, layout, or contract |

The full destination job processes all 682 records after algorithms and layouts are frozen. It may
include records from all four roles because it is a complete deterministic transformation, but it
does not load recommendation labels. System tuning uses only the 60 program-selection records.

## Controlled source-version assignment

The complete record set is real; its cache ages are controlled. The available data has no request
or cache-refresh trace from which an organic migration-anchor version could be recovered.

The frozen assignment:

1. sorts the 682 eligible prepared model user indices ascending and assigns stable record IDs
   `0..681`, while retaining the source-log raw user ID as audit metadata;
2. computes exact source counts by largest remainder from weights `0.20/0.30/0.50`;
3. shuffles those assignments with NumPy seed `58211`;
4. never reads recommendation labels or evaluation roles.

| Source | Records | Prefix tokens |
|---|---:|---:|
| theta0 | 136 | 221,999 |
| theta4 | 205 | 324,193 |
| theta10 | 341 | 541,593 |

This may be called a **complete controlled mixed-version cohort**. It may not be called organic,
request-derived, or production-distributed cache age.

## Baseline and action contract

### DroidSpeak adaptation

The closest external baseline is not defined as an arbitrary set of high-drift layers. DroidSpeak
profiles contiguous recomputation groups and uses a sender transition activation (`E` cache) to
start receiver-model recomputation. Scattered groups add transition state and propagate mismatch.
The HSTU adaptation therefore freezes:

- one contiguous current-model interval per candidate;
- widths `m in {2,4,6,8,12}`;
- every legal start position for each width, 53 intervals in total;
- old FP16 K/V outside the interval;
- one old-version FP16 pre-block hidden state when the selected start is greater than zero; a
  layer-zero start uses raw embeddings and needs no transition state;
- current-model recomputation inside the interval;
- the common label-free cache/score/top-100 views instead of DroidSpeak's task-label profiler.

The minimal HSTU reference executes full current blocks before the terminal interval layer and only
the terminal layer's `Norm + Wk/Wv`, because no later current hidden state is consumed. The existing
repository helper `migrate_contiguous_cache` is not this baseline: outside its interval it applies
current projections to old normalized states. Stage 1 implemented a separate reference whose
outside-interval values are the source old K/V and whose full-depth case matches exact replay.

For each `m`, the 60 program-selection records choose the interval with the best worst-view
recovery; ties prefer lower measured GPU cost and then the earlier start. The 60 certificate
records then select the minimum-cost frozen `m/interval` satisfying the same contract as CohortKV,
or exact if none passes. The 522 final-test records cannot change it.

This is named `selective_contiguous` or “DroidSpeak-adapted” in artifacts. It is a compatible
algorithmic baseline, not a reproduction of DroidSpeak's distributed LLM serving system.

### Action inputs

Every input is read lazily from buffered POSIX shards on the same `/data` ext4 source tier. One
untimed complete warmup precedes three measured repetitions; the page cache is not explicitly
evicted. Timed source reads are included. This is a common tier, not identical input bytes and not
a cold-device SSD benchmark.

| Action | Frozen source representation | Role |
|---|---|---|
| Reuse | old FP16 K/V | Zero-maintenance semantic anchor; not publishable synchronization |
| Cheap projection | FP16 normalized capsule | Compiler ablation |
| Compiled affine | FP16 normalized capsule | Primary candidate |
| Selective contiguous | old FP16 K/V + raw history; selected transition hidden only for start > 0 | External baseline |
| Residual-p | raw history + BF16 old hidden suffix `[p..L-1,T,H]` | Internal escalation tier, `p in {4,8}` |
| Exact | raw history | Current-model K/V reference |
| No transform | old FP16 K/V | Pure placement/transaction floor |

Source-shard creation, checkpoint loading, and offline tuning are outside job completion and must
be timed separately. Physical bytes for every representation remain required outputs.

Residual-p cannot be reconstructed from the default normalized capsule and one transition hidden
state. It needs every old pre-block hidden state from layer `p` through layer 15: 12.45 GiB for
`p=4` or 8.30 GiB for `p=8` over all 682 records. The currently verified fallback uses only `p=8`
for theta0 and theta10, which is still 5.83 GiB of auxiliary BF16 state. These bytes are not called
part of the 16.60-GiB default capsule and must enter source traffic and capsule-economics reporting.
If the auxiliary shard is not retained, the executable plan must be revised to fall through to
exact; Stage 4 may not pretend the existing residual fallback is available.

Before shard materialization, the implementation must verify that the resolved source is the
declared `/data` ext4 device and that at least 128 GiB is free. Candidate transition states used to
profile selective intervals exist only for the 60 program-selection records outside the timed
system job; final shards retain one frozen transition per source-target cohort.

The earlier verified compiler evaluated in-memory FP32 layerwise state. Stage 2 has now reapplied
the frozen label-free certificate to serialized FP16 capsules, prepared FP16 programs, and FP16
output without touching final-test records. All three primary plans pass and are directly
loadable. Real materialization rejected FP16 for the optional unnormalized hidden suffix, so that
auxiliary representation alone is BF16 at unchanged two-byte accounting. Engine
transport correctness then compares against the same selected method and numeric path resident on
the same serialized input, requiring finite output and `allclose(atol=0.02, rtol=0.02)`. Semantic
fidelity remains a separate comparison to FP32 current-model exact K/V and full-catalog scores.
Stage 3's `reference_fp32` widens that serialized FP16 capsule/program for arithmetic and writes
FP16 output; it is not the original FP32 fitted program or the semantic exact reference.

## Destination and timing contract

Every method publishes separate contiguous, unpadded FP16 K and V tensors
`[layers, valid_tokens, kv_width]` per extent, plus lengths and offsets. An implementation may use
dense length-bucketed execution internally, but any compaction, scatter, serialization, allocation,
or layout conversion needed to reach this output stays inside its measured boundary.

| Destination | Start | End | Durability |
|---|---|---|---|
| HBM | Frozen source manifest before its first read | Complete target K/V resident across active destination GPUs and manifest visible | Non-durable |
| DRAM | Frozen source manifest before its first read | Complete target K/V retained in pinned CPU extents and manifest visible | Non-durable |

Fresh target allocation occurs inside every timed job. Both destinations use
`streamkv_destination_manifest_v1`; commit requires all 682 record IDs exactly once. HBM omits D2H
by definition. DRAM includes D2H. Raw HBM and DRAM completion times are separate endpoint results.
The DRAM path preflights one maximum-size pinned extent and verifies that available host memory
covers its retained 33.20-GiB target plus the bounded source wave and publication queue; it may not
silently substitute pageable or non-retained output.
The one-GPU HBM endpoint retains 33.20 GiB of logical target K/V on a 47,699,722,240-byte A40.
Before timing, a capacity preflight must add assigned target bytes, model/program residency,
measured maximum-batch temporary memory, and allocator margin. It may reduce batch/in-flight values
only within the frozen grid. If batch 1 still cannot satisfy the endpoint, execution stops for an
explicit protocol revision; the point is not silently dropped or changed to a streaming sink.
Every warmup and measured repetition destroys the prior private target, synchronizes, and allocates
a fresh one. It also reopens and decodes the source shards; only the OS page cache remains warm,
not an application-level tensor or batch cache.

As a non-performance feasibility audit on 2026-07-27, cuda:0 successfully retained a
35,644,538,880-byte dummy target while theta11 recomputed a four-record, 2,047-token batch.
Peak allocated bytes were 37,518,565,376 with BF16 compute and 38,055,828,992 with FP32 compute;
8,647,802,880 bytes remained free at the FP32 output point. This removes the immediate one-A40
capacity blocker, but real extent overhead and engine queues still require the run-time preflight.

The job timer includes:

- source-manifest scan, buffered source read, decode, and pinning;
- destination allocation;
- H2D, compute, D2H where required;
- stage, coverage validation, coordinator overhead, and manifest commit.

The report separates `source_read`, `h2d`, `compute`, `d2h`, `stage`, `commit`, and total elapsed.
Compiler fit/certificate time is separate and is amortized over 682 records with both summed-work
and parallel critical-path accounting; Stage 2's `0.453 s/record` is the former, not observed
three-worker wall time. Training, checkpoint loading, source-shard creation, and offline tuning are
excluded and reported as setup.
Every method uses byte-weighted LPT over extents, with weight equal to its declared logical source
plus target bytes. Per-GPU records, tokens, input/output bytes, elapsed time, peak HBM, assigned-byte
imbalance, and aggregate source/staging/publication-queue peaks are mandatory; the aggregator
checks that device totals exactly match the complete run.

## Experiment matrix

| RQ | Frozen development experiment | Selection boundary | Final artifact |
|---|---|---|---|
| RQ2 | Existing theta0/4/10 verified programs, primary 70/80/90/30 contract, recovery targets 50/60/70/80/90% | Fit, selection, certificate roles only | Compiler/certificate cost, action/fallback, fidelity, threshold and amortization |
| RQ3 | Compiled, 53 selective-contiguous candidates, residual-p, cheap, reuse, exact | Selective interval on selection; certificate determines publishability | Cost-fidelity frontier, certificate outcome, frozen profiled action |
| RQ4 | Compiled/profiled-selective-diagnostic/exact on HBM and DRAM at 1/2/4 GPUs; residual and no-transform controls | Runtime grid tuned independently per method/destination/GPU count on selection users | Completion, throughput, bytes, peak memory, breakdown, manifest |
| RQ4 failure | Hash mismatch, semantic theta4 perturbation, four transaction failures | No final-record tuning | Detection/escalation, rework, visibility, cleanup |
| RQ5 | Capture, FP16, staged INT8 dequantization, workload-free break-even | Layout fixed before final report | Bytes, capture/dequant cost, fidelity, crossover |

RQ3 records 59 selection points for each of theta0/theta4/theta10: all 53 selective intervals,
residual p4/p8, compiled, cheap, reuse, and exact, for at least 177 points. The aggregate audits the
complete interval set per source-target pair. Interval selection and certification are also per
pair; one source cohort's winning interval cannot silently stand in for the other two.

Runtime tuning scans batch sizes `1/2/4`, bucket widths `16/32/64`, and in-flight depths `2/3/4`.
Compiled additionally compares packed and fused FP16 operators; exact compares BF16 and FP32
compute. Each method, destination, and GPU count is tuned independently on program-selection
records. The method's complete selection source is read once to establish the warm page-cache
condition; after correctness, every legal candidate receives one screen pass in seed-73421 order.
The fastest three receive one warmup and three measured passes, and every candidate result is
retained. That point-specific winner is frozen before any 682-record job.

RQ5 uses the 60 program-selection histories and theta11 on one GPU to time three matched paths:
fresh-K/V forward only; the same forward plus FP16 normalized-state device capture; and that
capture plus D2H, encoding, and buffered-POSIX persistence. Each receives one warmup and three
measured repetitions. INT8 is frozen as symmetric signed quantization with one FP32 absmax scale
per record and layer (`scale=max(abs(z))/127`, all-zero scale 1), then timed FP16 dequantization
during host staging. Its semantic certificate is reapplied on certificate users and final behavior
is reported without retuning; its complete-job endpoint is one-GPU HBM over all 682 records.

The time crossover is reported separately for FP16 and INT8 as
`ceil(capture_overhead / (exact - compiled - compiler_amortized))`, in seconds per record. A
nonpositive denominator is reported as no time break-even. Bytes, auxiliary fallback state, or
unknown update frequency are not converted into deployment cost without an external parameter.

## Failure and reader-visible state

The logical reader begins on the previously committed version. Before the theta11 commit it must
remain there; after a successful commit it switches once; after abort theta11 remains invisible.
Guard design uses only program-selection records and no recommendation labels. It selects the
lowest-overhead executable mechanism that detects the frozen integrity-valid theta4 perturbation
and preserves all unperturbed cohort certificates. The artifact records reference bytes/time,
normal no-fault overhead, false escalations, and detection phase. If no runtime sentinel qualifies,
an executable semantic preflight is allowed and the paper must drop the runtime-sentinel claim.
The mechanism is frozen before the six complete failure jobs.

With the Stage-4.6 per-cache lifecycle policy frozen, Stage 5 must test:

| Injection | Required outcome |
|---|---|
| Artifact hash mismatch before begin | Abort before first extent |
| Structurally valid, integrity-accepted semantic perturbation of theta4 program | Semantic preflight or runtime guard detection, exact fallback, complete corrected commit; runtime detection reports replaced records |
| Before first extent | No theta11 visibility |
| Mid-wave | Abort, previous version remains visible |
| During publication | Private staging reclaimed; no partial target |
| After coverage, immediately before commit | Abort, previous version remains visible |

Resume is optional until an extent journal proves at-most-one-wave redo. Atomic abort and
visibility are mandatory even if resume is omitted.
Every failure result records detection phase, abort versus corrected commit, old/current pointer
state, complete/partial target visibility, staging reclamation, cleanup time, and reworked record
count. A boolean “detected” without these state transitions does not complete the gate.

## Result schema and paper map

The final aggregate must validate against
`configs/cohortkv_single_config_v1/result.schema.json`. Raw stage files remain local under
`results/system/cohortkv_single_config_full_chain_v1/`; the aggregate path is
`final_summary_seed0.json`. The schema requires environment/source-cache state, compiler
accounting, the cost-fidelity frontier, exactly one aggregate run for every primary
method/endpoint/GPU combination, all six failure outcomes, capsule economics, and explicit negative
results. It also freezes the workload hash, record/token counts, component input bytes, capacity
preflight, and unpadded lengths/offset correctness, so controls cannot stand in for a missing
primary point. Environment metadata includes all four A40 capacities, source mount/device, software
versions, repository commit, and a hash of the actual code snapshot used by the run.

| Target manuscript slot | Required artifact | Current status |
|---|---|---|
| Table 6 / Figure 5, frozen compiler | RQ2 compiler section | Seed-0 deployed certificate and plans complete; new-seed replication open |
| Figure 6, closest-baseline frontier | RQ3 frontier points | 177-point seed-0 resident frontier complete; no selective action certifies |
| Table 7, operator paths | Correctness and resident operator report | Stage-3 full development distribution complete |
| Table 8 / Figure 7, complete job | RQ4 system runs and breakdown | Stage-4 normalized source loses; Stage-4.5 direct old K/V wins in scoped 1/2/4-GPU hot-HBM regime |
| Repeated-update lifecycle | Stage-4.6 chained migrate-or-exact policy | Balanced policy, independent certificate, and 682-record chain frozen |
| §8.6 escalation/failures | RQ4 failure records and manifests | Faults/visibility frozen; implementation open |
| Figure 8, capsule economics | RQ5 economics section | Fields frozen; implementation open |

No `TBD`, expected ordering, or desired speedup is filled from this document. A slot is updated
only after its result artifact exists; contradicted expectations are deleted or downgraded.

## Stage 1 downstream amendment

Stage 1 completed all 177 resident points and the disjoint selective certificate. Compiled repair
strictly dominates all 53 selective intervals for each source pair. No selective interval passes
the primary three-view contract, so exact is its publishable fallback.

The strongest profiled action is `m=12, layers=0..11` for theta0/theta4/theta10. Downstream system
work retains this action only as a certificate-failed diagnostic external baseline so that the
closest comparison is not hidden when it fails. It starts at layer zero and therefore reads old
FP16 K/V plus raw history, with no transition-hidden shard. Any Stage-4
`selective_contiguous` row must record `certificate_passed=false` and may not be described as a
certified or publishable target. The measured protocol and frozen decision are recorded in
`COHORTKV_STAGE1_FRONTIER_V1.md` and
`configs/cohortkv_single_config_v1/stage1_frontier_summary.json`.

## Stage 2 downstream amendment

Stage 2 reloaded serialized certificate shards and prepared runtime programs for all three source
pairs. Compiled full affine passes the primary 70% contract for theta0/theta4/theta10 at
`0.01651–0.01657x` resident exact cost. The primary fallback chains are `p8 -> exact`, `exact`,
and `p8 -> exact`; targets 50%–80% select compiled and 90% selects exact. The 522 final-test users
remain untouched.

The first materialization attempt also falsified one Stage-0 representation assumption:
unnormalized residual hidden states overflow FP16. The auxiliary residual suffix is therefore
BF16, while the default capsule, runtime program, and published K/V stay FP16. The three checked
plans and summary are under `configs/cohortkv_single_config_v1/`; detailed measurements and claim
boundaries are in `COHORTKV_STAGE2_COMPILER_V1.md`.

## Stage 3 downstream amendment

Stage 3 closes the compiled capsule/operator interface at the destination-ready output boundary.
The FP32-arithmetic transport reference, packed FP16, and fused FP16 now consume the same dense
length-bucketed capsule and write through one `execute_into` API into separate contiguous,
unpadded FP16 `[layers, valid_tokens, kv_width]` K/V extents with lengths and offsets.

All nine batch/bucket layouts were checked on the complete 60-record program-selection
distribution: 88,085 tokens and 1,443,184,640 valid FP16 K/V elements per layout. Every source and
dense-output padding value is zero, every dense-to-extent comparison is exact, and packed/fused
both pass the frozen transport tolerance against reference.

The 18-candidate resident screen freezes fused FP16, batch 4, and bucket width 32 as the default.
Its three full-distribution samples are `30.142/31.070/31.154 ms`; the fastest packed control has
a `61.970 ms` median. Every fused sample is below every packed sample, for a `1.995x` median
advantage. The exact bucket choice remains a Stage-4 retuning input rather than a robust ordering
claim among the close fused finalists.
This is resident operator evidence only. Stage 4 subsequently retunes every
method/destination/GPU point over the full Stage-0 grid and includes source read, allocation,
movement, publication, and commit.
The checked result is in `configs/cohortkv_single_config_v1/stage3_operator_summary.json`; details
are in `COHORTKV_STAGE3_OPERATOR_V1.md`.

## Stage 4 downstream amendment

Stage 4 materialized 2,523 hash-checked source shards and executes all 682 records through one
lazy-reader, extent, destination, and manifest interface. The frozen matrix contains compiled,
selective, exact, residual-p, and no-transform at HBM/DRAM destinations over 1/2/4 GPUs. All
30 points independently tune the Stage-0 runtime grid, pass complete-workload correctness and
capacity preflight, retain three timed repetitions, and publish complete duplicate-free
manifests.

Compiled remains 2.70–3.49× faster than the frozen certificate-failed selective diagnostic, but
it loses to exact at all six matched endpoints. Compiled HBM takes
`27.083/18.943/13.707 s` at 1/2/4 GPUs versus exact at
`18.881/9.644/5.742 s`; DRAM takes `22.567/12.231/15.662 s` versus
`18.886/9.391/5.448 s`. Source read/decode/pinning consumes `91.35%–96.91%` of compiled
completion. The 17.82-GB physical FP16 capsule source is therefore the measured blocker, while
exact reads only 89.1 MB of physical raw-history shards.

This negative endpoint result does not change the Stage-2 semantic recovery or Stage-3 resident
operator result, and it remains the frozen cold-source control. Details are in
`COHORTKV_STAGE4_SYSTEM_V1.md`.

## Stage 4.5 downstream amendment

Stage 4.5 closes a separate existing-old-K/V hot-HBM operating regime. It composes the frozen
capsule affine through the source model's stacked K/V projection and directly transforms the
already resident serving old K/V. The three FP16 programs total 100.78 MB; no extra per-record
source state or `Norm(x)` remains. Old extents are retired only after their replacements are
accepted.

All three source pairs retain the certificate, with 0.8810 minimum worst-view recovery and
0.0368× maximum certificate cost. Complete real fused transport covers all 682 records and
17.82 billion valid elements with zero tolerance mismatches. Complete 1/2/4-GPU jobs take
0.930/0.494/0.255 seconds versus paired HBM-resident raw-history exact at
18.695/9.729/4.766 seconds. Every compiled sample is below every exact sample.

This result does not rewrite the Stage-4 normalized-source failure and does not claim cold
filesystem or SSD speedup. Its equivalence and certificate begin from exact source-version K/V;
the result alone does not justify feeding a migrated output through a later adjacent program.
Stage 4.6 now addresses that boundary separately. The source-policy decision routes
capacity failure, missing old K/V, or an unverified program to exact; making the lifecycle policy
and fallback transactional and failure-safe is Stage 5.
Details are in `COHORTKV_STAGE4_5_SOURCE_PLAN_V1.md`.

## Stage 4.6 fixed lifecycle amendment

Stage 4.6 is one algorithm/lifecycle experiment, not another systems matrix. It fixes KuaiRand
4+12, seed 0, 16L/H512, one A40, hot HBM, and the 12 checkpoints theta0–theta11. The complete
workload starts with exact theta0 K/V and executes 11 recursive updates. At each update every
record is either directly migrated or exactly recomputed.

The first explored router was:

```text
exact if migration_depth >= max_migration_depth or risk_score >= risk_threshold
migrate otherwise
```

Its norm sketch had acceptable cumulative evidence but caused exact-refresh waves from 0.15% to
65.1% on the diagnostic full chain, so it was rejected. The frozen policy instead ranks the 11
edges by label-free fit one-hop cache error into exact budgets
`25/19/15/17/22/16/20/18/23/24/21%`; within each edge it refreshes depth-four caches first, then
older caches, with stable SHA256 tie-breaking. It computes no speculative candidate for routing.

The independent 60-record certificate costs 0.2142× all-exact cumulatively and passes every
fidelity, balance, peak-cost, and depth check. The full 682-record chain costs 0.2134×, has
0.2543× maximum step cost, minimum cache/score/top-100 values
0.9632/0.999950/0.9918, and actual exact fractions 14.956%–25.073% after record rounding. All
7,502 per-cache lineage rows rebuild from the policy and consume the previous actual output.
Every step also reports MeanRank, Catalog AUC, NDCG@100, Hit@100, state fidelity, exact fraction,
migration depth, and measured GPU cost relative to all-exact. No 1/2/4-GPU or HBM/DRAM grid is
part of Stage 4.6.

## Stage 0 completion

Stage 0 is complete because:

- the exact record set, roles, source assignments, upstream hashes, and target bytes are frozen;
- every action has a declared source representation and common physical tier;
- residual-p auxiliary state and the selective baseline's incompatible legacy helper are explicit
  instead of deferred implementation surprises;
- HBM/DRAM layout, allocation, visibility, and timing boundaries are explicit;
- RQ grids, tuning/final roles, failure points, output paths, and aggregate schema are fixed;
- the paper slots map to artifacts without creating a result claim.

Stages 1–4.6 have implemented and frozen the resident frontier, deployed compiler path, common
capsule/operator extent interface, full-cohort normal-path engine, and scoped direct-old-K/V
source plan plus continuous migrate-or-exact lifecycle. Guard/fallback and failure semantics are
the next open stage.
