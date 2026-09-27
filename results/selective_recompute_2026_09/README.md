# Four-baseline preparation record — 2026-09-26

**Status: complete — all15 edges,60 curves and12 figures; exit code0.**
Finished2026-09-26 13:58:50 Asia/Shanghai. Outputs include401,187 evaluated
requests,415 plot points (including Full/Reuse references), and
[12 figures with their source data](../../figures/out/selective_recompute_2026_09/).
The former tmux session was `evokv_selective_recompute_20260926`. The
[launch record](launch_record.json) preserves authorization and frozen sources;
`status.json` and `runtime.log` hold live progress. No paper text is changed.
[Programs](../../scripts/selective_recompute_2026_09/README.md)
and [configuration](../../configs/selective_recompute_2026_09/plan.json).

## Question and population

Evaluate layer-, tail-, deviation- and query-guided recomputation on all15
selected adjacent edges (three scales × five releases). Each edge has its own
fixed3,000-user evaluation panel,32 disjoint calibration users and16 disjoint
canary users. The four methods and all their budgets use identical requests.
The final request count is401,187 across15 panels.

The user explicitly chose a diagnostic population with observable Full–Reuse
loss. Existing Full/Reuse outcomes determine panel membership: reserve control
UIDs by fixed hash, then rank the remaining users once by mean class-balanced
pairwise-concordance advantage and retain the first3,000. No new baseline
scores enter this choice, and no favorable resampling is performed. This is an
**outcome-conditioned diagnostic**, not an unbiased population effect estimate.
The selection changes both label mix and gap size. Population and selected
metrics, source hashes, exact UIDs and all selected requests are preserved in
[the panel record](panels/README.md). Full/Reuse are reaggregated on each selected
panel; full-population AUC is never used as that panel's denominator.

## Execution semantics

- Parent K/V is built at the edge's cutover. New observed events use Current
  parameters; all queries at a timestamp precede that timestamp's event appends.
  Cache capacity is1024 and eviction occurs before appending to a full cache.
- Each scored request starts from this common rolling Reuse state. Repair is
  **request-local and ephemeral**: score with the repaired view, then discard
  it. Repaired views do not change later requests' serving cache. This evaluates
  single-edge read repair, not persistent multi-release adaptation.
- Layer replay uses actual captured producer boundary inputs. It replays a
  contiguous interval chosen on separate calibration users by teacher-logit
  MSE over every interval. No evaluation labels or evaluation baseline scores
  select layers. Teacher acquisition, reads and interval search are charged.
- Tail replay selects the most recent fraction of retained events and executes
  every Current layer conditioned on the unchanged inherited prefix.
- Deviation selects tokens using squared layer0 K/V difference between Current
  raw-input projections and inherited layer0 K/V. Query selects tokens using
  mean absolute first-layer native attention from the current request query.
  Both replay selected rows through all Current layers, attending to causally
  preceding updated selected rows and inherited unselected rows. They never
  splice exact teacher K/V into evaluation states.
  Deviation scores below `(32×dtype_epsilon)²×mean(K²+V²)` are zero ties,
  resolved chronologically. This fixed numerical rule prevents rounding noise
  in already-current first-layer rows from changing the selected positions.
- Repair raw inputs align with the serving-cache token order. Their time
  features are recomputed for the retained window (first delta0, raw interior
  gaps), while inherited K/V retains its original producing-time features.
  The saved Full anchor independently follows the original Full collator's
  raw-item tie order. Canary checks compare both saved controls separately and
  report any complete-repair versus saved-Full difference explicitly.
- Initially short histories use PyTorch throughout their singleton trajectory
  to avoid compiling a new Triton kernel at every growing length. Full-length
  user cohorts use Triton. Per-request and append histograms retain this backend
  distinction, and the FLOP ledger charges the corresponding executed shapes.
- Four GPU workers partition users by request count. Within each worker,
  grouping by initial history length and event count reduces ragged batches.
  Requests between observed events retain their causal order; consecutive
  events without an intervening request use native rolling-window bands of
  1,2,4,8,16 or32 events. Their actual band arithmetic enters the cost ledger.
- Models, history loading and ordinary Reuse replay are shared across the four
  methods. Each request computes a selector/ranking once for its nested budgets,
  and the query reader avoids copying the entire history cache. These execution
  savings do not remove per-method selector costs from the standalone ledger.

## Budgets and reporting

Token fractions:1/32,1/16,1/8,1/4,1/2. Layer counts are unique ceilings of
L×{1/16,1/8,1/4,1/2,3/4}, capped below the full stack. The already-measured
Current Full and Reuse are control endpoints. Each actual method point records
its selected interval/token count and arithmetic, not just its budget label.

Recovery=(method AUC−Reuse AUC)/(Full AUC−Reuse AUC). Relative compute equals
the method's extra FLOPs over Reuse divided by Full's extra FLOPs over Reuse.
The latter subtracts the ordinary Current append work that Full's per-request
reconstruction avoids; common initial Parent K/V and ordinary query reads
cancel. All selector work, repair work and method-specific profiling are
included. A shared experimental selector is charged in full for every
standalone deployment point. FLOPs use executed dense/tiled matrix shapes;
multiply-add=2, elementary operations=1. Sorting comparisons, memory and
elapsed time are recorded separately from FLOPs.

All15 edges and all declared budget points remain, including negative recovery
or values above100%. No clipping, envelope selection, favorable-edge filtering
or synthetic measured curves. Final output is12 figures (one scale per method,
five labeled release edges per figure), raw points CSV and source metadata.

## Validation and execution state

Focused CPU checks cover sparse replay against the native sequential reference,
zero/full endpoints, selected/unselected rows, query/deviation selectors,
timestamp ties and causal append/evict, short histories, immutable ephemeral
repairs, AUC cohort contributions, and arithmetic against actual CPU matmul
operators. GPU canaries use held-out users and real selected weights; synthetic
maximum-shape probes measure memory/speed only and are not quality evidence.

The final focused suite passed54 tests. [Readiness](readiness.json) binds the
execution sources and [canary evidence](probes/four_gpu/canary.pass.json).
The formal queue writes separate method outputs and seals completed user
units for resume. Maximum-shape stress covers the configured
1024-token context and user/query batches, with a70% PyTorch allocation cap;
two repeated cycles check that transient tensors are released. This is a
bounded resource check, not a claim of testing every possible long-run workload.

The saved Full and Reuse controls reproduce within7.16e-7 logit error on the
three-scale development canaries. On Max, rebuilding the cache-aligned inputs
differs from the saved Full path by up to0.02301049 because those input paths
order simultaneous events differently. This was already present before the
performance changes. Complete repair equals a rebuild of its own aligned
inputs; it is not asserted to equal the retained Full endpoint. All plotted
Full anchors remain the retained scores on the exact evaluated request IDs.

Development fixes retained under `probes/development_failures/`: the first
integration check exposed a missing auxiliary `weight` field in the Full
collator interface; supplying unit weight for this score-only control fixed it.
The next check passed Medium numerical controls but exposed repeated Triton
compilation for a growing short history. The PyTorch short-history path removes
that compilation cost. Both changes have focused regression checks.

The9,478-row optimization comparison passed for layer, tail and query repair,
but exposed536 deviation rows with unstable near-zero rankings. The original
[failed comparison](probes/four_gpu/optimization_check.json) remains intact.
The fixed precision-relative zero-tie policy was checked on affected actual
requests in all three scales: all15 selected-position arrays matched between
scalar and32-event replay, with maximum logit difference4.77e-7. The
[resolution](probes/four_gpu/optimization_resolution.json) records exact cases,
source hashes and the limited rerun scope. The full small canary population
was not rerun after this localized fix; the earlier four-card memory checks
and unchanged controls are retained, with the added arithmetic counted.

## Measured performance and resource estimate

The same three V0→V1 development panels, four workers per scale, gave the
following timings before and after band replay, cache-copy removal and grouping.
These are small-sample worker compute timings, including extra numerical
controls; they are not claimed as whole-job speedups.

| Scale | Slowest worker before → after | Reduction | Peak allocated / reserved GiB |
| --- | --- | --- | --- |
| Medium | 9.52s →6.51s | 31.6% | 12.51 /20.08 |
| Large | 29.05s →20.93s | 27.9% | 20.12 /29.80 |
| Max | 15.92s →6.53s | 59.0% | 20.37 /27.69 |

Each GPU has44.42GiB. The largest observed allocator reservation is67.1%;
maximum configured shapes passed on all four GPUs without batch reduction,
and retained allocation did not grow between the two repeated stress cycles.
Removing full-cache copies made the warm query reader about3× faster in the
Medium/Large batch probes. Four workers retain the70% allocator cap and halve
batches on an OOM; no users or budget points are discarded.

[Resource estimate](resource_estimate.json): all15 edges contain401,187 requests
and13,782,376 intervening event appends. Extrapolating measured rates, adjusting
history/append workload and including load/calibration gives about10.4h on four
GPUs. Use a broad5–21h planning range: the canaries have few users, while formal
cohorts offer more batching and omit the canary's repeated Full controls.
Progress reports will replace this extrapolation with completed-unit rates.

An interrupted-after-unit resume was simulated on Medium rank2. The worker
reused its sealed unit; every baseline output and the unit seal retained both
its content hash and modification time. The queue checkpoints every256 users
per worker and reports at edge boundaries or every two hours.
