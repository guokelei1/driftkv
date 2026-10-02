# Read-correction results and development history

## Current selection — 2026-09-30

Current Motivation uses **nonlinear Q-v5** (`cross_phi_joint8`, C128/256/512,
45 points) and the unchanged **nonlinear H-v4** (`map_all`, C64/128/256,
45 points). See [the final result entry](motivation_final/README.md).
The current 90-point figures remain under
`figures/out/read_correction_2026_09/motivation_final/`; the `v5/` figure set
retains all 135 historical comparison points.

The v1/v2/v3 and Max-diagnostic runners and the independent v4 queue are
retired. Shared fitting, reading and cost code remains in use. Historical
hashes and results are retained; old queues cannot resume against those
hashes in the cleaned source tree, while saved results remain plottable.
The bound historical Q-v2/H-v4 comparison input and script remain under
`motivation_final/analysis/inputs/` with their original hashes.

| Record | Scope |
| --- | --- |
| [Current Q-v5/H-v4](motivation_final/README.md) | Selected 90 points, method settings, results and figure links |
| [v2](v2/README.md) | Affine Q with output distillation; expensive history-interaction H |
| [v3](v3/README.md) | Fixed history structure with three fitting-cost endpoints |
| [v4](v4/README.md) | Nonlinear temporary K/V mapping; source of selected H |
| [v5](v5/README.md) | Nonlinear query features; source of selected Q |
| [Max diagnostics](v5/max_diagnostics/README.md) | Completed explanation probes and all counterexamples |

The sections below describe the completed **historical v1** experiment, not the
selected nonlinear Q-v5/H-v4 methods. Its standalone figures are retired;
retained summaries and scores remain the evidence.

## Historical v1 run

Status: **v1 exploration completed successfully on 2026-09-27 at 23:56 CST.**
Preparation passed17 focused tests,500 real rolling canary requests and the
resume check. The queue covers15 edges and401,187 requests on GPU0/1/2/3.
All15 edges and30 curves completed in7,099.68 seconds (1h58m20s), producing
90 measured points. `exit_status.txt` is0. Execution records
are in `status.json`, `launch_record.json` and `runtime.log`.

## Completed v1 results

The table uses an equal-weight mean across all15 edges, without selecting a
different calibration budget for each edge. Recovery is
`(method AUC - Reuse AUC) / (Full AUC - Reuse AUC)`; costs include fitting and
extra inference FLOPs relative to the existing Full-minus-Reuse denominator.
C is the number of calibration users, separate from the3,000 evaluation users.

| C | Query-only recovery | History-conditioned recovery | Query-only cost | History-conditioned cost | H positive edges |
|---:|---:|---:|---:|---:|---:|
| 32 | -20.17% | 29.54% | 0.29% | 3.91% | 11/15 |
| 128 | -23.42% | 44.88% | 1.14% | 6.79% | 15/15 |
| 512 | -17.27% | 53.01% | 4.58% | 18.37% | 15/15 |

History-conditioned correction beats query-only at the same C on42/45 points,
and has positive recovery on41/45 points. At C128 its mean recovery/cost is
66.39%/7.73% for Medium,54.27%/6.36% for Large, and13.99%/6.29% for Max.
The Max V4-to-V5 recoveries at C128/C512 are only0.24%/0.14%.

Query-only has positive recovery on17/45 points and at least one positive
budget on7/15 edges. One useful low-cost example is Medium V3-to-V4 at C32:
49.75% recovery at0.32% cost. This supports exploring the idea on some edges,
but does not establish broadly reliable query-only recovery.

Additional explicit user history shows a stronger recovery signal. It costs
more than query-only, but strong recovery does not always require a large
fraction of Full compute: Medium V0-to-V1 at C32 recovers99.72% at4.59% cost.
Individual curves are not necessarily monotonic. Negative and above100%
recoveries remain unmodified in `summary.json` and the plots. These are the
authorized development-panel results; no population or untouched-test claim is
made, and the paper has not been changed.

## 2026-09-28: cost interpretation and clearer figures

The original "Measured cost range" inset repeated the same points with an
independently enlarged axis. It was not another experiment. Its different
axis ranges made visual comparisons between Q and H confusing. The historical revision used a shared 0–25% cost axis and separated
calibration from inference. Those standalone figures are retired; the 90
measured points remain in `summary.json`. Current figures use the selected
Q-v5/H-v4 results linked above.

All90 measured points were checked against the recorded history histograms,
module configurations, calibration costs and Full-minus-Reuse denominator.
No missing H history-scan arithmetic was found. The equal-edge mean at C128
is1.14% for Q and6.79% for H. Their inference portions are0.00150% and2.93737%
respectively; the remaining cost is calibration. At N1024, per-request H
arithmetic is about2036 times Q correction arithmetic, but only2.5–3.2% of
a full history rebuild. H uses a small head-local network on existing K/V;
Full rebuilds all historical tokens through the backbone. C32/C128/C512 varies
the calibration-user count and fitting cost, not the inference architecture.

These are analytical FLOP estimates, including declared fitting estimates.
They do not measure latency or charge GPU memory traffic as FLOPs. Existing
synthetic resource probes show that H's full corrected-query latency is about
4.5–4.7 times Q's, but those probes are not a per-method latency measurement
of the completed3,000-user evaluation.

The v1 Q candidate had weak overall recovery. Its low training response MSE
did not establish good final ranking, and its calibration inputs differed from
the rolling request distribution. The completed
[post-fit diagnostic](query_only/development/postfit_probe_20260928/README.md)
varied correction strength and active layers on fixed 128-user subsets;
all diagnostic variants remain separate from complete-panel results.

## Scope

- Two independently stored methods: query-only affine correction and correction
  with an additional per-query full-history user-information branch.
- All 15 adjacent edges of the retained Medium/Large/Max V0–V5 models.
- Same outcome-conditioned 3,000-user panels and requests as the completed four
  selective-recomputation baselines. The user explicitly permits using these
  panels to compare and improve candidate methods; all new quality results are
  **development exploration**, not untouched qualification or a population estimate.
- Fitting users are disjoint from the union of the five evaluation/canary panels
  at each scale. All fitting histories precede release; candidate teachers use
  no feedback labels. Calibration budgets are nested C32/C128/C512.
- No quality gate. Every edge and signed outcome is retained. A useful signal on
  one edge can motivate another candidate without implying success on all edges.

Code and tests live under their method-specific directories; shared replay,
aggregation and cost code do not change the retained baseline execution sources.
Outputs are separated as
`<method>/development/<revision>/<scale>/<edge>/calibration_c*.{pt,json}`
and `rank0/shard_*.parquet`. Large weights and raw scores remain local.

## Development history

1. **v0**: both branches implemented. Three-scale C32 calibration and full rolling
   canaries completed; source, weights and all scores retained in
   `probes/v0/canary/`. Full/Reuse/zero-correction errors were at most 7.16e-7
   logits. These tiny canary groups are numerical checks: some have a nonpositive
   Full–Reuse gap or only one label class, so their normalized quality ratios do
   not establish recovery. The v0 calibration reader FLOP counts have a small
   known overcount; the main v1 calibration uses the corrected formulas.
2. **Resource checks**: four GPU checks covered configured 1024-token histories,
   evaluation cohorts and H backward minibatches. Batch sizes were increased
   after the first measurement. Final evaluation/query batches are 256/128/64,
   calibration batches 32/16/8. The largest measured reservation was 27.80 GiB
   of 44.42 GiB; allocator limit remains 70%. Repeated fixed-shape reads retained
   no growing allocations. These checks measure feasibility, not quality.
3. **v1**: per-head calibration-target RMS normalizes the H training output units.
   Saved output weights are folded back to original units. The inference
   function is unchanged; this addresses very different target scales and
   optimizer progress across layers. Original and normalized losses are kept.

The full fitting membership has been checked for valid pre-release history:
15×512 memberships, none empty. Preflight checked141 bound input files,
including the retained weights and request panels.

The initial128-user Medium V0→V1 development subset contains9,981 requests.
At C32, Q recovered35.91% and H116.11%, at0.71% and6.08% relative cost.
At C128, Q recovered−5.13% and H109.63%, at2.85% and14.46% cost.
All points remain in `probes/v1/development128/`. These users were the fixed
panel's most-request-active128 users for a throughput probe, so this is early
development signal, not the complete3,000-user result or a cross-edge conclusion.

The prelaunch estimate was 3.31h, with a 1.66–8.28h planning range;
`resource_estimate.json` retains its measurements and arithmetic. The actual
completed runtime is reported above.

## Quality and cost

Recovery and relative FLOPs use the same Full–Reuse denominators and request IDs
as the four-baseline experiment. Calibration cache construction, reads, fitting
and extra inference arithmetic are charged for every standalone method/budget.
Backpropagation, optimizer and solve counts contain explicitly identified
analytical estimates; they are not hardware performance-counter measurements.
Ordinary query reads are a common term. Existing user-history reads are not
charged again as an extra input; H pays for its additional query–history branch.

K/V is never rewritten by correction. Queries remain before same-time appends,
and observed events use the unmodified Current append/eviction path. Calibration
teachers use the cache-aligned event order; saved Full retains its original
input order. The known simultaneous-event ordering difference is not hidden by
regenerating Full scores.

## Execution and reports

The queue assigns one complete edge to each of GPU0/1/2/3, including calibration
then rolling evaluation, and takes the next edge when a GPU finishes. This
keeps calibration as well as evaluation parallel. Each method uses identical
requests; user units are sealed for resume. Quality does not stop the queue.
Numerical, input or runtime failures are reported and stop further execution.

The original v1 standalone plots are retired. Complete raw points remain in
`summary.json`; current figures are linked at the top of this page. The selected
v1 H-C512 comparison also remains in the 135-point v5 overview.

See [the exploration plan](../../docs/design/read_correction_probe_2026_09.md)
and [scripts](../../scripts/read_correction_2026_09/README.md).
