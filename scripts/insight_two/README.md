# Medium Insight 2 functional-boundary discovery

This directory is isolated from serving transitions. It studies one
`Parent -> Current` edge at a time and keeps three evidence classes separate:

1. **representation oracle** — where the Transformer reader first exposes a
   compact, causally sufficient response;
2. **persistence oracle** — whether a cutover response remains useful under
   real requests and Current appends;
3. **executable estimator** — whether that response can be generated from
   Parent state, Current parameters and pre-cutover evidence within 20% of
   Exact-All.

Exact stage deltas, release-basis coefficients and temporal coefficients are
diagnostics only. They never enter an executable frontier.

Current status: all tested KV-only estimator families are retired. The commands
below preserve reproducible route eliminators and diagnostics; none authorizes
population expansion, confirmation, or a frozen Design 1.

The fixed paired native-response route eliminator is:

```bash
PYTHONPATH=src:scripts python \
  scripts/insight_two/run_paired_native_response_preflight.py --device cuda:2
```

It keeps the two r4 factor arms and subtracts their responses after native
activation/aggregation.  Its UID-1930 result is a NO-GO against both
single-Current r8 controls; it is not authorized for population expansion.

The fixed defect-coordinate route eliminator is:

```bash
PYTHONPATH=src:scripts python \
  scripts/insight_two/run_defect_first_replay_preflight.py --device cuda:2
```

It assigns rank 2 to the Parent base and rank 4 to the finite-release defect,
with no rank sweep.  Its UID-1930 result is also a NO-GO and is not authorized
for population expansion.  Existing paired-native control cost and quality
remain authoritative in the dedicated runner above.

The fixed source-certificate route falsifier is:

```bash
PYTHONPATH=src:scripts python \
  scripts/insight_two/run_source_residual_closure_preflight.py --device cuda:3
```

It evaluates the frozen absolute-source and finite release-defect residual
closures at rank 4 with identical DEIM rows.  Both stay below 20% and satisfy
their algebraic full-rank limits, but both fail the fixed UID-1930 five-edge
quality gate and are dominated by the paired-native and single-r8 controls.
Their sampled-residual/DEIM core is also not a novel Transformer mechanism.
They are retired with no population expansion or rank/pivot/lift tuning.

The exact producer-state/read-version commutator diagnostic is:

```bash
PYTHONPATH=src:scripts python \
  scripts/insight_two/run_producer_reader_commutator_preflight.py --device cuda:3
```

It evaluates all four `F(reader, cache_producer)` paths with each reader's own
query semantics.  The near-commutation observation is oracle-only: its reverse
cross path reads Exact Current K/V and its endpoint formula mixes per-candidate
scores, so it is explicitly not a migration action or population authorization.

The activation-topology falsifier is:

```bash
PYTHONPATH=src:scripts python \
  scripts/insight_two/run_activation_boundary_replay_preflight.py --device cuda:3
```

It separates stable same-region deformation from branch crossings. Stable
topology does not imply a stable functional response, and native Current graph
discovery already exceeds the 20% budget. The route is retired.

## Tests

```bash
PYTHONPATH=src:scripts python -m pytest -q \
  tests/test_insight_two_functional_boundary.py \
  tests/test_insight_two_paired_native_response.py \
  tests/test_insight_two_defect_first_replay.py \
  tests/test_insight_two_source_residual_closure.py \
  tests/test_insight_two_producer_reader_commutator.py \
  tests/test_insight_two_activation_boundary_replay.py
```

## Completed representation diagnostics

The rank-0/rank-r boundary canary and 512-user discovery are run with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:scripts \
  torchrun --standalone --nproc_per_node=4 \
  scripts/insight_two/run_low_rank_canary.py --scope discovery

PYTHONPATH=src:scripts:scripts/insight \
  python scripts/insight_two/adjudicate_discovery.py
```

The authoritative discovery adjudication is `analysis_v2`; `analysis` is
retained with `INVALIDATED.md` because it incorrectly used gap weighting for
the primary aggregate.

The UID-disjoint release-basis ceiling uses:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:scripts:scripts/insight \
  torchrun --standalone --nproc_per_node=4 \
  scripts/insight_two/run_release_basis_diagnostic.py --scope discovery

PYTHONPATH=src:scripts:scripts/insight \
  python scripts/insight_two/adjudicate_release_basis.py --scope discovery
```

## Completed executable-estimator canaries

The parameter-map/compact-carrier estimator and its time-aligned correction
both failed their focused scientific gates, so neither has a 512-user run.
The final authorized no-new-predictor family is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:scripts:scripts/insight \
  torchrun --standalone --nproc_per_node=4 \
  scripts/insight_two/run_tail_functional_canary.py

PYTHONPATH=src:scripts:scripts/insight \
  python scripts/insight_two/adjudicate_tail_functional.py
```

It performs dependency-closed Tail-128 replay, costs 18.28%–19.02% of
Exact-All and is retired by its preregistered stop rule after a negative canary.

## Temporal diagnostics

The frozen-cutover persistence observation is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:scripts:scripts/insight \
  torchrun --standalone --nproc_per_node=4 \
  scripts/insight_two/run_temporal_persistence_diagnostic.py --scope discovery

PYTHONPATH=src:scripts:scripts/insight \
  python scripts/insight_two/adjudicate_temporal_persistence.py --scope discovery
```

The follow-up oracle asks whether one global or one-per-layer scalar tracks the
Current request correction along the cutover response direction:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:scripts:scripts/insight \
  torchrun --standalone --nproc_per_node=4 \
  scripts/insight_two/run_temporal_coefficient_diagnostic.py --scope discovery

PYTHONPATH=src:scripts:scripts/insight \
  python scripts/insight_two/adjudicate_temporal_coefficient.py --scope discovery
```

The 512-user result rejects both coordinate hypotheses: global/layerwise
recovery is 48.96%/65.04%, although all five edges remain positive. The next
diagnostic therefore preserves query-addressability instead of adding mapper
capacity. It evaluates a fit-free signed chronological K/V coreset:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:scripts:scripts/insight \
  torchrun --standalone --nproc_per_node=4 \
  scripts/insight_two/run_signed_response_coreset_diagnostic.py --scope canary
```

This coreset reads Current Exact K/V during construction and is oracle-only.
Its purpose is to determine whether the cross-version attention-response
operator can be read by real Current queries through the native kernel; it does
not authorize the subsequent sparse causal-replay constructor.

All outputs are raw-first and refuse overwrite. Confirmation users `[512,3000)`
remain unread until an executable Design 1 is frozen.
