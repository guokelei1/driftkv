# Fixed cheap projection plus deepest suffix-N full layers

> Historical structural-baseline record. The measurements remain valid within this protocol, but
> the suffix route is not the active D1 method.

This is the corrected evaluation of the latest layer-wise method. It is not a per-user reuse
policy and does not use JVP drift estimation.

The quality results below remain valid within this protocol, but the legacy full-block timing has been superseded by
the exactly equivalent terminal-projection operator in [INTERVAL_ORACLE.md](INTERVAL_ORACLE.md).

## Operator

For a cache produced by an old model:

- Every cheap layer applies the current `Wk` and `Wv` projections to that layer's cached old
  `Norm(x)` state.
- The deepest contiguous top-N region begins at the cached hidden state at the cheap/full split.
  All but its terminal layer run current full HSTU blocks; the terminal layer runs current
  `Norm + Wk/Wv` only.
- `top-N = number of layers` begins from current input embeddings and is exactly a current-model
  full K/V recomputation.
- Serving consumes the migrated prefix K/V cache and processes the latest behavior token with the
  current model.

The reusable operator is implemented in `src/hstu_kvcache/migration/layerwise.py` and evaluated with
the repaired incremental-serving protocol. It does not use a full-history stale-K/V forward,
padded `h[:, -1]`, invalid training targets, or a hand-written cost constant.

## Focused experiment

- Data: KuaiRand-1K, 14 base days followed by streaming three-day updates.
- Model: simplified six-layer HSTU, hidden size 96, four heads, head dimension 24, sequence length
  128, 5,000-item catalog, 770,496 parameters.
- Evaluation: 300 users per seed, full-catalog engaged-positive ranking, seeds 0-3.
- Staleness: fixed-input cumulative cache age from theta-0 to theta-1, theta-3, and theta-5.
- Time: CUDA-event measurement of GPU-resident cache migration, normalized to full prefix K/V
  recomputation for the same batch.
- Statistical unit: training seed; intervals are paired 95% t intervals over four seeds.

The short theta-0 to theta-1 cell is noisy because fresh recomputation itself has little benefit.
The table therefore reports the two cache ages where the method is identifiable. Quality entries
are recovery of the fresh-over-reuse gain in best rank and NDCG@100.

| Configuration | Time / full | Extra state / K/V | theta-0 to theta-3 rank / NDCG recovery | theta-0 to theta-5 rank / NDCG recovery |
|---|---:|---:|---:|---:|
| cheap all | 0.187 | 50.0% | 50.8% / 24.3% | 72.9% / 67.1% |
| cheap + top-1 full | 0.229 | 50.0% | 51.0% / 27.1% | 73.2% / 67.3% |
| cheap + top-2 full | 0.372 | 41.7% | 65.3% / 54.0% | 84.6% / 67.8% |
| cheap + top-3 full | 0.504 | 33.3% | 73.3% / 51.4% | 93.3% / 75.9% |
| cheap + top-4 full | 0.635 | 25.0% | 87.5% / 64.7% | 101.3% / 90.5% |
| cheap + top-5 full | 0.767 | 16.7% | 88.8% / 84.7% | 103.0% / 99.7% |
| full recompute | 1.000 | 0.0% | 100.0% / 100.0% | 100.0% / 100.0% |

Absolute fresh-over-reuse best-rank gains are 41.68 at theta-3 and 85.32 at theta-5. Cheap-all
recovers 21.18 and 62.24 ranks respectively. Top-5 recovers 37.02 and 87.86 ranks. In this table,
paired differences around 100% include zero, so they establish neither superiority nor equivalence.
Full recomputation is the version-consistency reference rather than a guaranteed task-metric upper
bound.

The paired comparisons sharpen the interpretation:

- Top-1 adds only 0.073 ranks over cheap at theta-3, with interval `[-0.027, 0.172]`, and 0.190
  ranks at theta-5, with interval `[0.050, 0.330]`. Its NDCG@100 changes are indistinguishable
  from zero. Terminal optimization reduces its cost substantially, but it remains a weak quality
  operating point.
- Top-2 and deeper configurations add reproducible best-rank recovery over cheap. At theta-3,
  top-2 adds 6.02 ranks with interval `[1.54, 10.49]`; at theta-5 it adds 9.95 ranks with interval
  `[4.53, 15.36]`.
- Top-5's paired intervals against full include zero at both theta-3 and theta-5 on Best Rank and
  NDCG@100, while it uses about 77% of optimized full-recompute compute. This is not an equivalence
  result.

Relative stale K/V error rises consistently from the first to the sixth layer: at theta-3 the
layer means are `[0.250, 0.446, 0.589, 0.658, 0.688, 0.826]`; at theta-5 they are
`[0.276, 0.453, 0.593, 0.683, 0.730, 0.886]`. This supports using the deepest contiguous suffix
rather than an arbitrary fixed subset.

## Interpretation

The core route is viable at this scale. Cheap-all is a strong low-cost endpoint, and adding a
deep full suffix gives a meaningful, measured quality-cost curve. The six-layer sweep is more
informative than the original three-layer check because it exposes several intermediate points.

There are also concrete negatives:

- Top-1 is structurally weak for a K/V-only cache: only its current norm changes the last-layer K/V
  relative to cheap. The old full-block implementation exposed this waste; the current operator no
  longer executes that block output.
- The method requires cached normalized states and one split hidden state. The table accounts for
  their capacity under FP16 storage, but not host-device transfer, allocator overhead, or cache
  admission cost.
- Timings measure GPU-resident batched kernels. They are not yet an end-to-end serving latency or
  memory-bandwidth result.
- Cumulative theta-0 is a controlled cache-age stress test, not a rollout with organically mixed
  cache versions. The simplified model, 5,000-item catalog, 300 users, and four seeds are sufficient
  for this gate, not for a final scale claim.

Terminal optimization, the held-out contiguous-interval oracle, streaming value control, and the
subsequent KuaiRand scale gates are complete. Current priorities are maintained only in
`docs/08_core_insights_and_roadmap.md`; this historical report does not define the next phase.

## Artifacts

- `src/hstu_kvcache/migration/layerwise.py`: reusable capture and migration operator.
- `tests/test_layerwise.py`: operator equivalence and hybrid-semantics tests.
- `scripts/layerwise_validity.py`: corrected serving evaluation and measured timing.
- `scripts/summarize_layerwise_validity.py`: seed-level paired aggregation.
- `results/validity/layerwise6l_seed{0,1,2,3}.json`: complete per-user six-layer runs.
- `results/validity/layerwise6l_multiseed_summary.json`: aggregate results and intervals.
- `results/validity/layerwise_seed{0,1,2,3}.json`: three-layer one-step and cumulative sanity runs.
- `experiments/validity/INTERVAL_ORACLE.md`: optimized timing and held-out interval decision.
