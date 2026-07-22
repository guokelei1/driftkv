# Terminal projection optimization and contiguous-interval oracle

This follow-up answers two questions left open by the six-layer suffix experiment:

1. Can the terminal full block be removed without changing the migrated K/V?
2. Does an arbitrary contiguous layer interval outperform the deepest suffix at comparable cost?

The answer at the current scale is: terminal projection is a deterministic improvement, while the
held-out evidence does not support replacing the suffix with arbitrary intervals.

## Optimized operator

For a one-based inclusive interval `[s, e]`:

- layers outside the interval use cached old `Norm(x)` with current `Wk/Wv`;
- layers `s..e-1` execute current full blocks so updated hidden states propagate;
- layer `e` executes only current `Norm + Wk/Wv`.

The terminal block's attention, gate, output projection, and residual cannot affect that layer's
prefix K/V and are therefore removed. A full interval `[1, L]` still produces exactly the same K/V
as current-model full recomputation.

All seven optimized suffix depths have K/V maximum absolute difference `0.0` from the legacy
operator on the evaluated batches. Unit tests also verify equality under padding and parameter
changes.

| Configuration | Optimized time / legacy operator | Optimized time / legacy full recompute |
|---|---:|---:|
| cheap all | 0.991 | 0.168 |
| suffix-1 | **0.624** | 0.204 |
| suffix-2 | 0.751 | 0.333 |
| suffix-3 | 0.804 | 0.450 |
| suffix-4 | 0.838 | 0.566 |
| suffix-5 | 0.864 | 0.684 |
| full recompute | 0.891 | 0.891 |

The largest relative saving is on suffix-1 because its only useful work is terminal normalization
and K/V projection. Even full K/V recomputation becomes about 10.9% faster because the last block
output is not needed. For a fair current comparison, costs below are normalized to this optimized
full recompute.

## Search and validation protocol

- Model and data: the existing six-layer validity-v1 KuaiRand setup.
- Cache ages: cumulative theta-0 to theta-3 and theta-5.
- Discovery: seed 0 evaluates all 21 contiguous intervals.
- Held-out validation: seeds 1, 2, and 3 evaluate only selected suffix and non-suffix candidates.
- Timing: GPU-resident CUDA events with five repeats; each run uses 300 evaluation users.
- Statistical unit: training seed. Discovery seed 0 is not used for held-out intervals below.

## Optimized suffix result

The table reports held-out means over seeds 1-3. Rank and NDCG columns are absolute gain over full
reuse. The K/V and recommendation output of each suffix are identical to the previous implementation;
only its migration cost changes.

| Configuration | Time / optimized full | theta-3 Rank / NDCG gain | theta-5 Rank / NDCG gain |
|---|---:|---:|---:|
| cheap all | 0.186 | 22.12 / 0.00062 | 67.55 / 0.00527 |
| suffix-2 | 0.371 | 29.15 / 0.00180 | 78.46 / 0.00514 |
| suffix-3 | 0.503 | 32.62 / 0.00171 | 85.74 / 0.00554 |
| suffix-4 | 0.635 | 39.34 / 0.00241 | 92.40 / 0.00643 |
| suffix-5 | 0.767 | 39.91 / 0.00301 | 92.62 / 0.00683 |
| full recompute | 1.000 | 46.24 / 0.00352 | 90.29 / 0.00702 |

Suffix-5 exceeding full on theta-5 Best Rank does not by itself establish superiority: full is the
version-consistency reference, and the paired evidence in this cell does not separate their task
quality. Across discovery and validation seeds, the normalized timings are highly stable: cheap
`0.186-0.187`, suffix-2 `0.371-0.372`, suffix-3 `0.503-0.504`, suffix-4 `0.635`, and suffix-5
`0.767-0.768`.

## Do non-suffix intervals help?

Two same-cost comparisons directly test moving propagation away from the deepest layers. Each row
is non-suffix gain minus suffix gain over the three held-out seeds; positive favors non-suffix.

| Cache age | Same-cost comparison | Best Rank difference, 95% t interval | NDCG@100 difference, 95% t interval |
|---|---|---:|---:|
| theta-3 | middle L3-L4 vs suffix L5-L6 | -5.27 `[-10.58, 0.03]` | **-0.00077 `[-0.00139, -0.00016]`** |
| theta-5 | middle L3-L4 vs suffix L5-L6 | **-7.74 `[-13.53, -1.94]`** | +0.00078 `[-0.00320, 0.00476]` |
| theta-3 | middle L3-L5 vs suffix L4-L6 | -6.20 `[-14.93, 2.53]` | -0.00018 `[-0.00321, 0.00284]` |
| theta-5 | middle L3-L5 vs suffix L4-L6 | -7.08 `[-17.62, 3.47]` | +0.00061 `[-0.00438, 0.00559]` |

The middle intervals lose Best Rank on all three validation seeds at both cache ages. Their NDCG
advantage at theta-5 appears on only two of three seeds, reverses at theta-3, and has intervals that
include zero except for the theta-3 result favoring the suffix. Early intervals such as L1-L3 and
L1-L4 are also worse on Best Rank on every validation seed and show no stable NDCG advantage.

## Decision

The current evidence supports retaining the **deepest optimized suffix** as the main operator.
Arbitrary interval selection creates metric-specific, seed-sensitive alternatives but does not
provide a reproducible improvement that justifies a dynamic planner. This does not prove suffix
optimality for larger models or other datasets; it means arbitrary-layer selection has not passed
the current gate and should not receive more engineering effort now.

The corrected `frozen / full reuse / full compute` control and the subsequent KuaiRand scale gates
have since passed. Current priorities are maintained only in
`docs/08_core_insights_and_roadmap.md`; this historical report does not define the next phase.

## Artifacts

- `src/hstu_kvcache/migration/layerwise.py`: optimized interval and suffix operators.
- `tests/test_layerwise.py`: exact K/V equivalence and interval-semantics tests.
- `scripts/interval_oracle.py`: discovery and held-out evaluation.
- `scripts/summarize_interval_validation.py`: seed-level aggregation.
- `results/validity/interval_oracle_seed0.json`: complete 21-interval discovery.
- `results/validity/interval_validation_seed{1,2,3}.json`: held-out selected configurations.
- `results/validity/interval_validation_summary.json`: paired aggregate statistics.
