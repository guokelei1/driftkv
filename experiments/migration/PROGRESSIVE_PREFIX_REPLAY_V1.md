# Progressive prefix replay v1

> Status: held-out-seed replication completed; retained as a structural baseline.

## Design objective

The online cache-migration operator must use only current-model HSTU primitives, expose a monotone
and measurable compute knob, and choose one batched action for an entire old/current model-version
cohort. It must not fit a per-user predictor or add learned migration weights.

For a model with \(L\) layers, the action ladder is:

1. `cheap`: project cached old `Norm(x)` with current `Wk/Wv`;
2. `prefix-p`: execute current blocks from layer 1 through layer \(p-1\), execute only current
   `Norm + Wk/Wv` at terminal layer \(p\), and use cheap projection for layers \(p+1\) through
   \(L\);
3. `full`: set \(p=L\), which is exactly current-model K/V recomputation.

There are \(L+1\) actions, so the version-level planner is linear rather than quadratic in model
depth. Every deployed action is composed only from original model blocks and projections.

## Discovery boundary

One motivation-aligned seed per dataset-tier cell was selected without using migration outcomes.
The frozen lexicographic rule first requires positive full-compute and stale-reuse streaming value,
then maximizes positive maintenance signs over BestRank/rank utility/NDCG@100, clipped BestRank
staleness tax, and cache-age correlation. These nine seeds are method discovery only:

| Cell | Discovery seed |
|---|---:|
| KuaiRand small / medium / large | 0 / 0 / 0 |
| QB small / medium / large | 0 / 1 / 0 |
| QK small / medium / large | 2 / 2 / 0 |

The exact candidates and scores are in
`results/motivation_scale/design_discovery_seeds.json`.

Discovery compared a 3x3 recent-token/deep-layer grid and every contiguous layer interval. Recent
partial replay was usually slower than replaying the full retained token span at length 128,
because splitting, concatenation, and small matrix shapes outweighed the saved arithmetic.
Arbitrary intervals improved only one KuaiRand cell materially over the prefix/suffix boundary
families. They did not justify an \(O(L^2)\) production planner.

## Frozen selector

For every theta-0/theta-11 version cohort:

1. choose 60 probe caches by a seeded permutation independent of labels and model outcomes;
2. fully recompute those probe caches only to measure exact K/V fidelity;
3. evaluate the \(L+1\) structural actions on the same probes;
4. choose the minimum measured GPU-cost action that closes at least 20% of the
   reuse-to-full relative K/V error gap;
5. fall back to full recomputation if no partial action reaches the target.

Task labels are not used by the selector. The selected prefix depth applies to the complete
version cohort, enabling length bucketing and homogeneous GPU batches.

The 20% target is the smallest global discovery target for which the prefix-only ladder improved
both BestRank and rank utility over reuse in all nine discovery cells. It is now frozen.

## Discovery result

On disjoint held-out users from the nine selected checkpoints, the frozen-rule simulation has:

- positive BestRank and rank-utility gains in 9/9 cells;
- positive NDCG@100 in 7/9 cells;
- mean selected kernel cost 0.482x full and maximum cost 0.679x;
- no learned migration parameters.

The selected actions range from cheap-only to prefix depths 1-5. A fixed deepest suffix reaches
positive BestRank in only 7/9 cells under the same fidelity target. This is evidence for
version-level structure selection, not evidence that shallow prefixes are universally optimal.

## Held-out-seed validation

The three non-discovery training seeds in every dataset-tier cell are confirmatory replication
units: 27 independently trained model-version chains in total. Every run keeps:

- theta-0 cache and theta-11 current model;
- at most 1,000 final-window users;
- 60 label-independent probe users and all remaining users for held-out evaluation;
- the complete prefix ladder and 20% fidelity target;
- three resident-GPU timing repetitions;
- full-catalog ranking.

The primary validation views are BestRank and rank utility. NDCG@100 remains secondary because the
motivation control itself has direction conflicts in QB medium. A strong pass requires every
dataset-tier cell to have mean selected cost below full and positive mean held-out BestRank and
rank-utility gain over reuse. Seed signs and intervals must still be reported; a positive mean
does not erase an unstable cell.

## Current system boundary

The measured cost is the resident-GPU migration kernel. The protocol records extra normalized
state, but it does not yet include cache reads/writes, probe admission, full-probe recomputation,
length-bucket scheduling, or end-to-end tail latency. Those costs are required before presenting
the method as a complete serving system.

## Held-out-seed outcome

Across the 27 non-discovery training seeds, the frozen 20% selector uses `0.456
[0.404, 0.507]x` full on average and achieves `0.237 [0.226, 0.247]` K/V recovery. Its strict
positive-mean BestRank/rank-utility gate passes 7/9 cells. QB-medium fails BestRank and QK-small
fails rank utility; both are motivation-weak cells in which exact full maintenance is itself
unstable.

The no-fit prefix ladder remains an interpretable structural control, but it is not the primary
operator. Under a common fit/probe/test split, the cohort-compiled projection reaches materially
higher K/V fidelity at about one quarter of this kernel cost, and exact-prefix replay is never
selected once compiled projection and residual transport share one action library. The current
primary design is documented in `experiments/migration/COHORT_TIERED_MIGRATION_V1.md`.
