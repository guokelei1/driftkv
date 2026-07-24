# Cohort-tiered cache migration v1

> Status: frozen held-out-seed replication completed on 2026-07-24.
> Primary protocol: `cohort_tiered_migration_v1`.

## 1. System abstraction

Caches are grouped by their `(old model version, current model version)` pair. The controller
calibrates once for that version cohort, chooses one homogeneous migration action, and batches all
caches in the cohort through the selected kernel. It does not predict a different action for each
user.

The production action library has three tiers:

1. a compiled affine K/V projection;
2. current-model prefix replay with residual-delta transport;
3. exact current-model recomputation.

Plain stale reuse remains the zero-cost baseline, not a migration tier. The old fixed suffix,
arbitrary contiguous intervals, and recent-token rectangles are discovery baselines only.

## 2. Compiled projection fast tier

For layer \(l\), let \(z_l\) be cached old-version
\(\operatorname{Norm}(x_l)\). Projecting it with the current K/V weights gives

\[
\widehat C_l^{\mathrm{cheap}}
=
z_l[W_l^K,W_l^V]_{\theta_t}.
\]

On a small cohort-level fit set, exact current caches expose the residual

\[
R_l=C_l(\theta_t)-\widehat C_l^{\mathrm{cheap}}.
\]

The controller fits a ridge-regularized rank-\(r\) affine map

\[
R_l\approx(z_l-\mu_l)A_{l,r}B_{l,r}+b_l.
\]

This factorization is never executed as two extra online multiplications. Before migration it is
folded into

\[
\widetilde W_l=[W_l^K,W_l^V]_{\theta_t}+A_{l,r}B_{l,r},
\qquad
\widetilde b_l=b_l-\mu_lA_{l,r}B_{l,r}.
\]

Every cache then uses one layer-batched affine projection
\(z_l\widetilde W_l+\widetilde b_l\), with the same online tensor shape for every selected rank.
Rank controls calibration capacity and generalization, not the online kernel shape.

## 3. Structural quality tier

When a compiled projection cannot meet a stricter fidelity target, action `residual-p` executes
the first \(p\) current-model blocks exactly. At its boundary it computes

\[
\Delta_p=x_p^{\theta_t}-x_p^{\theta_s}.
\]

For each deeper layer \(l>p\), it transports the residual-network displacement as

\[
\widehat x_l=x_l^{\theta_s}+\Delta_p
\]

and applies the current layer's `Norm + Wk/Wv`. This preserves the native model computation in
the replayed prefix and uses an additive residual-state approximation deeper in the network. It
adds no learned online operator. Setting the final action to exact recomputation supplies the
fidelity endpoint.

## 4. Frozen cohort controller

For every old/current version cohort:

1. assign 40 users to adapter fitting, 60 disjoint users to planning, and all remaining users to
   evaluation using a seeded partition independent of labels and outcomes;
2. fit ranks `2, 4, 8, 16, 32, 64, 96`, truncated to the model's supported width;
3. measure K/V fidelity and resident-GPU migration cost on the planning users;
4. for compiled ranks, use the median cost of the common projection kernel and choose the smallest
   rank satisfying the target;
5. compare that fast-tier candidate with the residual-depth ladder and exact recomputation;
6. choose the lowest-cost action meeting the target, while requiring a structural partial action
   to save at least 1% over full recomputation.

The primary paper operating point closes at least 50% of the stale-to-fresh K/V error gap. This
target is inherited from the earlier four-seed compiled-adapter protocol rather than tuned on the
nine capacity checkpoints. Targets 75% and 90% expose the quality-cost curve. In a deployment the
target is an SLA input; it is not a fixed layer count. No recommendation labels or ranking metrics
enter fitting or action selection.

## 5. Nine-checkpoint discovery

One checkpoint per dataset-capacity cell was selected by the frozen motivation-only rule in
`results/motivation_scale/design_discovery_seeds.json`. Method outcomes did not enter that
selection. The primary 50% controller selected only compiled projections:

| Cell | Action | GPU time / full | K/V recovery | BestRank gain | Rank-utility gain | NDCG@100 gain |
|---|---:|---:|---:|---:|---:|---:|
| KuaiRand small | rank 8 | 0.099 | 0.519 | +325.80 | +0.15422 | +0.00066 |
| KuaiRand medium | rank 16 | 0.135 | 0.564 | +780.68 | +0.32641 | +0.00203 |
| KuaiRand large | rank 32 | 0.150 | 0.550 | +1658.65 | +0.56469 | +0.00495 |
| QB small | rank 2 | 0.091 | 0.685 | +2.14 | +0.00026 | +0.00022 |
| QB medium | rank 2 | 0.121 | 0.536 | +6.83 | +0.01807 | -0.00032 |
| QB large | rank 32 | 0.150 | 0.540 | +19.62 | +0.01748 | +0.00114 |
| QK small | rank 2 | 0.088 | 0.525 | +8.01 | +0.01267 | +0.00126 |
| QK medium | rank 2 | 0.115 | 0.571 | +33.93 | +0.14983 | +0.00771 |
| QK large | rank 32 | 0.145 | 0.590 | +13.39 | +0.03678 | +0.00223 |

Across the nine cells, mean resident-GPU cost is `0.122x` full and mean held-out K/V recovery is
`0.564`. BestRank and rank utility improve over reuse in 9/9 cells; NDCG@100 improves in 8/9.
QB-medium is the sole NDCG conflict, and exact full recomputation is also negative on that metric
under the same split.

At the 75% target, six cells remain on a compiled projection while KuaiRand-large, QB-large, and
QK-large select residual depths 5, 6, and 7. The mean cost becomes `0.357x` full and mean K/V
recovery becomes `0.784`. This is secondary discovery evidence for a real tiered controller, not
a claim that 75% is the universally preferred deployment point.

## 6. Bounded architecture decisions

- Partial recent-token replay is rejected at the current sequence length because split, concat,
  and small-kernel overhead makes it slower than the matched complete-span action in all 54
  discovery comparisons.
- Arbitrary contiguous intervals are rejected because their \(O(L^2)\) planner improves only one
  discovery cell materially over boundary families.
- Fixed deep suffix replay is rejected as the transferable design because it fails in scaled QB
  and QK cells.
- Progressive exact-prefix replay remains a useful no-fit baseline, but it is not selected once
  the compiled fast tier and residual quality tier share one action library.
- Residual transport alone is a high-quality but high-cost ablation: its 50% operating point costs
  about `0.832x` full on the nine discovery checkpoints.

## 7. Frozen validation gate

The three non-discovery training seeds in every dataset-capacity cell are independent replication
units, for 27 held-out model-version chains. The architecture, split sizes, ranks, fidelity
targets, action library, and selector above must remain unchanged.

The primary gate is:

- mean selected resident-GPU cost below full in every cell;
- positive mean BestRank and rank-utility gain over reuse in every cell;
- seed signs and 95% seed-level intervals reported without treating users as independent
  replications.

NDCG@100 is secondary because the motivation control and exact full endpoint already have a
direction conflict in QB-medium.

## 8. Held-out-seed result

At the primary 50% target, all 27 held-out training seeds select the compiled projection tier.
Across these independent model-version chains:

- mean GPU time is `0.121 [0.112, 0.130]x` full;
- mean K/V recovery is `0.587 [0.547, 0.627]`;
- 25/27 test splits remain at or above the 50% probe-selected fidelity target;
- BestRank, rank utility, and NDCG@100 have positive signs in 20/27, 24/27, and 20/27 seeds.

The strict cell-level gate passes 6/9 cells. KuaiRand small/medium/large, QB-large, QK-small, and
QK-medium have positive mean BestRank and rank-utility gains below full cost. The failures are:

- QB-small rank utility is `-0.00003` on average, effectively a near-zero endpoint;
- QB-medium BestRank is `-9.48`, while exact full is also negative at `-9.71` in all three seeds;
- QK-large BestRank is `-0.002`, while exact full is only `+0.59` and has two positive signs.

This does not support a universal “migration always beats reuse” claim. It does show that the
operator closely tracks the attainable full endpoint. Selected and full have the same sign in
23/27 BestRank, 27/27 rank-utility, and 25/27 NDCG cases. Restricting the descriptive view to
positive full endpoints, the selected action is also positive in 18/20, 24/24, and 19/20 seeds,
with median gain recoveries of 0.907, 0.972, and 0.966. These pooled cross-cell diagnostics are not
inferential intervals.

The correct next abstraction is therefore two decisions:

1. an admission controller decides whether a version cohort has a useful maintenance endpoint;
2. conditional on admission, the tiered migration controller chooses a cache-fidelity/cost point.

The frozen v1 results remain unchanged. A new admission rule or a change from minimum eligible rank
to another compiled-rank rule requires a v2 protocol and fresh training seeds.

## 9. Current boundary

These measurements cover GPU-resident migration kernels plus the recorded one-time fit cost.
They do not yet include normalized-state reads, cache writes, adapter distribution, scheduler
queues, mixed cache ages, or end-to-end tail latency. The system claim remains incomplete until
the cohort planner is evaluated under organic version mixtures and its calibration cost is
amortized over real cohort sizes.

## 10. Reproduction

- Compiled operator: `src/hstu_kvcache/migration/low_rank.py`
- Residual structural operator: `src/hstu_kvcache/migration/layerwise.py`
- Unified discovery: `scripts/cohort_tiered_migration_search.py`
- Nine-cell runner: `scripts/run_cohort_tiered_migration_discovery.py`
- Held-out-seed runner: `scripts/run_cohort_tiered_migration_validation.py`
- Seed-level summary: `scripts/summarize_cohort_tiered_migration.py`
- Architecture-screen summary: `scripts/summarize_structural_design_discovery.py`
- Motivation checkpoint selector: `scripts/select_motivation_discovery_seeds.py`
