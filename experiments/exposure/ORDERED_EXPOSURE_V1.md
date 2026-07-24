# Ordered-exposure cross-dataset reproduction

> Status: motivation-alignment follow-up and aligned method gate completed on 2026-07-23.
> KuaiRand, Tenrec QB, and Tenrec QK reproduce the same pre-design logic over four training seeds.
> On the aligned method gate, QK cheap refresh transfers over four seeds, QB fails at seed 0, and
> the deepest-suffix curve remains KuaiRand-specific. ZhihuRec remains a negative boundary. Raw
> per-user outputs and checkpoints are local, ignored artifacts.
> A later protocol learns and compiles a shared low-rank residual adapter; it must not be pooled
> with this gate. See `experiments/migration/COMPILED_LOW_RANK_V1.md`.

## 1. Question and protocols

The pre-design question has three parts:

1. streaming updates improve a current model over a frozen model;
2. reusing a theta-0 prefix cache preserves a substantial part of that value rather than failing
   immediately;
3. the recoverable `full compute - full reuse` gap becomes larger for older caches.

Method transfer is evaluated separately. Sections 4.1-4.2 retain the original frozen top-50k
protocol and must not be used as if they were measured on the motivation-aligned cohorts.
Section 4.3 is the explicitly versioned aligned follow-up: all frozen candidates run on seed 0,
then only a material new quality-cost point expands to held-out seeds.

All datasets retain real negative exposures. A zero-feedback row enters context but is not a
next-item target. Tenrec positives are rows with click, like, follow, or share. No sampled or
fabricated unexposed item is inserted. Tenrec uses stable official within-user order and is an
ordinal replay, not calendar-time drift.

The model and optimization are shared: 6 layers, hidden size 96, 4 heads of width 24, sequence
length 128, 6 base epochs at `3e-4`, and 2 epochs per stream update at `1e-4`. Data settings are
calibrated before seed expansion to create a dense, identifiable ordered-exposure regime:

- KuaiRand uses its existing top-5k calendar-time validity protocol.
- QK uses top-5k, 5,000 users selected by retained base activity, 64 base exposures, and six
  8-exposure windows.
- QB uses top-50k and a fixed-horizon 5,000-user cohort. Eligibility requires at least 112 raw
  exposures, but does not inspect future labels or experiment outcomes. This removes changing-user
  composition from the cache-age comparison.

These are comparable mechanism tests, not one pooled dataset or an estimate of the population-wide
deployment effect. QB and QK are also two related Tenrec tables, so they are not claimed as two
independent data sources.

## 2. Materialized motivation cohorts

| Dataset | Catalog | Cohort rule | Base rows | Six stream windows | Theta-5 eval rows | Theta-5 positive-active users |
|---|---:|---|---:|---:|---:|---:|
| Tenrec QB | 50k | complete raw 64+6x8 horizon, no label filter | 311,572 | 223,399 | 37,079 | 4,730 |
| Tenrec QK | 5k | retained base activity only | 277,409 | 100,221 | 10,131 | 1,588 |

The QB fixed-horizon windows contain 37.1k–37.4k retained rows each; the old base-only cohort
shrunk from 35.8k to 19.4k rows and from 4,049 to 2,552 positive-active users. This composition
change was the main QB diagnostic. QK top-5k reduces the model from 5.09M to 0.77M parameters and
concentrates repeated item support. It produces a more stable maintenance denominator, but the
experiment does not identify catalog size alone as the causal factor.

Exact counts and provenance are tracked in
`results/dataset_audit/tenrec_qb_top50000_users5000_fixed_horizon_prepared.json` and
`results/dataset_audit/tenrec_qk_top5000_users5000_prepared.json`; compact NPZ files remain ignored.

The diagnostic path was deliberately narrow:

| Diagnostic | Result | Decision |
|---|---|---|
| QB top-5k, seed-0 gate | theta-5 streaming `+15.38`, maintenance `+1.88` | reject catalog reduction for QB |
| QK top-50k, four seeds | theta-5 maintenance `+20.16 [-8.90, 49.22]` | denominator unresolved |
| QK top-50k, 12L/H192 seed-0 | theta-5 maintenance `-7.00` | larger model alone does not fix it |
| Tenrec lifelong `task_0..3` files | inspected samples are positive-action-only | reject because true negative exposures are lost |

The accepted QK axis was fixed before its seed-0 run and then validated on seeds 1-3. The QB
fixed-horizon rule was motivated by row/user composition, not by outcome labels; theta-5
maintenance is positive on each of the three held-out seeds as well as the development seed.

## 3. Streaming-training value and accumulated cache loss

The three conditions are:

- `frozen`: theta-0 model, theta-0 cache, and theta-0 scoring head;
- `full reuse`: current model and scoring head consuming the theta-0 prefix cache;
- `full compute`: current model with a version-consistent current prefix cache.

The table reports BestRank improvement over `frozen`; maintenance is `full compute - full reuse`.
Positive values are improvements. Brackets are 95% t intervals over training seeds.

### 3.1 Three-dataset result, four seeds each

The primary compact comparison is:

| Dataset | Cache-age/maintenance Spearman | theta-5 full compute over frozen | theta-5 full reuse over frozen | theta-5 maintenance |
|---|---:|---:|---:|---:|
| KuaiRand | **0.925 `[0.845, 1.000]`** | **+484.34 `[462.15, 506.54]`** | **+399.02 `[370.79, 427.26]`** | **+85.32 `[53.74, 116.91]`** |
| Tenrec QB fixed horizon | **0.600 `[0.005, 1.000]`** | **+94.70 `[70.49, 118.90]`** | **+64.38 `[54.41, 74.35]`** | **+30.31 `[14.08, 46.55]`** |
| Tenrec QK top-5k | **0.700 `[0.475, 0.925]`** | **+47.34 `[29.20, 65.47]`** | **+34.34 `[20.55, 48.13]`** | **+13.00 `[5.70, 20.30]`** |

The Spearman statistic is computed within each seed across five cumulative cache ages and then
aggregated over seeds. It is positive with a seed-level 95% t interval above zero on all three
datasets. Absolute BestRank changes are not compared across catalogs; only the sign, partition,
and age structure are compared.

### 3.2 Tenrec QB fixed horizon

| Cache age | Full compute over frozen | Full reuse over frozen | Cache maintenance |
|---|---:|---:|---:|
| theta-1 | +0.48 `[-22.60, 23.56]` | +8.39 `[-1.86, 18.64]` | -7.91 `[-28.95, 13.14]` |
| theta-3 | +43.14 `[-7.98, 94.26]` | **+37.45 `[21.49, 53.41]`** | +5.69 `[-30.58, 41.97]` |
| theta-5 | **+94.70 `[70.49, 118.90]`** | **+64.38 `[54.41, 74.35]`** | **+30.31 `[14.08, 46.55]`** |

QB gives the intended delayed-onset pattern. A one-update cache does not need maintenance, while
streaming value and a recoverable cache gap are both clear by theta-5. Full reuse retains 68.0% of
the mean theta-5 BestRank streaming value. Theta-5 NDCG@100 maintenance is `+0.00102` with interval
`[-0.00008, 0.00212]`, so the strongest maintenance claim remains on BestRank.

### 3.3 Tenrec QK top-5k

| Cache age | Full compute over frozen | Full reuse over frozen | Cache maintenance |
|---|---:|---:|---:|
| theta-1 | **+11.52 `[6.20, 16.84]`** | **+7.24 `[4.64, 9.84]`** | **+4.28 `[1.18, 7.39]`** |
| theta-3 | **+21.45 `[12.98, 29.92]`** | **+16.52 `[12.58, 20.46]`** | +4.93 `[-2.11, 11.97]` |
| theta-5 | **+47.34 `[29.20, 65.47]`** | **+34.34 `[20.55, 48.13]`** | **+13.00 `[5.70, 20.30]`** |

QK now reproduces all three pre-design statements. The cumulative maintenance means over the five
ages are `4.29, 5.95, 4.95, 12.92, 12.99`: not strictly monotone at every adjacent point, but
consistently positive and larger for old caches in the seed-level age test.

### 3.4 Fixed-endpoint normalization and local jumps

Raw BestRank gains are retained only for within-dataset interpretation. Holding the final model,
evaluation users, histories, and targets fixed, the within-seed BestRank staleness tax
`(full compute - reuse) / (full compute - frozen)` is:

| Dataset | Coarse endpoint tax | Fine endpoint tax |
|---|---:|---:|
| KuaiRand | **0.176 `[0.115, 0.237]`** | **0.177 `[0.118, 0.236]`** |
| QB fixed horizon | **0.315 `[0.225, 0.404]`** | **0.350 `[0.263, 0.437]`** |
| QK top-5k | **0.276 `[0.168, 0.385]`** | **0.141 `[0.056, 0.227]`** |

The coarse values span 1.79x and fine values 2.48x, so the former
`484/95/47` raw full-compute scale difference is not a difference in normalized cache-loss
strength. Fine KuaiRand and QB have common local BestRank-tax jumps positive in 4/4 seeds; QK has
a large local jump in every seed but at different ages. The result motivates an update-aware
version-cohort trigger, not the stronger claim that every tuned periodic window fails. Exact
per-seed transitions, metric sensitivity, and protocol boundaries are in
`CACHE_VERSION_MATRIX_V1.md`.

### 3.5 ZhihuRec boundary

At theta-5, full compute improves BestRank over frozen by `+180.95` and full reuse by `+188.46`,
leaving maintenance at `-7.51`. Theta-3 maintenance is only `+4.19`. The same fixed update-strength
grid at `1e-4`, `2e-4`, and `4e-4` did not produce a stable accumulated gap. ZhihuRec therefore
reproduces streaming-training value but not stale-cache maintenance under the original protocol.
It is retained as a negative boundary rather than tuned into the three-dataset main result.

## 4. Frozen migration methods at theta-5

The evaluated configurations are reuse, cheap-all, deepest suffix-2/4/5, and full recompute. The
suffix operator uses terminal projection optimization and was not re-searched on either dataset.
Cost is measured GPU migration time normalized by optimized full recompute.

### 4.1 Tenrec QB, eight seeds

| Configuration | Cost / full | BestRank gain over reuse | 95% seed interval | Ratio of mean Rank gains |
|---|---:|---:|---:|---:|
| cheap all | 0.196 | +3.70 | `[0.85, 6.54]` | 23.0% |
| suffix-2 | 0.382 | +4.21 | `[1.11, 7.31]` | 26.2% |
| suffix-4 | 0.634 | +4.33 | `[0.48, 8.17]` | 26.9% |
| suffix-5 | 0.764 | +4.06 | `[-0.37, 8.50]` | 25.3% |
| full recompute | 1.000 | +16.08 | `[1.61, 30.55]` | 100% |

The cheap endpoint transfers: it gives a positive BestRank gain at about one fifth of full cost.
Suffix-2 and suffix-4 add a small mean gain, but deeper execution does not yield the strong,
monotone recovery curve seen on KuaiRand; suffix-5 reverses slightly. NDCG denominators are not
identifiable. This is a partial method transfer, not evidence that the fixed suffix is generally
near-full quality.

### 4.2 Tenrec QK, four seeds

| Configuration | Cost / full | BestRank gain over reuse | 95% seed interval | Ratio of mean Rank gains |
|---|---:|---:|---:|---:|
| cheap all | 0.196 | +11.19 | `[-8.83, 31.21]` | 55.5% |
| suffix-2 | 0.383 | +10.52 | `[-10.70, 31.74]` | 52.2% |
| suffix-4 | 0.636 | +8.99 | `[-16.87, 34.85]` | 44.6% |
| suffix-5 | 0.764 | +9.96 | `[-14.50, 34.43]` | 49.4% |
| full recompute | 1.000 | +20.16 | `[-8.90, 49.22]` | 100% |

The denominator is not identifiable and the suffix curve is absent. These ratios are descriptive
only. QK cannot currently support a migration-quality generalization claim.

### 4.3 Migration on the aligned motivation settings

The aligned follow-up does not re-select a layer interval. Seed 0 evaluates cheap-all, the frozen
suffix-2/4/5 configurations, and full recompute on exactly the fixed-horizon QB and top-5k QK
cohorts used by Section 3. Only a material seed-0 quality-cost point is expanded.

QB fails this screen. Full recomputation gains 21.95 BestRank on 4,730 users, while cheap gains
only 1.09 at `0.196x` cost and suffix-2/4/5 gain `-0.33/-2.07/+1.16`; every partial method has
negative NDCG@100 gain. This branch stops at seed 0 rather than spending three more seeds to
confirm a non-Pareto result.

QK gives the opposite seed-0 result:

| Configuration | Cost / full | BestRank gain | Rank recovery | NDCG@100 gain |
|---|---:|---:|---:|---:|
| cheap all | **0.197** | **+8.17** | **89.3%** | +0.00134 |
| suffix-2 | 0.384 | +7.75 | 84.7% | +0.00077 |
| suffix-4 | 0.636 | +6.29 | 68.8% | +0.00090 |
| suffix-5 | 0.764 | +7.40 | 80.9% | +0.00143 |
| full recompute | 1.000 | +9.15 | 100% | +0.00057 |

Every suffix loses BestRank relative to cheap while using more compute. Suffix-5's additional
NDCG over cheap is only 0.00010 at 3.9x its cost, so the frozen expansion rule retains cheap only.
Across four training seeds:

| Configuration | Cost / full | BestRank gain, 95% CI | Rank recovery | NDCG@100 gain, 95% CI |
|---|---:|---:|---:|---:|
| cheap all | **0.194 `[0.187, 0.200]`** | **+9.17 `[6.71, 11.63]`** | **70.6%** | +0.00125 `[-0.00091, 0.00341]` |
| full recompute | 1.000 | **+13.00 `[5.70, 20.30]`** | 100% | +0.00227 `[-0.00086, 0.00541]` |

Cheap improves BestRank in every seed, and its seed interval excludes zero. Its paired BestRank
difference from full is `-3.83 [-8.85, 1.19]`; failure to distinguish the two is not equivalence.
The NDCG denominator remains unresolved. The supported cross-dataset method claim is therefore
narrow but useful: current `Wk/Wv` projection refresh can recover a positive low-cost fraction of
the stale-cache gap on aligned QK. The stronger claim that deeper suffix propagation gives a
monotone Pareto curve remains unsupported outside KuaiRand.

## 5. Structural observation and failed scale gate

Mean relative stale K/V error by layer is:

| Dataset | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---:|---:|---:|---:|---:|---:|
| QB, 8 seeds | 0.393 | 0.478 | 0.541 | 0.574 | 0.598 | 0.642 |
| QK, 4 seeds | 0.277 | 0.333 | 0.362 | 0.449 | 0.511 | 0.631 |

The monotone depth pattern transfers cleanly and supports deepest suffix as a structural heuristic.
It does not guarantee that extra suffix computation improves a noisy ranking metric.

A predeclared 12-layer, hidden-192 single-seed gate kept the same users, windows, and update
settings. Streaming training remained useful at theta-5 (`+164.35` QK, `+234.42` ZhihuRec), but
maintenance was negative (`-7.00`, `-13.67`). Larger K/V drift therefore did not repair the task
gap, and this scale branch was stopped without method evaluation or seed expansion.

## 6. Decision

The motivation gate now passes on KuaiRand, QB, and QK:

- streaming training has positive long-horizon value;
- stale reuse preserves a substantial positive gain over the frozen model;
- version-consistent cache recomputation restores an additional old-cache gap;
- the cumulative BestRank maintenance gap has a positive age relationship over training seeds;
- fixed-endpoint BestRank staleness tax is on one scale across datasets, while local jumps and
  threshold-crossing ages show that age alone is not a calibrated maintenance trigger.

The result is not produced by selecting seeds or labels. QK used one predeclared catalog-size axis;
QB used a complete-activity-horizon restriction after the changing-cohort confound was identified.
Both decisions and the failed QB top-5k gate are retained in the record. The limitations are that
Tenrec has ordinal rather than global calendar time, QB conditions on future activity availability,
and QB/QK are related tables from one collection.

Within this fixed-suffix protocol, the method conclusion is mixed and must remain separate. Re-evaluation on the aligned
settings now gives four-seed support for cheap projection refresh on QK at 0.194x full cost and
70.6% mean BestRank recovery. Aligned QB fails its seed-0 partial-method gate, and no Tenrec result
supports the strong KuaiRand suffix curve. The subsequent compiled-adapter result supersedes cheap
as the current cross-dataset method anchor without relabeling these files. The next paper step is
therefore not another arbitrary layer search; it is mixed cache ages, state movement, throughput,
tail latency, periodic full recomputation, and update-aware version-cohort triggering.

## 7. Reproduction entry points

Prepare a compact stream:

```bash
python scripts/prepare_exposure_stream.py --dataset tenrec-qb \
  --cohort-selection complete_horizon \
  --output data/processed/tenrec_qb_top50000_users5000_fixed_horizon.npz \
  --metadata-output results/dataset_audit/tenrec_qb_top50000_users5000_fixed_horizon_prepared.json
python scripts/prepare_exposure_stream.py --dataset tenrec-qk --catalog-size 5000 \
  --output data/processed/tenrec_qk_top5000_users5000.npz \
  --metadata-output results/dataset_audit/tenrec_qk_top5000_users5000_prepared.json
python scripts/prepare_exposure_stream.py --dataset zhihurec
```

The training entry point is `scripts/motivation_validity.py --prepared-data <npz>`. Consistent
full-reuse/full-compute controls use `scripts/streaming_value_control.py`; frozen methods use
`scripts/interval_oracle.py` with `cheap_all interval_l5_l6 interval_l3_l6 interval_l2_l6
recompute`; method aggregation uses `scripts/summarize_exposure_methods.py`.

Tracked compact evidence:

- `results/validity/core6l_summary.json`
- `results/validity/streaming_control6l_summary.json`
- `results/exposure/qb_fixed_horizon_core_summary.json`
- `results/exposure/qb_fixed_horizon_streaming_control_summary.json`
- `results/exposure/qk_top5k_core_summary.json`
- `results/exposure/qk_top5k_streaming_control_summary.json`
- `results/exposure/qb_streaming_control_summary.json`
- `results/exposure/qb_method_summary.json`
- `results/exposure/qk_streaming_control_summary.json`
- `results/exposure/qk_method_summary.json`
- `results/exposure/aligned_method_gate_summary.json`
- `results/exposure/qk_top5k_aligned_method_summary.json`
- `results/exposure/zhihu_streaming_control_summary.json`
- `results/exposure/kuai_allages_streaming_control_summary.json`
- `results/exposure/qb_fixed_horizon_allages_streaming_control_summary.json`
- `results/exposure/qk_top5k_allages_streaming_control_summary.json`
- `results/exposure/cache_age_cross_dataset_summary.json`
- `results/exposure/cache_version_matrix_cross_dataset_summary.json`
- `results/exposure/cache_version_matrix_fine_cross_dataset_summary.json`
- `results/dataset_audit/tenrec_qb_top50000_users5000_fixed_horizon_prepared.json`
- `results/dataset_audit/tenrec_qk_top5000_users5000_prepared.json`
