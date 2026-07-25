# KuaiRand long-context compiled migration search

## Status

This is a seed-0 design exploration on the completed KuaiRand 4+12 model chain. Motivation,
training data, checkpoints, theta11/D16 endpoint, source versions, user split, batch size, and
serving semantics remain fixed. The search changes only the shared source-version-to-theta11
cache-migration program.

The fixed split contains 40 fit users, 60 label-free probe users, and 582 test users. Theta0,
theta4, and theta10 caches represent ages 11, 7, and 1. Every candidate consumes cached old
layerwise `Norm(x)` and compiles to the same `[16, 512, 1024]` affine projection. One FP32 program
contains 8,404,992 parameters and occupies 33,619,968 bytes.

Because later rounds were chosen after inspecting earlier test results, this document is
exploratory design evidence rather than a new independent replication. A frozen candidate still
requires new training seeds or datasets for confirmatory task-quality claims.

## Design loop

### Round 1: remove rank as an artificial online constraint

The old formal design fitted a rank-32 residual and then folded it into a dense projection. Once
compiled, rank 32 and rank 512 execute the same matrix shape, move the same program bytes, and
have the same online kernel cost. Rank therefore controls only offline statistical capacity, not
online work.

`kuairand_long_context_4plus12_compiled_rank_search_v1` fits rank 512 once and evaluates nested
ranks 16, 32, 64, 128, 256, and 512. The 60-user probe selected rank 512 by label-free
fresh-score cosine. On the 582 test users, rank 512 raises cache-fidelity recovery from the
rank-32 values `0.579/0.673/0.609` to `0.886/0.891/0.936` at ages `11/7/1`. Mean online cost
remains `0.064x` exact recomputation.

This is the main design improvement. The resulting operator is a full-affine version-cohort
transport, not an online low-rank factorization.

### Round 2: direct ridge search

`kuairand_long_context_4plus12_compiled_ridge_search_v1` solves the full affine regression
directly for ridge values `1e-5` through `1e-1`. The probe selected `1e-2` from a
`1.8e-5` mean score-cosine advantage over `1e-3`, but the test users rejected that choice:
at age 11, top-100 overlap fell from `0.9688` to `0.9633` and MeanRank absolute deviation from
fresh rose from `79.75` to `91.16`. The `1e-3` solution is numerically the same as the rank-512
candidate and remains the incumbent.

This negative round prevents a tiny proxy improvement from replacing a more robust program.

### Round 3: HSTU-attention-weighted full-affine transport

Uniform K/V regression treats every cached token equally even though the latest HSTU request does
not use them equally. For fit user \(u\), layer \(l\), prefix token \(j\), and head \(h\), the
calibration computes

\[
s_{ulj} =
\sqrt{\frac{1}{H}\sum_h
\phi(q_{ulh}^{\mathsf T}k_{uljh})^2},
\]

normalizes it to unit mean within the valid prefix, and caps it at eight. A convex mix between
uniform and \(s_{ulj}\) weights the same ridge regression from cached old `Norm(x)` to
`fresh - cheap` K/V. The learned correction is folded into one affine projection, so weighting
changes no online state, kernel, or matrix shape. Recommendation labels are not used.

The probe score cosine and top-100 overlap both improve monotonically from uniform weighting to
the full attention weight, selecting mix `1.0`. Relative to uniform rank 512, the three-age mean
test absolute deviations improve by `1.03%` for MeanRank, `1.04%` for AUC, and `1.57%` for
NDCG@100. This gain is modest but consistent with the HSTU data path and free online.

## Selected exploratory result

| Cache age | Cost / exact | K/V recovery | Score cosine | Top-100 overlap | MeanRank abs. deviation | AUC abs. deviation | NDCG@100 abs. deviation |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0.0640 | 0.8859 | 0.9997149 | 0.9693 | 78.77 | 0.001576 | 0.000671 |
| 7 | 0.0642 | 0.8909 | 0.9998832 | 0.9793 | 53.03 | 0.001061 | 0.000611 |
| 1 | 0.0642 | 0.9356 | 0.9999927 | 0.9948 | 12.10 | 0.000242 | 0.000110 |

Against the old rank-32 fast path, the selected program improves mean cache-fidelity recovery
from `0.620` to `0.904` at unchanged measured cost. Averaged over the three cohorts, absolute
deviation from fresh falls by `44.5%` for MeanRank, `44.5%` for AUC, and `47.3%` for NDCG@100.

At the harmful age-11 endpoint, stale reuse has MeanRank `10720.74`, the selected migration has
`9216.83`, and exact current-model recomputation has `9204.95`. The selected program recovers
`99.2%` of the signed MeanRank and AUC gap, `94.4%` of the NDCG@100 gap, and `90.9%` of the
Hit@100 gap while taking about `2.25 ms/user` versus `35.25 ms/user` for exact recomputation.

The age-7 stale cache happens to outperform the exact current model on D16 MeanRank and AUC.
Cache migration cannot preserve that accidental task gain while also promising current-model
semantics. This is not evidence for per-version reuse admission. It is evidence that full
recomputation is a semantic endpoint rather than a ranking upper bound.

## Decision

The exploratory fast tier is now:

1. fit a version-cohort shared, attention-use-weighted full-affine residual from old
   `Norm(x)` to current K/V;
2. compile it into one packed affine projection;
3. execute it unconditionally for stale cohorts.

The old rank-32 program is retired as a large-model default. Residual p8 is also dominated at the
three measured endpoints: it costs about `0.549x` exact while the selected compiled program costs
`0.064x` and has higher K/V recovery. Exact recomputation remains the endpoint. Structural replay
is retained only as a candidate for a stricter fidelity target not met by the compiled program;
it is not automatically a middle tier.

The next experiment is not another search on these 582 users. It is frozen replication of the
selected operator, followed by organically mixed version cohorts and end-to-end state movement.

The immediate follow-up implements the planned label-free verification stage with a new 60-user
certificate split and a 522-user final test. It publishes per-cohort action certificates and
ordered fallbacks without recommendation labels; see
`experiments/migration/VERIFIED_COHORT_COMPILER_V1.md`.

## Commands

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/search_kuairand_long_context_compiled_rank.py

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/search_kuairand_long_context_compiled_ridge.py

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/search_kuairand_long_context_attention_weighted.py
```

The corresponding local result files are:

- `results/motivation_scale/long_context_4plus12_compiled_rank_search_seed0.json`;
- `results/motivation_scale/long_context_4plus12_compiled_ridge_search_seed0.json`;
- `results/motivation_scale/long_context_4plus12_attention_weighted_search_seed0.json`.
