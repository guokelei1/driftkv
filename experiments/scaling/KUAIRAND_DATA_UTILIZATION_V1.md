# KuaiRand data-utilization-v1: top-50k with complete base chunks

## Why this follow-up was necessary

Increasing `max_items` did not by itself mean the training loop consumed all retained
interactions. The previous base iterator produced one latest truncated sequence per user per
epoch. At top-50k and sequence length 512, 1,276,718 base rows survive filtering, but the latest
iterator exposes only 432,681 context tokens per epoch.

The new optional `all_chunks` mode groups each user's complete chronological base history into
length-512 chunks with stride 511. The one-token overlap preserves the next-item pair across each
boundary without duplicating a target. During streaming updates, only chunks containing an engaged
target from the current date are executed. The old `latest` mode remains the default, so all prior
protocols are unchanged.

## Controlled protocol

Both modes use top-50k, sequence length 512, six layers, hidden size 96, four heads, six base
epochs, five three-day updates, two update epochs, 300 full-catalog evaluation users, and seeds
0-3. They differ only in training sequence construction.

| Training mode | Base sequences/epoch | Context tokens/epoch | Eligible base targets/epoch | Eligible stream targets |
|---|---:|---:|---:|---:|
| latest | 980 | 432,681 | 230,945 | 130,239 |
| all chunks | 3,007 | 1,278,745 | 620,958 | 130,239 |

The stream target count is exactly equal. The intervention raises base eligible targets by 2.69x
and context tokens by 2.96x; it does not manufacture extra stream labels.

This is still not full KuaiRand. Top-50k retains 1,563,816 of 11,713,045 standard-log rows
(13.35%) and only 50,000 of 2,119,510 base-period items. A leak-free all-base-item vocabulary would
retain 53.93% of the standard rows, but would require a 2.12-million-item embedding and ranking
head. KuaiRand-27K and the random-exposure log remain outside the experiment.

## Streaming value and maintenance gap

The theta-5 control uses the same users to compare the frozen theta-0 model, current-model full
reuse of a theta-0 prefix, and current-model full compute.

| Training mode | Full compute over frozen Best Rank | Full reuse over frozen | Maintenance Best Rank | Maintenance NDCG@100 |
|---|---:|---:|---:|---:|
| latest | 2,064.19 [1,966.65, 2,161.72] | 1,981.51 [1,897.99, 2,065.02] | 82.68 [40.23, 125.13] | 0.00109 [-0.00040, 0.00258] |
| all chunks | 3,837.67 [3,389.91, 4,285.44] | 2,952.11 [2,700.21, 3,204.02] | **885.56 [460.24, 1,310.88]** | **0.00250 [0.00169, 0.00330]** |

Full compute remains strongly better than frozen in every seed, so the larger maintenance gap is
not caused by a useless or degraded streaming model. Under all-chunks training, stale reuse retains
76.9% of the Best Rank value of streaming training and cache maintenance accounts for the remaining
23.1%; the corresponding NDCG share is 33.0%.

Across all five cache ages, the all-chunks pooled Best Rank maintenance gap is 455.35
[230.08, 680.61], versus 53.21 [13.49, 92.93] for latest-only training. Meanwhile cumulative
parameter distance is slightly smaller, 0.308 versus 0.326. The effect therefore cannot be
explained by parameter-norm distance alone.

## Fixed suffix at the stronger operating point

The table uses theta-0-to-theta-5 all-chunks checkpoints. Recovery is the ratio of cross-seed mean
Best Rank gain to the full gain. Full resident-GPU latency is 0.456 ms/user on one A40.

| Configuration | Cost / full | Best Rank gain, 95% CI | Rank recovery | NDCG@100 gain, 95% CI |
|---|---:|---:|---:|---:|
| cheap | **0.058** | 483.78 [218.87, 748.70] | 54.6% | 0.00187 [0.00060, 0.00314] |
| suffix-2 | 0.248 | 521.88 [281.07, 762.69] | 58.9% | 0.00215 [0.00082, 0.00349] |
| suffix-4 | 0.613 | 674.97 [442.91, 907.03] | 76.2% | 0.00263 [0.00179, 0.00348] |
| suffix-5 | 0.796 | 744.38 [493.28, 995.49] | 84.1% | 0.00260 [0.00191, 0.00328] |
| full | 1.000 | 885.56 [460.24, 1,310.88] | 100% | 0.00250 [0.00169, 0.00330] |

Cheap and suffix-2 remain distinguishable from full on paired Best Rank. Suffix-4 and suffix-5
paired intervals include zero, but this is not an equivalence result. Every partial configuration's
paired NDCG difference from full also includes zero. Compared with small-gap cells where recovery
can exceed 100%, this stronger problem produces a more conservative and interpretable curve.

## Single-seed large-model bridge gate

After the four-seed six-layer result passed, one pre-scoped seed-0 bridge used the same top-50k,
length-512, all-chunks data with 12 layers, hidden size 192, eight heads, and 11,859,456 parameters.
It is descriptive and is not pooled with the four-seed table.

Full compute improves Best Rank over frozen by 4,048.47; stale reuse retains 3,389.43 and leaves a
659.04 maintenance gap with NDCG@100 gain 0.00399. The fixed proportional suffix remains monotonic:

| Configuration | Cost / full | Best Rank gain | Recovery |
|---|---:|---:|---:|
| cheap | **0.054** | 402.55 | 61.1% |
| suffix-4 | 0.312 | 417.97 | 63.4% |
| suffix-8 | 0.653 | 546.05 | 82.9% |
| suffix-11 | 0.908 | 609.36 | 92.5% |
| full | 1.000 | 659.04 | 100% |

Full latency is 1.921 ms/user. This gate connects the separately replicated larger-model and
chunked-data axes without justifying another immediate four-seed matrix; it shows no failure that
would overturn either completed study.

## Interpretation and decision

- The earlier latest-only protocol materially underused the retained base trace.
- More complete training strengthens both the streaming-training value and the version-stale cache
  problem across four seeds.
- The optimized suffix still exposes a computation-quality curve at a 10x larger catalog and 4x
  longer context than the original main table.
- The high-quality endpoint remains expensive: suffix-5 uses about 0.8x full for 84% mean Rank
  recovery. Kernel and state-movement work, rather than another arbitrary layer search, is the next
  method bottleneck.
- Any future primary KuaiRand run should use chunked base training and record effective target counts.
  Existing latest-only results remain valid protocol-specific evidence but should not be presented
  as full-data results.

## Artifacts

- `scripts/kuairand_data_coverage.py`
- `scripts/motivation_validity.py --training-sequences {latest,all_chunks}`
- `scripts/scaling_validity.py`
- `scripts/streaming_value_control.py`
- `scripts/summarize_kuairand_data_utilization.py`
- `results/scaling/top50k_{latest,all_chunks}_{core,method}_seed{0,1,2,3}.json`
- `results/scaling/top50k_{latest,all_chunks}_streaming_control_seed{0,1,2,3}.json`
- `results/scaling/kuairand_data_utilization_summary.json`
- `results/scaling/top50k_all_chunks_large_{core,method}_seed0.json`
- `results/scaling/top50k_all_chunks_large_streaming_control_seed0.json`
- `checkpoints/scaling/top50k_{latest,all_chunks}_seed{0,1,2,3}/`
- `checkpoints/scaling/top50k_all_chunks_large_seed0/`
