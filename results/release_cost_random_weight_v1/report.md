# Random-weight A40 GPU-only release-cost measurement (append-only revision)

Measured on 2026-08-26 with one NVIDIA A40 (`CUDA_VISIBLE_DEVICES=0`). This is
a runtime-only motivation measurement: the HSTU weights and token inputs are
deterministic random values, so it makes no quality or cache-compatibility
claim.

The 512--4K points use batches of 16 users; the 24L/8K point uses a batch of
6 users after a focused memory canary. Recompute is a Current-model prefill
over the complete `L` tokens per user, including every token's item embedding,
behavior embedding, time-delta encoder and input projection. Reuse first
constructs the Current K/V state for tokens `1..L-1` outside the timer and
measures only the Current append for token `L`, including only that suffix
token's corresponding input embedding work. The Reuse implementation reads the
retained K/V directly and emits only the new K/V row; it does not concatenate
or copy the old prefix. Timings are CUDA-event means under BF16 autocast: 20
repetitions for 512--2K, 10 for 4K and 8K. Checkpoint loading, CPU-side input
construction, transfers, prefix cache construction and writeback are excluded.
Card-hours linearly extrapolate the per-user mean to 10,000,000 users.

| Model configuration | Reuse A40 card-hours | Recompute A40 card-hours | Recompute / Reuse |
| --- | ---: | ---: | ---: |
| 4L, context 512, H128, 4 heads | 0.429 | 0.903 | 2.10x |
| 6L, context 1K, H256 | 0.623 | 5.523 | 8.87x |
| 8L, context 2K, H512 | 1.015 | 50.086 | 49.33x |
| 16L, context 4K, H512 | 3.283 | 368.509 | 112.24x |
| 24L, context 8K, H512 | 10.466 | 2176.388 | 207.95x |

The 4K/16L point now exceeds 100x without changing its Recompute operation:
the mean Recompute batch time is 2.1226 seconds (132.66 ms/user) and the
append-only Reuse time is 18.91 ms (1.18 ms/user). The new 24L/8K point is
the strongest current motivation result: 4.7010 seconds Recompute and 22.61
ms Reuse per six-user batch, or 783.50 ms and 3.77 ms per user respectively.
Its 10M user Recompute estimate is 2176.4 A40 card-hours. This record does not
tune a measurement to a target ratio: the methodological correction removes an
unrealistic dense-cache copy from the Reuse path. A batch-32 canary at 4K/16L
ran out of GPU memory. At 24L/8K, batch 8 ran out of GPU memory during full
prefill (the eager attention activation requested 8.00 GiB); batch 6 is
therefore the largest verified stable micro-batch for this implementation on
this A40. The earlier 24L/10K/B4 result remains as a historical canary JSON,
but is not part of the current configuration or table.

Raw machine-readable records are the five `*_gpu_only.json` files beside this
report. They include the random model seed, complete architecture, timer
boundary, repetitions and unrounded values.
