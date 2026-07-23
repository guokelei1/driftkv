# Scaling-v1: fixed optimized suffix

## Scope

This phase freezes the optimized deepest-suffix operator before changing scale. Every propagation
region executes current full blocks except at its terminal layer, which executes only current
`Norm + Wk/Wv`. The tested budgets are approximately one third of the model, two thirds of the
model, all but the first layer, and the two endpoints cheap-all and optimized full recompute.

The phase contains four independent studies:

1. KuaiRand quality at active sequence lengths 32, 64, and 128, using the same six-layer
   checkpoints and four training seeds.
2. Resident-GPU operator cost over sequence lengths 16 through 512 and batch sizes 1 through 128.
3. KuaiRand depth scaling at 3, 6, and 9 layers with the same hidden size 96, four heads, head
   dimension 24, training protocol, and four seeds.
4. A controlled update path and a MovieLens-1M chronological-holdout transfer check.

No arbitrary interval was searched in this phase. Configuration names were fixed before the
four-seed runs.

## Resident-GPU shape cost

The table uses synthetic full-length tensors resident on one A40, CUDA events, 15 repetitions, and
optimized full K/V recompute as the denominator. Transfer, allocation, admission, and scheduling
remain outside the timed region.

| Sequence length, batch 32 | cheap | suffix-2 | suffix-4 | suffix-5 | full latency |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.184 | 0.378 | 0.634 | 0.756 | 1.97 ms |
| 128 | 0.189 | 0.377 | 0.636 | 0.768 | 2.08 ms |
| 256 | 0.111 | 0.285 | 0.616 | 0.786 | 4.77 ms |
| 512 | 0.058 | 0.248 | 0.612 | 0.795 | 14.41 ms |

The full operator is under-utilized through length 128 and then exposes its quadratic attention
cost. Cheap and suffix-2 become relatively cheaper at long context. Suffix-5 remains near 0.8x
because it still executes four complete blocks in the six-layer model.

| Batch size, length 128 | cheap | suffix-2 | suffix-4 | suffix-5 | full throughput |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.193 | 0.384 | 0.632 | 0.755 | 538 users/s |
| 32 | 0.191 | 0.376 | 0.640 | 0.770 | 15,547 users/s |
| 128 | 0.132 | 0.308 | 0.630 | 0.788 | 20,402 users/s |

Batching improves absolute throughput substantially. It does not remove the high-quality-end cost:
suffix-5 remains 0.76-0.79x full across the tested batch range. Optimized full K/V matches
`compute_kv` with maximum absolute error 0 for every shape.

## Active sequence length

The six-layer theta-0 to theta-5 comparison uses 300 users per seed and full-catalog ranking. The
table reports the ratio of cross-seed mean Best Rank gain to the corresponding full-recompute gain.

| Active length | Full Best Rank gain, 95% seed CI | cheap | suffix-2 | suffix-4 | suffix-5 |
|---:|---:|---:|---:|---:|---:|
| 32 | 134.46 [71.22, 197.69] | 44.9% | 54.3% | 75.4% | 85.9% |
| 64 | 95.58 [49.39, 141.76] | 63.7% | 75.5% | 93.0% | 96.9% |
| 128 | 85.32 [53.74, 116.91] | 72.9% | 84.6% | 101.3% | 103.0% |

The method remains a computation-quality curve at every tested length. Recovery above 100% at
length 128 does not by itself establish superiority: full is the version-consistency reference,
and this cell's paired evidence does not separate their task quality. Because the model was trained
at length 128, the absolute-gap trend across truncation lengths is descriptive, not a claim that
shorter production histories are intrinsically more version-sensitive.

## Model depth

Depth 3 and 9 were newly trained with the six-layer width and all three depths use four seeds. The
table compares proportional suffix budgets within each depth.

| Layers | cheap cost / recovery | one-third suffix cost / recovery | two-thirds suffix cost / recovery | Full Best Rank gain |
|---:|---:|---:|---:|---:|
| 3 | 0.202 / 66.7% | 0.277 / 67.0% | 0.554 / 94.6% | 106.45 |
| 6 | 0.187 / 72.9% | 0.373 / 84.6% | 0.636 / 101.3% | 85.32 |
| 9 | 0.181 / 70.8% | 0.399 / 87.4% | 0.668 / 95.3% | 80.13 |

The Pareto structure survives the first depth increase. The exact recovery is noisy and the
theta-0 to theta-5 parameter distances differ across depths, so this is not a depth-optimality
claim. It does remove the concern that the curve exists only at six layers.

## Controlled update magnitude

This study interpolates along the observed six-layer seed-specific direction

$$
\theta(\alpha)=\theta_0+\alpha(\theta_5-\theta_0),
$$

and evaluates the same theta-5 context and future positives. Intermediate points are controlled
parameter states, not deployable trained checkpoints.

| Alpha | Relative parameter distance | Stale K/V error | Full Best Rank gain, 95% seed CI | cheap / suffix-2 / suffix-4 / suffix-5 recovery |
|---:|---:|---:|---:|---:|
| 0.25 | 0.0390 | 0.2057 | 18.55 [14.08, 23.02] | 45.7 / 51.8 / 68.7 / 77.2% |
| 0.50 | 0.0779 | 0.3926 | 67.76 [43.69, 91.82] | 46.1 / 53.6 / 75.1 / 83.6% |
| 0.75 | 0.1168 | 0.5453 | 105.69 [75.01, 136.38] | 54.0 / 64.7 / 86.7 / 92.8% |
| 1.00 | 0.1558 | 0.6563 | 85.32 [53.74, 116.91] | 72.9 / 84.6 / 101.3 / 103.0% |

K/V inconsistency grows monotonically along the update direction. Best Rank does not: its gap
peaks at alpha 0.75 and falls at the trained endpoint. Parameter norm or cache error is therefore
an update-level severity feature, not a calibrated predictor of recommendation utility.

## MovieLens-1M transfer check

The `pilot20` records contain the same 5,923 users at three consecutive chronological holdouts.
The base model trains on each train history. Version 1 ingests the train target and evaluates the
dev target; version 2 ingests the dev target and evaluates the test target. The model has six
layers, hidden size 96, sequence length 128, four base epochs, two update epochs, and 1,000 fixed
evaluation users per seed. Full-catalog and provided 20-candidate rankings are both recorded.

| Version | Relative parameter distance | Full Best Rank maintenance gain, 95% seed CI | NDCG@100 gain, 95% seed CI |
|---:|---:|---:|---:|
| 1 | 0.0495 | 1.87 [0.87, 2.86] | 0.00001 [-0.00273, 0.00275] |
| 2 | 0.0790 | 1.48 [-6.37, 9.33] | 0.00034 [-0.00083, 0.00151] |

The two-update chain does not reproduce the strong KuaiRand maintenance gap. Candidate-20 gains
are also inconsistent. Recovery ratios are intentionally omitted because the fresh-over-reuse
denominator is too small. This is a valid negative boundary result: the operator transfers and
optimized full remains exact, but the problem strength has not generalized under this short
MovieLens stream.

All optimized full K/V comparisons have maximum absolute error 0. Full-vs-incremental hidden-state
parity is below `4.8e-6` after grouping equal effective lengths and evaluating matmuls at highest
precision.

## Decisions

- Keep the optimized deepest suffix frozen; all three KuaiRand scale axes preserve a useful curve.
- Do not revive arbitrary-layer or per-user selection. No new evidence supports it.
- Do not claim cross-dataset generality. A second dataset with a longer real update sequence is a
  prerequisite for that claim.
- Treat suffix-5 as a high-quality but modest-saving point. The next method work should be driven
  by end-to-end profiling, not another layer search.
- At the end of this phase, mixed cache versions, end-to-end state movement, and a stronger second
  stream remained open.

Subsequent work increased both data/model scale and repaired latest-only base-data underuse. See
`KUAIRAND_FACTORIAL_V1.md` and `KUAIRAND_DATA_UTILIZATION_V1.md`. Their stronger cells show that
full is a fidelity reference but not always a ranking-quality ceiling, so the “above 100% is
noise” shorthand is retired. A later data audit rejected Taobao UserBehavior as the primary second
stream because it lacks true unclicked impressions. The current cross-dataset plan is maintained
only in `docs/08_core_insights_and_roadmap.md`.

## Artifacts

- `scripts/operator_cost_scaling.py`
- `scripts/scaling_validity.py`
- `scripts/movielens_scaling.py`
- `scripts/summarize_scaling.py`
- `results/scaling/operator_cost_seed0.json`
- `results/scaling/sequence_length_seed{0,1,2,3}.json`
- `results/scaling/update_magnitude_seed{0,1,2,3}.json`
- `results/scaling/depth{3,9}_{core,method}_seed{0,1,2,3}.json`
- `results/scaling/movielens_seed{0,1,2,3}.json`
- `results/scaling/multiaxis_summary.json`
- `checkpoints/scaling/depth{3,9}_seed{0,1,2,3}/`
- `checkpoints/scaling/movielens_seed{0,1,2,3}/`
