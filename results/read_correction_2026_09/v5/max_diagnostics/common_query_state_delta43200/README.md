# Common-query cache-state diagnostic: both fixed time inputs

All three cases completed at both predefined query deltas, 0 and 43,200 seconds. This is the final mechanism probe; no further fitting or time selection was performed. Complete scalar comparisons, source paths and hashes are in [time_comparison.json](time_comparison.json). Per-user, append-bucket, teacher-alignment and resource summaries remain in each case's `summary.json` and in the original `../common_query_state/` directory.

## Construction and scope

- Use the first 64 already reserved calibration UIDs, without feedback labels. Retain only terminal histories of exactly 1,024 events: Max V0→V1 retains 55, Max V3→V4 retains 58, Large V0→V1 retains 52. Retained UID lists are identical between time controls.
- Every user within a case sees the same 32 uniformly sampled known-catalog candidates, seed `[17, 0]`, and the same fixed query delta. Candidate IDs are identical between the two time controls. Layer 0 therefore receives the same actual query tensor and N for every user and cache state.
- Compare pure Parent K/V against genuine Parent-prefix plus Current rolling appends. Balanced, label-free append targets are 0, 256, 512 and 896; timestamp-boundary rounding is retained. Both states end at exactly the same terminal events. No K/V is persisted by this probe.
- All 64 terminal event hashes, endpoints, per-user original query deltas and Full teacher histories align between capture paths. The zero-append bucket's paired target change is exactly zero in all six completed cases.
- The tested Q predictor is the original v5 cross-phi C512 joint artifact's layer 0. These 64 UIDs overlap that model's calibration users. This is a mechanistic intervention on known users with different common queries, not an independent quality evaluation.

For each candidate, the empirical best vector shared across users is the mean required correction. Its remaining MSE is the variance across users; pooled-state results average both states for the same users. This describes a finite-cohort first-layer constraint for a deterministic function of identical `(q,N)`. It is not a population lower bound or a claim that later layers cannot encode user/cache information.

The hidden-update analysis applies the actual frozen legacy projection:

`delta_hidden = linear_without_bias(delta_read, out_proj.weight) * silu(gate_proj(norm(query_embedding)))`.

Output bias and the block's residual connection cancel in a difference. Full-update normalization uses the actual Full history read plus query self contribution, followed by the original output projection and gate. Thus cross-model comparisons need not rely on raw read-coordinate magnitudes.

## Results: oracle residual variance

Percentages below are residual MSE divided by the MSE of zero correction, within the same space and cohort. Smaller values mean a common query-only vector can explain more of the required correction.

| Case | Delta, seconds | Read: pure | Read: mixed | Hidden: pure | Hidden: mixed | Hidden: pooled states |
|---|---:|---:|---:|---:|---:|---:|
| Max V0→V1 | 0 | 51.25% | 60.64% | 54.19% | 62.51% | 58.89% |
| Max V0→V1 | 43,200 | 50.85% | 60.34% | 51.99% | 61.15% | 57.07% |
| Max V3→V4 | 0 | 29.71% | 48.53% | 30.80% | 49.26% | 40.67% |
| Max V3→V4 | 43,200 | 29.22% | 48.31% | 31.12% | 50.05% | 41.15% |
| Large V0→V1 | 0 | 29.03% | 45.41% | 13.09% | 32.79% | 23.61% |
| Large V0→V1 | 43,200 | 27.27% | 43.95% | 12.24% | 32.10% | 22.87% |

The two targeted Max failures have a larger empirical hidden-space common-vector residual than the Large control at both fixed times. This is descriptive evidence of a stronger first-layer shared-mapping limitation in these cases, with no claim that it alone causes their final AUC behavior.

## Required target changes when only cache state changes

| Case | Delta, seconds | Paired hidden target change / pure target RMS | Paired change / Full update RMS | Mean state shift / Full update RMS |
|---|---:|---:|---:|---:|
| Max V0→V1 | 0 | 42.33% | 10.43% | 5.87% |
| Max V0→V1 | 43,200 | 41.87% | 10.30% | 6.00% |
| Max V3→V4 | 0 | 51.15% | 6.87% | 4.56% |
| Max V3→V4 | 43,200 | 50.58% | 6.44% | 4.31% |
| Large V0→V1 | 0 | 49.22% | 10.68% | 7.31% |
| Large V0→V1 | 43,200 | 49.32% | 9.88% | 6.80% |

State dependence is substantial in Large as well as Max. It is not a Max-specific mechanism. The frozen Q layer emits exactly the same tensor for the paired states because both q and N are held fixed, while the required correction changes. Append-bucket statistics are retained, including all zero-append controls; between-bucket differences are descriptive because the buckets contain different users.

## Actual Q512 error: retain, but do not interpret as online AUC

The following ratios compare the frozen Q512 prediction's hidden-update residual MSE with zero-correction MSE. They are not oracle quantities or quality metrics.

| Case | Delta, seconds | Pure error ratio | Mixed error ratio | Pooled error ratio |
|---|---:|---:|---:|---:|
| Max V0→V1 | 0 | 2.799× | 4.600× | 3.409× |
| Max V0→V1 | 43,200 | 4.075× | 7.918× | 5.388× |
| Max V3→V4 | 0 | 25.468× | 56.426× | 35.498× |
| Max V3→V4 | 43,200 | 5.870× | 10.734× | 7.457× |
| Large V0→V1 | 0 | 0.280× | 0.757× | 0.445× |
| Large V0→V1 | 43,200 | 2.315× | 3.113× | 2.590× |

Actual Q error is sensitive to the chosen time input, including a worsening Large control at 12 hours. It cannot support an online Max-only failure claim. The stable oracle/paired-target pattern provides the cleaner mechanism evidence.

The original 64 calibration users all had positive query deltas: medians were 52,200 seconds for Max1, 35,792.5 for Max4 and 39,655 for Large1. Both controls deliberately replace those user-specific times with one common time. All original quantiles remain in the summaries. At 0 seconds there were respectively 11,568, 12,644 and 10,722 adjacent tied-timestamp pairs; mapped-item and known-item inversions were zero in all three cases.

## Numerical checks and retained first attempts

The original Max1 run passed the stricter full-tensor `rtol=atol=2e-5` teacher check and was not rerun at delta 0. Large1 and Max4 initially stopped on sparse deep-layer differences after terminal event hashes had already matched. Large1's first check rejected only 18 of 3,276,800 elements, with maximum reported discrepancy 3.59e-5 in deeper layers.

Only the two unfinished cases were rerun. Their measured layer-0 teachers still had to pass the original strict tolerance; all-layer maximum differences and relative RMS differences were additionally recorded. Their whole-teacher relative RMS differences were about 5e-7, consistent with different capture batch grouping, while the exact terminal history hashes remained equal. First-attempt exits/settings and full appended logs remain in each original case directory. The original source is retained as `../common_query_state/source_before_layer_check_refinement.py`; the two original probe source files and both time-control outputs remain intact.

All final cases exited 0. Successful runtimes were about 88/100/48 seconds at delta 0 and 91/98/50 seconds at 12 hours for Max1/Max4/Large1. GPU peak allocation was approximately 9.46 GiB for Max and 6.54 GiB for Large. No additional experiment is pending.
