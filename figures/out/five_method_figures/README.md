# Historical five-method display data

The images and generator for this display are retired. Current measurements and
figure generators are in [the figure index](../../README.md).
This directory retains `selected_points.csv`, `selection.json` and the source
record. The upstream measurements are in
[`cost_ledger.json`](../three_scale_auc_preview/cost_ledger.json).

## Display convention

The former `five_methods_horizontal.pdf` selected 6L, 10L and 16L version
edges with a mean curve. Recovery was
`100 × (AUC_method − AUC_Reuse) / (AUC_Exact − AUC_Reuse)`.
Displayed recovery was clipped to [0,100]; means interpolated the clipped scale
curves at common theoretical-FLOPs positions. The first three low-cost points
were omitted from each shared-correction panel for readability.
These are historical display choices, not the current unfiltered 15-edge protocol.

The cost denominator was one Current Exact cache rebuild, with teacher
construction, fitting, profiling/conversion and additional reads included.
It differs from the current rolling Full-minus-Reuse denominator.

| Historical displayed mean | First point (cost, recovery) | Higher-cost point | Final point |
| --- | ---: | ---: | ---: |
| Layer recomputation | (18.6%, 22.5%) | (83.7%, 82.3%) | (100.0%, 91.3%) |
| Tail recomputation | (4.6%, 2.8%) | (61.8%, 52.0%) | (100.0%, 100.0%) |
| KV translation | (32.3%, 58.6%) | (59.4%, 71.5%) | (100.0%, 71.8%) |
| Shared read correction | (3.5%, 20.3%) | (35.9%, 19.5%) | (84.4%, 19.1%) |
| Shared correction with basic user signal | (3.6%, 51.6%) | (36.2%, 52.2%) | (85.2%, 51.4%) |

## Historical settings and methods

- Each edge used 10,000 user snapshots, 1,024 retained events, following-window
  known-target feedback and pooled ROC-AUC. Max's cohort was independently
  selected; Medium/Large used matched cohorts.
- Layer: profile every contiguous layer interval on 32 teachers with 16 fixed
  candidate queries, then choose one interval per width.
- Tail: replay 32/64/128/256/512/1024 recent events under the inherited prefix.
- KV translation: 256 teachers; 1/2/3/4 source layers; convert each user's K/V.
- Shared read fitting: 16 queries per teacher, ridge 0.01, teacher budgets
  32/64/128/256/512/1024/2048/4096/7144.
  The shared correction was `delta = N*b`; the basic-user-signal variant was
  `delta = N*b + T*r`, using native read response r.
  Neither is the current nonlinear Q-v5/H-v4 probe or the later Design.
- Scoring used GPUs with CPU teacher preparation. The plotted costs were
  analytical FLOPs, independent of execution time.

The old generator was `figures/src/five_method_selected.py`.
The separate [modified display record](../five_method_figures_modified/README.md)
documents illustrative transformations; those values are not measured gains.
