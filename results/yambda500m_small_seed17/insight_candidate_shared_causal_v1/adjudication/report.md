# Candidate-shared signed causal adjudication

Progression gate: **PASS**.

This adjudication uses signed per-head HSTU contributions without candidate-wise normalization. The real-exposed banks contain only actual same-UID, same-timestamp requests, and every raw artifact was sealed before labels were joined.

## Controlled 3,000-user width-64 intervention

| edge | reuse probability gap | shared probability gap | residual probability gap | shared gap recovery |
| --- | --- | --- | --- | --- |
| v0_to_v1 | 0.0048324748 | 2.0266765e-05 | 0.0048331993 | 99.580613 |
| v1_to_v2 | 0.0017097576 | 2.6420687e-05 | 0.0017079403 | 98.454712 |
| v2_to_v3 | 0.0014367486 | 2.8988187e-05 | 0.0014364175 | 97.982376 |
| v3_to_v4 | 0.004704316 | 1.7086925e-05 | 0.0047024515 | 99.636782 |
| v4_to_v5 | 0.0011376625 | 1.7206409e-05 | 0.0011371055 | 98.487565 |

## Real-exposed candidate distribution

| edge | widths | shared recovery min | shared recovery max | shared better residual |
| --- | --- | --- | --- | --- |
| v0_to_v1 | 4 | 99.643422 | 99.792507 | 4 |
| v1_to_v2 | 4 | 98.717919 | 99.498129 | 4 |
| v2_to_v3 | 4 | 99.054864 | 99.297598 | 4 |
| v3_to_v4 | 4 | 99.803964 | 99.842408 | 4 |
| v4_to_v5 | 4 | 99.458271 | 99.655212 | 4 |

Across nonzero head observations, the signed shared component carries 99.917046% of delta energy. Shared-only mean absolute logit gap to Current Exact is 5.5804434e-05, versus 0.015508635 for Reuse. The maximum shared-only absolute AUC/log-loss deltas to Exact across 20 edge-width cells are 9.390024e-05/1.4710608e-06.

Maximum native/full reconstruction errors are 9.5367432e-07/9.5367432e-07.

## Decision boundary

The signed causal and real-candidate-distribution gates pass. This supports a candidate-broadcast user-evidence component plus a small contextual residual as a real reader structure, rather than a norm-normalization artifact. It does **not** make shared/residual oracle interventions executable, admit a new action, or establish superiority over Design 0. The next allowed step is one frozen candidate-independent Current-HSTU evidence-basis mechanism at matched compute, carriers, raw I/O and state I/O.
