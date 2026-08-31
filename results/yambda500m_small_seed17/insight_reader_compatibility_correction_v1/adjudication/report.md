# Reader compatibility-correction adjudication

Status: **reader_stage_and_persistence_gates_passed**. No label was read.

This adjudicates a reader compatibility correction, not a materializable history evidence basis. The K/V-prefix stage is already query-dependent (`activated(qK)·V`), so an early boundary there does not imply raw K/V linear substitutability.

## Stage gate

Earliest frozen stable boundary: **kv_prefix_contribution**.

| source | stage | passing edges |
| --- | --- | ---: |
| controlled | av_aggregation | 5 |
| controlled | final_readout | 5 |
| controlled | kv_prefix_contribution | 5 |
| controlled | layer_hidden | 4 |
| controlled | u_gated_update | 4 |
| real_exposed | av_aggregation | 5 |
| real_exposed | final_readout | 5 |
| real_exposed | kv_prefix_contribution | 5 |
| real_exposed | layer_hidden | 5 |
| real_exposed | u_gated_update | 5 |

## Cross-request persistence gate

Evaluated stage: **av_aggregation**; passing edges: **5/5**.

| edge | pairs | users | median cosine | same-request recovery | prior recovery | coverage-scaled prior recovery | pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v0_to_v1 | 2208 | 567 | 0.982736 | 0.998390 | 0.812668 | 0.840596 | PASS |
| v1_to_v2 | 2183 | 560 | 0.981558 | 0.996871 | 0.768858 | 0.644569 | PASS |
| v2_to_v3 | 2385 | 570 | 0.965947 | 0.996061 | 0.706230 | 0.605954 | PASS |
| v3_to_v4 | 2442 | 601 | 0.981753 | 0.998794 | 0.861577 | 0.816982 | PASS |
| v4_to_v5 | 2146 | 585 | 0.979111 | 0.997517 | 0.713536 | 0.645629 | PASS |

Matched-cost layerwise broadcast-residual canary unlocked: **yes**.

All fixed time, append-count and remaining-old-state buckets are retained in `persistence_buckets.csv`.
