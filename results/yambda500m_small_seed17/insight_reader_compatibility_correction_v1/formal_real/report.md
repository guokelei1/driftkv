# Real-request reader-correction formal

Correctness: **PASS**; labels read: **no**.

## Same-request stage recovery

| edge | stage | mean recovery |
| --- | --- | ---: |
| v0_to_v1 | av_aggregation | 0.997692 |
| v0_to_v1 | final_readout | 0.986746 |
| v0_to_v1 | kv_prefix_contribution | 0.997692 |
| v0_to_v1 | layer_hidden | 0.992084 |
| v0_to_v1 | u_gated_update | 0.992084 |
| v1_to_v2 | av_aggregation | 0.994316 |
| v1_to_v2 | final_readout | 0.978220 |
| v1_to_v2 | kv_prefix_contribution | 0.994316 |
| v1_to_v2 | layer_hidden | 0.969123 |
| v1_to_v2 | u_gated_update | 0.969123 |
| v2_to_v3 | av_aggregation | 0.992577 |
| v2_to_v3 | final_readout | 0.976724 |
| v2_to_v3 | kv_prefix_contribution | 0.992577 |
| v2_to_v3 | layer_hidden | 0.971579 |
| v2_to_v3 | u_gated_update | 0.971578 |
| v3_to_v4 | av_aggregation | 0.998347 |
| v3_to_v4 | final_readout | 0.986756 |
| v3_to_v4 | kv_prefix_contribution | 0.998348 |
| v3_to_v4 | layer_hidden | 0.991864 |
| v3_to_v4 | u_gated_update | 0.991864 |
| v4_to_v5 | av_aggregation | 0.996349 |
| v4_to_v5 | final_readout | 0.974025 |
| v4_to_v5 | kv_prefix_contribution | 0.996350 |
| v4_to_v5 | layer_hidden | 0.982590 |
| v4_to_v5 | u_gated_update | 0.982590 |

## Adjacent-request persistence

| edge | stage | median cosine | median coverage-scaled recovery |
| --- | --- | ---: | ---: |
| v0_to_v1 | av_aggregation | 0.982736 | 0.840596 |
| v0_to_v1 | final_readout | 0.930318 | 0.492709 |
| v0_to_v1 | layer_hidden | 0.969711 | 0.646975 |
| v0_to_v1 | u_gated_update | 0.970394 | 0.646975 |
| v1_to_v2 | av_aggregation | 0.981558 | 0.644569 |
| v1_to_v2 | final_readout | 0.946471 | 0.138418 |
| v1_to_v2 | layer_hidden | 0.961885 | 0.002911 |
| v1_to_v2 | u_gated_update | 0.962781 | 0.002911 |
| v2_to_v3 | av_aggregation | 0.965947 | 0.605954 |
| v2_to_v3 | final_readout | 0.897329 | 0.000000 |
| v2_to_v3 | layer_hidden | 0.926583 | 0.000000 |
| v2_to_v3 | u_gated_update | 0.929703 | 0.000000 |
| v3_to_v4 | av_aggregation | 0.981753 | 0.816982 |
| v3_to_v4 | final_readout | 0.947400 | 0.319736 |
| v3_to_v4 | layer_hidden | 0.959926 | 0.500385 |
| v3_to_v4 | u_gated_update | 0.960898 | 0.501230 |
| v4_to_v5 | av_aggregation | 0.979111 | 0.645629 |
| v4_to_v5 | final_readout | 0.926916 | 0.000361 |
| v4_to_v5 | layer_hidden | 0.967525 | 0.227991 |
| v4_to_v5 | u_gated_update | 0.968748 | 0.227991 |
