# Real-request reader-correction canary

Correctness: **PASS**; labels read: **no**.

## Same-request stage recovery

| edge | stage | mean recovery |
| --- | --- | ---: |
| v0_to_v1 | av_aggregation | 0.994706 |
| v0_to_v1 | final_readout | 0.945195 |
| v0_to_v1 | kv_prefix_contribution | 0.994728 |
| v0_to_v1 | layer_hidden | 0.982661 |
| v0_to_v1 | u_gated_update | 0.982652 |
| v1_to_v2 | av_aggregation | 0.986396 |
| v1_to_v2 | final_readout | 0.954375 |
| v1_to_v2 | kv_prefix_contribution | 0.986405 |
| v1_to_v2 | layer_hidden | 0.948851 |
| v1_to_v2 | u_gated_update | 0.948872 |
| v2_to_v3 | av_aggregation | 0.987867 |
| v2_to_v3 | final_readout | 0.953245 |
| v2_to_v3 | kv_prefix_contribution | 0.987852 |
| v2_to_v3 | layer_hidden | 0.951818 |
| v2_to_v3 | u_gated_update | 0.951816 |
| v3_to_v4 | av_aggregation | 0.989930 |
| v3_to_v4 | final_readout | 0.960727 |
| v3_to_v4 | kv_prefix_contribution | 0.989930 |
| v3_to_v4 | layer_hidden | 0.958316 |
| v3_to_v4 | u_gated_update | 0.957585 |
| v4_to_v5 | av_aggregation | 0.979651 |
| v4_to_v5 | final_readout | 0.966800 |
| v4_to_v5 | kv_prefix_contribution | 0.979673 |
| v4_to_v5 | layer_hidden | 0.952054 |
| v4_to_v5 | u_gated_update | 0.951992 |

## Adjacent-request persistence

| edge | stage | median cosine | median coverage-scaled recovery |
| --- | --- | ---: | ---: |
| v0_to_v1 | av_aggregation | 0.987165 | 0.923200 |
| v0_to_v1 | final_readout | 0.952808 | 0.826922 |
| v0_to_v1 | layer_hidden | 0.978449 | 0.749850 |
| v0_to_v1 | u_gated_update | 0.978449 | 0.749850 |
| v1_to_v2 | av_aggregation | 0.982582 | 0.737474 |
| v1_to_v2 | final_readout | 0.954255 | 0.416297 |
| v1_to_v2 | layer_hidden | 0.964223 | 0.283078 |
| v1_to_v2 | u_gated_update | 0.964223 | 0.283078 |
| v2_to_v3 | av_aggregation | 0.958165 | 0.583706 |
| v2_to_v3 | final_readout | 0.881535 | 0.000000 |
| v2_to_v3 | layer_hidden | 0.920739 | 0.000000 |
| v2_to_v3 | u_gated_update | 0.925393 | 0.000000 |
| v3_to_v4 | av_aggregation | 0.975041 | 0.902213 |
| v3_to_v4 | final_readout | 0.935295 | 0.437058 |
| v3_to_v4 | layer_hidden | 0.949039 | 0.764514 |
| v3_to_v4 | u_gated_update | 0.949039 | 0.764514 |
| v4_to_v5 | av_aggregation | 0.979451 | 0.417116 |
| v4_to_v5 | final_readout | 0.922856 | 0.000000 |
| v4_to_v5 | layer_hidden | 0.960553 | 0.023217 |
| v4_to_v5 | u_gated_update | 0.960549 | 0.023217 |
