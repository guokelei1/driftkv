# Reader compatibility-correction stage localization

Scope: `canary`; correctness: **PASS**.

The correction is a same-request signed oracle derived from coherent Current-Exact and Parent-Reuse reader traces. It is not a materialized history basis or an executable action.

## Largest-width score-gap recovery

| source | edge | stage | recovery |
| --- | --- | --- | ---: |
| controlled | v0_to_v1 | av_aggregation | 0.990855 |
| controlled | v0_to_v1 | final_readout | 0.877152 |
| controlled | v0_to_v1 | kv_prefix_contribution | 0.990855 |
| controlled | v0_to_v1 | layer_hidden | 0.966793 |
| controlled | v0_to_v1 | u_gated_update | 0.966793 |
| controlled | v1_to_v2 | av_aggregation | 0.966936 |
| controlled | v1_to_v2 | final_readout | 0.837576 |
| controlled | v1_to_v2 | kv_prefix_contribution | 0.966935 |
| controlled | v1_to_v2 | layer_hidden | 0.837690 |
| controlled | v1_to_v2 | u_gated_update | 0.837694 |
| controlled | v2_to_v3 | av_aggregation | 0.953464 |
| controlled | v2_to_v3 | final_readout | 0.765661 |
| controlled | v2_to_v3 | kv_prefix_contribution | 0.953464 |
| controlled | v2_to_v3 | layer_hidden | 0.669272 |
| controlled | v2_to_v3 | u_gated_update | 0.669272 |
| controlled | v3_to_v4 | av_aggregation | 0.996219 |
| controlled | v3_to_v4 | final_readout | 0.955706 |
| controlled | v3_to_v4 | kv_prefix_contribution | 0.996219 |
| controlled | v3_to_v4 | layer_hidden | 0.967122 |
| controlled | v3_to_v4 | u_gated_update | 0.967122 |
| controlled | v4_to_v5 | av_aggregation | 0.972072 |
| controlled | v4_to_v5 | final_readout | 0.827068 |
| controlled | v4_to_v5 | kv_prefix_contribution | 0.972074 |
| controlled | v4_to_v5 | layer_hidden | 0.861671 |
| controlled | v4_to_v5 | u_gated_update | 0.861670 |
| real_exposed_canary | v0_to_v1 | av_aggregation | 0.997584 |
| real_exposed_canary | v0_to_v1 | final_readout | 0.996924 |
| real_exposed_canary | v0_to_v1 | kv_prefix_contribution | 0.997587 |
| real_exposed_canary | v0_to_v1 | layer_hidden | 0.989149 |
| real_exposed_canary | v0_to_v1 | u_gated_update | 0.989149 |
| real_exposed_canary | v1_to_v2 | av_aggregation | 0.998480 |
| real_exposed_canary | v1_to_v2 | final_readout | 0.996492 |
| real_exposed_canary | v1_to_v2 | kv_prefix_contribution | 0.998480 |
| real_exposed_canary | v1_to_v2 | layer_hidden | 0.992283 |
| real_exposed_canary | v1_to_v2 | u_gated_update | 0.992283 |
| real_exposed_canary | v2_to_v3 | av_aggregation | 0.977886 |
| real_exposed_canary | v2_to_v3 | final_readout | 0.994141 |
| real_exposed_canary | v2_to_v3 | kv_prefix_contribution | 0.977886 |
| real_exposed_canary | v2_to_v3 | layer_hidden | 0.989561 |
| real_exposed_canary | v2_to_v3 | u_gated_update | 0.989560 |
| real_exposed_canary | v3_to_v4 | av_aggregation | 0.995365 |
| real_exposed_canary | v3_to_v4 | final_readout | 0.984718 |
| real_exposed_canary | v3_to_v4 | kv_prefix_contribution | 0.995359 |
| real_exposed_canary | v3_to_v4 | layer_hidden | 0.989316 |
| real_exposed_canary | v3_to_v4 | u_gated_update | 0.989316 |
| real_exposed_canary | v4_to_v5 | av_aggregation | 0.996611 |
| real_exposed_canary | v4_to_v5 | final_readout | 0.981060 |
| real_exposed_canary | v4_to_v5 | kv_prefix_contribution | 0.996611 |
| real_exposed_canary | v4_to_v5 | layer_hidden | 0.990414 |
| real_exposed_canary | v4_to_v5 | u_gated_update | 0.990424 |

## Largest-width signed shared energy

| source | edge | stage | shared energy |
| --- | --- | --- | ---: |
| controlled | v0_to_v1 | av_aggregation | 0.999607 |
| controlled | v0_to_v1 | final_readout | 0.998570 |
| controlled | v0_to_v1 | kv_prefix_contribution | 0.999619 |
| controlled | v0_to_v1 | layer_hidden | 0.999277 |
| controlled | v0_to_v1 | u_gated_update | 0.999277 |
| controlled | v1_to_v2 | av_aggregation | 0.999420 |
| controlled | v1_to_v2 | final_readout | 0.998321 |
| controlled | v1_to_v2 | kv_prefix_contribution | 0.999485 |
| controlled | v1_to_v2 | layer_hidden | 0.998952 |
| controlled | v1_to_v2 | u_gated_update | 0.998952 |
| controlled | v2_to_v3 | av_aggregation | 0.999446 |
| controlled | v2_to_v3 | final_readout | 0.996231 |
| controlled | v2_to_v3 | kv_prefix_contribution | 0.999509 |
| controlled | v2_to_v3 | layer_hidden | 0.998795 |
| controlled | v2_to_v3 | u_gated_update | 0.998795 |
| controlled | v3_to_v4 | av_aggregation | 0.999406 |
| controlled | v3_to_v4 | final_readout | 0.998824 |
| controlled | v3_to_v4 | kv_prefix_contribution | 0.999450 |
| controlled | v3_to_v4 | layer_hidden | 0.999069 |
| controlled | v3_to_v4 | u_gated_update | 0.999069 |
| controlled | v4_to_v5 | av_aggregation | 0.999303 |
| controlled | v4_to_v5 | final_readout | 0.998691 |
| controlled | v4_to_v5 | kv_prefix_contribution | 0.999347 |
| controlled | v4_to_v5 | layer_hidden | 0.999076 |
| controlled | v4_to_v5 | u_gated_update | 0.999076 |
| real_exposed_canary | v0_to_v1 | av_aggregation | 0.999936 |
| real_exposed_canary | v0_to_v1 | final_readout | 0.999540 |
| real_exposed_canary | v0_to_v1 | kv_prefix_contribution | 0.999927 |
| real_exposed_canary | v0_to_v1 | layer_hidden | 0.999828 |
| real_exposed_canary | v0_to_v1 | u_gated_update | 0.999828 |
| real_exposed_canary | v1_to_v2 | av_aggregation | 0.749820 |
| real_exposed_canary | v1_to_v2 | final_readout | 0.999578 |
| real_exposed_canary | v1_to_v2 | kv_prefix_contribution | 0.749934 |
| real_exposed_canary | v1_to_v2 | layer_hidden | 0.749736 |
| real_exposed_canary | v1_to_v2 | u_gated_update | 0.749737 |
| real_exposed_canary | v2_to_v3 | av_aggregation | 0.998984 |
| real_exposed_canary | v2_to_v3 | final_readout | 0.999700 |
| real_exposed_canary | v2_to_v3 | kv_prefix_contribution | 0.999052 |
| real_exposed_canary | v2_to_v3 | layer_hidden | 0.998922 |
| real_exposed_canary | v2_to_v3 | u_gated_update | 0.998922 |
| real_exposed_canary | v3_to_v4 | av_aggregation | 0.999905 |
| real_exposed_canary | v3_to_v4 | final_readout | 0.999264 |
| real_exposed_canary | v3_to_v4 | kv_prefix_contribution | 0.999917 |
| real_exposed_canary | v3_to_v4 | layer_hidden | 0.999790 |
| real_exposed_canary | v3_to_v4 | u_gated_update | 0.999790 |
| real_exposed_canary | v4_to_v5 | av_aggregation | 0.999751 |
| real_exposed_canary | v4_to_v5 | final_readout | 0.999596 |
| real_exposed_canary | v4_to_v5 | kv_prefix_contribution | 0.999804 |
| real_exposed_canary | v4_to_v5 | layer_hidden | 0.999591 |
| real_exposed_canary | v4_to_v5 | u_gated_update | 0.999591 |

Maximum native/full reconstruction error: 2.8610229e-06.
No label was read.
