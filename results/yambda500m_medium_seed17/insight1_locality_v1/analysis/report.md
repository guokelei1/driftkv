# Medium Insight 1 locality diagnostic

All five frozen D14 adjacent edges and all 34 pre-specified Exact-KV splice configurations are included. No request label is used.

## Edge-equal best-observed frontier

| family | budget | cost | edge_equal_best_observed_recovery | minimum_edge_recovery | maximum_edge_recovery |
| --- | --- | --- | --- | --- | --- |
| layer | k1 | 0.166667 | 0.455614 | 0.157834 | 0.938777 |
| layer | k2 | 0.333333 | 0.483584 | 0.230331 | 0.938022 |
| layer | k3 | 0.5 | 0.843743 | 0.717618 | 0.985127 |
| layer | k4 | 0.666667 | 0.974351 | 0.944096 | 0.99502 |
| token | p10 | 0.0996094 | 0.300112 | 0.106848 | 0.466869 |
| token | p20 | 0.200195 | 0.411682 | 0.16155 | 0.640668 |
| token | p40 | 0.400391 | 0.569182 | 0.157513 | 0.812805 |
| token | p80 | 0.799805 | 0.897392 | 0.721425 | 0.97247 |
| window | w128 | 0.125 | 0.116177 | 0.0899997 | 0.140803 |
| window | w256 | 0.25 | 0.233981 | 0.186354 | 0.276613 |
| window | w512 | 0.5 | 0.460433 | 0.375759 | 0.531798 |
| window | w768 | 0.75 | 0.711092 | 0.629015 | 0.768869 |

## Globally fixed configuration winners

| family | budget | cost | config_id | edge_equal_recovery |
| --- | --- | --- | --- | --- |
| layer | k1 | 0.166667 | layer_1 | 0.455614 |
| layer | k2 | 0.333333 | layer_1_2 | 0.483584 |
| layer | k3 | 0.5 | layer_1_2_3 | 0.843743 |
| layer | k4 | 0.666667 | layer_1_2_3_4 | 0.974351 |
| token | p10 | 0.0996094 | token_p10_read_delta | 0.272226 |
| token | p20 | 0.200195 | token_p20_read_delta | 0.396705 |
| token | p40 | 0.400391 | token_p40_read_delta | 0.569182 |
| token | p80 | 0.799805 | token_p80_read_delta | 0.897392 |
| window | w128 | 0.125 | window_128_768_896 | 0.113573 |
| window | w256 | 0.25 | window_256_512_768 | 0.228551 |
| window | w512 | 0.5 | window_512_512_1024 | 0.460433 |
| window | w768 | 0.75 | window_768_256_1024 | 0.711092 |

Exact-KV splices are optimistic diagnostic interventions, not dependency-closed migration actions. The x-axis is theoretical KV coverage, not GPU FLOPs or wall time.
