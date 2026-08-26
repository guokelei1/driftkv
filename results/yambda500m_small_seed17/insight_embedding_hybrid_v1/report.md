# Embedding/input-stack versus Transformer origin

All hybrid producers are diagnostic interventions. The Current model remains the cache consumer and candidate/query scorer. These paths are not executable migration actions.

| edge | producer_path | requests | path_minus_exact_log_loss | mean_abs_probability_shift | mean_bernoulli_js | parent_all_gap | fraction_of_parent_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | current_exact | 128 | 0 | 0 | 0 | 0.005224 | 0 |
| v0_to_v1 | parent_all | 128 | 0.000967405 | 0.005224 | 5.32911e-05 | 0.005224 | 1 |
| v0_to_v1 | parent_item_embedding | 128 | 2.28729e-05 | 2.77482e-05 | 1.51108e-09 | 0.005224 | 0.00531168 |
| v0_to_v1 | parent_input_embeddings | 128 | -2.45601e-06 | 0.000247496 | 1.32888e-07 | 0.005224 | 0.0473767 |
| v0_to_v1 | parent_input_stack | 128 | 0.000214682 | 0.000850237 | 1.84609e-06 | 0.005224 | 0.162756 |
| v0_to_v1 | parent_transformer_blocks | 128 | 0.000894769 | 0.00437312 | 3.50608e-05 | 0.005224 | 0.837122 |
| v1_to_v2 | current_exact | 128 | 0 | 0 | 0 | 0.00164039 | 0 |
| v1_to_v2 | parent_all | 128 | 0.00028939 | 0.00164039 | 5.79362e-06 | 0.00164039 | 1 |
| v1_to_v2 | parent_item_embedding | 128 | 1.87815e-05 | 3.22809e-05 | 2.0818e-09 | 0.00164039 | 0.0196788 |
| v1_to_v2 | parent_input_embeddings | 128 | 8.13675e-05 | 0.000205702 | 7.58361e-08 | 0.00164039 | 0.125398 |
| v1_to_v2 | parent_input_stack | 128 | -4.3881e-06 | 0.000371693 | 2.85016e-07 | 0.00164039 | 0.226588 |
| v1_to_v2 | parent_transformer_blocks | 128 | 0.0003002 | 0.00142467 | 4.21166e-06 | 0.00164039 | 0.868495 |
| v2_to_v3 | current_exact | 128 | 0 | 0 | 0 | 0.00131009 | 0 |
| v2_to_v3 | parent_all | 128 | 0.000220293 | 0.00131009 | 3.97855e-06 | 0.00131009 | 1 |
| v2_to_v3 | parent_item_embedding | 128 | 1.51833e-05 | 3.25772e-05 | 2.10074e-09 | 0.00131009 | 0.0248664 |
| v2_to_v3 | parent_input_embeddings | 128 | 7.69867e-05 | 0.00025848 | 1.17682e-07 | 0.00131009 | 0.1973 |
| v2_to_v3 | parent_input_stack | 128 | 1.76136e-05 | 0.000255572 | 1.42289e-07 | 0.00131009 | 0.19508 |
| v2_to_v3 | parent_transformer_blocks | 128 | 0.000194565 | 0.00107419 | 2.6828e-06 | 0.00131009 | 0.819938 |
| v3_to_v4 | current_exact | 128 | 0 | 0 | 0 | 0.00498576 | 0 |
| v3_to_v4 | parent_all | 128 | -0.000452194 | 0.00498576 | 4.80277e-05 | 0.00498576 | 1 |
| v3_to_v4 | parent_item_embedding | 128 | 3.37723e-05 | 3.70172e-05 | 2.66638e-09 | 0.00498576 | 0.00742459 |
| v3_to_v4 | parent_input_embeddings | 128 | -1.96241e-05 | 0.000942428 | 1.50364e-06 | 0.00498576 | 0.189024 |
| v3_to_v4 | parent_input_stack | 128 | -0.000153807 | 0.00193024 | 6.2533e-06 | 0.00498576 | 0.387151 |
| v3_to_v4 | parent_transformer_blocks | 128 | -0.000383008 | 0.00322136 | 2.18568e-05 | 0.00498576 | 0.646111 |
| v4_to_v5 | current_exact | 128 | 0 | 0 | 0 | 0.000985241 | 0 |
| v4_to_v5 | parent_all | 128 | -0.000247931 | 0.000985241 | 2.39962e-06 | 0.000985241 | 1 |
| v4_to_v5 | parent_item_embedding | 128 | 4.21766e-05 | 3.94803e-05 | 3.00556e-09 | 0.000985241 | 0.0400718 |
| v4_to_v5 | parent_input_embeddings | 128 | 0.000130739 | 0.00027744 | 1.30814e-07 | 0.000985241 | 0.281596 |
| v4_to_v5 | parent_input_stack | 128 | -0.000185681 | 0.000411117 | 3.40623e-07 | 0.000985241 | 0.417276 |
| v4_to_v5 | parent_transformer_blocks | 128 | -8.3527e-05 | 0.00091205 | 1.78507e-06 | 0.000985241 | 0.925713 |

`parent_item_embedding` changes only the item embedding used to materialize history. `parent_input_embeddings` also changes behavior/time encoders; `parent_input_stack` additionally changes the input projection. `parent_transformer_blocks` keeps Current inputs but uses Parent HSTU blocks for cache production.

Parent item embeddings alone reproduce only 0.5%-4.0% of the Parent-all probability gap. The full Parent input stack reaches 16.3%-41.7%, while Parent Transformer blocks reach 64.6%-92.6%. The dominant origin is contextual block co-adaptation, not isolated item embedding drift. Hybrid ratios are separate interventions and are not additive parameter attributions.
