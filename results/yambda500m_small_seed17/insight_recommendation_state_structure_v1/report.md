# Recommendation-state structure across v0..v5

Scope: 3,000 fixed users (30.0% of the frozen Small population), five adjacent edges, 512 pre-cutover events per user, and 64 label-free candidate probes per user. No request label was read.

## Cross-user state-delta factorization

| edge | stage | qualified_items | held_out_user_item_action_samples | global_version_shift_R2 | item_centroid_R2 | item_excess_R2_over_global | item_action_R2 | item_action_increment_over_item |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | item_embedding | 1024 | 105369 | 0.134904 | 1 | 1 | 1 | -1.93292e-06 |
| v0_to_v1 | combined_input | 1024 | 105369 | 0.637634 | 0.681521 | 0.121111 | 0.933524 | 0.791272 |
| v0_to_v1 | layer0.k | 1024 | 105369 | 0.572599 | 0.618303 | 0.106935 | 0.897945 | 0.732628 |
| v0_to_v1 | layer1.k | 1024 | 105369 | 0.377704 | 0.416147 | 0.0617761 | 0.644536 | 0.391175 |
| v0_to_v1 | layer2.k | 1024 | 105369 | 0.314006 | 0.344574 | 0.0445609 | 0.497851 | 0.233858 |
| v0_to_v1 | layer3.k | 1024 | 105369 | 0.338824 | 0.36999 | 0.0471374 | 0.523137 | 0.243087 |
| v0_to_v1 | layer3.update | 1024 | 105369 | 0.170637 | 0.194382 | 0.0286297 | 0.322727 | 0.159313 |
| v1_to_v2 | item_embedding | 1024 | 101345 | 0.141381 | 1 | 1 | 1 | 1.67023e-06 |
| v1_to_v2 | combined_input | 1024 | 101345 | 0.675759 | 0.714856 | 0.120578 | 0.958494 | 0.854437 |
| v1_to_v2 | layer0.k | 1024 | 101345 | 0.695193 | 0.724399 | 0.0958173 | 0.880817 | 0.567551 |
| v1_to_v2 | layer1.k | 1024 | 101345 | 0.230227 | 0.261789 | 0.041001 | 0.468366 | 0.279835 |
| v1_to_v2 | layer2.k | 1024 | 101345 | 0.225574 | 0.261448 | 0.0463232 | 0.456466 | 0.264056 |
| v1_to_v2 | layer3.k | 1024 | 101345 | 0.313201 | 0.340603 | 0.0398986 | 0.465921 | 0.19005 |
| v1_to_v2 | layer3.update | 1024 | 101345 | 0.118584 | 0.137586 | 0.0215575 | 0.249242 | 0.12947 |
| v2_to_v3 | item_embedding | 1024 | 96005 | 0.141542 | 1 | 1 | 1 | 1.80317e-06 |
| v2_to_v3 | combined_input | 1024 | 96005 | 0.546689 | 0.594583 | 0.105655 | 0.922978 | 0.810018 |
| v2_to_v3 | layer0.k | 1024 | 96005 | 0.50845 | 0.557011 | 0.0987908 | 0.925873 | 0.832666 |
| v2_to_v3 | layer1.k | 1024 | 96005 | 0.301366 | 0.325498 | 0.0345411 | 0.479579 | 0.228437 |
| v2_to_v3 | layer2.k | 1024 | 96005 | 0.219445 | 0.243579 | 0.0309192 | 0.469599 | 0.298802 |
| v2_to_v3 | layer3.k | 1024 | 96005 | 0.251244 | 0.274306 | 0.0308002 | 0.410135 | 0.187171 |
| v2_to_v3 | layer3.update | 1024 | 96005 | 0.237471 | 0.254941 | 0.0229101 | 0.360894 | 0.142208 |
| v3_to_v4 | item_embedding | 1024 | 91822 | 0.114187 | 1 | 1 | 1 | -1.52427e-06 |
| v3_to_v4 | combined_input | 1024 | 91822 | 0.635984 | 0.666937 | 0.0850334 | 0.845962 | 0.537512 |
| v3_to_v4 | layer0.k | 1024 | 91822 | 0.579112 | 0.619154 | 0.0951386 | 0.866549 | 0.649593 |
| v3_to_v4 | layer1.k | 1024 | 91822 | 0.418421 | 0.439517 | 0.0362742 | 0.611084 | 0.306106 |
| v3_to_v4 | layer2.k | 1024 | 91822 | 0.460707 | 0.477917 | 0.0319119 | 0.638989 | 0.308519 |
| v3_to_v4 | layer3.k | 1024 | 91822 | 0.289286 | 0.311799 | 0.0316763 | 0.480507 | 0.245144 |
| v3_to_v4 | layer3.update | 1024 | 91822 | 0.150181 | 0.170366 | 0.0237524 | 0.308767 | 0.166821 |
| v4_to_v5 | item_embedding | 1024 | 86616 | 0.104334 | 1 | 1 | 1 | -9.61709e-07 |
| v4_to_v5 | combined_input | 1024 | 86616 | 0.549214 | 0.601705 | 0.116445 | 0.935073 | 0.836987 |
| v4_to_v5 | layer0.k | 1024 | 86616 | 0.53539 | 0.585215 | 0.10724 | 0.924801 | 0.818705 |
| v4_to_v5 | layer1.k | 1024 | 86616 | 0.356772 | 0.397552 | 0.0633982 | 0.626878 | 0.380658 |
| v4_to_v5 | layer2.k | 1024 | 86616 | 0.263803 | 0.295735 | 0.0433738 | 0.477191 | 0.257653 |
| v4_to_v5 | layer3.k | 1024 | 86616 | 0.278573 | 0.311868 | 0.0461512 | 0.473457 | 0.234822 |
| v4_to_v5 | layer3.update | 1024 | 86616 | 0.123268 | 0.138701 | 0.0176032 | 0.242023 | 0.119961 |

Centroids are fitted on one deterministic UID split and evaluated on disjoint users. Item and item-action columns therefore measure cross-user generalization, not within-sample reconstruction.

## Matched-budget semantic coreset

| edge | path | mean_abs_probability_gap_mean | output_gap_recovery_over_reuse_mean | top10_overlap_mean | rank_correlation_mean |
| --- | --- | --- | --- | --- | --- |
| v0_to_v1 | dense_current_tail128 | 0.00359887 | 0.248699 | 0.979067 | 0.990725 |
| v0_to_v1 | parent_reuse | 0.00483247 | 0 | 0.974667 | 0.987416 |
| v0_to_v1 | positional_pairs | 0.00360687 | 0.236127 | 0.9789 | 0.990766 |
| v0_to_v1 | same_item_pairs | 0.00353901 | 0.243245 | 0.9788 | 0.990735 |
| v0_to_v1 | typed_pairs | 0.00354291 | 0.243364 | 0.9789 | 0.990777 |
| v1_to_v2 | dense_current_tail128 | 0.0012714 | 0.202335 | 0.987467 | 0.996041 |
| v1_to_v2 | parent_reuse | 0.00170976 | 0 | 0.9834 | 0.994093 |
| v1_to_v2 | positional_pairs | 0.00132409 | -0.0177134 | 0.9873 | 0.996009 |
| v1_to_v2 | same_item_pairs | 0.00131041 | -0.141102 | 0.987067 | 0.996037 |
| v1_to_v2 | typed_pairs | 0.00131007 | -0.128463 | 0.987433 | 0.996021 |
| v2_to_v3 | dense_current_tail128 | 0.00109975 | 0.200894 | 0.986933 | 0.996896 |
| v2_to_v3 | parent_reuse | 0.00143675 | 0 | 0.983667 | 0.995114 |
| v2_to_v3 | positional_pairs | 0.00117534 | -0.144331 | 0.986733 | 0.996982 |
| v2_to_v3 | same_item_pairs | 0.00124117 | -0.242874 | 0.9871 | 0.996925 |
| v2_to_v3 | typed_pairs | 0.00122573 | -0.232168 | 0.987067 | 0.996958 |
| v3_to_v4 | dense_current_tail128 | 0.00351289 | 0.252417 | 0.989433 | 0.997459 |
| v3_to_v4 | parent_reuse | 0.00470432 | 0 | 0.985733 | 0.995694 |
| v3_to_v4 | positional_pairs | 0.00353641 | 0.209519 | 0.988833 | 0.997365 |
| v3_to_v4 | same_item_pairs | 0.00346253 | 0.238145 | 0.988867 | 0.99746 |
| v3_to_v4 | typed_pairs | 0.00347093 | 0.238024 | 0.989133 | 0.997465 |
| v4_to_v5 | dense_current_tail128 | 0.00089566 | 0.0849383 | 0.993933 | 0.999167 |
| v4_to_v5 | parent_reuse | 0.00113766 | 0 | 0.992367 | 0.998719 |
| v4_to_v5 | positional_pairs | 0.00100821 | -0.424641 | 0.993733 | 0.999079 |
| v4_to_v5 | same_item_pairs | 0.00104215 | -0.599902 | 0.993667 | 0.998904 |
| v4_to_v5 | typed_pairs | 0.00104498 | -0.571164 | 0.9939 | 0.99906 |

Every compact path retains the same Parent old-384 prefix, 64 Current carriers for recent-128 evidence, and represented mass two. Only the label-free pairing rule changes.

## Candidate-bank influence subspace

| edge | layer | candidate_influence_top_direction_fraction_mean | candidate_influence_effective_rank_mean | candidate_influence_rank90_median | exact_minus_reuse_influence_rank90_median | novel_to_prefix_effective_support_mean | recent_repeat_effective_support_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | 0 | 0.999805 | 1.00177 | 1 | 1 | 496.541 | 497.197 |
| v0_to_v1 | 1 | 0.999986 | 1.00018 | 1 | 1 | 381.52 | 381.36 |
| v0_to_v1 | 2 | 0.999992 | 1.0001 | 1 | 1 | 288.226 | 288.507 |
| v0_to_v1 | 3 | 0.999978 | 1.00025 | 1 | 1 | 201.814 | 201.997 |
| v1_to_v2 | 0 | 0.999764 | 1.0021 | 1 | 1 | 490.762 | 492.015 |
| v1_to_v2 | 1 | 0.999987 | 1.00016 | 1 | 1 | 369.247 | 369.252 |
| v1_to_v2 | 2 | 0.999986 | 1.00017 | 1 | 1 | 302.424 | 303.123 |
| v1_to_v2 | 3 | 0.999969 | 1.00034 | 1 | 1 | 240.848 | 241.294 |
| v2_to_v3 | 0 | 0.999752 | 1.0022 | 1 | 1 | 489.873 | 491.636 |
| v2_to_v3 | 1 | 0.999989 | 1.00014 | 1 | 1 | 362.782 | 362.758 |
| v2_to_v3 | 2 | 0.999991 | 1.00011 | 1 | 1 | 289.574 | 290.006 |
| v2_to_v3 | 3 | 0.999972 | 1.00031 | 1 | 1 | 222.445 | 222.76 |
| v3_to_v4 | 0 | 0.99977 | 1.00208 | 1 | 1 | 488.975 | 490.881 |
| v3_to_v4 | 1 | 0.999984 | 1.00019 | 1 | 1 | 354.054 | 354.25 |
| v3_to_v4 | 2 | 0.999984 | 1.00018 | 1 | 1 | 286.107 | 286.743 |
| v3_to_v4 | 3 | 0.999977 | 1.00027 | 1 | 1 | 214.04 | 214.56 |
| v4_to_v5 | 0 | 0.999681 | 1.00268 | 1 | 1 | 486.499 | 488.906 |
| v4_to_v5 | 1 | 0.999988 | 1.00015 | 1 | 1 | 340.052 | 340.437 |
| v4_to_v5 | 2 | 0.999986 | 1.00016 | 1 | 1 | 283.651 | 284.365 |
| v4_to_v5 | 3 | 0.999971 | 1.00032 | 1 | 1 | 211.586 | 211.834 |

The fixed 64-candidate panel takes up to 16 known recent repeats and 16 known old-only repeats, then fills with known novel-to-prefix items. This keeps high-OOV/low-repeat users in scope. Probes are not sampled negatives and are never joined to labels. Influence uses the norm of each position's pointwise-attention value contribution before candidate-wise normalization.

Elapsed wall time: 6.9 minutes.
