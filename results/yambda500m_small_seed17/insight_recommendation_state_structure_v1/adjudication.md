# Recommendation-state structure: adjudication

Formal scope: 3,000 fixed users (30% of Small), all five v0→v5 adjacent edges, 512 pre-cutover events and 64 label-free candidate probes per user-edge. No label or negative semantics entered the observation.

## Adjudicated insight

**Cross-version HSTU state mismatch is primarily a candidate-broadcast user-evidence compatibility field, not a collection of independent per-candidate token-retrieval failures.**

Every one of the 60,000 candidate influence matrices has rank-1@90%. The Exact−Reuse influence delta has rank-1@90% in 59,999/60,000 user-edge-layer cases, and the final readout delta has rank-1@90% in all 15,000 user-edge cases. Across edge/layer means, the first candidate-shared direction carries 99.9681%–99.9992% of normalized influence energy.

| edge | layer | candidate_rank90_one_fraction | candidate_top_direction_energy_mean | mismatch_influence_rank90_one_fraction | mismatch_readout_rank90_one_fraction | mismatch_readout_effective_rank_mean |
| --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | 0 | 1 | 0.999805 | 1 | 1 | 1.00821 |
| v0_to_v1 | 1 | 1 | 0.999986 | 1 | 1 | 1.00821 |
| v0_to_v1 | 2 | 1 | 0.999992 | 1 | 1 | 1.00821 |
| v0_to_v1 | 3 | 1 | 0.999978 | 1 | 1 | 1.00821 |
| v1_to_v2 | 0 | 1 | 0.999764 | 1 | 1 | 1.00851 |
| v1_to_v2 | 1 | 1 | 0.999987 | 1 | 1 | 1.00851 |
| v1_to_v2 | 2 | 1 | 0.999986 | 1 | 1 | 1.00851 |
| v1_to_v2 | 3 | 1 | 0.999969 | 1 | 1 | 1.00851 |
| v2_to_v3 | 0 | 1 | 0.999752 | 1 | 1 | 1.01976 |
| v2_to_v3 | 1 | 1 | 0.999989 | 1 | 1 | 1.01976 |
| v2_to_v3 | 2 | 1 | 0.999991 | 1 | 1 | 1.01976 |
| v2_to_v3 | 3 | 1 | 0.999972 | 1 | 1 | 1.01976 |
| v3_to_v4 | 0 | 1 | 0.99977 | 1 | 1 | 1.01023 |
| v3_to_v4 | 1 | 1 | 0.999984 | 1 | 1 | 1.01023 |
| v3_to_v4 | 2 | 1 | 0.999984 | 0.999667 | 1 | 1.01023 |
| v3_to_v4 | 3 | 1 | 0.999977 | 1 | 1 | 1.01023 |
| v4_to_v5 | 0 | 1 | 0.999681 | 1 | 1 | 1.01003 |
| v4_to_v5 | 1 | 1 | 0.999988 | 1 | 1 | 1.01003 |
| v4_to_v5 | 2 | 1 | 0.999986 | 1 | 1 | 1.01003 |
| v4_to_v5 | 3 | 1 | 0.999971 | 1 | 1 | 1.01003 |

## Where the shared field comes from

Held-out-user centroids show a strong typed entity coordinate at the input and layer-0 K, followed by rapid contextualization in the attention/gated update. Item identity adds a real but modest component beyond the global version shift; item-action typing explains much more early-layer delta. The update stages retain far less item-specific predictability.

| edge | stage | held_out_user_item_action_samples | item_centroid_R2 | item_excess_R2_over_global | item_action_R2 | item_action_increment_over_item |
| --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | combined_input | 105369 | 0.681521 | 0.121111 | 0.933524 | 0.791272 |
| v0_to_v1 | layer0.k | 105369 | 0.618303 | 0.106935 | 0.897945 | 0.732628 |
| v0_to_v1 | layer0.update | 105369 | 0.309673 | 0.046953 | 0.536056 | 0.327936 |
| v0_to_v1 | layer1.update | 105369 | 0.0663339 | 0.00951395 | 0.18517 | 0.127279 |
| v0_to_v1 | layer2.update | 105369 | 0.0834259 | 0.0030157 | 0.105165 | 0.0237174 |
| v0_to_v1 | layer3.update | 105369 | 0.194382 | 0.0286297 | 0.322727 | 0.159313 |
| v1_to_v2 | combined_input | 101345 | 0.714856 | 0.120578 | 0.958494 | 0.854437 |
| v1_to_v2 | layer0.k | 101345 | 0.724399 | 0.0958173 | 0.880817 | 0.567551 |
| v1_to_v2 | layer0.update | 101345 | 0.280872 | 0.0355303 | 0.478642 | 0.275014 |
| v1_to_v2 | layer1.update | 101345 | 0.134788 | 0.0163074 | 0.220027 | 0.0985183 |
| v1_to_v2 | layer2.update | 101345 | 0.167665 | 0.00841176 | 0.266302 | 0.118506 |
| v1_to_v2 | layer3.update | 101345 | 0.137586 | 0.0215575 | 0.249242 | 0.12947 |
| v2_to_v3 | combined_input | 96005 | 0.594583 | 0.105655 | 0.922978 | 0.810018 |
| v2_to_v3 | layer0.k | 96005 | 0.557011 | 0.0987908 | 0.925873 | 0.832666 |
| v2_to_v3 | layer0.update | 96005 | 0.208167 | 0.011511 | 0.279148 | 0.0896412 |
| v2_to_v3 | layer1.update | 96005 | 0.100729 | 0.00298148 | 0.127884 | 0.0301966 |
| v2_to_v3 | layer2.update | 96005 | 0.0898777 | 0.0214843 | 0.275013 | 0.203419 |
| v2_to_v3 | layer3.update | 96005 | 0.254941 | 0.0229101 | 0.360894 | 0.142208 |
| v3_to_v4 | combined_input | 91822 | 0.666937 | 0.0850334 | 0.845962 | 0.537512 |
| v3_to_v4 | layer0.k | 91822 | 0.619154 | 0.0951386 | 0.866549 | 0.649593 |
| v3_to_v4 | layer0.update | 91822 | 0.213269 | 0.0228614 | 0.321117 | 0.137083 |
| v3_to_v4 | layer1.update | 91822 | 0.0888379 | 0.0162183 | 0.282815 | 0.21289 |
| v3_to_v4 | layer2.update | 91822 | 0.167231 | 0.0133945 | 0.299942 | 0.159362 |
| v3_to_v4 | layer3.update | 91822 | 0.170366 | 0.0237524 | 0.308767 | 0.166821 |
| v4_to_v5 | combined_input | 86616 | 0.601705 | 0.116445 | 0.935073 | 0.836987 |
| v4_to_v5 | layer0.k | 86616 | 0.585215 | 0.10724 | 0.924801 | 0.818705 |
| v4_to_v5 | layer0.update | 86616 | 0.209778 | 0.0240308 | 0.354879 | 0.183621 |
| v4_to_v5 | layer1.update | 86616 | 0.0151822 | -0.00363389 | 0.0319221 | 0.016998 |
| v4_to_v5 | layer2.update | 86616 | 0.217493 | 0.00348899 | 0.246782 | 0.0374292 |
| v4_to_v5 | layer3.update | 86616 | 0.138701 | 0.0176032 | 0.242023 | 0.119961 |

This supports `shared typed coordinate + contextual user residual`; it does not support persisting raw item embeddings as the complete interface.

## Semantic coreset boundary

Same-item-first pairing raises the matched-item pair fraction from 3.29%–3.79% to 29.55%–30.13%, but same-item and typed pairing each beat positional pairing on mean probability gap in only 3/5 edges. Per-user win fractions remain 42.2%–50.0%. Raw identity/action equality is therefore not a stable substitutability test for contextual HSTU evidence.

| edge | path | mean_abs_probability_gap | aggregate_gap_recovery_over_reuse | gap_minus_positional_pairs | top10_overlap | rank_correlation |
| --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | positional_pairs | 0.00360687 | 0.253619 | 0 | 0.9789 | 0.990766 |
| v0_to_v1 | same_item_pairs | 0.00353901 | 0.267661 | -6.78574e-05 | 0.9788 | 0.990735 |
| v0_to_v1 | typed_pairs | 0.00354291 | 0.266854 | -6.39592e-05 | 0.9789 | 0.990777 |
| v1_to_v2 | positional_pairs | 0.00132409 | 0.22557 | 0 | 0.9873 | 0.996009 |
| v1_to_v2 | same_item_pairs | 0.00131041 | 0.23357 | -1.36775e-05 | 0.987067 | 0.996037 |
| v1_to_v2 | typed_pairs | 0.00131007 | 0.233766 | -1.40129e-05 | 0.987433 | 0.996021 |
| v2_to_v3 | positional_pairs | 0.00117534 | 0.181948 | 0 | 0.986733 | 0.996982 |
| v2_to_v3 | same_item_pairs | 0.00124117 | 0.136123 | 6.58379e-05 | 0.9871 | 0.996925 |
| v2_to_v3 | typed_pairs | 0.00122573 | 0.14687 | 5.03977e-05 | 0.987067 | 0.996958 |
| v3_to_v4 | positional_pairs | 0.00353641 | 0.248263 | 0 | 0.988833 | 0.997365 |
| v3_to_v4 | same_item_pairs | 0.00346253 | 0.263967 | -7.3876e-05 | 0.988867 | 0.99746 |
| v3_to_v4 | typed_pairs | 0.00347093 | 0.262182 | -6.548e-05 | 0.989133 | 0.997465 |
| v4_to_v5 | positional_pairs | 0.00100821 | 0.113789 | 0 | 0.993733 | 0.999079 |
| v4_to_v5 | same_item_pairs | 0.00104215 | 0.0839515 | 3.39445e-05 | 0.993667 | 0.998904 |
| v4_to_v5 | typed_pairs | 0.00104498 | 0.081471 | 3.67665e-05 | 0.9939 | 0.99906 |

## Design implication and boundary

The next mechanism should first repair or rematerialize a small Current-version **user evidence basis** shared across the candidate bank, then carry a smaller contextual residual. This is more recommendation-specific than query-by-query top-k token repair because the same persistent user state is amortized across many candidates.

Do not add candidate-specific Route or raw same-item/action GROUP from this result. The existing CAST + compact PATCH path remains a strong baseline. A new basis/anchor mechanism still needs a prospective task-quality experiment and runtime qualification. The observation is Small seed17 only and uses a controlled candidate bank, not an exposed-candidate quality result.
