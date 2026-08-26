# K/V mechanism probe

Diagnostic exact splices only; they are not executable migration actions.

| edge | path | requests | path_minus_exact_log_loss | mean_abs_probability_shift | mean_bernoulli_js |
| --- | --- | --- | --- | --- | --- |
| v0_to_v1 | reuse_parent_kv | 128 | 0.000967405 | 0.005224 | 5.32911e-05 |
| v0_to_v1 | current_exact_kv | 128 | 0 | 0 | 0 |
| v0_to_v1 | current_k_parent_v | 128 | 0.000481386 | 0.00292958 | 1.5458e-05 |
| v0_to_v1 | parent_k_current_v | 128 | 0.000567828 | 0.00231672 | 1.15873e-05 |
| v0_to_v1 | current_lower_1 | 128 | 0.00121586 | 0.00361402 | 2.26742e-05 |
| v0_to_v1 | current_lower_2 | 128 | 0.000392242 | 0.00100802 | 2.45773e-06 |
| v0_to_v1 | current_lower_3 | 128 | 2.47208e-05 | 0.000208942 | 8.57647e-08 |
| v0_to_v1 | stale_old384 | 128 | 0.000718486 | 0.00387575 | 2.96754e-05 |
| v0_to_v1 | stale_recent128 | 128 | 0.000419097 | 0.00129577 | 3.59813e-06 |
| v0_to_v1 | stale_old480 | 128 | 0.000902585 | 0.00488197 | 4.66097e-05 |
| v0_to_v1 | stale_recent32 | 128 | 0.000122796 | 0.000332163 | 2.50025e-07 |
| v0_to_v1 | refresh_old384 | 128 | 0.000419097 | 0.00129577 | 3.59813e-06 |
| v0_to_v1 | refresh_recent128 | 128 | 0.000718486 | 0.00387575 | 2.96754e-05 |
| v0_to_v1 | refresh_old480 | 128 | 0.000122796 | 0.000332163 | 2.50025e-07 |
| v0_to_v1 | refresh_recent32 | 128 | 0.000902585 | 0.00488197 | 4.66097e-05 |
| v0_to_v1 | parent_exact_model | 128 | 0.00683184 | 0.0253023 | 0.000886842 |
| v0_to_v1 | only_old384 | 128 | 0.00777257 | 0.0210494 | 0.00109168 |
| v0_to_v1 | only_recent128 | 128 | 0.0041997 | 0.0117141 | 0.000402417 |
| v0_to_v1 | only_old480 | 128 | -0.00824038 | 0.0197984 | 0.00112732 |
| v0_to_v1 | only_recent32 | 128 | 0.00773746 | 0.0169049 | 0.000747588 |
| v1_to_v2 | reuse_parent_kv | 128 | 0.00028939 | 0.00164039 | 5.79362e-06 |
| v1_to_v2 | current_exact_kv | 128 | 0 | 0 | 0 |
| v1_to_v2 | current_k_parent_v | 128 | 4.71438e-05 | 0.000804787 | 1.26796e-06 |
| v1_to_v2 | parent_k_current_v | 128 | 0.000243804 | 0.0011022 | 2.56293e-06 |
| v1_to_v2 | current_lower_1 | 128 | 0.000600593 | 0.0014962 | 4.39459e-06 |
| v1_to_v2 | current_lower_2 | 128 | 0.000296154 | 0.000807532 | 1.61107e-06 |
| v1_to_v2 | current_lower_3 | 128 | -7.28117e-05 | 0.000224244 | 1.82933e-07 |
| v1_to_v2 | stale_old384 | 128 | 0.000285327 | 0.00119668 | 3.16538e-06 |
| v1_to_v2 | stale_recent128 | 128 | -1.0707e-05 | 0.000474586 | 5.26699e-07 |
| v1_to_v2 | stale_old480 | 128 | 0.000257639 | 0.00151932 | 4.98781e-06 |
| v1_to_v2 | stale_recent32 | 128 | 2.95031e-05 | 0.000138049 | 4.71751e-08 |
| v1_to_v2 | refresh_old384 | 128 | -1.0707e-05 | 0.000474586 | 5.26699e-07 |
| v1_to_v2 | refresh_recent128 | 128 | 0.000285327 | 0.00119668 | 3.16538e-06 |
| v1_to_v2 | refresh_old480 | 128 | 2.95031e-05 | 0.000138049 | 4.71751e-08 |
| v1_to_v2 | refresh_recent32 | 128 | 0.000257639 | 0.00151932 | 4.98781e-06 |
| v1_to_v2 | parent_exact_model | 128 | -0.000412774 | 0.00242534 | 1.15092e-05 |
| v1_to_v2 | only_old384 | 128 | 0.0104028 | 0.0261258 | 0.00138776 |
| v1_to_v2 | only_recent128 | 128 | -0.00289492 | 0.0107918 | 0.000321531 |
| v1_to_v2 | only_old480 | 128 | 0.00609317 | 0.0231075 | 0.00114639 |
| v1_to_v2 | only_recent32 | 128 | 0.00440338 | 0.0171739 | 0.000781911 |
| v2_to_v3 | reuse_parent_kv | 128 | 0.000220293 | 0.00131009 | 3.97855e-06 |
| v2_to_v3 | current_exact_kv | 128 | 0 | 0 | 0 |
| v2_to_v3 | current_k_parent_v | 128 | 0.000128448 | 0.000687157 | 1.10548e-06 |
| v2_to_v3 | parent_k_current_v | 128 | 7.55835e-05 | 0.000657896 | 9.81108e-07 |
| v2_to_v3 | current_lower_1 | 128 | 5.87841e-05 | 0.000729761 | 1.51664e-06 |
| v2_to_v3 | current_lower_2 | 128 | -7.55851e-05 | 0.000637553 | 1.08122e-06 |
| v2_to_v3 | current_lower_3 | 128 | -2.31297e-05 | 9.5071e-05 | 1.51594e-08 |
| v2_to_v3 | stale_old384 | 128 | 5.60945e-05 | 0.00101904 | 2.36227e-06 |
| v2_to_v3 | stale_recent128 | 128 | 0.000160964 | 0.000338191 | 2.73682e-07 |
| v2_to_v3 | stale_old480 | 128 | 0.000185496 | 0.00123757 | 3.55254e-06 |
| v2_to_v3 | stale_recent32 | 128 | 3.43084e-05 | 9.95463e-05 | 2.078e-08 |
| v2_to_v3 | refresh_old384 | 128 | 0.000160964 | 0.000338191 | 2.73682e-07 |
| v2_to_v3 | refresh_recent128 | 128 | 5.60945e-05 | 0.00101904 | 2.36227e-06 |
| v2_to_v3 | refresh_old480 | 128 | 3.43084e-05 | 9.95463e-05 | 2.078e-08 |
| v2_to_v3 | refresh_recent32 | 128 | 0.000185496 | 0.00123757 | 3.55254e-06 |
| v2_to_v3 | parent_exact_model | 128 | 0.00167998 | 0.00503735 | 5.50818e-05 |
| v2_to_v3 | only_old384 | 128 | 0.00174201 | 0.0171852 | 0.000635654 |
| v2_to_v3 | only_recent128 | 128 | 3.55382e-05 | 0.00978308 | 0.000251469 |
| v2_to_v3 | only_old480 | 128 | -0.00246357 | 0.0163563 | 0.00066402 |
| v2_to_v3 | only_recent32 | 128 | 0.00399787 | 0.0167645 | 0.000766688 |
| v3_to_v4 | reuse_parent_kv | 128 | -0.000452194 | 0.00498576 | 4.80277e-05 |
| v3_to_v4 | current_exact_kv | 128 | 0 | 0 | 0 |
| v3_to_v4 | current_k_parent_v | 128 | -0.000292217 | 0.00250679 | 1.17325e-05 |
| v3_to_v4 | parent_k_current_v | 128 | -0.000275901 | 0.00241084 | 1.19394e-05 |
| v3_to_v4 | current_lower_1 | 128 | -0.000420527 | 0.00392576 | 2.89478e-05 |
| v3_to_v4 | current_lower_2 | 128 | 3.10241e-06 | 0.00206123 | 1.01632e-05 |
| v3_to_v4 | current_lower_3 | 128 | -9.92621e-06 | 0.000338189 | 2.63005e-07 |
| v3_to_v4 | stale_old384 | 128 | -0.000294634 | 0.00369032 | 2.66004e-05 |
| v3_to_v4 | stale_recent128 | 128 | -0.000269299 | 0.00131806 | 3.66293e-06 |
| v3_to_v4 | stale_old480 | 128 | -0.000385354 | 0.00466385 | 4.20691e-05 |
| v3_to_v4 | stale_recent32 | 128 | -0.000104148 | 0.000331671 | 2.50479e-07 |
| v3_to_v4 | refresh_old384 | 128 | -0.000269299 | 0.00131806 | 3.66293e-06 |
| v3_to_v4 | refresh_recent128 | 128 | -0.000294634 | 0.00369032 | 2.66004e-05 |
| v3_to_v4 | refresh_old480 | 128 | -0.000104148 | 0.000331671 | 2.50479e-07 |
| v3_to_v4 | refresh_recent32 | 128 | -0.000385354 | 0.00466385 | 4.20691e-05 |
| v3_to_v4 | parent_exact_model | 128 | -0.00125935 | 0.00989306 | 0.000181457 |
| v3_to_v4 | only_old384 | 128 | 0.00331469 | 0.0227318 | 0.00114956 |
| v3_to_v4 | only_recent128 | 128 | 0.00136782 | 0.011652 | 0.00042051 |
| v3_to_v4 | only_old480 | 128 | -0.00559553 | 0.0207194 | 0.000965054 |
| v3_to_v4 | only_recent32 | 128 | 0.0038341 | 0.0181847 | 0.000861751 |
| v4_to_v5 | reuse_parent_kv | 128 | -0.000247931 | 0.000985241 | 2.39962e-06 |
| v4_to_v5 | current_exact_kv | 128 | 0 | 0 | 0 |
| v4_to_v5 | current_k_parent_v | 128 | -0.000192872 | 0.000618525 | 9.97486e-07 |
| v4_to_v5 | parent_k_current_v | 128 | -7.17034e-05 | 0.000497492 | 5.34071e-07 |
| v4_to_v5 | current_lower_1 | 128 | -0.000153626 | 0.00107706 | 2.52508e-06 |
| v4_to_v5 | current_lower_2 | 128 | -0.00024797 | 0.00055015 | 8.79615e-07 |
| v4_to_v5 | current_lower_3 | 128 | 2.16394e-05 | 0.000163244 | 8.78114e-08 |
| v4_to_v5 | stale_old384 | 128 | -0.000112467 | 0.000760361 | 1.42596e-06 |
| v4_to_v5 | stale_recent128 | 128 | -0.000159789 | 0.000306426 | 2.56915e-07 |
| v4_to_v5 | stale_old480 | 128 | -0.00020417 | 0.000928538 | 2.16391e-06 |
| v4_to_v5 | stale_recent32 | 128 | -4.92605e-05 | 8.09584e-05 | 1.62997e-08 |
| v4_to_v5 | refresh_old384 | 128 | -0.000159789 | 0.000306426 | 2.56915e-07 |
| v4_to_v5 | refresh_recent128 | 128 | -0.000112467 | 0.000760361 | 1.42596e-06 |
| v4_to_v5 | refresh_old480 | 128 | -4.92605e-05 | 8.09584e-05 | 1.62997e-08 |
| v4_to_v5 | refresh_recent32 | 128 | -0.00020417 | 0.000928538 | 2.16391e-06 |
| v4_to_v5 | parent_exact_model | 128 | -0.000229776 | 0.0038839 | 2.59078e-05 |
| v4_to_v5 | only_old384 | 128 | -0.0214583 | 0.0233123 | 0.00128913 |
| v4_to_v5 | only_recent128 | 128 | -0.00416487 | 0.011199 | 0.00031582 |
| v4_to_v5 | only_old480 | 128 | -0.0118875 | 0.018951 | 0.000845741 |
| v4_to_v5 | only_recent32 | 128 | -3.81015e-05 | 0.0186544 | 0.000915525 |

Replacing either all K or all V with Current state reduces the output gap on every edge, but neither side alone dominates consistently. The mechanism is coupled K/V incompatibility, not a clean key-only or value-only failure.

Replacing the lower three of four layers with Current K/V recovers 83.4% to 96.0% of the absolute-probability gap. Replacing only layer 0 is insufficient and can worsen the gap. This supports early/middle dependency propagation; the splice remains diagnostic, not an action.

## Matched Parent/Current/Reuse overlap

| edge | requests | mean_matched_release_benefit | mean_reuse_harm | spearman_G_H | release_winner_fraction | positive_harm_on_release_winners_fraction | positive_harm_concentration_lift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | 128 | 0.00683184 | 0.000967405 | 0.785221 | 0.914062 | 0.963919 | 1.05454 |
| v1_to_v2 | 128 | -0.000412774 | 0.00028939 | 0.771001 | 0.484375 | 0.823129 | 1.69936 |
| v2_to_v3 | 128 | 0.00167998 | 0.000220293 | 0.823292 | 0.4375 | 0.959871 | 2.19399 |
| v3_to_v4 | 128 | -0.00125935 | -0.000452194 | 0.889106 | 0.820312 | 0.980254 | 1.19498 |
| v4_to_v5 | 128 | -0.000229776 | -0.000247931 | 0.488956 | 0.109375 | 0.499328 | 4.56528 |

This append-free probe uses Parent Exact under the Parent model, Current Exact under the Current model, and Parent K/V Reuse under the Current model on the same requests. It is the matched small-scale companion to the full-request descriptive overlap.

## Regional Utility x Staleness

| edge | region | requests | mean_utility | mean_regional_staleness | mean_refresh_recovery | positive_utility_fraction | spearman_utility_vs_staleness | positive_staleness_on_useful_fraction | positive_staleness_concentration_lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | old384 | 128 | 0.0041997 | 0.000718486 | 0.000548308 | 0.515625 | 0.194699 | 0.560154 | 1.08636 |
| v0_to_v1 | recent128 | 128 | 0.00777257 | 0.000419097 | 0.000248919 | 0.546875 | -0.178733 | 0.473517 | 0.865859 |
| v0_to_v1 | old480 | 128 | 0.00773746 | 0.000902585 | 0.00084461 | 0.53125 | 0.105315 | 0.542819 | 1.02178 |
| v0_to_v1 | recent32 | 128 | -0.00824038 | 0.000122796 | 6.482e-05 | 0.484375 | 0.0986198 | 0.479976 | 0.990918 |
| v1_to_v2 | old384 | 128 | -0.00289492 | 0.000285327 | 0.000300097 | 0.523438 | 0.0345232 | 0.524407 | 1.00185 |
| v1_to_v2 | recent128 | 128 | 0.0104028 | -1.0707e-05 | 4.06224e-06 | 0.5625 | 0.000818303 | 0.516053 | 0.917427 |
| v1_to_v2 | old480 | 128 | 0.00440338 | 0.000257639 | 0.000259886 | 0.609375 | 0.14239 | 0.624435 | 1.02471 |
| v1_to_v2 | recent32 | 128 | 0.00609317 | 2.95031e-05 | 3.17504e-05 | 0.46875 | 0.098099 | 0.491628 | 1.04881 |
| v2_to_v3 | old384 | 128 | 3.55382e-05 | 5.60945e-05 | 5.9329e-05 | 0.492188 | 0.0592783 | 0.277643 | 0.5641 |
| v2_to_v3 | recent128 | 128 | 0.00174201 | 0.000160964 | 0.000164199 | 0.40625 | -0.353055 | 0.419068 | 1.03155 |
| v2_to_v3 | old480 | 128 | 0.00399787 | 0.000185496 | 0.000185985 | 0.523438 | -0.0151014 | 0.22895 | 0.437397 |
| v2_to_v3 | recent32 | 128 | -0.00246357 | 3.43084e-05 | 3.47969e-05 | 0.414062 | -0.249479 | 0.327243 | 0.790323 |
| v3_to_v4 | old384 | 128 | 0.00136782 | -0.000294634 | -0.000182895 | 0.59375 | 0.164267 | 0.628826 | 1.05908 |
| v3_to_v4 | recent128 | 128 | 0.00331469 | -0.000269299 | -0.00015756 | 0.5 | 0.147518 | 0.550504 | 1.10101 |
| v3_to_v4 | old480 | 128 | 0.0038341 | -0.000385354 | -0.000348045 | 0.53125 | 0.176874 | 0.59714 | 1.12403 |
| v3_to_v4 | recent32 | 128 | -0.00559553 | -0.000104148 | -6.68397e-05 | 0.5 | 0.278618 | 0.596342 | 1.19268 |
| v4_to_v5 | old384 | 128 | -0.00416487 | -0.000112467 | -8.81424e-05 | 0.523438 | -0.157314 | 0.315596 | 0.602929 |
| v4_to_v5 | recent128 | 128 | -0.0214583 | -0.000159789 | -0.000135464 | 0.429688 | 0.094019 | 0.330407 | 0.768947 |
| v4_to_v5 | old480 | 128 | -3.81015e-05 | -0.00020417 | -0.000198671 | 0.578125 | 0.116296 | 0.326462 | 0.564691 |
| v4_to_v5 | recent32 | 128 | -0.0118875 | -4.92605e-05 | -4.3761e-05 | 0.429688 | 0.0271184 | 0.493692 | 1.14896 |

Utility removes and exactly recomputes the complementary history. Regional staleness replaces the same region with Parent K/V, and recovery starts from Parent K/V and refreshes that region. This aligns utility, staleness, and recovery on the same fixed region.
