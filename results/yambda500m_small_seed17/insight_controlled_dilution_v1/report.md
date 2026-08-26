# Controlled dilution

`anchor_fixed_384_old` adds real Current events without evicting the fixed 384 old tokens. `eviction_without_append` removes old tokens without adding Current events. `real_rolling` couples both as serving does. Query, candidate, label, and timestamp are fixed within each user/target comparison.

| edge | mode | target_append_count | users | mean_actual_count | mean_reuse_harm | mean_abs_probability_shift |
| --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | anchor_fixed_384_old | 0 | 32 | 0 | -1.00992e-05 | 0.00456226 |
| v0_to_v1 | anchor_fixed_384_old | 32 | 32 | 29.9688 | -0.000155184 | 0.00405956 |
| v0_to_v1 | anchor_fixed_384_old | 64 | 32 | 62.8125 | -0.00172847 | 0.00370124 |
| v0_to_v1 | anchor_fixed_384_old | 128 | 32 | 127.438 | 0.000468707 | 0.00347753 |
| v0_to_v1 | eviction_without_append | 0 | 32 | 0 | -0.000606214 | 0.00457374 |
| v0_to_v1 | eviction_without_append | 32 | 32 | 29.9688 | -0.000515993 | 0.00456922 |
| v0_to_v1 | eviction_without_append | 64 | 32 | 62.8125 | -0.000272757 | 0.00459099 |
| v0_to_v1 | eviction_without_append | 128 | 32 | 127.438 | -0.000101963 | 0.00453247 |
| v0_to_v1 | real_rolling | 0 | 32 | 0 | -0.000606214 | 0.00457374 |
| v0_to_v1 | real_rolling | 32 | 32 | 29.9688 | -0.000668526 | 0.00409569 |
| v0_to_v1 | real_rolling | 64 | 32 | 62.8125 | -0.00204083 | 0.0037428 |
| v0_to_v1 | real_rolling | 128 | 32 | 127.438 | 0.000328045 | 0.00346166 |
| v1_to_v2 | anchor_fixed_384_old | 0 | 32 | 0 | -0.0013368 | 0.00170252 |
| v1_to_v2 | anchor_fixed_384_old | 32 | 32 | 29.2812 | -0.000678238 | 0.00157278 |
| v1_to_v2 | anchor_fixed_384_old | 64 | 32 | 60.9688 | -0.00066049 | 0.00119345 |
| v1_to_v2 | anchor_fixed_384_old | 128 | 32 | 127.406 | -0.000785224 | 0.00134848 |
| v1_to_v2 | eviction_without_append | 0 | 32 | 0 | -0.00230834 | 0.00189734 |
| v1_to_v2 | eviction_without_append | 32 | 32 | 29.2812 | -0.00223935 | 0.00190435 |
| v1_to_v2 | eviction_without_append | 64 | 32 | 60.9688 | -0.00210322 | 0.00190882 |
| v1_to_v2 | eviction_without_append | 128 | 32 | 127.406 | -0.00192457 | 0.00186965 |
| v1_to_v2 | real_rolling | 0 | 32 | 0 | -0.00230834 | 0.00189734 |
| v1_to_v2 | real_rolling | 32 | 32 | 29.2812 | -0.00149494 | 0.00170839 |
| v1_to_v2 | real_rolling | 64 | 32 | 60.9688 | -0.00134882 | 0.00150662 |
| v1_to_v2 | real_rolling | 128 | 32 | 127.406 | -0.00146502 | 0.00146649 |
| v2_to_v3 | anchor_fixed_384_old | 0 | 32 | 0 | 0.000896216 | 0.00134228 |
| v2_to_v3 | anchor_fixed_384_old | 32 | 32 | 31.2188 | 0.000362975 | 0.00145685 |
| v2_to_v3 | anchor_fixed_384_old | 64 | 32 | 63.5312 | -0.000533886 | 0.00100167 |
| v2_to_v3 | anchor_fixed_384_old | 128 | 32 | 127.188 | 0.000729331 | 0.00135097 |
| v2_to_v3 | eviction_without_append | 0 | 32 | 0 | 0.000839879 | 0.00126732 |
| v2_to_v3 | eviction_without_append | 32 | 32 | 31.2188 | 0.000849361 | 0.00124302 |
| v2_to_v3 | eviction_without_append | 64 | 32 | 63.5312 | 0.000789582 | 0.0012821 |
| v2_to_v3 | eviction_without_append | 128 | 32 | 127.188 | 0.000779375 | 0.00128538 |
| v2_to_v3 | real_rolling | 0 | 32 | 0 | 0.000839879 | 0.00126732 |
| v2_to_v3 | real_rolling | 32 | 32 | 31.2188 | 0.000386359 | 0.00136461 |
| v2_to_v3 | real_rolling | 64 | 32 | 63.5312 | -0.00054981 | 0.000970898 |
| v2_to_v3 | real_rolling | 128 | 32 | 127.188 | 0.000642164 | 0.00134535 |
| v3_to_v4 | anchor_fixed_384_old | 0 | 32 | 0 | 0.00152203 | 0.00523143 |
| v3_to_v4 | anchor_fixed_384_old | 32 | 32 | 30.9375 | 0.00105591 | 0.00496501 |
| v3_to_v4 | anchor_fixed_384_old | 64 | 32 | 62.25 | 0.00192958 | 0.00431713 |
| v3_to_v4 | anchor_fixed_384_old | 128 | 32 | 125.812 | 0.00190069 | 0.00403557 |
| v3_to_v4 | eviction_without_append | 0 | 32 | 0 | 0.00106032 | 0.00549748 |
| v3_to_v4 | eviction_without_append | 32 | 32 | 30.9375 | 0.00127408 | 0.0055224 |
| v3_to_v4 | eviction_without_append | 64 | 32 | 62.25 | 0.00157166 | 0.00553754 |
| v3_to_v4 | eviction_without_append | 128 | 32 | 125.812 | 0.00153886 | 0.00537722 |
| v3_to_v4 | real_rolling | 0 | 32 | 0 | 0.00106032 | 0.00549748 |
| v3_to_v4 | real_rolling | 32 | 32 | 30.9375 | 0.000602981 | 0.00521199 |
| v3_to_v4 | real_rolling | 64 | 32 | 62.25 | 0.00203622 | 0.00465936 |
| v3_to_v4 | real_rolling | 128 | 32 | 125.812 | 0.00192633 | 0.00413941 |
| v4_to_v5 | anchor_fixed_384_old | 0 | 32 | 0 | 0.000160781 | 0.00109227 |
| v4_to_v5 | anchor_fixed_384_old | 32 | 32 | 31.4375 | 0.000909148 | 0.000992434 |
| v4_to_v5 | anchor_fixed_384_old | 64 | 32 | 63.625 | 0.000764348 | 0.00116879 |
| v4_to_v5 | anchor_fixed_384_old | 128 | 32 | 126.688 | 0.000275701 | 0.000905707 |
| v4_to_v5 | eviction_without_append | 0 | 32 | 0 | -0.000840533 | 0.00119704 |
| v4_to_v5 | eviction_without_append | 32 | 32 | 31.4375 | -0.000831979 | 0.00117331 |
| v4_to_v5 | eviction_without_append | 64 | 32 | 63.625 | -0.000712884 | 0.00118132 |
| v4_to_v5 | eviction_without_append | 128 | 32 | 126.688 | -0.00051466 | 0.00119415 |
| v4_to_v5 | real_rolling | 0 | 32 | 0 | -0.000840533 | 0.00119704 |
| v4_to_v5 | real_rolling | 32 | 32 | 31.4375 | 4.34881e-05 | 0.00110668 |
| v4_to_v5 | real_rolling | 64 | 32 | 63.625 | 9.42255e-05 | 0.001274 |
| v4_to_v5 | real_rolling | 128 | 32 | 126.688 | -0.000122663 | 0.000981083 |

## Gap reduction from 0 to about 128 events

| edge | mode | gap_at_zero | gap_at_128 | gap_reduction_fraction |
| --- | --- | --- | --- | --- |
| v0_to_v1 | anchor_fixed_384_old | 0.00456226 | 0.00347753 | 0.237761 |
| v0_to_v1 | eviction_without_append | 0.00457374 | 0.00453247 | 0.00902364 |
| v0_to_v1 | real_rolling | 0.00457374 | 0.00346166 | 0.243143 |
| v1_to_v2 | anchor_fixed_384_old | 0.00170252 | 0.00134848 | 0.207949 |
| v1_to_v2 | eviction_without_append | 0.00189734 | 0.00186965 | 0.0145907 |
| v1_to_v2 | real_rolling | 0.00189734 | 0.00146649 | 0.227083 |
| v2_to_v3 | anchor_fixed_384_old | 0.00134228 | 0.00135097 | -0.00647166 |
| v2_to_v3 | eviction_without_append | 0.00126732 | 0.00128538 | -0.01425 |
| v2_to_v3 | real_rolling | 0.00126732 | 0.00134535 | -0.0615716 |
| v3_to_v4 | anchor_fixed_384_old | 0.00523143 | 0.00403557 | 0.228592 |
| v3_to_v4 | eviction_without_append | 0.00549748 | 0.00537722 | 0.0218754 |
| v3_to_v4 | real_rolling | 0.00549748 | 0.00413941 | 0.247036 |
| v4_to_v5 | anchor_fixed_384_old | 0.00109227 | 0.000905707 | 0.1708 |
| v4_to_v5 | eviction_without_append | 0.00119704 | 0.00119415 | 0.00241721 |
| v4_to_v5 | real_rolling | 0.00119704 | 0.000981083 | 0.180412 |

Pure eviction leaves the output gap nearly unchanged. Fixed-old-state anchoring reduces the gap on four of five edges, and real rolling closely follows that anchor path. v2_to_v3 is the explicit counterexample.
