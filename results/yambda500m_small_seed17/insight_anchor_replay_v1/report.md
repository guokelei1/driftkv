# Pre-cutover tail replay bridge

`natural_current_append` adds genuinely new post-cutover events. `precutover_tail_replay` re-encodes the same number of already-known pre-cutover tail events and therefore adds no new behavior information. `eviction_without_append` removes the matched number of old tokens.

| edge | mode | target_count | users | mean_actual_count | mean_reuse_harm | mean_abs_probability_shift |
| --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | eviction_without_append | 0 | 32 | 0 | -0.000606214 | 0.00457374 |
| v0_to_v1 | eviction_without_append | 32 | 32 | 29.9688 | -0.000515993 | 0.00456922 |
| v0_to_v1 | eviction_without_append | 64 | 32 | 62.8125 | -0.000272757 | 0.00459099 |
| v0_to_v1 | eviction_without_append | 128 | 32 | 127.438 | -0.000101963 | 0.00453247 |
| v0_to_v1 | natural_current_append | 0 | 32 | 0 | -0.000606214 | 0.00457374 |
| v0_to_v1 | natural_current_append | 32 | 32 | 29.9688 | -0.000668526 | 0.00409569 |
| v0_to_v1 | natural_current_append | 64 | 32 | 62.8125 | -0.00204083 | 0.0037428 |
| v0_to_v1 | natural_current_append | 128 | 32 | 127.438 | 0.000328045 | 0.00346166 |
| v0_to_v1 | precutover_tail_replay | 0 | 32 | 0 | -0.000606214 | 0.00457374 |
| v0_to_v1 | precutover_tail_replay | 32 | 32 | 29.9688 | -0.000510843 | 0.0043623 |
| v0_to_v1 | precutover_tail_replay | 64 | 32 | 62.8125 | -0.000415935 | 0.00407658 |
| v0_to_v1 | precutover_tail_replay | 128 | 32 | 127.438 | -0.000294911 | 0.00353701 |
| v1_to_v2 | eviction_without_append | 0 | 32 | 0 | -0.00230834 | 0.00189734 |
| v1_to_v2 | eviction_without_append | 32 | 32 | 29.2812 | -0.00223935 | 0.00190435 |
| v1_to_v2 | eviction_without_append | 64 | 32 | 60.9688 | -0.00210322 | 0.00190882 |
| v1_to_v2 | eviction_without_append | 128 | 32 | 127.406 | -0.00192457 | 0.00186965 |
| v1_to_v2 | natural_current_append | 0 | 32 | 0 | -0.00230834 | 0.00189734 |
| v1_to_v2 | natural_current_append | 32 | 32 | 29.2812 | -0.00149494 | 0.00170839 |
| v1_to_v2 | natural_current_append | 64 | 32 | 60.9688 | -0.00134882 | 0.00150662 |
| v1_to_v2 | natural_current_append | 128 | 32 | 127.406 | -0.00146502 | 0.00146649 |
| v1_to_v2 | precutover_tail_replay | 0 | 32 | 0 | -0.00230834 | 0.00189734 |
| v1_to_v2 | precutover_tail_replay | 32 | 32 | 29.2812 | -0.00220973 | 0.00180727 |
| v1_to_v2 | precutover_tail_replay | 64 | 32 | 60.9688 | -0.00207436 | 0.00170501 |
| v1_to_v2 | precutover_tail_replay | 128 | 32 | 127.406 | -0.0019227 | 0.0015015 |
| v2_to_v3 | eviction_without_append | 0 | 32 | 0 | 0.000839879 | 0.00126732 |
| v2_to_v3 | eviction_without_append | 32 | 32 | 31.2188 | 0.000849361 | 0.00124302 |
| v2_to_v3 | eviction_without_append | 64 | 32 | 63.5312 | 0.000789582 | 0.0012821 |
| v2_to_v3 | eviction_without_append | 128 | 32 | 127.188 | 0.000779375 | 0.00128538 |
| v2_to_v3 | natural_current_append | 0 | 32 | 0 | 0.000839879 | 0.00126732 |
| v2_to_v3 | natural_current_append | 32 | 32 | 31.2188 | 0.000386359 | 0.00136461 |
| v2_to_v3 | natural_current_append | 64 | 32 | 63.5312 | -0.00054981 | 0.000970898 |
| v2_to_v3 | natural_current_append | 128 | 32 | 127.188 | 0.000642164 | 0.00134535 |
| v2_to_v3 | precutover_tail_replay | 0 | 32 | 0 | 0.000839879 | 0.00126732 |
| v2_to_v3 | precutover_tail_replay | 32 | 32 | 31.2188 | 0.000779593 | 0.00120574 |
| v2_to_v3 | precutover_tail_replay | 64 | 32 | 63.5312 | 0.000791137 | 0.00114665 |
| v2_to_v3 | precutover_tail_replay | 128 | 32 | 127.188 | 0.00078953 | 0.00100575 |
| v3_to_v4 | eviction_without_append | 0 | 32 | 0 | 0.00106032 | 0.00549748 |
| v3_to_v4 | eviction_without_append | 32 | 32 | 30.9375 | 0.00127408 | 0.0055224 |
| v3_to_v4 | eviction_without_append | 64 | 32 | 62.25 | 0.00157166 | 0.00553754 |
| v3_to_v4 | eviction_without_append | 128 | 32 | 125.812 | 0.00153886 | 0.00537722 |
| v3_to_v4 | natural_current_append | 0 | 32 | 0 | 0.00106032 | 0.00549748 |
| v3_to_v4 | natural_current_append | 32 | 32 | 30.9375 | 0.000602981 | 0.00521199 |
| v3_to_v4 | natural_current_append | 64 | 32 | 62.25 | 0.00203622 | 0.00465936 |
| v3_to_v4 | natural_current_append | 128 | 32 | 125.812 | 0.00192633 | 0.00413941 |
| v3_to_v4 | precutover_tail_replay | 0 | 32 | 0 | 0.00106032 | 0.00549748 |
| v3_to_v4 | precutover_tail_replay | 32 | 32 | 30.9375 | 0.000927755 | 0.00519236 |
| v3_to_v4 | precutover_tail_replay | 64 | 32 | 62.25 | 0.00100866 | 0.00488378 |
| v3_to_v4 | precutover_tail_replay | 128 | 32 | 125.812 | 0.000923781 | 0.00425753 |
| v4_to_v5 | eviction_without_append | 0 | 32 | 0 | -0.000840533 | 0.00119704 |
| v4_to_v5 | eviction_without_append | 32 | 32 | 31.4375 | -0.000831979 | 0.00117331 |
| v4_to_v5 | eviction_without_append | 64 | 32 | 63.625 | -0.000712884 | 0.00118132 |
| v4_to_v5 | eviction_without_append | 128 | 32 | 126.688 | -0.00051466 | 0.00119415 |
| v4_to_v5 | natural_current_append | 0 | 32 | 0 | -0.000840533 | 0.00119704 |
| v4_to_v5 | natural_current_append | 32 | 32 | 31.4375 | 4.34881e-05 | 0.00110668 |
| v4_to_v5 | natural_current_append | 64 | 32 | 63.625 | 9.42255e-05 | 0.001274 |
| v4_to_v5 | natural_current_append | 128 | 32 | 126.688 | -0.000122663 | 0.000981083 |
| v4_to_v5 | precutover_tail_replay | 0 | 32 | 0 | -0.000840533 | 0.00119704 |
| v4_to_v5 | precutover_tail_replay | 32 | 32 | 31.4375 | -0.000895286 | 0.00114727 |
| v4_to_v5 | precutover_tail_replay | 64 | 32 | 63.625 | -0.000860286 | 0.001094 |
| v4_to_v5 | precutover_tail_replay | 128 | 32 | 126.688 | -0.000874584 | 0.000972227 |

## Gap reduction from zero to about 128 tokens

| edge | mode | gap_at_zero | gap_at_128 | gap_reduction_fraction |
| --- | --- | --- | --- | --- |
| v0_to_v1 | eviction_without_append | 0.00457374 | 0.00453247 | 0.00902364 |
| v0_to_v1 | natural_current_append | 0.00457374 | 0.00346166 | 0.243143 |
| v0_to_v1 | precutover_tail_replay | 0.00457374 | 0.00353701 | 0.226671 |
| v1_to_v2 | eviction_without_append | 0.00189734 | 0.00186965 | 0.0145907 |
| v1_to_v2 | natural_current_append | 0.00189734 | 0.00146649 | 0.227083 |
| v1_to_v2 | precutover_tail_replay | 0.00189734 | 0.0015015 | 0.208627 |
| v2_to_v3 | eviction_without_append | 0.00126732 | 0.00128538 | -0.01425 |
| v2_to_v3 | natural_current_append | 0.00126732 | 0.00134535 | -0.0615716 |
| v2_to_v3 | precutover_tail_replay | 0.00126732 | 0.00100575 | 0.206396 |
| v3_to_v4 | eviction_without_append | 0.00549748 | 0.00537722 | 0.0218754 |
| v3_to_v4 | natural_current_append | 0.00549748 | 0.00413941 | 0.247036 |
| v3_to_v4 | precutover_tail_replay | 0.00549748 | 0.00425753 | 0.225549 |
| v4_to_v5 | eviction_without_append | 0.00119704 | 0.00119415 | 0.00241721 |
| v4_to_v5 | natural_current_append | 0.00119704 | 0.000981083 | 0.180412 |
| v4_to_v5 | precutover_tail_replay | 0.00119704 | 0.000972227 | 0.187811 |
