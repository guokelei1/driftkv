# Pre-cutover tail replay bridge

`natural_current_append` adds genuinely new post-cutover events. `precutover_tail_replay` re-encodes the same number of already-known pre-cutover tail events and therefore adds no new behavior information. `eviction_without_append` removes the matched number of old tokens.

| edge | mode | target_count | users | mean_actual_count | mean_reuse_harm | mean_abs_probability_shift |
| --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | eviction_without_append | 0 | 64 | 0 | 0.0021833 | 0.00506608 |
| v0_to_v1 | eviction_without_append | 32 | 64 | 30.6719 | 0.00223946 | 0.00507101 |
| v0_to_v1 | eviction_without_append | 64 | 64 | 63.2969 | 0.00233866 | 0.00504377 |
| v0_to_v1 | eviction_without_append | 128 | 64 | 127.547 | 0.00237299 | 0.00495565 |
| v0_to_v1 | natural_current_append | 0 | 64 | 0 | 0.0021833 | 0.00506608 |
| v0_to_v1 | natural_current_append | 32 | 64 | 30.6719 | 0.00237989 | 0.0044786 |
| v0_to_v1 | natural_current_append | 64 | 64 | 63.2969 | 0.00102401 | 0.00426673 |
| v0_to_v1 | natural_current_append | 128 | 64 | 127.547 | 0.00147741 | 0.00360013 |
| v0_to_v1 | precutover_tail_replay | 0 | 64 | 0 | 0.0021833 | 0.00506608 |
| v0_to_v1 | precutover_tail_replay | 32 | 64 | 30.6719 | 0.00208676 | 0.00480071 |
| v0_to_v1 | precutover_tail_replay | 64 | 64 | 63.2969 | 0.00198124 | 0.00447714 |
| v0_to_v1 | precutover_tail_replay | 128 | 64 | 127.547 | 0.00176884 | 0.00391361 |
| v1_to_v2 | eviction_without_append | 0 | 64 | 0 | -0.00101754 | 0.00191454 |
| v1_to_v2 | eviction_without_append | 32 | 64 | 30.2344 | -0.00107127 | 0.00194036 |
| v1_to_v2 | eviction_without_append | 64 | 64 | 62.3125 | -0.0010965 | 0.00193115 |
| v1_to_v2 | eviction_without_append | 128 | 64 | 127.562 | -0.00105782 | 0.00191495 |
| v1_to_v2 | natural_current_append | 0 | 64 | 0 | -0.00101754 | 0.00191454 |
| v1_to_v2 | natural_current_append | 32 | 64 | 30.2344 | -0.000335109 | 0.00172152 |
| v1_to_v2 | natural_current_append | 64 | 64 | 62.3125 | -0.000664313 | 0.00157145 |
| v1_to_v2 | natural_current_append | 128 | 64 | 127.562 | -0.00053355 | 0.00137006 |
| v1_to_v2 | precutover_tail_replay | 0 | 64 | 0 | -0.00101754 | 0.00191454 |
| v1_to_v2 | precutover_tail_replay | 32 | 64 | 30.2344 | -0.000950943 | 0.00181659 |
| v1_to_v2 | precutover_tail_replay | 64 | 64 | 62.3125 | -0.000894522 | 0.00168945 |
| v1_to_v2 | precutover_tail_replay | 128 | 64 | 127.562 | -0.00083529 | 0.00146654 |
| v2_to_v3 | eviction_without_append | 0 | 64 | 0 | -6.77991e-06 | 0.00126784 |
| v2_to_v3 | eviction_without_append | 32 | 64 | 31.1562 | 2.60619e-05 | 0.00124536 |
| v2_to_v3 | eviction_without_append | 64 | 64 | 63.4219 | 5.70321e-06 | 0.00125458 |
| v2_to_v3 | eviction_without_append | 128 | 64 | 127.188 | -1.19332e-05 | 0.00124737 |
| v2_to_v3 | natural_current_append | 0 | 64 | 0 | -6.77991e-06 | 0.00126784 |
| v2_to_v3 | natural_current_append | 32 | 64 | 31.1562 | 0.000124019 | 0.0012845 |
| v2_to_v3 | natural_current_append | 64 | 64 | 63.4219 | -0.000403102 | 0.00108725 |
| v2_to_v3 | natural_current_append | 128 | 64 | 127.188 | 0.000190395 | 0.00106189 |
| v2_to_v3 | precutover_tail_replay | 0 | 64 | 0 | -6.77991e-06 | 0.00126784 |
| v2_to_v3 | precutover_tail_replay | 32 | 64 | 31.1562 | -3.79143e-05 | 0.00120395 |
| v2_to_v3 | precutover_tail_replay | 64 | 64 | 63.4219 | -1.51668e-05 | 0.00114386 |
| v2_to_v3 | precutover_tail_replay | 128 | 64 | 127.188 | 4.18137e-05 | 0.000987093 |
| v3_to_v4 | eviction_without_append | 0 | 64 | 0 | 0.00311753 | 0.00505694 |
| v3_to_v4 | eviction_without_append | 32 | 64 | 30.5 | 0.0032137 | 0.00505898 |
| v3_to_v4 | eviction_without_append | 64 | 64 | 62.375 | 0.00333659 | 0.00504225 |
| v3_to_v4 | eviction_without_append | 128 | 64 | 126.688 | 0.00328099 | 0.00492797 |
| v3_to_v4 | natural_current_append | 0 | 64 | 0 | 0.00311753 | 0.00505694 |
| v3_to_v4 | natural_current_append | 32 | 64 | 30.5 | 0.00288052 | 0.00492717 |
| v3_to_v4 | natural_current_append | 64 | 64 | 62.375 | 0.00346751 | 0.00451981 |
| v3_to_v4 | natural_current_append | 128 | 64 | 126.688 | 0.00306895 | 0.0040225 |
| v3_to_v4 | precutover_tail_replay | 0 | 64 | 0 | 0.00311753 | 0.00505694 |
| v3_to_v4 | precutover_tail_replay | 32 | 64 | 30.5 | 0.00290893 | 0.00477789 |
| v3_to_v4 | precutover_tail_replay | 64 | 64 | 62.375 | 0.00280141 | 0.004492 |
| v3_to_v4 | precutover_tail_replay | 128 | 64 | 126.688 | 0.00246613 | 0.00391804 |
| v4_to_v5 | eviction_without_append | 0 | 64 | 0 | -0.000749809 | 0.000968229 |
| v4_to_v5 | eviction_without_append | 32 | 64 | 31.4531 | -0.000775878 | 0.000969914 |
| v4_to_v5 | eviction_without_append | 64 | 64 | 63.4375 | -0.000671777 | 0.000963716 |
| v4_to_v5 | eviction_without_append | 128 | 64 | 126.766 | -0.000611231 | 0.000965536 |
| v4_to_v5 | natural_current_append | 0 | 64 | 0 | -0.000749809 | 0.000968229 |
| v4_to_v5 | natural_current_append | 32 | 64 | 31.4531 | -0.00032823 | 0.000871979 |
| v4_to_v5 | natural_current_append | 64 | 64 | 63.4375 | 0.000230825 | 0.00107733 |
| v4_to_v5 | natural_current_append | 128 | 64 | 126.766 | -0.000298747 | 0.000917841 |
| v4_to_v5 | precutover_tail_replay | 0 | 64 | 0 | -0.000749809 | 0.000968229 |
| v4_to_v5 | precutover_tail_replay | 32 | 64 | 31.4531 | -0.00080658 | 0.00094219 |
| v4_to_v5 | precutover_tail_replay | 64 | 64 | 63.4375 | -0.00078346 | 0.000897256 |
| v4_to_v5 | precutover_tail_replay | 128 | 64 | 126.766 | -0.000723035 | 0.000796242 |

## Gap reduction from zero to about 128 tokens

| edge | mode | gap_at_zero | gap_at_128 | gap_reduction_fraction |
| --- | --- | --- | --- | --- |
| v0_to_v1 | eviction_without_append | 0.00506608 | 0.00495565 | 0.0217979 |
| v0_to_v1 | natural_current_append | 0.00506608 | 0.00360013 | 0.289366 |
| v0_to_v1 | precutover_tail_replay | 0.00506608 | 0.00391361 | 0.227488 |
| v1_to_v2 | eviction_without_append | 0.00191454 | 0.00191495 | -0.000214408 |
| v1_to_v2 | natural_current_append | 0.00191454 | 0.00137006 | 0.284393 |
| v1_to_v2 | precutover_tail_replay | 0.00191454 | 0.00146654 | 0.234002 |
| v2_to_v3 | eviction_without_append | 0.00126784 | 0.00124737 | 0.0161499 |
| v2_to_v3 | natural_current_append | 0.00126784 | 0.00106189 | 0.162443 |
| v2_to_v3 | precutover_tail_replay | 0.00126784 | 0.000987093 | 0.221439 |
| v3_to_v4 | eviction_without_append | 0.00505694 | 0.00492797 | 0.0255024 |
| v3_to_v4 | natural_current_append | 0.00505694 | 0.0040225 | 0.204558 |
| v3_to_v4 | precutover_tail_replay | 0.00505694 | 0.00391804 | 0.225215 |
| v4_to_v5 | eviction_without_append | 0.000968229 | 0.000965536 | 0.00278137 |
| v4_to_v5 | natural_current_append | 0.000968229 | 0.000917841 | 0.0520412 |
| v4_to_v5 | precutover_tail_replay | 0.000968229 | 0.000796242 | 0.17763 |

At 64 users per edge, pre-cutover tail replay reduces the output gap by 17.8%-23.4% on all five edges without adding new behavior information. Matched pure eviction changes it by -0.02%-2.55%. This closes the representation-versus-new-information bridge for the Small seed17 probe.
