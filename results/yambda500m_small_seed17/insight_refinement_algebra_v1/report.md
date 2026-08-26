# Typed-state refinement algebra probe

All paths are diagnostic interventions. Output-gap recovery is not rolling recommendation-quality recovery.

## CAST/PATCH decomposition

| edge | cast_recovery | patch_recovery | cast_then_patch_recovery | increment_over_best | parent_residual_on_cast_recovery | cast_shadowed_scope_max_kv_error | base_conditioned_patch_reconstruction_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | 0.560424 | 0.249956 | 0.681861 | 0.121438 | 0.795766 | 0 | 5.96046e-08 |
| v1_to_v2 | 0.488792 | 0.258461 | 0.626251 | 0.13746 | 0.681699 | 0 | 2.98023e-08 |
| v2_to_v3 | 0.640947 | 0.231547 | 0.726267 | 0.0853203 | 0.743569 | 0 | 5.96046e-08 |
| v3_to_v4 | 0.242936 | 0.223496 | 0.429433 | 0.186498 | 0.460224 | 0 | 5.96046e-08 |
| v4_to_v5 | 0.215583 | 0.193504 | 0.38722 | 0.171637 | 0.376768 | 0 | 2.98023e-08 |

## GROUP/PATCH order

| edge | carriers | group_then_patch_scaled_recovery | patch_then_group_scaled_recovery | patch_then_group_minus_group_then_patch |
| --- | --- | --- | --- | --- |
| v0_to_v1 | 8 | 0.203 | 0.197852 | -0.00514768 |
| v0_to_v1 | 16 | 0.215719 | 0.211407 | -0.00431118 |
| v0_to_v1 | 32 | 0.243964 | 0.240421 | -0.0035429 |
| v0_to_v1 | 64 | 0.25008 | 0.24791 | -0.00217005 |
| v0_to_v1 | 128 | 0.249956 | 0.249956 | 0 |
| v1_to_v2 | 8 | -0.253815 | -0.257538 | -0.00372378 |
| v1_to_v2 | 16 | 0.0774486 | 0.0671836 | -0.010265 |
| v1_to_v2 | 32 | 0.165494 | 0.158064 | -0.00742952 |
| v1_to_v2 | 64 | 0.230278 | 0.225019 | -0.0052592 |
| v1_to_v2 | 128 | 0.258461 | 0.258461 | 0 |
| v2_to_v3 | 8 | -0.394186 | -0.387455 | 0.006731 |
| v2_to_v3 | 16 | -0.0613333 | -0.0526138 | 0.00871952 |
| v2_to_v3 | 32 | 0.0934268 | 0.0963585 | 0.00293174 |
| v2_to_v3 | 64 | 0.176961 | 0.178994 | 0.00203295 |
| v2_to_v3 | 128 | 0.231547 | 0.231547 | 0 |
| v3_to_v4 | 8 | 0.106308 | 0.101396 | -0.00491244 |
| v3_to_v4 | 16 | 0.183659 | 0.179891 | -0.00376759 |
| v3_to_v4 | 32 | 0.209786 | 0.206932 | -0.0028533 |
| v3_to_v4 | 64 | 0.220502 | 0.218558 | -0.00194422 |
| v3_to_v4 | 128 | 0.223496 | 0.223496 | 0 |
| v4_to_v5 | 8 | -0.790647 | -0.784265 | 0.00638224 |
| v4_to_v5 | 16 | -0.171793 | -0.170497 | 0.00129608 |
| v4_to_v5 | 32 | -0.01175 | 0.00690906 | 0.018659 |
| v4_to_v5 | 64 | 0.0945859 | 0.107808 | 0.0132218 |
| v4_to_v5 | 128 | 0.193504 | 0.193504 | 0 |

## SCALE ablation

| edge | order | carriers | unscaled_recovery | scaled_recovery | scale_increment |
| --- | --- | --- | --- | --- | --- |
| v0_to_v1 | group_then_patch | 8 | -0.0364785 | 0.203 | 0.239478 |
| v0_to_v1 | patch_then_group | 8 | -0.036194 | 0.197852 | 0.234046 |
| v0_to_v1 | group_then_patch | 16 | -0.00137885 | 0.215719 | 0.217097 |
| v0_to_v1 | patch_then_group | 16 | -0.000947443 | 0.211407 | 0.212355 |
| v0_to_v1 | group_then_patch | 32 | 0.0654598 | 0.243964 | 0.178504 |
| v0_to_v1 | patch_then_group | 32 | 0.0657664 | 0.240421 | 0.174655 |
| v0_to_v1 | group_then_patch | 64 | 0.157091 | 0.25008 | 0.0929893 |
| v0_to_v1 | patch_then_group | 64 | 0.15641 | 0.24791 | 0.0915006 |
| v0_to_v1 | group_then_patch | 128 | 0.249956 | 0.249956 | 0 |
| v0_to_v1 | patch_then_group | 128 | 0.249956 | 0.249956 | 0 |
| v1_to_v2 | group_then_patch | 8 | -0.997477 | -0.253815 | 0.743662 |
| v1_to_v2 | patch_then_group | 8 | -0.994143 | -0.257538 | 0.736604 |
| v1_to_v2 | group_then_patch | 16 | -0.865681 | 0.0774486 | 0.94313 |
| v1_to_v2 | patch_then_group | 16 | -0.860088 | 0.0671836 | 0.927272 |
| v1_to_v2 | group_then_patch | 32 | -0.629171 | 0.165494 | 0.794665 |
| v1_to_v2 | patch_then_group | 32 | -0.620592 | 0.158064 | 0.778656 |
| v1_to_v2 | group_then_patch | 64 | -0.209306 | 0.230278 | 0.439584 |
| v1_to_v2 | patch_then_group | 64 | -0.199748 | 0.225019 | 0.424767 |
| v1_to_v2 | group_then_patch | 128 | 0.258461 | 0.258461 | 0 |
| v1_to_v2 | patch_then_group | 128 | 0.258461 | 0.258461 | 0 |
| v2_to_v3 | group_then_patch | 8 | -1.44074 | -0.394186 | 1.04656 |
| v2_to_v3 | patch_then_group | 8 | -1.43632 | -0.387455 | 1.04887 |
| v2_to_v3 | group_then_patch | 16 | -1.27677 | -0.0613333 | 1.21544 |
| v2_to_v3 | patch_then_group | 16 | -1.26896 | -0.0526138 | 1.21634 |
| v2_to_v3 | group_then_patch | 32 | -0.948621 | 0.0934268 | 1.04205 |
| v2_to_v3 | patch_then_group | 32 | -0.936162 | 0.0963585 | 1.03252 |
| v2_to_v3 | group_then_patch | 64 | -0.378721 | 0.176961 | 0.555682 |
| v2_to_v3 | patch_then_group | 64 | -0.365189 | 0.178994 | 0.544183 |
| v2_to_v3 | group_then_patch | 128 | 0.231547 | 0.231547 | 0 |
| v2_to_v3 | patch_then_group | 128 | 0.231547 | 0.231547 | 0 |
| v3_to_v4 | group_then_patch | 8 | -0.0481205 | 0.106308 | 0.154429 |
| v3_to_v4 | patch_then_group | 8 | -0.0477315 | 0.101396 | 0.149127 |
| v3_to_v4 | group_then_patch | 16 | -0.0132254 | 0.183659 | 0.196884 |
| v3_to_v4 | patch_then_group | 16 | -0.0126888 | 0.179891 | 0.19258 |
| v3_to_v4 | group_then_patch | 32 | 0.0436712 | 0.209786 | 0.166115 |
| v3_to_v4 | patch_then_group | 32 | 0.0436618 | 0.206932 | 0.163271 |
| v3_to_v4 | group_then_patch | 64 | 0.128762 | 0.220502 | 0.09174 |
| v3_to_v4 | patch_then_group | 64 | 0.128488 | 0.218558 | 0.09007 |
| v3_to_v4 | group_then_patch | 128 | 0.223496 | 0.223496 | 0 |
| v3_to_v4 | patch_then_group | 128 | 0.223496 | 0.223496 | 0 |
| v4_to_v5 | group_then_patch | 8 | -2.31705 | -0.790647 | 1.5264 |
| v4_to_v5 | patch_then_group | 8 | -2.31202 | -0.784265 | 1.52775 |
| v4_to_v5 | group_then_patch | 16 | -2.05299 | -0.171793 | 1.8812 |
| v4_to_v5 | patch_then_group | 16 | -2.04429 | -0.170497 | 1.87379 |
| v4_to_v5 | group_then_patch | 32 | -1.61342 | -0.01175 | 1.60167 |
| v4_to_v5 | patch_then_group | 32 | -1.60013 | 0.00690906 | 1.60704 |
| v4_to_v5 | group_then_patch | 64 | -0.857987 | 0.0945859 | 0.952573 |
| v4_to_v5 | patch_then_group | 64 | -0.84263 | 0.107808 | 0.950438 |
| v4_to_v5 | group_then_patch | 128 | 0.193504 | 0.193504 | 0 |
| v4_to_v5 | patch_then_group | 128 | 0.193504 | 0.193504 | 0 |

## Carrier-density frontier

| edge | order | carriers | carrier_density | represented_mass | recovery | increment_from_previous_density |
| --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | group_patch_scale | 8 | 0.0625 | 16 | 0.203 | nan |
| v0_to_v1 | group_patch_scale | 16 | 0.125 | 8 | 0.215719 | 0.0127187 |
| v0_to_v1 | group_patch_scale | 32 | 0.25 | 4 | 0.243964 | 0.0282454 |
| v0_to_v1 | group_patch_scale | 64 | 0.5 | 2 | 0.25008 | 0.00611636 |
| v0_to_v1 | group_patch_scale | 128 | 1 | 1 | 0.249956 | -0.000124328 |
| v0_to_v1 | patch_group_scale | 8 | 0.0625 | 16 | 0.197852 | nan |
| v0_to_v1 | patch_group_scale | 16 | 0.125 | 8 | 0.211407 | 0.0135552 |
| v0_to_v1 | patch_group_scale | 32 | 0.25 | 4 | 0.240421 | 0.0290137 |
| v0_to_v1 | patch_group_scale | 64 | 0.5 | 2 | 0.24791 | 0.00748922 |
| v0_to_v1 | patch_group_scale | 128 | 1 | 1 | 0.249956 | 0.00204572 |
| v1_to_v2 | group_patch_scale | 8 | 0.0625 | 16 | -0.253815 | nan |
| v1_to_v2 | group_patch_scale | 16 | 0.125 | 8 | 0.0774486 | 0.331263 |
| v1_to_v2 | group_patch_scale | 32 | 0.25 | 4 | 0.165494 | 0.0880451 |
| v1_to_v2 | group_patch_scale | 64 | 0.5 | 2 | 0.230278 | 0.0647843 |
| v1_to_v2 | group_patch_scale | 128 | 1 | 1 | 0.258461 | 0.0281835 |
| v1_to_v2 | patch_group_scale | 8 | 0.0625 | 16 | -0.257538 | nan |
| v1_to_v2 | patch_group_scale | 16 | 0.125 | 8 | 0.0671836 | 0.324722 |
| v1_to_v2 | patch_group_scale | 32 | 0.25 | 4 | 0.158064 | 0.0908805 |
| v1_to_v2 | patch_group_scale | 64 | 0.5 | 2 | 0.225019 | 0.0669546 |
| v1_to_v2 | patch_group_scale | 128 | 1 | 1 | 0.258461 | 0.0334427 |
| v2_to_v3 | group_patch_scale | 8 | 0.0625 | 16 | -0.394186 | nan |
| v2_to_v3 | group_patch_scale | 16 | 0.125 | 8 | -0.0613333 | 0.332853 |
| v2_to_v3 | group_patch_scale | 32 | 0.25 | 4 | 0.0934268 | 0.15476 |
| v2_to_v3 | group_patch_scale | 64 | 0.5 | 2 | 0.176961 | 0.083534 |
| v2_to_v3 | group_patch_scale | 128 | 1 | 1 | 0.231547 | 0.0545863 |
| v2_to_v3 | patch_group_scale | 8 | 0.0625 | 16 | -0.387455 | nan |
| v2_to_v3 | patch_group_scale | 16 | 0.125 | 8 | -0.0526138 | 0.334842 |
| v2_to_v3 | patch_group_scale | 32 | 0.25 | 4 | 0.0963585 | 0.148972 |
| v2_to_v3 | patch_group_scale | 64 | 0.5 | 2 | 0.178994 | 0.0826353 |
| v2_to_v3 | patch_group_scale | 128 | 1 | 1 | 0.231547 | 0.0525534 |
| v3_to_v4 | group_patch_scale | 8 | 0.0625 | 16 | 0.106308 | nan |
| v3_to_v4 | group_patch_scale | 16 | 0.125 | 8 | 0.183659 | 0.0773506 |
| v3_to_v4 | group_patch_scale | 32 | 0.25 | 4 | 0.209786 | 0.0261269 |
| v3_to_v4 | group_patch_scale | 64 | 0.5 | 2 | 0.220502 | 0.0107161 |
| v3_to_v4 | group_patch_scale | 128 | 1 | 1 | 0.223496 | 0.00299376 |
| v3_to_v4 | patch_group_scale | 8 | 0.0625 | 16 | 0.101396 | nan |
| v3_to_v4 | patch_group_scale | 16 | 0.125 | 8 | 0.179891 | 0.0784954 |
| v3_to_v4 | patch_group_scale | 32 | 0.25 | 4 | 0.206932 | 0.0270412 |
| v3_to_v4 | patch_group_scale | 64 | 0.5 | 2 | 0.218558 | 0.0116251 |
| v3_to_v4 | patch_group_scale | 128 | 1 | 1 | 0.223496 | 0.00493799 |
| v4_to_v5 | group_patch_scale | 8 | 0.0625 | 16 | -0.790647 | nan |
| v4_to_v5 | group_patch_scale | 16 | 0.125 | 8 | -0.171793 | 0.618854 |
| v4_to_v5 | group_patch_scale | 32 | 0.25 | 4 | -0.01175 | 0.160043 |
| v4_to_v5 | group_patch_scale | 64 | 0.5 | 2 | 0.0945859 | 0.106336 |
| v4_to_v5 | group_patch_scale | 128 | 1 | 1 | 0.193504 | 0.0989181 |
| v4_to_v5 | patch_group_scale | 8 | 0.0625 | 16 | -0.784265 | nan |
| v4_to_v5 | patch_group_scale | 16 | 0.125 | 8 | -0.170497 | 0.613768 |
| v4_to_v5 | patch_group_scale | 32 | 0.25 | 4 | 0.00690906 | 0.177406 |
| v4_to_v5 | patch_group_scale | 64 | 0.5 | 2 | 0.107808 | 0.100899 |
| v4_to_v5 | patch_group_scale | 128 | 1 | 1 | 0.193504 | 0.0856962 |

## Structural cost shape

| plan | raw_tokens | recomputed_token_layers | state_token_layers_read | state_token_layers_written | persistent_positions |
| --- | --- | --- | --- | --- | --- |
| CAST(all) | 0 | 0 | 2048 | 2048 | 512 |
| PATCH_exact(tail128) | 128 | 512 | 1536 | 512 | 512 |
| CAST(prefix)+PATCH_exact(tail128) | 128 | 512 | 2048 | 2048 | 512 |
| GROUP(128->8)->PATCH->SCALE | 8 | 32 | 1536 | 32 | 392 |
| PATCH_dense->GROUP(128->8)->SCALE | 128 | 512 | 1536 | 32 | 392 |
| GROUP(128->16)->PATCH->SCALE | 16 | 64 | 1536 | 64 | 400 |
| PATCH_dense->GROUP(128->16)->SCALE | 128 | 512 | 1536 | 64 | 400 |
| GROUP(128->32)->PATCH->SCALE | 32 | 128 | 1536 | 128 | 416 |
| PATCH_dense->GROUP(128->32)->SCALE | 128 | 512 | 1536 | 128 | 416 |
| GROUP(128->64)->PATCH->SCALE | 64 | 256 | 1536 | 256 | 448 |
| PATCH_dense->GROUP(128->64)->SCALE | 128 | 512 | 1536 | 256 | 448 |
| GROUP(128->128)->PATCH->SCALE | 128 | 512 | 1536 | 512 | 512 |
| PATCH_dense->GROUP(128->128)->SCALE | 128 | 512 | 1536 | 512 | 512 |

The report intentionally does not freeze a residual threshold, carrier-density contract, or budget compiler. Those require held-out rolling quality and a target-free residual estimator.
