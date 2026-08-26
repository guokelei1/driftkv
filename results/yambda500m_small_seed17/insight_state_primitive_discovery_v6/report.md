# Versioned-state primitive discovery

All paths are discovery interventions, not frozen executable actions. Translation is derived only from Parent/Current projection parameters; no Current target-KV fitting or label-driven state selection is used.

## Primitive recovery

| edge | path | requests | mean_abs_probability_gap | output_gap_recovery | path_minus_exact_log_loss | signed_log_loss_recovery | ROC_AUC | mean_bernoulli_js |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v0_to_v1 | reuse_parent | 252 | 0.00528663 | 0 | -0.000547608 | 0 | 0.450658 | 5.493e-05 |
| v0_to_v1 | current_exact | 252 | 0 | 1 | 0 | 1 | 0.444993 | 0 |
| v0_to_v1 | translate_joint | 252 | 0.00232388 | 0.560424 | 2.75096e-05 | 1.05024 | 0.445724 | 1.09272e-05 |
| v0_to_v1 | reader_bridge | 252 | 0.00567139 | -0.0727787 | -0.000633157 | -0.156224 | 0.450841 | 6.30764e-05 |
| v0_to_v1 | rematerialize_tail128 | 252 | 0.00396521 | 0.249956 | -0.000339607 | 0.379835 | 0.448648 | 3.07371e-05 |
| v0_to_v1 | synthesize_tail8 | 252 | 0.00556304 | -0.0522844 | -0.000388437 | 0.290665 | 0.451206 | 6.71209e-05 |
| v0_to_v1 | synthesize_landmark8 | 252 | 0.00547948 | -0.0364786 | -0.000379923 | 0.306213 | 0.451389 | 6.56544e-05 |
| v0_to_v1 | synthesize_sparse_oracle8 | 252 | 0.00547798 | -0.036194 | -0.000379651 | 0.30671 | 0.451389 | 6.56407e-05 |
| v0_to_v1 | synthesize_tail8_mass16 | 252 | 0.0056484 | -0.0684294 | 1.73187e-06 | 1.00316 | 0.447368 | 6.05181e-05 |
| v0_to_v1 | synthesize_landmark8_mass16 | 252 | 0.00421345 | 0.203 | 2.30473e-05 | 1.04209 | 0.44883 | 3.89181e-05 |
| v0_to_v1 | synthesize_sparse_oracle8_mass16 | 252 | 0.00424066 | 0.197852 | 3.52655e-05 | 1.0644 | 0.44883 | 3.93436e-05 |
| v0_to_v1 | synthesize_landmark32 | 252 | 0.00494057 | 0.0654597 | -0.000441712 | 0.193379 | 0.451572 | 5.29934e-05 |
| v0_to_v1 | synthesize_landmark64 | 252 | 0.00445615 | 0.157091 | -0.000455464 | 0.168265 | 0.453216 | 4.13384e-05 |
| v0_to_v1 | synthesize_sparse_oracle32 | 252 | 0.00493895 | 0.0657664 | -0.000442326 | 0.192258 | 0.451389 | 5.2973e-05 |
| v0_to_v1 | synthesize_sparse_oracle64 | 252 | 0.00445975 | 0.15641 | -0.000455885 | 0.167497 | 0.453034 | 4.13576e-05 |
| v0_to_v1 | synthesize_landmark32_mass4 | 252 | 0.00399689 | 0.243964 | -0.000418931 | 0.23498 | 0.451937 | 3.23856e-05 |
| v0_to_v1 | synthesize_landmark64_mass2 | 252 | 0.00396455 | 0.25008 | -0.000426404 | 0.221334 | 0.451023 | 3.11185e-05 |
| v0_to_v1 | synthesize_sparse_oracle32_mass4 | 252 | 0.00401562 | 0.240421 | -0.000417028 | 0.238455 | 0.451937 | 3.26806e-05 |
| v0_to_v1 | synthesize_sparse_oracle64_mass2 | 252 | 0.00397602 | 0.24791 | -0.000425593 | 0.222814 | 0.450841 | 3.13086e-05 |
| v0_to_v1 | retire_old128 | 252 | 0.00661336 | -0.250958 | -0.000722923 | -0.320148 | 0.451754 | 9.35549e-05 |
| v0_to_v1 | retire_recent128 | 252 | 0.0218999 | -3.1425 | 0.007895 | 15.4173 | 0.365497 | 0.00107044 |
| v0_to_v1 | retire_diversity128 | 252 | 0.0074738 | -0.413716 | 7.69941e-05 | 1.1406 | 0.447917 | 0.000106779 |
| v0_to_v1 | retire_old128_mass4over3 | 252 | 0.00661586 | -0.251431 | -0.00072143 | -0.317421 | 0.451572 | 9.36079e-05 |
| v0_to_v1 | retire_diversity128_mass4over3 | 252 | 0.00747656 | -0.414238 | 7.85491e-05 | 1.14344 | 0.447917 | 0.00010684 |
| v0_to_v1 | route_candidate384 | 252 | 0.00553555 | -0.0470832 | -0.000181181 | 0.669141 | 0.445541 | 6.17722e-05 |
| v0_to_v1 | route_qk384 | 252 | 0.004403 | 0.167145 | -0.00208003 | -2.79839 | 0.46345 | 3.94322e-05 |
| v0_to_v1 | route_candidate384_mass4over3 | 252 | 0.00553834 | -0.0476126 | -0.000179603 | 0.672023 | 0.445541 | 6.18168e-05 |
| v0_to_v1 | route_qk384_mass4over3 | 252 | 0.00440173 | 0.167385 | -0.00207898 | -2.79648 | 0.46345 | 3.94097e-05 |
| v0_to_v1 | translate_plus_rematerialize128 | 252 | 0.00168188 | 0.681861 | 4.47482e-05 | 1.08172 | 0.446272 | 5.7619e-06 |
| v0_to_v1 | translate_plus_synthesize8 | 252 | 0.00348212 | 0.341335 | 0.000127207 | 1.2323 | 0.447368 | 2.8565e-05 |
| v0_to_v1 | translate_plus_synthesize_landmark32_mass4 | 252 | 0.00174617 | 0.6697 | -5.13961e-05 | 0.906144 | 0.447734 | 6.92738e-06 |
| v0_to_v1 | translate_plus_synthesize_landmark64_mass2 | 252 | 0.00167471 | 0.683217 | -3.92421e-05 | 0.928339 | 0.447734 | 6.00139e-06 |
| v0_to_v1 | translate_plus_route_qk384 | 252 | 0.00578731 | -0.0947057 | -0.00153968 | -1.81164 | 0.461623 | 6.53214e-05 |
| v0_to_v1 | translate_plus_retire_old64 | 252 | 0.00300282 | 0.431998 | 2.88136e-05 | 1.05262 | 0.445906 | 2.08219e-05 |
| v0_to_v1 | translate_plus_retire_old128 | 252 | 0.00430407 | 0.185858 | -0.000103897 | 0.810271 | 0.444993 | 4.63064e-05 |
| v0_to_v1 | translate_plus_retire_diversity128 | 252 | 0.0047969 | 0.0926368 | 0.000654388 | 2.195 | 0.443713 | 5.19319e-05 |
| v0_to_v1 | translate_plus_route_qk384_mass4over3 | 252 | 0.00578493 | -0.0942562 | -0.00153869 | -1.80983 | 0.461623 | 6.52798e-05 |
| v0_to_v1 | translate_plus_retire_old64_mass8over7 | 252 | 0.00300351 | 0.431867 | 2.9462e-05 | 1.0538 | 0.445906 | 2.08316e-05 |
| v0_to_v1 | translate_plus_retire_old128_mass4over3 | 252 | 0.0043058 | 0.18553 | -0.000102398 | 0.813008 | 0.444993 | 4.63335e-05 |
| v1_to_v2 | reuse_parent | 247 | 0.00164229 | 0 | 0.000270742 | 0 | 0.661588 | 5.74769e-06 |
| v1_to_v2 | current_exact | 247 | 0 | 1 | 0 | 1 | 0.663666 | 0 |
| v1_to_v2 | translate_joint | 247 | 0.000839554 | 0.488792 | 9.33718e-05 | 0.655126 | 0.662512 | 1.51392e-06 |
| v1_to_v2 | reader_bridge | 247 | 0.00188567 | -0.148192 | 0.000369638 | -0.365279 | 0.661127 | 7.78962e-06 |
| v1_to_v2 | rematerialize_tail128 | 247 | 0.00121782 | 0.258461 | 0.000221366 | 0.18237 | 0.662742 | 3.15393e-06 |
| v1_to_v2 | synthesize_tail8 | 247 | 0.00329242 | -1.00477 | 0.000806404 | -1.9785 | 0.649815 | 2.60837e-05 |
| v1_to_v2 | synthesize_landmark8 | 247 | 0.00328044 | -0.997477 | 0.000767489 | -1.83477 | 0.650046 | 2.63245e-05 |
| v1_to_v2 | synthesize_sparse_oracle8 | 247 | 0.00327497 | -0.994143 | 0.000764516 | -1.82378 | 0.650046 | 2.62303e-05 |
| v1_to_v2 | synthesize_tail8_mass16 | 247 | 0.00287531 | -0.75079 | 0.000814044 | -2.00672 | 0.66205 | 1.79617e-05 |
| v1_to_v2 | synthesize_landmark8_mass16 | 247 | 0.00205913 | -0.253814 | 0.000272744 | -0.00739646 | 0.661588 | 1.01269e-05 |
| v1_to_v2 | synthesize_sparse_oracle8_mass16 | 247 | 0.00206525 | -0.257538 | 0.000237418 | 0.123081 | 0.66205 | 1.01687e-05 |
| v1_to_v2 | synthesize_landmark32 | 247 | 0.00267558 | -0.629171 | 0.000632266 | -1.33531 | 0.654894 | 1.6614e-05 |
| v1_to_v2 | synthesize_landmark64 | 247 | 0.00198603 | -0.209306 | 0.000459351 | -0.696641 | 0.658126 | 8.25555e-06 |
| v1_to_v2 | synthesize_sparse_oracle32 | 247 | 0.00266149 | -0.620592 | 0.000624638 | -1.30714 | 0.654894 | 1.64227e-05 |
| v1_to_v2 | synthesize_sparse_oracle64 | 247 | 0.00197034 | -0.199748 | 0.000451164 | -0.6664 | 0.658126 | 8.12064e-06 |
| v1_to_v2 | synthesize_landmark32_mass4 | 247 | 0.0013705 | 0.165494 | 0.000240782 | 0.110657 | 0.661819 | 4.21206e-06 |
| v1_to_v2 | synthesize_landmark64_mass2 | 247 | 0.00126411 | 0.230278 | 0.000203189 | 0.24951 | 0.662512 | 3.47516e-06 |
| v1_to_v2 | synthesize_sparse_oracle32_mass4 | 247 | 0.00138271 | 0.158064 | 0.000217175 | 0.197852 | 0.662973 | 4.29063e-06 |
| v1_to_v2 | synthesize_sparse_oracle64_mass2 | 247 | 0.00127275 | 0.225019 | 0.000189708 | 0.299301 | 0.662742 | 3.52615e-06 |
| v1_to_v2 | retire_old128 | 247 | 0.00436795 | -1.65966 | -0.000719687 | 3.65821 | 0.668283 | 4.54234e-05 |
| v1_to_v2 | retire_recent128 | 247 | 0.0240419 | -13.6392 | 0.0110874 | -39.9519 | 0.542475 | 0.00125832 |
| v1_to_v2 | retire_diversity128 | 247 | 0.00425114 | -1.58854 | 0.00067574 | -1.49589 | 0.662281 | 4.33228e-05 |
| v1_to_v2 | retire_old128_mass4over3 | 247 | 0.00436792 | -1.65965 | -0.000719237 | 3.65655 | 0.668283 | 4.54316e-05 |
| v1_to_v2 | retire_diversity128_mass4over3 | 247 | 0.00425271 | -1.58949 | 0.000676414 | -1.49837 | 0.662281 | 4.33564e-05 |
| v1_to_v2 | route_candidate384 | 247 | 0.00226752 | -0.380707 | 0.00033648 | -0.242809 | 0.663435 | 1.23649e-05 |
| v1_to_v2 | route_qk384 | 247 | 0.00675149 | -3.11101 | -0.000955372 | 4.52873 | 0.657202 | 7.94565e-05 |
| v1_to_v2 | route_candidate384_mass4over3 | 247 | 0.0022679 | -0.380933 | 0.00033716 | -0.245321 | 0.663435 | 1.23783e-05 |
| v1_to_v2 | route_qk384_mass4over3 | 247 | 0.00674862 | -3.10927 | -0.000955079 | 4.52764 | 0.657202 | 7.93995e-05 |
| v1_to_v2 | translate_plus_rematerialize128 | 247 | 0.000613804 | 0.626251 | 9.69349e-05 | 0.641965 | 0.662512 | 8.00315e-07 |
| v1_to_v2 | translate_plus_synthesize8 | 247 | 0.00301712 | -0.837141 | 0.000646025 | -1.38613 | 0.649354 | 2.40937e-05 |
| v1_to_v2 | translate_plus_synthesize_landmark32_mass4 | 247 | 0.000857786 | 0.47769 | 0.000113409 | 0.581117 | 0.662973 | 1.639e-06 |
| v1_to_v2 | translate_plus_synthesize_landmark64_mass2 | 247 | 0.000689985 | 0.579865 | 8.01255e-05 | 0.704052 | 0.663435 | 1.05097e-06 |
| v1_to_v2 | translate_plus_route_qk384 | 247 | 0.00700328 | -3.26433 | -0.00104055 | 4.84334 | 0.655586 | 8.6449e-05 |
| v1_to_v2 | translate_plus_retire_old64 | 247 | 0.00253187 | -0.541669 | -0.000669767 | 3.47383 | 0.66482 | 1.48128e-05 |
| v1_to_v2 | translate_plus_retire_old128 | 247 | 0.00411705 | -1.50689 | -0.00102642 | 4.79114 | 0.66759 | 4.24163e-05 |
| v1_to_v2 | translate_plus_retire_diversity128 | 247 | 0.00377525 | -1.29877 | 0.00047744 | -0.763453 | 0.662512 | 3.73233e-05 |
| v1_to_v2 | translate_plus_route_qk384_mass4over3 | 247 | 0.0070005 | -3.26264 | -0.00104026 | 4.84228 | 0.655586 | 8.63932e-05 |
| v1_to_v2 | translate_plus_retire_old64_mass8over7 | 247 | 0.00253175 | -0.541592 | -0.000669542 | 3.47299 | 0.66482 | 1.48149e-05 |
| v1_to_v2 | translate_plus_retire_old128_mass4over3 | 247 | 0.00411704 | -1.50689 | -0.00102601 | 4.78963 | 0.66759 | 4.24198e-05 |
| v2_to_v3 | reuse_parent | 256 | 0.00145258 | 0 | 0.000596295 | 0 | 0.599498 | 4.63543e-06 |
| v2_to_v3 | current_exact | 256 | 0 | 1 | 0 | 1 | 0.605351 | 0 |
| v2_to_v3 | translate_joint | 256 | 0.000521554 | 0.640947 | 0.000201466 | 0.662137 | 0.603344 | 6.54218e-07 |
| v2_to_v3 | reader_bridge | 256 | 0.0015882 | -0.0933592 | 0.000660832 | -0.108229 | 0.598662 | 5.58552e-06 |
| v2_to_v3 | rematerialize_tail128 | 256 | 0.00111624 | 0.231547 | 0.000493132 | 0.173008 | 0.600669 | 2.72408e-06 |
| v2_to_v3 | synthesize_tail8 | 256 | 0.00348049 | -1.39607 | -0.00109706 | 2.8398 | 0.614883 | 2.91365e-05 |
| v2_to_v3 | synthesize_landmark8 | 256 | 0.00354539 | -1.44074 | -0.0011195 | 2.87742 | 0.615385 | 2.99552e-05 |
| v2_to_v3 | synthesize_sparse_oracle8 | 256 | 0.00353896 | -1.43632 | -0.00111631 | 2.87208 | 0.615385 | 2.98267e-05 |
| v2_to_v3 | synthesize_tail8_mass16 | 256 | 0.00207802 | -0.430565 | 0.000846981 | -0.420405 | 0.598997 | 9.12291e-06 |
| v2_to_v3 | synthesize_landmark8_mass16 | 256 | 0.00202517 | -0.394187 | 0.000582427 | 0.0232571 | 0.6 | 8.68695e-06 |
| v2_to_v3 | synthesize_sparse_oracle8_mass16 | 256 | 0.0020154 | -0.387455 | 0.000625303 | -0.0486468 | 0.599331 | 8.51781e-06 |
| v2_to_v3 | synthesize_landmark32 | 256 | 0.00283054 | -0.948621 | -0.000698449 | 2.17131 | 0.609365 | 1.8653e-05 |
| v2_to_v3 | synthesize_landmark64 | 256 | 0.00200271 | -0.378721 | -0.000289488 | 1.48548 | 0.607191 | 9.15581e-06 |
| v2_to_v3 | synthesize_sparse_oracle32 | 256 | 0.00281244 | -0.936162 | -0.000688809 | 2.15515 | 0.60903 | 1.83768e-05 |
| v2_to_v3 | synthesize_sparse_oracle64 | 256 | 0.00198305 | -0.365189 | -0.000277903 | 1.46605 | 0.607191 | 8.95531e-06 |
| v2_to_v3 | synthesize_landmark32_mass4 | 256 | 0.00131687 | 0.0934272 | 0.000672556 | -0.127891 | 0.598495 | 3.71952e-06 |
| v2_to_v3 | synthesize_landmark64_mass2 | 256 | 0.00119553 | 0.17696 | 0.000504462 | 0.154007 | 0.601003 | 3.07807e-06 |
| v2_to_v3 | synthesize_sparse_oracle32_mass4 | 256 | 0.00131262 | 0.0963585 | 0.000705386 | -0.182947 | 0.598662 | 3.67988e-06 |
| v2_to_v3 | synthesize_sparse_oracle64_mass2 | 256 | 0.00119258 | 0.178994 | 0.000525308 | 0.119047 | 0.600836 | 3.0572e-06 |
| v2_to_v3 | retire_old128 | 256 | 0.00412346 | -1.8387 | 0.000687029 | -0.152162 | 0.596823 | 3.99229e-05 |
| v2_to_v3 | retire_recent128 | 256 | 0.0175237 | -11.0638 | 0.00191347 | -2.20893 | 0.583445 | 0.000723326 |
| v2_to_v3 | retire_diversity128 | 256 | 0.00378179 | -1.60349 | -3.12018e-05 | 1.05233 | 0.602007 | 3.57033e-05 |
| v2_to_v3 | retire_old128_mass4over3 | 256 | 0.00412541 | -1.84005 | 0.000688877 | -0.155261 | 0.596823 | 3.99492e-05 |
| v2_to_v3 | retire_diversity128_mass4over3 | 256 | 0.0037849 | -1.60563 | -2.92461e-05 | 1.04905 | 0.602007 | 3.5749e-05 |
| v2_to_v3 | route_candidate384 | 256 | 0.00207772 | -0.430362 | 0.001132 | -0.898387 | 0.595652 | 1.02079e-05 |
| v2_to_v3 | route_qk384 | 256 | 0.00799457 | -4.50369 | 7.43374e-05 | 0.875335 | 0.603344 | 0.000106362 |
| v2_to_v3 | route_candidate384_mass4over3 | 256 | 0.00207991 | -0.431866 | 0.00113376 | -0.90134 | 0.595652 | 1.02278e-05 |
| v2_to_v3 | route_qk384_mass4over3 | 256 | 0.00799098 | -4.50122 | 7.54279e-05 | 0.873506 | 0.603344 | 0.000106298 |
| v2_to_v3 | translate_plus_rematerialize128 | 256 | 0.00039762 | 0.726267 | 0.000168738 | 0.717023 | 0.603846 | 3.92946e-07 |
| v2_to_v3 | translate_plus_synthesize8 | 256 | 0.00322272 | -1.21861 | -0.00149155 | 3.50135 | 0.616722 | 2.70603e-05 |
| v2_to_v3 | translate_plus_synthesize_landmark32_mass4 | 256 | 0.000750994 | 0.482994 | 0.000348338 | 0.41583 | 0.601505 | 1.35382e-06 |
| v2_to_v3 | translate_plus_synthesize_landmark64_mass2 | 256 | 0.000569218 | 0.608134 | 0.000178071 | 0.70137 | 0.603846 | 6.99953e-07 |
| v2_to_v3 | translate_plus_route_qk384 | 256 | 0.00791101 | -4.44616 | -0.000270731 | 1.45402 | 0.602508 | 0.000109308 |
| v2_to_v3 | translate_plus_retire_old64 | 256 | 0.00235335 | -0.620115 | 5.29774e-05 | 0.911156 | 0.604515 | 1.33439e-05 |
| v2_to_v3 | translate_plus_retire_old128 | 256 | 0.00376709 | -1.59337 | 0.000282146 | 0.526836 | 0.600669 | 3.63322e-05 |
| v2_to_v3 | translate_plus_retire_diversity128 | 256 | 0.00340584 | -1.34467 | -0.000484651 | 1.81277 | 0.604013 | 3.23607e-05 |
| v2_to_v3 | translate_plus_route_qk384_mass4over3 | 256 | 0.0079074 | -4.44368 | -0.000269694 | 1.45228 | 0.602508 | 0.000109235 |
| v2_to_v3 | translate_plus_retire_old64_mass8over7 | 256 | 0.00235318 | -0.619998 | 5.36472e-05 | 0.910032 | 0.604515 | 1.33447e-05 |
| v2_to_v3 | translate_plus_retire_old128_mass4over3 | 256 | 0.00376803 | -1.59402 | 0.000283879 | 0.52393 | 0.600669 | 3.63471e-05 |
| v3_to_v4 | reuse_parent | 256 | 0.00496488 | 0 | 0.00023838 | 0 | 0.623484 | 4.62869e-05 |
| v3_to_v4 | current_exact | 256 | 0 | 1 | 0 | 1 | 0.625586 | 0 |
| v3_to_v4 | translate_joint | 256 | 0.00375873 | 0.242936 | 0.00025137 | -0.0544952 | 0.623484 | 2.41286e-05 |
| v3_to_v4 | reader_bridge | 256 | 0.00520118 | -0.047595 | 0.000261581 | -0.0973316 | 0.623484 | 5.14685e-05 |
| v3_to_v4 | rematerialize_tail128 | 256 | 0.00385525 | 0.223496 | 0.000223361 | 0.0630039 | 0.623322 | 2.83351e-05 |
| v3_to_v4 | synthesize_tail8 | 256 | 0.0052945 | -0.0663915 | 0.000585287 | -1.45527 | 0.621543 | 6.89819e-05 |
| v3_to_v4 | synthesize_landmark8 | 256 | 0.00520379 | -0.0481204 | 0.000594902 | -1.49561 | 0.621543 | 6.77322e-05 |
| v3_to_v4 | synthesize_sparse_oracle8 | 256 | 0.00520186 | -0.0477315 | 0.000595759 | -1.49921 | 0.621543 | 6.76674e-05 |
| v3_to_v4 | synthesize_tail8_mass16 | 256 | 0.00579821 | -0.167847 | -1.80471e-06 | 1.00757 | 0.62316 | 5.7416e-05 |
| v3_to_v4 | synthesize_landmark8_mass16 | 256 | 0.00443707 | 0.106308 | 2.61089e-05 | 0.890473 | 0.62591 | 3.81887e-05 |
| v3_to_v4 | synthesize_sparse_oracle8_mass16 | 256 | 0.00446146 | 0.101396 | 4.59776e-05 | 0.807125 | 0.625425 | 3.85509e-05 |
| v3_to_v4 | synthesize_landmark32 | 256 | 0.00474805 | 0.0436714 | 0.00039945 | -0.675687 | 0.623484 | 5.25223e-05 |
| v3_to_v4 | synthesize_landmark64 | 256 | 0.00432559 | 0.128762 | 0.000302409 | -0.268601 | 0.62591 | 3.96306e-05 |
| v3_to_v4 | synthesize_sparse_oracle32 | 256 | 0.0047481 | 0.0436618 | 0.000400701 | -0.680937 | 0.623645 | 5.24053e-05 |
| v3_to_v4 | synthesize_sparse_oracle64 | 256 | 0.00432695 | 0.128488 | 0.000304307 | -0.276564 | 0.626071 | 3.95739e-05 |
| v3_to_v4 | synthesize_landmark32_mass4 | 256 | 0.00392332 | 0.209786 | -1.23622e-05 | 1.05186 | 0.624939 | 2.88622e-05 |
| v3_to_v4 | synthesize_landmark64_mass2 | 256 | 0.00387011 | 0.220502 | 0.000156043 | 0.345399 | 0.624616 | 2.87706e-05 |
| v3_to_v4 | synthesize_sparse_oracle32_mass4 | 256 | 0.00393748 | 0.206932 | -3.07932e-06 | 1.01292 | 0.625425 | 2.90686e-05 |
| v3_to_v4 | synthesize_sparse_oracle64_mass2 | 256 | 0.00387976 | 0.218558 | 0.000161386 | 0.322989 | 0.624616 | 2.88988e-05 |
| v3_to_v4 | retire_old128 | 256 | 0.00615285 | -0.239276 | -0.000174466 | 1.73188 | 0.625101 | 9.24352e-05 |
| v3_to_v4 | retire_recent128 | 256 | 0.0233445 | -3.70193 | 0.00731745 | -29.6966 | 0.567362 | 0.00121559 |
| v3_to_v4 | retire_diversity128 | 256 | 0.00757863 | -0.52645 | 0.000256793 | -0.0772459 | 0.620573 | 0.000123556 |
| v3_to_v4 | retire_old128_mass4over3 | 256 | 0.00615551 | -0.239811 | -0.000172958 | 1.72556 | 0.625101 | 9.24977e-05 |
| v3_to_v4 | retire_diversity128_mass4over3 | 256 | 0.0075818 | -0.527088 | 0.00025838 | -0.0839024 | 0.620573 | 0.000123635 |
| v3_to_v4 | route_candidate384 | 256 | 0.00523687 | -0.0547846 | 5.77363e-05 | 0.757797 | 0.623645 | 5.40627e-05 |
| v3_to_v4 | route_qk384 | 256 | 0.00345932 | 0.303241 | -0.00113822 | 5.77481 | 0.637231 | 2.78029e-05 |
| v3_to_v4 | route_candidate384_mass4over3 | 256 | 0.00523995 | -0.0554049 | 5.8986e-05 | 0.752554 | 0.623645 | 5.41032e-05 |
| v3_to_v4 | route_qk384_mass4over3 | 256 | 0.0034577 | 0.303567 | -0.00113759 | 5.77218 | 0.637231 | 2.77704e-05 |
| v3_to_v4 | translate_plus_rematerialize128 | 256 | 0.00283279 | 0.429433 | 0.00023921 | -0.00348278 | 0.624292 | 1.38698e-05 |
| v3_to_v4 | translate_plus_synthesize8 | 256 | 0.00424859 | 0.14427 | 0.000608877 | -1.55423 | 0.620896 | 4.67265e-05 |
| v3_to_v4 | translate_plus_synthesize_landmark32_mass4 | 256 | 0.00286041 | 0.423871 | 1.27399e-06 | 0.994656 | 0.625425 | 1.46831e-05 |
| v3_to_v4 | translate_plus_synthesize_landmark64_mass2 | 256 | 0.00282227 | 0.431553 | 0.000169388 | 0.289418 | 0.624939 | 1.4097e-05 |
| v3_to_v4 | translate_plus_route_qk384 | 256 | 0.00351843 | 0.291335 | -0.000901845 | 4.78323 | 0.636422 | 2.8943e-05 |
| v3_to_v4 | translate_plus_retire_old64 | 256 | 0.00424552 | 0.144889 | -0.000855777 | 4.58998 | 0.631247 | 4.33741e-05 |
| v3_to_v4 | translate_plus_retire_old128 | 256 | 0.00537128 | -0.0818551 | -0.000160687 | 1.67408 | 0.624616 | 7.36567e-05 |
| v3_to_v4 | translate_plus_retire_diversity128 | 256 | 0.00634169 | -0.277312 | 0.000139858 | 0.413295 | 0.620411 | 9.03757e-05 |
| v3_to_v4 | translate_plus_route_qk384_mass4over3 | 256 | 0.00351613 | 0.291799 | -0.000901208 | 4.78056 | 0.636422 | 2.89089e-05 |
| v3_to_v4 | translate_plus_retire_old64_mass8over7 | 256 | 0.00424688 | 0.144614 | -0.000855218 | 4.58763 | 0.631247 | 4.33947e-05 |
| v3_to_v4 | translate_plus_retire_old128_mass4over3 | 256 | 0.00537388 | -0.0823807 | -0.000159153 | 1.66764 | 0.624616 | 7.37204e-05 |
| v4_to_v5 | reuse_parent | 256 | 0.00103057 | 0 | -0.000468934 | 0 | 0.507662 | 2.44723e-06 |
| v4_to_v5 | current_exact | 256 | 0 | 1 | 0 | 1 | 0.506995 | 0 |
| v4_to_v5 | translate_joint | 256 | 0.000808396 | 0.215583 | -0.000469697 | -0.00162884 | 0.507884 | 1.4645e-06 |
| v4_to_v5 | reader_bridge | 256 | 0.00109514 | -0.0626516 | -0.000425908 | 0.0917523 | 0.508106 | 2.72567e-06 |
| v4_to_v5 | rematerialize_tail128 | 256 | 0.000831149 | 0.193504 | -0.000394318 | 0.159119 | 0.508106 | 1.62609e-06 |
| v4_to_v5 | synthesize_tail8 | 256 | 0.00339153 | -2.29093 | -0.00077871 | -0.660598 | 0.513435 | 2.93678e-05 |
| v4_to_v5 | synthesize_landmark8 | 256 | 0.00341845 | -2.31705 | -0.000804468 | -0.715525 | 0.513435 | 2.97415e-05 |
| v4_to_v5 | synthesize_sparse_oracle8 | 256 | 0.00341326 | -2.31202 | -0.000804122 | -0.714788 | 0.513435 | 2.96454e-05 |
| v4_to_v5 | synthesize_tail8_mass16 | 256 | 0.00212598 | -1.06292 | 0.000667678 | 2.42382 | 0.505219 | 9.60745e-06 |
| v4_to_v5 | synthesize_landmark8_mass16 | 256 | 0.00184538 | -0.790647 | 0.000281227 | 1.59972 | 0.503886 | 8.63299e-06 |
| v4_to_v5 | synthesize_sparse_oracle8_mass16 | 256 | 0.00183881 | -0.784265 | 0.000288225 | 1.61464 | 0.50433 | 8.4987e-06 |
| v4_to_v5 | synthesize_landmark32 | 256 | 0.00269331 | -1.61342 | -0.000731338 | -0.559577 | 0.512103 | 1.81649e-05 |
| v4_to_v5 | synthesize_landmark64 | 256 | 0.00191478 | -0.857987 | -0.00061815 | -0.318204 | 0.511659 | 8.79919e-06 |
| v4_to_v5 | synthesize_sparse_oracle32 | 256 | 0.00267961 | -1.60013 | -0.000727669 | -0.551752 | 0.512325 | 1.79675e-05 |
| v4_to_v5 | synthesize_sparse_oracle64 | 256 | 0.00189896 | -0.84263 | -0.000613494 | -0.308275 | 0.511659 | 8.64697e-06 |
| v4_to_v5 | synthesize_landmark32_mass4 | 256 | 0.00104268 | -0.0117524 | -0.000273546 | 0.416663 | 0.508994 | 2.47741e-06 |
| v4_to_v5 | synthesize_landmark64_mass2 | 256 | 0.000933091 | 0.0945865 | -0.00035924 | 0.233921 | 0.507217 | 2.05534e-06 |
| v4_to_v5 | synthesize_sparse_oracle32_mass4 | 256 | 0.00102345 | 0.00690906 | -0.000258808 | 0.448092 | 0.508106 | 2.4194e-06 |
| v4_to_v5 | synthesize_sparse_oracle64_mass2 | 256 | 0.000919465 | 0.107808 | -0.000350127 | 0.253355 | 0.507662 | 2.02053e-06 |
| v4_to_v5 | retire_old128 | 256 | 0.00425291 | -3.12676 | -0.00147936 | -2.15473 | 0.527204 | 5.32706e-05 |
| v4_to_v5 | retire_recent128 | 256 | 0.0227314 | -21.0571 | -0.018926 | -39.3597 | 0.686209 | 0.00115824 |
| v4_to_v5 | retire_diversity128 | 256 | 0.00358537 | -2.47902 | 0.000662336 | 2.41243 | 0.505663 | 3.61704e-05 |
| v4_to_v5 | retire_old128_mass4over3 | 256 | 0.00425331 | -3.12715 | -0.00147473 | -2.14485 | 0.527204 | 5.32985e-05 |
| v4_to_v5 | retire_diversity128_mass4over3 | 256 | 0.00358761 | -2.4812 | 0.000667267 | 2.42295 | 0.505663 | 3.62062e-05 |
| v4_to_v5 | route_candidate384 | 256 | 0.00178226 | -0.729392 | 6.95622e-05 | 1.14834 | 0.500333 | 7.49228e-06 |
| v4_to_v5 | route_qk384 | 256 | 0.00723637 | -6.02173 | -0.00369188 | -6.87292 | 0.525205 | 9.36124e-05 |
| v4_to_v5 | route_candidate384_mass4over3 | 256 | 0.00178114 | -0.72831 | 7.40892e-05 | 1.158 | 0.500333 | 7.47867e-06 |
| v4_to_v5 | route_qk384_mass4over3 | 256 | 0.00723282 | -6.01828 | -0.0036887 | -6.86615 | 0.525205 | 9.35296e-05 |
| v4_to_v5 | translate_plus_rematerialize128 | 256 | 0.000631512 | 0.38722 | -0.000374561 | 0.201249 | 0.508106 | 9.18706e-07 |
| v4_to_v5 | translate_plus_synthesize8 | 256 | 0.00327568 | -2.17852 | -0.000795237 | -0.695841 | 0.513435 | 2.82871e-05 |
| v4_to_v5 | translate_plus_synthesize_landmark32_mass4 | 256 | 0.000832065 | 0.192616 | -0.000255102 | 0.455996 | 0.509438 | 1.6693e-06 |
| v4_to_v5 | translate_plus_synthesize_landmark64_mass2 | 256 | 0.000729562 | 0.292078 | -0.000342604 | 0.269398 | 0.507217 | 1.28095e-06 |
| v4_to_v5 | translate_plus_route_qk384 | 256 | 0.00714491 | -5.93298 | -0.00372645 | -6.94665 | 0.525872 | 9.34189e-05 |
| v4_to_v5 | translate_plus_retire_old64 | 256 | 0.00241274 | -1.34117 | -0.000859009 | -0.831834 | 0.516767 | 1.56137e-05 |
| v4_to_v5 | translate_plus_retire_old128 | 256 | 0.00418025 | -3.05625 | -0.00145969 | -2.11278 | 0.52787 | 5.30328e-05 |
| v4_to_v5 | translate_plus_retire_diversity128 | 256 | 0.003542 | -2.43694 | 0.000714037 | 2.52268 | 0.506107 | 3.58519e-05 |
| v4_to_v5 | translate_plus_route_qk384_mass4over3 | 256 | 0.00714136 | -5.92953 | -0.00372324 | -6.93981 | 0.525872 | 9.33364e-05 |
| v4_to_v5 | translate_plus_retire_old64_mass8over7 | 256 | 0.00241238 | -1.34083 | -0.000857007 | -0.827565 | 0.516767 | 1.56155e-05 |
| v4_to_v5 | translate_plus_retire_old128_mass4over3 | 256 | 0.00418063 | -3.05662 | -0.00145496 | -2.10271 | 0.52787 | 5.30629e-05 |

## Composition complementarity

| edge | composition | combined_recovery | best_component_recovery | increment_over_best |
| --- | --- | --- | --- | --- |
| v0_to_v1 | translate_plus_rematerialize128 | 0.681861 | 0.560424 | 0.121438 |
| v0_to_v1 | translate_plus_synthesize8 | 0.341335 | 0.560424 | -0.219089 |
| v0_to_v1 | synthesize8_over_retire_recent128 | -0.0522844 | -3.1425 | 3.09022 |
| v0_to_v1 | translate_plus_route_qk384 | -0.0947057 | 0.560424 | -0.655129 |
| v0_to_v1 | sparse_oracle8_over_retire_recent128 | -0.036194 | -3.1425 | 3.10631 |
| v0_to_v1 | mass_aware_sparse_oracle8 | 0.197852 | -0.036194 | 0.234046 |
| v0_to_v1 | mass_aware_qk_route384 | 0.167385 | 0.167145 | 0.000239205 |
| v0_to_v1 | translate_plus_mass_aware_retire64 | 0.431867 | 0.560424 | -0.128557 |
| v0_to_v1 | mass_aware_landmark32_over_landmark8 | 0.243964 | 0.203 | 0.0409642 |
| v0_to_v1 | translate_plus_synthesize_landmark32_mass4 | 0.6697 | 0.560424 | 0.109277 |
| v0_to_v1 | mass_aware_landmark64 | 0.25008 | 0.157091 | 0.0929889 |
| v0_to_v1 | translate_plus_synthesize_landmark64_mass2 | 0.683217 | 0.560424 | 0.122794 |
| v1_to_v2 | translate_plus_rematerialize128 | 0.626251 | 0.488792 | 0.13746 |
| v1_to_v2 | translate_plus_synthesize8 | -0.837141 | 0.488792 | -1.32593 |
| v1_to_v2 | synthesize8_over_retire_recent128 | -1.00477 | -13.6392 | 12.6344 |
| v1_to_v2 | translate_plus_route_qk384 | -3.26433 | 0.488792 | -3.75312 |
| v1_to_v2 | sparse_oracle8_over_retire_recent128 | -0.994143 | -13.6392 | 12.6451 |
| v1_to_v2 | mass_aware_sparse_oracle8 | -0.257538 | -0.994143 | 0.736604 |
| v1_to_v2 | mass_aware_qk_route384 | -3.10927 | -3.11101 | 0.00174518 |
| v1_to_v2 | translate_plus_mass_aware_retire64 | -0.541592 | 0.488792 | -1.03038 |
| v1_to_v2 | mass_aware_landmark32_over_landmark8 | 0.165494 | -0.253814 | 0.419309 |
| v1_to_v2 | translate_plus_synthesize_landmark32_mass4 | 0.47769 | 0.488792 | -0.0111019 |
| v1_to_v2 | mass_aware_landmark64 | 0.230278 | -0.209306 | 0.439584 |
| v1_to_v2 | translate_plus_synthesize_landmark64_mass2 | 0.579865 | 0.488792 | 0.0910729 |
| v2_to_v3 | translate_plus_rematerialize128 | 0.726267 | 0.640947 | 0.0853203 |
| v2_to_v3 | translate_plus_synthesize8 | -1.21861 | 0.640947 | -1.85956 |
| v2_to_v3 | synthesize8_over_retire_recent128 | -1.39607 | -11.0638 | 9.66773 |
| v2_to_v3 | translate_plus_route_qk384 | -4.44616 | 0.640947 | -5.08711 |
| v2_to_v3 | sparse_oracle8_over_retire_recent128 | -1.43632 | -11.0638 | 9.62748 |
| v2_to_v3 | mass_aware_sparse_oracle8 | -0.387455 | -1.43632 | 1.04887 |
| v2_to_v3 | mass_aware_qk_route384 | -4.50122 | -4.50369 | 0.00247103 |
| v2_to_v3 | translate_plus_mass_aware_retire64 | -0.619998 | 0.640947 | -1.26094 |
| v2_to_v3 | mass_aware_landmark32_over_landmark8 | 0.0934272 | -0.394187 | 0.487614 |
| v2_to_v3 | translate_plus_synthesize_landmark32_mass4 | 0.482994 | 0.640947 | -0.157953 |
| v2_to_v3 | mass_aware_landmark64 | 0.17696 | -0.378721 | 0.555682 |
| v2_to_v3 | translate_plus_synthesize_landmark64_mass2 | 0.608134 | 0.640947 | -0.0328128 |
| v3_to_v4 | translate_plus_rematerialize128 | 0.429433 | 0.242936 | 0.186498 |
| v3_to_v4 | translate_plus_synthesize8 | 0.14427 | 0.242936 | -0.0986656 |
| v3_to_v4 | synthesize8_over_retire_recent128 | -0.0663915 | -3.70193 | 3.63554 |
| v3_to_v4 | translate_plus_route_qk384 | 0.291335 | 0.303241 | -0.0119056 |
| v3_to_v4 | sparse_oracle8_over_retire_recent128 | -0.0477315 | -3.70193 | 3.6542 |
| v3_to_v4 | mass_aware_sparse_oracle8 | 0.101396 | -0.0477315 | 0.149127 |
| v3_to_v4 | mass_aware_qk_route384 | 0.303567 | 0.303241 | 0.0003266 |
| v3_to_v4 | translate_plus_mass_aware_retire64 | 0.144614 | 0.242936 | -0.0983217 |
| v3_to_v4 | mass_aware_landmark32_over_landmark8 | 0.209786 | 0.106308 | 0.103477 |
| v3_to_v4 | translate_plus_synthesize_landmark32_mass4 | 0.423871 | 0.242936 | 0.180935 |
| v3_to_v4 | mass_aware_landmark64 | 0.220502 | 0.128762 | 0.09174 |
| v3_to_v4 | translate_plus_synthesize_landmark64_mass2 | 0.431553 | 0.242936 | 0.188617 |
| v4_to_v5 | translate_plus_rematerialize128 | 0.38722 | 0.215583 | 0.171637 |
| v4_to_v5 | translate_plus_synthesize8 | -2.17852 | 0.215583 | -2.3941 |
| v4_to_v5 | synthesize8_over_retire_recent128 | -2.29093 | -21.0571 | 18.7662 |
| v4_to_v5 | translate_plus_route_qk384 | -5.93298 | 0.215583 | -6.14856 |
| v4_to_v5 | sparse_oracle8_over_retire_recent128 | -2.31202 | -21.0571 | 18.7451 |
| v4_to_v5 | mass_aware_sparse_oracle8 | -0.784265 | -2.31202 | 1.52775 |
| v4_to_v5 | mass_aware_qk_route384 | -6.01828 | -6.02173 | 0.00344779 |
| v4_to_v5 | translate_plus_mass_aware_retire64 | -1.34083 | 0.215583 | -1.55641 |
| v4_to_v5 | mass_aware_landmark32_over_landmark8 | -0.0117524 | -0.790647 | 0.778894 |
| v4_to_v5 | translate_plus_synthesize_landmark32_mass4 | 0.192616 | 0.215583 | -0.0229671 |
| v4_to_v5 | mass_aware_landmark64 | 0.0945865 | -0.857987 | 0.952574 |
| v4_to_v5 | translate_plus_synthesize_landmark64_mass2 | 0.292078 | 0.215583 | 0.0764955 |

## Primitive adjudication

| primitive | representative_path | decision | metric | edges | mean_output_gap_recovery | min_output_gap_recovery | max_output_gap_recovery |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TRANSLATE | translate_joint | advance | path_output_gap_recovery | 5 | 0.429736 | 0.215583 | 0.640947 |
| REMATERIALIZE | rematerialize_tail128 | advance_exact | path_output_gap_recovery | 5 | 0.231393 | 0.193504 | 0.258461 |
| SYNTHESIZE+REWEIGHT | synthesize_landmark64_mass2 | advance_approximate | path_output_gap_recovery | 5 | 0.194481 | 0.0945865 | 0.25008 |
| FUSE | translate_plus_rematerialize128 | advance_high_recovery_plan | path_output_gap_recovery | 5 | 0.570207 | 0.38722 | 0.726267 |
| FUSE | translate_plus_synthesize_landmark64_mass2 | advance_lower_compute_plan | path_output_gap_recovery | 5 | 0.518969 | 0.292078 | 0.683217 |
| ADAPT_READ | reader_bridge | reject_first_catalog | path_output_gap_recovery | 5 | -0.0849154 | -0.148192 | -0.047595 |
| RETIRE | retire_old128 | reject_first_catalog | path_output_gap_recovery | 5 | -1.42307 | -3.12676 | -0.239276 |
| RETIRE | retire_diversity128 | reject_first_catalog | path_output_gap_recovery | 5 | -1.32224 | -2.47902 | -0.413716 |
| ROUTE | route_candidate384 | reject_first_catalog | path_output_gap_recovery | 5 | -0.328466 | -0.729392 | -0.0470832 |
| ROUTE | route_qk384 | reject_first_catalog | path_output_gap_recovery | 5 | -2.63321 | -6.02173 | 0.303241 |
| REWEIGHT | mass_aware_landmark64 | required_by_synthesis | increment_over_unweighted_sidecar | 5 | 0.426514 | 0.09174 | 0.952574 |

## Cost shape

| primitive | plan | phase | raw_tokens | state_token_layers_read | state_token_layers_written | recomputed_token_layers | request_conditioned |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TRANSLATE | translate_joint | release | 0 | 2048 | 2048 | 0 | False |
| ADAPT_READ | reader_bridge | release_or_read | 0 | 2048 | 2048 | 0 | False |
| REMATERIALIZE | tail128 | release | 128 | 1536 | 512 | 512 | False |
| SYNTHESIZE | tail64_landmark_sidecar | release | 128 | 1536 | 256 | 256 | False |
| REWEIGHT | sidecar_multiplicity | release_or_read | 0 | 256 | 256 | 0 | False |
| FUSE | typed_segment_commit | release | 0 | 2048 | 2048 | 0 | False |
| RETIRE | retire_128 | release | 0 | 2048 | 1536 | 0 | False |
| ROUTE | candidate_qk_keep384 | request | 0 | 2048 | 0 | 0 | True |

Joint parameter-only translation recovers 21.6%-64.1% of the output gap on all five edges. Exact tail-128 rematerialization recovers 19.4%-25.8%. A 64-state landmark sidecar with multiplicity two recovers 9.5%-25.0%, while its unweighted counterpart is unstable. Translate plus exact rematerialization reaches 38.7%-72.6% and improves over the better component on all five edges. Translate plus the weighted 64-state sidecar reaches 29.2%-68.3% and is complementary on four of five edges.

Reader-only bridging, FIFO/diversity retirement, embedding routing, and layer-0 Q-K routing worsen output fidelity on most or all edges and leave the first primitive catalog. Candidate-mode companions are in `primitive_candidate_modes.csv`.
