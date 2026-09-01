# Medium D14/E14 direct long-age Reuse triangle

Status: **medium_D14_E14_direct_long_age_triangle_complete**. New non-adjacent cells: 10/10; full triangle including frozen adjacent cells: 15/15.

Every row is reported as E14. Each comparison uses only New and Reuse produced in the same run. Old is not recomputed for new cells; cross-run Current drift is recorded in JSON and never gates execution.

| Current | KV producer | Gap | Source | Requests | New AUC | Reuse AUC | New − Reuse AUC (pp) | Reuse AUC vs New | New loss | Reuse loss | Reuse loss vs New |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | v0 | 1 | frozen_adjacent | 137,168 | 0.658478 | 0.653302 | +0.5176 | -0.7860% | 0.300306 | 0.301090 | +0.2608% |
| v2 | v0 | 2 | new_direct_long_age | 131,047 | 0.641453 | 0.636504 | +0.4948 | -0.7715% | 0.329456 | 0.330681 | +0.3718% |
| v2 | v1 | 1 | frozen_adjacent | 131,047 | 0.639990 | 0.639055 | +0.0935 | -0.1461% | 0.329707 | 0.330148 | +0.1335% |
| v3 | v0 | 3 | new_direct_long_age | 131,783 | 0.649071 | 0.634869 | +1.4203 | -2.1881% | 0.342090 | 0.344878 | +0.8148% |
| v3 | v1 | 2 | new_direct_long_age | 131,783 | 0.649071 | 0.642730 | +0.6341 | -0.9770% | 0.342090 | 0.343331 | +0.3626% |
| v3 | v2 | 1 | frozen_adjacent | 131,783 | 0.648779 | 0.641689 | +0.7090 | -1.0928% | 0.342075 | 0.343246 | +0.3421% |
| v4 | v0 | 4 | new_direct_long_age | 133,837 | 0.650293 | 0.625144 | +2.5149 | -3.8673% | 0.330403 | 0.335010 | +1.3944% |
| v4 | v1 | 3 | new_direct_long_age | 133,837 | 0.650293 | 0.631882 | +1.8410 | -2.8311% | 0.330403 | 0.333513 | +0.9412% |
| v4 | v2 | 2 | new_direct_long_age | 133,837 | 0.650293 | 0.631410 | +1.8882 | -2.9037% | 0.330403 | 0.333607 | +0.9698% |
| v4 | v3 | 1 | frozen_adjacent | 133,837 | 0.650052 | 0.636473 | +1.3578 | -2.0888% | 0.330475 | 0.332452 | +0.5981% |
| v5 | v0 | 5 | new_direct_long_age | 138,734 | 0.639007 | 0.611726 | +2.7282 | -4.2694% | 0.307485 | 0.315063 | +2.4645% |
| v5 | v1 | 4 | new_direct_long_age | 138,734 | 0.639007 | 0.619252 | +1.9755 | -3.0915% | 0.307485 | 0.312819 | +1.7346% |
| v5 | v2 | 3 | new_direct_long_age | 138,734 | 0.639007 | 0.617506 | +2.1502 | -3.3649% | 0.307485 | 0.313864 | +2.0745% |
| v5 | v3 | 2 | new_direct_long_age | 138,734 | 0.639007 | 0.621755 | +1.7252 | -2.6999% | 0.307485 | 0.312053 | +1.4856% |
| v5 | v4 | 1 | frozen_adjacent | 138,734 | 0.638952 | 0.633640 | +0.5312 | -0.8314% | 0.307555 | 0.308266 | +0.2309% |
