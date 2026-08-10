# KuaiRand untied cache-producer screen

Primary diagnostic is 999-negative pairwise Recompute-over-Reuse. Candidate selection uses tuning users only; holdout values are reported after selection.

| candidate | LR | epochs | tuning min/mean | tuning update CE+ | holdout min/mean | holdout update CE+ |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| cp_lr100_e2 | 0.0001 | 2 | +0.104%/+0.107% | True | +0.127%/+0.130% | True |
| cp_lr200_e2 | 0.0002 | 2 | +0.014%/+0.016% | True | +0.064%/+0.066% | True |
| cp_lr500_e2 | 0.0005 | 2 | +0.034%/+0.037% | True | +0.043%/+0.048% | True |
| cp_lr1000_e2 | 0.001 | 2 | -0.031%/-0.024% | True | +0.035%/+0.038% | True |
| cp_lr500_e4 | 0.0005 | 4 | +0.015%/+0.020% | True | +0.016%/+0.021% | True |
