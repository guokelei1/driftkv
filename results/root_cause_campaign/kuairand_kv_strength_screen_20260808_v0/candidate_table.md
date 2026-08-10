# KuaiRand K/V-only strength screen

Primary workload is the first four engaged next-day targets for users with prefix length at least 256. Selection uses tuning users only.

| candidate | K/V update | tuning pairwise | holdout pairwise | holdout NDCG@10 | holdout MRR | holdout update pairwise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| kv_lr100_e2 | 0.083 | +0.477% | +0.493% | -1.032% | -0.430% | +0.568% |
| kv_lr200_e2 | 0.127 | +0.609% | +0.524% | -1.189% | -1.293% | +0.607% |
| kv_lr500_e2 | 0.229 | +0.573% | +0.474% | -2.291% | -2.199% | +0.591% |
| kv_lr1000_e2 | 0.385 | +0.433% | +0.420% | -2.039% | -2.052% | +0.638% |
| kv_lr500_e4 | 0.370 | +0.667% | +0.578% | -1.697% | -1.554% | +0.702% |
| kv_lr1000_e4 | 0.625 | +0.559% | +0.467% | -1.894% | -1.581% | +0.706% |
