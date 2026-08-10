# KuaiRand logged next-window Recompute-over-Reuse screen

Each user ranks the deduplicated in-catalog impressions from the untouched next natural day using one fixed pre-window query. Positives are engaged impressions and negatives are unengaged impressions.

| current | cache | age | users | candidates median/p95 | AUC | AP | NDCG@10 | NDCG@50 | window CE |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| θ2 | θ1 | 1 | 752 | 52/218 | -0.350% | -0.402% | -0.557% | -0.340% | +1.748% |
| θ3 | θ2 | 1 | 724 | 41/194 | +0.011% | +0.015% | +0.365% | -0.010% | +2.352% |
| θ4 | θ3 | 1 | 723 | 41/174 | +0.103% | -0.107% | +0.054% | -0.109% | +1.426% |
| θ5 | θ4 | 1 | 712 | 38/165 | +0.230% | -0.019% | -0.147% | -0.075% | -0.032% |
| θ6 | θ5 | 1 | 708 | 34/168 | +0.161% | +0.026% | +0.092% | +0.019% | -0.198% |
| θ7 | θ6 | 1 | 705 | 40/173 | -0.571% | -0.361% | -0.258% | -0.269% | -0.087% |
| θ8 | θ7 | 1 | 767 | 43/213 | -0.122% | +0.008% | +0.010% | +0.071% | +0.314% |
| θ8 | θ1 | 7 | 767 | 43/213 | +0.813% | +0.328% | +0.925% | +0.366% | +6.188% |
