# KuaiRand 自然日机会与参数路径报告

状态：development evidence，非正式论文结果。

| Edge | Positive targets | Fresh update CE 优势 | 全时域 stale CE tax | 95% user-cluster CI |
|---:|---:|---:|---:|---|
| 1 | 18065 | 0.263150 | 0.009648 | [0.002802033824495639, 0.017023803574830812] |
| 2 | 8838 | 0.172977 | 0.004183 | [-0.00021163065994233857, 0.007929860468400617] |

| Edge | 首个正样本 stale CE tax | 95% user-cluster CI |
|---:|---:|---|
| 1 | 0.074345 | [0.041630064323544505, 0.10741565227508545] |
| 2 | 0.024154 | [0.007374628819525242, 0.040512291900813575] |

scorer-only 实际训练保留 pooled full-update CE 收益 `22.01%`，未通过 25% hybrid gate。
KV-invariant tail + untied scorer 保留 `52.23%`，通过预声明的 50% primary gate，且 Reuse/Exact cache、hidden、NLL 均逐元素一致。

结论：自然日更新有强任务价值；stale 风险主要集中在模型发布后的最早请求。更扎实的机会是约束流式更新落在 K/V 不变子空间，而不是为所有相邻版本缓存拟合迁移器。
