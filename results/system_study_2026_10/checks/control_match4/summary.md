# 两个GPU入口的原生对照核对

同一任务的请求、用户、版本、标签、候选和时间完全匹配；全部政策保留同一请求集。

| 对照 | 协议 | 候选评分 | 最大logit差 |
| --- | --- | ---: | ---: |
| reuse | behavior_stride16_candidates16 | 3296 | 0 |
| reuse | original_feedback | 51 | 0 |
| rebuild | behavior_stride16_candidates16 | 3296 | 0 |
| rebuild | original_feedback | 51 | 0 |

这是4用户正确性联通证据，不据此选择方案或宣称总体质量优势。
