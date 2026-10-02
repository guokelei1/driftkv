# Max v2 → v3：E14结果

训练窗口 [245,259)，评价窗口 [259,273)。
评价完整性：complete

| 端点 | Parent AUC | Current AUC | 相对提升 | 原四项准入 | >1%目标 |
| --- | ---: | ---: | ---: | --- | --- |
| v3_e1 | 0.702906 | 0.700743 | -0.308% | False | False |
| v3_e2 | 0.702906 | 0.708698 | +0.824% | True | False |

完整指标见 [summary.json](summary.json)，原始分数及准入判定保持原样。

本分支从 V2 epoch2 训练，当前 V3 选用本分支 epoch2；epoch1 实体权重已删除。表中 Parent 按原训练分支记录，当前选定 V2@1 与 V3@2 的比较见 [Max 索引](../../README.md)。
