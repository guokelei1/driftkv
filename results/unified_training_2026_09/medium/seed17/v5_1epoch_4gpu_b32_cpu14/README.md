# Medium v4 → v5：1 epoch 结果

开发实验；固定 v4，训练窗口 [273,287)。

| 窗口 | Parent AUC | v5 AUC | 相对提升 | 提升百分点 |
| --- | ---: | ---: | ---: | ---: |
| E14 | 0.660402 | 0.682083 | +3.283% | +2.168 |

E14 原准入指标通过：True；相对 AUC 提升 >1%：True。

完整指标、请求数和判定见 summary.json；原始打分、seal 与逐窗口 adjudication 在 full_only/。
