# Large v3 → v4：2 epoch 结果

开发实验；固定 v3，训练窗口 [259,273)。

| 窗口 | Parent AUC | v4 AUC | 相对提升 | 提升百分点 |
| --- | ---: | ---: | ---: | ---: |
| E14 | 0.680587 | 0.689922 | +1.372% | +0.933 |

E14 原准入指标通过：True；相对 AUC 提升 >1%：True。

完整指标、请求数和判定见 summary.json；原始打分、seal 与逐窗口 adjudication 在 full_only/。
