# Medium V1 → V2：两 epoch 结果

开发实验；同一固定 V1，训练窗口 [231,245)。

| 窗口 | Parent AUC | V2 AUC | 相对提升 | 提升百分点 |
| --- | ---: | ---: | ---: | ---: |
| E14 | 0.639586 | 0.646865 | +1.138% | +0.728 |

E14 原准入指标通过：True；相对 AUC 提升 >1%：True。

完整指标、请求数和判定见 summary.json；原始打分、seal 与逐窗口 adjudication 在 full_only/。
