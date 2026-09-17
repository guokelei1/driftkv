# Large：本轮 V0–V5 工作链

**当前链尾暂定为 V5@1 epoch，按用户明确选择。** 10L/H320、10 heads、context1024、79,681用户，Yambda-500M，seed17。

统一权重入口：`seed17/checkpoints/v0..v5/checkpoint_100.pt`。[当前完整清单](seed17/checkpoints/chain.v0_v5.manifest.json)是本轮V0–V5的路径、哈希和选择状态入口。

| 版本 | Parent | 训练窗口（天） | Epoch | Global batch | 来源/状态 | Checkpoint |
| --- | --- | --- | ---: | ---: | --- | --- |
| V0 | — | [0,217) | 1 | 96 | 历史复用 | [权重](seed17/checkpoints/v0/checkpoint_100.pt) |
| V1 | v0 | [217,231) | 1 | 96 | 历史复用 | [权重](seed17/checkpoints/v1/checkpoint_100.pt) |
| V2 | v1 | [231,245) | 1 | 96 | 历史复用 | [权重](seed17/checkpoints/v2/checkpoint_100.pt) |
| V3 | v2 | [245,259) | 1 | 96 | 历史复用 | [权重](seed17/checkpoints/v3/checkpoint_100.pt) |
| V4 | v3 | [259,273) | 2 | 64 | 本轮重训，四项准入通过 | [权重](seed17/checkpoints/v4/checkpoint_100.pt) |
| V5 | v4 | [273,287) | 1 | 64 | 本轮重训，用户暂定 | [权重](seed17/checkpoints/v5/checkpoint_100.pt) |

## E14 父子 AUC

每行比较同一窗口中的parent/current；跨行绝对AUC不可直接比较。前三条边是历史结果，后两条为本轮重训结果。

| 版本边 | E14窗口 | Parent AUC | Current AUC | 相对提升 |
| --- | --- | ---: | ---: | ---: |
| v0 → v1 | [231,245) | 0.630066 | 0.651462 | +3.396% |
| v1 → v2 | [245,259) | 0.655420 | 0.668998 | +2.072% |
| v2 → v3 | [259,273) | 0.684236 | 0.703051 | +2.750% |
| v3 → v4 | [273,287) | 0.680587 | 0.689922 | +1.372% |
| v4 → v5 | [287,301) | 0.674620 | 0.683919 | +1.378% |

## V5 的暂定选择与证据范围

- 当前采用V5@1：相对AUC +1.378%，超过1%目标；但用户级平均loss改善的bootstrap置信区间下界未大于零，**原四项准入未全部通过**。这是用户暂定的工作链选择，原admission结果不修改，不标为自动准入或服务推广。
- V5的E14 [287,301)存在day300数据不完整的问题，只作为方向性开发结果。
- V5@2仍在[原运行目录](seed17/v5_epochs12_4gpu_b64_cpu14/README.md)保留：AUC=0.716664，相对+6.232%，四项准入通过；它是备选，不属于当前链尾。两个端点及所有结果均保留。

## 配置、来源和保存方式

- V0–V3为历史1 epoch/global96模型；V4为2 epochs/global64；V5采用连续2 epochs训练中的1 epoch端点/global64。均4卡，seed17。
- 本轮V4/V5沿用CPU14、fresh AdamW、LR=5e-5、weight decay=1e-4和用户等权BCE；V5两个epoch间不重置优化器。
- 六个模型的权重哈希、payload父哈希、训练窗口、epoch和配置已逐个核对。V4/V5实体权重移入统一目录，原路径保留相对符号链接；V5原多端点seal原样复制，选中条目为`v5_e1`。
- 原[四版本前缀清单](seed17/checkpoints/chain.manifest.json)被训练合同按哈希引用，保持原样；当前完整链另存，避免破坏旧合同。
- [V4训练评测](seed17/v4_2epoch_4gpu_b64_cpu14/README.md)及[V5双端点训练评测](seed17/v5_epochs12_4gpu_b64_cpu14/README.md)保留合同、配置、原始分数、seal和adjudication。其他历史模型不变。
