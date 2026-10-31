# RecFlow Max 训练与版本链交付

十个实际 Max checkpoint、原十九个窗口及追加九个时序窗口均已完成并核对。
历史上限1024，正式日更新和评测的实际历史全部1024；完整已记录排序候选，不额外抽样。
模型16层、宽320、10×32 heads，共1,154,235,993参数；其中1,140,971,840为物品embedding。

## 训练时间与数据量

“25分钟”是262144请求的一轮曝光预热，实测25.05分钟。
完整候选对齐为131072请求×2轮，实测52.82分钟；foundation两阶段合计1.30小时。
九次日更新合计843144请求，每日完整一轮约19–21分钟。
各训练阶段合计约4.27小时；含小模型、验证、两组评测和追加准备的连续预算计时为
7.50小时，低于10小时。以上为四张A40并行的墙钟时间，不是GPU小时数。
阶段计时的细节与全部SHA见[模型清单](reports/model_chain_inventory.json)。

参数规模主要来自embedding查表，候选共享一次历史编码；不能把这个模型按同参数量的稠密语言模型估算。
这次使用受控训练量，以原生验证选择checkpoint；未证明已经收敛或全量数据只需25分钟。

## 所有更新边

下表差值均为 **Rebuild−Reuse 的 observed-positive NDCG@10**，正数表示复用受损。
三个用户分组独立列出，不合并算显著性。主指标始终NDCG@10，Recall@10及@5/@20保留在逐窗文件中。

| 更新边 | 原开发差值 | 未见用户确认差值 | 训练分组时序差值 | 时序 UID 95% 区间 | 时序相对损失 |
| --- | ---: | ---: | ---: | --- | ---: |
| M0→M1 | +0.004207 | +0.006245 | +0.005473 | [+0.004019, +0.006974] | +13.89% |
| M1→M2 | +0.000155 | +0.000865 | +0.000617 | [+0.000030, +0.001178] | +1.51% |
| M2→M3 | +0.001047 | +0.000530 | +0.000840 | [+0.000290, +0.001433] | +2.04% |
| M3→M4 | -0.000668 | +0.000023 | -0.000523 | [-0.001047, -0.000018] | -1.29% |
| M4→M5 | +0.001441 | +0.001678 | +0.001421 | [+0.000840, +0.001983] | +3.60% |
| M5→M6 | -0.000233 | -0.000413 | -0.000827 | [-0.001276, -0.000374] | -2.05% |
| M6→M7 | -0.000269 | +0.000043 | +0.000803 | [+0.000328, +0.001288] | +1.97% |
| M7→M8 | +0.000794 | +0.000804 | +0.000959 | [+0.000372, +0.001552] | +2.20% |
| M8→M9 | -0.000352 | +0.000576 | -0.000101 | [-0.000558, +0.000338] | -0.24% |

原未见用户确认只有M0→M1和M4→M5两条名义区间排零；原开发三条。
额外D37的M8→M9差值−0.00069964、区间跨零，完整保留在[全部28窗CSV](reports/version_chain_all_populations.csv)。
追加组中，复用下降的名义区间排零版本为[1, 2, 3, 5, 7, 8]；反向版本为[4, 6]；区间跨零版本为[9]。
第二条时序边区间下界很接近零，第四条反向边上界也很接近零，不能夸大。

## 追加时序组的含义

在原结果之后，按用户允许扩人口的授权，在D26原5197训练居民中排除384 fit与128 validation缓存校准UID，
再以固定UID哈希选2048人。同一名单用于全部九窗，370105完整请求，无曝光/无正例请求也保留计数。
每个目标checkpoint的实际训练截止日均严格早于对应评测日；人口不按未来活跃交集、标签或分数选择。
名单中2031人已出现于foundation训练目标；后续窗口少数活跃UID尚未被父版监督，均按固定名单保留。
因此这是**训练分组居民的未来请求时序留出，是原研究之后的追加分析**，不能改称独立未见用户确认。
原确认结果不回写。区间是名义配对UID95% bootstrap，不是多重检验后的整体显著性。

## 工程核对与后续使用

原链60项CPU检查、新人口/时间边界10项检查通过；真实Max四卡训练/完整AdamW恢复、最后不完整batch检查通过。
新评测先对原64真实请求逐位对照：Rebuild、parent native、Reuse三路分数最大差全为0，逐请求指标一致。
全部28窗的650,240个请求、256,803,319个候选实例已与原始候选ID/标签逐项核对并独立重算指标。
九个追加窗口的当前原生模型优于固定基线的数量为9/9。
正反结果、所有checkpoint、完整训练状态、原始分数和代码绑定均保留。

当前cache比较是**匹配同一因果前缀的版本快照**。持续在线写入/FIFO和论文具体校准方法仍属于后续实验，
本轮结果不证明长历史优于短历史，也不证明论文方法已有效。Rebuild不是质量的理论上界。

| 模型 | 本阶段训练日 | 每轮请求 | 轮数 | checkpoint |
| --- | --- | ---: | ---: | --- |
| M0 | D24, D25, D26 | 131072 | 2 | [step_004096.pt](/home/gkl/work/evokv/experiments/recflow_request_ranking/runs/max_budgeted/attempt_1/alignment/step_004096.pt) |
| M1 | D27 | 99931 | 1 | [step_001562.pt](/home/gkl/work/evokv/experiments/recflow_request_ranking/runs/max_budgeted/attempt_1/update_d27/step_001562.pt) |
| M2 | D28 | 94878 | 1 | [step_001483.pt](/home/gkl/work/evokv/experiments/recflow_request_ranking/runs/version_chain/attempt_1/training/v2/step_001483.pt) |
| M3 | D29 | 92693 | 1 | [step_001449.pt](/home/gkl/work/evokv/experiments/recflow_request_ranking/runs/version_chain/attempt_1/training/v3/step_001449.pt) |
| M4 | D30 | 97036 | 1 | [step_001517.pt](/home/gkl/work/evokv/experiments/recflow_request_ranking/runs/version_chain/attempt_1/training/v4/step_001517.pt) |
| M5 | D31 | 94945 | 1 | [step_001484.pt](/home/gkl/work/evokv/experiments/recflow_request_ranking/runs/version_chain/attempt_1/training/v5/step_001484.pt) |
| M6 | D32 | 90662 | 1 | [step_001417.pt](/home/gkl/work/evokv/experiments/recflow_request_ranking/runs/version_chain/attempt_1/training/v6/step_001417.pt) |
| M7 | D33 | 91231 | 1 | [step_001426.pt](/home/gkl/work/evokv/experiments/recflow_request_ranking/runs/version_chain/attempt_1/training/v7/step_001426.pt) |
| M8 | D34 | 92144 | 1 | [step_001440.pt](/home/gkl/work/evokv/experiments/recflow_request_ranking/runs/version_chain/attempt_1/training/v8/step_001440.pt) |
| M9 | D35 | 89624 | 1 | [step_001401.pt](/home/gkl/work/evokv/experiments/recflow_request_ranking/runs/version_chain/attempt_1/training/v9/step_001401.pt) |

M0另继承D12–23的预热；各后续模型继承实际父版的完整AdamW状态。模型清单的日期是当前训练阶段日期。
证据入口：[原链审计](runs/version_chain_verification/attempt_1/completion_audit.json)、
[追加组审计](runs/temporal_cohort/attempt_1/completion_audit.json)、[最终交付核对](reports/final_training_handoff.json)。
