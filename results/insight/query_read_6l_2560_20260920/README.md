# 六层 Yambda：256 校准／2560 评价的共享读取修正

2026-09-20 完成。固定共享规则 `b+Aq+Tr` 在两条预选边上分别恢复 **8.05%／17.90%**
的逐用户概率差距，2164／2154 名用户优于 Reuse。去掉直接历史响应后的 `b+Aq`
主指标均为负。结果支持当前用户响应参与的跨用户共享修正；全部负尾、pilot 和
canary 结果在下文与原始记录中保留。

## 设置与冻结过程

- [固定配置](../../../configs/insight/query_read_6l_2560_20260920.json)：当前 consolidated
  Medium 的 V1→V2、V2→V3，cutover day245／259；seed17、6L/H192/6 heads、
  context1024，实际为 legacy ELU+1。两条边均已有 Full-only 准入，权重与映射 hash 经核验。
- 256 校准、128 pilot、2560 评价 UID，三组互斥。评价组从既有 development 原顺序中
  排除 pilot、校准和两个保留确认组，再筛 `n_theta0>=1024` 后取前2560；合格候选共3970。
  所有评价用户在 day217 前已满1024历史。这是既有开发人口中的扩大评价，
  本次未参与校准或 pilot；final／confirmation 未读。
- 每个 cutover 使用严格早于该时刻的1024条历史。各组分别以完整组历史构造固定候选库，
  按 recent16、old-only16、发布前 popularity bank 补足64个候选；不用未来标签。
  校准每人取索引 `0,4,...,60` 的16个候选，评价取32个奇数索引候选。

`q` 是 Current 每层的实际读取查询，`r` 是本层读取用户旧缓存所得的历史响应。
两组为共享 `b+Aq` 与 `b+Aq+Tr`，各用相同256×16预算逐层拟合，FP64 ridge=.01；
归一化及参数只来自校准组。两个分支各自已修正的下层产生下一层 query，教师与源响应
在该分支相同 incoming query 下配对。实现以 `r/1024` 拟合响应率，回注入时乘回1024。

高层 `q` 已承接用户历史，两组都执行完整原生历史读取。比较 `b+Aq+Tr` 与 Reuse
回答共享规则能否迁移；比较它与 `b+Aq` 回答直接加入本层用户历史响应的贡献。
本组不运行每用户 oracle，也不输入摘要。

128人 pilot 只观察一次后，扩大评价原样加载 pilot 保存的256人校准参数。
两边规则 SHA256 分别为
`c6777131c8e0dc8955e1128716eb7fa5a22092b3ff43988cb9acd596d627910b`、
`ee487b7b5bc9f8e6a9859c28a0ddaa2bd37eebbcc3de23cb9336f7e67d05b142`。
执行器核验配置、源码、模型绑定和规则 hash；没有根据 pilot 调参数，也没有在2560人上重拟合。
扩大阶段新增校准教师人数为0，原256人教师成本仍计入总校准费用。

## 2560 用户全部结果

逐用户 `e_m` 是32候选上相对 Exact 的概率绝对误差均值。
主指标 `U=mean_UID(1-e_m/e_Reuse)`；辅助 `P=1-mean_UID(e_m)/mean_UID(e_Reuse)`。
下表 U/P 均为百分比，不裁剪；改善人数统计 `e_m<e_Reuse`。

| 版本边 | 路径 | U (%) | P (%) | 概率 MAE | 优于 Reuse |
| --- | --- | ---: | ---: | ---: | ---: |
| V1→V2 | Reuse | 0.00 | 0.00 | 0.009520489 | 0/2560 |
| V1→V2 | Exact | 100.00 | 100.00 | 0 | 2560/2560 |
| V1→V2 | `b+Aq` | -818.36 | -148.73 | 0.023679896 | 1034/2560 |
| V1→V2 | `b+Aq+Tr` | 8.05 | 65.54 | 0.003280634 | 2164/2560 |
| V2→V3 | Reuse | 0.00 | 0.00 | 0.006848068 | 0/2560 |
| V2→V3 | Exact | 100.00 | 100.00 | 0 | 2560/2560 |
| V2→V3 | `b+Aq` | -200.01 | -3.56 | 0.007091556 | 1270/2560 |
| V2→V3 | `b+Aq+Tr` | 17.90 | 72.85 | 0.001859288 | 2154/2560 |

加入直接用户响应后，相对 `b+Aq` 的平均概率误差下降86.15%／73.78%，
2247／2216个用户的误差更低。相对 Reuse，`b+Aq+Tr` 仍有396／406人不改善。

| 版本边／规则 | UID 恢复率中位数 (%) | p10 (%) | 最低 (%) |
| --- | ---: | ---: | ---: |
| V1→V2 `b+Aq` | -23.70 | -676.51 | -166447.94 |
| V1→V2 `b+Aq+Tr` | 74.23 | -57.56 | -5582.21 |
| V2→V3 `b+Aq` | -3.28 | -475.83 | -20122.32 |
| V2→V3 `b+Aq+Tr` | 75.70 | -60.81 | -9652.86 |

最小逐用户 Reuse MAE 为 `2.58450e-5`／`5.25777e-5`，未触发冻结的近零分母停止条件。
小分母会放大负恢复尾部，因此同时保留绝对误差、U、P与全部逐用户结果。
这是单训练 seed 的概率保真度诊断，不是推荐 AUC、摘要必要性或连续缓存寿命的测量。

## 保留的 pilot 与 canary

128-user pilot 不与上述2560评价用户重叠。两条规则均按原设置进入扩大评价。

| 版本边／规则 | U (%) | P (%) | 概率 MAE | 优于 Reuse |
| --- | ---: | ---: | ---: | ---: |
| V1→V2 `b+Aq` | -228.66 | -67.97 | 0.015657915 | 56/128 |
| V1→V2 `b+Aq+Tr` | 12.17 | 69.13 | 0.002878026 | 105/128 |
| V2→V3 `b+Aq` | -140.02 | 20.47 | 0.006135152 | 75/128 |
| V2→V3 `b+Aq+Tr` | 34.71 | 80.08 | 0.001536809 | 111/128 |

32校准／8评价的 V1→V2 canary 质量结果也保留：`b+Aq` U=-997.68%、P=-482.24%、
MAE=0.029437294、0/8改善；`b+Aq+Tr` U=-298.83%、P=-42.21%、
MAE=0.007189815、6/8改善。canary 核准的是数值、数据与资源执行，并未通过质量筛选。

## 检查、成本与运行资源

[canary](canary/canary.pass.json)的 FP64 正规方程残差为6.32e-16量级，
FP32 callback 与参考计算通过既定容差；真实 native／Exact replacement 端点检查通过。
完整2560评价两边端点最大误差不超过2.38419e-7。
单 A40、FP32读取／FP64求解、TF32关闭、Torch backend，求解器单CPU线程。

[成本记录](analysis/cost.md)按实现形状估算标量运算，包含 pilot 的一次校准。
普通32候选读取为228.530 M/user；依赖闭合 cache-only 重建为4.1971 G/user。

| 规则 | 每发布校准 K | 32候选每用户额外修正 | 额外修正／普通读取 | 本次2560用户含K／重建 |
| --- | ---: | ---: | ---: | ---: |
| `b+Aq` | 3.9593 T | 2.658 M | 1.163% | 36.913% |
| `b+Aq+Tr` | 3.9723 T | 16.961 M | 7.422% | 37.374% |

最后一列为 `(K+2560*C_extra)/(2560*C_rebuild)`，每人读取一次32候选。
它计兼容性新增算术，不是端到端时间比；普通原生读取仍需执行，多次请求继续累计修正。
共享服务不用评价用户教师；评价中的 Exact 仅用于测量。完整分项和近似见
[cost.json](analysis/cost.json)。

初始预算10分钟、约10GiB；canary 后的保留估计约210秒，pilot峰值8.74GiB。
扩大运行实测读史268.62秒，两边执行30.95／30.37秒，三项合计329.94秒；
超过canary推算，仍在初始10分钟预算内。流式评价峰值2.72GiB。
这些是研究流程开销，不作为服务速度比较。

## 证据与复现入口

- [diagnostic/summary.json](diagnostic/summary.json)、[metrics.csv](diagnostic/metrics.csv)、
  [per_user.csv](diagnostic/per_user.csv)：全部汇总、未裁剪逐用户结果与计时。
- [configuration.json](diagnostic/configuration.json)：固定输入、执行设置及依赖 hash。
  两份 NPZ 保存 UID、候选、query deltas 与 Reuse／Exact／两种共享规则的原始 logits。
- [pilot/summary.json](pilot/summary.json)及两份 `*_rules.pt`：原始校准及128人结果；
  [canary/summary.json](canary/summary.json)保留32/8探针全部结果。
- [run_query_read_probe.py](../../../scripts/design/run_query_read_probe.py)执行实验；
  [query_read_probe.py](../../../scripts/design/query_read_probe.py)实现共享修正。
  扩大评价使用 `--calibration-from .../pilot`，与pilot相同配置并另给新输出目录。
- [shared_read_insight.py](../../../figures/src/shared_read_insight.py)只读原始结果重算U/P并生成
  [论文表](../../../figures/tables/insight_shared_read.tex)；
  [analyze_query_read_probe.py](../../../scripts/design/analyze_query_read_probe.py)只读重建成本分解。

[旧128人 item／response 实验](../user_information_6l_20260920/README.md)及
[摘要实验](../shared_read_6l_20260920/README.md)完整保留。本次按用户修改后的问题与规模
固定新诊断，不在这些实验之间按结果择优。当前完整 Design 的实现和验证仍是后续工作。
