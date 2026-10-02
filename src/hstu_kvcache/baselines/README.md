# 缓存重算与保留对比原语

## 当前四种Motivation基线

按层、尾部、早期层偏差引导、当前query引导重算分别位于
`layer_recompute/`、`tail_recompute/`、`deviation_recompute/`、`query_recompute/`。
共享任意位置重放在 `sparse_recompute.py`。它们是方法思想的HSTU适配，
不宣称逐项复刻原系统。

全部15条边的预算曲线已经完成，入口与结果见
[四种重算实验](../../../scripts/selective_recompute_2026_09/README.md)。
本轮每个请求从同一滚动Reuse状态出发，临时重算选中区域并复用其他K/V；
评分后丢弃临时修复。新事件按原生路径追加，候选query不进入持久历史。

## 保留的方法原语

| 目录 | 接口与用途 |
| --- | --- |
| [layer_recompute/](layer_recompute/README.md) | `capture_state`、`recompute_interval`及连续层区间profiling；保存真实入口hidden |
| [tail_recompute/](tail_recompute/README.md) | `recompute_tail`在继承前缀条件下重放最近的真实事件 |
| [deviation_recompute/](deviation_recompute/README.md) | 根据第一层K/V变化选择位置，再经当前各层重放 |
| [query_recompute/](query_recompute/README.md) | 根据当前query的第一层attention选择位置，再重放 |
| [kv_translate/](kv_translate/README.md) | 保留跨层、跨head的K/V ridge映射原语，供后续对比使用 |

LR与KT的方法来源记录仍在各自 `paper_design.md`。
K/V翻译原语不属于当前四条重算曲线；其有效性与成本须由对应实验说明。

原语不自动选择评价用户、接受模型发布或读取数据。LR使用真实保存的层入口；
token重算使用与缓存对齐的历史事件，并读取未选位置的继承K/V。
局部重算得到的K/V不自动等于Full对应位置；费用包括选择、校准与实际重放。

## 当前检查与历史接口

当前四基线的数值、因果时间流、用户面板、计算量和同请求聚合检查位于
`tests/selective_recompute/`。LR/TR 的原语测试也在该目录各方法子目录中；
KT 的基本公式与状态检查为 `tests/test_baseline_kv_translate.py`。

旧 `scripts/design/run_insight1_competitors.py` 功能探针及其专属测试已退休；
旧AUC、splice和oracle结果保留为历史证据，不作为当前执行入口。
现存Design接口见[脚本说明](../../../scripts/design/README.md)，
论文设计以[设计入口](../../../docs/paper_design.md)及正文为准。

共享模型、原生缓存与追加操作继续复用 [models/](../models/README.md)。
`HSTUKVCache` 布局为 `[layers,batch,sequence,kv_width]`，本身不包含原始事件、
producer或层入口hidden；所需附加状态由对应方法维护。
