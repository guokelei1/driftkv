# 缓存重算原语

当前Motivation包含按层、尾部、早期层偏差引导和当前query引导四种重算。
它们是方法思想的HSTU适配，不宣称逐项复刻原系统。
全部15条边的预算曲线已完成，见[实验与保留范围](../../../scripts/selective_recompute_2026_09/README.md)。

| 目录 | 接口与用途 |
| --- | --- |
| [layer_recompute/](layer_recompute/README.md) | `capture_state`、`recompute_interval`及连续层区间profiling |
| [tail_recompute/](tail_recompute/README.md) | `recompute_tail`在继承前缀条件下重放最近真实事件 |
| [deviation_recompute/](deviation_recompute/README.md) | 根据第一层K/V变化选择位置，再经当前各层重放 |
| [query_recompute/](query_recompute/README.md) | 根据当前query的第一层attention选择位置，再重放 |

共享任意位置重放在`sparse_recompute.py`；LR方法来源在其`paper_design.md`。
旧K/V翻译分支及专用说明、测试已退休，历史结果保留在结果目录。

每个请求从同一滚动Reuse状态出发，临时重算后丢弃修复；原生新事件继续追加，
候选query不进入持久历史。LR保存真实层入口，token重算使用与缓存对齐的真实事件，
读取未选位置的继承K/V。局部重算不自动等于Full对应位置，费用包括选择、校准与重放。

数值、因果时间流、用户面板、计算量和同请求聚合检查位于`tests/selective_recompute/`。
共享模型与缓存操作见[models/](../models/README.md)。`HSTUKVCache`布局为
`[layers,batch,sequence,kv_width]`，额外事件、producer和层入口由具体方法维护。
论文方法见[设计入口](../../../docs/paper_design.md)。
