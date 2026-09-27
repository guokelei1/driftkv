# 偏差引导重算

本目录实现本次四 baseline 诊断中的 HSTU 适配，不声称复刻 CacheBlend。

- 固定用第 0 层作为 early layer。以 Current 的 embedding、norm 和 K/V 投影处理当前保留的全部原始事件。
- 每个位置的分数为 `mean((K_current-K_reused)^2) + mean((V_current-V_reused)^2)`；不读取标签，不计算上层 Full K/V。
- 按分数选前 n 个位置，分数相同时选择较早的位置。重放前恢复时间顺序。
- 选中位置在所有当前层重新执行；每层 attention 读取该层新生成的选中 K/V 和原样保留的未选中 K/V。上层隐藏输入来自本次选中位置的下层执行，不能用免费 Full hidden 填补。
- 各请求、预算从同一滚动 Reuse 状态开始；修复仅服务本次 query，随后丢弃。零预算返回原缓存；全历史预算直接 Full 重算，省略不必要的选择。

`core.py` 提供 `layer0_deviation_scores(...)` 和 `recompute_deviation(..., n=...)`；任意位置重放复用上级 `sparse_recompute.py`。输入为无 padding 等长 batch，原始历史与 K/V 位置对应。正式实验的请求窗口、首项时间差和成本分摊由共同评测程序记录。

成本包括全历史 embedding、norm、K/V 投影、差值与归约，以及选中位置的实际重放。排序比较次数单列，不冒充浮点 FLOPs。重放的稠密 attention 会计算随后被 mask 的未来位置；按实际矩阵尺寸计费。query chunk 默认 64，仅控制 attention 临时空间，不改变选中的位置或数学结果。

[测试](../../../../tests/selective_recompute/deviation_recompute/test_deviation_recompute.py) 检查 early-layer 分数、没有执行上层块、选择与执行衔接、两端不产生选择开销；[共同数值参考](../../../../tests/selective_recompute/test_sparse_replay.py) 将批量任意位置重放与原生逐事件执行比较。
