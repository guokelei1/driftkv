# Query 引导重算

本目录实现本次四 baseline 诊断中的 HSTU 适配，不声称复刻 ProphetKV。

- 使用当前真实请求的 transient query，在 Current 第 0 层投影 Q，并与继承的第 0 层 K 做 attention 打分。
- 保留模型真实 scale、位置偏置和 pointwise activation。HSTU 没有 softmax；按 activated attention 的绝对值排序，跨 head 求均值。当前评测每行一个候选；API 若传多个候选，则再跨候选平均。
- 固定选择分数最高的 n 个历史位置，再按历史顺序通过全部当前层重放；没有 Current Full K/V 或标签参与选择。
- 未选中位置保留继承 K/V。选中位置的上层输入由本次下层执行产生，attention 因果读取混合状态。
- 各预算从同一滚动 Reuse 状态开始，修复仅服务当前 query，不写回持续状态。零预算直接 Reuse；全历史预算直接 Full 重算，省略不必要的选择。

`core.py` 提供 `query_attention_scores(...)` 和 `recompute_query(..., n=...)`。公共重放执行器在上级 `sparse_recompute.py`；支持无 padding 等长 batch。query 自身不成为持久历史。

选择额外执行 query embedding、norm、Q 投影、QK 和 importance 归约。此后修复缓存上的完整 query scoring 仍需执行，不能把两次 query 工作重叠扣掉。选中历史重算、排序比较及矩阵实际形状分别计入共同成本记录。

[测试](../../../../tests/selective_recompute/query_recompute/test_query_recompute.py) 使用原生 query 前向实际捕获的 Q 核对选择分数，并验证选择后的重放及端点；[共同数值参考](../../../../tests/selective_recompute/test_sparse_replay.py) 验证真实因果依赖和未选中状态不变。
