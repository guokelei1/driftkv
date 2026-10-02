# TR：尾部 token 重算

当前四基线实验已完成全部 15 条边，见
[完整结果](../../../../results/selective_recompute_2026_09/README.md)。
每个请求选择最近的一段真实历史，在继承前缀条件下通过全部 Current 层重放，
临时修复只用于当前评分，随后丢弃。该实验支持当前 6、10、16 层模型。

## 选择与执行

选择理由是 recency：优先更新离当前请求最近的事件。
本方法不训练 token 选择器，也不使用评价标签或 Full K/V 选择位置。
当前预算是历史长度的 `1/32,1/16,1/8,1/4,1/2`；
取整规则与实际重算数量由共同评测器记录，所有预算从同一 rolling Reuse 状态出发。

对 N 个保留事件，实际重算 m=min(n,N)：

1. 保留前 N−m 个事件的全部继承 K/V。
2. 读取与缓存对齐的尾部 item、behavior 和 time_delta，保持内部原始时间间隔。
3. Current 逐层重放尾部。因果 attention 读取继承前缀和允许读取的新尾部位置。
4. 拼接未变前缀与新尾部，供完整 Current query 读取；不将这份修复缓存写回服务状态。

尾部首 token 的时间差不能独立重置；共同评测器对完整保留窗口首项采用 delta=0，
再截取尾部。候选 query 不成为历史，真实事件在原始服务状态上继续追加和淘汰。

## 接口

[core.py](core.py) 是
[hybrid_tail_refresh](../../models/state_transition.py) 的薄封装：

```python
from hstu_kvcache.baselines.tail_recompute import recompute_tail

repaired = recompute_tail(
    current, cache, item_ids, behaviors, time_deltas, n=128)
```

输入无 padding、batch 内等长，原始事件与缓存逐位置对应。
函数不原地修改输入；n=0 返回原缓存，n≥N 重算完整历史。
执行使用确定性的 eval/no-grad 路径，随后恢复原 training 状态。

第 0 层尾部 K/V 来自 Current 原始事件投影；上层仍读取继承前缀，
因此局部尾部通常不等于 Current Full 的精确尾部。
没有继承前缀的全历史端点才与同一对齐输入的 Full 重算一致。
滚动前缀还可能含已淘汰上下文，不能把“同模型”单独当作与窗口 Full 一致的条件。

## 计算量与检查

令 p=N−m，inclusive causal attention 每层有效注意力依赖数为
`m×p + m×(m+1)/2`。这不是实际稠密矩阵 FLOPs；
完整账本按执行形状计算，含尾部对整个继承前缀的读取。
原语返回完整拼接缓存，可能复制前缀，不能用理想的仅写尾部字节数代替实际内存代价。

[原语测试](../../../../tests/selective_recompute/tail_recompute/test_baseline_tail_recompute.py)
检查逐事件参考、端点、未选前缀和 append/evict；
[状态迁移参考](../../../../tests/test_state_transition.py)与
[时间因果检查](../../../../tests/test_yambda_incremental_time_contract.py)覆盖共同基础。
这些数值检查与完成的单边请求级实验不建立连续多发布适配有效性。
