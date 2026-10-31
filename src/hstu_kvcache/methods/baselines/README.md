# 局部重算基线

这些是 Layer、Tail、Deviation 和 query-guided 重算的 HSTU 实现。

| 目录 / 文件 | 用途 |
| --- | --- |
| [layer_recompute](layer_recompute/README.md) | 真实层入口、连续层区间重算和 profiling |
| [tail_recompute](tail_recompute/README.md) | 在继承前缀条件下重放最近事件 |
| [deviation_recompute](deviation_recompute/README.md) | 按第一层 K/V 偏差选择位置 |
| [query_recompute](query_recompute/README.md) | 按当前 query 的注意力选择位置 |
| `sparse_recompute.py` | 任意位置的共享稀疏重放 |
| `serving.py` | query 评分 |
| `protocol/` | 人口、校准、执行与计算成本辅助 |

每个请求从共同滚动 Reuse 状态出发，临时重算后丢弃修复；原生新事件继续追加，
候选 query 不进入持久历史。Layer 保存真实层入口，token 重算使用与缓存对齐的真实事件。
Reuse / Rebuild 的公共轨迹在 [evaluation/trajectory.py](../../evaluation/trajectory.py)，
评测策略在 [evaluation/baselines.py](../../evaluation/baselines.py)。
KV Map 独立放在相邻的 [kv_map](../kv_map/)；Figure 9 的策略组合留在各实验目录。
