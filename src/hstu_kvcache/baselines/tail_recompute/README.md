# TR：尾部 Token 重算

日期：2026-09-16。状态：基本重算原语已实现，CPU 数值检查通过；正式连续质量评价未开展。共同口径见 [上级说明](../README.md)。

## 当前接口

实现位于 [core.py](core.py)：

```python
from hstu_kvcache.baselines.tail_recompute import recompute_tail

updated = recompute_tail(current, cache, item_ids, behaviors, time_deltas, n=128)
```

输入是与缓存逐位置对齐的原始事件，无 padding，batch 内等长；不要重置尾部首 token 的时间差。函数返回更新缓存，不原地修改输入；`n=0` 直接返回原对象。它复用模型确定性的 eval／no-grad 重放路径，正常调用后恢复模型原来的 training 状态。

第一版不做 n 的自动搜索、版本准入或实验编排；下文相应段落仍是后续比较设计。

## 方案定义

每次获准发布时，对用户当前保留的 N 个历史事件，取 `m=min(n,N)`，保留前 `N-m` 个事件的全部旧 K/V，用 Current 权重重放最后 m 个事件，更新它们在全部六层的 K/V。n 是发布级配置，不根据测试用户标签挑位置。

这是一种 recency 策略：假设靠近当前请求的历史更值得更新。它不做 token 重要性预测、不改变历史，也不从 Current Full 中抽出尾部 K/V 拼进去。该假设是否带来质量／成本收益由后续实验回答。

首版每个发布只刷新一次，之后正常追加和淘汰；“每次请求前再刷新 n 个 token”是不同的成本语义，不放进同一个结果行。

## 执行过程

1. 将 K/V 截断到前 `N-m` 个位置，得到旧前缀。
2. 从共同因果历史取得尾部原始 item、behavior 和 time_delta；保持原始顺序、绝对事件对应及尾部首 token 的时间差。
3. 用 Current 在旧前缀条件下重放尾部。尾部内的 token 按原模型因果 mask 读取旧前缀和允许读取的尾部位置。
4. 保留原前缀，安装重算尾部。长度仍为 N，原始 producer 与本次重算目标分别记录。
5. 后续 query 只读缓存；新增真实事件在本分支缓存上写入，滚动窗口遵守逐 token 先淘汰再追加的顺序。

现有 [hybrid_tail_refresh](../../models/state_transition.py) 实现第 1–4 步的张量核心；当前 `recompute_tail` 是其薄封装，不另写模型前向。`n=0` 在 baseline 接口定义为 Reuse，因为底层函数只接收正 width。`n>=N` 没有旧前缀，与同一保留原始历史的 Current Full 重算对齐。版本来源元数据由后续调用方维护。

## “重算”不意味着尾部完全精确

尾部第 0 层 K/V 来自 Current 的原始事件投影；上层 hidden 还会读取旧前缀，所以通常仍带旧状态误差。正确名称是“旧前缀条件下的 Current 尾部重放”，不能声称拿到了 Current Full 的精确尾部。

若前缀是同一模型对相同静态历史计算的正确缓存，尾部重放应与 Full 一致。经历滚动淘汰的持久前缀可能包含更早上下文影响，不能把这个条件扩展成“只要模型相同就总与当前窗口 Full 相同”。

## n 的选择与成本

后续最小开发可比较 `n∈{32,128,256}`，并设 `0` 和 `N` 两个端点控制；这些是探针候选，不是已冻结的正式实验配置。先测一个小 n 的执行和资源，再决定是否需要完整开发表。各用户按 `min(n,N)` 执行，不能删除短历史用户以改善结果。

在独立的发布前 development UID 上选择 n，记录质量与成本；最终评价冻结。首版不引入按用户动态调整 n 的预测器。

对 inclusive causal attention，令 `p=N-m`，每层有效注意力对数为：

\[
m p + \frac{m(m+1)}{2}.
\]

六层重算 token-layer 数为 `6m`，但不能据此断言耗时就是 Full 的 `m/N`：尾部仍读取整个旧前缀，而且稠密实现通常会计算被 mask 的位置。上述是有效依赖计数，不是实际矩阵乘法 FLOPs；实际算子尺寸、原始事件读取、KV 搬运及拼接复制均要记录。其他 diagonal 设置以实际 checkpoint 为准。

原语返回拼接后的完整缓存，可能复制未变前缀；首版先照实计费，不把理论上的“只写尾部”当成现实现的全部内存开销。后续若改用仅返回新 K/V 的接口，再验证和计量实际节省。

## 后续职责与最小检查

目录内只需 `tail_refresh` 薄封装和 n 的配置／开发选择；共同 runner 负责历史、发布、指标及成本记录，不需要专用训练器。

- 复用 [test_state_transition.py](../../../../tests/test_state_transition.py) 的同模型尾部／Full 对照，以及 [时间因果测试](../../../../tests/test_yambda_incremental_time_contract.py)。
- 检查 `n=0`、`n>=N`、未选前缀不变、尾部首个 time_delta 未重置。
- 在同一实际状态上，批量尾部重放与逐事件参考比较；若切换到滚动追加场景，使用相同的先淘汰规则。
- 两次发布之间实际 append／evict，第二轮输入必须是第一轮已处理的缓存；不需要期望它等于 Current Full。

本方案无需拟合参数，但后续 n 的开发选择和 Full 教师仍有成本。当前 [CPU 测试](../../../../tests/test_baseline_tail_recompute.py) 覆盖尾部／逐事件参考、端点和两次迁移间的 append／evict；这些检查不提供推荐质量或性能收益结论。
