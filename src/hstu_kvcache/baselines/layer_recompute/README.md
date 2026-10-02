# LR：按连续层区间重算

当前四基线实验已完成 Medium/Large/Max 的全部 15 条边。
本方法在每个请求上临时重算选定连续层区间，其余层复用原 K/V，评分后丢弃修复。
[完整结果与成本](../../../../results/selective_recompute_2026_09/README.md)
记录实际选层与执行；本原语本身不负责模型发布准入或数据划分。

## 选择什么、为什么

每条边预留 32 名独立校准用户，每用户 16 个固定、无标签的目录候选。
为各个连续层区间实际执行重算，再比较 Current 读取与 Full 教师的 logit MSE。
同一重算层数选择误差最小的区间，冻结后用于该边全部评价用户。
Teacher 获取、所有候选执行和查询的工作量均计入每条独立预算曲线。

预算层数为去重后的 `ceil(L×{1/16,1/8,1/4,1/2,3/4})`，不包含完整层栈。
实现支持当前 6、10、16 层；六层的 21 个非空区间只是一个例子。
独立校准与实际选择逻辑在
[calibrate.py](../../../../scripts/selective_recompute_2026_09/calibrate.py)。

## 原语与入口状态

实现位于 [core.py](core.py)，接收无 padding、batch 内等长的对齐历史：

```python
from hstu_kvcache.baselines import layer_recompute as lr

state = lr.capture_state(parent, item_ids, behaviors, time_deltas)
repaired = lr.recompute_interval(
    current, state, item_ids, behaviors, time_deltas, (2, 3))
```

`LayerRecomputeState` 保存 K/V 及各层真实的 norm 前 hidden `E_l`，
布局为 `[B,N,hidden]`。K/V 本身不能免费还原这些入口。
`capture_state` 和 `append` 通过临时 norm pre-hook 捕获入口；
`retain_latest` 同步淘汰相应状态。变换不原地修改输入，未变入口可共享只读存储。
模型执行时进入 eval/no-grad，返回时恢复原 training 状态。

当前实验随共同 Parent 构建及 Current 原生 append 保存所有可能的非零层入口。
当 hidden 与 kv_width 相等、精度相同，附加入口张量相对双份 K/V 为
`(L−1)/(2L)`：六层约 41.7%。预先固定单个入口会有不同存储要求，
不能用其约 8.3% 代替本实验六层自由选区间的状态量。
原语维护不等于已实现连续发布的策略与计费。

## 区间执行

1. 起点 a=0 时用 Current embedding 处理保留事件；a>0 时使用实际保存的 `E_a`，
   不读取教师 hidden。
2. 顺序执行 Current 的 a..b 层，每层处理完整保留历史，保留模型原生因果 mask、
   norm、attention、gate 和 residual。
3. 只替换区间内的 K/V 和实际产生它们的入口。区间外 K/V、入口保持原值，
   不能用 block b 的输出覆写未执行的 block b+1 入口。
4. 当前候选 query 仍经过全部 Current 层，读取临时混合状态。

`a>0` 的入口可能包含旧模型误差；“Current 执行过”不等于 Full 对应 K/V。
完整区间从 Current embedding 开始时，与同一对齐历史的 Full 一致。
经历淘汰后的旧入口可能带更早上下文，也不能视为当前窗口的重新编码。
实际最后一层的完整 block 工作照实计费，未按尚未实现的只投影 K/V 优化扣费。

`enumerate_intervals` 与 `profile_intervals` 提供同步候选执行和评分回调；
预算、UID 划分和完整成本属于实验调用方。
`forward_stale_kv` 仍执行全长 attention，不能代替跳过复用层的执行器。

## 方法来源与检查

借鉴 DroidSpeak 的“连续区间选择＋保存入口状态＋部分执行”，
[来源笔记](paper_design.md)记录原论文。这里是 HSTU 推荐任务适配，
不包含原系统的 vLLM、跨 GPU 通信或动态调度。

[数值测试](../../../../tests/selective_recompute/layer_recompute/test_baseline_layer_recompute.py)
覆盖入口捕获、完整/局部参考、append/evict 后更换起点及评分隔离。
共同滚动测试检查临时修复不进入后续持久状态。旧发布时一次性修复设想
不描述当前请求级实验，也不构成连续多发布质量验证。
