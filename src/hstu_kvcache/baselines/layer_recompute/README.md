# LR：DroidSpeak 启发的按层重算

日期：2026-09-16。状态：基本原理已实现并通过 CPU 数值检查，完整质量／成本评价未开展。共同执行与评价口径见 [上级说明](../README.md)。层编号使用 `0..5`。

[原论文核心 Design 笔记](paper_design.md) 单独保存 DroidSpeak 的方法；本文记录本仓库的 HSTU 适配设计。

## 当前接口

实现位于 [core.py](core.py)，当前接口同步执行、只接收无 padding 的等长 batch：

```python
from hstu_kvcache.baselines import layer_recompute as lr

state = lr.capture_state(parent, item_ids, behaviors, time_deltas)
updated = lr.recompute_interval(current, state, item_ids, behaviors, time_deltas, (2, 3))
# 滚动窗口先按每个新增事件需要腾出空间，再追加；不要整批追加后才截断。
updated = lr.retain_latest(updated, 1023)
updated = lr.append(current, updated, new_item_ids, new_behaviors, new_time_deltas)
```

`LayerRecomputeState` 保存 K/V 和按真实层号索引的 `boundary_inputs`。初始化和追加通过短期 norm pre-hook 捕获真实 E，完成后移除 hook。变换不原地修改输入；未变化的 E 可共享存储，调用方应按只读状态使用。模型执行时进入 eval／no-grad，随后恢复 training 状态。

`enumerate_intervals(6)` 给出空动作及 21 个区间；`profile_intervals(..., score=callback, intervals=...)` 对同一输入独立重算并返回分数和重算层数。调用方负责数据及最终区间选择，没有隐含教师查询或自动调参；层数不是完整成本计量。

当前 [CPU 测试](../../../../tests/test_baseline_layer_recompute.py) 覆盖真实 E 捕获、完整／局部参考、追加和淘汰后换起点、候选评分隔离。下文关于完整开发选择、成本和 producer 谱系的部分仍是后续实验设计。

## 论文中借鉴什么

来源：Yuhan Liu 等，*DroidSpeak: KV Cache Sharing Across Fine-tuned Model Variants*，NSDI 2026，pp.319–338。[正式论文](https://www.usenix.org/system/files/nsdi26-liu-yuhan.pdf)。

- §3.2、Fig.4–5：不同模型对的层敏感性不同，不能固定认为某几层最重要。
- §4.1、Fig.6–7：执行时重算连续层区间，复用区间外 KV；重算入口需要 sender 保存的层输入 E。连续区间减少从复用切到重算的边界。
- §4.2：离线 profile 候选区间，建立质量与重算代价的关系；§4.3–4.4 将配置交给实际存取和 partial-prefill 执行。

在 HSTU 中保留“区间选择＋入口状态＋部分执行”的结构。LLM 生成任务、vLLM 集成、跨 GPU 传输重叠和基于队列的动态调度不属于首版。以下具体策略是本仓库设计，不能写成论文原方法的逐项复刻。

## 首版选择策略

每次发布为所有用户选择一个连续区间 `[a,b]`，重算区间内所有历史位置，区间外 K/V 保留。六层只有 21 个非空连续区间，另设空区间 Reuse 控制；不先引入任意层子集、用户级预测器或多段搜索。

完整开发阶段的区间 profiler 职责如下；当前只提供上述执行＋评分回调原语：

1. 在独立、发布前开发历史上准备本分支真实输入状态及 Current Full 教师。
2. 对每个候选区间调用真正的区间执行器，再走 Current 完整 query 路径；不能以“替换 Full KV 的对应层”代替该执行。
3. 记录开发质量／输出偏差、实际 block 工作、边界状态与 I/O，形成质量—成本表。K/V 差异或单层敏感性可作解释，不单凭参数变化大小选层。
4. 在事先给定的预算下按开发指标选定区间，并冻结到该次发布。首个 CPU 探针只检查少数区间的数学与资源，不默认启动完整搜索；正式预算值、开发目标及平局规则随实验配置确定。

空区间配置是方法自己的“保留缓存”动作；模型拒绝发布则更早发生，所有方法都保留服务父版本。两者不能混淆。空区间只保证 K/V 行为与 Reuse 相同，LR 的附加入口维护仍须计费；全区间即使 K/V 与 Exact 相同，也要单列附加状态的成本。

## 为重算保存入口 hidden

令 `E_l` 是历史 token 进入 block l **在 norm 之前的完整 hidden**，布局 `[B,N,hidden]`。HSTU block 同时需要 norm 输入和 residual，仅存 norm 后向量不够，K/V 本身也不能免费还原该输入。

首版允许后续发布选择不同起点，因此从初始缓存构建及每次真实 native append 开始，捕获 `E_1..E_5` 五个可能入口。`a=0` 时直接由 Current 对原始历史执行 embedding，不保存旧 `E_0`。这些状态随对应 token 追加和淘汰，记录实际来源及继承误差。

这是本场景为跨发布重新选层付出的存储选择：若 hidden 和 kv_width 相等、精度一致，五个入口相对六层双份 K/V 增加 `5/(2×6) ≈ 41.7%` 的持久张量字节，不含元数据。单个入口只需约 8.3%，但那要求预先固定起点；不能一边按每次发布自由选入口，一边只给一个入口的成本。后续若研究固定起点版本，应单列其限制和节省。

已有只含 K/V 的线上状态不能直接补出这五个入口。后续开发可在共同初始 Parent 构建时捕获它们，并收取额外写入费；若从已有持久状态中途安装，重建所需历史／回放的成本必须另计。

## 区间执行

对发布前真实状态 `C`：

1. `a=0`：从相同的保留事件及其原始时间差得到 Current embedding；否则读取真实 `E_a`，不读取 target 教师 hidden。
2. 顺序执行 Current 的 block `a..b`，每层处理完整保留历史，保留原配置的因果 mask、norm、attention、gate 和 residual；得到该区间的新 K/V。
3. 替换且只替换 `[a,b]` 的 K/V。其他层直接复用，不在其上运行历史 attention。
4. 区间内的 `E_l` 更新为这次实际产生该层 K/V 的 block 输入；区间外的 `E_l` 与 K/V 一起保留。特别是即使算出了 block b 的输出，也不单独覆写未执行的 block b+1 的入口记录，以维持该记录与其实际缓存生产过程的对应。
5. 随后的候选 query 仍执行 Current 的全部六层，读取混合 K/V；native append 则在所有层生成新增 K/V 和相应入口记录。

首版直接调用现有完整 block，区间最后一层的输出计算也照实计费。后续可在确有收益时研究只投影最后一层 K/V 的优化，不能提前按尚未实现的优化计时。

`a>0` 时，重算所用 `E_a` 仍可能带旧版本及早期历史误差，因此“当前权重计算过”不等于 Current Full 的对应 K/V。`a=0,b=5` 使用 Current embedding 才能与相同保留历史的 Full 重算对齐。滚动淘汰后的旧 hidden 可能保留已淘汰上下文的影响，也不要求它等于重新编码当前窗口的 hidden。

## 模块边界与现有代码

| 计划职责 | 复用基础／需要补充 |
| --- | --- |
| 区间 profiler | 已有 `enumerate_intervals`／`profile_intervals`；数据与选择规则交给调用方 |
| 入口状态 | 已有 `capture_state`／`append`／`retain_latest`；完整版本来源记录待接入 |
| 区间执行器 | 已有 `recompute_interval`，复用 [HSTUBlock.forward](../../models/block.py) 和 [HSTUKVCache](../../models/kv_cache.py) |
| 工作与存储计量 | 当前只返回 profiler 的重算层数；完整入口读写、query／append 成本待计量 |

[forward_stale_kv](../../models/attention.py) 仍执行全长 attention，当前实现还会计算随后弃用的 K/V 投影；它可以作参考，但不能当成“跳过复用层”的低成本执行器。[project_exact_layer0_segment](../../models/state_transition.py) 仅解决第 0 层投影，不等于任意层选择已经实现。

## 最小验证与未决项

- 在未经过淘汰的同模型静态历史上，用真实捕获的 E 验证区间 K/V 与原结果一致；全区间与 Full 对齐，空区间保留缓存。
- 用真实跨版本输入检查执行器与显式逐 block 参考一致，未选层不变；不要求它与 target Full 一致。
- 一条小轨迹包含两次发布、区间起点改变、append、evict 和 query，确认所用入口确实存在、各层来源和对应事件一致。不能为了第二次发布而重新建干净 Parent 状态。

选择目标、预算、发布前开发样本量尚未冻结；这些在有实际开发信号后确定。论文对 LLM 模型对迁移的结果不能替代本方案的推荐质量或多发布连续行为证据。
