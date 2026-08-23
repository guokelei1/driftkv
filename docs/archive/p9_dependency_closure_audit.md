# Archived: P9 Dependency-Closure Audit

更新日期：2026-08-21。

本审计依据当前 HSTU 实现的 `embed_inputs → block.norm → attention K/V projection → causal attention → next layer hidden` 数据流，判断 diagnostic splice 是否可由生产时持有的 raw history 与 parent K/V 执行。

## 结论表

| 候选操作 | 所需输入 | 依赖闭包 | 当前角色 |
| --- | --- | --- | --- |
| No-op | parent K/V | 完整 | executable baseline |
| Layer-0 exact segment projection | 该段 raw item/action/time，加首 token 的前驱 timestamp | 完整 | 首选 partial candidate |
| Layer-0 exact full projection | 全 prefix raw events | 完整 | partial candidate |
| 任意 layer>0 exact segment splice | current lower-layer hidden；其又依赖更早 causal prefix | 不完整 | diagnostic only |
| All-layer exact middle/tail splice | 每层 current hidden boundary；KV-only state 不包含它 | 不完整 | diagnostic only |
| Hybrid tail replay | 截断到 tail 前的 parent K/V + raw tail；用 current model incremental replay | 完整，但语义不同于 exact splice | executable candidate |
| Prefix rectangle replay | raw prefix `[0,b)`，逐层 current replay | 完整 | executable，但可能接近 Exact 成本 |
| Exact All | 全 raw prefix | 完整 | executable reference |

## 为什么 layer 0 是特殊情况

Layer-0 的 K/V 是当前位置 input embedding 经 normalization 和 current K/V projection 的结果。K/V projection 本身不读取其他历史位置；relative-position bias 只进入 attention aggregation，不进入 K/V。因此任意 layer-0 segment 可以从对应 raw events 独立重建，并与 Current Full 的 layer-0 K/V 精确一致。

实现必须保留两个边界：

- temporal delta 仍相对原历史前一事件计算，不能把 segment 首 token 的 delta 重置为零；
- item/behavior/time embedding、normalization 和 K/V projection全部来自 current model。

## 为什么上层孤立 splice 不合法

Layer `l>0` 的 K/V 来自 layer `l-1` 的 hidden。该 hidden 已经过 causal attention，因此位置 `j` 依赖 `0..j` 的 lower-layer K/V。只持有 parent K/V 时，无法直接得到 Current Full 的 lower hidden boundary。

所以“从离线 Current Full 拿出 layer-2 middle K/V 并覆盖旧状态”是定位上界，不是系统动作。要精确得到它，闭包会向下层和更老 prefix 扩张；若把这些计算成本隐藏，frontier 将失真。

## 冻结的最小 executor 候选

下一步只实现并回放：

1. `No-op`；
2. `Layer0-Recent128`；
3. `Layer0-Middle`；
4. `Layer0-Full`；
5. `HybridTail-32`；
6. `HybridTail-128`；
7. `Exact-All`。

前三个 layer-0 action 用于验证诊断 exact-KV 能否被真实投影器逐元素复现；HybridTail 使用真实 incremental executor，不宣称等价于 diagnostic exact tail。

这些 action 是 development action-space，由 P9 tomography 导出，必须回放到完整 P8 cells 和全部 seed。最终方法仍需在未查看时间边 qualification。
