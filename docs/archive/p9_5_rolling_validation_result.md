# Archived: P9.5 Expanded Rolling-Lineage Validation

P9.5 将真实 release-cutover 物化、逐事件 append 和 append 前滚动淘汰扩展到全部 24 个 `release × model × seed` cells。每个 cell 在 label-free fidelity 与 quality companion 中各使用 40 个事前按 lineage strata 选择的请求，共 1,920 次请求评测。24/24 cells 均通过 Exact、R0 和数据计数门。

## Fidelity 结果

R0 的全部动作仍为精确零，说明真实 rolling executor 没有破坏 output-only negative control。

下表为先对每个 seed 的 JS 等权求均值、再相对 No-op 计算的恢复率：

| Release / Model | L0 Recent128 | L0 Middle | L0 Full | HybridTail128 |
|---|---:|---:|---:|---:|
| R1 edge1 / M0-F | 33.8% | 37.3% | 20.5% | 53.9% |
| R1 edge1 / M1 | 44.7% | 67.8% | 87.4% | 55.6% |
| R1 edge2 / M0-F | 13.7% | 22.7% | 37.2% | 38.4% |
| R1 edge2 / M1 | 60.3% | 78.3% | 97.9% | 60.3% |
| R2 / M0-F | 80.4% | 95.6% | 99.2% | 83.4% |
| R2 / M1 | 89.1% | 92.3% | 89.4% | 94.7% |

R1-edge1 M0-F 的 layer-0 actions 有一个 seed 为负恢复，而 HybridTail128 三 seed 均为正。M1-R2 的 layer-0 actions 同样存在一个负恢复 seed；其高聚合恢复主要来自风险更大的 seed。HybridTail128 在 M1-R2 三 seed 均为正。这些结果禁止“一个固定 partial action 适用于所有 release/seed”的表述，并支持先比较 release-level action，再判断是否需要 state-level scheduler。

## Lineage 稳健性

非 R0 cells 中，CurrentOnline-vs-ReuseOnline 的 S 与旧 request-local S 通常相差约 1%；CurrentOnline 与旧 CurrentFull 的差异则远低于非 R0 staleness。因而旧 tomography 的机制方向是稳健的，但正式 frontier 仍必须只使用 rolling-lineage executor。

## 边界与下一步

这是扩大后的 development validation，不是全人群 frontier：每个 cell/view 只有 40 个 stratified 请求，quality companion 没有足够样本形成统计结论，migration cost 也尚未按 state/user 去重。

下一步实现 state-keyed executor：每个用户在 cutover 只执行一次动作，按用户时间流复用状态并计量 token-layer work、KV/history bytes 与 batched runtime。随后才能绘制 No-op / Partial / Exact frontier。scheduler 仍未授权。
