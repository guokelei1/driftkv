# Archived: P9.6 State-Keyed Transition Cost

P9.6 将迁移成本按 release 时唯一用户状态计费，而不是按同一用户的多个 query 重复计费。edge1 与 edge2 分别包含 2,974 和 3,129 个 F fidelity 用户状态。

## 逻辑工作量

以 Exact-All 为 1，edge1 的主要动作成本为：

| Action | token-layer work | attention-pair work | new KV write |
|---|---:|---:|---:|
| Layer0-Recent128 | 6.48% | 0% | 6.48% |
| Layer0-Middle | 12.50% | 0% | 12.50% |
| Layer0-Full | 25.00% | 0% | 25.00% |
| HybridTail128 | 25.92% | 44.12% | 25.92% |
| Exact-All | 100% | 100% | 100% |

Layer0-Full 仍需读取全部 raw history token，但只投影第一层 K/V；HybridTail128 读取 parent prefix K/V 并在全部四层重放尾部，因此必须同时报告 old-KV read，而不能只看 token-layer work。

## GPU 原型计时

在 GPU0、R2 M0-F seed17、64 个 512-token 状态、batch 16 下，迁移动作本身的平均时间为：

| Action | ms/state |
|---|---:|
| Layer0-Recent128 | 0.0418 |
| Layer0-Middle | 0.0402 |
| Layer0-Full | 0.0402 |
| HybridTail128 | 0.1851 |
| Exact-All | 0.4744 |

该计时不含 checkpoint load、storage read、H2D、post-cutover append 和 query scoring。Layer0 三个宽度的时间接近，说明当前小型 GPU kernel 受固定 launch/embedding overhead 主导；论文结果仍需同时提供逻辑工作、bytes 和端到端批处理时间。

结合 P9.5，partial action 已同时显示 fidelity recovery 与明显低于 Exact 的开发成本。但 P9.5 只有 stratified validation requests，尚不能称为全人群 frontier。下一步必须实现按 uid 共享 cutover 状态和时间流的 full-population rolling executor，再进行状态级 fidelity/cost join。
