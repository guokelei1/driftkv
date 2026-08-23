# Archived: P9.4 Rolling-Lineage 审计与 Canary

## 裁决

P8/P9 的旧执行器按请求重算保留下来的 parent prefix。长度计算正确，但当 512-token 状态发生滚动淘汰时，这不等价于 release 时物化一次、随后逐事件 append/evict 的持久化上层 K/V。F fidelity 请求中约 86.5%–87.0% 发生淘汰，因此旧结果只保留为 request-conditioned 机制证据。

冻结后的真实 lineage canary 使用以下语义：release 时分别物化 parent 与 current 状态；每个 post-cutover listen 在 current model 下逐个追加；cache 满时在追加前淘汰最旧位置；query 只瞬时读取最终状态。

Canary 全部门通过：

- Exact-All 与 CurrentOnline 最大 logit 差为 `0`；
- R0 No-op 和全部动作的最大 JS 均为 `0`；
- snapshot/suffix 计数不一致为 `0`；
- R2 fidelity 的在线 No-op JS 为 `6.48465e-4`，旧 request-local JS 为 `6.49965e-4`，比例为 `0.9977`；
- CurrentOnline 与旧 CurrentFull 的平均 JS 仅 `6.04e-8`。

R2 合法动作在 18-request fidelity canary 上的恢复率为：

| Action | JS recovery |
|---|---:|
| Layer0-Recent128 | 72.7% |
| Layer0-Middle | 95.5% |
| Layer0-Full | 98.2% |
| HybridTail-128 | 79.0% |
| Exact-All | 100% |

quality canary 也保留同方向 fidelity recovery，但只有 18 个请求，不能作为统计质量结论。下一步必须扩展到全部 model/seed/release，并重新测量真实执行成本后才能绘制正式 frontier。

## 证据边界

- 旧 P8/P9 数字不删除，但标记为 request-conditioned diagnostic。
- 当前 canary 是 development correctness gate，不是 paper qualification。
- 尚未授权 scheduler。
- 若扩展矩阵中结构消失，应在 rolling lineage 下重做 tomography，不得回退引用旧 frontier。
