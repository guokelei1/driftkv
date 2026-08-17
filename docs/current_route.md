# EvoKV 当前路线（37D）

更新日期：2026-08-17。

本文是当前项目的执行入口。`docs/newset.md` 是完整论证稿；本文压缩研究边界、证据分级和下一步。旧资料仍可追溯，但除非本文明确列为当前输入，否则不得当作当前事实或论文结果。

## 1. 研究问题

EvoKV 研究推荐模型发布新版本后，旧版本生成的持久化用户 KV 状态何时可以直接复用，何时需要近似演进、局部重算或完整重算。

当前模型已由模型发布流程决定；EvoKV 不判断新模型是否应该发布，只处理发布后的状态兼容性和演进成本。允许的动作是：`no-op reuse`、快速恢复、选择性重算、exact recompute。

## 2. 不再承担的证明负担

- 不证明 HSTU 优于 SASRec、BERT4Rec 等推荐算法。
- 不要求每条相邻版本边都有正的 Recompute-over-Reuse gap。
- 不把 Full 当作 ranking 理论上界；负 gap 可能是模型发布负迁移，而非 evaluator 错误。
- 不声称 HSTU 稳定利用严格自然顺序；当前用语是 `history-conditioned persistent state`。
- 不用事后分数混合、指标缩放、挑选有利边、挑选用户子群或目标 K/V 回归制造 gap。
- 不在基模问题尚未闭环前继续扩展 KuaiRand 八版本链，也不把 D2/D3 系统探索提前当作论文主线。

## 3. 最小系统闭环

### 3.1 Compatibility profiler

在少量 canary 状态上比较 Full、Reuse 和少量局部恢复，收集与目标标签无关的兼容性特征：输出分布差异、Top-K overlap、rank displacement、score margin、层/头敏感度、状态年龄、历史长度和用户活跃度。canary 只用于估计风险和冻结阈值，不用于拟合 old-KV 到 exact-new-KV 的自由映射器。

### 3.2 State evolution controller

控制器根据版本变化、用户状态、请求时刻和可用预算选择四条动作路径，并支持状态随 append 逐步稀释风险。重点不是寻找在所有 workload 上都有效的单一 mapper，而是验证分层决策是否比永远 Reuse 或永远 Recompute 更合理。

### 3.3 论文闭环

1. 证明持久化 KV 与模型版本绑定，更新后会出现混合语义。
2. 证明兼容性具有版本、用户、层和请求级异质性。
3. 用 profiler/controller 选择 no-op、近似、局部或 exact 路径。
4. 在 fidelity/质量约束下报告节省的重算、IO 和发布后 burst 成本。

## 4. 旧材料中保留的有效内容

| 材料 | 保留理由 | 当前用法 |
|---|---|---|
| v69 连续 block-release 链 | 已建立真实版本 lineage、时间窗口和 Reuse/Recompute 对齐执行链 | 开发 workload，不是 formal result |
| 旧 44.666 GiB 两卡链 | 证明容量、长历史重建和大状态工作负载确实存在 | motivation/capacity evidence，不再扩展 |
| v80/v81/v82 一类结果 | 提供兼容、异质性和历史信号强弱的候选 workload | 只在冻结协议下复核，不拼接旧矩阵 |
| 首请求、publish cutoff、online lineage 审计 | 直接决定旧 cache 是否具有合法血缘 | 所有新实验的 correctness invariant |
| raw-score、no-history、popularity、Full 对照 | 区分模型有效性、历史贡献和 popularity shortcut | 最小 sanity check |
| 失败的 query/score 混合、near-100% mapper、逐边调参记录 | 给出泄漏、捷径和过拟合边界 | negative evidence，禁止复活 |
| D2/D3 生命周期、存储和硬件机制代码 | 未来系统实现可能复用接口和成本模型 | future work，不影响当前质量协议 |

## 5. 当前实验顺序

1. 固定一个小规模、可重复的三版本 workload，先完成 routine / moderate / major 三种更新区域。
2. 每个区域只做 Random、Popularity、No-history、Full 四项最小模型 sanity check；不重新开启模型算法横向比较。
3. 在同一模型版本和同一请求上比较 Reuse、selective/approximate、Exact，报告绝对质量、fidelity、动作比例和理论成本。
4. 用不参与阈值选择的时间段验证 controller；失败版本和负 gap 原样保留。
5. 小规模闭环成立后，再扩展到 Yambda 与 RecFlow；最后才做大状态、多级存储和硬件系统验证。

## 6. 资料使用规则

- 当前路线只认本文和 `docs/newset.md`；协议细节需写入新的冻结 contract 后才能产生论文结果。
- `docs/legacy/README.md` 中列出的文件可以解释来路，但不定义当前路线。
- 任何结果必须带 workload、版本 lineage、时间边界、候选、score、seed 和证据等级。
- 历史结果不跨协议拼表；明确无复用价值的临时日志、cache 和生成物不进入新的研究结论。
- 新代码必须服务 profiler、controller、四条动作路径或其验证；仅用于已否决分支的 runner 不再扩展。
