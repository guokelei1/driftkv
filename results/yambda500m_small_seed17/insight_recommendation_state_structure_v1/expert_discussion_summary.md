# EvoKV v0–v5 推荐状态结构探索：专家讨论稿

更新日期：2026-08-28  
状态：Small/seed17 正式结构 observation、signed causal/真实曝光候选复核、reader-stage 定位、跨真实请求持久性、compact-probe AV broadcast residual score canary、无 translated-prefix 物化的轻量 PRO 正确性/成本、轻量 PRO 五边全人口 rolling quality，以及最新 progressive PRO 的五边无标签 error decomposition 与 C32/C48/C64 fidelity frontier 均已完成。原 C32 PRO 在 217,584 个真实请求上相对 Reuse 的 AUC 为 5/5 正向、log-loss 为 3/5 正向，五边平均 AUC 与 log-loss 均改善；因此总体 Design viability 判为通过。最新增量没有形成可冻结升级：C64 虽在 cutover/rolling 的 relative L2 上均为 5/5 优于 C32，但未通过 rolling absolute-direction 门，且 C48/C64 非单调。当前正式设计仍为 C32 lightweight PRO；事前严格 quality gate 未过，也不准入 serving lineage、额外 seed/runtime 或长训练。

> **口径更正（2026-08-27）：** 当前已经证明的是 candidate-shared **reader compatibility correction**，不是“存在可物化的 candidate-shared history evidence basis”。历史侧错误是分布式、上下文化的；已观察到的是它经过 Current HSTU reader 后在候选维度收敛为共享的用户级修正。下文保留首个失败机制的原合同名称 `evidence_measure_basis` 作为审计标识，但不再把该名称当作已证实结论。

## 0. 一页结论

这次探索的出发点是：已有 `CAST + compact PATCH + GROUP/SCALE` 能构成一条完整的强基线，但它更像通用的状态压缩/修复流水线，还没有把推荐系统中“同一份用户状态被大量候选反复读取”的结构变成核心 Insight。

我们在固定的 HSTU-native `v0→v1→...→v5` 模型链上，对 **3,000 名固定用户**（Small 冻结人口的 30%）完成了五条相邻发布边的 label-free 内部状态观测。每个 user-edge 使用 cutover 前 512 个历史事件和 64 个固定候选，共形成：

- 15,000 个 user-edge；
- 60,000 个 user-edge-layer candidate influence 矩阵；
- 64 个 candidate probe/user-edge；
- UID-disjoint 的 1,520 fit users / 1,480 held-out users；
- 全流程不读取请求 label，也不把未曝光 candidate 当作负样本。

最强的观察是：

> **跨版本 HSTU 的历史侧误差是分布式、上下文化的，但经 Current reader 的 query-dependent aggregation 后，会形成一份被整个候选集合共同消费的用户级 compatibility correction，而不是 64 个候选各自发生一次独立的 token retrieval failure。**

支撑它的三组结果是：

1. 60,000/60,000 个 Current candidate×history influence 矩阵均为 rank-1@90%；第一 candidate-shared 方向平均承载 99.9681%–99.9992% 的能量。Exact−Reuse influence 差分在 59,999/60,000 个矩阵上为 rank-1@90%，最终 readout 差分在 15,000/15,000 个 user-edge 上为 rank-1@90%。
2. 跨用户 state-delta 分解表明，item/action 的共享坐标在输入和 layer-0 K 上很强，但经过 HSTU aggregation 与 U gate 后迅速变成 user-context residual。layer-0 K 的 item-action held-out R² 为 86.7%–92.6%，而 deeper update 中 item 相对 global shift 的额外 R² 只剩 −0.4%–2.9%。
3. raw same-item/action 并不是稳定的 contextual substitutability：语义配对将 same-item pair 比例从约 3.3%–3.8% 提高到 29.6%–30.1%，却只在 3/5 条边改善平均 probability gap，逐用户胜率仅 42.2%–50.0%。

专家随后指出，原始 influence 使用 contribution norm、candidate-wise normalization 和受控 candidate bank，可能人为强化共同方向。我们按这一意见追加了一轮事前冻结的决定性验证：使用不做 candidate normalization 的 signed、逐 head 干预，改变 bank width，并在真实 same-UID/same-timestamp exposed requests 上做 raw-first 复核。结果是：受控 3,000 用户 width-64 上 shared-only 恢复 97.98%–99.64% 的 Reuse 概率缺口；真实曝光分布的 20 个 edge×width 组合全部由 shared 优于 residual，恢复 98.72%–99.84%，shared-only 的平均绝对 logit gap 为 `5.58e-5`，Reuse 为 `1.55e-2`。因此 **candidate-shared reader compatibility correction 已从低秩观测升级为 signed causal reader structure**，并且不是只由受控候选或 norm/normalization 造成的伪影。这不证明历史 token 可以被压成某种线性 evidence basis。

但专家建议的第二道门没有通过。我们只实现了一个与 Design 0 在参数映射 FLOPs、Current compute、64 carriers、recent-128 raw I/O 和 448 个物化状态位置上完全匹配的最小 evidence-measure basis。它在五条边、每边 32 名无标签 canary 用户、共 1,598 个请求上，mean absolute logit gap **0/5 条边不弱于 Design 0**，因此按冻结合同停止，没有读取 canary label，也没有启动 formal rolling AUC/log-loss。当前结论必须分开：**结构发现是强正结果；这一种朴素的 `CAST value measure + Current anchor residual` 可执行化是明确负结果。** 现有 Design 0 继续作为 strong baseline；本轮仍不支持增加 per-candidate Route、raw semantic GROUP 或 embedding selector。

按最新专家意见追加的决定性实验进一步收敛了结论。阶段定位显示 shared correction 最早已经出现在 query-dependent 的 `activated(qK)·V` prefix contribution，而不是 raw K/V；受控和真实请求的 K/V-contribution、AV 两阶段均为 5/5 edge 过门。真实请求持久性覆盖 11,364 对相邻请求，AV correction 的五边中位方向 cosine 为 `0.9659–0.9827`，coverage-scaled 前序修正恢复 `60.60%–84.06%`，5/5 通过。由此唯一解锁的 compact-probe AV sidecar 用 32 个 Current carriers 加两条固定单 probe reader path，在 1,805 个无标签请求上相对 Design 0 的 logit gap 为 4/5 边改善（`−36.7%/−54.2%/−62.9%/+11.0%/−58.0%`）。这是首个可执行方向的正 score canary，但 `v3→v4` 是明确反例，而且尚无 AUC/log-loss、额外 seed、runtime 或 scale 资格。

最新一轮进一步删除了这个方向中最重的旧组件：不再对 384 个旧位置逐一生成或写回 translated K/V。版本对的 joint linear map 被推入一次固定 probe 的 AV read；recent-128 仍预先固定压成 32 个 Current carrier，但 carrier 只在未变换的 Parent prefix 上做 dependency-closed contextual replay。新 action 的 translated-prefix 物化位置为 0，只写四层共 512 个 sidecar scalar。held-out 32 用户×五边的 v2 验证中，融合 read 相对同语义 materialized reference 的最大相对 L2 为 `4.73e-6`，sidecar replay 最大绝对误差为 `3.58e-7`；32-carrier sidecar 相对旧 40.5%-of-Full extractor 的五边平均方向 cosine 为 `0.9983–0.9993`，幅值比中位数为 `0.9828–0.9982`。公平理论口径为 `0.057 GFLOPs/user`、Exact-All 的 `9.1%`；16-carrier diagnostic 为 `5.2%`，但对旧 sidecar 的方向和相对误差明显较弱，因此不替换事前主配置 32。该轮只通过正确性、结构和成本门，不包含 request-score 或行为质量结论。

在机制不变、`carrier32/latest-item/mass4/coverage-decay` 全部冻结后，随后完成了五条 E14 全人口 rolling quality。每条 edge 的 PRO raw 先独立封存，再与既有 sealed Design 0 精确对齐请求，最后才读取 label；五边共 217,584 个请求，基线最大重放误差为 `9.54e-7`。PRO 相对 Reuse 的 AUC **5/5 全部提高**，提高 `0.00493–0.16934` pp；log-loss **3/5 提高**，两条变差仅为 `1.87e-5` 和 `3.75e-5`。五边非加权平均为 `+0.06641` AUC pp、`−7.65e-5` log-loss。因事前严格 gate 要求两项各 4/5，它如实记为 qualification FAIL；但按“平均为正且过半 edge 正向”的总体可行性标准，Design viability 为 PASS。这个后验推进裁决不改写原 gate，也不允许再用同一五边证明后续调参版本。

最新专家建议的增量也已按 raw-first、label-free 协议完成。64-user/edge 的 error decomposition 表明，C32 双 probe 对 Exact shared AV 的 `cosine≥0.90` 门只在 cutover 2/5、rolling 0/5 edge 通过；oracle projection 的纯幅值改进门为 0/5+0/5。两条固定 probe 自身却在 5/5 edge 高度一致（cosine `0.99998–0.99999`、norm ratio `0.99958–1.00012`），所以问题不是 latest-item probe 的随机性；old/recent segment decay 也只在 2/5 edge 不差于 global decay。下一批 held-out 64-user/edge 的 10.52%/14.54%/18.64% C32/C48/C64 frontier 中，C64 relative L2 相对 C32 在 cutover 和 rolling 都是 5/5 改善，无标签 score gap 为 5/5、4/5 改善；但 C64 的 absolute-direction 门仍只有 3/5、0/5，而且 C48 的五边平均 relative L2 略低于 C64。按事前规则，这不是单调、可冻结的 progressive precision axis，因此增量升级不入选，保留原 C32 设计且不重读旧五边 quality label。

## 1. 为什么做这次探索

已有机制实验已经回答了一部分“怎样低成本转换 persistent K/V”的问题：大范围共享变换、局部 dependency-closed replay、carrier 压缩和 evidence mass 保留可以拼成一条可执行路径。但如果论文主 Insight 只写成“旧状态 + 最近重算 + 压缩”，它与通用 cache repair、RAG compression 或 state migration 太相似，推荐系统本身只剩应用背景。

这次探索没有继续增加新 operator，而是回到已经存在的六个模型版本，询问三个更推荐系统化的问题：

1. 同一用户状态被一组候选读取时，candidate 之间的读取模式是否真的彼此独立？
2. 跨版本状态变化中，是否存在可以跨用户泛化的 item/action 共享坐标；它在 HSTU 内部何处转化成用户上下文残差？
3. 如果相同 item/action 表示相似语义，是否可以在相同 carrier budget 下稳定构造更好的 semantic coreset？

这三问分别对应 candidate-bank subspace、cross-user delta factorization 和 matched-budget semantic coreset 三个实验。它们在运行前写入同一 prospective observation contract，五条边全部报告，没有按结果挑边或挑用户。

## 2. 实验范围与协议

| 项目 | 固定设置 |
| --- | --- |
| 模型与数据 | EvoKV-HSTU-S，Yambda-500M Small，training seed 17 |
| 版本链 | `v0→v1`、`v1→v2`、`v2→v3`、`v3→v4`、`v4→v5` |
| cutover day | 231、245、259、273、287 |
| 用户人口 | 3,000 名；Small 冻结 10,000 用户的 30% |
| 用户资格 | day 231 前至少有 512 次 listen；按既有 label-free selector rank、再按 UID 固定选择 |
| 历史 | 每条边取该用户 cutover 前最后 512 个事件；同一 3,000 人贯穿五条边 |
| candidate bank | 每 user-edge 固定 64 个：最多 16 个 recent repeat、最多 16 个 old-only repeat，其余由 prefix 中未出现的已知高频 item 补齐 |
| delta split | UID hash 固定拆成 1,520 fit / 1,480 held-out users，只用于跨用户 centroid 拟合和评价 |
| 标签边界 | 不读取未来/当前请求 label；candidate 不被解释为负样本；不做 candidate admission |
| 执行边界 | 不训练新模型，不改变 cache lineage，不读取 theta3，不启动 Medium/Large |

正式运行耗时 416.6 秒。candidate influence trace 与模型原始 score 的五边最大绝对误差为 `7.15e-7`，说明观测 trace 没有改变被测 score。正式输出只保留 compact aggregates，没有保存展开 K/V 或 attention tensors。

完整冻结输入、checkpoint hash 和协议见 [contract](../../../configs/contracts/yambda500m_small_hstu_native_recommendation_state_structure_v1.yaml)；原始聚合见 [report](report.md) 和 [adjudication](adjudication.md)。

## 3. 实验一：candidate-bank influence subspace

### 3.1 做法

对每个 user-edge，分别构造：

- **Current Exact**：Current 模型产生并读取完整 Current cache；
- **Parent Reuse**：Current reader 直接读取 Parent 模型产生的 persistent cache。

对 64 个 candidate，在 HSTU 每一层记录 512 个历史位置的 pointwise-attention value contribution norm，得到 `64 candidates × 512 history positions` 的 influence 矩阵。每个 candidate 的 history influence 归一化后，计算：

- Current influence 矩阵的 90% 能量秩与 effective rank；
- `Current Exact − Parent Reuse` influence 差分的谱；
- 64 个 candidate 的最终 hidden readout 差分谱；
- recent repeat、old-only repeat、novel-to-prefix 三类 candidate 的 effective support 和绝对 logit shift。

这里的 rank 是“固定一名用户时，candidate 维度上的共享程度”，不是模型参数矩阵的 rank。

### 3.2 结果

| edge | Current 第一共享方向能量（四层均值范围） | Current rank-1@90% | Exact−Reuse influence rank-1@90% | readout delta effective-rank 均值 |
| --- | ---: | ---: | ---: | ---: |
| v0→v1 | 99.9805%–99.9992% | 12,000/12,000 | 12,000/12,000 | 1.0082 |
| v1→v2 | 99.9764%–99.9987% | 12,000/12,000 | 12,000/12,000 | 1.0085 |
| v2→v3 | 99.9752%–99.9991% | 12,000/12,000 | 12,000/12,000 | 1.0198 |
| v3→v4 | 99.9770%–99.9984% | 12,000/12,000 | 11,999/12,000 | 1.0102 |
| v4→v5 | 99.9681%–99.9988% | 12,000/12,000 | 12,000/12,000 | 1.0100 |
| **总计** | **99.9681%–99.9992%** | **60,000/60,000** | **59,999/60,000** | **1.0082–1.0198** |

最终 `Exact−Reuse` readout 差分在五条边的 15,000/15,000 个 user-edge 上均为 rank-1@90%。这不是只在某一条 release edge 出现的现象。

三类 candidate 的结果也非常接近。例如 layer 0 上：

| edge | recent / old / novel effective support | recent / old / novel 绝对 logit shift |
| --- | --- | --- |
| v0→v1 | 497.20 / 497.22 / 496.54 | 0.05722 / 0.05726 / 0.05705 |
| v1→v2 | 492.02 / 492.04 / 490.76 | 0.01879 / 0.01881 / 0.01851 |
| v2→v3 | 491.64 / 491.67 / 489.87 | 0.01371 / 0.01370 / 0.01417 |
| v3→v4 | 490.88 / 490.91 / 488.98 | 0.05290 / 0.05289 / 0.05486 |
| v4→v5 | 488.91 / 488.91 / 486.50 | 0.01222 / 0.01220 / 0.01239 |

### 3.3 当前解释

对于这个 HSTU ranking workload，同一 persistent user state 的主要读取形状在 candidate 之间高度共享；版本错配也主要沿 candidate-shared 方向传播。因此，主要功能效应更像“每用户一次、由整个候选集合共同消费的 reader compatibility correction”，而不是“每 candidate 一次的 token selection”。这使 Insight 直接来自推荐 ranking 的 candidate amortization，而不是把推荐历史当作普通文档 token。它是否能物化为跨请求 sidecar，必须由阶段定位与持久性实验另行决定。

这组谱结果本身仍不能证明某个 basis/anchor 实现会恢复 AUC。尤其 influence 使用了各 head contribution 的 norm 并做 candidate-wise history normalization，可能放大共同形状。这个当时保留的伪影风险已由第 11 节的 signed/head-wise、无 normalization、变宽度与真实 exposed candidate 干预复核；机制可执行性则仍需单独过门。

## 4. 实验二：跨用户 state-delta factorization

### 4.1 做法

对同一个历史事件，计算不同版本中的状态变化：

~~~text
delta(stage, item, action, user-context)
  = Current(stage) - Parent(stage)
~~~

每条边选择由最多不同用户覆盖的 top-1,024 shared items，在 1,520 名 fit users 上分别拟合：

- 全局 version-shift centroid；
- item centroid；
- item-action centroid。

然后只在 1,480 名 UID-disjoint held-out users 上评价 R²。观察位置覆盖 combined input、各层 normalized input、Q/K/V、AV、attention output、U gate、gated update 和 hidden output。每个 item-action group 在 fit/test 中都至少需要 8 个样本。

### 4.2 结果

| edge | combined-input item R² | layer-0 K item R² | layer-0 K item-action R² | layer-0 update item R² | layer 1–3 update：item 相对 global shift 的额外 R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0→v1 | 68.2% | 61.8% | 89.8% | 31.0% | 0.3%–2.9% |
| v1→v2 | 71.5% | 72.4% | 88.1% | 28.1% | 0.8%–2.2% |
| v2→v3 | 59.5% | 55.7% | 92.6% | 20.8% | 0.3%–2.3% |
| v3→v4 | 66.7% | 61.9% | 86.7% | 21.3% | 1.3%–2.4% |
| v4→v5 | 60.2% | 58.5% | 92.5% | 21.0% | −0.4%–1.8% |

有两个重要的方向性变化：

1. 在 combined input 和 layer-0 K，item identity 能跨用户解释相当一部分版本变化；action type 进一步把 layer-0 K 的解释率提高到 86.7%–92.6%。因此，不能再用“isolated embedding 效果弱”推导“item identity 与版本兼容性无关”。
2. 一旦经过 HSTU 的 attention aggregation 与 U gate，item-only 的可预测性明显下降。layer-0 update 的 item R² 只有 20.8%–31.0%，更深 update 中 item 相对 global shift 的额外解释率只剩 −0.4%–2.9%。

### 4.3 当前解释

更符合结果的分解不是“共享 item shift”或“完全个性化 residual”二选一，而是：

> **输入与 early K/V 存在跨用户共享的 typed entity/action coordinate；HSTU aggregation 和 gating 随后把它转化成 user-context residual。**

所以 isolated embedding replacement 不是完整迁移接口，但共享 typed coordinate 也不是可以忽略的噪声。合理机制需要同时处理便宜、共享的版本坐标和较小、依赖上下文的 residual。

## 5. 实验三：matched-budget semantic coreset

### 5.1 做法

固定每个用户的 old-384 Parent prefix，仅处理 recent-128。所有 compact 路径都用 64 个 Current carrier 表示 128 个 recent events，每个 carrier 的 represented mass 固定为 2；唯一变化是配对规则：

1. **positional pairs**：相邻两个事件配对；
2. **same-item-first**：优先相同 item；
3. **typed pairs**：依次优先相同 item-action、相同 item、相同 action。

每对都取更晚事件作为 carrier。比较它们到 Current Exact 的 mean absolute probability gap、top-10 overlap 和 rank correlation。这个对照严格保持 history scope、carrier count、Current recompute 数量与 represented mass 相同，因此只测试“raw semantic identity 是否给出更好的合并关系”。

### 5.2 结果

same-item-first 将 same-item pair fraction 从 3.29%–3.79% 提高到 29.55%–30.13%；typed pairing 的 same-action 比例约为 98.2%。但 score 结果并未稳定改善：

| edge | positional gap | same-item gap | typed gap | same-item / typed 逐用户胜率 |
| --- | ---: | ---: | ---: | ---: |
| v0→v1 | 0.0036069 | 0.0035390 | 0.0035429 | 48.2% / 50.0% |
| v1→v2 | 0.0013241 | 0.0013104 | 0.0013101 | 44.5% / 46.7% |
| v2→v3 | 0.0011753 | 0.0012412 | 0.0012257 | 42.2% / 44.4% |
| v3→v4 | 0.0035364 | 0.0034625 | 0.0034709 | 49.7% / 49.3% |
| v4→v5 | 0.0010082 | 0.0010422 | 0.0010450 | 44.3% / 44.6% |

same-item 和 typed rule 都只在 3/5 条边改善 edge-level mean gap；即使平均 gap 略好，逐用户胜率仍不超过 50%。这不是一个可作为 action admission 依据的稳定关系。

### 5.3 负结论

raw item/action equality 说明事件在离散语义上相同，却不能保证它们在 HSTU 的时序上下文中可互换。推荐历史中的 evidence 仍受 occurrence position、surrounding history、version-specific projection、aggregation 和 represented mass 共同影响。

因此本结果明确否定：

- 直接把 same-item/action 当作稳定 GROUP relation；
- 认为语义匹配率提高就等价于 Current-state approximation 提高；
- 用这一轮结果把 semantic GROUP 加入 scale action frontier。

## 6. 三个实验如何连成一个 Insight

三个实验分别约束了“读取端”“状态来源”和“压缩关系”：

~~~text
Candidate-bank observation
  同一用户状态的读取与版本错配主要沿 candidate-shared direction
        ↓
Cross-user delta factorization
  early state 有 shared typed coordinate，随后变成 contextual residual
        ↓
Semantic coreset counterexample
  raw item/action equality 不能直接定义 contextual substitutability
        ↓
Candidate-shared reader compatibility correction
  是已通过 signed causal 验证的功能结构；是否可跨请求物化仍待验证
~~~

基于当前证据，可以做如下分级裁决：

| 命题 | 当前状态 |
| --- | --- |
| persistent HSTU state 在候选集合上呈 candidate-broadcast 读取结构 | 五边、3,000 用户谱观测与真实 candidate signed intervention 强支持 |
| Exact−Reuse 的主要错配同样是 candidate-shared | 受控 5/5、真实 20/20 edge×width 因果干预支持 |
| early cross-version delta 含共享 item/action coordinate | UID-disjoint held-out 上支持 |
| aggregation/gating 将其转成 contextual residual | 多 stage R² 下降支持 |
| raw same-item/action 是稳定 coreset relation | 被五边 matched-budget 对照否定 |
| 应为不同 candidate family 分别执行 token Route | 当前 support/logit-shift 结果不支持 |
| candidate-shared reader correction 已恢复同请求 reader gap | signed oracle 已支持；不是可执行 action |
| 某个 candidate-shared basis/anchor 已恢复 rolling AUC | **未验证**；首个历史侧 matched-cost canary 已 0/5 失败并停止 |
| 新机制已证明更快、更省 I/O 或可扩展到 Medium/Large | **未验证** |

## 7. 对当前设计的影响

### 7.1 保留什么

现有 `CAST + GROUP→PATCH→SCALE` 仍然是一条有执行语义、有 rolling 结果和理论成本分析的 **Design 0 strong baseline**。本轮 observation 没有废除这些底层 primitive，也没有增加 scale action。

### 7.2 改变什么

论文的核心表述不再把四个 operator 本身包装成四条 Insight，而是优先强调推荐 ranking 的结构：

> 一份 persistent user evidence 被整个 candidate bank 摊销读取；release-time compatibility repair 应首先按用户发生一次，再跨候选共享。

`CAST` 可以被解释为处理共享 version coordinate 的 Design 0 方法，dependency-closed PATCH 处理 contextual residual；但这只是对已有路径的重新定位，不表示它已经实现了新的 evidence-basis mechanism。

### 7.3 下一步验证边界

第一个按此方向冻结的 `CAST value measure + Current anchor residual` 已完成 canary 并失败。这个负结果否定了朴素历史 V 线性替代，但没有否定 reader correction。按最新专家意见，在讨论第二个机制前只回答两件事：修正最早在 K/V contribution、AV、U-gated update、layer hidden 还是 final readout 形成；以及同一用户在真实请求组之间其方向、幅值和恢复能力是否持久。只有两道门都通过，才允许一个直接来自 Current HSTU intermediate state 的 per-user layerwise broadcast residual canary。

如果该唯一机制被两道门解锁，仍须在新 prospective contract 下满足：

1. 从 Current model 重建少量 candidate-independent evidence anchors/basis；
2. 显式保留 coverage 与 evidence mass；
3. 用较小 contextual residual 补足 aggregation/gating 后的用户特异变化；
4. 同一 basis 对整个 candidate bank 复用，而不是 request-time per-candidate 选 token；
5. 先通过 score-replay correctness canary，再评价全部 frozen seeds/edges 的 task quality、GPU runtime 和 I/O。

以上仍是设计约束，不表示新机制已获授权或第一个负结果可以调参覆盖。history V sum、PCA/SVD、raw semantic GROUP 与 per-candidate selector 均不在下一轮分叉中。

## 8. 结果边界与可能的替代解释

1. **只有 Small/seed17。** 结果覆盖了 30% Small 用户和全部五条相邻边，但 training seed、模型规模和数据域仍然单一，不能直接外推到 population-scale、Medium/Large 或 RecFlow。
2. **原始 candidate bank 是受控 probe，但已新增真实分布。** 初轮 repeat/novel panel 仍不是曝光分布；第 11 节新增了真实 same-UID/same-timestamp banks，不能把其中未标注的候选当负样本，也不能外推到其他 ranking system。
3. **低秩的主要伪影风险已被因果复核显著降低。** signed/head-wise、无 normalization、不同 width 和实际 exposed candidates 均复现 shared dominance；这仍不等于某个压缩实现自动正确。
4. **R² 不是迁移收益。** centroid factorization 只说明哪些 delta component 可以跨用户预测，不说明直接写入该 centroid 会保持 dependency closure 或恢复 AUC。
5. **semantic coreset 的负结果只否定 raw identity rule。** 它不否定使用 contextual similarity、functional equivalence 或 learned-but-label-free basis；这些需要新的冻结协议。
6. **首个机制只到 canary 且失败。** 它 0/5 不弱于 Design 0，未解锁 formal rolling qualification；当前不能声称 basis 优于 Design 0，也不能用本轮结果选择 edge、carrier 数或改 pair/budget。
7. **没有 runtime 结论。** “每用户一次并跨 candidate 摊销”是计算结构上的方向，不等于已经测得更低 latency、bandwidth 或 makespan。

## 9. 第一轮讨论问题（后续状态见第 11–12 节）

以下问题保留为第一轮审计记录；其中 signed causal、真实候选、functional object 与最小 canary
已由第 11–12 节推进，当前待讨论问题以第 12.6 节为准：

1. **Insight 是否足够 recommendation-specific？** “candidate-broadcast user-evidence field”是否准确抓住 ranking workload 与单 query context processing 的本质差异？是否需要更精确地区分 retrieval、ranking 和 generative recommendation？
2. **signed causal 证据是否已足够排除结构伪影？** 第 11 节已完成逐 head、变 width、真实 candidate 干预；还需要额外 seed 才能提升 paper claim，还是可留到机制通过后？
3. **第一个 value-measure basis 失败后，正确的 basis 对象是什么？** 是否必须直接来自 Current HSTU intermediate state，而不是 history anchors 或 signed V measure？
4. **shared coordinate 与 contextual residual 应怎样分工？** 第一个 `CAST measure + Current anchor` 分工失败；下一次是否只保留“direct Current intermediate basis”这一条机制分叉？
5. **contextual substitutability 如何定义？** raw item/action 已失败；下一轮应使用哪种 label-free functional criterion，且怎样避免 target-K/V fitting 和 qualification leakage？
6. **第二个机制的最小充分验证是什么？** 不同 width、真实 exposed candidate 和 signed intervention 已完成；额外 seed 应在机制 canary 前还是在 Small/seed17 五边质量通过后执行？
7. **论文 claim 的保守边界是否合适？** 当前建议只 claim state structure discovery，把 task-quality recovery、runtime 和跨规模泛化留给后续 prospective 实验。

## 10. 可复核材料

- 冻结契约：[yambda500m_small_hstu_native_recommendation_state_structure_v1.yaml](../../../configs/contracts/yambda500m_small_hstu_native_recommendation_state_structure_v1.yaml)
- 正式观测脚本：[probe_recommendation_state_structure.py](../../../scripts/insight/probe_recommendation_state_structure.py)
- 裁决脚本：[adjudicate_recommendation_state_structure.py](../../../scripts/insight/adjudicate_recommendation_state_structure.py)
- 原始聚合报告：[report.md](report.md)
- 正式裁决：[adjudication.md](adjudication.md)
- 机器可读摘要：[summary.json](summary.json) 与 [adjudication.json](adjudication.json)
- candidate common basis：[candidate_common_basis.csv](candidate_common_basis.csv)
- state factorization：[state_factorization_adjudicated.csv](state_factorization_adjudicated.csv)
- semantic coreset：[semantic_coreset_adjudicated.csv](semantic_coreset_adjudicated.csv)
- 当前论文概念定位：[paper_design.md](../../../docs/paper_design.md)
- 完整 motivation/observation 上下文：[motivation_observations.md](../../../docs/motivation_observations.md)

## 11. 专家意见后的决定性验证与最小机制裁决

本节记录收到专家意见后新增的完整一轮工作。它不是对前述结果的事后重解释，而是两个新的 prospective gate：先判断 candidate-shared direction 是否有 signed causal 意义，再只实现一个 matched-cost mechanism candidate。合同、候选宽度、用户、edge、路径和停止条件均在运行前冻结。

### 11.1 专家意见怎样改变了实验

专家认可 recommendation ranking 中“一份用户状态被整组候选共同读取”的方向，但指出三个可能的伪影来源：contribution norm、candidate-wise normalization 和受控构造 candidate bank。因此新增验证刻意去掉这些因素：

1. 直接使用每层、每 head 的 signed `attention_weight @ V`，不取 contribution norm，不做 candidate normalization；
2. 对同一 user-edge 的 Exact−Reuse signed delta 分解为 candidate mean broadcast `shared` 与零均值 `residual`；
3. 显式执行 `Reuse+shared`、`Reuse+residual` 和 `Reuse+shared+residual`，而不只看谱；
4. 受控 bank 使用固定 nested width `8/16/32/64`；真实 exposed bank 使用同 UID、同 timestamp 的真实请求组，固定 width `2/4/8/16`，不加入 sampled/synthetic negative；
5. 真实请求 raw score 先生成并封存，之后才连接质量标签。

这些 shared/residual 路径仍是 causal oracle intervention，不是可执行 cache action。

### 11.2 受控 3,000 用户 signed causal 结果

五条相邻 release edge、固定 3,000 用户和四种 candidate width 全部完成。native reader 与 intervention trace、`shared+residual` 与 Current Exact 的最大重构误差均为 `9.54e-7`。

在最大的 width-64 上，按 edge 聚合的结果为：

| edge | Reuse probability gap | shared-only gap | residual-only gap | shared gap recovery |
| --- | ---: | ---: | ---: | ---: |
| v0→v1 | 0.00483247 | 0.00002027 | 0.00483320 | 99.58% |
| v1→v2 | 0.00170976 | 0.00002642 | 0.00170794 | 98.45% |
| v2→v3 | 0.00143675 | 0.00002899 | 0.00143642 | 97.98% |
| v3→v4 | 0.00470432 | 0.00001709 | 0.00470245 | 99.64% |
| v4→v5 | 0.00113766 | 0.00001721 | 0.00113711 | 98.49% |

shared-only 在 5/5 edge 上显著优于 residual-only；residual-only 基本与 Reuse 重合。最大的受控 bank 上，逐 edge 的 signed shared energy fraction 为 99.77%–99.81%；width-8 相对 width-64 的 shared direction cosine 均值约为 `0.99998–0.99999`。因此结论不依赖只报告 width-64。

### 11.3 真实 exposed candidate 复核

真实候选来自 frozen `requests_fidelity` manifest 中同 UID、同 query timestamp 的实际请求组。五条 E14 edge 各覆盖 1,272–1,300 名有至少一个二候选组的用户；按四种 width 展开后共有 24,407 个 bank 和 68,764 个 request-width observation。没有把未曝光 item 加入 bank，也没有把候选解释为负样本。

结果为：

- 20/20 个 `edge × width` 组合中，shared-only probability gap 小于 residual-only；
- 按组合聚合的 shared gap recovery 为 98.72%–99.84%；
- 非零 signed head observation 中，shared component 平均承载 99.9170% 的 delta energy；
- shared-only 到 Current Exact 的平均绝对 logit gap 为 `5.58e-5`，Reuse 为 `1.55e-2`，相差约 278 倍；
- 20 个组合中，shared-only 与 Current Exact 的最大绝对 ROC-AUC 差为 `9.39e-5`，最大绝对 log-loss 差为 `1.47e-6`；
- native score 和 full-delta reconstruction 的五边最大误差仍为 `9.54e-7`。

真实候选内部 pairwise accuracy 不是逐格完全相同：很小的 score perturbation 会翻转 near-tie。部分真实候选子群上 Current Exact 自身的 log-loss 也可能不优于 Reuse，所以这里正确的因果解释是“shared-only 重建 Current reader 的决策”，而不是“这个带选择偏差的 exposed subgroup 证明 Current 在每个质量切片都更好”。

据此，专家提出的第 1 道门（signed causal intervention）通过，第 3 道门中的“真实 candidate distribution 外部性”也通过。它们支持：

> Current HSTU reader 的主要跨版本兼容性项，确实是每用户一次、跨候选 broadcast 的 signed evidence component；candidate/context residual 很小。

### 11.4 唯一最小机制：matched-cost signed evidence measure

因果门通过后，我们没有扩充 operator，只冻结了一个机制候选。Design 0 对 recent-128 做固定相邻配对，PATCH 64 个 later-event carrier，并把每个 Current carrier 的 V 乘以 represented mass 2。新候选保留相同 pair 和 Current anchor key，但把一个 size-2 pair 的 signed value 写成：

~~~text
V_basis(pair)
  = CAST(V_earlier) + Current(V_later_anchor)
  = [CAST(V_earlier) + CAST(V_later)]
    + [Current(V_later_anchor) - CAST(V_later)]
~~~

第一项是 candidate-independent signed shared value measure，第二项是 Current dependency-closed contextual residual；右式中的 anchor CAST 相消，因此不执行。为了不增加 CAST cost，完整 512-state 把 32 个最旧位置的 joint K/V CAST 预算重分配为 64 个 recent earlier-event value-only CAST：

| 预算轴 | Design 0 | evidence-measure basis |
| --- | ---: | ---: |
| joint K/V CAST 等价位置 | 384 | `352 joint + 64 value-only = 384 joint-equivalent` |
| Current PATCH carriers | 64 | 64 |
| raw repair region | recent 128 | recent 128 |
| materialized / nominal state | 448 / 512 | 448 / 512 |
| mass arithmetic | 32,768 scalar multiply | 32,768 signed add |

这个机制不读取 candidate bank，不拟合 target Current K/V，不使用 PCA/SVD、learned selector、semantic GROUP 或 per-candidate Route。

### 11.5 最小机制 canary：0/5，按合同停止

canary 固定每条 edge 每个 rank 的前 8 名 label-free 用户，即每 edge 32 人、总计 160 user-edge 和 1,598 个真实 rolling 请求。Current、Reuse 和 Design 0 均先与既有 sealed formal raw 重放校验，最大绝对 logit error 为 `7.15e-7`。canary 全程未读取 label。

| edge | requests | Design 0 mean abs logit gap | evidence basis gap | 相对 Design 0 |
| --- | ---: | ---: | ---: | ---: |
| v0→v1 | 282 | 0.01955888 | 0.02302571 | +17.7% |
| v1→v2 | 370 | 0.01480140 | 0.01594743 | +7.7% |
| v2→v3 | 428 | 0.01854624 | 0.01880406 | +1.4% |
| v3→v4 | 292 | 0.02918902 | 0.03362815 | +15.2% |
| v4→v5 | 226 | 0.01921453 | 0.02157264 | +12.3% |

事前 progression gate 要求至少 4/5 edge 不弱于 Design 0，实际为 **0/5**。因此正式 full-population rolling AUC/log-loss 没有启动；不能为了挽救机制去读 canary label、改变 pair、移动 32-position budget 或选择性报告 edge。

这个负结果说明：从“shared causal component 存在”到“把 pair 的 signed V 相加就得到合格 basis”之间仍缺少关键语义。可能缺失的对象包括 Current HSTU 的非线性 key-response、跨层 dependency closure、gate-conditioned interaction，或者 shared base 与 contextual residual 的正确分工；本轮数据不能在这些解释中事后选一个。

### 11.6 更新后的分级结论与专家决策点

| 命题 | 更新后状态 |
| --- | --- |
| norm/normalization 是否制造 candidate-shared 假象 | signed、无 normalization 干预强烈反驳该替代解释 |
| 结构是否只存在于受控 candidate bank | 真实 exposed candidate 五边、四宽度复现 |
| shared component 是否因果决定 Exact−Reuse score gap | 受控 5/5、真实 20/20 组合通过 |
| shared-only 是否重建 Current rolling score/quality | oracle 路径近似重建；不是可执行 action |
| `CAST value measure + Current anchor residual` 是否是合格 basis | canary 0/5，明确否定 |
| 新 basis 是否优于/不弱于 Design 0 rolling AUC/log-loss | 未进入正式质量评价，不能声称 |
| 是否应扩充 Route、semantic GROUP 或 selector | 否；专家意见和当前证据都不支持 |

该轮结束时的待讨论问题已经由最新专家意见具体化并完成实验：不再猜 history basis，而是定位 reader correction、验证跨请求持久性，并且只在两门通过后测试一个直接来自 Current HSTU AV 中间状态的 compact-probe broadcast residual。结果见第 12 节。

新增可复核材料：

- signed causal 合同：`configs/contracts/yambda500m_small_hstu_native_candidate_shared_causal_v1.yaml`；
- signed causal 正式裁决：`results/yambda500m_small_seed17/insight_candidate_shared_causal_v1/adjudication/report.md`；
- 五边真实 exposed quality：`results/yambda500m_small_seed17/insight_candidate_shared_causal_v1/formal_exposed/report.md`；
- 最小机制合同：`configs/contracts/yambda500m_small_hstu_native_evidence_measure_basis_v1.yaml`；
- 最小机制 canary：`results/yambda500m_small_seed17/insight_evidence_measure_basis_v1/canary/report.md`。

本文是面向专家讨论的单篇汇总；数值裁决仍以对应的 sealed aggregate、raw seal 和 adjudication 为准。若讨论后改变机制解释，应新增 prospective contract 验证，不应回写、调参覆盖或选择性替换本轮结果。

## 12. Reader correction 的阶段、持久性与唯一机制 canary

本节记录第二轮专家意见后的完整工作。专家要求首先修正 claim：已证明对象是 candidate-shared **reader compatibility correction**，而不是可物化的 history evidence basis；随后只回答 stage localization、cross-request persistence 和一个 Current-HSTU layerwise broadcast residual 三个问题。我们按此顺序冻结合同，前一门未通过时后一项不会执行。

### 12.1 事前协议与口径

阶段按冻结因果顺序检查：

1. `kv_prefix_contribution`：每个历史位置的 signed `activated(qK)·V`，尚未沿 history 求和且不含 transient self；
2. `av_aggregation`：逐 head prefix 加 self 的聚合；
3. `u_gated_update`：output projection、normalization 与 U gate 后的 update；
4. `layer_hidden`：residual 加 update；
5. `final_readout`：最终 norm 后、score head 前的 hidden。

每一层都在同一个当前 hidden state 上比较 Current-source 与 Parent-Reuse tensor，将 signed delta 分成 candidate mean broadcast 与零均值 residual，再沿 shared-only path 动态推进。这避免把上游误差在后层重复注入。没有做 norm、candidate normalization、target-K/V fitting 或 label 读取。受控 observation 固定 3,000 用户、五边和 width `8/16/32/64`；真实 observation 固定 same-UID/same-timestamp exposed bank、width `2/4/8/16`。

阶段门要求 shared energy 至少 95%、同请求 probability-gap recovery 至少 90%，并分别在受控与真实请求中至少 4/5 edge 通过。这里的 recovery 沿用前一轮 causal adjudication 的 ratio-of-means；Exact−Reuse 近零的请求仍保留 raw 记录，但不会因逐请求零分母制造极端比率。

### 12.2 最早形成阶段

所有 3,000-user 受控记录完成，最大 native/full reconstruction error 为 `3.81e-6`。真实请求正式观察覆盖五边合计 3,063 个 user-edge 和 15,338 个 eligible request group，最大 native error 为 `2.38e-7`。阶段过门数为：

| stage | controlled passing edges | real-exposed passing edges | 裁决 |
| --- | ---: | ---: | --- |
| `kv_prefix_contribution` | 5/5 | 5/5 | **最早稳定边界** |
| `av_aggregation` | 5/5 | 5/5 | 作为首个 post-aggregation persistence stage |
| `u_gated_update` | 4/5 | 5/5 | 更晚，不用于 stage selection |
| `layer_hidden` | 4/5 | 5/5 | 更晚，不用于 stage selection |
| `final_readout` | 5/5 | 5/5 | 更晚，不用于 stage selection |

受控 ratio-of-means recovery 在 `kv_prefix_contribution/AV` 五边均为 `97.93%–99.64%`；真实 exposed 为 `99.26%–99.83%`。K/V contribution 与 AV 的 score recovery 几乎相同，因为二者之间只是对 history contribution 求和并加上 Exact/Reuse 共同的 self term。

这个结果不能被写成“raw K/V 已有可物化 basis”。`kv_prefix_contribution` 已包含 candidate query 的 K-dependent activation；正确结论是：

> 分布式、上下文化的历史侧误差，在 Current reader 第一次执行 query-dependent `activated(qK)·V` 时，就已经呈现 candidate-shared compatibility correction；AV 聚合保留并消费这个修正。

### 12.3 跨真实请求持久性

持久性只在同一用户、同一 E14 edge 的连续 eligible request groups 之间评价。由于冻结 coverage 规则从 512 个旧位置开始，cutover 时不足 512 条历史的用户仍进入同请求阶段观测，但不进入 persistence pair。最终 AV persistence 覆盖 11,364 对相邻请求：

| edge | pairs / users | median direction cosine | same-request recovery | prior-request recovery | coverage-scaled prior recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0→v1 | 2,208 / 567 | 0.982736 | 0.998390 | 0.812668 | 0.840596 |
| v1→v2 | 2,183 / 560 | 0.981558 | 0.996871 | 0.768858 | 0.644569 |
| v2→v3 | 2,385 / 570 | 0.965947 | 0.996061 | 0.706230 | 0.605954 |
| v3→v4 | 2,442 / 601 | 0.981753 | 0.998794 | 0.861577 | 0.816982 |
| v4→v5 | 2,146 / 585 | 0.979111 | 0.997517 | 0.713536 | 0.645629 |

冻结门要求 cosine 至少 0.90、coverage-scaled recovery 至少 0.50，并在至少 4/5 edge 同时通过；实际为 **5/5**。固定时间间隔、append-count 与 remaining-old-state buckets 全部保留，没有按结果选择时间片。它支持“release-time per-user sidecar 可以跨多个请求复用”，而不只是同一 request bank 内共享。

同时，后层并不自动更适合作为 sidecar。以 `v1→v2` 为例，AV 的 scaled prior recovery 为 64.46%，而 U/hidden 仅 0.29%、final readout 为 13.84%；`v2→v3` 的三个后层中位恢复均接近 0。方向 cosine 很高但功能恢复很低，说明不能只看向量方向或 shared energy 来选择注入层。

### 12.4 唯一机制：compact-probe AV broadcast residual

两门通过后，我们只冻结并执行一个机制，没有扩大 operator 空间。它在 cutover 做：

~~~text
Parent Reuse state
  + CAST old-384
  + recent-128 固定连续 group-of-4 -> 32 个 Current dependency-closed carriers × mass 4
  = 一次性 disposable Current probe source

latest pre-cutover item，query delta = 0
  -> Current reader 分别读取 probe source 与 Parent Reuse
  -> 取各层 AV delta
  -> 存为 per-user layerwise sidecar

每个真实 candidate 读取 Parent Reuse
  -> 在各层 AV 广播 sidecar × remaining-old coverage
~~~

这个 sidecar 不读取请求 candidate bank，不用 sampled negative，不拟合 Current target K/V，不做 history-V sum、PCA/SVD、raw semantic GROUP 或 per-candidate selector。生成只用固定的最近历史 item probe；之后对整组候选共享。

成本不超过 Design 0：两者都 CAST old-384、读取 recent-128；Design 0 replay 64 个 Current carriers，而机制 replay 32 个 carriers 加两条单 probe reader path。每层 incremental attention pairs 从 `384×64 + Σ1..64 = 26,656` 降为 `384×32 + Σ1..32 + 2×512 = 13,840`，projection/MLP token 为 34 对 64。持久化只新增四层共 512 个 AV scalar，而不写回 448-position compact cache。

### 12.5 无标签 score canary 结果

canary 固定 prior 3,000-user population 与 E14 请求的交集，再按既有 balanced assignment 取每 rank 前 8 人。五边共 160 user-edge、1,805 个真实请求。Current/Reuse/Design 0 与先前 sealed formal raw 重放，连同 probe path 的最大误差为 `7.15e-7`；全程未读 label。

| edge | requests | Design 0 mean abs logit gap | AV sidecar gap | 相对变化 | 不弱于 Design 0 |
| --- | ---: | ---: | ---: | ---: | --- |
| v0→v1 | 368 | 0.01780833 | 0.01127125 | −36.71% | 是 |
| v1→v2 | 518 | 0.01459615 | 0.00668389 | −54.21% | 是 |
| v2→v3 | 385 | 0.01633045 | 0.00605577 | −62.92% | 是 |
| v3→v4 | 249 | 0.03230288 | 0.03584708 | +10.97% | **否** |
| v4→v5 | 285 | 0.01758990 | 0.00738515 | −58.01% | 是 |

冻结 progression gate 为至少 4/5 edge 不弱于 Design 0，实际 **4/5 通过**。与首个 history-V mechanism 的 0/5 相比，这是实质进展：直接从 Current HSTU AV reader 中间状态生成并跨候选广播，确实比朴素线性历史 basis 更接近已发现结构。

但 `v3→v4` 的 +10.97% 是必须保留的机制反例。当前不能声称机制全面优于 Design 0，也不能据此调 probe item、group size、coverage scale 或选择性报告四条正边。合同规定无论 canary 通过与否都返回专家讨论，不自动读 label、不启动 full-population rolling AUC/log-loss、不做 action admission。

### 12.6 更新后的核心结论

现在可以把这次探索收束为四层证据：

1. **强结构正结果：** Exact−Reuse 的主要功能误差是 candidate-shared signed reader correction；
2. **形成位置：** 它最早在 query-dependent `activated(qK)·V` 形成，AV 是首个可用 post-aggregation 边界；
3. **持久性正结果：** AV correction 跨真实请求稳定，方向与 coverage-scaled 功能恢复均为 5/5 过门；
4. **可执行性初步正结果：** compact-probe AV sidecar 的无标签 score canary 4/5 胜过 Design 0，但尚未通过质量、额外 seed、runtime 或 scale 验证。

推荐系统化的 headline 因而应写成：

> **State generation is user-contextual, while state consumption is candidate-amortized. Cross-version history error becomes a persistent, candidate-shared compatibility correction at the HSTU reader’s query-dependent aggregation boundary.**

建议专家下一轮重点裁决：

1. 是否认可 `activated(qK)·V` 是“reader 形成边界”而非“history basis”的准确术语；
2. 4/5 score canary 是否足以授权事前冻结的 full-population rolling quality，还是应先增加 seed；
3. `v3→v4` 反例应作为必须由 label-free safety gate 捕获的 edge，还是说明 latest-item probe 本身不足；
4. 在不调本 canary 超参数的前提下，下一阶段最小证据应优先是 AUC/log-loss、同机 runtime/I/O，还是额外 seed/规模外推。

新增可复核材料：

- stage/persistence 合同：`configs/contracts/yambda500m_small_hstu_native_reader_compatibility_correction_v1.yaml`；
- 正式阶段与持久性裁决：`results/yambda500m_small_seed17/insight_reader_compatibility_correction_v1/adjudication/report.md`；
- 全量 persistence buckets：`results/yambda500m_small_seed17/insight_reader_compatibility_correction_v1/adjudication/persistence_buckets.csv`；
- 唯一机制合同：`configs/contracts/yambda500m_small_hstu_native_av_broadcast_residual_v1.yaml`；
- 唯一机制 canary：`results/yambda500m_small_seed17/insight_av_broadcast_residual_v1/canary/report.md`。

## 13. 轻量 PRO：把版本变换推入一次 reader read

### 13.1 为什么旧 extractor 仍然不够轻

旧 AV sidecar 已经把持久化对象从 448-position compact K/V 降为四层共 512 个 scalar，但生成器仍先
对 old-384 的每个用户、每层、每位置执行 joint K/V CAST。按与既有成本表相同的口径，CAST 单独
就是 Exact-All 的 32.2%；因此整个 extractor 仍约为 40.5%。版本 map 可以跨用户只构造一次，GPU
batching 也可以减少 kernel overhead，但二者都不会消除逐用户 `384×d²` 的算术。

最新专家意见据此把主方法收敛为 **Per-user Reader Offset（PRO）**：只需要 AV correction 时，不应
先物化没人会持久化的 translated prefix。若每层 Parent joint state 为 `z=[K;V]`，参数 map 分成
`A_K/A_V`，则固定 probe 的 mapped-prefix read 可写成：

~~~text
K' = z A_K, V' = z A_V
activated(q K'^T) V'
  = (activated((q A_K^T) z^T) z) A_V
~~~

实现因此只变换一次 probe query，直接流式读取 Parent joint state，在 history sum 后再做一次
value-space transform；函数不返回 translated prefix cache，也没有该状态的 writeback。

多头 HSTU 带来一个必须诚实处理的成本边界：如果 16/32 个 carrier 的每个 query 也执行上述 wide
joint-state scan，完全等价的 16/32-carrier 方案分别会达到 Full 的 20.6%/40.0%，32-carrier 又失去
轻量性。因此冻结的 lightweight PRO 不这样做。它只对最终固定 probe 使用 reader-pushed map；
recent carrier 在未变换 Parent old-prefix 上执行普通 dependency-closed Current replay。最终流程是：

~~~text
Parent persistent K/V
  -> recent-128 fixed groups -> 32 Parent-conditioned Current carriers
  -> latest-item single probe:
       fused mapped read of Parent old-384
       + native read of 32 Current carriers
       - native read of full Parent Reuse
  -> four layerwise AV offsets (512 scalar)
  -> discard carriers; bounded-horizon candidate-shared serving
~~~

这里仍保留 parameter-only version-map **语义**，但彻底取消了旧式 per-position CAST state 的物化和
持久化。Design 0 与旧 40.5% extractor 只保留为实验 reference/baseline，不是主方法的前置阶段。

### 13.2 事前合同、v1 失败和协议修正

第一份合同固定五边、每边 selector-order 前 32 名用户、16/32 carrier 轴、latest-item probe，以及
`2e-5` fused-reference 最大绝对误差。它的成本门和零物化结构门通过，但绝对 AV 误差达到
`0.015625`，因此 v1 按合同保留为 **FAIL**。诊断表明 AV 是未归一化的大幅值中间张量：同一批记录
的最大相对 L2 仅 `4.69e-6`，最终 sidecar replay 最大绝对误差仅 `4.77e-7`。失败来自把 logit
replay 的 dimensionful 绝对阈值错误复制到 AV tensor，不是可以静默改写的通过结果。

随后新增 v2 合同，显式引用并保留 v1 failure，不改变模型、map、probe、carrier、group 或 scale。
v2 按 FP32 dot/reduction 的尺度无关误差界冻结 `relative L2 <= 1e-4`，同时继续要求最终 replay
`abs <= 2e-5`；验证人口换成 selector-order 第 33–64 名、未参与 v1 的 32 个用户。这个版本化协议
修正避免用同一批样本把失败阈值事后调成通过。

### 13.3 v2 正确性与成本结果

held-out v2 共 320 条 user-edge×carrier observation，全程不读取 request candidate、score 或 label：

| 门 | 结果 | 裁决 |
| --- | ---: | --- |
| fused-reference 最大相对 L2 | `4.7269e-6` | 通过 `1e-4` |
| sidecar replay 最大绝对误差 | `3.5763e-7` | 通过 `2e-5` |
| action 内物化 translated-prefix positions | `0` | 通过 |
| 16-carrier 理论 FLOPs | `0.0326 GFLOPs/user`，Full 的 `5.2%` | 通过成本门 |
| 32-carrier 理论 FLOPs | `0.0572 GFLOPs/user`，Full 的 `9.1%` | 通过成本门 |
| FP32 sidecar write | `2.0 KiB/user` | 512 scalar |
| Parent state read | `2.0 MiB` unique；`5.0 MiB` conservative logical streams | runtime 待测 |

版本 map 的 pseudoinverse 每条 release edge 构造一次而非每用户一次；本次单卡过程除首次初始化
`0.207s` 外，其余 edge 约 `0.018s`。这些是 canary 过程记录，不是生产 latency claim。

32-carrier 是事前 primary，16-carrier 只用于小成本轴诊断。相对旧 materialized-prefix sidecar：

| edge | 32-carrier direction cosine mean | norm ratio median | relative L2 mean |
| --- | ---: | ---: | ---: |
| v0→v1 | 0.998947 | 0.982817 | 0.050964 |
| v1→v2 | 0.999312 | 0.993264 | 0.036392 |
| v2→v3 | 0.999207 | 0.998217 | 0.036690 |
| v3→v4 | 0.998317 | 0.990304 | 0.064914 |
| v4→v5 | 0.999087 | 0.993858 | 0.045625 |

16-carrier 的五边 cosine mean 为 `0.8761–0.9646`、relative L2 mean 为 `0.2494–0.4976`，明显弱于
primary 32。由于 carrier 轴和 primary 已在结果前冻结，这个观察不构成事后选择；后续正式机制仍
只携带 32-carrier 配置。

### 13.4 正确性/成本门结束时的阶段裁决

轻量 PRO 已经回答三个必要问题：reader-pushed joint map 与 materialized read 在 scale-aware FP32
口径下等价；新 action 不物化 translated prefix；完整 per-user 算术在公平口径下约为 Full 的 9.1%。
它还高度保持旧 sidecar 的方向和幅值，因此不是为了降成本而换成完全不同的 correction。

该轮刻意没有读取 request score 或行为 label。sidecar 相似不等于 rolling quality 相同，9.1%
FLOPs 也不等于 90.9% latency reduction；Parent state bandwidth、raw recent-history I/O、kernel
utilization 和 serving sidecar read 仍需独立 runtime。当时的合同裁决是：

> **轻量 PRO 通过无标签正确性、零物化结构和理论成本门；返回专家讨论。尚未解锁 score canary、
> full-population AUC/log-loss、action admission、额外 seed 或长训练。**

新增可复核材料：

- v1 合同与保留失败：`configs/contracts/yambda500m_small_hstu_native_pro_lazy_reader_v1.yaml`、
  `results/yambda500m_small_seed17/insight_pro_lazy_reader_v1/correctness_cost/`；
- v2 scale-aware held-out 合同与通过结果：
  `configs/contracts/yambda500m_small_hstu_native_pro_lazy_reader_v2.yaml`、
  `results/yambda500m_small_seed17/insight_pro_lazy_reader_v1/correctness_cost_v2/`；
- 实现与成本模型：`scripts/insight/pro_lazy_reader.py`、`scripts/insight/pro_lazy_cost.py`；
- 单元测试：`tests/test_pro_lazy_reader.py`。

### 13.5 五边全人口 rolling quality

正确性/成本门之后新增独立 prospective quality 合同，机制不再改变：primary `carrier=32`、
recent-128 固定 chronological group-of-4、latest pre-cutover item probe、mass 4、AV injection 和
`max(0,512-evictions)/512` coverage decay 全部冻结。满 512 历史的用户生成一次 2 KiB sidecar；
cutover 历史不足 512 的用户按事前 label-free 规则使用 Reuse No-op。Design 0 只从既有 sealed raw
读取同请求比较值，不参与 PRO 执行。

正式运行前，五边各取每 rank 前 8 名冻结分配用户，共 160 user-edge、1,598 个无标签请求。基线
重放最大误差为 `7.15e-7`，translated-prefix positions 为 0；PRO 对 Current 的 mean absolute
logit gap 在 5/5 edge 均低于 Design 0，因此解锁全人口 raw-first 运行。正式结果为：

| edge | requests | PRO−Reuse AUC (pp) | PRO−Reuse log-loss | Reuse harm recovered | PRO release-gain retained | PRO−Design 0 AUC (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0→v1 | 43,186 | +0.091723 | −0.00019437 | 30.0% | 82.1% | −0.180581 |
| v1→v2 | 41,655 | +0.169338 | −0.00023962 | 84.8% | 95.2% | −0.142113 |
| v2→v3 | 43,092 | +0.004930 | +0.00001871 | 3.3% | 53.6% | −0.104520 |
| v3→v4 | 43,945 | +0.045681 | +0.00003750 | 23.5% | 不稳定（−220.1%） | −0.044873 |
| v4→v5 | 45,706 | +0.020374 | −0.00000476 | 32.0% | 80.3% | +0.286137 |

五边非加权平均 `PRO−Reuse` 为 `+0.066409` AUC pp、`−0.00007651` log-loss。与较重的 Design 0
相比，PRO 在前四边 AUC 较低，但在 Design 0 原先失败的 `v4→v5` 上反超 `+0.286137` pp；因此
lightweight PRO 不是无损复现旧方法，而是一条成本更低、恢复幅度较小但 AUC 边际更稳定的路径。

`v3→v4` 需要单独解释但不能删除。该 edge 的 E14 Full-only `Current−Parent` AUC 只有
`0.046331` pp，release-gain retention 分母极小；同时 rolling Current Exact log-loss
`0.33651878` 本身差于 Reuse 的 `0.33643834`。PRO 的 `0.33647584` 位于二者之间：它在 AUC 上
向 Current 恢复，同时在 log-loss 上也向这个较差的 Current 目标移动。因此这条结果更像 release
排序收益与概率校准目标冲突，不是 reader-pushed map 数值失准或 sidecar 完全失效。`v2→v3` 则是
真正的弱恢复 edge：AUC 只提高 `0.004930` pp，log-loss也轻微恶化，应进入未来 label-free safety
signal 的 development evidence。

### 13.6 双重裁决：方法可继续，当前版本未获上线资格

事前 formal gate 要求 AUC 与 log-loss 各至少 4/5 edge 不差于 Reuse，并要求五边均值同向。实际
AUC 为 5/5、log-loss 为 3/5，均值两项均通过，所以严格 gate 因单项 edge count 记为 **FAIL**。
这个记录不能在看到结果后改写。

同时，完整结果足以支持更上层的研究裁决：若 Design viability 的标准是“总体均值为正，且至少
过半 edge 正向”，轻量 PRO 明确 **PASS**。它证明了一个 9.14%-of-Full、零 translated-prefix、
candidate-amortized 的单一设计可以恢复真实推荐质量，而不是只有内部向量相似性。这里没有
`Design 0 -> Design 1` 串联；服务方法只有 PRO，Design 0 是历史强对照，Reuse/Exact 是 fallback/
上界。

因此下一阶段可以研究 admission/calibration 或机制改进，但必须遵守两个边界：

1. 本次五边及 label 已成为 development evidence；不能在其上调 probe、carrier、mass 或 coverage
   后，再把同一 Small/seed17 结果当成无偏 qualification。
2. 改进版本应先冻结 label-free signal 和 No-op/Reuse fallback，再在新 training seed 或新冻结
   release edge 上验证；runtime 仍需实测，理论 9.14% 不等于 GPU latency 9.14%。

新增可复核材料：

- quality 合同：`configs/contracts/yambda500m_small_hstu_native_pro_lazy_rolling_quality_v1.yaml`；
- 五边 canary：`results/yambda500m_small_seed17/insight_pro_lazy_reader_v1/rolling_quality_v1/canary/`；
- 五边正式 raw、seal、逐边 adjudication 与总表：
  `results/yambda500m_small_seed17/insight_pro_lazy_reader_v1/rolling_quality_v1/formal/`；
- evaluator、raw-first adjudicator 与 runner：
  `scripts/evaluate_yambda500m_hstu_native_onehop_reuse_raw.py`、
  `scripts/insight/adjudicate_pro_lazy_quality.py`、
  `scripts/insight/run_pro_lazy_rolling_quality.py`。

## 14. 最新专家增量意见：从轻量 PRO 走向自校准 progressive PRO

本节是给下一轮专家讨论使用的统一增量记录。它不改写第 13 节已经封存的结果，也不把尚未运行的
方案写成实验结论。当前可以确定的是：轻量 PRO 已经证明结构路线成立，但它对 Exact reader
correction 的恢复仍偏弱；最新建议是在同一个 PRO 内提高 estimator fidelity，而不是重新发散到
semantic group、embedding selector、per-candidate route、history-token basis 或新的 operator catalog。

### 14.1 已验证事实与最新解释

已完成的证据保持不变：

| 项目 | 已验证结果 | 当前含义 |
| --- | ---: | --- |
| PRO 理论计算量 | Full 的 `9.14097%` | 已是低成本 candidate-amortized reader；不是完整重算 |
| translated-prefix 物化 | `0` positions | 版本变换已经推入一次 Current reader read |
| rolling AUC | 相对 Reuse `5/5` 正向 | 结构路线具备真实推荐质量可行性 |
| rolling log-loss | 相对 Reuse `3/5` 正向 | 当前 estimator/calibration 尚不够稳定 |
| `v2→v3` | AUC 仅 `+0.004930` pp | 是真正的弱恢复 edge，应作为 fidelity/admission 开发证据 |
| `v3→v4` | AUC `+0.045681` pp；log-loss `+0.00003750` | Current Exact 的 log-loss 本身差于 Reuse，不能要求迁移方法在该指标上优于目标 Current |

因此最新专家判断不是“PRO 的核心 Insight 失败”，而是：现有 `carrier32 + 单 latest-item probe +
全局 coverage decay` 是一个正确但低保真的 correction estimator。它已经抓住稳定方向，却可能低估、
误估不同 layer 和不同历史段在滚动过程中的 correction 幅值。

### 14.2 拟冻结的单一设计核心

后续仍只有一个服务设计，暂称 **self-calibrating progressive PRO**，不存在串联的 Design 0/Design 1：

1. release 时按用户读取一次 Parent persistent state，并用少量 Current-version carriers 执行
   candidate-independent reader probe；
2. 每层持久化稳定 correction direction，而不是物化一份 translated history prefix；
3. 使用两条固定、与候选无关的 probe 估计每层幅值，避免单 latest-item probe 偶然失准；
4. 将 correction 分成 old-prefix 与 recent-context 两个幅值分量；后续 append/eviction 只更新这两个
   scalar coverage，不重新生成用户 sidecar；
5. 同一 sidecar 在整个 candidate bank 上广播和摊销；只有 probe disagreement 触发事前冻结的
   carrier precision 升级或 No-op/Reuse fallback。

形式上，每层 correction 写成：

```text
delta_AV_l(t) = d_hat_l * [alpha_old,l * c_old(t) + alpha_recent,l * c_recent(t)]
```

其中 `d_hat_l` 是两条固定 probe 共同估计的持久方向；`alpha_old,l`、`alpha_recent,l` 是 release-time
幅值；`c_old(t)` 与 `c_recent(t)` 只由已发生的 append/eviction 计数决定，不读取未来 label。对于当前
`old384 + recent128` 布局，old coverage 先随前 384 次 eviction 衰减，recent coverage 在此期间保持，
随后才在最后 128 次 eviction 中衰减。这比原先把全部 512 positions 用一个线性 scalar 同步衰减
更符合真实 cache lineage，同时仍满足“per-user once, candidate-amortized”。

carrier 数量不是三个不同设计，而是同一 PRO 的 fidelity/cost 轴：`C32` 为默认约 10% 档，`C48`
和 `C64` 分别用于约 15%/20% 档。最终只冻结一个 deployment policy；不会在看到质量 label 后按
edge 选择配置。

### 14.3 下一轮先做的 label-free oracle decomposition

现有五条 Small/seed17 edge 及其 label 已经成为 development evidence，不能再用于改进版本的正式
AUC/log-loss qualification。它们仍可用于以下无标签机制诊断：

1. 逐层比较 PRO correction 与 Exact shared AV correction 的 direction cosine；
2. 比较 norm ratio，判断误差是否主要来自 amplitude underestimation；
3. 同时看 cutover 与真实 rolling request，区分 release-time extractor 误差和 coverage decay 误差；
4. 比较 single-probe 与 dual-probe disagreement，判断一个持久方向是否足够；
5. 在同一统一 PRO 上报告约 10%/15%/20% Full FLOPs 的 label-free fidelity frontier。

该分解只允许读取模型内部 state、真实已曝光 candidate 和 request 时间线，不读取行为 label，不进行
target-KV fitting，也不改变已经封存的 quality 结果。如果稳定方向不能跨 layer/request 成立，才按
事前规则考虑第二 component；不会重新开放新的结构目录。

### 14.4 五边 label-free oracle decomposition 结果

正式分解使用每边 64 个 cutover 用户和 64 个真实 rolling 用户；每个 rolling 用户固定取 first/last
eligible exposed request。共封存 10,240 条逐层记录和 4,480 条 score 记录，最大 reader trace 误差
为 `2.86e-6`，所有 raw seal 复核一致，行为 label 未读取。

| 冻结问题 | cutover | rolling | 裁决 |
| --- | ---: | ---: | --- |
| C32 dual-probe direction cosine ≥0.90 | 2/5 edge | 0/5 edge | 稳定方向门 FAIL |
| oracle amplitude 使 relative L2 至少下降 25% | 0/5 edge | 0/5 edge | amplitude-dominant 门 FAIL |
| 两条 probe 的 cosine/norm 一致性 | 5/5 edge | release-time 固定 | PASS，但说明第二 probe 基本冗余 |
| segment decay 不差于 global decay | — | 2/5 edge | FAIL，frontier 保留 global decay |

逐 edge 的 probe cosine 中位数为 `0.999984–0.999995`，norm ratio 为
`0.999583–1.000119`。因此 latest item 与 recent-window-start item 给出的 correction 几乎相同，
增加同类 candidate-independent probe 没有补足缺失方向。old/recent component 自身的方向 cosine
仅约 `0.47–0.78`，但事前第二 component 规则绑定的是两条 probe disagreement；实际为 0/5 edge，
所以没有在看到结果后追加 component。

segment decay 的五边平均 relative L2 略低于 global，但只在 `v0→v1`、`v3→v4` 两边逐 edge
不差，未达到冻结的 4/5 门。故原来“old384 先衰减、recent128 后衰减”的直觉没有获得足够一致的
模型内部证据，正式 frontier 继续使用单一 global coverage decay。

结论是：专家关于“先分方向与幅值”的诊断非常有价值，但数据不支持把 C32 误差归结为幅值低估。
当前误差同时包含 extractor direction 和 rolling transport drift；只校准 scalar 不够。

### 14.5 10%/15%/20% carrier fidelity frontier

随后冻结新的 held-out label-free 样本：cutover selector-order 第 65–128 名，以及 rolling eligible
order 第 65–128 名，与 decomposition 人口错开。C32/C48/C64 仍是同一个双 probe、单 direction、
global-decay PRO，只改变 carrier 数量。正式 raw 包含 11,520 条逐层记录、4,800 条 score 记录和
7,680 条 self-convergence 记录；最大正确性误差为 `2.38e-6`。

| point | 理论 Full FLOPs | sidecar | cutover 五边平均 relative L2 | rolling 五边平均 relative L2 |
| --- | ---: | ---: | ---: | ---: |
| C32 | 10.52% | 2,080 B/user | 0.60851 | 0.73331 |
| C48 | 14.54% | 2,080 B/user | **0.58437** | **0.69700** |
| C64 | 18.64% | 2,080 B/user | 0.58541 | 0.70121 |

C64 相对 C32 的 relative L2 在 cutover 与 rolling 均为 **5/5 edge 改善**，五边均值也均降低；
对 Current Exact 的 mean absolute logit gap 在 cutover 5/5、rolling 4/5 不差。这说明增加 carrier
确实带来部分 fidelity 收益，不是完全无效的增量。

但事前完整选择门没有通过：C64 的 `median cosine≥0.90` 只在 cutover 3/5、rolling 0/5 edge
成立；同时 C48 的五边平均 relative L2 在两个 phase 都略低于 C64，并不位于 C32 与 C64 之间。
因此这不是一个单调、可由“更多 carrier=更高精度”解释的 frontier。按照结果前冻结的规则，不能
看到 C48 较好后把它事后选成 Design，也不能只用 C64 相对 C32 的改善忽略 absolute-direction 门。

最终裁决为：

> **progressive PRO 增量诊断有效，但本轮升级不入选。正式设计继续保留已完成真实质量验证的
> C32 lightweight PRO；不启动旧五边 AUC/log-loss 重测，不获得 serving、new-seed、runtime 或
> scale admission。**

这不是核心 Insight 或原 PRO 的反证。原 C32 的 AUC 5/5 正向结果仍然成立；被否定的是一个更窄的
增量假设：仅靠第二条近乎等价的 probe、layer scalar、segment decay 和更多同构 carriers，就能形成
自校准且单调的升级。

### 14.6 给专家的最新讨论点与材料

建议专家本轮只讨论两件事：

1. 是否同意把 progressive 增量记录为“有一致 fidelity 收益、但未形成可冻结的单调 precision axis”，
   因而保留 C32 主设计；
2. 是否到此停止同一 Small/seed17 上的 estimator 调整，把后续资源优先用于原 C32 的新 seed/runtime
   qualification，而不是继续在已成为 development evidence 的五边上调 carrier/probe。

新增可复核材料：

- decomposition 合同：
  `configs/contracts/yambda500m_small_hstu_native_progressive_pro_decomposition_v1.yaml`；
- decomposition formal raw、seal、aggregate 与裁决：
  `results/yambda500m_small_seed17/insight_progressive_pro_v1/decomposition_v1/formal/`；
- frontier 合同：
  `configs/contracts/yambda500m_small_hstu_native_progressive_pro_frontier_v1.yaml`；
- frontier formal raw、seal、cost、aggregate 与裁决：
  `results/yambda500m_small_seed17/insight_progressive_pro_v1/frontier_v1/formal/`；
- progressive primitives、执行器与 raw-first adjudicator：
  `scripts/insight/progressive_pro.py`、
  `scripts/insight/probe_progressive_pro_decomposition.py`、
  `scripts/insight/adjudicate_progressive_pro_decomposition.py`、
  `scripts/insight/probe_progressive_pro_frontier.py`、
  `scripts/insight/adjudicate_progressive_pro_frontier.py`。

给专家的一句话摘要：

> 原 C32 PRO 的 9.14%-Full、AUC 5/5 正向结论保持；最新无标签增量证明更多 carrier 可在 5/5 edge
> 降低 C64 对 C32 的内部 relative L2，但双 probe 几乎完全冗余、纯幅值/segment-decay 假设未过门，
> 且 C48/C64 非单调、rolling absolute direction 仍未达标，因此不事后挑配置，保留 C32 主设计。

## 15. Small 收口与 Medium 推进裁决（2026-08-28）

专家最新裁决已接受：Small/seed17 的作用是发现机制、打通实现和暴露边界；不再在相同五边上调整
carrier、probe 或 decay。正式 Insight 冻结为 persistent candidate-shared AV compatibility correction，
正式 Design 冻结为原 C32 lightweight PRO。完整冻结记录见
`small_insight_design_freeze_2026-08-28.md`。

下一步优先建立 Medium Full-only release environment，而不是直接把 C32 搬过去读质量。经过数据审计，
day150 会使现有 Medium 30k population 中 3,346 人缺少基础历史，并需重选约 11.2% 用户及重建 item
mapping。因此最终沿用既有 day217 boundary，只比较 D7 与 D14：

- 共享 Medium v0：30k fixed UIDs、6L/H192/context1024、foundation `[0,217)`；
- D7：10 个增量 candidate，逐 edge 评测 E3/E7；
- D14：4 个增量 candidate，逐 edge 评测 E3/E7/E14；
- complete source 只到 `[0,300)`，partial day300 排除，不为增加版本数使用残缺评测窗口。

这一步先只训练、seal 和裁决 Parent/Current Full。Reuse、PRO、Medium core Insight gates、额外 seed 与
runtime 都保持锁定；长训练仍须 prospective contract、资源估算、focused canary 和用户显式 launch。
具体代码改造、窗口表与资源预算见
`docs/medium_scale_training_plan.md`。

更新后给执行者的一句话：

> **Small discovery 到此冻结；下一阶段按 day217 的既有 Medium population/mapping 准备一份共享 v0、
> D7×10 与 D14×4 的 Full-only prospective scan，先验证模型更新环境，再决定是否解锁冻结 C32 的
> Medium scale qualification。**
