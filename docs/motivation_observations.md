# 核心 Motivation 与 Observation

更新日期：2026-08-26

本文是当前论文最详细的结果文档。它只记录已经观察到的 HSTU-native motivation 和版本化状态 observation；没有观察到的系统设计效果、最终 EvoKV action 或 scheduler 结论不写成已验证结果。

## 1. 当前结论

在 Yambda-500M Small 的 HSTU-native foundation 上，模型更新带来的发布收益与 persistent KV 的跨版本兼容性是两个不同问题：

- Current Full 在多条连续版本边上优于 Parent Full；
- 直接复用父版本 prefix KV 会使当前模型的 rolling 质量低于 Current Exact Rolling；
- 在四条常规正收益边上，One-hop Reuse 侵蚀了 25.5%–47.9% 的模型发布 AUC 收益；
- 直接 Reuse 的损失随 producer version age 增长，在 long-age direct matrix 中呈严格单调关系；
- request-level Reuse harm 与 matched Parent→Current benefit 稳定重合；
- controlled dilution 表明少量 Current state 的重新锚定比单独 eviction 更能解释 gap 衰减；
- pre-cutover tail replay 在不加入新行为信息时也能形成 Current-version anchor；
- recommendation risk 对 candidate 是否为历史新颖项具有稳定差异，HSTU 机制则表现为 K/V 共同不兼容与 early/middle propagation；
- 完整活跃用户分析和 hybrid producer 诊断不支持 isolated item-embedding drift；主要来源是 contextual Transformer co-adaptation；
- parameter-only joint state translation 与 Current residual rematerialization 在五条 edge 上稳定互补；
- HSTU sidecar state 必须保留 aggregation mass；无权重压缩会系统性破坏读取语义；
- 事前固定的 one-release `CAST(prefix384) + GROUP/PATCH(recent128->64) + SCALE(2)`
  已在五条完整 D14/E14 rolling edge、217,584 个请求上执行；它在 4/5 条 edge 上提高 Reuse
  AUC，前三条分别保留 97.2%、117.9% 和 87.3% 的既有 Full-only 发布收益；
- v4->v5 上固定计划反而比 Reuse 低 0.265765 AUC point，因此当前结果证明的是“一次转换链可执行且
  经常有效”，不是一个可对所有 release 无条件采用的固定计划；
- 这支持“版本化 persistent state 会阻碍新模型收益兑现”这一 motivation；
- 这还没有证明 recursive lineage debt、最终 migration policy 或 scheduler 应该采用哪种具体设计。

## 2. 当前结果的实验对象

- 数据：Yambda-500M，Explicit Feedback；
- 时间线：约 300 天；
- foundation：Day 0–217；
- 当前 Small：固定 UID hash 人口，约 10,000 用户；
- 模型：HSTU-native，4L/H128/context512；
- seed：17；
- item mapping：foundation cutoff 前固定，未来 item 使用 256 个稳定 OOV bucket；
- 版本链：v0 → v1 → v2 → v3 → v4 → v5；
- 核心切片：每次 update D=14 天，发布后观察 E=14 天；
- 评测：同一用户、同一 causal history、同一 query/target/candidate、同一当前模型与 readout。

## 3. 对照定义

### 3.1 Full-only release gain

Parent Full 和 Current Full 都在同一未来请求上完整重算：

~~~text
Release gain = Current Full - Parent Full
~~~

该差值只回答新模型是否比父模型好。它不读取 Reuse、JS、KV distance、release debt 或 scheduler 输出，因此不会用 compatibility 结果决定模型 admission。

### 3.2 Rolling reuse harm

Current Exact Rolling 和 One-hop Reuse Rolling 共享同一 rolling 执行语义：

- Current Exact Rolling：cutover prefix 由当前模型重算，随后当前模型 append；
- One-hop Reuse Rolling：cutover prefix 使用父版本 KV，随后当前模型 append；
- 两条路径使用相同的 query、target、candidate、eviction 和 append 规则。

~~~text
Reuse harm = Current Exact Rolling - One-hop Reuse Rolling
~~~

对 log-loss 使用相反方向：

~~~text
Reuse loss = LogLoss(One-hop Reuse) - LogLoss(Current Exact)
~~~

## 4. D=14、E=14 核心 AUC 结果

所有 AUC 差值单位为 percentage points。

| 版本边 | Parent Full | Current Full | 发布收益 | Current Exact Rolling | One-hop Reuse | Reuse 损失 | 侵蚀比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 → v1 | 0.669198 | 0.681196 | +1.199879 | 0.681769 | 0.678709 | +0.306014 | 25.5% |
| v1 → v2 | 0.663884 | 0.670146 | +0.626216 | 0.670493 | 0.668496 | +0.199662 | 31.9% |
| v2 → v3 | 0.611481 | 0.614582 | +0.310144 | 0.615001 | 0.613514 | +0.148693 | 47.9% |
| v3 → v4 | 0.619490 | 0.619953 | +0.046331 | 0.620994 | 0.619054 | +0.193984 | 418.7%* |
| v4 → v5 | 0.549289 | 0.551495 | +0.220611 | 0.551858 | 0.551221 | +0.063736 | 28.9% |

* v3→v4 的绝对 Reuse 损失真实存在，但 Full-only 发布收益很小，导致百分比分母放大。它是高风险尾部案例，不是 headline 平均效果。

四条常规正收益边的侵蚀比例为 25.5%–47.9%。这组结果同时展示了：

1. 新模型可以确实变好；
2. 父 KV 可以阻碍新模型兑现收益；
3. 侵蚀比例不是固定常数；
4. 不能因为某一条边不 harmful 就否定整个问题，也不能因为某一条边很大就声称所有 release 都有害。

### 4.1 固定 One-Release 方案的完整 rolling AUC

在不改变上述五条正式 D14/E14 edge、请求人口、Current reader、label 或 rolling 语义的前提下，
新增一条事前固定的 `Our` 路径：对旧 384-position prefix 做 parameter-only CAST，将 recent-128
按相邻 evidence 两两 GROUP 为 64 个 carrier，用 Current model PATCH，再以 represented mass 2
执行 SCALE。该路径只处理一次 Parent->Current 转换，不递归复用自己的近似结果。

为严格沿用前一组 motivation 的口径，令：

~~~text
G_full = AUC(Current Full) - AUC(Parent Full)
AUC_old_ref = AUC(Current Exact Rolling) - G_full

Retained(path)
  = [AUC(path) - AUC_old_ref]
    / [AUC(Current Exact Rolling) - AUC_old_ref]
  = 1 - [AUC(Current Exact Rolling) - AUC(path)] / G_full
~~~

这里的 `AUC_old_ref` 只是把既有 Full-only gain 平移到相同 rolling 轴上的参考点，不是新增的一条
模型路径。它使下面两列精确对应“`AUC(Reuse/Our)-AUC(old)` 除以
`AUC(Recompute)-AUC(old)`”的既有问题定义。作为不依赖该分母的 companion，同时报告
`Our-Reuse` 和：

~~~text
Reuse harm recovered
  = [AUC(Our) - AUC(Reuse)]
    / [AUC(Current Exact Rolling) - AUC(Reuse)]
~~~

| 版本边 | 请求数 | Recompute AUC | Reuse AUC | Our AUC | Reuse 保留收益 | Our 保留收益 | Our - Reuse（pp） | Reuse harm 挽回 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 -> v1 | 43,186 | 0.681769 | 0.678709 | 0.681432 | 74.5% | 97.2% | +0.272304 | 89.0% |
| v1 -> v2 | 41,655 | 0.670493 | 0.668496 | 0.671611 | 68.1% | 117.9% | +0.311450 | 156.0% |
| v2 -> v3 | 43,092 | 0.615001 | 0.613514 | 0.614609 | 52.1% | 87.3% | +0.109449 | 73.6% |
| v3 -> v4 | 43,945 | 0.620994 | 0.619054 | 0.619960 | -318.7%* | -123.2%* | +0.090555 | 46.7% |
| v4 -> v5 | 45,706 | 0.551858 | 0.551221 | 0.548563 | 71.1% | -49.4% | -0.265765 | -417.0% |

正式 E14 population 上，五条 edge 都满足 `Recompute AUC > Reuse AUC`；此前小型 probe 中出现的
方向反转不属于这组正式 motivation 数据。Our 在前四条 edge 改善 Reuse，前三条恢复最明显；
v1->v2 超过 100% 表示 Our 的聚合 AUC 略高于 Recompute，不表示其 state 比 Exact 更精确。

* v3->v4 的 Full-only gain 只有 0.046331 point，远小于 rolling Reuse harm，因此 retained ratio
对很小的绝对变化极其敏感。该 edge 应主要阅读 `Our-Reuse=+0.090555 point` 和
`harm recovered=46.7%`，不把负百分比作为 headline。v4->v5 则不是分母伪影，而是固定计划的真实
反例：它比 Reuse 和 Recompute 都差。

该实验在 label join 前封存固定计划及输入 hash，并要求重放的 Current/Reuse logit 与既有 sealed raw
一致；五条 edge 的最大绝对重放误差不超过 `9.54e-7`。因此它补上了完整 rolling AUC 资格，但尚未
补上端到端 GPU、raw-history I/O、state I/O 和 makespan 成本，也不能使用这五条 qualification label
事后调整 `r/c`。

## 5. Companion 质量与表示结果

| 版本边 | Full-only PR-AUC Parent → Current | Rolling PR-AUC Current → Reuse | Reuse − Current event log-loss | 用户等权 log-loss 差 | Bernoulli JS |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0 → v1 | 0.193987 → 0.206501 | 0.206536 → 0.204757 | +0.000502 | +0.000089 | 1.69e-05 |
| v1 → v2 | 0.186811 → 0.194637 | 0.194268 → 0.192455 | +0.000274 | +0.000110 | 3.18e-06 |
| v2 → v3 | 0.152918 → 0.153150 | 0.153849 → 0.153327 | +0.000195 | −0.000041 | 2.31e-06 |
| v3 → v4 | 0.159239 → 0.160395 | 0.159553 → 0.159566 | −0.000080 | −0.000095 | 2.27e-05 |
| v4 → v5 | 0.133822 → 0.135537 | 0.134760 → 0.134218 | +0.000042 | +0.000005 | 2.31e-06 |

AUC 的损害不需要在所有 loss 口径上同幅出现。它可能集中在排序边界、部分用户或部分请求，因此主结论使用配对 AUC release gain 与 rolling Reuse harm，同时保留 PR-AUC、log-loss、Brier、用户等权差异和 JS 作为 companion。

## 6. 版本年龄 observation

在 direct long-age matrix 中，producer 版本越旧，Current 与 Reuse 的质量差异越大。D=14 的 long-age 结果显示：

- one-hop 是最小年龄的直接父版本对照；
- 更老 producer 的 Reuse 损失随版本年龄严格单调增长；
- v4←v0 的 ROC-AUC 损失约为 one-hop 的 3.6 倍；
- 这支持 version age 是 persistent-state compatibility 的重要维度。

该结论是 direct Reuse age observation，不是 recursive lineage debt 的证明。Recursive lineage 需要单独的真实 append/eviction chain 和预注册质量评测。

### 6.1 Insight first pass（描述性）

在不重跑训练的前提下，`scripts/insight/analyze_first_pass.py` 对五条 D14/E14 edge 的
217,584 个配对请求做了第一轮 request-level 分析：

- `G = loss(Parent Full) - loss(Current Full)` 与
  `H = loss(Reuse) - loss(Current Exact Rolling)` 的 Spearman 相关在五条 edge 上均为正，
  范围为 0.323–0.606；
- 正 Reuse harm 落在 `G > 0` 请求上的集中程度，相对这些请求的人口占比为
  1.10–3.57 倍；
- remaining-old fraction 与 Current–Reuse absolute probability shift 的 Spearman 相关在
  五条 edge 上为 0.573–0.739；remaining-old state 为零时的平均 shift 比 remaining fraction
  大于 0.75 时低约 28–74 倍；
- append count 与 remaining-old fraction 的 Spearman 相关为 −0.956 至 −0.981，因此现有
  rolling trace 本身不能区分 eviction 与 current-version anchor effect。

这些结果支持“Reuse harm 与新模型的 request-level benefit 有结构性重合”作为下一步
History Utility × State Staleness 实验的候选方向。它仍是 seed17 Small 上的描述性 discovery；
v3→v4 在平均 log-loss 上的 `G` 和 `H` 均为负，也说明不能把该 observation 写成所有质量
口径上的统一正效应。完整聚合见
`results/yambda500m_small_seed17/insight_first_pass_v1/report.md`。

在每条 edge 的 128 个 append-free、512-token 请求上进一步使用 matched Parent Exact、
Current Exact 和 Current-read-Parent-KV 对照后，request-level `G/H` Spearman 为 0.489–0.889，
positive-harm concentration lift 为 1.05–4.57。该小规模 matched 结果说明 benefit/harm overlap
不是简单由 Full 与 rolling 路径混合造成的；它仍需更大 cohort/seed confirmation。

### 6.2 Small History Utility 与 HSTU 机制 probe

随后在每条 edge 的前 256 个活跃 UID 首请求上，用 Current Full、recent-128 和 recent-32
做了 history truncation probe。Utility 与 Reuse harm 的相关很弱：recent-32 的 Spearman
仅为 0.051–0.123，recent-128 为 −0.007–0.152；正 harm 的 concentration lift 也在
0.90–1.26 之间，没有形成稳定分离。candidate repeat、历史多样性、organic fraction 和
recent/old overlap 的方向跨 edge 改变。当前结果不支持按这些用户或推荐语义直接冻结
selective migration policy。需要注意，当前 persistent history 是带 organic/non-organic
behavior 的 listen token；like/dislike 是请求 label，不是 persistent action token。

将 Utility、Regional Staleness 和 refresh recovery 对齐到同一个 old384/recent128 或
old480/recent32 区域后，相关仍跨 edge/region 改变：Spearman 为 −0.353–0.279，positive
staleness concentration lift 为 0.44–1.19。当前结果因此明确不支持
`History Utility × Regional Staleness` 作为已经成立的中心规律，也不支持据此冻结
semantic-region scheduler。

在每条 edge 的 128 个 append-free、512-token 请求上，进一步进行了诊断性 exact K/V
splice。只替换 Current K 或只替换 Current V 都能在所有 edge 上减少输出 gap，但不存在稳定的
单侧主导；把 lower 3/4 layers 的 K/V 替换为 Current 后，可恢复 83.4%–96.0% 的 absolute
probability gap，而只替换 layer 0 不充分并可能放大 gap。这支持 K/V 共同不兼容和 early/middle
dependency propagation，不支持 key-only repair 或单层 hotspot 设计。这些 splice 仍然只是
干预，不是 executable action。

### 6.3 Controlled dilution 与 recommendation-specific risk

在每条 edge 的 32 个真实重用户上固定最终 query/candidate，比较三条路径：保留固定 384 个
旧 token 并只追加 Current state、只 eviction 旧 token 而不追加、以及真实 rolling。追加约
128 个 Current events 后：

- anchor-only 在 v0→v1、v1→v2、v3→v4、v4→v5 上减少 17.1%–23.8% 的 output gap；
- pure eviction 在这四条 edge 上只减少 0.2%–2.2%；
- real rolling 减少 18.0%–24.7%，与 anchor-only 接近；
- v2→v3 是明确反例，三条路径均未改善。

因此现有 observation 中“remaining-old 越少、gap 越小”的相关不能解释为 eviction 因果。
更准确的机制候选是：少量 Current-version state 可以在大量旧状态仍然存在时重新锚定当前
模型对历史的读取。

在完整 217,584 个请求上，candidate mode 进一步显示稳定的 recommendation-specific
差异：novel-to-prefix candidate 的 Current−Reuse ROC-AUC 损失在五条 edge 上为
0.041–0.443 pp，并且每条 edge 都大于 recent-repeat 的 0.009–0.289 pp。Parent→Current
release gain 并不总是集中在 novel cohort，因此当前只支持“novel candidate compatibility
risk 更高”，不能写成“旧 KV 总是在压制 novel capability”。

同用户、同 timestamp 的 11,124 个真实 positive-negative pair 没有稳定的 pairwise accuracy
下降，harmful/beneficial flip 方向跨 edge 改变。因此 aggregate AUC harm 不能改写成普遍的
local pair inversion。

### 6.4 Pre-cutover replay bridge 与 representation-drift origin

为了区分“Current representation”与“发布后新行为内容”，随后把 controlled probe 扩到每条
edge 的 64 个 eligible 用户，并加入 `precutover_tail_replay`：它不读取任何发布后行为，只用
Current model 从依赖边界开始重放约 128 个发布前 tail token。结果在五条 edge 上均降低
17.8%–23.4% 的 absolute-probability gap；同等数量的 pure eviction 为 −0.02%–2.55%。自然
Current append 的恢复为 5.2%–28.9%，方向为正但跨 edge 波动更大。由此，small current-state
anchor 不再只是对自然 append 的解释：在没有新行为信息时，Current-version representation
本身已经能稳定提供主要恢复。

embedding origin 分析进一步扩大到每条 edge 的全部 active-user first request，共 23,051 个
请求。对 fixed in-vocabulary mapping 测得：

- candidate embedding drift、prefix mean/max drift、high-drift mass 和 candidate–history
  embedding geometry 与 Reuse harm/输出 gap 的相关均弱且跨 edge 变号；
- request-level mean item-embedding drift 与真实 behavior/time/context 下的 layer-0 K/V drift
  相关仅为 −0.16 到 0.14；
- 用 Parent item embedding、Current 其余参数生成 history cache，只产生 Parent-all Reuse gap
  的 0.5%–4.0%；连同 behavior/time encoder 与 input projection 后为 16.3%–41.7%；
- 保持 Current 输入、只使用 Parent Transformer blocks 生成 cache，则产生 Parent-all gap 的
  64.6%–92.6%。

因此当前 source insight 是 contextual Transformer co-adaptation，而不是 isolated item
embedding drift。embedding 参数确实更新，但 raw item distance 既不能解释 layer-0 state drift，
也不能稳定预测 request harm。hybrid 比例是不同干预的输出差异，不能相加解释为参数贡献率。
当前不继续做 high-drift-item causal replay 或 embedding-aware
scheduler；这不是遗漏实验，而是由 full-user correlation 与 hybrid intervention 共同给出的
negative adjudication。未来 item 使用稳定 OOV bucket，结果只支持 OOV exposure 分析，不能
外推为每个真实新 item 的独立 embedding drift。

### 6.5 历史 capability discovery：转换结构、互补性与 state mass

在五条 edge 共 1,267 个 append-free、512-token 请求上，进一步打开状态转换原语层。该实验
使用相同 Current reader，并复现 sealed Parent Reuse/Current Exact baseline；所有结果是
output-fidelity capability，不等同于完整 rolling quality qualification。

- 不读取 raw history、也不拟合 target K/V 的 parameter-only joint layerwise translation，在
  五条 edge 均恢复 21.6%–64.1% 的 Current–Reuse output gap，平均 43.0%；
- dependency-closed tail-128 rematerialization 恢复 19.4%–25.8%，平均 23.1%；
- Translate 后再 rematerialize recent-128，合计恢复 38.7%–72.6%，且相对最佳单原语在五条
  edge 都额外恢复 8.5%–18.6%；
- 把 recent-128 压成 64 个 Current landmark states、并携带 `mass=2`，恢复 9.5%–25.0%；
  无 multiplicity 的同一 sidecar 平均 recovery 为 −23.2%；
- Translate 与 weighted 64-state sidecar 组合恢复 29.2%–68.3%，在四条 edge 对最佳单原语
  有正互补；
- reader-only key bridge、FIFO/diversity Retire-128、embedding route 和 layer-0 Q–K route
  当前多数或全部恶化，因此退出当前 capability list。

这产生两个新的核心 insight。第一，projection geometry 与 recent contextual residual 是可分、
可组合的误差来源；否则 Translate 与 exact replay 不会在所有 edge 稳定互补。第二，HSTU
persistent state 的语义不仅是 K/V 内容，还包括 state density 和 aggregation mass；这是
unnormalised pointwise aggregation 下压缩、退休和路由必须遵守的接口合同。

该实验中的 Translate/Rematerialize/Synthesize/Retire/Route 是 capability 路径，不再作为
当前原语设计。其负结果和原始数字继续保留。

### 6.6 Typed State Refinement Algebra probe

新的聚焦 probe 在同一批五条 edge、1,267 个 append-free 请求上，将旧宏动作分解为
`CAST / PATCH / GROUP / SCALE`。它复现 sealed Current/Reuse logit 的最大误差为 `7.2e-7`，
不拟合 target K/V、不用 label 选 scope、不做 score mixing。

- Parent raw-history tail replay 生成的 additive residual 加到 CAST base 的同一 tail scope 后，恢复
  37.7%–79.6% output gap，平均 61.2%；它在五条 edge 都比 CAST/PATCH 较好的单项多
  10.3–23.5 percentage points；
- exact PATCH 完全覆盖同 scope CAST 时，只 CAST prefix 与先 CAST full state 的最终 K/V
  误差为 0，支持 compiler dead-code elimination；
- `GROUP->PATCH->SCALE` 与 `PATCH->GROUP->SCALE` 在 64 carriers 上的平均绝对 recovery
  差为 0.49 percentage point，方向跨 edge 改变；当前更像 cost order，不是两个质量机制；
- carrier 数 8->16->32->64->128 的两种顺序共有 40 个密度增量，39 个非负；
- carriers=8/16/32/64、两种顺序、五 edge 共 40 个非平凡 SCALE 消融，mass-aware
  路径全部更好；64 carriers 上额外改善 9.0–95.3 percentage points。

这些数字支持四个独立语义轴：version coordinate、contextual payload residual、ordered
evidence-to-carrier coverage 和 aggregation mass。但 additive payload 不保证质量单调，8/16 carriers
在部分 edge 仍为负 recovery，因此当前不冻结 residual threshold 或 carrier-density 安全阈值。

### 6.7 最新 Insight 裁决：三条观察推导四阶段流水线

上述结果不再以“四个并列原子操作”作为论文主叙事。当前更准确的三条 Insight 是：

1. **分布式失配与非对称修复**：只换 layer 0 不足，lower 3/4 layers 联合才恢复
   83.4%–96.0%；Tail-128 exact replay 在五条 edge 都有效，但只恢复
   19.4%–25.8%。这支持“Tail 是便宜且有用的 dependency-closed repair boundary，但
   单独 Tail 不完整”。
2. **共享且可转换的版本变化**：parameter-only joint layerwise K/V CAST 在所有 edge
   正恢复，平均 43.0%；CAST 与 contextual PATCH 在不同 scope 和 same scope 上都有额外恢复。
   因此大范围共享变化可以先翻译，昂贵重读留给上下文变化。
3. **保持证据质量的紧凑重算**：128->64 carriers 时，先 GROUP 后 PATCH 与先 dense
   PATCH 再 GROUP 的 output recovery 平均只差 0.49 point；但 compact state 必须保留
   represented mass，否则会系统性恶化。GROUP 和 SCALE 因而是一条 mass-aware compact
   rematerialization Insight，不是两条并列 headline。

它们推导的 provisional 流水线是：

~~~text
PLAN(repair width r, carrier count c)
  -> CAST(large stable region)
  -> GROUP(repair region r -> c) -> PATCH(c Current carriers)
  -> SCALE(represented mass) -> UNION/COMMIT
~~~

`PLAN` 是高层选择器，不是第五种状态算子。GROUP 要在 PATCH 之前执行才能降低主要重算；
SCALE 只恢复 occurrence mass，不会凭空恢复过度压缩丢掉的具体语义。

当前 4L/context512 结构计数中，dense Tail-128 重算 Exact-All 的 25.0% token-layers
和 43.7% causal attention pairs；`GROUP(128->64)->PATCH->SCALE` 下降为 12.5% 和
20.3%。这证明主要结构 work 下降，但还没有包含 CAST、kernel utilization、KV bandwidth
和 makespan 的端到端成本证明。

固定 `r=128,c=64` 的完整 rolling 评测现已完成：它在 4/5 edge 上提高 Reuse AUC，但在
v4->v5 失败。因此 rolling quality 不再是“完全未测”的缺口；当前缺口变为解释该跨 edge 失效边界、
在不使用 qualification label 调参的前提下预注册安全选择/回退，以及完成端到端成本资格。

三个关键观察缺口仍保留：

- 现有数据证明 recent state 具有高读取效用、Tail replay 稳定有效，但没有等宽
  old/middle/recent/random-128 干预，因此不能写“Tail 是最敏感的位置”；
- aggregate CAST 有效不等于每层、每个 token quartile 都有效；尚未测 head 时不能写
  “head selection 已被否定”；
- GROUP+SCALE 必须在同机、同 batch 下与 Exact-All、Tail-128 和 CAST+Tail 实测 CUDA
  time、raw/state I/O 与 persistent bytes，才能写端到端更便宜。

对应聚合见：

- `results/yambda500m_small_seed17/insight_history_utility_probe_v1/report.md`；
- `results/yambda500m_small_seed17/insight_kv_mechanism_probe_v1/report.md`；
- `results/yambda500m_small_seed17/insight_controlled_dilution_v1/report.md`；
- `results/yambda500m_small_seed17/insight_recommendation_semantics_v1/report.md`；
- `results/yambda500m_small_seed17/insight_anchor_replay_v2/report.md`；
- `results/yambda500m_small_seed17/insight_embedding_origin_v1/report.md`；
- `results/yambda500m_small_seed17/insight_embedding_hybrid_v1/report.md`；
- `results/yambda500m_small_seed17/insight_state_primitive_discovery_v6/report.md`；
- `results/yambda500m_small_seed17/insight_refinement_algebra_v1/report.md`。

## 7. 当前 observation 的系统含义

目前可以形成的 insight 证据链是：

~~~text
Reuse harm 优先落在 Current 原本改善的请求上
  -> 风险对 novel-to-prefix candidate 更强
  -> mismatch 主要来自 contextual Transformer co-adaptation，而非孤立 embedding drift
  -> mismatch 跨层传播，Tail 是便宜有用但不完整的 causal repair boundary
  -> 共享版本变化由 CAST 翻译，用户上下文变化由局部 PATCH 重读
  -> GROUP 在 PATCH 前减少 Current carriers，SCALE 保留 represented occurrence mass
  -> typed refined states 通过 UNION -> COMMIT 形成合法 persistent view
~~~

这条链条直接支持的机制不再是单一 Tail Replay，而是 insight-guided Versioned Neural-State
Refinement Pipeline。当前证据支持 `CAST / PATCH / GROUP / SCALE`
作为四个底层 instruction semantics，但论文主叙事是三条 Insight 和四阶段流水线；
`SLICE / UNION / COMMIT` 属于寻址、组合和生命周期。Tail Replay、Translate-All 和 weighted Landmark-64
都只是编译宏计划。该裁决准入 instruction set，不准入 scale scheduler、target-free residual estimator
或固定阈值。

在论文组织上，这三条 Insight 直接放入 `Insight-Driven State Refinement` 的 Design Insights 小节，
并分别以 Design implication 接到 CAST、contextual PATCH 和 GROUP+SCALE；不再单独设置一个面向
Continuous/Runtime 的通用 Insight 章节，也不增加重复的 Design Principles 层。

这条证据链的直接终点是 **One-Release State Refinement**，不是完整 multi-release system。
long-age direct Reuse harm 随 producer age 增大，说明持续版本化状态管理值得研究；但当前没有执行过
bounded-debt plan selection、最大近似深度、sampled Current-Exact feedback 或质量触发 Rebase 实验。
因此论文结构应明确分成：已有 Insight 推导一次转换；下一阶段独立验证 debt-bounded evolution
闭环；最后才实现和度量 GPU Transformation Runtime。

当前不支持的设计方向同样明确：

- 不按 recent/old History Utility 象限冻结 semantic-region scheduler；
- 不使用单一 K repair、单一 V repair 或 layer-0 hotspot；
- 不使用 raw item-embedding drift 或 high-drift × fanout cohort 冻结 selective scheduler；
- 不把 aggregate AUC harm 写成普遍的 pairwise inversion；
- 不把自然 post-cutover append 当成稳定 action；它跨 edge 的恢复幅度明显波动。
- 不把 tail replay 的 output-gap recovery 直接改写成最终线上质量收益。
- 不把当前失败的 reader bridge、Retire 或 Route 放进第一版 catalog 凑原语数量。
- 不把 payload residual 的张量可加性改写成 recommendation quality 单调性。
- 不把 39/40 density 增量直接冻结为跨 seed/scale 的 `rho_min` 或 `mu_max`。

## 8. 结论边界与反例

可以写：

> 在约 300 天真实交互时间线上，使用约 72% 时间跨度建立初始模型，再以约 4.7% 时间跨度进行更新并观察紧随其后的约 4.7% 时间跨度。多个相邻版本边显示，新模型的 AUC 发布收益会被直接复用父版本 KV 稳定侵蚀；四条常规正收益边的侵蚀比例为 25.5%–47.9%。

不可以写：

- 所有模型更新都必然有害；
- 418.7% 是典型效果；
- One-hop 结果已经证明 recursive cache debt；
- one-hop CAST/PATCH/GROUP/SCALE 已经证明多次近似演化受控；
- producer-age 单调结果已经确定 debt estimator、`tau/H`、shadow rate 或 rebase threshold；
- sampled output fidelity 已经等价于真实 rolling recommendation-quality 保证；
- 当前结果已经证明某个最终 migration policy；
- 当前结果已经证明 scheduler、partial action 或 executor 的线上收益；
- 当前 typed plan 已经证明 GPU batching、makespan 或 serving isolation；
- eviction 是 rolling dilution 的主要因果机制；
- 64-user tail replay 的 output-gap recovery 已经证明所有 seed、规模和 workload 的质量收益；
- 64-user tail replay probe 已经代表额外 seed、Medium/Large 或第二 workload；
- raw item embedding drift 已经构成可用的 migration predictor；
- 一条固定计划的 4/5 rolling AUC 改善已经证明稳定的跨 edge quality-cost frontier 或 latency 优势；
- novel candidate 承载了所有 Parent→Current release gain；
- HSTU-native Small 结果已经代表 M/L 或 RecFlow。

完整 CSV、PR-AUC 和 log-loss companion 以本文件中的固定表为准；结果源文件仍保存在 results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3/ 下。

## 9. 复现入口

当前结果对应的合同和代码入口：

- configs/contracts/yambda500m_small_hstu_native_rolling_recipe_matrix_v3.yaml
- configs/contracts/yambda500m_small_hstu_native_d14_onehop_reuse_diagnostic_v1.yaml
- configs/contracts/yambda500m_small_hstu_native_d14_onehop_reuse_completion_v2.yaml
- scripts/run_yambda500m_hstu_native_rolling_recipe_matrix_v3.py
- scripts/run_yambda500m_hstu_native_d14_onehop_reuse.py
- scripts/run_yambda500m_hstu_native_d14_onehop_reuse_completion_v2.py
- scripts/run_yambda500m_hstu_native_d14_direct_long_age_reuse.py
- scripts/summarize_yambda500m_hstu_native_d14_auc_coverage.py
- scripts/insight/analyze_first_pass.py
- scripts/insight/probe_history_utility.py
- scripts/insight/probe_kv_mechanism.py
- scripts/insight/analyze_recommendation_semantics.py
- scripts/insight/probe_controlled_dilution.py
- scripts/insight/probe_anchor_replay.py
- scripts/insight/analyze_embedding_origin.py
- scripts/insight/probe_embedding_hybrid.py
- scripts/insight/probe_refinement_algebra.py
- configs/contracts/yambda500m_small_hstu_native_d14_one_release_refinement_auc_v1.yaml
- scripts/insight/run_one_release_auc.py
- scripts/insight/adjudicate_one_release_auc.py
- results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3/d14_one_release_refinement_auc_v1/auc_summary.md

这些入口只负责复现固定合同和结果，不授权根据结果新增 edge、调 recipe 或改变最终系统设计。
