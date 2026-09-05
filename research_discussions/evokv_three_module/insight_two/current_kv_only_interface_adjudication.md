# Insight 2 / Design 1：当前 KV-only 接口阶段裁决

日期：2026-09-03  
状态：**阶段裁决已冻结；本文只裁决 post-hoc KV-only 接口**  

> 2026-09-03 后续说明：本文的“无 Design”结论只适用于
> `existing Parent KV-only + no new producer state + no new training + <20%`。专家讨论已经把正向
> 后继 prospective mechanism 是在 state creation 时主动写入 Migration Sketch，并以 common-mode
> certificate-error objective 和 Full/append 同源的 clean writer $G_v$ 形成闭环；见
> [统一论文材料](../../../docs/insight2_design1_expert_brief.md)。该后继方向不改变本文任何负结果，且其
> 方法 recovery 仍为 pending。
范围：Yambda-500M Medium `v0..v5`、单次相邻 `Parent -> Current`、现有 Parent persistent K/V、
无新训练、无 Current-Exact target fitting、理论计算 `0%--20% Exact-All`

## 1. 裁决先行

当前证据支持一个重要、但必须准确表述的 Transformer-specific 边界：

> **跨版本误差可以在一次 query-conditioned reader aggregation 之后变得非常紧凑；但这种
> finite-query functional compactness 是 reader-conditional 的，并不意味着 Parent K/V 中已经存在一个
> query 之前可低成本生成、可供未来请求持续读取的 Current functional state。**

更直白地说：**被压紧的是一次读取后的 response，不是读取前的可迁移 state。**

这一区分解释了目前全部正负结果：

- S4 exact functional intervention 很强，因为它直接使用了 Current Exact response；
- single Current-r8 replay 很强，因为它直接近似生成 Current contextual trajectory；
- token support、moments、paired subtraction、source residual、suffix self-probe 和 static transport
  都没有提供新的 Current contextual information，因此不能从表示紧凑自动跃迁到合法 constructor；
- 一旦真正执行 Current historical interaction，native attention work 又会越过 `20%` 预算。

因此，当前不能冻结一个 paper-worthy Design 1。数值最强的 generic single-C8 只保留为硬对照，旧 PRO
只保留为历史先验；二者都不能因结果好而改写成创新方法。当前也不启动 32/512-user formal、confirmation、
新训练或 GPU 搜参。

## 2. 已经成立的正向 observation

### 2.1 Reader aggregation 确实形成紧凑功能边界

在冻结的 Medium ranking workload 上：

- S4 shared response rank-0/rank-1 oracle recovery 为 `95.34%/99.46%`；
- full activation-region representation recovery 为 `99.57%`；
- release response rank-8 oracle 为 `94.18%`；
- rolling cutover/current response direction cosine 为 `0.9460`。

这些结果共同说明，分布在 token、layer 和 K/V 中的 compatibility error 经过 Current reader 的
query--key interaction、value aggregation 与 residual computation 后，会在有限 recommendation query
workload 上收敛到紧凑 response range。这是 Insight 1 之后真正值得保留的科学现象。

### 2.2 该收敛不是 token support sparsity

Exact-state R64 carrier、Parent-conditioned carrier 与 recursive closure 的 recovery 分别为
`-45.32%/-34.31%/-19.62%`。activation graph 虽有 `87.32%` endpoint branch agreement，但 crossing-only
causal oracle 只有约 `.211` mean recovery；主要误差来自同一 activation region 内的连续 K/V 与
residual deformation，而不是少数 crossing edges。

所以已经可以冻结：

> **functional compactness does not imply token-support sparsity or stable interaction-topology locality.**

## 3. 表示紧凑为何没有转化为 constructor

一个 persistent migration object 不只要在冻结 query panel 上低秩；它还必须在 Current Exact 不可见时
生成，并对未来 native queries、append 和 eviction 保持语义。也就是必须同时满足：

1. `representation`：少量状态足以读取主要 response；
2. `generation`：只从 Parent state、raw history 和两版权重合法地产生该状态；
3. `evolution`：cutover 后可随 Current append/eviction 闭合演化。

现有结果通过了第 1 项，却没有通过第 2、3 项。有限 candidate panel 上的低秩只说明多个已给定 outputs
相关；它不能反向提供产生这些 outputs 所缺少的 Current historical trajectory。

### 3.1 Query-independent exact quotient 不存在于当前 reader

legacy `ELU+1` 的 negative branch 含有

\[
\sum_i e^{q^\top k_i}v_i.
\]

对一般 keys，有限指数函数没有固定有限维 exact feature state。实际 full-Exact all-history affine oracle
也在三条 edge 上为负，最差为 `-13.0716`。另外，256 个 model-native layer-0 item queries 已在全部
`30/30` Current head witnesses 上张成完整 32 维 head space；不存在可删除的 exact key nullspace。

### 3.2 Current contextual trajectory 是缺失的信息源

Parent joint K/V 可以恢复 Parent normalized token coordinates，却不能生成 Current residual trajectory。
若不形成 Current trajectory，解析 transport 只能得到 contextual-freeze coordinate mapping；若形成它，
前五个 non-terminal blocks 仅 Current native QK 已为

\[
1{,}007{,}616{,}000 = 21.1183\%\ \text{Exact-All},
\]

QK+AV 为

\[
2{,}015{,}232{,}000 = 42.2367\%\ \text{Exact-All},
\]

尚未计 projection、gate、residual、writeback 或 sidecar。由此形成当前接口的三分法：

- 不生成 Current trajectory：退化为 mapping/contextual freeze；
- 原生生成 Current trajectory：attention 本身越过预算；
- 压缩或抽样该 trajectory：回到 generic low-rank/kernel/sampling/cache-compression family。

## 4. 经过数值与创新双门的候选裁决

| family | 最强当前证据 | 裁决原因 |
| --- | --- | --- |
| generic single Current-r8 | `.9372` mean recovery，`17.8953%` | 数值硬对照；xKV-adjacent generic compression，不是 Design |
| paired native response | `.9012`，`18.2810%` | 被 single-r8 支配；paired compression/control variate |
| Parent-base + release defect | `.508`，`18.4567%` | 数值失败；base-plus-delta prior art |
| source-certified finite defect | `.662`，`19.4726%` | 数值失败；DEIM/sampled-residual 与 control-variate 碰撞 |
| producer/reader commutator | raw oracle `.8831` | centered decision 不稳；依赖 Current Exact reverse path与 score mixing |
| all-history affine moments | full-Exact oracle mean `-2.658` | representation 本身失败；linear-attention/fast-weight state |
| migration-ready source tape | native QK+AV floor `42.2367%` | 状态额外 `+216.7%`，计算越界 |
| natural causal suffix | Tail-128 `-.0876`，约 `18.28%` | suffix 对错误 lineage 也自洽；只有 query coverage，没有 target-state information |
| sparse head/circuit replay | 第三个 cache layer单 head closure `35.09%` | dense `W_O`/gate/residual 使 head closure跨层饱和；低成本窗口退化为 locality |
| attention gauge / release algebra | gauge mismatch `5.21%--11.89%`；真实 `Delta W` rank `180--192` | release 不在 symmetry orbit；截断后退化为 LoRA/JVP/compression |
| causal state ports | 仅 prospective architecture | memory ports已有；Parent-sufficient 不推出 Current-sufficient，且无 release/delete law |

上述失败不是一次超参数搜索的结果。每一族都在运行前或固定单点后接受了：full-rank correctness limit、
no-target API、完整成本、matched generic control 和 prior-art reduction test。失败后没有调 rank、probe、
selector、layer、edge 或 confirmation population。

## 5. 当前可以写成什么 Insight，不能写成什么

### 5.1 可以写成阶段性 Transformer insight

可以准确写成：

> Transformer reader 会把分布式 cache mismatch 压缩成有限请求上的紧凑功能响应；但这种压缩发生在
> query-dependent interaction 之后，不是 Parent state 自带的可生成 quotient。因而 functional boundary
> 的 existence 与 migration object 的 constructibility 是两个独立问题。

它比“AV 很低秩”更一般，也解释了为什么局部重要性、固定 offset、response moments、interaction graph
和 paired residual 都不能自动成为迁移算法。

### 5.2 当前不能写成最终 Insight 2 / Design 1

不能写：

- `Insight 2 = candidate-shared AV correction`；
- `Design 1 = PRO` 或 generic C8；
- “response operator 低秩，所以可迁移”；
- “少数 heads/ports 是新的 Transformer state”；
- “gauge + residual”“memory token + mapper”或“base + defect”是新机制；
- 当前接口下已经找到 `80%--90% @ 0%--20%` 的论文方法。

## 6. 重新打开 Design 1 所需的新信息

下一候选不能再从现有 response 表示中换一种压缩坐标。它必须先给出一个新的
**Current-information source** 或 **finite-release causal law**，并同时回答：

1. 为什么删除 Parent-specific term 后，matched-compute Current-only control 会稳定变差；
2. 为什么 constructor 不读 Current Exact target，却能区分 Parent state 已合并、Current producer需要区分
   的历史；
3. 为什么 append 顺序满足可检查的 commuting relation，而 eviction 具有明确 lineage/delete 语义；
4. 为什么核心机制在删掉“release/recommendation”两个应用名词后，仍不是 mapper、SVD、sampling、
   linear attention、selective recomputation 或 memory token；
5. 为什么完整 constructor 严格低于 `20%`，且强于 `.9372 @ 17.8953%` generic control。

一个 model--system co-design 只有在先构造出非 regression 的 release law 后才值得申请新训练。单纯增加
causal ports 不够：若 `P_P(h_1)=P_P(h_2)`，port-only exact migration 要求
`P_C(h_1)=P_C(h_2)`；append 还要求在可达状态上满足

\[
T\circ F_P=F_C\circ T.
\]

这两个条件目前没有具体、非 mapping、又不冻结 history semantics 的实现。因而当前不以“先训练看看”
代替方法构造，也不自动授权 Small/Medium training。

## 7. 执行决定

1. 保留所有既有正负 evidence、contracts、hashes 与 raw seals；confirmation `[512,3000)` 继续 unread。
2. 停止当前 KV-only family 的 rank/probe/layer/head/suffix/operator 扫描，不启动新的 formal GPU run。
3. generic single-C8 只作为未来任何方法必须超过的数值硬对照；不进入论文贡献表。
4. PRO 只作为 Small 历史先验和“functional migration 有可能便宜”的证据；不作为 active Design。
5. 若未来提出具体 release law，先做 primary-prior-art reduction、toy exactness 和 CPU cost falsifier；只有
   通过后才建立新的 Small prospective training contract并请求用户明确 launch。

当前正确停止点是：**Insight 2 已得到一个可信的接口边界，但 paper-worthy 的正向 Design 1 尚未找到。**
在这一状态下诚实保留空缺，比把 `.937` generic compression 或 `.901` paired response 包装成新方法更符合
论文要求。
