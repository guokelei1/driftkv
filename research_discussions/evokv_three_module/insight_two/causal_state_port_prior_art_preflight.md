# Causal state port co-design：prior-art 与可行性审计

日期：2026-09-03  
状态：**严格 NO-GO（不能作为当前 Design 1，也不能在现有 Medium `v0..v5` 上验证）；仅保留一个需要新训练授权的 prospective falsification canary**  
范围：只做原始论文、因果接口与可实现性审计；未运行代码或训练，未修改 frozen contract、seal、raw result 或既有 checkpoint

## 1. 裁决先行

本轮审计的问题是：能否把少量 model-native history state ports 做成真实 causal separator，使未来
recommendation query、append 和 eviction 只经过 ports；模型 release 时只迁移或重算 ports，从而绕开
完整 token-level KV rematerialization。

严格结论是 **NO-GO**，理由不是这种架构没有价值，而是它目前不能同时满足本文需要的四项要求：

1. **causal separator 本身不是新方法。** [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881)
   已把读/写 memory tokens 作为 segment 间唯一递归通路；
   [Block-Recurrent Transformer](https://arxiv.org/abs/2203.07852) 已用一组 fixed-size state vectors 与
   token block 双向 cross-attention；[RetNet](https://arxiv.org/abs/2307.08621) 则给出可与并行形式等价的
   recurrent state。把旧历史挡在少数 ports 后面，是成熟的 recurrent-memory architecture family，不能
   单独承担 EvoKV 的 Insight 2 或 Design 1。
2. **一个 port 对本版本是充分状态，不代表它对下一版本也是充分状态。** 令 Parent/Current 的历史端口为
   `P_P(h)` 与 `P_C(h)`。只读 Parent port 的无损 release transform `T` 存在，当且仅当
   `P_P(h1)=P_P(h2)` 蕴含 `P_C(h1)=P_C(h2)`。独立训练的两个 producer 没有理由满足这个 fiber
   refinement 条件；memory-token mask 不会替我们创造它。
3. **精确、紧凑、producer 可自由更新、保留一般 full-attention expressivity 四者不能同时保证。**
   固定小端口可以精确复现“本来就只允许通过该端口读取历史”的 ported architecture；它不能在有限精度下
   无损保留一般 softmax Transformer 对任意旧 token 的 query-dependent random access。若要对任意 Current
   producer 精确，则只能冻结 port semantics、把 release 限制为同一 state machine 的坐标变换，或保留足够
   replay information；三者分别牺牲 producer 更新自由、方法新颖性或成本。
4. **append 容易，eviction 并不由 recurrence 自动闭合。** 通常的 nonlinear recurrent port 能吸收新事件，
   却没有从累计状态中精确删除最旧事件的逆操作。若用 independent segment ports 解决 drop lineage，就变成
   已有 segment memory/compression 的变体；若保留 raw segment 以重算，则 port 不再是唯一持久迁移对象。

所以，不能把以下组合写成论文方法：

```text
memory tokens + version tag + learned upgrader + append/eviction bookkeeping
```

它分别与 recurrent memory、backward-compatible representation mapping 和普通 cache lineage 相撞，而且
没有解除跨版本 exactness 的基本障碍。

唯一仍可能有论文意义的边界，不是“增加 ports”，而是设计一个 **version-compatible state-machine
family**：release upgrade 必须在可达状态上满足 transition homomorphism/commuting law，并且时间分解本身
提供可验证的 deletion algebra。当前没有这样的具体机制，也没有证据表明它能在不冻结 producer 的情况下
保持质量和 `<20%` 成本。因此它只能作为未来研究问题，不能作为当前 Design 1 候选。

## 2. 什么才叫真实 causal state port

对模型版本 `v`、已封存历史 `h`、release 后 suffix `s` 和当前请求 `q`，令 port producer 为

\[
z_v=P_v(h),
\]

后续 transition/readout 为

\[
y_v(h,s,q)=G_v(z_v,s,q).
\]

只有当计算图中从 `h` 到任何未来节点的所有有向路径都经过 `z_v`，`z_v` 才是 causal separator。它有两个
重要限定：

- 这是一个**架构性质**，不是通过 cosine、SVD、probe recovery 或蒸馏损失得到的经验结论；
- 它只保证 `z_v` 对**同一个 ported architecture** 充分，不保证它等价于原 full-attention HSTU 的完整
  token state，也不保证另一版本能够消费它。

若还保留任何跨 segment 的 token K/V、retrieved raw block 或 hidden-state cache，ports 就不是唯一 separator。
因此：

- RMT 的 segment 间 recurrence 可以满足这一结构定义；
- Landmark Attention 不满足，因为 landmark 负责选择、随后仍读取被选 block 的 exact tokens；
- 原始 Block-Recurrent Transformer 也不由 recurrent states 单独分隔，因为它同时保留上一 block K/V 的
  sliding-window path；必须删掉这条 path 后才会得到纯 port variant；
- Transformer-XL 和 Compressive Transformer 的 memory 本身仍是 per-position/per-layer activation sequence，
  不是少量统一 ports。

## 3. 跨版本精确迁移的充分必要条件

### 3.1 Fiber criterion

希望存在一个不读取原历史的 release operator `T_{P->C}`，使所有合法历史都满足

\[
T_{P\rightarrow C}(P_P(h))=P_C(h).
\]

这样的确定性 operator 存在，当且仅当

\[
P_P(h_1)=P_P(h_2)
\Longrightarrow
P_C(h_1)=P_C(h_2).
\]

证明很直接：

- 必要性：若两个历史拥有同一个 Parent port，确定性的 `T` 对二者只能产生同一个输出；
- 充分性：若 Current port 在 Parent port 的每个等价类上恒定，就可以在 `P_P` 的 image 上定义唯一的 `T`。

这说明问题不在 mapper 是否足够深。若 Current producer 区分了 Parent producer 已经合并的两个历史，任何
port-only mapper、attention adapter、flow 或 lookup 都不可能恢复丢失的信息。增加训练数据只能改善分布内
近似，不能改变这个不可辨识性。

### 3.2 Append consistency 自动要求 commuting law

令版本 `v` 的 append transition 为

\[
P_v(h\mathbin\Vert x)=F_v(P_v(h),x).
\]

若 `T` 真能在任意 cutover boundary 精确迁移，则在所有可达状态上必然满足

\[
T(F_P(z,x))=F_C(T(z),x).
\]

这是 state-machine homomorphism，而不是额外的 heuristic loss。它意味着：先在 Parent 下 append 再迁移，
与先迁移再在 Current 下 append，必须得到同一 Current state。只拟合 release 时的 endpoint port，不能保证
用户继续追加行为后误差不会沿 Current dynamics 放大。

一个真正有论文高度的 co-design 至少要把这条 commuting relation 变成结构保证或可检验 invariant；把它
当作一项 distillation loss，则仍然是 approximate state alignment，且需要和既有 backward-compatible
representation learning 区分。

### 3.3 Eviction 需要另一套 algebra

对 sliding history `x_1,...,x_N`，精确淘汰最旧事件需要某个 delete transition `D_v` 满足

\[
D_v(P_v(x_1\Vert\cdots\Vert x_N),x_1)
=P_v(x_2\Vert\cdots\Vert x_N).
\]

普通 gated recurrent memory 只定义 forward `F_v`，并不提供 `D_v`。如果 `F_v` 丢失信息，delete 在一般
情形下不存在。实际可选项只有：

1. 让 port update 具有 group/monoid decomposition，可从累计 sufficient statistics 中减掉旧贡献；这会
   强烈限制 transition family，并与 linear attention、RetNet/SSM 的 recurrent sufficient-statistic 路线
   接近；
2. 保持多个 immutable segment ports，整段淘汰时直接 drop；这给出 segment-granular lineage，但 state 大小
   随 active segments 增长，且每个 segment 的跨版本 fiber problem 仍然存在；
3. 保留 raw events 或 merge-tree children，淘汰时重算受影响路径；它可以精确，但 migration object 已经不再
   是单个小 port，计算/I/O 必须完整入账；
4. 接受 approximate eviction；此时必须单独报告 append/eviction drift，不能声称 exact separator closure。

因此“RNN state 天然支持 append/eviction”是不正确的。它天然支持 append；delete 是独立而且更强的接口。

## 4. Expressivity：精确的是新架构，不是原 Transformer

### 4.1 固定小端口不能无损实现一般 random-access attention

考虑长度 `N`、item vocabulary 大小 `M` 的历史；未来 query 可以要求返回任意一个位置的 item/value。所有
历史共有 `M^N` 种。若一个 finite-precision port 只有 `B` bits，而

\[
B < N\log_2 M,
\]

则至少两个不同历史必须映射到同一个 port。存在一个位置查询能区分这两个历史，因此同一个 port 不可能对
所有 query 都给出正确结果。标准 attention 通过保留每个位置的 K/V 可以实现这种 query-dependent retrieval；
固定小状态不能在最坏情形下保留同一 function class。

对连续输入也不能用“一个实数可以编码无限信息”绕过工程约束：在有限精度下上述计数直接成立；要求连续、
稳定的低维 neural encoder 时，高维开放集到更低维空间也不能保持全局单射。

这不否认 ported Transformer 在 recommendation 分布上可能很好。它只限定 claim：

- 可以说“architecture-exact”：cache port 后与该 ported model 从头运行严格一致；
- 可以实证“task-sufficient”：在冻结 workload 上接近 dense HSTU；
- 不能说“用少量 ports 无损保持一般 Transformer expressivity”。

### 4.2 为什么“producer 自由更新”再次破坏 exactness

即使 Parent port 在 Parent task 上完全充分，Current producer 也可能学习一个 Parent 没保留的新历史区分。
这正是 Full Current 可能优于 Full Parent 的来源之一。若仍要求从 Parent port 精确得到 Current port，就必须
选择以下一种约束：

- **冻结 producer/port semantics。** Current 只更新 reader 或 port 之后的模块；exact reuse 成立，但本质
  接近 Activated LoRA 的 prefix-invariant strategy，且不能探索新的 history encoding；
- **只允许 state-coordinate conjugacy。** Current producer 在功能上与 Parent producer相同，只更换可逆
  坐标；可以用解析 transform 迁移，但这正是 mapping，而且不会增加 port 中的历史信息；
- **使用 version-independent sufficient state。** 所有 release 共享同一 canonical history algebra；这把
  可更新部分移到 reader，仍然是在冻结 state semantics；
- **保存 replay witness。** 从 raw items/segment trace 重新运行 Current producer；producer 可以自由变化，
  但成本和 I/O 随 witness 大小增长；
- **只要求近似。** 可以联合训练 Current producer 和 upgrader，但其方法边界落入 representation alignment、
  distillation 或 learned state translation，不能仅靠“它用于 recurrent port”获得创新性。

所以，**不存在一个对任意 model update 都成立的、小而精确、无需 replay、又不限制 producer 的 state port。**
这是信息约束，不是尚未找到更好的 optimizer。

## 5. Primary-source collision matrix

| 工作 | 它已经覆盖什么 | 为什么不能直接成为 EvoKV 新 Design | 与 port separator 的准确关系 |
| --- | --- | --- | --- |
| [Transformer-XL](https://arxiv.org/abs/1901.02860) | segment-level recurrence；缓存上一 segment 每层 hidden sequence；relative position 支持 state reuse | state 仍按 token/layer 增长且是 producer-specific；无跨 release contract | 旧 segment 通过整段 cached states 进入未来，不是少量 port |
| [Compressive Transformer](https://arxiv.org/abs/1911.05507) | 把被淘汰的 per-layer activations 压成 secondary FIFO memory，并用 reconstruction/attention loss训练 compressor | compression 与 eviction 都已有；是 lossy model-specific memory；无跨版本迁移 | raw old activations 被 compressed activations 替代，但仍是 position sequence，不是 version-stable separator |
| [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881) | 在 segment 首尾放 read/write memory tokens；updated write memory 递归传给下一 segment；仅少量 global tokens 传递历史 | “memory tokens 作为跨段功能状态”已被直接覆盖；没有 release migration、delete 或兼容性保证 | 是最直接的纯 port prior art；相对 RMT 计算图可 exact cache |
| [Block-Recurrent Transformer](https://arxiv.org/abs/2203.07852) | fixed-size recurrent state vectors 与 token block 互相 cross-attend，并用 gate 更新 state | block state、state IDs、gated update 都已有；原架构还保留 sliding K/V | recurrent state 不是唯一 separator，除非删掉其跨 block K/V path |
| [Landmark Attention](https://arxiv.org/abs/2305.16300) | landmarks 先选相关 block，再加载 exact token block；保留 random access 与 CPU offload | landmark 是 router/index，不是充分 history state；迁移 landmark 不能替代旧 block KV | 明确不是 port-only separator |
| [RetNet](https://arxiv.org/abs/2307.08621) | `S_n=gamma S_{n-1}+K_n^T V_n`；parallel/recurrent/chunkwise 三种形式等价，推理 O(1) state | fixed recurrent sufficient state 与 exact append algebra 已有；它移除 softmax、改变 operator，且 state 随参数变化 | 对 retention architecture 是 exact separator，但不是一般 softmax Transformer 的无损 port |
| [GRU4Rec](https://arxiv.org/abs/1511.06939) / [Mamba4Rec](https://arxiv.org/abs/2403.03900) | recommendation 中用 recurrent/SSM hidden state 总结 user sequence；Mamba4Rec 明确利用 recurrent inference | “推荐系统适合持久用户级递归状态”早已成立；没有跨 model-release state compatibility | 证明 workload motivation 不新，也暴露相同的 versioned hidden-state 问题 |
| [Activated LoRA](https://arxiv.org/abs/2504.12397) | adapter 只在 invocation 以后激活，因此之前的 base KV 完全不变并可精确复用 | exact compatibility 来自不修改 cache producer；这正是本路线试图避免的 producer-freeze escape | 是 exact cross-model reuse 的强控制组 |
| [Learning Backward Compatible Embeddings](https://arxiv.org/abs/2206.03040) | 在 model update 中训练兼容 embedding/aligner，并在真实 recommender application 上验证 | “给低维 state 加兼容损失或 mapper”已有直接 representation-version 先例 | 不处理 recurrent transition/append/delete，但会击穿简单 port aligner 的 novelty |
| [ReCache](https://arxiv.org/abs/2608.19662) | 用 resource-wise attention 切断 resource 之间的 contextual path，生成 composition-invariant reusable KV blocks | 结构化 mask 产生 reusable representation 已是明确方法；不涉及 model version | 说明“用因果 mask 制造可复用边界”本身也不新 |

### 5.1 几个容易混淆的结论

1. RMT 已经足以否定“我们首次发现 memory tokens 是功能状态”的 claim。
2. Landmark Attention 保留 exact raw blocks，所以它能保留 random access，但也因此没有解决旧 history state
   的 release compatibility。
3. RetNet 的 recurrent form 与 parallel form 等价，是**新 operator 内部**的 exactness；它不是把任意
   softmax attention cache 无损压成固定 state。
4. aLoRA 说明 cross-model exact cache reuse 可以通过 architecture/training co-design 实现，但它的关键是
   activation 前权重不变，不是把已经发生语义变化的 state 转成新 state。
5. backward-compatible embeddings 说明“新版本学习兼容旧向量空间”不是空白；若本路线最后只剩一个
   port mapper 和 compatibility loss，达不到用户要求的论文高度。

## 6. 是否还存在 paper-level novelty boundary

### 6.1 下面这些都不够

- 在 HSTU 前缀中加入若干 `[MEM]` tokens；
- 让未来 candidate 只 cross-attend memory tokens；
- 给 port 加 version ID、release embedding 或 adapter；
- 学一个 Parent-port 到 Current-port 的 MLP/LoRA/flow；
- 用 consistency/distillation loss 让两版 ports 接近；
- 把 ports 按 segment 保存，以便 drop 一个 segment；
- 在 release 时从 raw history 重跑一个小 RNN writer。

最后一项在系统上可能实用，但方法上是“cheap recurrent encoder + version-specific reader”；若没有新的状态
契约或迁移 invariant，它仍是已有 memory/recurrent recommender 的系统组合。

### 6.2 唯一值得重新开启的边界

一个可能达到论文高度的对象必须同时给出：

1. **structural separation**：sealed history 到未来的所有路径严格经过 bounded ports；
2. **release law**：存在不是事后 target-state regression 的可执行构造，并在可达 states 上满足
   `T o F_P = F_C o T`；
3. **information monotonicity contract**：Current release 不得要求 Parent port 已丢失的新历史区分，或必须
   明确提供额外 witness；不能把这个事实藏在平均 compatibility loss 中；
4. **delete/lineage algebra**：append、segment seal、evict 和 release cutover 的顺序具有可验证等价性；
5. **model improvement freedom 的实证边界**：Current Full 确实提高，而 port contract 没有退化成冻结整个
   history producer；
6. **强控制优势**：胜过 RMT-style memory、frozen producer/aLoRA-style compatibility 与 generic learned
   aligner，而不是只和 Reuse 比。

可以把这个研究对象概括为 “version-compatible causal state machine”，但目前不要创造正式名字，也不要把
它写进 Design 1。最关键的 release law 仍未被构造；若最后只是 learned `T`，仍然没有越过 novelty gate。

### 6.3 exactness 的最终判定

问题“一个 exact ported Transformer 能否在不冻结 producer、也不 full segment recompute 的情况下保持
expressivity？”需要分两层回答：

- **相对于 ported architecture 自身：可以。** mask 把 port 设为真正 graph separator 后，缓存/恢复 port
  可以 bitwise 或数值等价地复现该架构；producer 也可以在每个 release 内正常训练。
- **相对于任意独立更新的 Current port 或一般 full-attention Transformer：不可以普遍保证。** 除非
  Parent port 对 Current port 满足 fiber criterion；在 producer 自由变化时没有这个保证。固定小 port 也
  无法在最坏情形下保留任意 token random access 的 function class。

因此，所谓 exact co-design 必须承认至少一个限制：冻结/约束 port semantics、限制 model family、扩大
state 到足以保存 history，或读取 replay witness。不存在免费的第五种情况。

## 7. 对当前 EvoKV 路线的意义

### 7.1 现有 Medium `v0..v5` 不能验证这条路线

当前六个 checkpoint 是既有 HSTU，并未用 causal port mask、port recurrence 或 version-compatible release
law 训练。推理时事后插入 memory tokens 或把某个 pooling state 当 port，不能证明 causal separation，
因为历史 token 的 contextual paths 已经参与全部层的 K/V、gate 和 residual。它只会成为另一个 compression
或 mapper experiment。

所以：

- 不应在 UID1930 上实现一个 post-hoc port prototype；
- 不应把现有 single-r8 的 `.9372` 解释成 port feasibility；
- 不应为这条路线读取 `[512,3000)` confirmation；
- 不应修改 frozen v0..v5 contract 或补训练；
- 若无新的 release law，连新训练 canary 也不值得启动。

### 7.2 它不是 Insight 2 的当前答案

这条审计最多支持一个 negative boundary：

> 把 history dependency 压到显式 recurrent ports，可以从架构上消除 token-level causal closure；但版本兼容
> 不由 causal separation 自动产生。一个本版本充分的 state，除非其 history equivalence classes 被后续
> release 保留，否则不能只从旧 state 精确升级。

这是有用的 discussion/theory boundary，但还没有导出一个 `<20%`、高恢复、paper-worthy 的 Design 1。

## 8. 最小 prospective experiment：只用于重新开启，不是当前执行计划

如果以后要重新考虑这条 co-design，最小实验必须是**新训练的单 edge falsification canary**，而不是在
现有 Medium checkpoint 上加 mapper。它需要新的 prospective contract、资源估计、focused canary 和用户
明确 launch；本轮没有训练授权。

### 8.1 最小设置

- 数据：Yambda 的冻结小规模 development population；只用一个 Parent->Current 时间 edge；
- 模型规模：Small architecture pilot，不启动 Medium/Large；
- history：固定 context 与固定 segment size；只训练一对 release；
- port：一个预先固定的 port budget，不扫 token 数、层位或 rank；
- 任务：先做现有 explicit-feedback workload；只有机制通过后才扩 next-item；
- 训练：Parent 正常训练；Current 必须在新窗口提升 Full quality，同时接受预先冻结的 state-contract 约束。

### 8.2 四个必须同跑的 arm

1. `RMT-style port only`：只有 causal separator，没有 release constraint；
2. `frozen producer`：port producer 保持 Parent，不迁移，只更新 Current reader，作为 aLoRA-like exact control；
3. `generic port aligner`：从 Parent port 回归 Current Exact port，作为 mapping/compatibility control；
4. `proposed release-law model`：只有在先提出非 regression 的 commuting/deletion construction 后才允许加入。

Dense HSTU Parent/Current 只作为 quality 与 Exact-All cost reference，不要求 port model bitwise 等价 dense
HSTU。

### 8.3 必须执行的顺序干预

对同一历史和 request，比较：

```text
Current exact port
Parent port direct reuse
Parent -> Current migrated port
migrate -> append
append under Parent -> migrate
migrate -> append -> evict
Current exact rebuild after the same append/evict sequence
```

必须报告 port tensor error 之外的 ranking output、AUC/log-loss recovery，并检查：

- cutover/append 两种顺序是否一致；
- append `1/8/32` 个事件后 recovery 是否持续；
- whole-segment eviction 后是否与 Current exact rebuild 对齐；
- 所谓 Current improvement 是否来自 reader，还是 producer 实际上被 compatibility objective 冻住；
- migration 是否完全不读取 raw sealed segment；若读取，bytes/FLOPs 必须计入。

### 8.4 重新开启门槛

只有同时满足以下条件，才把它升级为后续研究方向：

- ported Current Full 相对 ported Parent Full 有真实质量提升，并接近 dense Current 的 task quality；
- migration constructor `<20%` 对应 ported Exact-All，且不是把大部分成本转移到 cache-build/replay I/O；
- probability recovery 至少 `.80`，目标 `.90`；
- append 后不系统漂移，eviction lineage 有明确语义；
- proposed arm 稳定胜过 generic aligner 与 frozen-producer control；
- 方法贡献来自可陈述、可验证的 release/transition algebra，而不是 memory-token 数量、mapping capacity 或
  compatibility-loss 权重。

任一失败即维持 NO-GO。尤其是：若 generic aligner 已达到相同结果，或 proposed arm 只能靠几何 mapping
解释，就不符合本项目的创新门槛。

## 9. 最终建议

1. **当前不执行。** 不在现有 v0..v5 上写代码、不做 GPU preflight、不读 confirmation。
2. **当前不晋级。** causal ports 不应写成 Insight 2 或 Design 1；最多进入 Discussion 的 impossibility/
   co-design boundary。
3. **若保留 future work，只保留 precise question：** 能否构造一个对相邻 release 可验证 commuting、对
   append/eviction 闭合、同时不冻结 history semantics 的 compact state machine？
4. **在回答这个 construction 以前，不训练。** “先训练 RMT 再看看能否对齐”只会得到 memory tokens 加
   mapper，已经可以预判达不到 paper-height gate。

严格裁决：**NO-GO for Design 1；prospective canary 仅在先出现新的 algebraic release mechanism 后才有
资格申请授权。**
