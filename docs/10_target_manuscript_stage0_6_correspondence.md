# Target manuscript 与 Stage 0–6 当前实现对照

> 状态：2026-07-28。本文记录的是**重写前目标稿**与 Stage 0–6 证据的对照；旧稿可从
> Git 历史恢复。重写结果已经落在
> [`manuscript_v3_target_en.md`](../paper/cohortkv/manuscript_v3_target_en.md)。
> 本文不改变实验语义或研究状态。
> 当前事实仍以
> [`08_core_insights_and_roadmap.md`](08_core_insights_and_roadmap.md)、
> [`eval_protocol.md`](eval_protocol.md) 和
> [`final_summary_seed0.json`](../results/system/cohortkv_single_config_full_chain_v1/final_summary_seed0.json)
> 为准。
>
> 本文的“剩余证据”和 Stage 7 表述是 Stage 0–6 冻结时的差距快照，不是当前实施队列。
> 当前 D2 的设计与执行状态分别见
> [`DESIGN2_FINAL_PLAN.md`](future_design/DESIGN2_FINAL_PLAN.md) 和
> [`DESIGN2_FOUR_STAGE_EXECUTION.md`](future_design/DESIGN2_FOUR_STAGE_EXECUTION.md)；
> Stage B 当前 W1/W2 已完成、W4 未冻结的事实见
> [`DESIGN2_STAGE_B_HANDOFF.md`](future_design/DESIGN2_STAGE_B_HANDOFF.md)。
>
> **当前命名覆盖：** Stage 0–6 的 compiler、direct-old-K/V operator、bounded renewal 和
> transaction closure 现在整体归入 D1。当前 D2 是 immutable-action physical-wave
> compilation：D1 决定 what must be recomputed，D2 决定 fixed work 如何移动、组批、执行和
> 发布。本文 §5 的旧“三贡献槽位”是重写时快照，已被
> `08_core_insights_and_roadmap.md` 的 D1/D2 结构取代。

## 1. 结论

Stage 0–6 没有“照原稿实现完”，而是完成了一次有实质修正的单配置研究闭环：

- **保留下来的主线**：model-version-stale HSTU K/V、version-pair shared repair、
  label-free semantic contract、direct-old-K/V transform、matched destination transaction。
- **实质改变的主线**：主运行输入从额外 `Norm(x)` capsule 变为已有 old K/V；固定历史
  15%–25% refresh 变为增长历史下的 bounded-renewal H12；新增行为的 append 被移到
  migration 之后并排除在 migration 计时之外；完整 runtime guard 被收缩为 job-level
  preflight、执行前 exact fallback 和代表性 copy-on-write fault closure。
- **被降级或淘汰的支线**：normalized-capsule full-cohort 路径、逐 cache risk threshold、
  jagged 性能主张、SSD/INT8/capture economics、per-wave runtime rework/resume。
- **仍未完成的论文证据**：当前 16L/H512 配置的新 training seeds、selective frontier 和
  H12 lifecycle 的预声明跨 seed/容量/数据集复现，以及一个包含完整 state movement 的
  growing-history end-to-end 系统结果。

因此，目标稿是一个**较好的研究蓝图**，但已经不是一篇适合直接补数字投稿的最终论文。
它应当按当前证据重写，而不是继续在旧结构中填 `TBD`。

## 2. 核心设计逐项对照

状态含义：

- **保留**：目标稿逻辑与当前实现一致；
- **替换**：逻辑目标保留，但关键机制已经变化；
- **降级**：已经实现，但更适合放在 Implementation 或负结果，而不是核心 design；
- **部分完成**：只完成了明确收缩后的边界；
- **Stage 7**：不属于当前 seed-0 冻结结果。

| 目标稿中的模块或主张 | Stage 0–6 的实际状态 | 判断 | 新论文应如何处理 |
|---|---|---|---|
| 模型更新使持久 K/V 发生 version staleness；reuse 便宜，exact replay 昂贵（§1–2） | 问题定义、stale-inference 语义和跨表/容量 motivation 均保留 | **保留** | 继续作为问题主线，但不声称每个数据集、容量或时间点都有正的 ranking maintenance gain |
| Version cohort `(source_version, target_version)` 是编译、批处理、放置和发布单位（§3） | Compiler、program、extent、manifest 和 lineage 均携带 version identity | **保留** | 强调它是工作组织与摊销单位，不是“某用户可以安全 reuse”的预测器 |
| 每条记录保存所有层的旧 `Norm(x)` migration capsule（§3.1、§4.1） | `Norm(x)` 仍用于 fit、certificate 和 reference；主 hot runtime 不再保存它 | **替换** | 明确区分 calibration representation 与 runtime representation，不能再把 capsule 写成主运行状态 |
| 以 `fresh - cheap` 为 residual，学习 version-pair shared affine 并折叠进一次 projection（§4.1） | Stage 2 已发布三组 serialized FP16 program；证书和 executable fallback plan 均通过 | **保留且完成** | 仍是核心算法，但应与 direct-old-K/V reparameterization 合并为一个完整 design |
| Cache error、score cosine、Top-100 overlap 的 label-free contract（§4.2） | Seed-0 的 fit/selection/certificate/final-test 角色隔离、阈值 sweep 和部署表示证书已完成 | **保留；统计复现待 Stage 7** | 保留为 compiler contract；不能把最终 AUC/NDCG/Hit 反向用于 routing |
| `compiled → residual-p → exact` action library（§4.3） | Compiled 与 exact 可执行；residual-p 被修正为需要 raw history 和 BF16 hidden suffix，无该状态时直接跳 exact | **保留但收缩** | 把 residual-p 写成有条件的 structural tier，不得暗示默认 capsule 足够执行 |
| DroidSpeak-style selective-layer action 可能成为强候选（§4.3、§8.4） | 53 个 interval、177 个 frontier 点已完成；最强 selective 约 `0.698×` exact 且证书失败，compiled 同时更便宜且恢复更高 | **外部 baseline 完成，结论降级** | 保留为 independently tuned、certificate-failed diagnostic；publishable fallback 是 exact |
| 将 certified affine 通过 source K/V projection 的 right inverse 组合到 old K/V（§4.5） | Stage 4.5 完成：condition number `5.97–10.74`，额外 per-record state 为 0，三组 program 共 `100.78 MB` | **保留且成为主路径** | 从旧稿的后置优化提升为核心算法的一半；同时披露 full-row-rank、provenance 和 exact-source 假设 |
| 独立的 fused source-to-K/V operator 作为 Design 2（§5） | Reference、packed、fused 已共用 unpadded extent API；fused 对 packed 的 resident 中位优势为 `1.995×` | **完成，但应降级定位** | 保留技术细节和消融；它是实现 compiled/direct transform 的关键工程，不足以单独承担与 compiler 同级的科学贡献 |
| Length bucketing 与 jagged/page compaction（§5.2） | Bucketing 保留；jagged 正确但端到端无显著收益 | **一正一负** | Bucketing 放 Implementation；jagged 作为负结果，不再作为设计主线 |
| Destination-oriented 1/2/4-GPU HBM/DRAM full-cohort engine（§6） | 682 records、30 个 method/destination/GPU 点均完成并通过 transaction/correctness gate | **保留且完成** | 继续作为系统闭环，但 LPT、queue、manifest mechanics 等标准机制放 Implementation |
| Normalized capsule 能把 kernel 优势转成 full-cohort 优势（§6、§8.5） | 0/6 matched endpoint 胜过 exact；17.82-GB source 的 read/decode/pinning 占 `91.35%–96.91%` | **已证伪** | 保留为决定性负结果，用来解释为什么必须改成 direct old K/V |
| Direct-old-K/V hot-HBM complete job（§4.5、Table 8b） | 1/2/4 GPU 为 `0.930/0.494/0.255 s`，paired exact 为 `18.695/9.729/4.766 s`，即 `20.11×/19.71×/18.72×`；该路径以约 35.6 GB 已有 old K/V 为前提，exact 的 raw-history source 仅约 89 MB | **保留且完成** | 严格限定为 existing-old-K/V hot-HBM、same-location、prepublished-program data plane；同时报告 standing HBM，不把它写成相同 source-byte footprint、cold storage 或 serving SLO 结果 |
| 固定历史、edge severity + age、15%–25% exact、depth≤4 lifecycle（§8.6） | Stage 4.6 完成，累计成本 `0.2134×`；但它只是 fixed-history control | **完成但被主结果取代** | 保留为机制演进或附录，不再作为最终 lifecycle headline |
| 逐 cache norm/risk threshold 判断谁 exact（目标稿 lifecycle 前身） | 会造成 `0.15%–65.1%` exact refresh wave；norm shift 对真实误差的平均 Spearman 仅 `0.0341` | **淘汰** | 作为负结果；不得恢复“强 per-cache risk predictor”主张 |
| 真实增长历史的连续迁移 | 目标稿没有完整吸收；Stage 4.7–4.9 新增 canonical-date causal history、previous-actual recursion 和 next-window evaluation | **当前 D1 lifecycle 核心** | 并入 D1 的 action-plane/bounded-renewal contract；不得再称当前 Design 2 |
| 新行为窗口在 migration 前由 source model append | Stage 4.9 已纠正为先迁移 retained prefix，再由 target model append | **旧语义被替换** | 正文必须使用新顺序；Stage 4.7/4.8 数字只能作为旧协议诊断 |
| 新行为 append 计入 lifecycle/migration time | Stage 4.9 将 target-model append 单独计量并排除在 `U/E` 两边之外 | **计量边界被替换** | 主 migration 指标写成 matched retained-prefix `ΣU/ΣE`；若报告 cache-ready E2E，必须另跑最快 same-output exact |
| H12 bounded renewal 与 token-debt scheduler | 目标稿未覆盖；H12 为冻结候选，`U/E=0.100017`，AUC/NDCG@100/Hit@100 ratio 为 `1.000039/0.997463/1.000000`；token-debt 为 `0.071319` 成本端点 | **当前 D1 lifecycle 核心** | H12 属于 D1 的 bounded-renewal mechanism；token-debt 只作为无 per-cache deadline 的成本下界 |
| Runtime sampling guard、逐 wave 自动升级和 re-migration（§6.2） | 未实现。Stage 5 只实现固定 job-level preflight、semantic canary 和执行前 cohort-wide exact fallback | **部分替换** | 不写 “automatic runtime guard completed”；准确写成 pre-execution fallback，runtime invalidation/rework 仍开放 |
| Destination transaction 与 failure injection（§6.4–6.5） | 两卡 COW normal、semantic fallback、mid-job abort、pre-commit abort 均通过；两个 abort 后旧 682 records 全部可读 | **最小闭环完成** | 作为系统可靠性证据；必须与 `20×` 的 extent-reclaim 性能 mode 分开，后者 `abort_safe=false`，而 COW mode 没有吞吐主张；before-first/publication fault、journal、resume 和跨 job retirement 仍不声称 |
| Durable SSD、remote object、automatic tier selection（§6、§8.7） | POSIX/remote 只有接口正确性；没有具名物理 SSD/GDS/remote 性能 | **降为 optional post-v1** | 从主贡献和完成条件删除；不能把 warm filesystem source read 称为 SSD destination 结果 |
| RQ5 capsule capture、INT8、break-even economics（§8.8） | 只完成基于 Stage 2/4/4.5 artifact 的 source-state accounting；没有新 capture/INT8/full-cohort break-even matrix | **主问题已失去必要性** | 删除独立 RQ5；把 capsule 负结果、program bytes 和 offline setup cost 合并进系统成本分析 |
| Stage 6 “完整论文矩阵” | Stage 6 是 CPU-only deterministic assembly，校验 18 个 source artifacts 并发布八个 sidecars，没有重跑 GPU matrix | **报告闭环完成** | 作为 artifact/reproducibility 说明，不得包装成新的算法或性能实验 |

## 3. 目标实验网格与当前证据差距

| 目标稿的证据要求 | 当前已有证据 | 仍然缺少什么 | 论文处理 |
|---|---|---|---|
| RQ1：跨数据表和 3×3 容量说明问题存在 | KuaiRand/QB/QK motivation、capacity 和 27-chain mechanism evidence 已完成 | 第三个真正独立工业域并不存在；QB/QK 是同一 Tenrec collection 的相关表 | 可以写，但限定为 cross-table/capacity，不写 universal generality |
| RQ2：27-chain compiled mechanism replication | 3 tables × 3 capacities × 3 non-discovery seeds 已完成；mean cost `0.1211×`、K/V recovery `0.5867` | 这组较小容量链不能替代当前 16L/H512 deployed compiler 的新 seed 复现 | 保留为 mechanism replication |
| RQ2：冻结的 16L/H512 compiler 跨新 training seeds | 只有 seed-0 deployed artifact/certificate | 目标稿 Table 6 的新 seeds、per-seed final task/cost/certificate | **Stage 7 必做** |
| RQ3：compiled 对 selective 的 primary frontier | KuaiRand-long seed-0 的 177 点和独立 certificate 已完成 | 无 | 可作为 controlled primary result |
| RQ3：跨 seed、KuaiRand-medium、QB-large 的 selective frontier | 未完成 | 预声明的新 seed/容量/数据集 cells | **Stage 7 必做或收缩 dominance claim** |
| RQ4：full-cohort matched HBM/DRAM transaction | Normalized source 的 30 点矩阵完成；direct-old-K/V hot-HBM 1/2/4 GPU 完成 | Direct route 没有 cold/DRAM/SSD 泛化，也不需要为 v1 强行补齐 | 以 hot-HBM operating regime 报告 |
| RQ4：连续增长历史的低成本、高任务质量 | H12 11-edge same-device retained-prefix `U/E` 和最终任务指标完成 | 新 training seeds、跨容量/表复现；H12=12 但当前链只有 11 edges，未观察完整 renewal cycle | **Stage 7 核心**，并保留 horizon 限制 |
| RQ4：连续链的完整 end-to-end state movement | Stage 4.9 使用 groupwise CPU host staging，H12 另报 662.87 GB logical H2D/D2H movement | 没有一条 GPU run 同时覆盖 11-edge growing history、hot-HBM full cohort、完整 movement 和 transaction | 若投 systems venue，这是最值得补的一条代表性整合证据；否则必须把主张拆开 |
| RQ4：fallback 和 failure safety | Stage 5 的 preflight exact fallback 与两个 COW abort case 完成 | Per-wave sentinel、online invalidation/rework、journal/resume | 当前 v1 不必补，但不得写成 runtime recovery system |
| Compiler/setup 的总体经济性 | 已汇总 Stage 2 fit/runtime-prepare/certificate：一次性 setup `308.90 s`，direct program composition `1.55 s`；一张卡 682-record exact 仅 `18.69 s` | 尚无基于 cohort size、复用次数和 version-pair 生命周期的实测总体 break-even | `20×` 只能称 prepublished-program data-plane speedup；若不补 break-even，就把 overall economics 明确列为限制 |
| RQ5：capsule economics | FP16 capsule 的负结果、direct route 零额外 state、program/offline cost ledger 已完成 | INT8/capture/quantized run/SSD break-even | 删除独立 RQ；均列 optional |
| Artifact-to-claim 完整性 | Stage 6 的 18 sources、8 sidecars、schema、claim map 和 code snapshot 均通过 | 重写前目标稿有 35 个 `TBD`，其中大量证据仍属于 Stage 7 或 optional post-v1 | 当前稿已改为 0 个占位符，开放证据以 limitation 明示；manuscript ledger 已重新生成 |

## 4. 重写前目标稿本身是不是一篇好论文

我的判断是：**它是一份好的 research blueprint，但不是一篇好的当前投稿稿件。**

| 维度 | 判断 | 原因 |
|---|---|---|
| 问题定义 | **强** | “完整但版本陈旧的 K/V”区别于 eviction、same-model restoration 和普通 append；边界清楚且有实际系统代价 |
| 研究纪律 | **强** | 强调相同 source/destination/commit boundary、training seed 才是 replication unit、负结果不隐藏、exact 不是 ranking upper bound |
| 核心机制 | **中上** | Shared affine compilation + direct-old-K/V reparameterization 形成了完整机制；但 affine fitting、pseudoinverse、fused GEMM 单独看都不是强新算法 |
| 原稿三 Design 划分 | **需要重构** | Fused operator 和 transaction engineering 被抬得过高，真正新的 repeated-migration lifecycle 反而被放在 Evaluation 的 “RQ4 continued” |
| 系统证据 | **强但分段** | Kernel、full cohort、hot-HBM、lifecycle、COW failure 各自扎实；但尚无一个 run 把 growing history、完整 movement、hot-HBM 和 transaction 全部合在一起 |
| 总体经济性 | **尚未闭合** | 数据面很快，但当前 682-record accounting 中一次性 compiler/certificate setup 比一次 exact job 大一个数量级以上；论文尚无总体 break-even 主张 |
| 统计与泛化 | **当前偏弱** | 27-chain mechanism replication 很有价值，但 deployed 16L/H512、selective frontier 和 H12 仍主要是 seed-0 |
| 写作一致性 | **不合格于直接投稿** | §8.1 把 RQ2 写成 replicated，但 Table 6 仍是 TBD；evidence grid 写 RQ3 三链，正文却承认只有 primary seed-0 |
| 主结果组织 | **过满** | `0.121×` mechanism、`20×` hot-HBM、`0.213×` fixed lifecycle、`0.100×` corrected lifecycle 同时争夺 headline，容易让审稿人怀疑比较边界不一致 |
| 审稿攻击面 | **较大但可控** | HSTU-specific affine/full-row-rank 条件、单主配置、H12 heuristic、host-staged lifecycle、相关 Tenrec tables、无 request trace 都必须主动收缩 |
| 总体 | **值得写，但必须重写** | 当前仓库已经比原目标稿更诚实、更完整；如果仍按旧稿结构填数，反而会削弱这项工作 |

目标稿最值得保留的不是“三个 design 的名字”，而是三种写作纪律：

1. 从可验证的 version-staleness 问题出发；
2. 把 kernel、source state、destination 和 publication boundary 放在同一个成本合同中；
3. 允许实验否定原方案，并把负结果转化为下一项设计依据。

## 5. 历史重写建议（已被当前 D1/D2 主线取代）

当前 canonical paper story 是：

1. **M1/D1：** stale reuse 与 all-exact full-history replay 之间的 semantic–compute dilemma；
   D1 通过 version-cohort migration 和有界 exact reset 形成 immutable logical action
   sparsity。
2. **M2/D2：** row-sharded exact maintenance 的 physical cost 不随 logical work 自动线性
   下降；D2 通过 `(S,R)` extents、segmented suffix-only state 和 merged exact pool 将固定
   actions 降低为 physical sparsity。
3. **D3：** communication-aware semantic selection、organic mixed versions 和跨 wave
   renewal 仍是未来方向，不属于当前 D2。

以下三槽位是本文件产生时的历史建议，仅用于理解为何旧稿需要重写：

当前证据更适合下面三个贡献槽位：

1. **Version-cohort compiler with direct old-K/V reparameterization**
   将 shared residual 编译成 affine，再组合到已有 old K/V；normalized capsule 只用于
   calibration/reference。这里同时包含 label-free empirical semantic gate 和
   exact-terminated plan。“Certificate” 应始终解释为 held-out empirical contract，而不是
   distribution-free 或形式安全证明。

2. **Bounded migrate-or-exact lifecycle for growing histories**
   对 previous-actual cache 连续更新，以 deterministic bounded renewal 决定 migration 或
   exact；最终评价轴是 migration GPU cost 与 AUC/NDCG@100/Hit@100，而不是把中间
   K/V fidelity 或 norm shift 当 task oracle。

3. **Destination-oriented runtime and transactional closure**
   Fused direct-write operator、length bucketing、1/2/4-GPU placement、matched HBM/DRAM
   transaction、job-level preflight、exact fallback 和 COW commit/abort 共同构成系统实现；
   不再把每一项标准工程分别包装成独立创新。

相应地，评测可以收敛为四个问题：

- **RQ1：** version-stale K/V 的机会和适用边界是什么？
- **RQ2：** compiled/direct migration 相对 reuse、cheap、selective 和 exact 的
  cost–semantic frontier 是否复现？
- **RQ3：** 在增长历史和连续模型更新下，bounded renewal 能以多少 migration cost
  保留多少最终推荐质量？
- **RQ4：** 这些收益在 full-cohort source/destination/transaction 边界下是否仍成立，
  其容量、movement、fallback 和 failure 边界是什么？

原 RQ5 不再独立存在。Normalized capsule economics 应作为“为什么初始系统失败、为什么
direct old K/V 必要”的系统分析；INT8、SSD、GDS、remote 和 online resume 均为未来工作。

## 6. 写新稿时不能直接沿用的旧句子

- 不能写 frozen 16L/H512 compiler 已 multi-seed replicated。
- 不能写 selective dominance 已跨 seed、容量和数据集成立。
- 不能把 Stage 4.9 的 `0.100017×` 称为 end-to-end state-movement 或 full-HBM lifecycle。
- 不能写 H12 已观察完整 12-edge renewal cycle；当前只有 11 次更新。
- 不能写 automatic runtime guard、online rework 或 resume 已完成。
- 不能把 Stage 5 的 correctness closure 写成 throughput、durability 或 serving result。
- 不能把 warm filesystem、POSIX interface 或 host staging 写成 durable SSD result。
- 不能把 Stage 6 描述成一次新的完整 GPU rerun。
- 不能把 fixed-history `0.2134×` 与 corrected growing-history `0.100017×` 混为同一协议。
- 不能把 27-chain 小容量 mechanism replication 当作当前 16L/H512/H12 的 replication。
- 不能把 `20×` hot-HBM data-plane speedup 写成包含 `308.90 s` compiler/certificate setup 的
  overall speedup。
- 不能暗示逐 extent reclaim 的高速路径同时具有 Stage 5 COW 的 abort safety。
- 不能把 empirical semantic “certificate” 写成分布外或形式化正确性保证。

## 7. 当前最重要的论文风险

按优先级排序：

1. **Total-economics gap**：当前一次性 Stage-2 setup 为 `308.90 s`，另有 `1.55 s`
   direct-program composition；同一 682-record 一卡 exact job 仅 `18.69 s`。若 program
   不能跨足够大的 cohort 或足够多次 update 摊销，`20×` 数据面结果并不等于整体收益。
2. **Replication gap**：当前 deployed compiler、selective frontier 和 H12 lifecycle 仍以
   seed-0 为主；这是 Stage 7 的首要任务。
3. **Evidence-composition gap**：hot-HBM `20×` 和 growing-history `0.100×` 来自不同
   边界，不能在文字中暗示它们已经组成一个 monolithic end-to-end result。
4. **Novelty framing**：不要分别夸大 affine、pseudoinverse、Triton 或 COW；创新应落在
   version-pair amortization、zero-extra-state transform、bounded lifecycle 和完整系统合同的组合。尤其需要正面对比直接 `old K/V → fresh K/V` affine、普通
   `z_old → fresh K/V` ridge，以及合适的低秩/正交映射，否则容易被评价成“线性回归加工程”。
5. **Safety/performance mode split**：`20×` 路径通过逐 extent reclaim 降低峰值但不是
   abort-safe；Stage 5 COW 路径证明了 abort safety 却没有吞吐结果。两者不能合并成一句
   “既快又 failure-safe”。
6. **Generality gap**：主系统链只有 simplified HSTU/KuaiRand；QB/QK 是相关表，且没有
   request trace。跨架构、真实 serving 和 SLO 不是当前主张。
7. **Heuristic attack**：H12 是预注册、label-free、带 per-cache deadline 的系统策略，
   不是最优调度定理；应报告完整 Pareto 和选择依据，而不是称其为 optimal。
8. **External novelty audit**：本文只评价仓库内的逻辑与证据，没有完成面向投稿日期的
   全量 primary-source related-work novelty 审计；投稿前仍需单独执行。

## 8. 关键证据入口

- 原目标结构：
  [`manuscript_v3_target_en.md`](../paper/cohortkv/manuscript_v3_target_en.md)
- 当前研究事实：
  [`08_core_insights_and_roadmap.md`](08_core_insights_and_roadmap.md)
- 当前实验语义：
  [`eval_protocol.md`](eval_protocol.md)
- Stage 1 selective frontier：
  [`stage1_frontier_summary.json`](../configs/cohortkv_single_config_v1/stage1_frontier_summary.json)
- Stage 2 deployed compiler：
  [`stage2_compiler_summary.json`](../configs/cohortkv_single_config_v1/stage2_compiler_summary.json)
- Stage 3 operator：
  [`stage3_operator_summary.json`](../configs/cohortkv_single_config_v1/stage3_operator_summary.json)
- Stage 4 full-cohort matrix：
  [`stage4_system_summary.json`](../configs/cohortkv_single_config_v1/stage4_system_summary.json)
- Stage 4.5 direct old-K/V：
  [`stage4_5_source_plan_summary.json`](../configs/cohortkv_single_config_v1/stage4_5_source_plan_summary.json)
- Stage 4.9 corrected lifecycle：
  [`stage4_9_same_device_confirmation_seed0.json`](../results/system/cohortkv_single_config_full_chain_v1/stage4_9_same_device_confirmation_seed0.json)
- Stage 5 COW closure：
  [`stage5_full_cow_theta0_theta1_seed0.json`](../results/system/cohortkv_single_config_full_chain_v1/stage5_full_cow_theta0_theta1_seed0.json)
- Source-state/offline-cost accounting：
  [`stage5_source_state_accounting_seed0.json`](../results/system/cohortkv_single_config_full_chain_v1/stage5_source_state_accounting_seed0.json)
- Stage 6 frozen aggregate：
  [`final_summary_seed0.json`](../results/system/cohortkv_single_config_full_chain_v1/final_summary_seed0.json)
- 当前稿 manuscript disposition（0 个 `TBD`）：
  [`stage6_tbd_disposition_seed0.json`](../results/system/cohortkv_single_config_full_chain_v1/stage6_tbd_disposition_seed0.json)
