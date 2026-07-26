# CohortKV 单配置全链路开发计划

> 状态：2026-07-27 起生效。本文档定义当前唯一的实施顺序。
>
> 当前阶段只在最成熟的 KuaiRand 长上下文配置上完成一次论文级 vertical slice。它是
> **开发与集成阶段**，不是新的多 seed 确认性证据。完成并冻结 v1 后，才进入新 seed、
> 多数据集和多模型容量扩展。
>
> 研究事实与实验语义仍分别以
> [08_core_insights_and_roadmap.md](08_core_insights_and_roadmap.md) 和
> [eval_protocol.md](eval_protocol.md) 为准。目标论文
> [manuscript_v3_target_en.md](../paper/cohortkv/manuscript_v3_target_en.md)
> 描述希望闭合的论文形态，但其中的预期方向和 `TBD` 不是实现合同，也不是研究事实。

## 1. 目标、固定配置与阶段边界

本阶段的目标不是追求某一组预写数字，而是让论文中的
`compiler -> capsule/operator -> destination engine -> complete manifest`
在一个真实配置上完整闭合，并让所有主要 baseline、消融、失败路径和成本项可以在统一协议下
运行。

默认开发配置是：

- 数据：`kuairand_long_context_4plus12_exploration_v1`；
- 时间边界：D1-D4 为 base，D5-D16 为 updates，固定 theta11/D16 endpoint；
- 模型：16 层、hidden/K/V width 512、最大历史长度 2,048，约 0.181B 参数；
- 训练链：seed 0 的 theta0-theta12；
- 迁移 cohort：theta0、theta4、theta10 到 theta11；
- 质量用户角色：40 fit、60 program selection、60 certificate、522 final test；
- 系统 workload：当前 682 个 eligible records 的完整固定集合；source-version 分配必须由
  Stage 0 冻结的、与标签无关的规则生成；
- 设备：1/2/4 张 NVIDIA A40；
- 主 capsule/K/V 表示：FP16；exact 路径可使用独立调优的 BF16 compute，但必须发布相同的
  FP16 目标 K/V。

如果实际 artifact 与上面任一项不一致，先更新本文档和新 protocol，再运行实验。不得让脚本
默认值悄悄改变论文 workload。

单配置阶段允许开发和调优，但必须遵守两个边界：

1. compiler 的 fit、selection、certificate 和 final-test 用户不能混用；
2. baseline 或系统参数可以在预先声明的 development role 上调优，最终 workload 不能反向
   用来选方法。

## 2. 执行原则：目标稳定，实现可变

### 2.1 不为预写设计强行找结果

目标稿中的 full affine、Triton kernel、sentinel、SSD 等是当前最合理的候选实现，不是不可
修改的教条。任何阶段发现前提错误、机制无效或成本不成立时，应当：

1. 暂停受影响的后续阶段；
2. 保存负结果和失败原因；
3. 回到该模块的逻辑目标；
4. 选择同一接口下更简单或更有效的实现；
5. 为实质变化建立新 protocol；
6. 重新运行受影响的下游实验，并按真实结果修改论文。

实现不同没有问题，但不能在改变语义后继续复用旧结果或旧 protocol 名称。

### 2.2 三个论文模块的逻辑目标

模块接口比当前实现手段更稳定：

- **Compiler 的目标**：把 source/target version pair 的共享适配移出 per-record 路径，
  发布带语义证书和 fallback 的不可变 program。
- **Operator 的目标**：以最少的中间状态把 capsule 变成最终布局 K/V，正确处理长度与
  padding。
- **Engine 的目标**：在有界工作内存下处理完整固定 cohort，并在显式 destination 上只发布
  一个完整 target-version manifest。

当某个首选机制失败时，优先做“同槽替换”，而不是增加第四、第五项贡献。

### 2.3 证据与协议纪律

- 单配置结果统一标记为 development/controlled evidence。
- 训练 seed 才是后续统计复现单位；系统 timing repeats 只描述稳定性。
- 每个 speedup 都必须声明 source、destination、dtype、layout、durability 和包含的阶段。
- HBM、DRAM 和 filesystem 是不同 endpoint，不能互相换算成算法 speedup。
- exact 是 current-model K/V 的语义 reference，不是 ranking metric 的保证上界。
- 负结果进入记录，不通过改指标、改用户或换 protocol 名称隐藏。

## 3. 目标模块的基本设计

### 3.1 Design 1：Version-cohort migration compiler

默认设计沿用目标稿 §4。

对 cohort `(v,t)`，当前 projection 先产生：

```text
cheap = old Norm(x) @ current P + current bias
residual = fresh current K/V - cheap
```

compiler 在 fit users 上学习 attention-use-weighted full-affine residual，并在发布前折叠为
一组新的 projection weights/bias。执行期仍是一种与 current projection 同形状的 affine
program，而不是逐 record 执行额外 adapter。

默认 action library 是：

1. compiled affine；
2. residual-\(p\) structural replay；
3. exact recomputation。

这里的 residual-\(p\) 不是只依赖默认 capsule 的算子。它需要 raw history，以及从第
\(p\) 层到末层的每层旧 pre-block hidden state；这部分是单独计费、单独持久化的辅助
representation。没有该 representation 时，合法 fallback 是 exact，而不是从 normalized
capsule 猜回 hidden state。

reuse 是零维护 anchor，不是可发布的同步 action。cheap projection 是必要消融。verified
plan 使用 cache error、score cosine 和 top-100 overlap 三个无标签视图，记录成本、证书、
selected action 和有序 fallback。

如果 full affine 在当前配置上不能稳定通过合同，可以在同一 compiler 接口中：

- 选择 residual-\(p\) 或 exact；
- 发布 affine 与 selective/structural action 组成的单一混合 program；
- 缩小主张为特定 fidelity 区间的 Pareto 优势。

不得为了保留“full affine 总是被选中”的句子而放宽 final-test 规则。

### 3.2 Design 2：Migration capsule 与 capsule-to-K/V operator

默认 capsule 保存每层旧版本 `Norm(x)`、record ID、valid length 和 migration anchor。
served K/V target 与 migration anchor 必须始终分离；近似生成 theta11 K/V 不会自动把
capsule anchor 改成 theta11。

operator 的三个实现层是：

1. FP32 reference，作为数值 oracle；
2. packed FP16 framework path，作为强 library baseline；
3. fused direct-write path，融合 affine、bias、length mask 和 K/V split，直接写最终布局。

首选实现是现有 Triton kernel。若代表性 workload 上 Triton 相对 library GEMM 的优势消失，
可以改用 library GEMM 加 fused epilogue，或连续 staging extent 加异步 scatter。只要最终
仍是一个明确的 capsule-to-K/V operator，就不需要强行保留某种 kernel 实现。

Jagged/page compaction 已是当前 trace 上的负结果。除非 full-cohort profile 暴露新的 padding
或 launch 瓶颈，否则不再搜索它；默认保留 dense/length-bucketed path。

### 3.3 Design 3：Destination-oriented full-cohort engine

默认 job contract 沿用目标稿 §6：

```text
begin(job, target_version, expected_record_ids)
  -> stage(extent_id, target K/V)*
  -> commit(complete target manifest)
  -> or abort()
```

engine 负责：

- 从 source manifest 惰性读取 capsule 或 raw history；
- 按 source cohort 和 length bucket 组织 batch；
- 保持 program resident；
- 以 byte-weighted LPT 将 extents 分给 1/2/4 GPU；
- 用 bounded waves 和 publication queue 限制瞬时工作集；
- 让 compiled、selective-layer、residual-\(p\) 和 exact 走同一 destination transaction；
- 在完整、无重复 coverage 后才提交 manifest。

HBM 与 DRAM 是主 endpoint。POSIX filesystem 是同一 transaction 的次级实例；remote object
只保留接口正确性，不做网络性能主张。

如果物理 cohort extent 重排不值得，可用 secondary-index sorted gather；如果三流重叠长期
失衡，可改 wave/buffer 策略。逻辑目标仍是 bounded full-cohort transformation，不要求保留
某一种组织方式。

### 3.4 Runtime guard 与 failure boundary

默认候选是目标稿中的 per-wave sentinel：抽样检查当前 cohort 的 label-free semantic view，
违反 published contract 时单调升级，并在 commit 前重迁移该 cohort 已生成的 extents。

sentinel 必须先回答三个问题：

- 它实际观察什么 reference，reference 的生成和保存成本是多少；
- 统计规则能否在当前 wave/cohort 大小下成立；
- 它的开销与误报/漏报是否优于只做离线 certificate。

如果 runtime sentinel 在语义上不成立或成本过高，不强行实现。可以改成：

- job 启动前的 exact canary/preflight verification；
- 每 cohort 的固定 verification wave；
- 只保护 artifact corruption 的 checksum/version validation；
- 失败后直接使用 published fallback 重启该 cohort。

论文必须按最终实际机制命名，不能把 preflight check 写成在线 sentinel。

## 4. 单配置实验设计

| Paper RQ | 本阶段实验 | 主要比较 | 主要输出 |
|---|---|---|---|
| RQ1 opportunity | 复用现有 KuaiRand motivation/capacity evidence | frozen、reuse、fresh、固定周期诊断 | staleness tax、age/drift/task 边界 |
| RQ2 compiler | 当前长上下文 cell 的 verified compiler 与阈值扫描 | reuse、cheap、compiled、residual-\(p\)、exact | certificate、cost、fidelity、task deviation、compile amortization |
| RQ3 closest baseline | 单配置 selective-layer cost-fidelity frontier | compiled ladder、selective-layer、reuse/exact anchors | certified Pareto frontier、独立调优结果 |
| RQ4 system | 682-record full-cohort、1/2/4 GPU、HBM/DRAM | compiled、certified selective、exact、no-transform | completion、throughput、bytes、峰值内存、breakdown、commit |
| RQ4 robustness | forced degradation 与 failure injection | normal、escalated、abort、resume | detection、rework、visibility、cleanup |
| RQ5 economics | FP16/INT8 capsule 与创建成本 | no capsule/exact、FP16、INT8 | bytes、capture/dequant cost、fidelity、break-even |

RQ1 不因本计划重新训练。RQ2-RQ5 在本阶段只建立单配置完整证据模板；跨 seed 和跨数据集结论
留给 v1 冻结后的下一阶段。

## 5. 分阶段实施

### Stage 0：冻结实验蓝图和 artifact contract

状态：**已完成（2026-07-27）**。冻结产物见
[`COHORTKV_SINGLE_CONFIG_FULL_CHAIN_V1.md`](../experiments/system/COHORTKV_SINGLE_CONFIG_FULL_CHAIN_V1.md)
和 `configs/cohortkv_single_config_v1/`。

先写协议，不写新性能结论。

任务：

- 冻结 682-record complete job manifest 和与标签无关的 source-version 分配；
- 明确 40/60/60/522 质量角色与系统 workload 的关系；
- 为 compiled、selective-layer、residual-\(p\) 和 exact 声明真实可执行的输入表示及 source
  tier；其中 residual-\(p\) 必须显式读取 `[p..L-1]` 旧 hidden suffix，不能把单个
  transition hidden 或默认 normalized capsule 当成充分输入；
- 冻结 HBM/DRAM 的 destination、dtype、layout、allocation 和 commit 边界；
- 定义每个 RQ 的候选配置、调优 role、final role、指标和结果 schema；
- 明确哪些 setup、materialization、compile、read、compute、write、commit 时间被计入；
- 预先定义 failure injection 点和 reader-visible state；
- 建立 paper table/figure 空壳与 artifact-to-claim map。

完成条件：

- 任一实验都能回答“比较谁、从哪里开始、在哪里结束、谁可以调参、结果写到哪里”；
- 新性能结果不会写入现有 resident-kernel 或 controlled-64-user protocol。

### Stage 1：开发最近邻 baseline 和统一 anchor

这是独立阶段，不能等 engine 做完后再临时拼 baseline。

外部 baseline：

- 实现 DroidSpeak-adapted contiguous-layer recomputation 的 HSTU 适配；
- 为每个候选连续区间保存一个 transition hidden state，区间内执行 current-model blocks，
  区间外复用 old K/V；
- 不复用现有 `migrate_contiguous_cache` 充当该 baseline：它在区间外执行 current
  projection，而不是复用 old K/V；先实现独立 reference，并验证区间外逐元素等于 source
  K/V、全深度区间等于 exact；
- 在独立 development users 上对每个 `m` 的所有合法连续区间做 label-free profiling；
- 扫描 `m in {2,4,6,8,12}`；
- 使用与 CohortKV 相同的 label-free fidelity view 和 measured GPU cost；
- 输出 resident-GPU frontier，并冻结进入 full-cohort job 的 certified `m`。

以上 grid 对 theta0/theta4/theta10 分别完成：每个 pair 包含 53 个 selective interval，加
p4/p8、compiled、cheap、reuse、exact，共 59 点、总计至少 177 点；source pair 之间不共享
winner，aggregate 必须校验 interval 集合完整。

端点和内部 baseline：

- stale reuse；
- exact current-model recomputation；
- cheap current projection；
- residual-\(p\)；
- no-transform placement；
- FP32 reference、packed FP16、fused FP16；
- bucketing on/off。

Residual-\(p\) 的 Stage 1 测试还必须证明 `p` 对应的旧 hidden suffix 是充分输入，并记录其
额外逻辑/物理字节。当前 verified plan 的 theta0/theta10 `p=8` fallback 需要 5.83 GiB
辅助状态；若不保留它，就在新 plan 中删除该 fallback 并直接接 exact，不能到 engine
阶段再临时重算旧 hidden。

HCache-style same-model restoration 在语义上产生错误版本，不作为实测性能 baseline；只在 related
work 中说明不适用。固定周期重算只服务 RQ1 的 policy boundary，不进入 RQ3 算法 frontier。

完成条件：

- 每条 baseline 有独立正确性测试和明确计时边界；
- selective-layer 不是用 final workload 选出来的；
- 可以生成第一张单配置 cost-fidelity frontier；
- 如果 selective-layer 支配当前 compiled 路径，暂停 Stage 2-4 的主张扩张，先决定是否
  收缩论文或设计同槽混合 program。

### Stage 2：收口 compiler 模块

复用现有 full-affine search 和 verified compiler，不在 522 final users 上继续搜索。

任务：

- 固定 attention-weighting、ridge、fit token sampling 和 action library；
- 让 compiler 从 checkpoint pair、角色 manifest 和 protocol 生成 program artifact；
- 严格验证 source/target、shape、dtype、hash 和 program path；
- 现有 certificate 使用的是内存 FP32 layerwise state；用同一冻结合同在 serialized FP16
  capsule、prepared runtime program 和 FP16 output 上重新验证一次，不重新选阈值；
- 输出完整 certificates、selection reason 和 executable fallback chain；
- 扫描 recovery target `{50,60,70,80,90}%`，但只把预先冻结的主合同用于选择；
- 测量 fit、compile、certificate 和 full-catalog score 成本；
- 给出 cohort-size amortization floor，不把 kernel-only break-even 当端到端结论；
- 验证 plan 可以被 engine 直接加载，而不依赖评测脚本内隐状态。

完成条件：

- theta0/4/10 -> theta11 都能从原始声明输入重新生成或加载可验证 plan；
- final users 只用于最后报告；
- compiler 的一次性成本、程序字节和 fallback 都进入 artifact；
- 若主 action 改变，记录真实选择，不强行保持 full affine。

### Stage 3：收口 capsule/operator 模块

任务：

- 固化 capsule 与 output extent 的版本、长度、dtype 和 layout contract；
- 对 reference、packed、fused 三路径做全 valid-element 与 padding-zero 验证；
- 记录 fused epilogue 各阶段、临时 tensor 和目标写入的 profile；
- 在完整 length 分布上重新确认 bucket width，但只使用 Stage 0 声明的 development role；
- 验证 dense direct-write 为主路径；
- 保留 jagged/page correctness 与负性能结论，不重新搜索无新瓶颈的 layout；
- 为后续 HBM direct 和 host-staged path 提供同一 operator API。

完成条件：

- operator 在代表性 shape 和完整 length 分布上数值正确；
- packed 与 fused 的比较包含相同输出 layout；
- 如果 Triton 优势不稳定，选择更简单的 library-GEMM 方案并更新论文，而不是继续微调到
  某个偶然 batch 获胜。

### Stage 4：开发 full-cohort core engine

先闭合正常路径，再增加 sentinel 和失败恢复。

任务：

- 定义 source capsule/raw-history shard manifest；
- 实现真正的 lazy shard reader，不把所有 CPU batches 先转换成 tuple；
- 对 source、wave、publication queue 和 target 分别记录峰值内存；
- 让 compiled、selective-layer、residual-\(p\) 和 exact 共用 job/extent/manifest 接口；
- 为 exact 和 selective 路径实现与 compiled 独立调优的 batching/pipeline；
- 完成 DRAM host-staged 与 HBM direct 两种 destination；
- DRAM 先探测最大 pinned extent，并检查 retained target、source wave 和 publication queue
  的 host capacity；不得在失败后静默改成 pageable 或不保留输出；
- 在每个 HBM 点计时前执行容量 preflight，把 retained target、model/program、
  maximum-batch temporary 和 allocator margin 全部算入；单卡点若在冻结 grid 的 batch 1
  下仍不可行，暂停修订协议，不得静默删点；
- 运行 1/2/4 GPU extent placement，记录每卡 bytes、time 和 imbalance；
- 记录 source scan、H2D、compute、D2H、stage、commit 和总 elapsed；
- 加入 no-transform 以测量纯 movement floor；
- 验证完整、无重复 record coverage。

完成条件：

- 682 records 从声明 source 到 committed manifest 全部跑通；
- working set 对 host-staged path 由 wave/queue 而不是 cohort 总大小决定；
- compiled、selective 和 exact 在各自相同 destination 下可公平比较；
- 得到真实 bottleneck breakdown。

这是单配置阶段的主要系统判断点。如果 source I/O、target write 或 calibration 使 compiled 在
有意义 fidelity 下没有 Pareto 点，先暂停 Stage 5-6，重新设计 source representation、
pipeline 或论文主张。

### Stage 5：接通 guard、escalation 与 failure semantics

任务：

- 先做 sentinel/preflight 的 reference 与成本设计实验；
- 只用 program-selection users，记录 reference bytes/time、正常 job overhead、theta4
  perturbation detection 和 unperturbed cohort false escalation，选定最低 overhead 的可执行
  guard，不预设名称；
- 将 verified plan 的 fallback chain 接入 engine；
- 支持 per-cohort monotone escalation；
- 在 commit 前使旧 action 产生的 extents 失效并重迁移；
- 在 manifest 中记录每个 cohort 的最终 action；
- 注入 structurally valid、重新发布后能通过 integrity/hash 检查的 theta4 语义退化程序，
  验证 semantic guard 将其升级到 published exact fallback；若运行中发现，记录并替换所有
  已生成 theta4 extents；
- 在 first extent、mid-wave、publication、pre-commit 注入异常；
- 验证 reader 只看到上一个 committed manifest；
- 测量 abort cleanup；
- 仅在语义清楚时加入 extent journal 与最多一 wave 重做的 resume。

完成条件：

- 没有失败能暴露 partial target version；
- corrupted program 不会静默提交；
- fallback 与 rework 开销被测量；
- 如果最终采用 preflight 而不是 runtime sentinel，目标稿同步更名和收缩主张。

### Stage 6：补 capsule economics 和次级 storage endpoint

这一阶段不新增论文模块。

任务：

- 测量在已经生成 fresh K/V 的 forward 中捕获 `Norm(x)` 的额外时间和写入字节；
- 报告 FP16 capsule 相对 K/V 的 logical/physical bytes；
- 实现固定的 symmetric signed INT8 storage layout：每 record、每 layer 一个 FP32 absmax
  scale，`scale=max(abs(z))/127`，全零 tensor 使用 scale 1；在 staging 中反量化到 FP16；
- 测量 INT8 对 source bytes、dequant time、operator time 和 final fidelity 的影响；
- 在 60 个 program-selection histories 上分别计时 fresh-K/V-only、加 device capture、加
  D2H/encode/POSIX persistence；用实测 capture、migration、exact、compiler amortization
  构造 `ceil(capture/(exact-compiled-compiler_amortized))`，分 FP16/INT8 报告，分母非正就
  明确写无 time break-even；
- 若目标稿仍保留 SSD 结果，在具名 NVMe、记录 mount/filesystem/cache/fsync 条件下运行
  compiled 与 exact 的相同 POSIX transaction；
- remote backend 保持 interface-only。

完成条件：

- capsule 的 standing cost 和创建成本不再是隐藏项；
- INT8 失败也作为结果，FP16 可以继续是保守默认；
- SSD 只是一种 destination 结果。若写入完全支配且没有新的系统结论，可以降到边界或附录，
  不为了保留 SSD 字样继续增加 GDS、压缩或另一个存储子系统。

### Stage 7：冻结并运行一次完整单配置论文矩阵

Stage 0-6 完成后冻结代码、配置、programs、baseline rules 和 artifact schema。

最终运行：

- RQ2 compiler/certificate/threshold/amortization；
- RQ3 selective-layer 与内部 action Pareto；
- RQ4 1/2/4 GPU HBM/DRAM full-cohort；
- RQ4 guard、escalation、failure 和 resume；
- RQ5 capture、FP16/INT8 和 break-even；
- 保留时再运行具名 SSD。

输出：

- 原始与汇总 JSON；
- protocol record；
- correctness report；
- timing breakdown；
- peak-memory/bytes report；
- paper figures/tables；
- artifact-to-claim map；
- negative-results log；
- 对目标稿每个 `TBD` 的“已测、删除、降级或仍开放”状态。

最终 workload 不能再用于调优。若整合运行发现设计错误，回到对应 Stage，建立新 protocol，
修改实现后只重跑受影响及其下游阶段。

## 6. 同槽替换与暂停规则

| 观察到的问题 | 首选处理 | 禁止做法 |
|---|---|---|
| full affine 不通过合同 | 发布 residual/selective/exact，或设计同接口混合 program | 降低 final-test 标准来保住 full affine |
| selective-layer 更强 | 收缩 dominance claim；必要时把它作为 compiler action | 使用较弱或未独立调优的 baseline |
| Triton 不稳定优于 packed | 改为 library GEMM + fused epilogue/direct layout | 只报告有利 shape |
| jagged/page 仍无收益 | 保留 dense bucket 和负结果 | 继续搜索到 final trace 偶然获胜 |
| source I/O 吞掉计算优势 | 报告瓶颈并调整 representation/source tier | 排除 source read 后继续称端到端 |
| runtime sentinel reference 不成立 | 改 preflight/canary 或只保护 artifact corruption | 保留没有真实观测量的 sentinel 叙事 |
| 4-GPU scaling 受共享 I/O 限制 | 报告 saturation point 与瓶颈 | 用 kernel scaling 代替 job scaling |
| INT8 fidelity 明显下降 | 保留 FP16，报告空间边界 | 在量化结果上改变 certificate 定义 |
| SSD 只得到共同写入上界 | 降为 destination boundary/附录 | 把跨 endpoint 时间称为算法收益 |

暂停不是失败。发现最初设计错误并尽早停止，通常比完整实现一个错误机制更有研究价值。

## 7. 完成定义与下一阶段

“单配置全链路开发完成”要求：

1. RQ2-RQ5 的单配置实验设计全部有可运行实现；
2. 最近邻 selective-layer baseline 和所有端点/内部消融已完成；
3. compiler、operator、engine 三个模块通过各自正确性和接口 gate；
4. full cohort 在至少 DRAM/HBM 上完成同 destination 的 compiled/selective/exact 比较；
5. 最终 guard 与 failure semantics 有真实 end-to-end 证据；
6. capsule 成本被完整计量；
7. 所有论文主张都有 artifact，所有不成立的预期都已从目标稿删除或降级；
8. 一次冻结后的完整集成运行完成，不混入此前的 adaptive 数字。

之后按以下顺序扩展：

1. 在当前配置的新 training seeds 上复现冻结 v1；
2. 扩到预先选择的数据集和模型容量；
3. 将扩展结果划分为 discovery 与 untouched validation；
4. 如果跨数据集暴露结构失败，设计 v2 并在未参与设计的 seed/cell 上重新验证。

多数据集结果不需要全部漂亮；它们的作用是验证适用范围和暴露边界，而不是为 v1 选择一个
有利的测试集合。
