# CohortKV 单配置全链路开发计划

> 状态：2026-07-27 起生效。本文档定义当前唯一的实施顺序。
>
> 当前阶段只在最成熟的 KuaiRand 长上下文配置上完成一次论文级 vertical slice。它是
> **开发与集成阶段**，不是新的多 seed 确认性证据。完成并冻结 v1 后，才进入新 seed、
> 多数据集和多模型容量扩展。
>
> 当前进度：Stage 0–4.6 已完成并冻结。Stage 4 的 FP16 normalized-capsule 文件源路径在
> 0/6 个 matched endpoint 上快于 exact；Stage 4.5 已改为直接从现有 HBM old K/V
> 重参数化，消除了额外 `Norm(x)`，并在 full-cohort 1/2/4 卡点均通过端到端门槛。该证据
> 原本只覆盖“exact old K/V 经过一次迁移”的固定更新；Stage 4.6 现已从 theta0 exact
> K/V 连续执行 11 次真实递归更新，并冻结均衡的 `migrate-or-exact` 生命周期。下一步是
> Stage 5 的 guard、自动 fallback、transactional rework 和 failure visibility。
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

1. FP32-arithmetic reference，作为同一份 serialized FP16 capsule/program 的 transport 数值
   oracle；它不是原始 FP32 fitted program，也不是 exact current-model K/V；
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
| RQ3 closest baseline | 单配置 selective-layer cost-fidelity frontier | compiled ladder、selective-layer、reuse/exact anchors | Pareto frontier、certificate 结果、独立调优结果 |
| RQ4 system | 682-record full-cohort、1/2/4 GPU、HBM/DRAM | compiled、frozen profiled selective diagnostic、exact、no-transform | completion、throughput、bytes、峰值内存、breakdown、commit |
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

状态：**已完成（2026-07-27）**。实现、177 点 resident frontier 和冻结决策见
[`COHORTKV_STAGE1_FRONTIER_V1.md`](../experiments/system/COHORTKV_STAGE1_FRONTIER_V1.md)
与 `configs/cohortkv_single_config_v1/stage1_frontier_summary.json`。

这是独立阶段，不能等 engine 做完后再临时拼 baseline。

外部 baseline：

- 实现 DroidSpeak-adapted contiguous-layer recomputation 的 HSTU 适配；
- 为每个 start > 0 的候选连续区间保存一个 transition hidden state；start = 0 直接从 raw
  embedding 开始。区间内执行 current-model blocks，区间外复用 old K/V；
- 不复用现有 `migrate_contiguous_cache` 充当该 baseline：它在区间外执行 current
  projection，而不是复用 old K/V；先实现独立 reference，并验证区间外逐元素等于 source
  K/V、全深度区间等于 exact；
- 在独立 development users 上对每个 `m` 的所有合法连续区间做 label-free profiling；
- 扫描 `m in {2,4,6,8,12}`；
- 使用与 CohortKV 相同的 label-free fidelity view 和 measured GPU cost；
- 输出 resident-GPU frontier；certificate 决定该 baseline 是否可发布，同时冻结一个
  与 certificate 成败无关的、只用于公平系统比较的 profiled action。

以上 grid 对 theta0/theta4/theta10 分别完成：每个 pair 包含 53 个 selective interval，加
p4/p8、compiled、cheap、reuse、exact，共 59 点、总计至少 177 点；source pair 之间不共享
winner，aggregate 必须校验 interval 集合完整。

端点和内部 baseline：

- 本阶段 resident frontier 已实测 stale reuse、exact current-model recomputation、cheap
  current projection 与 residual-\(p\)；
- no-transform placement 只在 Stage 4 的共同 destination transaction 中才有非零、可比的
  movement cost；
- packed FP16 / fused FP16 属于 Stage 3 的同输入 operator 对照；
- bucketing on/off 属于 Stage 4 的独立 runtime tuning control。

Residual-\(p\) 的 Stage 1 测试还必须证明 `p` 对应的旧 hidden suffix 是充分输入，并记录其
额外逻辑/物理字节。当前 verified plan 的 theta0/theta10 `p=8` fallback 需要 5.83 GiB
BF16 辅助状态；若不保留它，就在新 plan 中删除该 fallback 并直接接 exact，不能到 engine
阶段再临时重算旧 hidden。

HCache-style same-model restoration 在语义上产生错误版本，不作为实测性能 baseline；只在 related
work 中说明不适用。固定周期重算只服务 RQ1 的 policy boundary，不进入 RQ3 算法 frontier。

完成结果：

- 本阶段实测的 selective 与 resident anchors 已有独立正确性测试和明确计时边界；其余
  placement/operator/runtime controls 已冻结接口和所属后续阶段，不伪造 Stage 1 数值；
- 60 个 program-selection 与 60 个 certificate records 完成，522 个 final-test records
  未参与 Stage 1 评估；
- theta0/theta4/theta10 各有 59 点，53 个区间均完整，总计 177 点；
- compiled 在三个 pair 上均以更低成本和更高 worst-view recovery 严格支配全部 53 个
  selective 点；
- 没有 selective action 通过 70% 三视图合同，故可发布 action 按预案回退 exact；
- 为避免在 Stage 4 隐藏失败 baseline，仍冻结每个 pair 的最高 worst-view action
  `m=12, layers=0..11`，只作为 diagnostic external baseline 走相同 destination
  transaction。它从 layer 0 开始，因此 full-cohort shard 不需要 transition hidden；
  论文和结果必须显式标记 certificate failed，不能称为 certified/publishable action。

若后续其它配置出现 selective-layer 支配 compiled，仍需暂停主张扩张，先决定是否收缩论文
或设计同槽混合 program。

### Stage 2：收口 compiler 模块

状态：**已完成**。冻结证据见
`configs/cohortkv_single_config_v1/stage2_compiler_summary.json` 和
`experiments/system/COHORTKV_STAGE2_COMPILER_V1.md`。

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

完成结果：

- 三个 pair 均从冻结 checkpoint、source program、角色 manifest 和 protocol 生成并回读
  FP16 runtime program；三个 executable plan 均通过版本、shape、dtype、path、hash 和
  frozen-input 校验；
- 60 个 certificate users 上的 deployed certificate 全部通过，522 个 final-test users
  未执行；主 action 都是 compiled full affine；
- 主合同 fallback 为 theta0/theta10 的 `p8 -> exact` 与 theta4 的 `exact`；
- compiled 在重新加载的 serialized 表示上为 `0.01651–0.01657x` resident exact cost，
  cache recovery 为 `0.8810–0.9365`，worst recovery lower bound 为
  `0.8391–0.9231`；该 cost 不含 Stage 4 source read 和 destination publication；
- `{50,60,70,80}%` 都选择 compiled，`90%` 全部选择 exact；
- 三 pair 的 historical fit、runtime prepare 和 certificate 分别合计
  `31.243 s`、`4.316 s`、`273.343 s`；这是三个 version-pair 的 summed work，不是
  三 worker 并行 wall time。在 682 records 上 summed-work accounting 为
  `0.453 s/record`，理想并行 critical-path accounting 为 `0.153 s/record`；两者都不是
  端到端 break-even；
- 实际 shard 暴露旧 unnormalized hidden suffix 会在 FP16 大量溢出，因此 residual
  auxiliary representation 改为同为 2 bytes/element 的 BF16。默认 normalized capsule、
  compiled program 和 output K/V 仍为 FP16；逻辑目标和主 action 未改变。

### Stage 3：收口 capsule/operator 模块

状态：**已完成（2026-07-27）**。统一接口、完整 development length 分布结果和冻结决策见
[`COHORTKV_STAGE3_OPERATOR_V1.md`](../experiments/system/COHORTKV_STAGE3_OPERATOR_V1.md) 与
`configs/cohortkv_single_config_v1/stage3_operator_summary.json`。

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

完成结果：

- reference、packed、fused 现在都通过同一个 `execute_into` API，把 dense
  length-bucketed capsule 写入连续、去 padding 的 FP16 `[L,T,Dkv]` K/V extent；旧 dense
  输出只保留作兼容与 padding-zero oracle；
- 60 个 program-selection records 共 88,085 个有效 token；`batch {1,2,4} × bucket
  {16,32,64}` 九种 layout 均在全部 1,443,184,640 个有效 FP16 K/V 元素上验证，source 和
  三路径 dense padding 均为零，dense 与 extent 输出逐元素相同；
- 18 个 resident 候选按 seed-73421 完整筛选，冻结默认值为 fused FP16、batch 4、bucket
  32；三次完整分布为 `30.142/31.070/31.154 ms`；
- 最快 packed 对照为 batch 4、bucket 64，中位数 `61.970 ms`；所有 fused 样本都小于
  所有 packed 样本，中位优势 `1.995x`，因此没有触发退回 library path；相近的 fused
  finalist 之间不声明稳定排序，Stage 4 仍须重调；
- 代表性四条 2,047-token shape 上，reference/packed/fused 为
  `14.610/5.378/2.729 ms`；packed 临时峰值 402,612,224 bytes，fused 在预分配 target
  之外无 global temporary；
- jagged/page 的既有全元素正确性与负性能结论继续保留，没有重开 layout 搜索；
- 这些数值只冻结 Stage-4 compiled resident 默认值。Stage 4 仍必须对每个
  method/destination/GPU count 独立搜索完整 grid，不能把 Stage-3 结果当端到端选择。

### Stage 4：开发 full-cohort core engine

状态：**已完成（2026-07-27）**。协议、30 点结果和冻结决策见
[`COHORTKV_STAGE4_SYSTEM_V1.md`](../experiments/system/COHORTKV_STAGE4_SYSTEM_V1.md) 与
`configs/cohortkv_single_config_v1/stage4_system_summary.json`。

先闭合正常路径，再增加 sentinel 和失败恢复。

任务：

- 定义 source capsule/raw-history shard manifest；
- 实现真正的 lazy shard reader，不把所有 CPU batches 先转换成 tuple；
- 对 source、wave、publication queue 和 target 分别记录峰值内存；
- 让 compiled、冻结的 profiled selective diagnostic、residual-\(p\) 和 exact 共用
  job/extent/manifest 接口；
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
- compiled、profiled selective diagnostic 和 exact 在各自相同 destination 下可公平比较，
  且 selective 的 certificate-failed 状态不会被隐藏；
- 得到真实 bottleneck breakdown。

完成结果：

- 682 records、1,087,785 prefix tokens 的 30 点矩阵完整执行并冻结；18 个主点和 12 个
  controls 均通过完整 coverage、17,822,269,440 valid-element transport correctness、五次
  capacity preflight 和 atomic manifest commit；
- compiled 在六个 matched endpoint 上都比 certificate-failed selective diagnostic 快
  `2.70–3.49x`，但在 `0/6` 个点上快于 exact；
- HBM 1/2/4 GPU 的 compiled/exact 时间为
  `27.083/18.943/13.707 s` 与 `18.881/9.644/5.742 s`；DRAM 为
  `22.567/12.231/15.662 s` 与 `18.886/9.391/5.448 s`；
- compiled 的 source read/decode/pinning 占总时间 `91.35%–96.91%`。当前
  17,823,519,546-byte physical FP16 capsule source 吞掉了 resident operator 的计算优势；
- no-transform 在两卡后饱和，compiled DRAM 四卡也慢于两卡，确认共享 source path 是扩展
  瓶颈；
- 因此 Stage 4 正常链路完成，但 end-to-end Pareto gate 失败。按本计划的暂停规则，不直接
  推进 Stage 5。

### Stage 4.5：source/state-footprint 优化

状态：**已完成并冻结**。这是 Stage 4 负结果触发的同槽修正，不新增第四个论文模块。冻结
记录是 `configs/cohortkv_single_config_v1/stage4_5_source_plan_summary.json`。

最终没有强行保留最初的 normalized-capsule supply 设计。胜出路径利用旧模型每层 stacked
K/V projection 的满行秩性质，把已经冻结的 capsule affine 通过最小范数右逆合成为
`existing old K/V -> repaired K/V` 的直接 affine。三个 FP16 program 合计
100,777,103 bytes；额外 per-record source state 和 `Norm(x)` 都是 0。旧 K/V 是 serving
本来就有的状态，按 extent 在 replacement 被 transaction 接受后回收，不与完整 new K/V
双份共存。

该路径重新通过三视图 certificate：三个 source pair 全部选择 `compiled_old_kv`，最低
worst-view recovery 为 0.8810，最高 certificate cost 为 0.0368× exact。真实 fused
transport 覆盖全部 682 records、17,822,269,440 个 valid elements，零 tolerance
mismatch，最大绝对误差 0.01172。完整 HBM job 在 1/2/4 卡分别为
0.930/0.494/0.255 s，paired raw-history-HBM exact 为 18.695/9.729/4.766 s；五次
compiled repetition 全部快于对应点的所有 exact repetition。

这个结论严格限定为 **existing-old-K/V hot-HBM regime**。它不是 cold filesystem、SSD
或自动 tiering 的结果。性能 replay 使用等 shape/dtype/layout/occupancy 的 old-K/V
起始值，真实 old-K/V 数值路径由独立 full-transport artifact 覆盖；两种证据不得互相
替代。pinned-DRAM normalized capsule 虽能形成速度上界，但保留约 17.86 GB host state
且 preload 24.7–39.5 s，因此只保留为 backup/负 economics 结果。

#### 4.5.1 唯一硬目标

Stage 4 已经证明 compiled 的算术不是瓶颈：HBM 两卡/四卡的 compiled compute component
只有 `0.244/0.118 s`，但完整 job 为 `18.943/13.707 s`，而 matched exact 为
`9.644/5.742 s`。因此本阶段不是继续优化一个已经很短的 kernel，而是修复
**compiled 所需 source state 的表示、放置、供给和生命周期**，让算术优势能够保留到完整
job。

本阶段的首要成功条件是：

> 在相同 source-tier 假设、相同 HBM 目标布局和相同 manifest 完成边界下，compiled 的
> 完整-cohort completion time 必须稳定低于 paired exact，同时其额外状态、创建成本和峰值
> 容量全部可见。

压缩、解压、HBM 常驻、pinned-DRAM 常驻、流水化或直接从旧 K/V 变换都只是候选手段，不是
实现合同。仅把 17.82-GB capsule 预装进 HBM、然后在计时和容量报告中把这份代价隐藏起来，
不算完成。反过来，如果不压缩但通过已有状态复用、合理放置或流式回收形成了真实、容量可行
的端到端 Pareto 点，也可以通过。

#### 4.5.2 必须回答的四个问题

1. **时间去哪了**：把文件打开/反序列化、pageable-to-pinned、H2D、decode、affine、
   target allocation 和 commit 分开计时，并以 resident-source ceiling 验证 Stage 4 的
   负结果确实来自 source supply。
2. **最少需要保存什么**：比较 FP16 `Norm(x)`、量化 capsule、更小 sufficient latent，
   以及从已经存在的 old K/V 直接重参数化或恢复 compiled 输入；不能预设额外
   `Norm(x)` 一定是最终表示。
3. **状态放在哪里**：明确 hot HBM、warm pinned DRAM 和 cold exact 的适用边界。每个
   compiled/exact 比较必须给 exact 的 raw history 至少同样有利的 source tier，不能用
   HBM-resident compiled 对 filesystem exact。
4. **更新时如何共存**：计量 old K/V、source state、new K/V、模型、program、temporary
   和 allocator margin；测试 extent-wise overwrite/reclamation，避免没有必要的三份大
   状态同时驻留。

#### 4.5.3 候选方向

优先按“能最快回答问题”的顺序探索，而不是一次组合所有优化：

- **上界与定位**：FP16 capsule 的 HBM-resident ceiling，以及已经 decode/pin 的
  DRAM-resident ceiling；它们用于给出可达到的时间上界，不自动成为最终方案。
- **消除额外状态**：利用 old K/V 已经是 serving cache 的事实，测试
  `old K/V -> current K/V` 的直接 compiled 变换或可逆重参数化，并尽可能边读边覆盖旧
  extent。
- **缩小状态**：INT8/FP8 capsule、分层 scale、其他更紧凑 latent；必须同时报告
  metadata、量化/反量化时间、H2D bytes 和 semantic fidelity。
- **减少供给代价**：融合 decode/dequant 与 affine、按 GPU 预分片、NUMA/并行 reader、
  pinned buffer 复用和异步搬运。因为当前 compute 极短，单纯 overlap 只有在实际减少
  critical path 时才保留。
- **分层放置**：hot cohort 使用 HBM 或 pinned DRAM source，cold cohort 直接 exact。
  当前数据没有真实 hotness trace，所以只报告容量、驻留次数和 break-even 参数曲线，不
  声称学得了线上调度策略。

候选可以组合，但每次组合前要有单项证据说明它解决了哪个 measured bottleneck。发现首选
设计错误时，允许快速切换到逻辑目标相同的表示或执行路径，不为预写机制强行实现。

#### 4.5.4 四步开发流程

**4.5-A：建立 ceiling 和预算。**

- 在 60 个 program-selection records 上实现 matched resident-source micro/full-path
  control；
- 分别测 HBM capsule、pinned-DRAM capsule、resident old K/V 和 resident raw-history
  exact；
- 给单卡和四卡算出允许 source/decode/movement 使用的时间预算、带模型和 temporary 的
  容量预算；
- 若 resident ceiling 本身仍不胜 exact，先回到 source representation，不进入大规模
  工程优化。

**4.5-B：筛选 representation 与 supply path。**

- 每个候选登记 source representation、logical/physical/metadata bytes、创建路径、
  source tier、decode、H2D、compute、输出误差、semantic recovery 和峰值容量；
- 只使用 program-selection role 选候选，不读取 certificate/final-test labels；
- 先单候选、小规模、一 seed；只有改变 time/space Pareto frontier 的候选才保留；
- representation 冻结后，在 disjoint certificate role 上重新应用既有三视图合同。失败
  就淘汰或建立明确的新语义 protocol，不能用系统速度绕过 fidelity。

**4.5-C：完整 cohort 代表点。**

- 首轮只运行 `compiled:hbm:1` 与 `compiled:hbm:4`，并在新 protocol 下重跑 paired
  exact；两种方法从对应的同等级 resident/warm source tier 开始，结束于相同 FP16
  target extent 和 committed manifest；
- 单卡是容量压力点：new K/V 本身约 `33.2 GiB`，再常驻当前 FP16 capsule 会达到约
  `49.8 GiB`，超过 A40 容量；候选必须压缩、流式供给或按 extent 回收；
- 四卡是完整 hot-tier 可行性点：三份当前 FP16 状态约 `20.75 GiB/GPU`；
- 两点分别覆盖最紧容量边界和完整 hot-tier 多卡可行性，不在每轮加入中间卡数。

**4.5-D：冻结和一次性扩展。**

- 冻结一个最简单的胜出 source plan：representation、placement、capture/preload、
  decoder/operator、reclamation、capacity preflight 和 cold exact fallback；
- 只有该 source plan 在代表点形成稳定 Pareto 改进后，才扩到两卡、DRAM 和必要
  baseline；中间迭代不得反复跑 Stage-4 的 30 点矩阵；
- 最终只运行一次受影响的论文矩阵。Stage-4 FP16 文件源结果作为独立负 protocol 保留，
  不覆盖、不混合。

#### 4.5.5 计量与判定

每个正式候选至少报告：

- 至少五次完整 repetition、median、离散度，以及 source read/decode/pin、H2D、compute、
  allocation、commit 和 total；
- paired exact 的同边界结果；最终判定使用新 protocol 的 paired exact，Stage-4 exact
  数字只作为初始预算；
- logical/physical/metadata/source-traffic bytes，standing HBM/host bytes，峰值
  old/source/new overlap 和每卡余量；
- capture/materialization、preload、eviction 和重建成本；resident source 另报需要多少
  次更新或复用才能摊销；
- transport correctness、完整 record coverage、manifest 原子性，以及冻结后的
  certificate 结果。

性能通过不能只来自一次噪声样本：代表点的 compiled median 必须低于 paired exact，且
差值要超过两边完整 repetition 的实测波动带。主路线要求单卡和四卡都通过。若只有一个
容量明确的 hot/resident operating regime 通过，则只能冻结为**范围受限的 source
policy**：该范围外必须走 exact，论文也只能声明该范围，不能外推为全局收益。

#### 4.5.6 Stage 4.6 放行条件

Stage 4.6 只有在以下条件全部满足后才开始：

1. 至少一个明确、可部署的 source policy 在其声明 operating regime 内稳定快于
   same-boundary exact；主目标是单卡和四卡均通过；
2. 新表示没有破坏冻结的 semantic-repair 合同，或相应新合同已经独立冻结；
3. standing state、capture/preload、old/source/new overlap 和 capacity preflight
   全部进入 artifact，未把 HBM residency 当免费条件；
4. engine 获得一个冻结的 source-plan 接口，并保留范围外或容量不足时的 exact
   fallback；
5. 目标稿已按实际 operating regime 修改 claim。

若经过有界候选搜索仍没有任何 operating regime 快于 exact，这不是 Stage 4.6 的普通放行
结果：应冻结负结果、重新评估 compiled 端到端主张或方法路线，不能一边保留全局 speedup
叙事一边名义上推进后续阶段。

### Stage 4.6：连续迁移生命周期与逐 cache exact refresh

状态：**已完成并冻结（2026-07-27）**。冻结产物为
`configs/cohortkv_single_config_v1/stage4_6_lifecycle_policy.json` 和
`stage4_6_lifecycle_summary.json`；实验记录见
[`COHORTKV_STAGE4_6_LIFECYCLE_V1.md`](../experiments/system/COHORTKV_STAGE4_6_LIFECYCLE_V1.md)。
Stage 4.5 证明的是一次固定更新：

```text
exact C_v(x) -> direct migration -> approximate target C_hat_t(x)
```

它没有证明下一轮可以把 `C_hat_t(x)` 当成 exact `C_t(x)` 再送入
`t -> t+1` 的 direct program。若直接重复，上一轮误差会经过新 affine 继续传播；Stage 4.5
回收 exact old extent 后，也不能再用原始 anchor 消除这项不确定性。当前受控
`theta0/theta4/theta10 -> theta11` workload 是三个 exact-source 的一次性边，不是连续
迁移链。

本阶段把每轮更新定义为逐 cache 的二选一。每个 cache 都必须到达当前 target version，
没有“继续 stale reuse”这一第三种正常动作：

1. **lightweight migrate**：执行 Stage 4.5 的 direct-old-K/V affine；
2. **exact refresh**：用当前模型重放 raw history，产生 exact current K/V，并把该 cache 的
   累计迁移误差和连续迁移深度归零。

路由目标是控制**当前模型 K/V 语义误差和误差累积**，不是预测某个用户是否会获得推荐收益。
推荐标签、在线点击、用户级 task gain 和已经退役的 per-user drift/JVP/Fisher 路线均不得
作为 admission oracle。

#### 4.6.1 状态语义

每个 cache/extent 至少保存：

- `served_version`：当前 K/V 声明服务的模型版本；
- `last_exact_version`：最近一次 exact refresh 的版本；
- `state_kind`：`exact` 或 `migrated`；
- `migration_depth`：自最近一次 exact refresh 后的连续轻量迁移次数；
- `risk_score` 及其组成项；
- 本轮 action、输入/输出 program hash 和 lineage；
- 若采用递推误差预算，保存上一轮预算及本轮传播后的预算。

`served_version=t` 只表示状态面向 `theta_t`，不等价于 exact `C_t(x)`。只有 exact refresh
能设置 `state_kind=exact`、`last_exact_version=t` 和 `migration_depth=0`。Stage 5 的
manifest、rework 和 fallback 必须遵守这些字段，不能把 migrated output 伪装成 fresh
anchor。下一轮 migrated cache 使用的是 `served_version -> new_target` program；不得因为
它的 `last_exact_version` 更老，就假装输入仍是那个旧版本的 exact K/V。

#### 4.6.2 可部署判别量与候选 router

最初实现探索了“最大迁移深度 + 逐 cache 风险阈值”：

```text
if migration_depth >= max_migration_depth or risk_score >= risk_threshold:
    exact refresh
else:
    lightweight migrate
```

其中 `max_migration_depth` 防止任何 cache 无限连续迁移，`risk_score` 尝试让真正高风险的
cache 提前 exact。其数学依据如下。

若上一轮状态满足 `C_hat_t = C_t + e_t`，当前 direct affine 为
`T_t(y)=yB_t+c_t`，则累计误差可以拆成：

```text
C_hat_(t+1) - C_(t+1)
= [T_t(C_t) - C_(t+1)] + e_t B_t
   current one-hop residual     propagated prior error
```

这不是零误差证明，但给出了可以校准和递推的结构：exact refresh 令 `e_t=0`；轻量迁移则
同时承担本跳 residual 和上一跳误差经 `B_t` 的放大。Stage 4.6 的风险分数围绕这两个量
建立，而不是假设误差不会累积。

第一版先测 correction magnitude，再实现 fused K/V norm-ratio sketch。两者都保持小而
可解释：

```text
next_risk =
    calibrated_one_hop_error(normalized_correction_magnitude)
    + propagation_gain * previous_risk
```

其中：

- `normalized_correction_magnitude =
  ||T(KV)-KV|| / (||KV||+eps)`，是 action 前或轻量候选计算时可见的逐 cache 量；
- `calibrated_one_hop_error` 只在 40 fit records 的 exact-referenced recursive
  trajectories 上做单调分桶或 isotonic calibration，不训练复杂用户模型；
- `propagation_gain` 由当前 direct program 和 fit trajectory 校准，并保留保守上界；
- record 的最终 `risk_score` 取各层风险的高分位或最大值，具体聚合只在
  program-selection role 上选一次。

实际结果否定了把这个阈值冻结为主策略。全局 correction magnitude 与一跳误差的 Spearman
相关接近零；较强的 norm-ratio threshold 虽在 selection role 上超过 matched-random p95，
但它只优化累计成本/精度，没有单轮峰值目标。诊断 full chain 的 exact 数为
`39/3/35/209/444/1/65/53/139/431/105`，即每轮 `0.15%–65.10%`，形成不可接受的同步
refresh wave。该结果保留为负诊断，不作为冻结 policy。

最终 policy 是一个确定性的 **balanced age/deadline + edge-severity quota**：

```text
edge_severity_t = median_fit(one_hop_cache_error_q090 at theta_t -> theta_(t+1))
configured_exact_fraction_t = rank(edge_severity_t) mapped into [0.15, 0.25]
mandatory = caches with migration_depth >= 4
optional exact = oldest remaining caches, then stable SHA256 tie-break
migrate = all other caches
```

11 条边的配置 exact fraction 为
`25/19/15/17/22/16/20/18/23/24/21%`。模型变化较大的边分到更多 exact budget；
逐 cache 的年龄优先级提前消化临近 deadline 的 cache；硬深度上限可以覆盖软配额。由于
decision 在 candidate 之前产生，最终策略不会为了路由而先算后丢 migration candidate，
router GPU cost 为零，调度 CPU 时间单独报告。

这是一套在冻结 selection role 上选择、在独立 certificate 和 full chain 上验证的有界
启发式，不是数学全局最优声明。它能机械保证 deterministic action 和最大深度；`15%–25%`
峰值合同则是本配置上的实证合同，完整 cohort 因单条记录取整实际为 `14.956%–25.073%`。
若未来版本的 deadline 数超过预算，必须提高 budget 或 exact，不得突破深度上限。

当前数据没有可信 request-arrival/hotness trace，且热度本身不等于状态误差，因此不能把
构造的用户热度作为主 router 证据。将来若有真实 serving trace，热度最多用于给 cost/SLO
加权，仍不能替代 K/V fidelity risk。

selection role 只复用一条预计算 transition DAG。它比较 threshold、periodic、
fixed-quota 和 severity-bounded quota，不为候选重复完整 cohort；只有改变 Pareto frontier
或暴露新的系统约束的候选才保留。最终选择 `max_migration_depth=4`、
base exact fraction 20%、severity amplitude 5%。

#### 4.6.3 连续链实验

唯一开发配置固定为：

- KuaiRand 4+12、seed 0、16 layers、hidden/K/V width 512、history 2,048；
- `theta0 -> theta1 -> ... -> theta11`，共 12 个 checkpoint、11 次连续更新；
- 从完整 `theta0` exact K/V 开始；
- 11 跳始终使用 Stage 0 冻结的同一组 682 条 history/prefix，只改变 model version 和 cache
  state，以隔离累计迁移误差；这不是用户 history 随日期增长的 serving trace；
- 1×A40、hot-HBM source/target boundary；
- 不增加其他 seed、数据集、模型大小、GPU 数、HBM/DRAM/SSD endpoint。

在每一版本保存 exact current K/V 作为**离线评价 reference**，并执行真实的递归输入：

```text
chosen migrated records: C_hat_t -> direct(t -> t+1) -> C_hat_(t+1)
chosen exact records:     raw history -> exact theta_(t+1) -> C_(t+1)
```

全量 exact reference 不得被 router 读取，也不得计入混合策略的系统时间；策略成本只计本轮
真正选择的 exact records、migrated records、risk/router 和共同 publication。每一项 GPU
成本实测，不用手写常数。

开发与最终运行只有三层：

1. **小规模定结构**：40 fit records 生成 11 条相邻 affine、edge severity 和完整
   recursive transition DAG；60 program-selection records 共享一条 DAG，检查 threshold、
   periodic 与少量 balanced quota，不重复 full-cohort job。推荐标签不参与选择。
2. **独立证书**：冻结 policy 后，在从未参与选择的 60 certificate records 上连续运行
   11 跳；必须同时通过 fidelity、累计成本、单轮成本、exact-fraction range 和 depth 合同。
3. **一次最终链**：在完整 682-record workload 上，从 theta0 exact K/V 连续走完 11 次，
   只运行冻结的混合策略和必要的 all-exact reference。522 final-test records 提供主要推荐
   结果，其余 records 仍参与完整 cache transformation，但不反向调阈值。

每一跳的混合策略与同版本 all-exact 使用同一个 current checkpoint、同一 history、同一
engaged positive 和现有 stale-inference evaluator；唯一差别是 prefix K/V 来自混合
生命周期还是 current-model full recompute。每一跳报告：

- MeanRank、Catalog AUC、NDCG@100、Hit@100 及 paired difference；
- 在 reuse-to-exact denominator 稳定时报告 recommendation recovery；denominator 接近零
  时只报告绝对指标和 paired difference，不制造百分比；
- cache error、score cosine 和 top-100 overlap，作为状态语义诊断；
- 本跳 migrate/exact record 数、exact fraction、migration-depth 分布；
- migrate、exact、router、publication GPU 时间，本跳 `cost/all_exact`；
- 从 theta0 累积到当前版本的 cost ratio 和最终 theta11 cumulative ratio。

最终图以 cumulative `cost/all_exact` 为横轴：主选择曲线画 label-free
cache/score/top-100 fidelity，独立结果面板画每个真实推荐指标相对 all-exact 的 paired
gap/recovery。selection role 从同一条 trajectory 导出少量候选；完整 682-record 链只验证
冻结 operating point，不再跑 30 组方法/endpoint/GPU 组合。matched-random 只用于记录
threshold 的选择能力；它不能推翻 threshold 的单轮波峰负结果。

#### 4.6.4 代码与 artifact 边界

实现保持为现有 compiler/operator/engine 内部的一个小 lifecycle planner：

- `src/hstu_kvcache/migration/lifecycle.py`：定义 cache lineage、实验用 risk calibration、
  冻结的 balanced age/deadline policy 和纯函数 decision；
- `src/hstu_kvcache/migration/stage46_chain.py`：按 checkpoint 顺序推进 exact/migrated
  records，保证下一跳消费真实上一跳输出，并汇总逐版本指标与成本；
- `scripts/compile_cohortkv_stage4_6_edges.py`：生成 theta0→theta1 到 theta10→theta11 的
  11 条相邻 direct programs 和 risk calibration；
- `scripts/evaluate_cohortkv_stage4_6_lifecycle.py`：在 fit/selection roles 上复用 transition
  DAG 比较 threshold/periodic/balanced 候选并推荐 policy；
- `scripts/run_cohortkv_stage4_6_full_chain.py`：只运行一次完整 682-record 冻结策略链与
  all-exact evaluation reference；
- `scripts/freeze_cohortkv_stage4_6.py`：发布并校验
  `stage4_6_lifecycle_policy.json` 和 `stage4_6_lifecycle_summary.json`。

单元/集成测试覆盖 exact reset、depth 上限、risk threshold、balanced quota、年龄优先、
deterministic routing、
lineage/hash、上一跳真实输出复用、router 不可访问 evaluation exact tensor，以及 exact
reference 不进入 mixed-policy cost。Stage 4.6 不修改 Stage 4.5 冻结 summary；若 program
语义改变，新增 amendment/protocol。

Stage 4.5 的 direct affine 形式和 fused operator 默认保持不变。若 exact-source program
在 migrated-input distribution 上失败，可以在相同 program ABI 下对 migrated source
重新拟合或按 migration-depth bucket 发布 program，但必须建立新 protocol、重新证书，
不能把原有一跳等价性证明外推到连续链。

#### 4.6.5 完成与暂停条件

Stage 4.6 完成要求：

1. 真实迁移输出被实际作为下一跳输入，且完整 `theta0 -> theta11` 轨迹可复现；
2. 每个 cache 的 exact/migrated lineage 和 migration depth 无歧义；
3. 11 个版本跳点都有混合策略与 all-exact 的真实推荐精度、状态 fidelity 和实测 GPU
   cost 对比，没有只报最后一个有利点；
4. 冻结策略在独立 certificate records 上满足预先声明的逐跳与最终合同；
5. 目标 operating point 是约 `0.2–0.3×` cumulative all-exact cost 下获得约
   `0.8–0.9` 或更高的 label-free exact-relative fidelity，同时真实 MeanRank/AUC/NDCG/Hit
   与 all-exact 保持较小 paired gap；这是成功目标而非允许修改数据/指标的硬编码结果。
   若未达到，完整报告曲线和负结果；
6. policy 不只控制累计成本，还必须控制每轮 exact fraction 和 cost peak；逐 cache
   threshold 即使超过 matched random，也不能在形成 refresh wave 时获得主策略声明；
7. 不存在无限连续迁移，并生成只依赖 action 前信息的不可变 lifecycle-policy artifact，
   供 Stage 5 执行。

如果在合理 exact fraction 下仍无法控制累计误差，则停止连续迁移主张。可以把论文明确
收缩为“一份 exact anchor 最多迁移一次”，但不能在没有证据时继续声称支持长期 streaming
cache lifecycle。该 policy 属于现有 engine 的 maintenance planner，不新增第四个论文
模块。

完成结果：

- 60-record selection 的冻结 balanced point 为 `0.2305×` cumulative exact GPU cost，
  最差三视图 fidelity `0.9542`，每轮 exact `15%–25%`；
- 独立 60-record certificate 为 `0.2142×`，最高单轮 `0.2814×`，最差 cache
  fidelity/score cosine/top-100 overlap 为 `0.9613/0.999759/0.9898`，全部合同通过；
- 682-record 最终链为 `0.2134×`，最高单轮 `0.2543×`，最差三视图为
  `0.9632/0.999950/0.9918`；11 轮 exact 数为
  `171/130/102/116/163/171/136/123/157/164/143`；
- 7,502 条逐 cache lineage 已机械重建：每一跳消费上一跳实际输出，exact 全部归零，
  migration depth 不超过 4，program hash 与相邻版本一致；
- 522 final-test records 的逐轮 mixed-minus-exact 最大绝对差为 MeanRank `4.171`、
  Catalog AUC `8.35e-5`、NDCG@100 `3.49e-4`、Hit@100 `0.00384`。这些是单 seed
  development outcomes，不是独立统计复现；
- 调度 11 轮累计 CPU 时间约 `23.4 ms`，不计入 GPU cost ratio，且已单独披露。

### Stage 5：接通 guard、escalation 与 failure semantics

状态：**下一执行阶段**。Stage 4.6 已冻结“哪些 cache migrate、哪些 exact”，本阶段
只把该 lifecycle policy、fallback 与 transaction semantics 接入完整 engine。direct
operator 默认保持 Stage 4.5 形式；若 Stage 4.6 已按新 protocol 修改 program 语义，则以
其冻结 artifact 为准。

任务：

- 先做 sentinel/preflight 的 reference 与成本设计实验；
- 只用 program-selection users，记录 reference bytes/time、正常 job overhead、theta4
  perturbation detection 和 unperturbed cohort false escalation，选定最低 overhead 的可执行
  guard，不预设名称；
- 让 lifecycle planner 在 job 开始前把每条记录确定为 `migrate` 或 `exact`，并让两种
  action 进入同一 target transaction；
- 将 verified plan 的 fallback chain 接入 engine；
- 支持 per-record planned exact refresh，以及 program/cohort 失效时把所有受影响的
  migrated records 单调升级到 exact；
- 在 commit 前使旧 action 产生的 extents 失效并重迁移；
- 在 manifest 中记录每条记录的最终 action、lineage、last exact version 和 migration
  depth；
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

这一阶段不新增论文模块。Stage 4.5 已经为胜出 source plan 计量了 program standing
bytes、零额外 per-record state、old/new overlap 和 extent reclamation；本阶段补齐被淘汰
normalized-capsule 表示的跨表示生命周期经济学与次级 backend，而不是改变胜出主路径。

任务：

- 验证 direct-old-K/V 胜出路径不需要独立 capsule capture/encode，并测量三个 direct
  program 的一次性编译与发布成本；保留 FP16 `Norm(x)`、INT8 capsule 与 exact raw
  history 作为空间/创建经济学对照；
- 报告各表示相对 K/V 的 logical/physical/metadata bytes；
- 实现固定的 symmetric signed INT8 storage layout：每 record、每 layer 一个 FP32 absmax
  scale，`scale=max(abs(z))/127`，全零 tensor 使用 scale 1；如果 Stage 4.5 已选择其他
  表示，INT8 作为固定空间对照而不是强制替换胜出方案；
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
- 连续 `theta0 -> theta11` lifecycle、逐 cache migrate/exact router、matched-budget
  baselines 与累计误差；
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
| source I/O 吞掉计算优势 | 保持 Stage 4.5 direct-old-K/V hot policy；范围外 exact | 用 resident 结果外推 cold/SSD |
| 连续迁移误差无法被低比例 exact 控制 | 提高 refresh 比例、缩短最大 depth，或收缩为 one-hop | 把一跳证书当作链式证明 |
| 自适应 risk 不优于相同预算 random/fixed-depth | 冻结更简单的 periodic policy | 用 oracle exact error 伪装在线判别 |
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
5. Stage 4.5 冻结的 source policy 在声明 operating regime 内稳定优于 same-boundary exact，
   并有完整的时间与容量计量；
6. Stage 4.6 对 migrated-output 再迁移有真实连续链证据，并冻结逐 cache
   `migrate-or-exact` lifecycle policy；
7. 最终 guard 与 failure semantics 有真实 end-to-end 证据；
8. capsule/source-state 成本被完整计量；
9. 所有论文主张都有 artifact，所有不成立的预期都已从目标稿删除或降级；
10. 一次冻结后的完整集成运行完成，不混入此前的 adaptive 数字。

之后按以下顺序扩展：

1. 在当前配置的新 training seeds 上复现冻结 v1；
2. 扩到预先选择的数据集和模型容量；
3. 将扩展结果划分为 discovery 与 untouched validation；
4. 如果跨数据集暴露结构失败，设计 v2 并在未参与设计的 seed/cell 上重新验证。

多数据集结果不需要全部漂亮；它们的作用是验证适用范围和暴露边界，而不是为 v1 选择一个
有利的测试集合。
