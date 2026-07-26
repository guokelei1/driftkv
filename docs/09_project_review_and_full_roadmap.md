# CohortKV 项目全面评审与路线图（2026-07-26）

> 本文档是对整个项目（不只是论文）的一次全方位、发散式评审：包括创新性评判、架构批评、
> 设计上的不足与可再创新点、实验与评测标准、baseline 体系、数据与模型规模、统计协议、
> 写作与投稿策略、artifact 准备、风险与审稿攻击面、以及按优先级排列的行动路线。
> 立场：以 SOSP/OSDI/NSDI/ATC/EuroSys（CCF-A systems）审稿人的标准做尽量苛刻的自审。

---

## 0. 一句话总判断

**这不是"想法不够"的工作，是"实验还没长到想法尺寸"的工作。**
问题定义（流式训练使持久 K/V 成为 model-versioned derived state）、关键洞察
（version-pair 级共享 affine residual 可 fold 进投影，per-record 路径保持一次 GEMM）、
三层架构（compiler / operator / engine）都达到 CCF-A 选题与设计水准；
但系统证据（64 records、2 GPU、无最近邻 baseline、engine 无性能数字）目前只有
workshop~弱会强度。差距是可执行的：基础设施已备好，缺的是把 Gate 1/2/4/7 跑完。

---

## 1. 当前工作的价值盘点（哪些是真资产）

### 1.1 问题层资产
- **占位了一个真实空隙**：HCache（同模型恢复）、vLLM/CachedAttention（同模型内存管理）、
  MTServe（跨访问持久化但无版本变换）、DroidSpeak（LLM 微调变体选择性重算）都没有做
  "successive streaming versions 的编译式 K/V 迁移"。这个空隙随生成式推荐（HSTU 系）
  流行会变得越来越重要，选题有时间红利。
- **动机链条有数据支撑且诚实**：23.1% staleness tax（KuaiRand Top-50k）；三表正的
  maintenance gap；3×3 screen 证明 age/capacity 不单调；drift-utility 相关性仅 0.020 的
  负结果。"不能用 age/drift/task 做 admission oracle" 这一反直觉观察本身就是贡献。
- **语义立场清晰**：exact 是 semantic reference 而非 ranking 上界；cohort 是执行单元而非
  安全预测器。这两个立场堵住了大量审稿攻击。

### 1.2 机制层资产
- **可编译的修复面**：HSTU 每层 `K,V = P·Norm(x)+b` 暴露了 old normalized state 这个
  "半价"中间量（capsule = 50% K/V 体积），residual 在 version-pair 级结构化 →
  fold 成一个 affine program。这是本项目最核心的知识贡献。
- **27 chains 复现**：0.121× cost、0.587 recovery，多 seed、跨三表三容量——这是全项目
  统计上最硬的一块，别的都可以被打折，这块不会。
- **Label-free 证书**：cache/score/top-100 三视图 + bootstrap/Wilson 下界 + 成本预算 +
  fallback 链。方法论上完整，虽然目前只有 seed-0 adaptive 证据。

### 1.3 系统层资产
- Triton fused direct-write operator（1.19× over packed、端到端 1.176×）、
  长度分桶（863→643 rec/s 的反向消融）、两 GPU 1.951× scaling、11.22× vs tuned BF16 exact
  （边界匹配、诚实声明）。
- Destination transaction 四后端（HBM/DRAM/POSIX/remote）的正确性测试体系：
  manifest-last、duplicate rejection、abort 清理、byte-exact readback。
- 证据分级纪律（replicated / controlled / interface-validated）与 claim-evidence matrix、
  review log、open gaps 的流程化管理——这套 process 本身在 rebuttal 和 artifact evaluation
  时是加分项。

---

## 2. 当前设计/架构的不足（自我批评清单）

### 2.1 Compiler 层
1. **Verified full-affine compiler 是 seed-0 adaptive 的**。最漂亮的数字
   （0.064×、0.887–0.936 recovery）统计上会被审稿人打折。必须冻结后在新 seed 复现（Gate 1）。
2. **证书成本没有被摊销分析完整呈现**：认证要为 probe users 重算 exact K/V + 全目录打分。
   目前只承诺"报告一次性成本"，没有系统的 amortization 曲线（cohort 大小 vs 分摊成本、
   多小的 cohort 不值得编译）。这是一个必答题。
3. **Action library 偏薄**：affine / residual-p / exact 三层。中间地带还有很多可探索：
   - per-layer 混合 action（浅层 affine、深层 residual）；
   - 低秩 + 稀疏组合修复；
   - 分位置（token-position-dependent）修复——目前 attention-use 加权只影响拟合权重，
     没有位置相关的程序结构；
   - 分长度段（length-bucket-specific）程序。
4. **程序复合（program composition）未研究**：v→t1 的程序和 t1→t2 的程序能否复合成 v→t2？
   误差如何累积？这直接决定"错过若干次更新的用户"是否需要为每个 (v,t) 对单独编译。
   这可能是**架构上最值得的下一个创新点**（见 §4.1）。
5. **训练侧信息完全未利用**：编译器只看 checkpoint 前后的黑盒状态。其实训练系统手里有
   参数 delta、优化器状态、甚至梯度统计。用 ΔW 直接预测 residual 结构（例如
   residual ≈ z·ΔP + 高阶项）可以把校准样本量降到极小，甚至做到"训练结束即出程序"。
   目前 JVP/Fisher 路线被判负，但那是 per-user 路线；per-version-pair 的参数侧先验没试过。
6. **证书阈值（70/80/90/30）是手工常数**：没有敏感性分析（阈值扫过去，选择和最终质量如何变），
   审稿人会问"这些数怎么来的"。

### 2.2 Operator 层
7. **单一 kernel、单一硬件**：只有 A40。至少加一种架构（A100/H100 或 4090），
   否则 "Triton kernel 优势" 的普适性存疑。Tensor-core 利用率、roofline 分析都没有。
8. **1.19× 的 kernel margin 不厚**：packed baddbmm 已经很强。卖点应更明确地从
   "kernel 更快"转向"直接写 destination layout、消除 epilogue、地址稳定"——即
   **系统集成价值**而非纯 kernel 价值。论文已部分这样写，但评测可以更针对性
   （例如测 epilogue 各成分的时间分解）。
9. **Jagged 布局是负结果**：诚实保留了，但意味着变长问题实际靠 32-token bucket 兜底。
   若未来 cohort 长度分布更散（真实生产分布），bucket 策略需要重新搜索并给自动化方法。

### 2.3 Engine / destination 层
10. **三大贡献之一无性能数字**（最刺眼）。HBM/DRAM 有正确性、POSIX 无 SSD 实测、
    remote 无网络。Gate 2 + Gate 4 之前，这一层只能叫 "implemented architecture"。
11. **Fallback 链不会自动执行**：plan 里序列化了 fallback order，但 coordinator 不会
    在运行时自动升级。声称有 escalation 机制但从未 end-to-end 演练过，审稿人会抓。
12. **无 durable resume/恢复**：进程崩溃后不可幂等续跑。作为 "update job" 系统这是
    功能短板（可以声明范围外，但最好至少设计出 journal 方案写进 future work 并给接口）。
13. **无跨 GPU P2P 发布**：HBM 模式要求计算即目的地。多机、NVLink 拓扑感知都没有。
14. **wave/backpressure 参数手工设定**：wave size、in-flight batches=3、bucket=32 都是
    搜出来的常数，没有自适应机制，也没有跨负载的鲁棒性验证。

### 2.4 与 serving 的接口（scope 之外但审稿人必问）
15. **更新窗口期间前台请求怎么办**：读旧版本？阻塞？双版本并存的内存预算？
    manifest 切换的读侧一致性协议只字未提。至少要有一节把接口讲清楚
    （"serving reads version manifest X until Y commits"），否则系统显得悬空。
16. **capsule 的生产路径**：capsule 在什么时候由谁物化？serving 时顺带写出（几乎免费）
    还是离线批量生成（很贵）？这决定 50% 额外空间之外还有多少隐藏成本。目前评测从
    "capsule 已物化"之后开始，创建成本是黑箱。

### 2.5 数据与模型规模
17. 简化 HSTU、最大 0.181B；原版 HSTU 是万亿级生产系统。至少要展示随模型加大
    （1B 左右、更深/更宽）成本比与 recovery 的趋势线不塌。
18. KuaiRand-1K + Tenrec（QB/QK 同源）——没有第二个独立长上下文域。Gate 6 里已排除
    Taobao UserBehavior（曝光语义不符），可考虑：MovieLens-25M 拉长会话（弱）、
    Amazon Reviews 时间切片（中）、或去找带时间戳的工业公开集。哪怕加一个
    "架构外推"实验（LLM 上做同样的 cross-version affine 修复可行性 probe）也能大幅
    扩展 generality 叙事（见 §4.6）。

---

## 3. 评判标准：一篇 CCF-A systems 论文的验收清单（对照打分）

| 维度 | CCF-A 期望 | 当前状态 | 打分 |
|---|---|---|---|
| 问题新颖性与重要性 | 新问题或旧问题新压力，有真实负载证据 | 新问题 + 多表动机数据 | ★★★★★ |
| 关键洞察 | 一句话可说清、可泛化 | version-pair 级可编译残差 | ★★★★★ |
| 架构完整性 | 各层职责清晰、单元一致 | cohort 贯穿三层，清晰 | ★★★★☆ |
| 端到端规模 | 完整负载、多机/多卡、真实存储层 | 64 records / 2 GPU / 无 SSD | ★★☆☆☆ |
| Baseline 强度 | 与最近邻工作实测对比 | 无 DroidSpeak 式 baseline | ★★☆☆☆ |
| 统计与复现 | 多 seed、冻结协议 | 算法层 27 chains 硬；编译器/系统 seed-0 | ★★★☆☆ |
| 消融完整性 | 每个机制单独归因 | bucketing/LPT/persistent 有；证书阈值/权重无 | ★★★☆☆ |
| 失效与鲁棒性 | 注入失败、恢复语义 | abort 有测试；崩溃恢复无 | ★★★☆☆ |
| 开销诚实披露 | 内存/编译/摊销全量化 | capsule 50% 已披露；创建成本与摊销曲线缺 | ★★★☆☆ |
| Artifact 可评估性 | 一键复现主表 | 流程文档极好，缺打包 | ★★★★☆ |

**结论**：两个 ★★ 项（规模、baseline）就是 reject 的直接原因；把它们提到 ★★★★，
其余小项顺手补齐，就是一篇有竞争力的 ATC/EuroSys 论文；再加模型规模与存储层，
可冲 SOSP/OSDI。

---

## 4. 架构上还可以再创新的方向（发散，按含金量排序）

### 4.1 版本代数：程序复合与增量编译（最推荐的下一个大点）
把 migration program 视为版本图上的边，研究其**代数性质**：
- **复合**：`Φ(v→t2) ≈ Φ(t1→t2) ∘ Φ(v→t1)`？affine 复合仍是 affine（可直接 fold），
  关键是误差累积曲线。若复合可用，任意落后用户只需要 O(版本数) 条边而非 O(对数) 条程序。
- **增量编译**：新 checkpoint 到达时，用上一对的程序热启动拟合（warm-start ridge），
  把编译时间从"每对从头"降为"每步增量"。配合训练侧 ΔW 先验（§2.1-5）可能把
  校准样本降到个位数用户。
- **版本图调度**：多个 stale 源共存时，是逐边迁移汇聚到 t，还是各自直达 t？
  这变成一个带误差约束的最短路/生成树问题——非常"系统论文"的问题形状。
这一组能把论文从"一个 pair 的编译"升级为"版本流上的持续维护系统"，
和 streaming 叙事完全咬合，而且大部分靠现有基础设施就能做。

### 4.2 更新-服务共同设计（把 scope 边界往前推半步）
- 双版本 manifest 的读侧协议：serving 固定读 committed manifest，更新 job 在旁路进行，
  commit 即原子切换；给出切换期间的内存峰值模型（old K/V + new K/V + capsule 的
  同驻峰值）与分波释放策略。
- **在线优先级**：按用户下次访问概率排迁移顺序（这不是 safety prediction，只是调度），
  可以用 LPT 的同一框架做 "expected-hit-weighted" placement——一个很自然的增量实验。
- SLO 共存实验：迁移 job 与前台推理同卡运行，用 MPS/MIG 或 stream priority 隔离，
  报告前台 P99 退化 vs 迁移吞吐的 trade-off 曲线。这是 NSDI 口味的实验。

### 4.3 Capsule 压缩与存储协同（把 50% 开销变成卖点）
- INT8/FP8 capsule + per-layer scale：测 recovery 损失曲线；若 INT8 基本无损，
  capsule 降到 25% K/V，叙事从"空间换时间"变成"低成本影子状态"。
- 低秩 capsule：`Norm(x)` 的谱衰减多快？存 rank-r 投影可否再省一半？
- 与编译器联动：拟合时直接在压缩域进行（compile-for-compressed），程序吸收反量化。
- 存储布局：capsule 按 cohort 分组顺序写，为 Gate 2 的 lazy scan 准备顺序读带宽。

### 4.4 自动 fallback 与运行时验证（把证书从离线搬到在线）
- 运行时轻量哨兵：对每个 wave 抽样 k 条记录算 cheap 侧证书指标（cache error 即可，
  无需打分），超阈值自动升级到 residual-p / exact 并记录到 manifest。
- 这补齐 §2.3-11 的短板，同时创造一个新叙事："certified compilation + runtime guard"，
  和数据库的 plan + adaptive re-optimization 类比，审稿人会喜欢这个类比。

### 4.5 多租户与资源治理
- 多个 (v,t) job 并发、共享 GPU 池：程序驻留表的置换策略、带宽配额、
  cohort 间公平性。目前是单 job 世界观；哪怕给出设计草案也能显著加厚 "engine" 层。

### 4.6 架构外推：脱离 HSTU 的普适性 probe
- 关键洞察依赖 "Norm(x) → P" 的数据路径，而这在标准 Transformer 里同样成立
  （pre-LN 结构的 K/V 投影）。做一个小规模 LLM continual-pretrain 的 cross-version
  affine 修复 probe（哪怕 125M 模型、只测 cache recovery），能把贡献从
  "推荐系统机制"抬到 "版本化 K/V 的一般方法"，是 rebuttal 的战略储备。
- 反方向也要测：post-LN / 无 bias / GQA 变体下修复面是否仍存在。

### 4.7 与训练系统的接口（Ekko 补集）
- 定义 checkpoint 发布时训练侧应携带的元数据（版本号、层签名、ΔW 范数摘要），
  让编译器可以零通信地决定 rank/正则/是否需要更强 action。
- "训练结束 N 秒内程序就绪" 可以成为一个新的端到端指标（update-to-ready latency），
  与 Ekko 的 update latency 叙事形成呼应但不重叠。

---

## 5. 实验路线图（具体配置级）

### 5.1 P0 —— 决定生死的三个实验

**E1（Gate 2）Full-cohort identical-boundary destination benchmark**
- 负载：KuaiRand 4+12 全部合格记录（全 cohort，theta0/4/10 → theta11 组织版本），
  lazy capsule shard reader（顺序读，报告读带宽）。
- 两路径同事务发布：compiled（selected action + 强制触发至少一次自动升级）vs
  独立调优 exact；目的地各测 HBM 和 pinned-DRAM。
- 规模轴：1/2/4 GPU；报告 completion、tokens/s、物理/逻辑字节、峰值
  source/HBM/staging/target 内存、backpressure 事件数、程序摊销、manifest commit 时间。
- 失效注入：首 extent 前、wave 中、publication 中、commit 前各一次，验证不可见性与清理。
- 通过标准：4 GPU 线性度 ≥85%；compiled/exact 端到端加速比在全 cohort 上不塌
  （目标 ≥8×）；commit 时间 < 总时间 5%。

**E2（Gate 7）最近邻 baseline：selective-layer recomputation（DroidSpeak 式）**
- 实现：对每层计算 cross-version K/V 距离，选 top-m 层用当前模型重算、其余复用旧值；
  m 扫 {2,4,6,8,12}；同样走 label-free 证书与同一发布边界。
- 附加两条陪跑 baseline：no-transform placement（纯搬运旧 K/V，量化"迁移 vs 只移动"）
  与 HCache 式 same-model restore（说明为什么同模型恢复不解决版本问题——预期它语义不达标，
  这是设计好的失败展示）。
- 报表：cost/exact vs semantic recovery 的 Pareto 前沿图，CohortKV 各 action 与
  selective-layer 各 m 同图。**这张图很可能成为论文的招牌图。**

**E3（Gate 1）冻结编译器多 seed 复现**
- 冻结全部超参与 70/80/90/30 契约，写 hash 进协议文件；在 ≥2 个新 training seed
  （或接受的外部 checkpoint）上跑完整 4+12 流程；报告证书通过率、选中 action 家族、
  最终 recovery 分布、编译墙钟与摊销。
- 顺带做证书阈值敏感性：recovery target ∈ {50,60,70,80,90}%，画选择切换图（回应 §2.1-6）。

### 5.2 P1 —— 显著加分

**E4（Gate 4）物理 SSD POSIX**：具名 NVMe，报告序列化字节、fsync 策略、写放大、
  带宽利用、compiled vs exact 同端点完成时间；顺带 MTServe 式 page-chunk 摆放对照。
**E5（Gate 5）Capsule 经济学**：创建成本（serving 顺带 vs 离线批量）、FP16/INT8/低秩
  三种布局的空间-recovery 曲线、更新频率 break-even 点（每天 N 次更新时 capsule 是否回本）。
**E6 硬件泛化**：A100 或 4090 上复跑 operator microbenchmark 与 E1 的 1-GPU 点。
**E7 模型规模趋势**：0.18B → ~0.5B → ~1B（层/宽扩展），只测 cost ratio 与 cache recovery
  趋势线（不必全任务指标），证明洞察不随规模消失。
**E8 SLO 共存**（§4.2）：迁移与前台推理同卡，P99 退化 vs 迁移吞吐曲线。

### 5.3 P2 —— 战略储备 / 下一篇
- E9 程序复合与增量编译（§4.1）——可能独立成章甚至下一篇论文的核心。
- E10 LLM 架构外推 probe（§4.6）。
- E11 多 job 并发资源治理（§4.5）。
- E12 运行时哨兵 + 自动升级的在线验证（§4.4，其最小版并入 E1 的"强制升级"要求）。

### 5.4 统一评测规范（所有新实验必须遵守）
- 冻结协议先于运行；训练 seed 为统计单元；同一 boundary 语句写进结果 JSON 的 metadata。
- 每个加速比必须注明 baseline、endpoint、residency；HBM-vs-host 差异永不称为 operator 加速。
- 每个近似结果携带三视图证书值；exact 永远称 semantic reference。
- 负结果照常入库（drift、jagged 的先例保持）。
- 结果落盘 → 更新 claim matrix → 改 Results → 改 Discussion → 最后才动 Intro/Abstract
  （维持现有 results-to-paper 协议）。

---

## 6. 论文本身的进一步改进（manuscript 层面）

### 6.1 结构与叙事
1. **图太少（4 张）且全是架构/流程图，没有一张数据图**。顶会系统论文通常 8–12 张图。
   急需：① E2 的 cost-fidelity Pareto 图（招牌）；② 27-chain 的 cost/recovery 散点
   （按表/容量着色）；③ 1/2/4-GPU scaling 柱状；④ 长度分桶消融；⑤ age vs drift vs
   task-utility 的"不可校准性"图（把 §3.2 表格可视化，动机会强很多）；
   ⑥ capsule 空间-精度曲线（E5 后）。把现有 Table 3/6/9 的一部分转成图。
2. **Abstract 略超 249 词**，投稿前按 venue 限制再压一遍。
3. **§9.2 依赖回引 §3 的表**：评测章自证性弱。E1/E2 落地后，把 §3 压缩为纯动机、
  §9 全部用新的 full-cohort 数据自立。
4. 考虑给 §5 加一个 2–3 行的编译器伪代码/流程框（compile(v,t) → program artifact），
  系统读者喜欢有个可指认的 API。
5. Related work 的 DroidSpeak 段在 E2 完成后要改写成实测对比而非声明性区分。
6. 中文版 `manuscript_v2_zh.md` 已落后于英文版，投稿前决定是否维护双语。

### 6.2 术语与一致性守则（持续执行）
- 六个受控术语（migration anchor / served K/V target / version cohort / compiled repair /
  exact recomputation / destination manifest）在新章节中禁止同义替换。
- 全部新增数字必须带方向词（only / as high as / substantial）。
- "exact" 只作语义参照；正确性词汇固定为 element-for-element / byte-exact / lossless。

---

## 7. 投稿与备赛策略

### 7.1 档位与顺序
- **主目标**：E1+E2+E3 完成 → **ATC 或 EuroSys**（CCF-A，胜算实在）。
- **冲刺目标**：再加 E4+E7 → SOSP/OSDI（需要存储层与规模趋势撑住"大系统"观感）。
- **备选改叙事**：FAST（存储/事务角度，E4 为主）或 VLDB（版本化派生状态管理角度）。
  MLSys 亦可但非 CCF-A。
- **时间线策略**：以最近的 ATC/EuroSys 截稿倒排 E1→E2→E3；E4/E7 视窗口决定进主文
  还是 rebuttal 弹药。

### 7.2 预演审稿（rebuttal 弹药库，提前写好答案）
1. "为什么不直接重算？GPU 很便宜" → 用 E1 的全 cohort 字节/时间账 + 更新频率放大
  （每天 N 次 × 用户量）回答；补充 update-to-ready latency 叙事。
2. "DroidSpeak 已经做了 cross-model reuse" → E2 Pareto 图 + workload 差异表（Table 10）。
3. "50% capsule 空间不可接受" → E5 的 INT8/低秩曲线 + break-even 分析。
4. "模型太小" → E7 趋势线 + 明确声明机制依赖的是数据路径结构而非规模。
5. "单 seed" → E3 冻结复现 + 现有 27-chain。
6. "近似 K/V 影响业务指标怎么办" → 证书三视图 + signed recovery + exact 非上界的
  age-7 实证 + 运行时哨兵（E12 最小版）。
7. "任务收益有时为负，系统还有意义吗" → 立场重申：系统交付的是 current-model 语义
  一致性（部署纪律问题），不是排名收益预测；这正是 label-free 契约存在的原因。

### 7.3 Artifact Evaluation 准备
- 一键脚本复现主表（Table 6/7/9 + 未来 E1/E2 表），固定 seed 与版本锁。
- 提供小型化 "kick-the-tires" 配置（单 GPU、子采样 cohort、<30 分钟）。
- checkpoint 与数据的再分发许可核查（KuaiRand/Tenrec 条款）；不能分发的给再训练脚本。
- claim-evidence matrix 直接随 artifact 发布——这是本项目独有的加分项。

---

## 8. 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| E1 全 cohort 下加速比明显缩水（I/O 变主导） | 中 | 高 | 提前测 capsule 顺序读带宽；叙事备份：报告 compute-bound 与 I/O-bound 两个 regime，本身就是有价值的系统结论 |
| E2 selective-layer baseline 意外地好 | 中 | 高 | 若其 Pareto 点接近，强调 affine 的正交性与可组合（affine+selective 混合 action 反而成为新贡献）；提前实现混合 action 以攻为守 |
| E3 新 seed 上证书不通过/选中更贵 action | 低-中 | 中 | 契约本就允许升级——失败即展示 fallback 机制工作；但若普遍失败需回滚"cheap 优先"叙事 |
| 4-GPU 硬件/时间不可得 | 中 | 中 | 至少 2-GPU 全 cohort + 模拟器外推声明；或借用集群窗口 |
| 数据集扩展受阻（无合格新域） | 高 | 中 | 用 §4.6 架构外推 probe 补 generality；文中维持诚实边界声明 |
| 时间预算不足以全做 | 高 | — | 严格按 P0→P1 顺序；P0 三项做完即可投 ATC/EuroSys，不贪 |

---

## 9. 行动清单（按序执行）

1. [P0] E1 full-cohort destination benchmark（含自动升级演练 + 失效注入）
2. [P0] E2 selective-layer / no-transform / same-model-restore 三 baseline + Pareto 图
3. [P0] E3 冻结编译器多 seed 复现 + 阈值敏感性
4. [P1] E5 capsule 经济学（创建成本、INT8/低秩、break-even）
5. [P1] E4 具名 SSD POSIX 性能
6. [P1] E6 第二种 GPU、E7 模型规模趋势
7. [P1] 论文数据图重做（§6.1-1 的六张图）、§3/§9 职责重分、DroidSpeak 段改写
8. [P2] E8 SLO 共存、E12 运行时哨兵
9. [P2] E9 程序复合/增量编译（下一个大创新点，视进度决定进本文还是下一篇）
10. [持续] claim matrix / review log / open gaps 与每个新结果同步；中文稿同步决策

---

## 10. 附录：capsule 存储开销的发散设计空间（2026-07-26 增补）

> 背景：审稿人对本文最可预期的攻击是"你多存了 50% 的 Norm(x)，凭什么说这是划算的"。
> 本节发散所有能减少/消解这笔开销的方向，并给出进论文的建议。§4.3 是早期雏形，本节取代之。

### 10.1 统一性洞察：线性编解码器可折叠进编译程序（本节最重要的一条）

在线路径是单次仿射映射 `y = z·P̂ + b̂`。因此**任何对 z 的线性存储变换都能在编译期折叠进程序，
每条记录的在线成本为零**：

- 存 `w = z·C`（编码），执行 `y = w·(C⁺P̂) + b̂` —— 解码矩阵直接乘进程序权重；
- 这统一了三类看似不同的技术：
  1. **INT8/FP8 per-channel 量化**：反量化 scale 是对角线性变换 → 折叠；
  2. **cohort 共享低秩投影**：per-(v,layer) 在校准集上拟合 PCA/SVD 基，记录只存 r 维系数
     → 折叠，且 GEMM 的 K 维从 H 降到 r，**存储和计算同时下降**；
  3. **零存储模式（高风险高回报）**：K_v = z_v·Wk_v 本身就是 z 的线性编码。若 Wk_v 可
     （伪）逆，则 z_v ≈ K_v·Wk_v⁺，`y = K_v·(Wk_v⁺·P̂) + b̂` —— **不存 capsule，直接从
     已驻留的旧 K/V 迁移**，仍是一次 GEMM。保真度取决于 Wk 的条件数（D_kv<H 时必有损），
     但现成的证书恰好能逐 cohort 判定该模式是否可发布。
- 这把"多存 50%"的弱点反转为编译思想的自然延伸：**capsule codec 是编译的一部分，
  证书认证的是 codec∘program 的复合体**。叙事从"空间换时间"升级为
  "存储格式是编译器的一个自由度"。

### 10.2 候选技术清单（按可行性排序）

| # | 技术 | 存储 | 在线成本 | 风险 | 备注 |
|---|---|---|---|---|---|
| 1 | INT8 per-channel（staging 反量化） | 25% K/V | 0（已收敛进 target E5） | 低 | 已在 target 中 |
| 2 | INT8 scale 折叠进程序（免反量化） | 25% | 0，且省 staging 转换 | 低 | 10.1-①，几行矩阵代数 |
| 3 | cohort 共享低秩 r=H/2、H/4 | 25%→12.5% | **为负**（GEMM 变小） | 中 | 需测 Norm(x) 谱衰减；证书把关 |
| 4 | 2+3 组合（低秩+INT8） | ~6% K/V | 为负 | 中 | "capsule ≈ 元数据"级别 |
| 5 | 无损压缩（zstd/lz4）只用于 SSD/DRAM 层 | 视熵而定 | CPU 解压 | 低 | **与流水线天然重叠**：E1 预期 full-cohort 是 I/O-bound → 压缩=有效源带宽倍增器，可能反而加速端到端 |
| 6 | 长度阈值策略：短记录不存 capsule，直接 exact 重放 | 按分布 | 0 | 低 | break-even 对长度参数化即得，workload-free |
| 7 | 升级即刷新：cohort 被迫 exact 重算时顺带重写 capsule 锚点 | 0 增量 | 0（顺带） | 低 | 免费的锚点新鲜度维护 |
| 8 | 注意力分层精度：高注意力 token FP16、低注意力 INT4 | ~15-20% | 0（折叠） | 中高 | 复用 §4.1 的 attention-use 统计；实现较繁 |
| 9 | 零存储 K-逆模式 | **0%** | 0 | 高 | 10.1-③；D_kv=H 的配置最有戏；哪怕只在部分 cohort 通过证书也是好结果 |

### 10.3 进论文的建议（2026-07-26 定调：base + 扩展分层）

**Base 方案（进论文的主体，solid、无保真度风险）**：
- **压缩 capsule 存储层 + 流水线重叠**：INT8 per-channel（2×，已在 E5 target）
  叠加无损压缩（zstd/lz4）作为 SSD/DRAM 层的存储格式；解压与迁移波
  （wave pipeline）重叠，目标是在 I/O-bound regime 下把压缩变成
  **有效源带宽倍增器**——省字节的同时可能反而加速端到端。
- **CPU vs GPU 解压是测量项而非先验决定**：
  - CPU 解压（staging 阶段，lz4 单核 ~3-4 GB/s）：不占 GPU SM，但 PCIe H2D
    传的是解压后字节，只省 SSD→DRAM 段带宽；
  - GPU 解压（nvCOMP LZ4/GDeflate）：压缩字节直接 H2D，**PCIe 有效带宽倍增**，
    但解压 kernel 与迁移 GEMM 抢 SM，需 stream 重叠测干扰。
  - 预判：SSD 源 → CPU 解压赢；DRAM 源且 H2D 瓶颈 → GPU 解压赢。
    两个都测，"解压位置 vs 瓶颈段" 本身就是一张好图。
- 优点：技术成熟、无损（无需新证书理论）、workload-free、直接打在 E1
  预期的 I/O-bound regime 上；不会给审稿人"堆砌"观感。

**扩展方案（降级为可选，不一定进论文；可行性存疑且有堆砌风险）**：
- #2 INT8 scale 折叠进程序、#3 cohort 共享低秩（"compiled capsule codec"
  作 §4.5）：理论上是 Design 1 的自然深化，但会增加叙事复杂度；
  仅当 base 方案数据出来后仍需更强空间故事时再考虑。
- #9 零存储 K-逆模式：高风险彩蛋。若顺手测得任一 cohort 通过证书可写一句
  "capsule-free migration"；不专门投入。
- **不做**：#8（实现繁、增益与 #4 重叠）；任何依赖请求热度的分层放置（serving 边界）。

### 10.4 对应实验增量（按 base/扩展分层）

**Base（并入 E5 / E1）**：
- E5d（升级为主实验）：INT8+zstd/lz4 压缩比实测；CPU 解压 vs GPU 解压
  （nvCOMP）两条路径的端到端方向，在 E1 的 SSD 源与 DRAM 源两个配置下各测；
  报告压缩比、解压吞吐、SM 干扰（与迁移 GEMM 同卡时的吞吐退化）、
  端到端完成时间变化方向。

**扩展（可选，视 base 结果决定）**：
- E5a：Norm(x) 逐层谱衰减曲线（一次离线 SVD，半天）——决定 #3 的 r；
- E5b：codec 折叠正确性 + 复合证书（复用现有证书代码）；
- E5c：低秩/INT8/组合三条 recovery-vs-bytes 曲线（Figure 8 扩展）;
- E5e：#9 在 D_kv=H 配置上的条件数与证书通过率。


---

*本文档为一次性全景评审快照；后续每完成一个 P0/P1 项，应在 review log 记录并回改本文档相应条目的状态。*
