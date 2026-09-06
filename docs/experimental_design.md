# EvoKV 具体实验设计

更新日期：2026-09-06

状态：**Medium KV-only discovery 已完成；Cross-Version Cache Adaptation 尚为 prospective；
六层完整原型计划已按用户澄清完成 review，实现、Translator calibration 和方法实验均尚未运行。**

本文记录 [论文总体设计](paper_design.md)对应的实验方案草稿，不是已封存的运行合同。
本阶段的开发顺序、当前决定与 review 结论以 [design/plan.md](design/plan.md) 为入口，
探索经过集中记录在 [design/iterations.md](design/iterations.md)。先把四组件接成六层完整 v0，
再联动迭代；本文的 S0–S5 是研究诊断维度，不是必须先后达标的阶段。
现有 V0–V5 backbone 保持冻结。单边时共享 Translator 为 edge-specific；连续路径每个目标版本
共享一个支持实际 producer 的 Translator。用户已于 2026-09-06 撤销笼统 target-KV fitting 禁令，
摘要/KV 衍生监督与小规模共享校准进入本阶段开发范围。下文保留单边公式及多版本衔接；
旧单边合同只解释原实验，不作为新方法开发的禁令，也不因本轮澄清改写。

本文不修改任何 sealed motivation contract、checkpoint、data hash、release window、seed、workload、
metric、raw result 或 adjudication。当前论文证据、Medium/Large 模型及其必要过程记录保持原路径；
废弃 Small 和旧探索的原始结果已删除，旧结论与失败说明压缩归档，见
[结果索引](../results/README.md)。
本文沿用先前草稿的状态语义、诊断名称和成本定义；实现与检查按当前研究问题分步展开，
不把完整生命周期工程作为首个想法实验的前置条件。本文没有新建实验合同或启动运行。
Design 正文按组件与流水线叙述；本文件保留复现所需的技术定义和评价检查。
这些检查用于确认接口按定义工作、评价近似方法效果，不要求各模块与 Current Full 逐元素等价。

### 研究迭代方式

本仓库以论文想法的快速验证为主。先用六层 HSTU-native、内存中、顺序执行原型接通摘要、
共享翻译、paired read 与持续维护；真实摘要参考和 learned 方法可以同轮测量，不等某个模块完善
才接下一个。用最小样本检查关键数值和联动问题，再迭代各部分。不先做通用框架、事务执行器、
异常恢复或完整序列化兼容。检查聚焦当前会影响结果的公式、时间因果、数据划分与指标口径。

小规模探索记录假设、必要配置、结果和下一步即可，更新现有实验记录，不为每次改动另建合同
或执行全量测试。已有的数值对照或小样本运行能覆盖风险时，直接作为相应检查/canary。
范围内的小规模共享校准记录监督来源、UID 划分和资源，不为每个 loss/摘要变更设审批；
长训练、正式人口评价和未开放数据仍遵循各自协议与授权边界。
工程功能和边界用例只在实际运行或论文论点依赖它们时补充。

## 1. 固定资产与范围

### 1.1 数据与人口

| 项目 | 当前设置 |
| --- | --- |
| 数据 | Yambda-500M unified Medium |
| 固定训练人口 | 30,000 users |
| 已封存 Insight population | 3,000 label-free users，跨五 edge 固定 |
| full-history eligibility | 首个 cutover 前至少 1,024 events |
| eligible population | 21,200 users |
| 历史长度 | 1,024 |
| candidate panel | 每 user-edge 64 candidates，来自 pre-cutover history/global bank |
| item vocabulary | 1,380,509 known + 256 stable OOV buckets |

现有 3,000-user population 继续使用封存 UID 顺序和 \([5,3000,64]\) candidate panel。扩大到
15,000/30,000 只用于方法冻结后的 scale/qualification，不得用来重新选择配置。

### 1.2 六个冻结模型

使用 seed 17、HSTU-native CC、6 layers、hidden 192、6 heads、context 1,024、约 266.3M 参数的
D14 direct-parent chain：

| Version | training window | adjacent cutover |
| --- | --- | --- |
| V0 | [0,217) | — |
| V1 | [217,231) | day 231 |
| V2 | [231,245) | day 245 |
| V3 | [245,259) | day 259 |
| V4 | [259,273) | day 273 |
| V5 | [273,287) | day 287 |

观察边为 V0→V1、V1→V2、V2→V3、V3→V4、V4→V5。每条 edge 的 Translator 在两版
checkpoint Full-only admission 后训练；backbone 参数和 readout 均不更新。

这些模型只训练了 explicit-feedback candidate-conditioned binary ranking。现阶段不能把其结果外推为
next-item/retrieval。RecFlow、theta3 和新 backbone 不进入本 Design 1 执行范围。

### 1.3 单边 reference paths

每条 edge、同一历史和 query 构造：

1. Parent Exact：Parent reader + Parent-produced prefix；
2. Current Exact：Current reader + Current-produced prefix；
3. Current Reuse：Current reader + Parent-produced prefix。

compatibility target 是 Current Exact − Current Reuse。模型 admission 先于 Reuse/Design evaluation；
近零或反号 Full–Reuse gap 按预注册 No-op 规则处理。

## 2. 既有 Insight protocol

### 2.1 Stage taxonomy

| Stage | Transformer term | HSTU adapter |
| --- | --- | --- |
| S0 | input representation | item/action/time input；query token |
| S1 | normalized input and Q/K/V | producer K/V；reader Q；transient self K/V |
| S2 | query–key interaction | raw/activated qK |
| S3 | position value contribution | activated qK × V，尚未 history sum |
| S4 | raw context aggregate | position-summed per-head historical response |
| S5 | transformed update | output projection、gate/FFN update |
| S6 | residual state | residual + update |
| S7 | final representation | final norm 前后 hidden |
| S8 | readout | CC logit/probability |

Read-time correction 的 HSTU 主注入点是 S4：聚合后 normalization 之前的可加 historical
aggregate；合并历史与模型规定的暂态贡献后，按原模型顺序执行 normalization、gate、output
projection 和 residual。S2/S3 仍是诊断 tensor，不自动成为 persistent action。

### 2.2 既有用户/candidate 防泄漏

| Split | selector-order indices | 既有用途 |
| --- | --- | --- |
| focused canary | [0,32) | instrumentation 与资源 |
| discovery | [0,512) | Insight stage/representation development |
| confirmation | [512,3000) | 配置冻结后一次读取 |

每个 64-candidate panel 的偶数 32 个作为 anchors，奇数 32 个作为 held-out。现有 confirmation users
在新 sketch schema、Translator、loss 和成本冻结前继续 unread。

Translator calibration 从整个 3,000 Insight cohort 之外选择 UID，开发开始时记录划分，正式评价
前再冻结最终协议。最终未见用户确认与 fitting 分离；逐用户拟合诊断单列，不混作未见泛化。
上述 confirmation 禁读边界继续保留，但 3,000 用户已参与 Insight 1，不能称为研究中全未见。
新方法的独立确认用户预留与接触范围按 [当前计划](design/plan.md)明确，不能看结果后换划分。

### 2.3 已完成证据

- Motivation、Full/Reuse 与 Medium locality 已封存；
- S4 shared/low-rank oracle 显示 functional contraction；
- fixed correction 的持续性原始证据仍保留；tail、sparse carriers、paired replay、release-algebra
  等退休探索的正负结论保存在归档中，不再表示对应原始结果和实现均可直接使用；
- ordinary KV、Full/Reuse、response instrumentation 和 diagnostic delta injection 已有代码。

这些只决定为什么进入新的 state interface，不是 EvoKV 方法结果。

## 3. Prospective state 与 Translator

### 3.1 Source summary

参考 schema 每段最多 64 个事件、每段每层两个固定事件区间 slots。边界对齐的 1,024-event
窗口有 16 段、每层 32 slots；滑动边界与 release 提前封段的额外容量需单列：

\[
S_g^p=W(C_g^p;m_g).
\]

首版 writer \(W\) 在同一次运行的 release family 内固定、可加减；开发可联改 schema，
序列化按后续实际需要实现。Source 保存实际持久化 K/V 的累加量、mass/count、
mask、time/position；读取和翻译用均值载荷，每个 segment 保存 source version、schema、
generation 和边界 metadata。assignment 随事件固定，不能随窗口移动重新分组。
Source 只需持久化累加量，翻译和读取时计算均值，避免重复保存均值副本。累加与扣除使用同一份
实际持久化 K/V 数值；累加精度、空 slot 和删除后的数值误差纳入状态检查及存储计量。

Translator 只预测表示 payload：

\[
\widehat X_u^c=X_u^p+T_{p\rightarrow c}(X_u^p,m_u),
\qquad \widehat S_u^c=(\widehat X_u^c,m_u).
\]

count、mask、time/position 不由 Translator 猜测。参考 Translator 输入该用户存活 Parent
摘要的全部 segments 与各层载荷，允许跨段交互；whole/partial eviction 后刷新全部剩余 Parent
翻译结果。segment-local 只能作为输入受限的独立变体，不能默认具有相同可预测性。
混合来源时这里的 Parent 扩展为当前目标以外的所有活跃 producer。revision 跟踪本目标译者的
实际 source 输入；只追加未进入该输入的 current 段不会触发旧段重翻译。稳定事件序号与时间
保存在 source，窗口坐标按请求计算；旧段内容变动后，在下一次修正读取之前完成同步刷新。

### 3.2 Paired HSTU read

\[
\widetilde A_\ell^c(q_\ell)
=
A_\ell^{\mathrm{base}}(q_\ell)
+
\sum_g
\left[
R_{c,\ell}(q_\ell,\widehat S_g^c)
-
R_{c,\ell}(q_\ell,S_g^p)
\right].
\]

base 使用实际修正轨迹的同层 query 读取存储缓存，不从另一次 Reuse 前向搬运张量。
参考 slot read 为 \(n_j\phi(\gamma q^\top\bar k_j+\bar b(q,j))\bar v_j\) 乘原模型统一缩放；
不能按 slot 数重新归一化。验证单事件 slot 的 segmented reference、mask 和代表位置语义。
代表时间采用存活事件平均时间，代表位置采用平均事件序号向下取整后对应的当前窗口位置，
再使用原模型的偏置索引或桶化规则；checkpoint 未启用的偏置项保持关闭。
批量追加必须提供逐前缀摘要或按事件顺序推进，暂态候选不入摘要。若实际路径存在跨 segment
normalization，先改为可组合的接口，不能让 Translator 吞掉代数错误。

softmax attention 不使用上述 normalized-output 求和；它需要组合 numerator/normalizer \((N,Z)\) 后再
统一归一化。本轮完整实现只验证 HSTU-native 路径。

### 3.3 Translator objectives

backbone 冻结。以下是首版共享校准的初始目标，开发可按闭环收益调整；
calibration users 可产生 Parent/Current Exact summary 和 response teacher：

\[
\mathcal L_{\mathrm{sketch}}
=
\sum_{\ell,g}
\left\|
D_\ell^{-1}(\widehat X_g^c-X_g^c)
\right\|_2^2,
\]

\[
\Delta A_\ell^*(q)
=
A_{c,\ell}(q;C_\ell^c)
-
A_{c,\ell}(q;C_\ell^p),
\]

\[
\widehat{\Delta A}_\ell(q)
=
R_{c,\ell}(q,\widehat S_\ell^c)
-
R_{c,\ell}(q,S_\ell^p),
\]

\[
\mathcal L_T
=
\lambda_s\mathcal L_{\mathrm{sketch}}
+
\lambda_r
\sum_\ell
\left\|
\widehat{\Delta A}_\ell(q)-\Delta A_\ell^*(q)
\right\|_2^2.
\]

最终 functional query 必须来自部署式 Reuse + paired-correction closed loop。Current Exact query
teacher forcing 可作辅助，不得替代 closed-loop training/evaluation。
四次读取使用同一 query、mask、坐标和 Current reader，避免将 query 轨迹差异混入历史响应差。
\(D_\ell\) 是校准集确定的固定 K/V 尺度，不做逐样本幅值归一化。
校准前缀只包含查询之前可见的事件，覆盖追加查询和部分淘汰后的摘要形态。
淘汰教师从匹配的既有 Parent/Current 缓存删除相同条目，沿用服务的保留与坐标语义。

**混合状态的目标集合。** 记 \(\mathcal O_t\) 为当前目标以外 producer 的存活事件。
此时 \(C^p\) 替换为本分支的实际混合 cache，\(C^c\) 为按匹配保留/追加规则维护的当前目标参考。
上式的两次完整 K/V 响应都限制在 \(\mathcal O_t\)，source 与教师摘要使用相同事件归属，
重建 loss 也只作用于这些 translated slots。当前版本新增事件的响应误差单列为后代误差；
全历史误差与最终预测仍照常测量。若改为用旧段摘要补偿全历史差分，作为新的联动目标记录和比较，
不能在未说明的情况下改变监督集合。当 \(\mathcal O_t\) 为空时直接旧段修正为零，
不代表当前分支的普通 K/V 已与独立 Exact 轨迹相同。

用户已明确允许摘要拟合及 K/V-derived supervision，\(\mathcal L_{\mathrm{sketch}}\) 不再触发
单独许可或新监督合同。小校准的输入、目标和预算随配置记录；若用评价用户目标状态拟合，
明确列为对应诊断/对照并计入成本，不把这些结果同时报告为该用户上的未见泛化。
v0 保留当前请求六层 reader 的梯度，持久事件/跨发布轨迹 detach；更改早期译者后重新生成
受影响的后续混合输入，避免训练、评价与真实方法的状态分布错配。
optimizer step 后不复用旧 translated payload；最终固定译者后重新生成后续发布所需的实际轨迹。

### 3.4 连续发布设计的技术衔接

论文第 4.5 节 State Maintenance Across Releases定义同一 release family 内的混合来源迁移：每个目标版本共享一个
Translator，输入按事件顺序排列的待迁移 Source，并附各段 producer 标记；跨段、跨层读取后
输出新目标载荷。原始 Source 作为每次转换的输入，上一轮 translated sketch 只承担当时的读取
视图。稳态每段保留 Source 和一个目标视图，过渡缓冲与在途引用另计峰值存储。

其校准输入需要沿既定发布与事件顺序保留实际迁移、追加生成的混合来源状态，并由最新目标模型
提供教师。请求绑定目标模型和 source revision，旧目标迟到结果不能提交；新目标视图未就绪或
来源未覆盖时使用该目标的 Reuse，并计入覆盖与总体质量。来源退出以缓存和在途任务不再引用
为条件，旧目标 Translator 在相关服务与迁移任务结束后退出在线使用。

这些是设计接口定义。连续优化指 cache 从写入到跨多次发布的实际演化，包括长驻旧段、
新旧 producer 混合、淘汰和继承误差。首版原型即纳入小型顺序多版本轨迹，按
[plan §3.5](design/plan.md)推进，配置中记录发布顺序、混合来源与时间可见性。
不为每次局部修改重建合同。正式多版本评价前再封存发布序列、实际 Source
谱系、来源覆盖、混合校准人口、目标切换、资源和完整轨迹评价；现有五条相邻边不能作为连续迁移结果。
本次只建立计划，不新建运行合同或启动多版本校准。
首轮同步原型先测量视图就绪后的机制与准备/刷新成本；完整服务时段评价需加入就绪前 Reuse
请求及其后代写入，不能用同步原型声称异步迁移覆盖或完整等待期质量。

## 4. 按研究问题推进

下面列出表示、翻译、质量、持续性和成本的研究问题，不是每次探索必须完整走过的工程门禁。
根据当前假设选择最小实验；正式结论应覆盖其依赖的问题。已经发现的表示或数值错误需要
明确解释，不能用更复杂 Translator 掩盖。

### S0 — 最小代数与数值检查

- 用一个 tiny reference 对照检查 HSTU S4 分段聚合与当前 read 路径；
- 核对该路径使用的 count、mask、time/position 和缩放；
- 当前假设涉及追加或淘汰时，再检查对应 writer 更新；
- 需要安排长作业时，用同一小样本估计内存和耗时。

先修正会污染结果的代数错误。静态表示实验无需先实现 serialization、schema mismatch、
generation atomicity、并发事务或全部 empty/eviction 组合；涉及这些运行路径时再做必要检查。

### S1 — Exact-summary representation reference

直接从 frozen Exact caches 构造：

\[
S^p=W(C^p),\qquad S^c=W(C^c).
\]

不训练 Translator。对 held-out users/queries 比较：

\[
\Delta A_{\mathrm{sketch}}^*(q)
=R_c(q,S^c)-R_c(q,S^p),
\]

\[
\Delta A_{\mathrm{full}}^*(q)
=A_c(q;C^c)-A_c(q;C^p).
\]

报告 layer/head/edge response recovery 和 end-to-end exact-sketch injection。S1 用真实摘要检查
差分压缩误差，S2 检查翻译误差；两者可以同轮进行。真实摘要不是 learned 方法的严格效果上界，
响应训练可能补偿压缩误差。若 S1 不足，记录表示损失，结合闭环证据联改 schema、reader 或目标，
不把 S1 达标当作开始 Translator 的门槛，也不能声称已经验证“忠实翻译即可恢复响应”。

预注册表示 ablations：segment/slot count、K-only/V-only、count/time、paired versus
Current-sketch-only。qualification users 不参与选择。

### S2 — Translator calibration 与 generalization

以该轮记录的 schema，在 UID-disjoint calibration users 上训练 \(T_{p\rightarrow c}\)。网络 class、
capacity、loss weights、optimizer、sample count 和 stopping rule 随每轮配置保存；开发期可根据
证据联动修改，正式 confirmation 前冻结，不能用 qualification/scale 结果调参。

在 unseen users、histories 和 queries 上报告：

- payload error relative to true target summary；
- response recovery relative to full differential；
- 与 S1 真实摘要参考的差距，以及输出对真实摘要的偏离；
- calibration population、slots 和 capacity curves；
- 五条 adjacent edges 的完整结果。

若 S1 强而 S2 弱，裁决为 estimator failure；不能把 oracle 当方法结果。

### S3 — End-to-end single-edge quality

在逐层 closed-loop 路径比较 Current Exact、Reuse、No-op、Exact-summary reference、
EvoKV、direct mapper 和 generic controls。higher-is-better 指标：

\[
\rho_M
=
\frac{M_{\mathrm{method}}-M_{\mathrm{reuse}}}
{M_{\mathrm{exact}}-M_{\mathrm{reuse}}}.
\]

lower-is-better 指标使用方向一致定义。主要报告 response recovery、AUC、PR-AUC、log-loss、Brier、
rank agreement 和 user-equal companion。

成熟目标：在预声明的一次性 release budget 0%–20% 内至少 80% quality recovery，90% 为 stretch goal。
v0 先检验低计算、实质恢复的方向，不先锁死阈值；高成本低恢复只说明原型待改，不能宣布方法有效。
全部冻结 edges/seeds 均报告，最终质量主指标、达标口径和小 gap 处理在确认前固定。

### S4 — Closed-loop append 与 eviction

从迁移后的 edge 连续 append 真实 chronological events，并执行 fixed-window eviction。测量：

- Current-produced descendant K/V、response 和 task error；
- residual 随 append horizon 衰减、稳定或放大；
- whole/partial eviction 后的全用户 Parent 摘要 refresh；
- whole Parent-segment retirement；
- Parent 全部淘汰后的 endpoint；
- append/eviction 的因果顺序；先顺序模拟，仅在实际评估并发执行时检查事务行为。

Current producer tag 只证明参数/格式归属，不证明 exactness。开发时联合检查后代误差、最终质量
与刷新成本，按证据调整训练形态或有明确预算的局部维护；不能只因 producer 相同而忽略误差。
正式评价超过预先规定的容差时，不认定持续性成立；局部维护也必须给出效果与成本，不能自动算修复。
Parent 全部淘汰只消除直接旧版本项，不消除新增 K/V 的继承误差。所有对照采用相同的状态保留、
追加、淘汰与位置规则；不能一条路径保留 K/V、另一条每次重建滑动窗口，却把全部差距算作版本误差。

### S5 — Measured systems cost

分别测量 writer、calibration teacher/training、existing-cache backfill、population translation、
storage/I/O、per-request paired read、全摘要 refresh 和另行定义时的 rebase。
理论估计不得冒充 GPU/system result。
早期机制验证只需足以判断可行性的计算量、内存和小样本耗时；完整系统测量在论文相应
成本论点需要时展开，不作为每次算法尝试的前置工作。

## 5. Cost contract

一次性 release 与持续 request 使用不同分母：

\[
B_{\mathrm{release}}
=
\frac{
F_{\mathrm{teacher}}+F_{\mathrm{train}}
+F_{\mathrm{backfill}}
+F_{\mathrm{translate}}
}{
F_{\mathrm{Exact\mbox{-}All}}
},
\]

\[
C_{\mathrm{request}}
=
\frac{
C_{\mathrm{paired\ sketch\ read}}
}{
C_{\mathrm{ordinary\ history\ read}}
}.
\]

这里 \(F\) 均为计算量。生命周期 aggregate 只有在服务 horizon 预先冻结后才报告，累计绝对工作量
后统一除以分母，不混加比例与 FLOPs。I/O、storage 和实际运行时间单列。参考估计属于不同账本：

| Quantity | Reference estimate | Ledger |
| --- | ---: | --- |
| Current summary translation | Translator 未确定；旧 0.60% 不作为当前估计 | one-time release |
| two 32-slot payloads | 同精度为 6.25%；FP32 Source + 16-bit target/KV 为 9.375%，metadata、边界和缓冲另计 | storage |
| two 32-slot reads / 1,024 history | 约 6.25% attention work | recurring request |

calibration teacher 的最低人口摊销约为
\(N_{\mathrm{cal}}/N_{\mathrm{target}}\) 次 Exact-All。3,000 users 对 30,000 target population
为 10%；若正式分母只包含 21,200 eligible users，则为 14.15%，实际值还受 history length 影响。
prospective contract 必须冻结 target population、eligibility 和 history-weighted Exact-All denominator。
优先复用已有 Parent K/V 与 source summary；Translator training、response probes、backfill I/O 和 release write 另计。

论文默认以一次性 population release compute 的 0%–20% 为主门，不称为总生命周期预算；
同时计入 writer、paired read、全摘要 refresh，报告 latency、throughput、P99 和相对 Exact-All
在固定服务时段的 break-even。

## 6. 对照、失败与 fallback

### 6.1 Matched controls

以下保留早期草稿中的对照候选供查阅，不是现有实现或待办清单；其中部分旧路线代码已清理。
首版最小对照和后续按假设增加对照的方式见 [当前计划](design/plan.md)，不因此恢复退休路线。

- Current Exact、Reuse、No-op；
- Exact-summary reference；
- Translated-response-only add；
- Parent→Current raw KV/sketch ridge 或 MLP mapper；
- direct prediction 与 residual Translator；
- fixed offset、PRO、generic Current-r8；
- no response loss、no sketch loss、teacher-forced-only；
- no count/time、global slots、chronological segments；
- ordinary mixed-trajectory append 与 bounded replay/rebase。

Translator 本身是 mapping。创新必须由 producer-time sketch、paired functional read 与 lifecycle 的
完整收益承担；若直接 mapper 在相同监督、容量和成本下等价，论文不能只靠命名保留 Design claim。

### 6.2 Failure rules

- S0 failure：修正 state algebra；
- S1 failure：记录表示损失，根据闭环证据联改摘要与 reader/Translator；
- S2 failure：报告 estimator failure，开发配置变化记录在 design/ 中；不得据确认结果调参；
- S3 failure：不继承 oracle contraction；
- S4 failure：增加独立论证的 bounded refresh，或拒绝 persistent-state claim；
- cost failure：降低 slots/refresh 或拒绝 operating point，不改变 workload/denominator。

Current 已准入但 schema/hash/translation 缺失、generation 不一致时回退 Reuse；模型 admission
失败则保持 Parent 服务和谱系。迁移尚未完成的用户与请求计入覆盖和完成时间。No-op 对无显著
compatibility gap 的 edge 仍是合法结果。

## 7. 实现状态与执行顺序

### 7.1 已有

- frozen V0–V5 HSTU checkpoints；
- ordinary K/V、Full/Reuse 和 state transition primitives；
- stage/response instrumentation；
- diagnostic response-difference injection；
- Insight 1/2 的论文证据与诊断实现。旧 KV-only controls 多已退出活动代码目录，
  不能再整体列为现成实现。

### 7.2 未实现

- summary writer、backfill 和 segment add/subtract；
- exact-sketch S1 harness；
- edge-specific Translator 和 calibration pipeline；
- 用于实验的 paired reader；
- closed-loop append/contamination control；
- release executor 与 measured cost。

serialization、并发事务与通用恢复留到实验确实需要时。本阶段按 design/plan.md 先连通六层
完整 v0，追加、淘汰和再次发布从初版即进入顺序原型，成本随小样本同步估计；
S0–S5 为定位误差和支撑结论的维度，不是逐模块完成后才能前进的开发顺序。
小校准的 Current-derived supervision 已在本阶段范围内，无需为此另设审批。formal GPU population job
和长训练需要 focused canary、资源估计和用户明确 launch。相关小样本检查已通过且路径未变时直接复用，
不为文本或局部无关改动重跑；超过 30 分钟的作业使用 detached execution。

四张 GPU 0/1/2/3 均在 allowlist，但同一时刻至多运行一个 four-rank long job。UID shard 可并行，
edge/checkpoint 串行；CPU mapping、join 和 aggregation 可安全并行。

## 8. 协议禁区与 out of scope

- 不修改或覆盖 sealed contracts、hashes、raw、negative results 或 invalidations；
- 不使用 future labels、score mixing、selected-edge reporting；
- 明确 target-state/per-user fitting 的用户与用途，不将拟合用户上的结果混作未见泛化；
- 不把 diagnostic Exact-KV splice 加入 executable frontier；
- 不因 qualification outcome 调 population、slots、loss、capacity、edge 或 metric；
- 不把 request count 当统计重复；
- 不把 existing-cache backfill、calibration 或 paired-read overhead 记为零；
- 不把 Current-produced state 未经验证称为 clean/exact；
- 多版本开发记录真实 source-version routing、混合校准与发布顺序；正式评价再冻结协议，
  不据旧单边结果声称持续性，翻译不串联上一轮 translated sketch；
- 不在没有 \((N,Z)\) adapter 时声称 softmax Transformer 使用相同 segment-sum 公式；
- 不重新训练 V0–V5 backbone，不启动 RecFlow、theta3 或 next-item long training。
