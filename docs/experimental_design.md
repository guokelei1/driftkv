# EvoKV 具体实验设计

更新日期：2026-09-05

状态：**Medium KV-only discovery 已完成；Sketch-to-Sketch State Migration 为 prospective
Design 1；实现、Translator calibration 和方法实验均尚未运行。**

本文把 [论文总体设计](paper_design.md)具体化为可执行、可否证的实验协议。当前执行范围仍为一次
相邻 Parent → Current migration。现有 V0–V5 backbone 保持冻结，不重新训练；五条 edge 分别
校准共享、edge-specific Translator。论文新增的连续发布状态管理属于设计范围，尚不由这里的
单边执行协议覆盖，也不改变任何既有合同。

本文不修改任何 sealed motivation contract、checkpoint、data hash、release window、seed、workload、
metric、raw result 或 adjudication。历史探索和负结果继续由原结果与探索日志维护。
本次仅同步论文设计中的状态语义、诊断名称和成本定义；没有新建实验合同或启动运行。
Design 正文按组件与流水线叙述；本文件保留复现所需的技术定义和评价检查。
这些检查用于确认接口按定义工作、评价近似方法效果，不要求各模块与 Current Full 逐元素等价。

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

Sketch paired correction 的 HSTU 主注入点是 S4：聚合后 normalization 之前的可加 historical
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

Translator calibration 需要新增、与整个 3,000 Insight cohort UID-disjoint 的 population，并在
prospective contract 中封存。不得把同一用户同时用于 Translator fitting 和 final confirmation。

### 2.3 已完成证据

- Motivation、Full/Reuse 与 Medium locality 已封存；
- S4 shared/low-rank oracle 显示 functional contraction；
- fixed correction、tail、sparse carriers、paired replay、release-algebra 和多类 KV-only constructors
  已形成正负证据；
- ordinary KV、Full/Reuse、response instrumentation 和 diagnostic delta injection 已有代码。

这些只决定为什么进入新的 state interface，不是 Sketch-to-Sketch 方法结果。

## 3. Prospective state 与 Translator

### 3.1 Fixed Source Sketch

参考 schema 每段最多 64 个事件、每段每层两个固定事件区间 slots。边界对齐的 1,024-event
窗口有 16 段、每层 32 slots；滑动边界与 release 提前封段的额外容量需单列：

\[
S_g^p=W(C_g^p;m_g).
\]

writer \(W\) 固定、可加减、可序列化。Source 保存实际持久化 K/V 的累加量、mass/count、
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

两版 backbone 冻结。calibration users 可产生 Parent/Current Exact Sketch 和 response teacher：

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

当前仓库规则禁止 target-KV fitting，而 \(\mathcal L_{\mathrm{sketch}}\) 使用 Current-KV-derived
teacher。任何 S2 Translator 训练前必须建立 prospective contract，明确允许 disjoint calibration
population 上的共享 edge-level supervision，并继续禁止 evaluation-user/per-user target fitting。
本文档不是 launch authorization。

### 3.4 连续发布设计的技术衔接

论文 Design 1 第 3.5 节定义同一 release family 内的混合来源迁移：每个目标版本共享一个
Translator，输入按事件顺序排列的待迁移 Source，并附各段 producer 标记；跨段、跨层读取后
输出新目标载荷。原始 Source 作为每次转换的输入，上一轮 translated sketch 只承担当时的读取
视图。稳态每段保留 Source 和一个目标视图，过渡缓冲与在途引用另计峰值存储。

其校准输入需要沿既定发布与事件顺序保留实际迁移、追加生成的混合来源状态，并由最新目标模型
提供教师。请求绑定目标模型和 source revision，旧目标迟到结果不能提交；新目标视图未就绪或
来源未覆盖时使用该目标的 Reuse，并计入覆盖与总体质量。来源退出以缓存和在途任务不再引用
为条件，旧目标 Translator 在相关服务与迁移任务结束后退出在线使用。

这些是设计接口定义。后续多版本执行需单独封存发布序列、实际 Source 谱系、来源覆盖、混合
校准人口、目标切换、资源和完整轨迹评价；现有五条相邻边不能作为连续迁移结果。
本文件下述 S0–S5 保持单边含义，本次不新建运行合同或启动多版本校准。

## 4. 分阶段 admission

前一 gate 未通过时，不得用更复杂 Translator 掩盖问题。

### S0 — Algebra、writer 与 lifecycle canary

- 验证 HSTU S4 对不相交 history segments 可加；
- 验证 count、mask、time/position 和 mass-aware sketch read；
- 验证 writer add/subtract、partial/full eviction 和 empty slots；
- 验证 serialization、schema/hash mismatch、generation atomicity 和 Reuse fallback；
- 测 tiny-batch writer/backfill/read FLOPs、bytes 和 runtime。

如果 S0 algebra 失败，修正 state interface；不能训练补偿 mapper。

### S1 — Exact-Sketch representation reference

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
差分压缩误差，S2 再检查翻译误差。真实摘要不是 learned 方法的严格效果上界；响应训练可能补偿
压缩误差。若 S1 不足，先停止当前 schema 的推进、审视表示假设；不能仅扩容 Translator 并声称
已经验证“忠实翻译即可恢复响应”。这是一项设计推进规则，不是普遍不可能性定理。

预注册表示 ablations：segment/slot count、K-only/V-only、count/time、paired versus
Current-sketch-only。qualification users 不参与选择。

### S2 — Translator calibration 与 generalization

固定 S1 schema 后，在 UID-disjoint calibration users 上训练 \(T_{p\rightarrow c}\)。网络 class、
capacity、loss weights、optimizer、sample count 和 stopping rule 在 development 阶段冻结。

在 unseen users、histories 和 queries 上报告：

- payload error relative to true Current Sketch；
- response recovery relative to full differential；
- 与 S1 真实摘要参考的差距，以及输出对真实摘要的偏离；
- calibration population、slots 和 capacity curves；
- 五条 adjacent edges 的完整结果。

若 S1 强而 S2 弱，裁决为 estimator failure；不能把 oracle 当方法结果。

### S3 — End-to-end single-edge quality

在逐层 closed-loop 路径比较 Current Exact、Reuse、No-op、Exact-Sketch reference、
Translated-Sketch、direct mapper 和 generic controls。higher-is-better 指标：

\[
\rho_M
=
\frac{M_{\mathrm{method}}-M_{\mathrm{reuse}}}
{M_{\mathrm{exact}}-M_{\mathrm{reuse}}}.
\]

lower-is-better 指标使用方向一致定义。主要报告 response recovery、AUC、PR-AUC、log-loss、Brier、
rank agreement 和 user-equal companion。

目标：在预声明的一次性 release budget 0%–20% 内至少 80% quality recovery，90% 为 stretch goal。
全部冻结 edges/seeds 均报告；允许在合同中预注册四条边达门，但不隐藏其余边。

### S4 — Closed-loop append 与 eviction

从迁移后的 edge 连续 append 真实 chronological events，并执行 fixed-window eviction。测量：

- Current-produced descendant K/V、response 和 task error；
- residual 随 append horizon 衰减、稳定或放大；
- whole/partial eviction 后的全用户 Parent 摘要 refresh；
- whole Parent-segment retirement；
- Parent 全部淘汰后的 endpoint；
- append/eviction 并发与事务顺序。

Current producer tag 只证明参数/格式归属，不证明 exactness。若 descendant contamination 超过预注册
tolerance，Design 1 不准入，除非另行定义并验证 bounded replay/rebase 或 clean-write mechanism。
Parent 全部淘汰只消除直接旧版本项，不消除新增 K/V 的继承误差。所有对照采用相同的状态保留、
追加、淘汰与位置规则；不能一条路径保留 K/V、另一条每次重建滑动窗口，却把全部差距算作版本误差。

### S5 — Measured systems cost

分别测量 writer、calibration teacher/training、existing-cache backfill、population translation、
storage/I/O、per-request paired read、全摘要 refresh 和另行定义时的 rebase。
理论估计不得冒充 GPU/system result。

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
优先复用已有 Parent KV/Sketch；Translator training、response probes、backfill I/O 和 release write 另计。

论文默认以一次性 population release compute 的 0%–20% 为主门，不称为总生命周期预算；
同时计入 writer、paired read、全摘要 refresh，报告 latency、throughput、P99 和相对 Exact-All
在固定服务时段的 break-even。

## 6. 对照、失败与 fallback

### 6.1 Matched controls

- Current Exact、Reuse、No-op；
- Exact-Sketch reference；
- Current-Sketch-only add；
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
- S1 failure：拒绝或重定义 sketch；
- S2 failure：报告 estimator failure，配置变化需新 prospective decision；
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
- Insight 1/2 evidence 与 KV-only controls。

### 7.2 未实现

- fixed Sketch schema/writer/serialization；
- backfill、segment add/subtract 和 transactional lifecycle；
- exact-sketch S1 harness；
- edge-specific Translator 和 calibration pipeline；
- production paired reader；
- closed-loop append/contamination control；
- release executor 与 measured cost。

执行顺序严格为 S0 → S1 → S2 → S3 → S4 → S5。S0/S1 可先做 CPU/tiny-GPU canary。S2 触及
Current-derived supervision，必须先有新合同。任何 formal GPU population job 都需要 focused canary、
资源估计和用户明确 launch；超过 30 分钟的作业使用 detached execution。

四张 GPU 0/1/2/3 均在 allowlist，但同一时刻至多运行一个 four-rank long job。UID shard 可并行，
edge/checkpoint 串行；CPU mapping、join 和 aggregation 可安全并行。

## 8. 协议禁区与 out of scope

- 不修改或覆盖 sealed contracts、hashes、raw、negative results 或 invalidations；
- 不使用 future labels、score mixing、selected-edge reporting；
- 不对 evaluation users 做 Current target-state 或 per-user fitting；
- 不把 diagnostic Exact-KV splice 加入 executable frontier；
- 不因 qualification outcome 调 population、slots、loss、capacity、edge 或 metric；
- 不把 request count 当统计重复；
- 不把 existing-cache backfill、calibration 或 paired-read overhead 记为零；
- 不把 Current-produced state 未经验证称为 clean/exact；
- 不用当前单边合同运行多版本校准或任意 source-version routing；论文设计中的混合来源扩展
  另建协议，翻译不串联上一轮 translated sketch；
- 不在没有 \((N,Z)\) adapter 时声称 softmax Transformer 使用相同 segment-sum 公式；
- 不重新训练 V0–V5 backbone，不启动 RecFlow、theta3 或 next-item long training。
