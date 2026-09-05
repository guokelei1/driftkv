# Insight 2 / Design 1 探索日志

本文件只追加重要且可持久化的研究记录。raw evidence、hash 与正式 adjudication 放在
`results/yambda500m_medium_seed17/insight2_functional_boundary_v1/`；这里记录每轮为何做、
观察到什么、哪些解释被排除以及下一步只改变什么。

## 2026-09-02：Iteration 0 — 研究定位与资产盘点

### 输入方向

- Insight 1 已完成：local token/layer/K/V importance 不能直接成为 Design 1 的主要迁移抽象。
- Insight 2 与 Design 1 重开；existing candidate-shared AV correction 和 lightweight PRO 是强先验，
  不是冻结答案。
- 单次跨版本迁移；理论计算甜点区 `0%–20%`；目标 functional recovery `80%–90%+`，
  允许一条 edge 失败但必须完整报告。

### 已核对资产

- Yambda-500M Medium D14 `v0..v5` 六个 formal checkpoint 均存在；架构为
  HSTU-native CC 6L/H192/6 heads/context1024，seed 17。
- Medium fixed population 为 30,000；Insight 1 已封存一套从 21,200 个
  full-history eligible users 中确定的 3,000-user label-free population，以及
  `[5, 3000, 64]` candidate panels，可直接复用而不重新选择用户或候选。
- Insight 1 的 34 个 Exact-KV locality splice 已完成五边观察：10%/20% token budget 的
  edge-equal best recovery 约 30.0%/41.2%，单层 16.7% budget 约 45.6%；这些是 optimistic
  diagnostic coverage，不是可执行成本，支持继续寻找非局部功能边界。
- 四张 NVIDIA A40 46 GiB 当前空闲；工作盘约 721 GiB 可用。正式执行仍遵守“一条四 rank
  job、edge/checkpoint 串行、先 canary 后 formal”。

### 文档问题

当前 `docs/paper_design.md`、`docs/experimental_design.md`、`docs/insight_develop_map.md`
把 candidate-shared AV/PRO 写成已冻结 Insight/主设计；这与本轮方向冲突。既有数字应作为
observations 保留，但结论层必须降级为候选证据，并恢复 architecture-neutral boundary search。

### 首轮假设

首轮只检验：position-indexed S2/S3 error 是否在 S4–S7 中出现更紧凑、anchor-to-heldout
可迁移的 response。rank0 和 low-rank query-conditioned correction 同时报告，避免把协议写成
为 PRO 寻证据。

### 下一步

实现不修改核心 model forward 的 stage adapter、数值 reconstruction tests 和 32-user/five-edge
focused canary。通过 Gate A 后再决定是否启动 512-user discovery run。

## 2026-09-02：Iteration 1 — 最早的紧凑功能边界

### 执行与口径修正

- 32-user canary 通过后，在 discovery `[0,512)` 上完成五条 edge、S3–S7 的
  anchor-to-heldout intervention；共 51,200 条无标签 record，wall time 355.24 秒，峰值
  reserved memory 3,818 MiB/卡。
- 第一版汇总误把 gap-weighted recovery 当 primary，违反“先 per-user、再 edge-equal”的协议。
  原目录保留并写入 `INVALIDATED.md`；正式裁决在 `discovery_functional_boundary/analysis_v2/`，
  不删除错误路径。

### 结果

| stage / rank | edge-equal recovery | minimum edge | >=90% edges | FP32 scalars/user |
| --- | ---: | ---: | ---: | ---: |
| S4 aggregated context / rank0 | 0.9534 | 0.9446 | 5/5 | 1,152 |
| S4 aggregated context / rank1 | 0.9946 | 0.9891 | 5/5 | 4,608 |
| S7 final representation / rank0 | 0.9274 | 0.8949 | 4/5 | 192 |
| S7 final representation / rank1 | 0.9837 | 0.9735 | 5/5 | 768 |
| S5/S6 rank0 | 0.7982 | — | — | 1,152 |

S4 是最早通过 Gate B 的 position-free boundary。S7 更紧凑，但当前 CC readout 是线性标量头，
candidate-shared S7 offset 只产生同用户 common logit bias，不能恢复 within-user ranking，因而不作为
主设计边界。S5/S6 在当前 same-hidden additive intervention 中代数等价，不能包装为两条发现。

### 裁决

成立的是 **S4 response 的表示上界**，不是 AV estimator。Exact anchor correction 仍读取完整
Current Exact state；所有 positive-rank coefficient 也属于 oracle。下一轮只改变 estimator，边界与
candidate split 不再调。

## 2026-09-02：Iteration 2 — 第一个合法 functional-probe estimator 失败

冻结 `Parent K/V + Current/Parent 参数 + fixed history probes + compact Current carriers` 的
parameter-only/Parent-conditioned estimator，并预注册 16 个 `C8/16/32/64 × P1/2/4/8` 点，理论
generation cost 为 Exact-All 的 2.30%–18.65%。32-user/five-edge instrumentation 全部通过，
但最好点 `C16/P2` 的 edge-equal recovery 为 `-0.9288`、minimum edge `-1.8453`、0/5 edge 达
80%；增加 probe 几乎无效，增加 carrier 只提高 tensor cosine、没有改善 score recovery。

结论：S4 boundary 的存在不意味着 parameter-only map 加 Parent-conditioned carrier response 能估计
其幅值。按 canary stop，不启动 512 users。

## 2026-09-02：Iteration 3 — 修复 time semantics 仍不能救回 estimator

旧 probe 把 query delta 固定为 0，而 target 使用 `cutover - last_event_time`。新合同只修这一项，
固定 `C32/C64`，其余不变。time-aligned canary 的 C32/C64 recovery 为 `-1.0590/-1.1410`；相比
zero-time baseline `-1.2993/-1.4062` 有改善，但仍为 0/5 edge 达门。该 family 正式退休，不用
时间修正后的结果反向调 carrier、probe 或 scale。

## 2026-09-02：Iteration 4 — release-level response subspace

这一轮只做结构诊断，不拟合 executable predictor：前 64 个 discovery users 构造每层 release
basis，后 448 个 UID 隔离用户使用各自 Exact coefficient 做 oracle projection；同时用八个固定
history queries 测 query-sampling ceiling。512-user run 用时 75.37 秒，labels absent。

| diagnostic | recovery | minimum edge | >=90% edges |
| --- | ---: | ---: | ---: |
| Exact fixed-history-query mean | 0.9470 | 0.9358 | 5/5 |
| release basis rank4 | 0.8947 | 0.7906 | 3/5 |
| release basis rank8 | 0.9418 | 0.9316 | 5/5 |
| release basis rank32 | 0.9469 | 0.9358 | 5/5 |

Calibration cohort 的 mean per-layer rank@90 为 1.53，rank1/rank2/rank4 energy 为
0.8884/0.9604/0.9893。这支持“response subspace 很小”，但每个 evaluation user 的 projection
coefficient 来自 Exact functional target，不能作为 action。Population calibration 会增加 predictor
和新的成本摊销口径，当前没有授权；不执行、不把 oracle basis 放进 frontier。

## 2026-09-02：Iteration 5 — 时间持久性否定“固定 offset”

冻结 cutover anchor S4 correction 一次，在真实 E14 request timeline 上保持两条 cache 都只用 Current
append、同时间戳 query-before-append。主裁决始终用固定 held-out panel；真实 exposed items 是伴随。
32-user canary 的数值门通过后，在 512 users、五条 edge、10,447 个 request groups 上完成 discovery，
wall time 601.99 秒，峰值 reserved memory 4,536 MiB/卡。

| quantity | 512-user result |
| --- | ---: |
| same-request S4 oracle recovery | 0.9339 |
| cutover/current direction cosine | 0.9460 |
| current/cutover norm ratio | 0.7697 |
| fixed unscaled correction recovery | -34.2189 |
| remaining-Parent-coverage scaled recovery | 0.3385 |
| coverage-scaled positive edges | 5/5 |

coverage-scaled recovery 随时间为 `[0,1d) 0.5204 / [1,3d) 0.5526 / [3,7d) 0.3509 /
[7,14d) 0.2108`，未达到 Gate C 的 0.70。fixed offset 的极端负值未裁剪；约 6.83% 请求的
Reuse probability gap 小于 `1e-4`，但将 denominator floor 固定为 `1e-4` 后主值也只有 0.3965，
不改变裁决。

### 新结构解释

功能 response 的**方向**在 Current append 与 Parent eviction 下仍高度稳定，但其**系数**随 query
time、layer 与 cache composition 演化。于是“persistent migration object = 一个固定 AV offset”被
否定；更准确的候选对象是低维 response basis 及其演化坐标。它仍需受控 coefficient projection
验证，不能由 cosine 直接宣布成立。

## 2026-09-02：Iteration 6 — 最后一个无新 predictor 的 <=20% estimator

只复用已有 dependency-closed Tail-128：在 Parent first-896 K/V 上按 Current 顺序重放 recent128，
用 1/2/4 个固定 history query 读取 mixed-minus-Parent S4 response，丢弃 transient mixed cache，只写
1,152-scalar sidecar。理论 generation compute 分别为 18.2842%/18.5303%/19.0224%。

Instrumentation 通过：single-probe mixed-reader replay error `1.91e-6`，Parent cache/prefix change 0。
但 P4 recovery 为 `-0.0878`，minimum edge `-0.3518`，只有 2/5 正向；P1/P2/P4 几乎相同。
按预注册 stop rule 退休此 family，不启动 512，不修改 tail width、位置、probe 或 scale。

这同时排除了一个重要捷径：94.70% 的 fixed-history-query ceiling 先计算了完整 Current 1024-state，
其成本至少约 101.97% Exact-All；不能把“query 数少”偷换成“prefix contextualization 便宜”。

## 2026-09-02：Iteration 7 — 时间坐标诊断（进行中）

基于 Iteration 5 的高 direction cosine，冻结两个 oracle：一个全局 least-squares coefficient，或每层
一个 coefficient，将 cutover S4 direction 投影到当前 request correction。32-user canary 中，全局
1-scalar recovery 为 0.5815；6-scalar layerwise recovery 为 0.7274；两者五边均正，但都未过 0.80。
同请求完整 S4 shared correction ceiling 仍为 0.9447。

因为 layerwise 值接近门槛且 canary user 数小，按同一预注册协议执行 512-user discovery；不读取
confirmation，也不把 oracle coefficient 写成 estimator。下一条日志只追加完整 discovery 裁决。

## 2026-09-02：Iteration 7 补充 — 时间坐标完整裁决

512-user discovery 覆盖五条 edge、10,447 个真实 request groups，用时 624.72 秒，峰值 reserved
memory 4,536 MiB/卡。主结果如下：

| representation diagnostic | edge-equal recovery | minimum edge | positive edges |
| --- | ---: | ---: | ---: |
| one global oracle coefficient | 0.4896 | 0.3333 | 5/5 |
| six layerwise oracle coefficients | 0.6504 | 0.4940 | 5/5 |
| same-request full S4 shared correction | 0.9339 | 0.9125 | 5/5 |

global/layerwise projection relative L2 分别为 0.2099/0.1665。即使 coefficient 由当前请求的
Current Exact correction 直接求最小二乘，固定 cutover direction 加 1 或 6 个 oracle coordinate 也
无法达到 0.80 门；因此失败不能归因于 coefficient estimator。

这一结果否定“persistent object = 固定 response basis + 少量独立时间标量”。更准确的下一假设是：
迁移对象必须保留一个由真实 Current query 读取的紧凑 response operator，使 query、layer、time 和
cache composition 通过 Transformer 自身的 interaction 联合决定系数。标量实验只承担机制证伪，
不进入 Design 1。

## 2026-09-02：Iteration 8 — 从 correction vector 转向 signed response operator

### 为什么这是新机制而不是增加 mapper 复杂度

Iteration 7 已经给了 Current-Exact oracle coefficient，却仍只有全局 `48.96%`、逐层 `65.04%`；
因此下一轮明确禁止 ridge/MLP、score fitting 和自由 learned prompt。新假设把完整 Parent attention
response 保留为 control variate，只表示两版本 attention measures 的 signed residual：

~~~text
R_hat_Current(q)
  = R_Parent(q)
    + sum_{i in stratified landmarks} (1/pi_i)
        [rho(q, k^Current_i) v^Current_i
         - rho(q, k^Parent_i) v^Parent_i]
~~~

未来 held-out candidate 用自己的 Current q 经过模型原有 activation 读取上述 K/V entries；不接收
广播 offset。每个 entry 同时携带时间 stratum、原位置和 lineage，使后续 eviction 可以删除 support，
而不是预测一个时间幅值。可执行版本将利用 Current layer-0 K/V 的 dependency-free projection 作为
seed，再让固定 landmarks 按历史顺序读取 Parent prefix 与 earlier residual 做 causal replay。

这与普通 prefix/memory token 的区别不在名字，而在四件事同时成立：version-residual decomposition、
Reuse control variate、causal defect propagation 和 lineage-aware native query read。近期 cross-model
KV transfer 的 ridge/MLP 路线、同模型 KV compression 的 learned/compensation token 都作为明确
related-work boundary，不作为本方案的实现替代。

### 冻结的最小 oracle

合同：
`configs/contracts/yambda500m_medium_hstu_native_insight2_signed_response_coreset_v1.yaml`
（SHA256 `e3ad86a028ed101575496620ea93dc9e1e2a428c9407ad709943bb3d96e5099a`）。

- Exact Current/Parent cache 只用于构造 fit-free chronological midpoint coreset，因此本轮仍是
  representation oracle，不进入成本 frontier；
- 固定 `R={8,16,32,64,128}`，全部 user/layer/head/edge 使用同一规则；`R=1024` 只做 Exact
  reconstruction correctness；
- construction 不使用 candidate；只在 frozen odd-32 held-out panel 上评价；
- canary 的 R128 recovery 至少 `0.70` 且至少 4/5 edge 正向，才允许 512 discovery；
- discovery 中 R64 或 R128 至少 `0.80` 且至少 4/5 edge 正向，才支持 compact native-query
  response-operator 表示；若 R128 低于 `0.70`，固定 midpoint signed coreset family 直接退休。

只有这一级通过，下一轮才会另立合同实现不读取 Current Exact upper-layer K/V 的 sparse causal
replay constructor。当前不因理论成本估计看似低于 20% 而提前准入 action。

## 2026-09-02：Iteration 8 补充 — chronological landmark 反例与 family 退休

正式 odd-32 canary 覆盖五条 edge，`R=1024` instrumentation 的最大 logit/readout 误差分别为
`4.768e-7/2.861e-6`，确认 signed replacement 实现可以数值重建 Exact。紧凑 coreset 的 edge-equal
recovery 则为：

| landmarks | edge-equal recovery | minimum edge | positive edges |
| ---: | ---: | ---: | ---: |
| 8 | -2.7572 | -4.6163 | 1/5 |
| 16 | -1.2864 | -2.8975 | 1/5 |
| 32 | -1.0109 | -2.0547 | 1/5 |
| 64 | -0.0719 | -0.6908 | 3/5 |
| 128 | 0.0967 | -0.8886 | 3/5 |

`R=128` 未达到预注册的 `0.70` canary 门，因此不启动 512 discovery，也不开发基于 chronological
midpoint 的 causal constructor。该 family 正式退休。单个 UID 预检曾出现 `R=128` recovery `0.8768`，
但它没有证据地位，且正式 population canary 已经否决以此选点或调参。

这个反例否定的是“时间均匀节点足以积分 response defect”，不是任意 native-query operator。当前
Medium checkpoint 使用 `ELU(qK)+1`，raw qK 具有很大的正负幅值；response 主要由 query 与 key
在地址空间中的半空间关系决定。chronological midpoint 不控制这种 kernel coverage，因而即使时间上
均匀也可能系统遗漏决定 response 的 key 区域。

## 2026-09-02：Iteration 9 — attention-address hypothesis（仅预检，未形成证据）

下一轮只改变一个科学变量：保持完整 Parent control response、paired Current-minus-Parent atoms、
native query read、held-out panel 和所有评价指标不变，把 landmark rule 从 chronological position
换成 **layer-0 cross-version attention-address coverage**。layer-0 Current K 可由 raw event 独立投影，
因此这一选择变量将来可以落到合法 constructor，而不需要 Current upper-layer Exact state。

一个 UID、`v0->v1` 的非正式预检使用 normalized `[K0_Current, K0_Parent]`、deterministic
farthest-first 和 Voronoi mass，得到 `R=8/16/32/64/128` recovery
`-1.0283/-0.8278/0.6227/0.9832/0.9287`。它只说明新假设值得立 prospective contract；不能用来
选择 `R=64`、声称通过，或继续在该 UID 上调规则。正式实验必须冻结 nested grid，并在同一 odd-32、
五 edge 上同时报告全部点。

若 address-aware coreset 显著超过已封存的 chronological control，Insight 2 才可能进一步收紧为：

> Cross-version state error is non-local in token/time coordinates, but its signed functional defect
> contracts in the Transformer reader's attention-address space.

即便 oracle 通过，单纯的 key clustering 也不是 Design 贡献。Design 1 的论文准入必须同时包含
cross-version signed defect、Reuse control variate、native query read、causal defect replay 与 lineage
transport；address landmarks 只负责找到这项机制的 quadrature support。任何 ridge/MLP、自由 virtual
token、per-user output mapping 或 qualification 调参都不进入该路线。

## 2026-09-02：Iteration 9 补充 — attention-address coreset 正式反例

odd-32、五 edge 正式 canary 的 `R=1024` 数值重建通过：最大 logit/readout 误差为
`4.768e-7/2.861e-6`，nested selector、mass conservation、cache immutability 全部通过。紧凑点结果为：

| landmarks | edge-equal recovery | minimum edge | positive edges |
| ---: | ---: | ---: | ---: |
| 8 | -2.1787 | -4.1756 | 0/5 |
| 16 | -0.7876 | -2.5183 | 2/5 |
| 32 | -1.2751 | -2.9520 | 2/5 |
| 64 | -0.4532 | -1.9112 | 2/5 |
| 128 | 0.3719 | -0.1914 | 3/5 |

`R=128` 虽比 chronological control 高 `0.2752`，仍未达到 `0.70` canary 门，且只在 3/5 edge
正向。因此不启动 512 discovery，attention-address real-state coreset family 退休。这个结果说明 key
coverage 不能替代 cluster 内的 value mass 与 activation-branch 变化；继续换距离、聚类数或单 UID 规则
只会成为 sampler tuning，不满足论文贡献标准。

## 2026-09-02：Iteration 10 — user attention cone 与 exact response moments

回到 HSTU reader 公式后发现一个与前两轮不同的结构。Medium 使用 `ELU(s qK)+1`；对固定 positive
set `P`，positive response 精确等于：

~~~text
B_P + s q M_P,
B_P = sum_{i in P} v_i,
M_P = sum_{i in P} k_i outer v_i.
~~~

这不是 Taylor fitting，也不是把模型改成 linear attention；它是原 kernel 在固定 sign cone 内的精确
代数。推荐场景的待证命题是，同一用户的多 candidate query 占据稳定 cone，因此历史 reader 可由
user-level response moments 表示。

在冻结 UID `1930`、`v0->v1` 的非正式预检中：

- anchor-unanimous positive/negative interior 覆盖每层 head-state 的约 `76.6%–91.7%`，其 held-out
  sign stability 接近 `0.9999`；negative-branch signed response norm 近乎为零；
- 但 Current/Parent sign crossing 本身约占 `8.3%–23.3%`，不能把两版本强行放进同一 positive set；
- 分别对 Current 与 Parent 用 anchor majority-positive set 构造 moments 后，coherent held-out recovery
  为 `0.9997`；
- 完整 Parent moments 加 Current sampled moments 时，chronological `R=128` recovery 为 `0.6222`，
  layer-0 address quadrature `R=64/128` 为 `0.6528/0.9593`。

这个预检改变了核心对象：address landmarks 不再作为压缩 K/V 的贡献，而只可能作为估计 additive
Current moments 的工具。候选 persistent state 是每 user/layer/head 的 signed `Delta B/Delta M`；Medium
仅 `38,016` scalars，约为完整 Current K/V 的 `1.61%`。future query 通过 `Delta B+s q Delta M`
自然得到不同 correction。

下一轮必须在同一 odd-32、五 edge 正式检验 full moment representation 与 sampled moment estimator。
full moments 若不能达到 edge-equal `0.90`、最差 edge `0.80`，该结构直接否决；address-R128 若不能
达到 `0.70` 且 4/5 edge 正向，则不开发 sparse causal replay。单 UID 数值不参与最终选择。

## 2026-09-02：Iteration 10 补充 — activation-region 表示通过，采样构造失败

正式 32-user、odd-32 held-out、五 edge canary 已完成，instrumentation、cache immutability、
`38,016` 个 persistent floating scalars 和 `1.611328125%` Current-KV storage ratio 均通过。
完整 Current/Parent positive-branch moments 的 edge-equal probability-gap recovery 为
`0.9957`，最差 edge 为 `0.9892`，五边全部为正。held-out query 对 anchor-majority region 的
Current/Parent sign agreement 分别为 `0.99445/0.99446`；Current/Parent sign crossing 为
`0.08292`。Current/Parent negative-branch response-norm fraction 为 `0.03758/0.04366`。

这个结果支持的严格表述是：当前 legacy `ELU+1`、无 relative-bias reader 中，推荐 query 共享一个
高度稳定的 activation region；region 内的 **positive affine bulk** 可以用 response moments 表示，
而 negative branch 与 query-specific region flips 构成非仿射 remainder。它不是完整 attention response
的精确等式：正式 recovery 很高但不等于 1，negative remainder 也没有被静默称为零。

同一 canary 中，absolute Current-moment quadrature 失败：chronological/address `R=128` recovery 分别为
`-2.9455/0.2733`，address 最差 edge 为 `-0.5077`，虽有 4/5 edge 正向，仍远低于预注册的 `0.70`
launch gate。因此不启动 512 discovery；不再通过更换 clustering、distance 或 sample count 调整该
constructor。full row 只冻结为 representation evidence，不能准入 Design 1。

此外，related-work audit 修正了 claim 边界：原始 HSTU 论文使用 SiLU 并包含 relative attention bias；
当前代数只属于仓库这组 legacy `ELU+1`/no-bias Medium checkpoints。“cone”只是无 bias 时 activation
region 的特例，不能写成一般 HSTU theorem；`B/qM` 也与 fast-weight/linear-attention sufficient state
有直接重叠，不能单独作为设计创新。下一轮只有在利用 activation-region 结构改变合法 release-time
计算图，并闭合 lineage update 与 region-exit detection 后，才可能进入论文 Design。

## 2026-09-02：Iteration 11 — paired functional-delta causal closure 的正式反例

### 机制与预注册边界

这一轮没有把 moments、control variate、address selector 或 token coreset 当成创新。唯一待检验的新机制是
`recursive paired functional-delta closure`：完整 Parent cache 作为 base path；按真实时间 commit 的
Current-minus-Parent response defect 既服务未来推荐 query，也被后续 carrier 读取，从而通过 Current
Transformer block 递归生成自己的 upper-layer continuation。`R=N` singleton 时该 recurrence 数值恢复
Exact Current；`R64` 的实际总理论计算上界必须不超过 Exact-All 的 `20%`。

v1 prospective contract 为
`configs/contracts/yambda500m_medium_legacy_pointwise_insight2_paired_functional_delta_v1.yaml`
（SHA256 `605fc606d1e1ba0f78c17841a6580d807a97cf25d6775dbfedb6ee320a460e83`）。首次执行在写出任何
metric row 以前因 metadata string 被错误传给 `np.isfinite` 而终止；partial raw 与 failure record 均保留，
failure record SHA256 为 `6a86f03550dff22c734d98539a61519e3ee4fd756339feb8bcbd79e484df02de`。

v2 只修复这一 finite-check type guard，没有改变 user、support、weight、method、metric、cost 或 gate。
execution-only amendment contract SHA256 为
`45d6d5affc2626f527547e74e2af76e41677d9c4da98b872f753abf4d50bc8b2`。正式 32-user、五 edge
canary 共写出 1,920 个 metric rows 与 160 个 diagnostic/correctness rows，用时 `84.02s`，四卡峰值
reserved memory `3324 MiB`；construction 不读取 candidate、label 或 Current Exact upper-layer K/V。

### 正式结果

| object / path | edge-equal recovery | minimum edge / other gate |
| --- | ---: | ---: |
| full-history activation-region representation, P8 | 0.9951 | 0.9873 |
| same-support Exact-state carrier oracle, R64 | -0.4532 | quality gate fail |
| Parent-conditioned independent carriers, R64 | -0.3431 | control |
| recursive paired closure, R64 | -0.1962 | quality gate fail |
| exact-Current-layer0 consistency ablation, R64 | -0.3765 | not a legal improvement |
| affine compiler, R64 | -0.1853 | tracks native closure |

recursive closure 相对 matched independent control 的 edge-equal 增益为 `+0.1468`，在 4/5 edge 上为正；
五条 edge 的增益依次为 `-0.0326/+0.0498/+0.1902/+0.4799/+0.0469`。但 160 个 user-edge pair 的
paired bootstrap 95% CI 为 `[-0.0060, 0.3834]`，包含零；而且 closure 自身仍低于 Reuse。R128 closure
只有 `0.3703` 左右的 edge-equal recovery，且理论计算约 `27.10%`，只具有 diagnostic 身份。

R64 实际最大 neural、selection、total compute fraction 分别为 `14.37%/3.22%/17.59%`；所有 user 与
任意 64 个 unique positions 的审计上界都在 `20%` 内。因此否定结果不是成本超界造成的。canary 的
Design gate 失败，不启动 512-user discovery。

### 新的结构结论与下一轮约束

这一轮同时建立了两个不能再混淆的事实：

1. 全历史经过 reader aggregation 后的 positive-affine bulk 极其紧凑，并且跨 held-out request
   因果充分；
2. 这个紧凑输出**不具有 token-support sparsity**。无论真实 Exact carrier、chronological support、
   attention-address support，还是在相同 support 上加入 causal recursion，都不能可靠代替 dense
   history reduction。

因此当前可冻结的是反例，而不是 Design：**functional compactness does not imply support sparsity**。
下一轮不再修改 selector、distance、cluster、carrier 数或 mapper。新的候选必须让全部历史贡献进入
一个 bounded-cost associative/dense reduction，只在 aggregation 后压缩 query/operator/layer 维度；
若只是把新的 token sampler 接到现有 paired ledger 上，即使数值改善也不满足论文创新门槛。

## 2026-09-03：Iteration 12 — 从 token support 转向稠密 history modes

### 被排除的三条捷径

在不读取 confirmation 用户、不启动训练的条件下，先完成三个机制与成本 preflight：

1. 直接把 `Parent KV -> Current KV` 聚合成 release-global operator，在 UID `1930` 的五条 edge 上只有
   `0.64/0.26/-1.95/0.13/0.61` 的近似 recovery；它既没有稳定质量，也仍是 mapping，路线退休。
2. 从现有 KV 反演 Parent hidden，再做 release tangent/JVP，KV-only 的三个必要 dense transform 下界已达
   Exact-All 的 `28.48%`；假设额外保存约三倍 Parent execution tape 的乐观窗口才约 `18.34%`。它不符合
   当前 persistent-state interface，不能进入 Design。
3. 历史 prefix moments 的闭合成本超过 `56%`，且 historical activation region 本身不稳定；request-side
   compactness 不能被偷换为对全部 historical queries 的 layerwise closure。

这些反例进一步约束下一步：不能靠 mapper、额外 source tape 或把单请求 reader 代数直接扩展成全历史
重放。

### 新的结构观察

同一 UID、五 edge 的纯 oracle 诊断显示，exact joint `Delta[K,V]` 沿 token/history 轴具有很强低秩结构：

- 每层独立 rank-8 oracle recovery 为
  `0.9989/0.9985/0.9928/0.9995/0.9999`；
- 只用 dependency-free layer-0 exact defect 的 rank-8 basis 投影所有上层 exact defect，recovery 为
  `0.9981/0.9732/0.9396/0.9302/0.8574`；
- Parent state 自己的 token basis 不能稳定承载 release defect。

因此值得检验的科学线索不是少量 token，而是 **support-dense、mode-compact 的 release drift**。它与
Insight 1 不矛盾：所有历史位置都可以参与，但共同变化只占少数稠密 history modes。

一个合法的单臂 rank-8 Current reduced replay 加 layer-0 shared-mode splice，在该 UID 五 edge 上得到
`0.8610/0.9173/0.9852/0.9473/0.9753`。但 related-work 审计发现，xKV 已经覆盖跨层共享的
`N x r` token basis 与 layer cores；ShadowKV、Palu、EigenAttention、MobiLoRA、ForkKV 等也分别覆盖
sequence-specific subspace、低秩 reader 或 base-plus-residual。因此这条路径即使数值好，也只能作为
generic compression control，不能成为 Design 1。

## 2026-09-03：Iteration 13 — 等分辨率 finite-release differential 的信号与成本否决

### 为什么继续研究双版本递推

把 Parent 与 Current 都按完全相同的预注册数值分辨率运行，并逐层传播两者的有限差分：

~~~text
Xc_l = Xp_l + D_l
Xp_{l+1} = Fp_l(Compress(Xp_l, r))
D_{l+1} = Fc_l(Compress(Xc_l, r)) - Xp_{l+1}
~~~

固定两臂各 `rank=4, oversample=4, power=1`，再从 approximate layer-0 `Delta[K,V]` 形成 rank-8
`U0`，UID `1930` 五 edge recovery 为：

~~~text
0.8701 / 0.9014 / 0.9850 / 0.8686 / 0.7059
~~~

这个信号比“两个压缩结果相减”更具体，因为 matched ablation 出现了三个可证伪现象：

- 同预算的 Current-only rank-4 为 `0.447/0.203/0.314/0.808/-0.394`，远低于对称双臂；
- 把总 rank 8 不对称分成 Parent/Current `2/6` 或 `3/5` 会严重失败，说明收益不是给 Current 更多容量；
- shared 与 independent Gaussian seed 几乎不改变结果，而强迫两臂共用一个 rank-4 data basis 会失败。

当前最合理的解释是：重要对象可能是 **同一数值分辨率下的两版本计算差分**；对称近似会抵消与 release
无关的 truncation bias，而不对称压缩或单边 Current 轨迹会把 numerical error 写进 migration delta。
这仍是待 population 验证的机制假设，不是已经冻结的 Insight 2。

同时必须保留一个不利事实：paired 五边均值约 `0.866`，而前一轮 single-arm rank-8 的五边均值约
`0.937`。两组还不是 formal matched comparison，但计算规模相近，现有数据不能支持“paired 已优于
generic compression”。paired rank4 相对 Current-only rank4 的增益只证明 equal-rank subtraction 值得
研究；只有后续 matched-compute control 显示 release-specific recurrence 的独立收益，才能把它提升为
设计机制。

代码已经把它实现为显式 base-plus-defect recurrence，并通过 16 个 CPU tests，包括 factorized native
attention 等价、paired recurrence 与两条等分辨率轨迹等价，以及 full-token-rank 时
`Exact Parent + paired defect = Exact Current` 的 exact-limit invariant。该 invariant 只证明算法语义，
不证明低 rank 质量或成本。

### 20% 成本门的硬裁决

完整逐算子 ledger 给出的双臂成本为 `1,206,907,124 FLOPs/user = 25.2952% Exact-All`。即使第六层只
生成最终所需 K/V、完全删除 dead Q/attention/gate/output，合法最低仍为
`1,041,218,120 = 21.8226%`，超过 20% cap `86,961,531 FLOPs/user`。单臂冻结 control 也为
`22.8028%`。

所以本轮不建立 32-user formal contract，不通过更换 FLOP 口径、事后降低 oversampling 或选择有利 edge
准入。当前双臂路径只保留为新的 structural lead：下一轮必须在不破坏“同分辨率差分/误差抵消”的前提
下消去一条历史 attention 轨迹。优先否证的具体问题是，exact Parent persistent cache 能否直接充当
matched control variate；若它只是普通 cache compression 的重新组合，或五 edge 信号消失，则立即退休。

## 2026-09-03：Iteration 14 — 14.15% static Parent-cache shortcut 的决定性反例

为检验是否能省掉第二条 historical trajectory，同时保留 paired error cancellation，本轮固定
Current `rank4/os4/power1`，对每层 exact Parent joint `[K,V]` 事后执行相同 rank-4 range finder，再按
同一个 rank-8 layer-0 `U0` 编译 signed cores。UID、edge、candidate 和 numerical schedule 均未改变。

五边结果为：

| path | v0→v1 | v1→v2 | v2→v3 | v3→v4 | v4→v5 | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current r4 − exact Parent | 0.5109 | 0.2680 | 0.3742 | 0.7407 | -0.1173 | 0.3553 |
| Current r4 − static matched Parent-cache r4 | 0.3858 | 0.2153 | 0.3467 | 0.7608 | -0.3754 | 0.2666 |
| Current r4 − Parent trajectory r4 | 0.8701 | 0.9014 | 0.9850 | 0.8686 | 0.7059 | 0.8662 |

static shortcut 的完整 prospective cost 为 `675,118,152 FLOPs/user = 14.1496% Exact-All`，成本通过，
但质量彻底失败。因此它明确退休，不通过提高 rank、power iteration 或更换 seed 修补。

这个反例把 paired lead 从“两个低秩 cache 相减”推进成一个更具体的 Transformer 机制：Parent 与
Current 的 approximation error 必须经过各自的 attention、gate、residual 和下一层 recompression，
才可能保持 trajectory correlation；对已经形成的 Parent K/V 做相同静态压缩不会产生这种误差相消。
所以省算不能发生在 trajectory 结束以后。

下一步检查的是一个语义等价、而非近似调参的内部省算机会。两臂的初始 range finder 目前先各自物化
完整 `embed_inputs`，再做四次 matrix application；但 HSTU input formation 在 dropout=0 时是线性算子：

~~~text
X = (item + behavior + time_features W_time^T) W_in^T.
~~~

因此可以用 `X@R` / `X^T@Q` 的 matrix-free application 直接形成完全相同的初始 factors，不物化
temporal projection 或 `N×H` in-projection output。若逐项 ledger 证明这使原 paired recurrence 严格
低于 20%，它只是解除成本阻塞的 executor component；论文机制仍必须由 trajectory-correlated
finite-release error cancellation 和 matched-compute functional gain承担。

## 2026-09-03：Iteration 15 — matrix-free input 解除 paired 机制的成本阻塞

原 paired KV-only 账本的 `21.8226%` 并不是 attention recurrence 的不可约下界。两臂都先物化
`N×H` temporal projection 与 post-`in_proj` history，再对这个 dense matrix 做 initial range finder；
但固定 range finder 实际只需要 `X@R` 与 `X^T@Q`。对当前 HSTU：

~~~text
X = (E_item + E_behavior + Phi W_time^T) W_in^T.
~~~

通过矩阵结合律，可以从 embedding lookup、`Phi` 和两组权重直接执行这些 operator applications。
这不改变 rank、oversampling、power iteration、Gaussian seed 或所得 factor 的数学定义，也不删除
Parent trajectory。

CPU tests 已验证 `X@R`、batched `X@R`、`X^T@Q` 及 power0/power1 factor reconstruction 与 dense
路径在 float32 tolerance 内等价，并禁止 helper 暗中调用 dense temporal/in-projection forward。固定
`rank4/os4/power1` 每臂完整计入两次 right apply、两次 transpose apply、两次 QR 和 small
Gram/eigh/rotation后，input-plus-initial-factor 从 `101,441,366` 降为 `18,033,494 FLOPs`。

替换双臂相同部分后：

~~~text
paired final-KV-only = 874,402,376 FLOPs/user
                     = 18.3264% Exact-All
20% margin           = 79,854,212.8 FLOPs/user = 1.6736 pp
~~~

每臂另有 32,768 次 sin/cos、16 次可 release-global 预计算的 frequency exp、1,536 Gaussian draws、
393,216 个 embedding lookup scalars，均单列而未冒充零成本。该结果只说明 paired mechanism 现在有一个
诚实的预算内 executor；matrix-free randomized range finding 是系统组件，不是 Insight 2 或论文创新。
下一门仍是 population-level error cancellation 与 matched single-arm comparison。

## 2026-09-03：Iteration 16 — early formation 信号与 rank-handoff 否证

在保持 Current arm、layer-0 basis、rank、seed 和 held-out odd-32 不变的条件下，固定
`d={1,3,5,6}` 截断 matched Parent trajectory。UID `1930` 五 edge 的 probability-gap recovery 均值为：

```text
d1=.528, d3=.831, d5=.868, d6=.866
```

`d1 -> d3` 的 `+30.3pp` 跃升说明 equal-resolution subtraction 需要穿过早期
attention--gate--residual blocks，不能在 layer 0 后立刻退化成静态 cache subtraction。`d3` 的严格成本
为 `16.9631%`；但它相对 `d6` 在 `v1->v2` 仍低 `10.39pp`，所以“固定三层即普适 formation boundary”
尚未成立。

为单独检验 upper gap 是否只是 Current capacity 不足，只执行一个事前固定的 handoff：前三层
Parent4+Current4，之后停止 Parent 并把 active rank 合并为 Current8。其均值只有 `.836`，相对 d3
仅 `+0.55pp`，且远低于 single-arm Current8 的 `.937`。原 ledger 为 `20.1084%`；与已验证的
matrix-free input component 组合后为 `16.6122%`。成本通过不能挽救机制失败，因此 handoff 和
“upper autonomous transport”均明确不接纳为 Design。

本轮只保留一个较窄、仍需 population 证伪的观察：paired numerical cancellation 的主要收益可能在
早期 contextual formation 中出现。下一步不再扫描 depth/rank，而检查这一 trajectory defect 到达
S4 functional response 后是否出现 single-arm 没有的因果优势；并独立审计 exact Parent K/V 能否作为
逐层 trajectory checkpoint，让算法只传播 finite-release defect。

## 2026-09-03：Iteration 17 — paired trajectory 到 S4 后数值过门、创新门失败

固定 paired r4/r4、P8 history probes 与 held-out odd-32，把同一 approximate trajectory 从 shared-`U0`
K/V splice 改为 S4 signed affine moments。五边 probability recovery 为：

```text
paired KV       .8703/.9015/.9849/.8685/.7059, mean=.8662
paired S4       .8725/.9212/.9487/.9335/.8226, mean=.8997
single-r8 S4    .9137/.9983/.8544/.9814/.9367, mean=.9369
full Exact S4   .9997/.9998/.9960/1.0000/.9992, mean=.9990
```

matrix-free input 后 paired S4 完整构造为 `938,047,624 FLOPs/user = 19.6603%`，数值与成本都达到探索
目标；但它只在 1/5 edge 胜过 single-r8 functional control，且只是 paired compression 接已有
ELU+1 moment compiler。global 38,016-scalar moment 也不能删除任意 eviction row；segment ledger 与
single-arm 完全同构，没有 paired-specific persistence closure。故这条路线**因为机制和 prior-art
边界失败而退休**，不能用接近 90% 的数字包装成 Design。

## 2026-09-03：Iteration 18 — Parent KV checkpoint 信息可逆，但 delta execution 成本下界失败

对 frozen v0..v5 的 36 个 Medium block 做 model-only audit，joint `[K,V]` projection 全部 rank 192，
condition number `15.92--35.23`。只要 Parent cache 额外保存每 token/layer 一个 RMS denominator，便可
无学习恢复 exact pre-block state；metadata 只有 `6,144` scalars，即 Parent KV 的 `0.2604%`。这修正了
“RMS scale ambiguity 是原则障碍”的旧猜测。

真正障碍是 K/V 没有保存 historical query 与 HSTU gate coordinates。即使采用病态但最便宜的 K-only
inverse，六层 checkpoint decode 加前五层 Q/gate 两组 dense transforms 已为
`1,207,959,552 = 25.3173% Exact-All`；稳定 joint decoder 为 `34.8113%`，均尚未计算任何 Current
defect 或 attention delta。joint-KV→Q/G exact maps 也全 rank，rank@90 仍约 69--76。故
Parent-anchored finite-difference execution 是有科学语义的路线，但当前 KV-only/20% interface 下被
决定性否决，不运行 UID/GPU。

## 2026-09-03：Iteration 19 — K/V finite interaction 与 common-projection response control 否证

在 Current coherent query path 上做 exact finite endpoint decomposition：

```text
R_CC - R_PP = (R_CP-R_PP) + (R_PC-R_PP)
              + (R_CC-R_CP-R_PC+R_PP).
```

K-only/V-only 五边均值仅 `.518/.563`；去掉 finite K×V interaction 在四边可恢复 `.847--.917`，但
`v0->v1` 为 `-1.107`。interaction/joint norm 随 depth 约从 `.130` 增至 `.348`，却没有稳定的 K/V
cancellation law。因此完整 finite interaction 是安全语义的一部分，但不是可单独迁移的 compact object。

最后测试一个不使用 ELU moments 的 native-response control variate：在同一 Current-r8 per-layer span
内分别读取 projected Current 与 projected exact Parent，response 做差后加回完整 Parent response。
它有 full-rank exact limit、成本 `19.4941%`，但五边为
`.2722/.9337/.8316/.9878/.9362`，mean `.7923`；同 runner generic single-r8 为 `.9364`，shared-layer0
control 为 `.9372`。v0 严重过校正，故 common-projection response 路线退休。

当前硬结论是：在 KV-only 与 20% cap 下，所有已测合法 Current information source 最终都退化为
generic reduced Current replay。下一轮只允许检验一种新的 state organization：在 Transformer 递推中
分别保持 Parent base 与 finite-release defect，让 mode budget 直接服务 defect，而非再压缩完整 Current
absolute state；若它仍被 single-r8 或 base-plus-delta prior art 支配，则转向 migration-ready state
co-design，而不再扫描 rank/probe/layer。

## 2026-09-03：Iteration 20 — probe-free affine invariant 的 representation-level 强反例

为判断 S4 functional state 是否能完全脱离 query probes，固定 strongest single Current-r8 replay，
对每层/head 的全部历史位置直接保存：

```text
B = sum_i V_i,   M = sum_i K_i^T V_i,
delta r(q) = delta B + q delta M.
```

该路径不扫 feature/probe/mask，matrix-free 完整成本为 `885,299,968 = 18.5548% Exact-All`。但五边
recovery 为 `-.108/.717/-12.995/.925/-1.827`，mean `-2.658`。使用 full Current Exact 构造同一
representation 的 oracle 也为 `-.057/.733/-13.072/.920/-1.907`，说明失败来自 functional object，
不是 rank8 constructor。

正式 attention-cone 结果中 negative-response norm 约 4%，但这不约束 Current-minus-Parent signed
residual：把少量 negative-logit rows 错误延拓到 `1+qk` 支路，会产生大版本差并经 gate/residual 放大。
因此冻结新的负边界：**query-dependent activation geometry 是 functional migration object 的必要部分；
不能用全局 affine/fast-weight moments 删除。** 此外该状态与标准 linear-attention outer-product memory
同构，即使数值通过也不足以承担创新。本 family 不再增加 polynomial/random features。

## 2026-09-03：Iteration 21 — paired native-response placement 有增益但仍被 single-r8 支配

为排除 paired r4/r4 只是被 shared-`U0` 或 P8 moment 放错边界，固定保存两臂各自的 per-layer factors，
让 Current query 原生读取：

```text
exact Parent response + native response(Current-r4) - native response(Parent-r4).
```

该路径不使用 `U0`、probe、moment 或 mapper。full-token-rank 恢复 Exact Current，`P=C` 时 correction
严格为零；row eviction 与已有 exact Current suffix 的 segment read 均有单测。UID1930 五边为
`.8725/.9214/.9549/.9335/.8236`，mean `.9012`：比 paired K/V 的 `.8662` 提高 `3.49pp`，说明完整
activation 后做差确实过滤部分合成误差；但只在 1/5 edge 胜 single-r8 reduced cache（mean `.9364`），
只在 2/5 edge 胜 single-r8 shared-`U0`（mean `.9372`）。

matrix-free constructor 为 `872,238,088 = 18.2810%`，两臂 sidecar 为 `67,584` scalars（FP32
`270,336B`）；每层两次 factor read、每 query 增量 `1,218,816` FLOPs。成本通过但质量/创新硬门失败，
故 **NO-GO / RETIRE**，不扩用户。新的负边界是：response placement 不能补回 reduced trajectory 未保留
的 Current information；下一轮不得继续扫相同 trajectory 的 rank/basis/subtraction boundary。

## 2026-09-03：Iteration 22 — migration-ready source tape 仍无法跨过 native attention floor

作为 KV-only 失败后的 model--system fallback，本轮允许 Parent cache producer 在原始 Full 生成时保存
execution cut，使 constructor 沿 exact Parent trajectory 只推进 finite defect。full-rank/native 极限下，
`D_{l+1}=D_l+U_l^C-U_l^P` 可逐层恢复 finite Current computation graph；这是真正的非 mapping 语义，
也不等于一阶 JVP。

信息最小性审计表明，generic nonzero-gate cut 需 `X0 + 5U + 5Q + 5G + 5A = 21` 个 `N x H`
field；但 `O=U/SiLU(G)` 在 gate=0 不可识别、近零不稳定。固定布局必须再存五层 post-output `O`，共
26 fields，即现有 9 MiB K/V 之外再加 `19.5 MiB/user`（`+216.7%`）；30k users 额外
`571.29 GiB`。只存 O 再反解 A 也不可靠，真实 Parent output-projection condition number 为
`970--37,149`。

更决定性的是，Parent `A/O/U` 全免费保存在 tape 后，前五层 exact Current historical response 仍各需
一次 native QK 和 weighted-V：总计 `2,015,232,000 FLOPs/user = 42.2367% Exact-All`，尚未计
input、projection、normalization、defect compression、gate 或 sidecar。低于 20% 只能再次删除
query-dependent activation geometry、使用已被 full-Exact oracle 否证的 positive-affine moments。
因此本轮在 cost gate 停止，不运行 UID/GPU；source tape 明确超出当前 KV-only contract，也没有越过
single-C8 `17.8953%/.937` 硬对照。

## 2026-09-03：Iteration 23 — defect-coordinate closure 不能替代绝对 Current information

按 Iteration 19 预注册的唯一 state organization，逐层分别保持 Parent base rank 2 与
Parent-to-Current defect rank 4；Current 输入是两者 factor 的精确和、effective rank 6：

```text
B_next = Compress2(F_parent(B))
D_next = Compress4(F_current(B + D) - F_parent(B))
```

初始 `B` 与 `D` 分别由 matrix-free Parent input 和 `(Current-Parent)` input operator 形成；末层只形成
K/V。primary reader 不使用 P8/moment，而执行 exact Parent native response 加 approximate Current 与
approximate Parent 的 native-response difference。固定 UID1930 五边 probability recovery 为
`.554/.690/.549/.821/-.076`，mean `.508`，只有 `1/5 >= .80`。matched ordinary absolute P2/C6 更差，
为 `-1.346/-1.125/-4.040/-.044/-2.507`；这说明 defect coordinates 的确抑制了一部分不匹配误差，
但远不足以形成迁移方法。

保守、显式 materialization ledger 为 `880,621,524 = 18.4567% Exact-All`，factor sidecar 为
`67,584` FP32 scalars。full-rank exact-limit、matrix-free input identity 与 native reader exact-limit 已有
单测。由于质量 gate 决定性失败，且 base-plus-defect/rank allocation 与 ForkKV/MobiLoRA 等 prior-art
边界重叠，本固定 B2+D4 路线 **NO-GO / RETIRE**，不调 rank、不扩 32 users。

同 runner 对 paired P4/C4 native-response 的复核只作 control；其权威结果、成本和 NO-GO 裁决统一引用
`paired_native_response_preflight.md`：`.9012` 仍被 single-r8 `.9364/.9372` 支配。本轮因此关闭“继续
改变同一 reduced trajectory 的坐标分配或 subtraction placement”这一族，而不是把一次 matched-control
失败保留成新 Design lead。

## 2026-09-03：Iteration 24 — producer-state / reader-version 近交换是正 oracle，但没有 constructor

固定四条 exact path (F(r,p)=Reader_r(KV_p,q_r))，其中 reader 与 cache producer 分别取 Parent/
Current；每个 reader 使用自己的 query embedding、Q/gate/residual/readout。直接检验：

```text
F(C,C) ~= F(C,P) + F(P,C) - F(P,P).
```

UID1930 五边 raw probability-gap recovery 为
`.9657/.7651/.9333/.9427/.8087`，mean `.8831`、4/5 edge ≥ `.80`；两个 reader 测到的
producer-state score effect cosine 为 `.99930--1.00000`。S4 layer0 的 candidate-centered L2 recovery
也为 `.857/.915/.806/.930/.845`。因此保留一条正结构观察：**在相邻 continual release 中，producer
state 的有限影响对 reader update 近似一阶不变，reader 与 state update 的 mixed finite difference
通常较小。**

但 raw score 主要受 candidate-shared calibration shift 支配；center 后的 final-score recovery 为
`-4.697/.884/.713/.805/.402`，只有 2/5 edge 过 `.80`。S4 跨层也不单调（layer2 一边 `-.695`、
layer3 五边 `.234--.788`），不能据此冻结单层 functional boundary。

更根本地，reverse path `F(P,C)` 读取 Exact Current upper-layer K/V；按公式执行还需每 candidate 跑
Parent reader 并在 score 端混合。把它 probe-compile 会退化成 mapping，保存 response operator 又回到
已审计但缺 Current constructor 的 signed/native-response state；用 approximate Current cache 替换则只是
generic compression 加 control variate。因此 oracle observation 通过而 **Design NO-GO**：不开 formal
用户，不把 `.8831` 包装成合法恢复率，等待真正 no-target 的 finite-state-effect source。

## 2026-09-03：Iteration 25 — exact source certificate 不能闭合 release trajectory

本轮把 exact Parent persistent K/V 当作 source execution certificate，而不是 Current tensor target。
joint Parent K/V projection 满列秩，因此可在四个固定 DEIM rows 恢复 exact Parent normalized query，
读取完整 exact causal Parent prefix，并得到原生 attention/gate block update。随后只检验两个冻结构造：

1. 将 exact-minus-reduced Parent update residual 插值后同时加入 Parent/Current 两臂；
2. 用 `N_P_exact + (Nhat_C-Nhat_P)` 与 `KV_P_exact + (KVhat_C-KVhat_P)` 形成 anchored logical
   Current endpoint，只将它与 paired approximate update-difference 的 residual 加入 Current arm。

两者均不读取 Current Exact、candidate 或 label；均有 zero-release 与 full-token-rank exact limit。显式
ledger 分别为 `905,666,998 = 18.9816%` 和 `929,090,998 = 19.4726% Exact-All`。但固定 UID1930
五边 recovery 仅为：

```text
absolute source residual:  -.003/.729/.896/.911/.686, mean .644
finite release defect:      .212/.672/.667/.945/.815, mean .662
matched paired-native:      .872/.917/.955/.933/.824, mean .900
single Current r8 controls: .913/.997/.852/.981/.937, mean .936
                            .861/.914/.985/.947/.975, mean .936
```

absolute correction 会在下一层经两版不同的 RMSNorm、query--key activation、gate 与参数产生非零
cross-version response；finite-defect residual 避免了这个语义错误，却仍不能由四个 point tests 决定
distributed/query-dependent release defect。它只有 2/5 edge 达 `.80`，对两个 single-r8 都是 0/5 胜。

独立 novelty gate 同样失败：`Q(Q[I])^-1 residual[I]` 是标准 DEIM/sampled-residual hyper-reduction，
exact/approximate attention residual 又与 control-variate 工作直接相邻；四个 test rows 并非 Transformer
因果图特有的守恒量或闭合泛函。因此本轮冻结为数值与 novelty **双重 NO-GO / RETIRE**，不调
rank/pivot/lift、不扩用户。后续若仍使用 sparse residual test，必须先给出 DEIM 无法还原且直接控制
future reader response 的 Transformer-only causal invariant；否则当前最诚实结论仍是尚无论文级 Design。

## 2026-09-03：Iteration 26 — 当前 KV-only 接口总裁决

本轮不再增加另一个 rank/probe 变体，而把 Iteration 21--25 后仍可能提供新 Current information 的出口
逐项审计到底。全部过程继续只研究单个相邻 edge；没有读取 label、confirmation `[512,3000)`，没有
训练或启动新的 formal GPU job。

首先，activation interaction graph 的 Parent/Current branch agreement 为 `.8732`，但 crossing-only
coherent causal intervention 五边为 `.7778/.0635/.0342/.0519/.1276`，mean 约 `.211`。不读取 Current
target 的 recursive graph replay虽有 `.833` mean，却需要先发现 Current historical QK graph；五层 QK
单项为 `21.1183%`，QK+AV 为 `42.2367%`。所以 stable topology 不是 stable functional response，主体是
same-region continuous deformation。

随后四类新的信息源均在运行 population 前关闭：

1. **Natural causal suffix**：对任意合法初始 cache，Current append 都满足
   `A_C(A_C(S,b1),b2)=A_C(S,b1||b2)`；错误 Parent lineage 同样自洽。因此 suffix 只有 query coverage，
   没有 Current target-state information。最直接的 Tail-128 已为 `-.0876` mean recovery、约
   `18.28%`，不新增 runner。
2. **Release circuit / heads**：attention heads 只在单层 QK/AV 内独立；dense `W_O`、full-width gate、
   residual 与下一层 norm 使任意 upper-head exact closure重新包含全部 lower heads。第三个 cache layer
   的 single-head Current closure 已为 `35.0915%`；唯一 `<20%` 窗口退化成 early head-wise K/V
   replacement，与 locality/selective recomputation prior art 重合。
3. **Release algebra**：30/30 layer-edge 的最佳 head permutation 均为 identity，但 QK/VO gauge-invariant
   mismatch 仍为 `5.21%--11.89%`，gauge tangent只解释约 `5%--7%` movement；30 个 block matrix 的
   `Delta W` exact numerical rank 为 `180--192`。model-native layer-0 query witness 又在 30/30 heads
   满 rank。gauge、structured update、secant/commutator、native-query quotient 与 finite moments 五类
   exact shortcut全部关闭；截断后分别退化为 LoRA/JVP、mapping 或 generic compression。
4. **Causal state ports**：RMT/Block-Recurrent/RetNet 已覆盖 causal memory/ports 本身。更重要的是，
   Parent-sufficient state 不自动 Current-sufficient：port-only exact transform 要求 Parent port 的 fibers
   被 Current producer保持；append 还要求 `T o F_P = F_C o T`，eviction 需要独立 delete algebra。
   当前 checkpoint无法事后创造这些性质，“memory token + upgrader”仍是已有方法组合。

加上前述 paired-native `.9012 @ 18.2810%`、defect-coordinate `.508 @ 18.4567%`、source finite-defect
`.662 @ 19.4726%` 与 strongest generic single-C8 `.9372 @ 17.8953%`，当前证据形成统一边界：

> **Finite-query functional compactness is reader-conditional, not generator-closed.** Transformer reader
> 在 query-dependent aggregation 后会把分布式 mismatch 压紧，但 Parent K/V 并不因此包含一个可在
> query 前、无 Current target、低于 20% 生成的 compact Current quotient。

这是一条可信的阶段性 Transformer insight，却不是可执行 Design。当前 `v0..v5 + KV-only + no new
training + <20%` 下没有 paper-worthy active candidate；single-C8 只作 hard numerical control，PRO 只作
历史 baseline。停止新的 UID/rank/probe/layer/head/suffix/operator 扫描，不建立 formal/confirmation
合同。重新开启前必须先提出 generic replay 无法还原的 Current-information source 或 finite-release
causal law；co-design 也只有先有非 regression 的 release homomorphism 与 delete/lineage law 才有资格
申请新的 Small training authorization。完整裁决见 `current_kv_only_interface_adjudication.md`。

## 2026-09-03：Iteration 27 — 专家讨论统一稿与措辞收敛

本轮没有新增实验、读取 UID/label/confirmation、启动训练或修改任何 sealed evidence，而是把 Motivation、
Insight 1、Iteration 1--26 的正负结果和 Design 空缺整理为单篇
[`docs/insight2_design1_expert_brief.md`](../../../docs/insight2_design1_expert_brief.md)。统一稿按
`D512/C32/P1/model-only` 标记证据强度，明确 `.9372 @ 17.8953%` 只是固定 UID 的 generic hard control，
不能当作 population 或 paper result。

本轮进一步收敛了 S4 的称谓：它是 **the earliest observed response-contraction stage in the current
HSTU reader**，不是已经找到的 persistent migration boundary。当前最稳的候选 Insight 2 表述为：

> **The reader contracts the mismatch, but that contraction is not itself a migratable state.**

对应 Design 结论保持不变：`READ -> ESTIMATE -> PERSIST -> INJECT/UPDATE` 只是准入合同，不是具体方法。
若要重新开启正向 Design，必须先提出 generic replay 无法还原的 Current-information source，或带有
release/append/delete 语义的 Transformer causal law；在此之前不启动新的 formal、confirmation 或训练。

## 2026-09-03：Iteration 28 — 专家裁决后的 Migration Sketch Design 冻结

专家讨论把上一轮“Design 空缺”进一步收敛为明确方向：真正需要改变的不是再从 Parent KV 中寻找一个
post-hoc constructor，而是让 state producer 在普通 K/V 生成时主动写入一份小型、跨 release 稳定的
用户摘要。本文据此将 scoped Insight 2 与 Design 1 统一为：

> **Query-conditioned aggregation 会把 distributed stale-state mismatch 收缩成紧凑的用户级功能差异；
> conventional KV cache 的缺口，是没有提前保存一份可供未来 reader 解释并随 lineage 演化的摘要。**

本轮没有新增 UID、label、confirmation、GPU job 或模型训练，也没有修改任何 sealed raw/result。完成的
工作是把 Design 明确成可证伪的 reference mechanism，并写入
[`docs/insight2_design1_expert_brief.md`](../../../docs/insight2_design1_expert_brief.md)：

1. release-family 内冻结的 canonical sketch writer，在 Current 未知时随 Parent K/V 一起写入 state；
2. 每个版本只用本版本 Full response 自认证并冻结少量 per-layer native K/V slots，不拟合
   Parent→Current paired target；
3. Current query 在 S4 分别读取 Current view 与实际 producer-lineage view，将 response difference 加到
   ordinary Reuse context；
4. chronological segments 带 producer tag。Current-produced segment 的两个 view 相同，correction
   自动为零；Parent segment eviction 时两 view 同步删除；
5. reference steady-state `M=32` 账本中，单个 Current view decode 约为 `0.5934% Exact-All`，两 view
   加 canonical state 约 `600 KiB/user`，每请求两次 sketch read 约为 prefix QK+AV 的 `6.25%`；cutover
   split 的保守 `M=34` 上界分别为 `0.6305%`、`637.5 KiB`、`6.64%`。这些是解析 accounting，不是
   方法 runtime 或 recovery 结果。

本轮同时冻结创新门：slot pooling、decoder、compression 和 subtraction 各自都不构成贡献；必须由
`stable producer-written state + version-local certification + lineage-paired native interpretation +
append/eviction semantics` 共同成立，
并在 matched compute 下胜 Current-only sketch、memory-token/mapper、frozen-producer 和 generic C8。
否则方法降级为 generic compression，不靠命名保留创新主张。

当前已有 `95.34%/99.46%` S4 数字仍只属于 oracle Insight 证据。Migration Sketch 在 `<20%` 下的
functional recovery、rolling persistence 和 task quality 全部保持 `pending`。下一步是先冻结 prospective
`BUILD/MIGRATE/READ/APPEND/DROP` 合同与 toy exactness，再申请独立 C32 canary；现有
`[512,3000)` confirmation 继续 unread，新 Small/Medium 训练仍需独立合同与用户显式授权。

## 2026-09-03：Iteration 29 — common-mode certificate 与 clean state writer

本轮没有新增 UID、label、confirmation、GPU job 或模型训练，也没有修改 sealed raw/result。对 Iteration
28 做 paper-level red-team 后发现两个结构问题：第一，两个 same-version decoder 相减仍可能只是普通
memory distillation；第二，若 Current append 从 corrected mixed-lineage request trajectory 生成 K/V，有限
correction error 会污染 descendant cache，`producer=c` 不能使状态误差自动归零。

因此，Iteration 29 supersede Iteration 28 的**机制、append 与成本口径**，但不回写其历史记录。当前
reference mechanism 收紧为：

1. foundation 用目标 edge 之外的 pseudo releases 训练
   $\mathcal L_{cm}=\|\epsilon_a-\epsilon_b\|^2$；每个真实版本再对预先冻结的 reference-release residual
   训练 $\mathcal L_{anchor}$。目标 edge 不做 per-user Parent→Current fit，但必须公开 foundation 使用了
   paired pseudo-release compatibility supervision；
2. per-segment certificate、random segment-mask/drop consistency 和 same-version shadow rollout 分别约束
   segment identifiability、任意 lineage subset 与 query drift；
3. native interpreter $D_v$ 进入正式 state writer，而不是 release-only auxiliary head：

   \[
   C_{v,i}^{\mathrm{Full}}
   =G_v(r_i,S_{<i})
   =C_{v,i}^{\mathrm{append}}.
   \]

   $G_v$ 不读取 mixed-lineage serving hidden，由此切断 approximation 向新 K/V 的传播；
4. 每个 segment 保存原 entry writer state 与 compact raw witness。partial Parent boundary drop 从原 entry
   replay 原 atoms、聚合 surviving suffix，并原子刷新 $D_p(S_g)$ 与 $D_c(S_g)$；Current boundary 只刷新
   $D_c(S_g)$；
5. 固定 query/cache 下只有 local identity
   $\widetilde A-A^*=\epsilon_c-\epsilon_p$。真实 rollout 还要单列 query-trajectory drift；ordinary
   trajectory-writer control 还会多出 descendant state-propagation error。

创新口径不再是
`stable state + decoder + subtraction + segmentation` 的组件组合，而是一个可证伪命题：**在 state
creation 时把有限摘要训练成跨版本 certificate error 的共同部分，使 native response difference 相消；
并让 Full/append 共用同一 state producer，避免相消误差被重新写回。** 核心 matched controls 为
`lambda_cm=lambda_a=0`、Current-only、auxiliary-$D_v$、ordinary trajectory writer、memory token、mapper
与 generic C8。若这些对照解释收益，方法降级为 generic memory/compression。

成本账本同步修正：native view 输入为 $H+2$，因此 one-view decode 是 $4LM(H+2)H$。`M=32/34` 分别为
`28,606,464 / 30,394,368 FLOPs = 0.5996% / 0.6370%` sealed legacy Exact-All reference；包含 paired
views、canonical slots、every-segment entry states、contextual-atom sums、global live state 与 boundary
witness 示意下界的 subtotal 为 `641 / 681 KiB/user`。partial Parent drop 上界为
$B C_\phi+3,575,808$ FLOPs 加 witness I/O、约 36 KiB view 写；Current boundary 为
$B C_\phi+1,787,904$ 加 I/O、约 18 KiB view 写。view-only Parent 项等于 `0.07494%` legacy reference。
prospective contract 必须重封 $\mathrm{ExactAll}^{G}$。
BUILD、$G_v$、实际 event schema/allocator、I/O 与 runtime 仍为 pending。

证据边界同时收紧：当前 legacy-HSTU/seed17/Medium ranking 的 S4 `95.34%/99.46%` 只证明 oracle
functional contraction；`v0..v5` 没有 $G_v$，legacy retrospective fitting 不能证明新 Design。最终方法
需要 prospective Parent-before-Current pair、migration-aware Full vs unconstrained Full admission、clean
rolling endpoint 与上述 matched controls；该 pair 还必须先独立封存 Full$^G$ vs Reuse$^G$ gap，near-zero
gap 按 No-op 处理，不能借用 legacy denominator。任何新训练仍需 prospective contract、focused canary、资源
估计和用户显式授权；本轮没有发起实验。
