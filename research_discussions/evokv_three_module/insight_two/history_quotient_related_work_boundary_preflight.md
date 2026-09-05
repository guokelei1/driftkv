# History-mode release differential：related-work 与 novelty boundary 预检

日期：2026-09-03
状态：**独立文献审计；未运行 GPU、未建立或修改合同、不是 Design 1 冻结稿**

## 1. 裁决摘要

当前候选可以写成：对一个用户长度为 `N` 的历史，以 rank `r=8` 做 token/history-axis
reduced Current prefill；从合法的 approximate layer-0 release defect

\[
D_0=[K^{C,red}_0-K^P_0\mid V^{C,red}_0-V^P_0]
\]

形成用户级正交 basis `U_0 in R^(N x r)`，并为每层保存

\[
C^K_l=U_0^T(K^{C,red}_l-K^P_l),\qquad
C^V_l=U_0^T(V^{C,red}_l-V^P_l).
\]

reader 消费

\[
\widehat K_l=K^P_l+U_0C^K_l,\qquad
\widehat V_l=V^P_l+U_0C^V_l,
\]

而不把完整 migrated K/V 持久化。

文献审计后的结论是：

1. **上述表面组合还不足以成为可信 novelty claim。** sequence-axis projection、prompt-specific
   history basis、跨层共享 token basis、low-rank cache core、base/anchor 加 residual/delta、以及 native
   low-rank reader 都已有直接先例。
2. 最危险的表示撞车是 [xKV](https://arxiv.org/abs/2503.18893)。它已经把多层 K/V 横向拼接，保存
   一个跨层共享的 `N x r` token basis 和每层 `r x d` core；这与“固定用户 `U_0` + 上层 cores”在
   代数形式上近乎同构。
3. [DroidSpeak](https://arxiv.org/abs/2411.02820) 已明确研究同架构、不同 fine-tuned model/version
   对相同 context 的 KV reuse；因此不能声称首次提出跨版本 KV sharing 或旧模型 history 被新模型
   消费的问题。
4. [MobiLoRA](https://aclanthology.org/2025.acl-long.1140/) 已保存 anchor KV 与跨 adapter delta KV；
   [ForkKV](https://arxiv.org/abs/2604.06370) 已把 cache 拆成共享 base 与轻量 residual，并设计
   ResidualAttention 在 reader 中组合两者。因此“Parent + signed delta”和“不物化完整 cache 的
   native read”也不能单独成为贡献。
5. 当前路线唯一可能守住的核心，不是 low rank 或 `U_0`，而是：**只从 dependency-free layer-0
   release defect 取得 history basis，然后让 Parent 与 Current 两版 Transformer 在同一个 history
   quotient 中进行有因果依赖的 finite-release differential replay；上层 coefficient 由这条递推产生，
   而不是读取 Current Exact、学习 Parent-to-Current mapper，或先独立压缩一份 Current cache。**

这项组合在本次检索到的原始论文中没有直接等价物，但这不是法律或学术意义上的新颖性保证。更重要的
是，仓库当前的 single-arm Current reduced replay 若到最后才读取 Parent 并做 subtraction，仍然可以被
reviewer 合理描述为“generic Current compression + xKV-style residual encoding”。它应当作为 control，
不能直接冻结为 Design 1。

## 2. 先固定比较坐标：不同论文的 low-rank 轴并不相同

令普通 cache 为 `X in R^(N x d)`。至少有三类不能混写的低秩对象：

- **history/token axis**：`X ~= U C`，`U in R^(N x r)`。`U` 描述哪些稠密历史模式共同变化。当前
  `U_0`、Linformer 的 sequence projection、ShadowKV 和 xKV 属于或触及这一类。
- **feature/head axis**：`X ~= H B`，或缓存 `X U_f`，其中 `U_f in R^(d x r)`。Palu、EigenAttention、
  OjaKV 和 Gated Subspace Inference 主要属于这一类。
- **parameter delta**：`W_C-W_P ~= A B`。LoRA、DeltaZip 等处理模型参数或参数差分；它不自动推出
  activation/cache delta 沿 history axis 低秩。

轴不同意味着这些方法不等价，但不能据此忽略 prior art。当前候选在 history axis 上最接近
ShadowKV/xKV；其 base-plus-residual layout 又与 MobiLoRA/ForkKV 接近。论文必须把贡献放在跨版本
causal construction，而不是“我们与 feature-axis compression 不同”。

另外，`signed core` 不是区别点。SVD/factorization 的 core 本来就是任意实数；而且
`U C=(U R)(R^{-1}C)` 存在 basis gauge ambiguity。正负号只是 residual representation 的必要性质，
不是研究发现。

## 3. 指定 KV/activation compression prior art

| 工作 | 已有机制 | 与当前候选重叠 | 仍然不同的部分 | 对 claim 的约束 |
|---|---|---|---|---|
| [Linformer](https://arxiv.org/abs/2006.04768) | 以 `E,F in R^(N x k)` 沿 sequence axis 投影 K/V；还实验了 head、K/V 和全层共享同一个 projection | token-axis reduction；一个 projection 跨层/K/V 共享；低维 reader | 它是训练后的单版 attention 架构，projection 通常 model-global，不保留 exact Parent complement，也不处理 release delta | 不能声称 sequence-axis K/V projection、layerwise shared projector 或 linear-complexity low-rank attention 是新点 |
| [Palu](https://openreview.net/forum?id=LWMS4pk2vK) | 分解 K/V projection weights，缓存低维 latent，再重构或把 reconstruction factor 融入后续算子；含 group-head decomposition 和 rank allocation | cache latent/core；低秩 projection；native reconstruction/fusion | feature axis、单版模型、model-global basis；没有 persistent Parent 或版本差分 | 不能声称低秩 KV latent、projection-weight factorization、core reconstruction、rank allocation 或 kernel fusion；若只对 `Delta W_K/V` 套 Palu/LoRA，更不是新机制 |
| [EigenAttention](https://aclanthology.org/2024.findings-emnlp.899/) | 从 calibration Q/K/V activation 取 per-layer/head feature basis，并修改 projection 使 attention 在低维空间执行 | activation-derived basis、低维 cache、reader 原生消费 | 静态 dataset-level feature basis；单版 cache；不迁移旧状态 | 不能声称 activation-calibrated subspace 或 low-rank-space attention；“离线 basis + 用户 coefficient”属于已有范式 |
| [OjaKV](https://aclanthology.org/2026.findings-acl.494/) | prefill/decode 中用 Oja rule 在线更新 context-aware K/V feature basis，并混合保存 full-rank 高误差 token | user/context-adaptive basis；append 后更新；低秩 cache 与 native/reconstructed read | feature axis，basis 从当前产生的 K/V 更新；不是固定 layer-0 history quotient，也没有 Parent-to-Current delta | 不能声称用户自适应 subspace、online PCA、低频 refresh 或 hybrid full/low-rank storage |
| [ShadowKV](https://proceedings.mlr.press/v267/sun25b.html) | 对每个 prompt 的 pre-RoPE key 在线 SVD，保存 `A in R^(N x r)` 与 head-specific core；发现同一 sequence 与 continuation 共享 subspace | sequence/user-specific history factor；prefill-derived basis；continuation reuse；按需重构 | 通常逐层分解已完整产生的单版 key，value 主要 offload；没有跨版本 defect 或 exact Parent base | 不能声称每用户 history basis、同一用户后续请求复用 basis、prefill SVD 或 sparse row reconstruction |
| [xKV](https://arxiv.org/abs/2503.18893) | 对层组 cache 横向拼接后 SVD：`[X_l1,...,X_lW] ~= A[B_l1,...,B_lW]`，其中 `A in R^(N x r)` 是跨层共享 token basis | **与一个 `U_0` 跨层共享、每层保存 core 几乎同构**；还用 CKA 证明 left singular/token geometry 跨层对齐 | xKV 读取已经形成的单版 full Current caches；basis 由目标层组联合决定；目标是压缩而非 release migration | 不能声称跨层 dominant left singular vectors 对齐、共享 token basis、layer-specific signed cores。只把 xKV 用在 exact `Current-Parent` 上仍只是 oracle/control |
| [Gated Subspace Inference](https://arxiv.org/abs/2605.03109) | 以 activation feature basis 和 weight image 加速 linear maps；depth cascade 只在 layer 0 做 full SVD，后层继承并少量修正 | “layer-0 seed 向深层传播”“跨层 coherent subspace”“小 residual correction”的叙事 | feature axis、单版 inference、目标是 weight bandwidth；没有 KV migration 或两版差分。当前仅为 arXiv v1 | 不能把“layer-0 subspace 穿过 Transformer depth”单独写成 Insight 2；必须限定为 release differential 的 history quotient |

最关键的阻断是 xKV，而不是 Palu。Palu/EigenAttention/OjaKV 说明低秩 cache reader 和动态 subspace 已
经成熟；xKV 则直接覆盖了当前表示形式。`joint K/V`、把 window 扩到所有层、basis 正交化，或把 rank
固定为 8，都只是 factorization 范围和超参数变化，不足以单独产生论文创新。

## 4. 跨模型、跨 adapter 与 cache residual prior art

### 4.1 问题本身已有直接先例

[DroidSpeak](https://arxiv.org/abs/2411.02820) 的问题设定已经包括：两个模型权重不同但架构相同，
updated chatbot model 读取 older model 曾处理过的同一 history，并通过离线 profiling 选择 critical
layers 重算、其余层复用。它甚至报告 critical layer identity 对同一 model pair 跨输入相对稳定。

因此 EvoKV 仍可贡献 recommendation-specific evidence 和不同机制，但以下话术不可用：

- “首次发现模型升级使 KV cache 失效”；
- “首次在不同模型版本间复用 KV”；
- “首次用少量新版本计算迁移旧 cache”；
- “首次研究相同架构、相邻版本的 state compatibility”。

Insight 1 可以把 DroidSpeak 的 layer-local premise 作为重要比较边界：在 HSTU persistent recommendation
history 上，critical-layer/local splice 是否因 causal closure 和 workload 变化而不稳定。该对比必须由
实验支持，不能通过改名回避 prior art。

### 4.2 mapper/translator 已经是一条拥挤路线

[Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893) 使用 target Current Exact calibration
pairs 拟合 per-head closed-form ridge mapper，并选择多个 source layers；
[Latent Cache Flow](https://openreview.net/pdf?id=PvnR2LDCOs) 使用 learned low-dimensional cache channel
和 receiver residual update；[Mixture-of-Translators](https://arxiv.org/abs/2607.28979) 进一步使用多个
translator 与 target-side correction。这些工作已经覆盖 source-to-target linear/nonlinear mapping、
cross-layer source selection、low-dimensional cache channel 和 learned residual update。

所以，如果 `C_l` 来自下列任一过程，EvoKV 都应明确降级为 mapper baseline，而不是 Design：

- 用任意用户或 calibration corpus 的 `(Parent KV, Current Exact KV)` pair 回归；
- 用 Current Exact response、score 或 label 拟合 coefficient；
- 用上层 Current Exact `Delta KV` 做 SVD/聚类/字典学习后持久化；
- 训练一个 neural adapter 直接把 Parent state 翻译成 Current state。

当前候选的科学价值必须来自“执行两版已知 Transformer 计算图的一条受限差分路径”，而不是从 target
states 学一个更小的 translator。

### 4.3 anchor/base 加 delta/residual 也已有直接先例

[MobiLoRA](https://aclanthology.org/2025.acl-long.1140/) 对相同 prefix 的不同 LoRA adapter 保存 anchor
KV 和 layer-wise delta KV，并对 delta 做 error-bounded encoding。它先产生目标 adapter cache，再压缩
与 anchor 的差异；不是合法的低成本 release constructor，但已阻止“跨模型 cache delta encoding”这一
宽泛 claim。

[ForkKV](https://arxiv.org/abs/2604.06370) 利用 LoRA 结构把 cache 写为大规模共享 base cache 和轻量
adapter residual cache，并以 ResidualAttention 在 SRAM 中组合 base/residual。它也明确承认上层 hidden
随 adapter 分叉，使共享 base 在第一层以上成为近似。与当前候选的差别是：ForkKV 的 residual 来自
LoRA feature-rank parameterization，当前路线面对任意 full-model adjacent release，并希望沿 history
axis 构造 finite-release state defect。尽管如此，以下点都不能再单独声称新：

- cache 的 base-plus-residual layout；
- 多模型共享一个 base/anchor cache；
- 每个变体只保存小 residual；
- attention kernel 在不写回完整 cache 的情况下消费 base 与 residual；
- residual connection 使模型间 state drift 较缓。

[Activated LoRA serving](https://arxiv.org/abs/2512.17910) 还说明，通过改变 adapter 的激活边界，可以
让 base 与 adapted model 在边界前精确共享 prefix cache。这是 model-system co-design 的另一类解法；
EvoKV 不要求重新训练或限制 release，但不能把“模型变化下仍复用 prefix”当作首次提出。

## 5. model delta、LoRA 与 tangent 的边界

[LoRA](https://openreview.net/forum?id=nZeVKeeFYf9) 已把 parameter update 写成
`W_C=W_P+BA`；[DeltaZip](https://doi.org/10.1145/3689031.3717468) 已系统化 base model 加压缩 full-model
delta 的多变体 serving。因此这些事实都不是 Insight 2：

- 相邻 release 的 `Delta W` 幅值较小或低秩；
- base weights 可共享、只保存 model delta；
- `X Delta W` 可以用 low-rank factors 加速；
- parameter delta 的 spectrum 可以指导 rank。

更重要的是，低秩 `Delta W` 不推出低秩 `Delta X_l` 或 `Delta KV_l`。后者还包含从所有下层传播的 state
term、normalization、attention activation、gate 和 residual interaction。

JVP/forward-mode tangent 同样是标准工具；例如 [Pearlmutter 的 R-operator](https://doi.org/10.1162/neco.1994.6.1.147)
和后续 forward-mode AD 都已能沿计算图传播 parameter direction。不能把“沿网络传播 `Delta theta`”
本身作为 novelty。若未来使用 tangent，还必须承认：一次 JVP 只给 Parent 点的一阶项，不在 full-rank
极限恢复有限版本差异。真正可能形成机制贡献的是针对 persistent Transformer state 的 **finite-release
paired recurrence、可执行 state interface、严格 exact-limit 与 0--20% cost envelope**；不是自动微分。

当前 rank-8 reduced Current prefill 避开了一阶余项，但若它完全是 single-arm Current execution，也会
退化为 generic low-rank inference。两条风险必须同时避开。

## 6. 哪些点绝不能写成创新

以下内容可以作为实现组件、diagnostic 或 related-work connection，但不能作为论文的核心新点：

1. rank-8、SVD、randomized range finder、QR、moment、采样或聚类；
2. KV 或 `Current-Parent` tensor 低秩；
3. 沿 token/history axis 压缩 K/V；
4. 一个用户/sequence 有自己的 basis；
5. 同一用户的 continuation 或后续请求复用 basis；
6. 一个 basis 在多个 head、K/V 或 layer 间共享；
7. 从 layer 0 初始化 subspace 并在 depth 上继承；
8. 每层保存 signed core，或把 exact Parent 与低秩 delta 相加；
9. reader 不在 HBM 中物化完整 migrated cache；
10. model/version KV transfer、linear mapping、learned translator 或 residual adapter；
11. low-rank `Delta W`、base-plus-model-delta serving 或 JVP；
12. recommendation 用户状态会被很多请求重复读取。最后一点是很强的系统动机和 amortization 条件，
    但 ShadowKV/prefix-cache/multi-turn work 已经覆盖 repeated-context reuse 的一般价值。

尤其不能把下式当作 Design 的定义：

\[
C_l=U_0^T(K_l^{C,Exact}-K_l^P)
\]

无论 recovery 多高，它都读取了上层 Current Exact，只能叫 `layer0-U / exact-core oracle`。把所有层 exact
delta 联合做 SVD 则应明确叫 `xKV-on-Delta oracle`。

## 7. 最小可信机制组合

在现有 prior art 下，最小可辩护候选应同时包含以下条件；删掉其中任一关键条件都很容易退化为已有
compression、delta encoding 或 mapper。

### 7.1 合法的起点

`U_0` 只从该用户 raw history 在 Parent/Current layer-0 的 dependency-free release defect 形成。basis
constructor 不读取：

- 上层 Current Exact K/V、hidden、response 或 score；
- qualification 用户的 label；
- Parent/Current target-state calibration pairs；
- candidate-specific query。

range finder/SVD 只是固定的数值实现，不是 novelty。

### 7.2 Parent 必须进入层间生成，而不是只在最后 subtraction

需要在固定 `U_0` quotient 中执行一条 **paired Parent/Current recurrence**。每层的 Current coefficient
应由：上一层携带的 paired/differential state、两版该层参数，以及该层原生 Transformer primitive 共同
产生。Parent persistent state 是递推的输入，不只是最终编译 sidecar 时才读取的 tensor。

一个很直接的机制判据是：

> 如果把 constructor 中的 Parent 删除，只运行一份 rank-8 Current compressor，最后再减
> `U_0^T Parent KV`，输出完全不变，那么它仍是 generic compression control，不是两版本迁移机制。

当前 `single-arm Current reduced prefill -> final Parent subtraction` 应保留作 matched-compute baseline。
真正候选必须证明 paired differential evolution 在相同 rank/compute 下带来额外恢复，或者提供 single-arm
方法不具备的 causal/exact-limit invariant。

### 7.3 只持久化 cross-version defect

最终状态仍是已有 exact Parent K/V 加一份用户级 sidecar：一个 `U_0` 和每层 `Delta K/Delta V` core。
不能另存一份完整 compressed Current cache 并把它重新命名为 migration；也不能丢掉 Parent complement。

### 7.4 reader 保持原生 query dependence

reader 必须计算与 materialized
`(K_P+U_0 C_K, V_P+U_0 C_V)` 相同的 HSTU attention semantics。K correction 会改变 nonlinear attention
weight，不能把 Parent response 与 delta response 线性相加后假装普适。对其他 Transformer，只能要求
各自 adapter 实现相同 cache semantics，不能声称 HSTU 的 activation identity 普遍成立。

### 7.5 full-rank exact-limit

当 `r` 达到有效历史长度、所有数值截断关闭，并使用完整合法 source state 时，paired recurrence 应按层
归纳恢复 finite-release Current trajectory/KV。否则它只是一个有用近似，不足以说明在传播“版本差分”。

该 exact-limit 不能由 Current Exact coefficient 注入实现，也不能把一次 Parent-point JVP 冒充 finite
release exactness。

### 7.6 single-edge persistent migration semantics

算法只处理一个 `Parent -> Current` edge。用户历史 sidecar 在 release 时生成，并被后续 ranking/retrieval
请求共同读取；它不涉及跨 `V0 -> ... -> V5` 的 recursive multi-version accumulation。五条 rolling edge
是五个独立实验 repeat，不是一个连续迁移算法。

上述组合可以用一句不过度造词的话概括为：

> **由 layer-0 release defect 固定用户 history subspace，在该共享 subspace 中耦合传播两版
> Transformer 的 finite-release difference，并只向 exact Parent cache 写入各层低秩 correction。**

这句话中的新意必须落在“耦合传播两版 finite-release difference”；`history subspace` 和 `low-rank
correction` 本身都有强 prior art。

## 8. 必须设置的直接 controls 与 falsification tests

在冻结任何 Design 1 合同前，至少需要以下 matched controls：

1. **Current-only reduced replay**：只做 rank-8 Current prefill，最后与 Parent 相减。它代表 generic
   low-rank inference + residual encoding，也是当前字面候选的风险下界。
2. **Linformer/random history projection**：固定、model-global 或随机 `U`，排除任意 sequence projection
   都有效。
3. **ShadowKV/xKV-style Current compression**：对已形成的 Current cache 做 per-layer 或 cross-layer
   history factorization；它不合法进入 0--20% Design frontier，但给 compression ceiling。
4. **xKV-on-Delta oracle**：读取所有层 exact delta 后联合分解，回答 residual-domain rank ceiling；明确
   generation cost 超过 100%。
5. **Layer0-U / exact-core oracle**：`U_0` 合法，但上层 core 读取 Current Exact。它只检验表示充分性。
6. **DroidSpeak-style layer recompute/reuse**：在相同 edge/workload 上体现 Insight 1 与跨模型系统 prior。
7. **closed-form mapper**：若实现，仅作为 target-fitted cross-model transfer baseline，不进入 Design。
8. **paired quotient replay**：Parent 在每层进入 recurrence；只此项有资格成为当前 novelty candidate。

建议用以下可证伪门决定是否继续，而不是先给方法命名：

- **version-essential gate**：paired 方法在相同 rank、同一 `U_0`、同一 FLOP 下必须稳定优于 Current-only
  replay。若无增益，Parent-aware mechanism 没有必要性，贡献退化为 compressor。
- **release-specific gate**：layer-0 `Delta KV` 得到的 `U_0` 必须优于由 Parent-only、Current-only、随机
  或 model-global basis 得到的表示。否则“release differential”只是选 basis 的叙事。
- **no-target gate**：执行 trace 证明没有任何上层 Current Exact、target response、score 或 label read。
- **causal gate**：逐层 coefficient 只能依赖 raw history、Parent persistent interface、两版参数和更早
  已形成的 coefficient。
- **exact-limit gate**：toy/full-rank test 逐层恢复 Current native computation，而不是只匹配最终 score。
- **functional gate**：在 Current reader intervention 中恢复 response、user representation 与 ranking/
  retrieval gap；tensor energy/cosine 不能代替因果恢复。
- **edge gate**：五条 edge 全报告，user-equal 后 edge-equal；不按有利 edge、layer 或 head 选择结果。
- **budget gate**：主点处在 `0--20% Exact-All` per-user generation cost，并报告 reader overhead、storage
  与 I/O；`80--90%` 仍是最终 Design target，preflight 不下调它。

本文件不冻结这些门的数值细节，也不授权 32-user 或 scale GPU launch。

## 9. 严格 FLOP 与 storage 语义

当前机制在拥有正式 cost ledger 前不能声称处于甜点区。至少要逐项计算：

- Current raw embedding/input formation；
- 每层 bounded range finder 的所有 matrix multiply、power iteration、QR 和 small SVD；
- factorized RMSNorm、Q/K/V projection core、output projection 与 gate；
- native nonlinear attention 的 `N x N` score/activation，以及 `N^2 r` contractions；rank-8 不会自动
  消除 attention matrix；
- 每个 residual/gate rank-expansion boundary 的 dense materialization 与下一次 compression；
- approximate layer-0 defect、`U_0` basis construction；
- 每层 `U_0^T Parent K/V`、`U_0^T U_l` alignment 和 core products；
- sidecar construction/write；
- reader 每个请求读取 exact Parent 加 `U_0/core` 的额外 FLOP、byte 与 kernel workspace。

`Exact-All` baseline 必须使用同一个 multiply/add 口径。已有 Parent K/V 的原始生成不计入 release-time
generation FLOP，但 release 时读取它的 bytes/I/O 不能消失；任何 Current Exact prefill 都使该方法成为
`>=100%` oracle。model-edge global preprocessing 与 per-user generation 分开报告，不能把 user-specific
basis work 摊到请求数上来满足 `0--20%`。

sidecar 的最小 scalar 数为

\[
N r + 2 L r H,
\]

分别对应一个共享 `U_0` 和每层 K/V core；dtype、orthogonalization metadata、长度/mask 与 alignment
state 另报。这个紧凑存储公式与 xKV 的 factor storage 近似同构，不能作为 novelty 证据。

## 10. 论文表述边界

### 可以安全作为研究问题的表述

> 对持续更新的 stateful Transformer recommender，跨版本 defect 是否能够由 dependency-free layer-0
> history modes 初始化，并沿两版 reader 的因果计算路径在一个固定用户 subspace 中传播，从而以小于
> full KV rematerialization 的成本形成对推荐决策足够的 persistent correction？

### 若实验成立后，可能使用的贡献表述

> 我们发现，版本差分的 token support 虽然稠密，但其跨层 causal evolution 可由 layer-0 release
> defect 确定的用户 history subspace 承载。基于此，我们在该 subspace 中耦合执行 Parent/Current
> differential，并把各层 correction 作为 exact Parent cache 的 sidecar 供原生 reader 使用。

其中“发现”必须同时有 dense-support、cross-layer subspace transport 和 causal intervention 证据；仅有
SVD energy 不够。

### 不应使用的表述

- “我们提出首个 token-axis low-rank KV cache”；
- “我们首次发现 KV 在不同层共享低秩 basis”；
- “我们首次提出 user-specific / context-aware KV subspace”；
- “我们首次用 layer-0 basis 贯穿 Transformer”；
- “我们首次保存 base KV 加 signed delta”；
- “我们首次实现无需重构完整 KV 的 residual attention”；
- “我们首次迁移不同模型版本的 KV cache”；
- “我们的创新是 rank-8/SVD/randomized projection”。

## 11. 最终 preflight 判定

当前候选分成两个层次：

- **不应作为主 Design 的版本**：rank-8 single-arm Current reduced prefill，随后以 `U_0` 投影并在最后
  减 Parent core。它是一个重要、合法、可能很强的 systems control，但 prior-art 组合风险过高。
- **值得继续验证的版本**：同一 `U_0` 下的 Parent/Current paired finite-release differential
  recurrence；不读上层 Current Exact，Parent 在每层参与 coefficient formation，最终只输出 exact
  Parent 的低秩 state correction，并具有 full-rank exact-limit。

截至 2026-09-03，本次检索没有找到同时具备“任意 full-model adjacent release、persistent exact Parent
cache、dependency-free layer-0 release basis、无 target-state fitting、两版共享 history quotient 的
causal recurrence、low-rank signed sidecar、native reader、full-rank finite-release exact-limit”的原始
论文。可是，相关工作的并集已经覆盖除 causal paired construction 外几乎所有表面组件。因此只有当
paired construction 本身带来稳定、可测、matched-budget 的额外恢复时，这条路线才达到 Design 1 所需
的论文高度；否则应诚实地归类为已有 KV compression/residual encoding 的组合。

## 12. 本次使用的原始论文与官方页面

- [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)
- [Palu: Compressing KV-Cache with Low-Rank Projection](https://openreview.net/forum?id=LWMS4pk2vK)
- [EigenAttention: Attention in Low-Rank Space for KV Cache Compression](https://aclanthology.org/2024.findings-emnlp.899/)
- [OjaKV: Context-Aware Online Low-Rank KV Cache Compression](https://aclanthology.org/2026.findings-acl.494/)
- [ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference](https://proceedings.mlr.press/v267/sun25b.html)
- [xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction](https://arxiv.org/abs/2503.18893)
- [Gated Subspace Inference for Transformer Acceleration](https://arxiv.org/abs/2605.03109)
- [DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving](https://arxiv.org/abs/2411.02820)
- [Cross-Model KV Cache Transfer in LLM Families](https://arxiv.org/abs/2608.03893)
- [Latent Cache Flow: Model-to-Model Communication Without Text](https://openreview.net/pdf?id=PvnR2LDCOs)
- [Mixture-of-Translators](https://arxiv.org/abs/2607.28979)
- [MobiLoRA](https://aclanthology.org/2025.acl-long.1140/)
- [ForkKV](https://arxiv.org/abs/2604.06370)
- [Efficient Multi-Adapter LLM Serving via Cross-Model KV-Cache Reuse with Activated LoRA](https://arxiv.org/abs/2512.17910)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://openreview.net/forum?id=nZeVKeeFYf9)
- [DeltaZip: Efficient Serving of Multiple Full-Model-Tuned LLMs](https://doi.org/10.1145/3689031.3717468)
- [Fast Exact Multiplication by the Hessian](https://doi.org/10.1162/neco.1994.6.1.147)

检索只使用上述原始论文、正式 proceedings/DOI 页面与作者论文页面；没有以综述、博客或搜索摘要作为
claim 依据。
