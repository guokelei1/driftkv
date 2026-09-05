# Cross-version attention-response operator：prior-art collision preflight

日期：2026-09-03  
状态：**严格 NO-GO：当前 `signed response operator / K--V finite difference / cone moments` 组合不足以承担 Insight 2 或 Design 1**  
范围：只审计论文 claim 与相关工作边界；未运行实验、未修改合同、seal、raw result 或 frozen docs

## 1. 裁决先行

截至 2026-09-03，对原始论文与官方 proceedings 的审计给出以下结论：

1. **“attention 后的 functional/model-visible error 比 raw KV error 更重要”已经被直接提出。**
   [HeadQ](https://arxiv.org/abs/2605.03562) 明确把 key error 定义在 query-visible score quotient，
   把 value error定义在 attention-weighted readout，并实现 low-rank side code 与 read-time logit correction；
   [OptR](https://arxiv.org/abs/2608.02691) 直接以 post-attention、post-`W_O` output error 优化 KV；
   [CacheBridge](https://arxiv.org/abs/2609.00891) 又把 cross-model cache transfer 的校准目标改成
   receiver-composed attention sensitivity。EvoKV 不能再把“从 storage space 转向 reader-functional
   space”本身写成 Insight 2。
2. **K-address / V-content 的精确有限差分以及所谓 `Delta K x Delta V` interaction 不是新分解。**
   OptR 的精确等式已经将 output error 写成 key-induced routing change 与在 perturbed attention 下读取
   `Delta V` 的 value term；展开后后者显式含有 `Delta attention x Delta V`。在 fixed affine region 内，
   `Delta attention` 对 `Delta K` 为线性，这正是当前候选所称的 `Delta K x Delta V`。
3. **固定 activation region 的 `B/M` moments 是 linear-attention / fast-weight state 的特例。**
   `B=sum V` 与 `M=sum k outer V` 可合并为一个 augmented-feature outer-product memory；其 query read、
   additive update、difference state 都已落在 linear attention 与 fast-weight 的标准代数中。固定 cone
   是一个可能有价值的 workload observation，但不是方法创新。
4. **“signed attention response”也不是可独立主张的新机制。**
   [Differential Transformer](https://proceedings.iclr.cc/paper_files/paper/2025/file/00b67df24009747e8bbed4c2c6f9c825-Paper-Conference.pdf)
   已用两个 attention maps 的差做 cancellation；
   [Efficient Attention via Control Variates](https://arxiv.org/abs/2302.04542) 已用 exact/approximate
   attention 的 control-variate 语言构造低成本 attention。两个 release endpoint 的语义不同仍可作为
   问题边界，但“取差”“signed”“control path”本身都不能承担 novelty。
5. **persistent per-user recommendation KV 以及 cross-model KV handoff 都已有直接先例。**
   [CollectiveKV](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4078c8b648dc107aedbdf561dd4edc2a-Abstract-Conference.html)
   已研究 sequential recommendation 中每用户长期 KV 的存储，并拆成全局共享 KV 与低维用户特有 KV；
   [DroidSpeak](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan) 已研究同架构、不同
   fine-tuned model 的 KV 复用与选择性重算；[Semantic Cache Distillation](https://arxiv.org/abs/2606.07684)、
   [Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893) 和
   [CacheBridge](https://arxiv.org/abs/2609.00891) 已覆盖 learned/closed-form state transfer、compact
   semantic codes、intermediate patch boundary 与 attention-aligned mapping。
6. **当前候选剩下的最窄空白只是一个尚未实现的约束组合，不是已经成立的方法：**已有 exact Parent
   per-user state，在单个相邻 release 上，不读取 Current Exact upper state、不用 paired target-state
   calibration，以少量 Current execution 生成可跨 recommendation requests 持久复用的 release defect。
   本轮没有找到直接等价的原始论文，但“若干已有条件的交集尚未被显式发表”不等于论文级机制。
7. 仓库自己的 matched control 还给出更强的否证：paired functional path 被 single Current reduced
   replay 支配，见 [paper-height mechanism audit](paper_height_mechanism_audit.md)。因此当前既没有
   prior-art-safe 的机制 claim，也没有 version-essential 的数值证据。
8. 同日完成的 [all-history affine response falsifier](all_history_affine_response_preflight.md) 进一步
   排除了“先把 estimator 调准即可”的解释：full-Exact `B/M` representation oracle 的五边 recovery 为
   `-.0569/.7333/-13.0716/.9195/-1.9074`，edge mean `-2.6766`；single-r8 executable 为 `-2.6575`。
   因此最简单的 affine response state 不仅 prior-art 不新，其 representation ceiling 本身也失败。

**最终裁决：不要把 cross-version signed attention-response operator、`D_K/D_V/D_KV`、或
piecewise-affine moments 冻结为 Insight 2 / Design 1。** 它们可以保留为分析语言、negative result、
oracle 或某个未来方法的 implementation adapter；当前没有足够 novelty。

## 2. 先把候选对象写准确

对固定 query `q`，令 attention weights/activations 为 `A(q,K)`，一层的 history response 为

\[
R(q;K,V)=A(q,K)V.
\]

若以 Parent 为基点，写

\[
K_C=K_P+\Delta K,\qquad
V_C=V_P+\Delta V,
\]

以及

\[
A_C=A(q,K_C)=A_P+\Delta A,
\]

则存在精确有限差分

\[
\begin{aligned}
R(q;K_C,V_C)-R(q;K_P,V_P)
&=\Delta A\,V_P+A_P\,\Delta V+\Delta A\,\Delta V\\
&=\Delta A\,V_P+A_C\,\Delta V.
\end{aligned}
\]

这里：

- `Delta A V_P` 是 K 通过 query--key interaction 改变地址/routing 后对旧 content 的作用；
- `A_P Delta V` 是旧 routing 对新 content 的直接读取；
- `Delta A Delta V` 是 routing change 与 content change 的有限 interaction；
- 对 softmax 或一般非线性 kernel，`Delta A` 是 `Delta K` 的非线性函数，不能把第三项普遍写成字面
  双线性的 `Delta K Delta V`；
- 对当前 legacy `ELU+1` reader 的固定正分支，`Delta A` 对 K 为线性，此时该 interaction 才可具体
  化为 `q Delta K^T Delta V` 型项。

[OptR 的 Equation 8](https://arxiv.org/html/2608.02691v1#S3.SS2) 已给出同一个精确二项形式：
key-induced term 使用 `Delta p` 读取原始 `V`，value-induced term 使用 perturbed `p_tilde` 读取
`Delta V`。因为 `p_tilde=p+Delta p`，第二项已经包含 `Delta p Delta V`。因此不能声称：

- 首次把 K 与 V 作用分成 address/content；
- 首次给出非 Taylor 的 finite K/V decomposition；
- 首次发现或保留 `Delta K x Delta V`；
- 现有工作只保留 direct K/V term 而本文首次考虑 interaction。

还有一个重要限定：上述等式**固定了同一个 q**。完整 Parent→Current trajectory 一般同时有
`Delta q`、normalization、gate、residual 与下层 state drift。逐层在 Current query 上拆 K/V 是合法的
diagnostic intervention，但它不是完整 cross-version Transformer finite difference。论文中若把它称为
完整 release operator，会过度主张。

## 3. Claim-by-claim 碰撞矩阵

| 潜在 claim | 最直接 primary-source collision | 未被覆盖的精确边界 | 裁决 |
|---|---|---|---|
| raw KV error 经 reader 后变成更紧凑的 functional error | [HeadQ](https://arxiv.org/abs/2605.03562)：model-visible score/readout geometry；[OptR](https://arxiv.org/abs/2608.02691)：post-`W_O` output objective | Parent→Current release，而非 single-model quantization | **已碰撞；release 只改变 error source，不改变 functional-coordinate claim** |
| K 是 address、V 是 content | HeadQ 的 score-visible K / readout-visible V；OptR 的 key-induced/value-induced exact split | HSTU pointwise activation 与 recommendation query distribution | **已碰撞；只能做 workload-specific evidence** |
| `D_K+D_V+D_KV` 是新 finite-response decomposition | OptR Eq. 8 精确包含 perturbed-routing × value-error；[LoFAST](https://proceedings.mlr.press/v235/havens24a.html) 已系统分析 attention 对输入 perturbation 的 local sensitivity | release interaction 在不同 edge/depth 的实证必要性 | **代数不新；可作为 negative/empirical observation** |
| signed difference of two attention responses cancels common error | [Differential Transformer](https://proceedings.iclr.cc/paper_files/paper/2025/file/00b67df24009747e8bbed4c2c6f9c825-Paper-Conference.pdf)；[EVA](https://arxiv.org/abs/2302.04542) | 两臂是两个真实 release endpoint，且 Parent state 已物化 | **subtraction/cancellation 不新；endpoint semantics 只是边界** |
| fixed cone 中 response 由 `B/M` 精确表示 | [Transformers are RNNs](https://proceedings.mlr.press/v119/katharopoulos20a.html)；[Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html)；[ReLA](https://aclanthology.org/2021.emnlp-main.523/) | user query 是否长期留在同一 activation cone | **moment state 已碰撞；cone stability 可是新观察但不足以成 Design** |
| piecewise activation 产生可迁移 region | ReLA 使用 ReLU attention；[Attention is a smoothed cubic spline](https://arxiv.org/abs/2408.09624) 已从 spline/activation-region 角度刻画 attention | 两个 release 间 region persistence 与 recommendation workload | **理论形态不新；只能主张经严谨验证的特定现象** |
| 一个 user-specific compact state 跨层/请求复用 | [xKV](https://arxiv.org/abs/2503.18893) 的 shared token basis + per-layer cores；CollectiveKV 的用户级 cache | per-user state 是 release defect，不是单版 compressed cache | **表示不新；必须证明新 constructor** |
| exact Parent base + compact signed residual，reader 原生消费 | [MobiLoRA](https://aclanthology.org/2025.acl-long.1140/) 的 cross-adapter KV delta encoding；[ForkKV](https://arxiv.org/abs/2604.06370) 的 base/residual cache 与 ResidualAttention | 任意 full-model adjacent release、不是 LoRA parameterization | **layout/kernel pattern 已碰撞** |
| 同架构旧模型 cache 被新模型读取，并只做少量新计算 | [DroidSpeak](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan) | per-user long-lived recommender state、release-time offline migration | **问题与 bounded recomputation 已碰撞；workload 尚不同** |
| compact functional state transfer / intermediate patch boundary | [SCD](https://arxiv.org/abs/2606.07684)：semantic codes + normalized pre-attention Patch | SCD 依赖 paired source/target full prefills与 learned translator；EvoKV 禁止 target-state fit | **宽泛 framework claim 已碰撞；no-target constructor 仍开放** |
| attention-aligned cross-model transfer 优于 coordinate KV fit | [Cross-Model KV Transfer](https://arxiv.org/abs/2608.03893) 报告 attention-output similarity；[CacheBridge](https://arxiv.org/abs/2609.00891) 以 causal attention sensitivity 校准 mapper | 无 calibration、用户现场构造、persistent recommendation | **该 Insight 已被直接覆盖** |
| recommendation 中每用户持久 KV 值得压缩/共享 | [CollectiveKV](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4078c8b648dc107aedbdf561dd4edc2a-Abstract-Conference.html) | 模型 release 后的 state compatibility，而非单版 cache storage | **系统动机不新；cross-version 才是剩余限定** |

## 4. Piecewise-affine / cone moments 的精确边界

当前 legacy reader 采用 pointwise, unnormalized `rho(z)=ELU(z)+1`。在一个 query region 内若正集合
`P={i:q k_i>=0}` 固定，则正分支 response 为

\[
R_+(q)=\sum_{i\in P}(1+s qk_i)v_i
=B_P+s qM_P,
\]

其中

\[
B_P=\sum_{i\in P}v_i,\qquad
M_P=\sum_{i\in P}k_i v_i^\top.
\]

把 feature 扩维为

\[
\phi_q(q)=[1,sq],\qquad \phi_k(k)=[1,k],
\]

就有

\[
R_+(q)=\phi_q(q)
\left(\sum_{i\in P}\phi_k(k_i)v_i^\top\right).
\]

括号内正是 linear attention / fast-weight literature 中的 additive outer-product memory。
[Transformers are RNNs](https://proceedings.mlr.press/v119/katharopoulos20a.html) 已用 separable kernel
feature 与矩阵结合律把 attention 写成 recurrent finite state；
[Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html)
则明确把 state 写成 key/value activation 的 additive outer products，并研究对 memory mapping 的更新。
因此：

- `B/M` 的维数固定、可 append、可 eviction、可由 query 读取，都不是新性质；
- `Delta B=B_C-B_P`、`Delta M=M_C-M_P` 只是两个已有 moment states 的线性相减；
- signed moments 或 positive/negative moments 分开存也只是 feature/state bookkeeping；
- 多个 cone、cone dictionary、Taylor/polynomial moments若只是在不同 region 保存更多 feature moments，
  仍属于已有 kernel-feature / piecewise-polynomial approximation 范式。

真正尚可报告的是一个**实证现象**：同一用户的 recommendation queries 是否在多个请求、候选与 append
之后仍共享一个低复杂度 activation-region family。它可以解释为什么某个 adapter 有效，但本身不能提供
低成本 Current moment constructor，也不能把标准 outer-product memory 变成新 Design。

仓库最新 falsifier 还说明不能从“negative branch 平均 response fraction 小”推出全历史 affine state
足够。[all-history affine response preflight](all_history_affine_response_preflight.md) 使用完整 Current
Exact K/V 构造同一 `B/M` state，五边仍有三边为负，最差 edge 为 `-13.0716`。这已经在 estimator 之前
否决了 probe-free all-history affine representation。保留 query-conditioned region 虽可避免该反例，
但会重新引入 region mask、boundary remainder 与 request dependence，并不会改变 moments 的 prior-art
归属。

### 4.1 HSTU 名称不能替 legacy ELU adapter 背书

原始 HSTU 论文 [Actions Speak Louder than Words](https://arxiv.org/html/2402.17152v3) 在其公式中对
attention nonlinearity 使用 **SiLU**，并包含 relative attention bias；SiLU 不是 piecewise affine。
仓库已明确当前 Medium checkpoint 是 legacy `ELU+1`、无 relative-bias 实例。因此：

- exact cone `B/M` 不能写成 HSTU 一般性质；
- 更不能写成 Transformer 一般性质；
- 最多可写成 functional-migration framework 在一个 legacy pointwise reader 上的专用 compiler；
- 若 Insight 2 依赖 exact affine cone，它与用户要求的 architecture-neutral Transformer insight 相冲突。

## 5. K/V finite difference、Taylor 与 influence 的边界

一阶展开通常保留

\[
\delta R\approx J_K\,\delta K+J_V\,\delta V,
\]

而忽略二阶及更高 interaction。attention 的 local sensitivity/Jacobian 分析本身已有成熟先例；例如
[LoFAST](https://proceedings.mlr.press/v235/havens24a.html) 对标准 dot-product self-attention 的输入扰动
给出细粒度 local sensitivity，并把界写到 attention matrices 与未扰动输入上。

有限差分保留 interaction 是正确的，但不能由此推出新颖性：

1. `Delta A Delta V` 是把乘积 `(A+Delta A)(V+Delta V)` 展开的直接结果；
2. OptR 已在 exact output decomposition 中通过 `A_tilde Delta V` 保留它；
3. HeadQ 已把 K perturbation 放到 score/Fisher geometry、把 V perturbation放到 readout geometry；
4. CacheBridge 甚至明确说自己的 first-order surrogate丢弃 K--V blocks，这说明“保留还是丢弃 interaction”
   已是现有方法的显式设计选择，而不是未被识别的空白。

所以当前 `D_K/D_V/D_KV` 最安全的论文角色是：

- 作为 causal diagnostic，证明单独修 K 或 V 不稳定；
- 作为反例，说明只依赖一阶 K+V correction 在某些 release 会过冲；
- 作为 correctness test，检查实现是否保留 native endpoint interaction。

它不能单独导出低成本 state constructor。若 Design 最后只是把三个 term 分别低秩化、量化、采样或分配
rank，reviewer可以准确地把它归类为已有 perturbation decomposition 加 generic compression。

## 6. Persistent recommendation state migration：真正剩下多少

### 6.1 已被占据的外围

[CollectiveKV](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4078c8b648dc107aedbdf561dd4edc2a-Abstract-Conference.html)
已经建立以下场景：Transformer sequential recommender、每用户 KV、用户量导致的持久存储压力、共享
global KV 与低维 user-specific KV、多个模型与数据集上的 KV compression。因此下列句子不能再作为
贡献：

- recommendation 的 KV 与 LLM session KV 不同，因为它按用户长期持久化；
- 每用户 KV 需要紧凑表示；
- user-specific state 可以与共享 state 解耦；
- 同一用户的后续请求会重复消费 compact KV。

[DroidSpeak](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan) 则明确把 continuously updated
fine-tuned models 列为 cross-model cache-sharing use case，并用少数层重算加其余层复用解决同架构模型
间 handoff。因此“旧版本 cache 被新版本 model 消费”和“以 bounded recomputation 避免 full prefill”
也不是首次提出。

cross-model transfer 的空间随后已非常拥挤：

- [Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893)：paired KV traces、per-head closed-form
  ridge、cross-layer sources、RoPE stripping；
- [SCD](https://arxiv.org/abs/2606.07684)：100--500 paired prefixes 上两模型 full prefill，学习 low-rank
  Reuse codes 与 normalized pre-attention Patch；
- [CacheBridge](https://arxiv.org/abs/2609.00891)：receiver-aligned head support、query-conditioned causal
  attention weighting、bounded sufficient-statistics construction；
- [MobiLoRA](https://aclanthology.org/2025.acl-long.1140/)：跨 LoRA adapter 的 KV delta encoding；
- [ForkKV](https://arxiv.org/abs/2604.06370)：共享 base cache、adapter-specific residual cache、在 attention
  kernel 内重构/读取。

因此“functional state migration framework”这一宽标题过大。SCD 已经直接使用 semantic state transfer
与 intermediate transition patch；CacheBridge 已经把 receiver functional damage 放进 cross-model
mapping；ForkKV 已经把 base/residual state 做成 native reader interface。

### 6.2 仍未被本轮直接找到的最窄 problem setting

本轮没有找到一篇原始论文同时满足以下全部条件：

1. source 是**已经物化**的 per-user Parent Transformer history state，而非 handoff 时重新执行 source
   prefill；
2. target 是同一 recommender 的单个 adjacent release；
3. constructor 不读取 Current Exact upper-layer K/V、response 或 paired target calibration traces；
4. 不训练 Parent→Current mapper、translator、codebook 或 target-fitted basis；
5. 只做 `0--20% Exact-All` 的 Current-version per-user computation；
6. correction 在 release time 生成后由多个 ranking/retrieval requests 重复读取；
7. state 有 append/eviction 与 lineage 语义。

这可以定义 EvoKV 的**问题边界**，但不能自动成为贡献。它仍缺少一个已有方法无法还原的新构造原理。
尤其是：

- “在推荐系统里做 xKV/HeadQ/SCD”是应用迁移；
- “不训练 mapper，改用 SVD/moments”是算法替换；
- “Parent response 加 signed sidecar”是 ForkKV/MobiLoRA 邻近的 base-plus-residual layout；
- “两个 release 都低秩执行再相减”是 generic compression + control variate；
- “由于同一用户多请求所以 amortize”是系统动机，不是 mechanism novelty。

## 7. 当前绝不能写的创新表述

以下话术应从未来 Insight 2 / Design 1 草稿中明确禁止：

1. 首次提出 reader-level、functional、model-visible 或 attention-output cache correction；
2. 首次发现 K 控制 address、V 携带 content；
3. 首次将 K/V error 精确拆成 direct 与 interaction terms；
4. 首次考虑 `Delta K x Delta V`；
5. 首次使用 signed attention response、两路 attention 相减或 Parent control variate；
6. 首次把历史变成 `B/M` outer-product moments；
7. 首次在 activation region/cone 中把 attention 写成 affine/polynomial form；
8. 首次持久化每用户 recommendation KV 或 compact user state；
9. 首次做 cross-model/cross-version KV reuse、state transfer 或 selective recomputation；
10. 首次使用 exact base + compact residual、reader-time reconstruction 或 sidecar；
11. 首次发现 raw KV MSE 不等价于 downstream/attention-output fidelity；
12. 首次提出 intermediate functional boundary、semantic code 或 pre-attention patch。

可以保留但必须降级的表述：

- “我们在 HSTU recommendation release workload 上观察到 interaction 的 edge heterogeneity”；
- “legacy pointwise reader 的用户 queries 在若干请求上保持 stable cone”；
- “full exact moments 给出很高 representation ceiling”；
- “单侧 K/V correction 与局部 state splice 不能稳定恢复”。

这些是 domain-specific findings 或 negative evidence，不是完成的 Design。

## 8. 还可能成立的最小新颖命题

经过碰撞后，唯一值得继续审查的命题应收窄为：

> 对已经持有 exact Parent per-user state 的 adjacent recommender release，存在一种不读取 Current
> Exact target state、不做跨模型拟合的 **finite-release causal constructor**。它用 Parent state、
> raw history、Parent/Current 参数与少量 Current execution，直接产生 persistent user-level release
> defect；Parent 路径在每层生成中不可删除，并且该 defect 在 matched compute 下比 Current-only
> compressed replay 更接近 Current reader behavior。

这个句子仍只是一个待证 proposition。要达到 PRO 那种论文高度，未来方法必须带来至少一个不能被
compression/mapping/moments 还原的 invariant，例如：

- **version-essential recurrence**：删除 Parent arm 会严格改变逐层生成，而不是只在最后做
  `compressed Current - Parent`；
- **no-target construction**：不通过任何 Current Exact trace、score 或 response 拟合 state；
- **finite endpoint correctness**：full-rank/无截断极限逐层恢复真实 Current endpoint，而不是一次
  Parent-point Taylor/JVP；
- **functional causality**：上层 correction 只依赖合法的更早 release state，不能借 upper Current
  Exact closure；
- **matched-compute advantage**：在同 FLOPs/storage 下稳定优于 Current-only compressor，这一点证明
  Parent/Current pairing产生了新信息，而不是增加组件；
- **persistent lineage**：同一对象跨请求读取并随 append/eviction 合法演化。

当前 paired route 已在仓库 canary 中输给 matched single-Current control，所以“matched two-endpoint
approximation error cancellation”目前也**没有通过**。在出现新的 Current-information source 或新的
causal invariant 之前，不能把上述 proposition 写成已发现的 Insight 2。

## 9. 对下一轮研究的硬 gate

新的候选在启动 population experiment 以前，应先逐项回答：

| gate | 必须满足 | 失败后归类 |
|---|---|---|
| Prior-art identity | 核心对象不是 moments、mapping、low-rank code、base+residual 或 signed subtraction | 已有方法组合 |
| Version-essential | 删除 Parent arm后，在相同 Current work 下结果明显改变并稳定变差 | generic Current compression |
| No-target | constructor trace 不读 Current Exact upper states/response/label，不用 paired target calibration | SCD/CacheBridge/mapper 类 |
| Exact endpoint | full-rank limit 对 toy/full path 逐层恢复 Current native semantics | heuristic approximation |
| Interaction | K routing、V content 与其 finite interaction由 native reader保留，而非只加 final vector | first-order/output offset |
| Cost | 完整 per-user constructor `<=20% Exact-All`，不漏 input、QR、reader overhead 或 metadata | 不在 Design 1 甜点区 |
| Matched baseline | 同 runner、同 FLOP 下优于 single Current reduced replay，不能只报 absolute recovery | 无迁移特异收益 |
| Persistence | cross-candidate、后续请求、append/eviction 与 fallback 全部可审计 | 单请求 correction |

对当前 operator/moment 路线，前三层门已经失败：对象与 prior art 重叠，paired 路径没有
version-essential superiority，probe-free full-Exact affine representation oracle 也失败。继续调 rank、
cone count、probe、moment order 或 K/V budget 只会变成参数搜索，不能修复 novelty。

## 10. Primary-source ledger 与精确使用边界

下表只列本次裁决实际依赖的原始论文或官方 proceedings。2026 年若干条目仍是 arXiv v1；它们不是
peer-review 质量背书，但已足以构成投稿时必须正面处理的公开 prior art。本页不是穷尽性或法律意义的
novelty opinion。

| Source | 本页依赖的原始内容 | 不应误读为 |
|---|---|---|
| [HSTU, ICML 2024](https://arxiv.org/html/2402.17152v3) | published HSTU attention uses SiLU、relative bias、pointwise aggregation | legacy `ELU+1` cone identity |
| [Transformers are RNNs, ICML 2020](https://proceedings.mlr.press/v119/katharopoulos20a.html) | separable kernel feature、associative finite recurrent state | cross-version migration |
| [Linear Transformers Are Secretly Fast Weight Programmers, ICML 2021](https://proceedings.mlr.press/v139/schlag21a.html) | additive key/value outer-product memory 与 query read | release-specific constructor |
| [Sparse Attention with Linear Units, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.523/) | ReLU attention、normalization/gating、piecewise-linear activation | 本文 exact ELU implementation |
| [Attention is a smoothed cubic spline](https://arxiv.org/abs/2408.09624) | ReLU attention/masked/cross attention 的 spline view | persistent cache algorithm |
| [Differential Transformer, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/00b67df24009747e8bbed4c2c6f9c825-Paper-Conference.pdf) | 两个 softmax attention maps 相减以 cancellation | 两个 model releases 的 state transfer |
| [Efficient Attention via Control Variates, ICLR 2023](https://arxiv.org/abs/2302.04542) | control-variate view of approximate/exact attention | exact Parent persistent cache semantics |
| [LoFAST, ICML 2024](https://proceedings.mlr.press/v235/havens24a.html) | standard attention 对 input perturbation 的 local sensitivity | finite release constructor |
| [HeadQ](https://arxiv.org/abs/2605.03562) | model-visible K score quotient、V readout geometry、low-rank side code、logit correction | cross-version KV migration；主实验为 single-model quantization |
| [OptR](https://arxiv.org/abs/2608.02691) | exact key/value-induced post-`W_O` output decomposition，perturbed attention 下读取 `Delta V` | model-release setting；对象为 INT2 quantization |
| [xKV](https://arxiv.org/abs/2503.18893) | cross-layer shared token basis、per-layer core、selective reconstruction | Parent→Current construction |
| [CollectiveKV, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4078c8b648dc107aedbdf561dd4edc2a-Abstract-Conference.html) | sequential recommendation 中 per-user KV、global/shared 与 user-specific state | cross-version compatibility |
| [DroidSpeak, NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan) | same-architecture fine-tuned models、cross-model KV reuse、selected-layer recomputation | dependency-closed recommendation migration |
| [MobiLoRA, ACL 2025](https://aclanthology.org/2025.acl-long.1140/) | cross-adapter KV delta encoding、reuse/eviction | arbitrary full-model release |
| [ForkKV](https://arxiv.org/abs/2604.06370) | base/residual caches 与 reader-time ResidualAttention | arbitrary model-delta semantics；依赖 LoRA structure |
| [Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893) | paired trace ridge mapper、cross-layer sources、attention-output fidelity | no-target、no-mapping constructor |
| [Semantic Cache Distillation](https://arxiv.org/abs/2606.07684) | paired full prefills、low-rank semantic codes、normalized pre-attention Patch | 已物化 Parent-only release migration |
| [CacheBridge](https://arxiv.org/abs/2609.00891) | receiver attention-aligned calibration、bounded sufficient statistics、affine transfer | calibration-free per-user construction |

## 11. 最终审计结论

当前 candidate bundle 的每个核心部件都能被现有工作单独覆盖：

```text
signed response subtraction
  -> differential attention / control variates

K-address + V-content + finite interaction
  -> model-visible KV geometry / exact output-aware decomposition

piecewise cone + B/M moments
  -> ReLU/spline attention + linear-attention/fast-weight state

persistent user sidecar + native reader
  -> recommendation KV compression + base/residual KV systems

cross-model functional transfer
  -> DroidSpeak / ridge transfer / SCD / CacheBridge
```

把这些部件串在 Parent→Current recommender release 上，确实形成一个尚不完全相同的 workload，但目前
只是 **known components under a narrower deployment contract**。再加上 paired route 没有胜过 generic
single-arm control、all-history affine state 又在 Full-Exact representation ceiling 上失败，本页的严格
结论只能是：

> **没有足够 novelty；当前方向不能作为 Design 1。下一候选必须引入新的、可证伪的 finite-release
> causal construction principle，而不是继续扩充 response decomposition 或 moment vocabulary。**
