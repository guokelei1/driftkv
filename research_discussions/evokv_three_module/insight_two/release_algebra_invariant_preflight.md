# Release algebra / invariant preflight：当前 KV-only 接口下的穷尽式机制审计

日期：2026-09-03  
状态：**严格 NO-GO（针对当前 `v0..v5`、Parent K/V-only source、`<20%` 约束）；不是 Insight 2 / Design 1 冻结稿**  
范围：只读 checkpoint 与 Transformer 代数审计；未训练、未读 discovery/confirmation user、未运行 GPU、未修改合同或 seal

## 1. 裁决先行

本轮专门寻找一种比 PRO 更有机制高度、同时又不是 mapper、generic compression、sampling 或参数调节的
`Parent -> Current` 有限版本代数。允许的输入是：exact Parent K/V、raw history、Parent/Current 两版权重；
constructor 不得读取 Current Exact state、candidate panel、label 或 future request，并须低于 `20% Exact-All`。

结论是：**在当前接口和六个 frozen Medium endpoint 上，没有找到可执行的 exact/finite-release
algebra 或 invariant。** 这不是因为某个 rank 没调好，而是五类可能删除 Current 历史执行的结构出口均被
逐项关掉：

1. **attention gauge quotient 不适用真实 release。** 六层五边全部不是 Q/K、V/O gauge 加 head
   permutation；最佳 permutation 在全部 `30` 个 layer-edge 上都是 identity，gauge-invariant endpoint
   mismatch 仍为 `5.21%--11.89%`。最优 infinitesimal gauge tangent 平均只解释约 `5%--7%` 的
   Q/K 或 V/O 参数移动。
2. **endpoint update 没有 exact low-rank provenance。** `30` 个 `192 x 192` block matrix 的
   `Delta W` 数值秩为 `180--192`；用实际 full rank 执行全部 direct parameter terms 本身已约为
   `92.60%--93.28% Exact-All`。即使只保留 cache production 必需的前五个完整 block 加末层 K/V，
   并把 full-rank factor自动切回更便宜的 dense matmul，`27` 个 transforms 仍为 `42.72%`，尚未计算
   attention、nonlinearity 或 residual。`rank@90` 确实很低，但截断以后就是已有 low-rank delta/JVP
   数值近似，不是 exact release algebra。
3. **checkpoint 不含可重放的 optimizer/path provenance。** 六个 payload 都只保存 endpoint model；
   release contract 使用 fresh AdamW per checkpoint，未保存 optimizer moments、minibatch gradients 或
   中间 parameter path。即使保存，AdamW 的 coordinatewise moments 也不会把 observed full-rank endpoint
   delta 变成一个免费 exact factorization。
4. **native recommendation query 没有 exact key nullspace。** 对每个 Current endpoint，固定使用
   model vocabulary 的 item ID `1..256`、time `0`、默认 query type/action 形成合法 layer-0 query；
   六个 head 全部达到 numerical rank `32/32`。因此对 native query set，
   `q delta-k = 0 for all q` 的 head-wise annihilator 已经是零空间，不能从 query symmetry 得到更小的
   exact K quotient。
5. **nonlinear reader 不存在独立于历史长度的有限低阶 query-independent sufficient statistic。** legacy `ELU+1` 的
   positive branch 可写成有限 `B/M` moments，但 negative branch 是 `exp(q k^T)`，对一般 q/k 具有
   无限 separation rank；若要求 exact，只能保留随历史增长的逐 token 信息。SiLU gate 与 RMSNorm 又让
   parameter secant 具有非终止高阶项。仓库的
   full-Exact all-affine representation oracle 已在三条 edge 为负，直接排除了把近似 closure 当 exact
   invariant 的可能。

剩下的唯一路径是沿 Current causal trajectory 执行真实 contextualization。当前 Parent K/V-only
source 要恢复这个 trajectory，已有乐观 source-coordinate floor 为 `25.3173%`，稳定 joint decode
版本为 `34.8113%`；它们还没有开始 Current finite defect。也就是说，**exactness、当前 source
interface 与 `<20%` 三者不能同时成立。**

这份裁决不声称对任意程序作复杂度论上的不可能性证明。它穷尽的是当前 Transformer straight-line
graph 中能够合法省掉依赖的结构类别：function-preserving symmetry、structured endpoint update、native
query quotient、finite aggregation closure 和 cached causal separator。若不属于这五类，一个新方案就
必须近似/拟合/采样完整 Current trajectory，因而回到本项目已经明确排除的 generic family。

## 2. 审计对象与方法

### 2.1 Frozen endpoints

只读以下 sealed Medium checkpoint：

- `v0`: `shared_v0/checkpoint_100.pt`；
- `v1..v4`: `D14/checkpoints/v*/checkpoint_100.pt`；
- `v5`: `D14/v5_extension_v1/checkpoint/checkpoint_100.pt`。

每个模型为 legacy pointwise HSTU：`L=6,H=192,heads=6,d=32,N<=1024`。本轮没有选择 UID，也没有读取
任何 recommendation request。checkpoint payload 的 top-level keys 经只读检查，六版均没有
`optimizer` 或 `optimizer_state_dict`。

### 2.2 Model-only measurements

参数谱覆盖每层五个 square matrix：Q/K/V/out/gate，共 `30` 个 matrix/edge。数值秩阈值固定为

\[
192\,\epsilon_{fp32}\,\sigma_{max},
\]

并同时报告 Frobenius-energy `rank@90`、`rank@99`；没有根据 user quality 选择 rank。

Gauge audit 按真实六 head 切分。PyTorch row-vector convention 下，每个 head 的两个可观察 invariant 为

\[
A_h=W_{Q,h}^{\mathsf T}W_{K,h},\qquad
C_h=W_{V,h}^{\mathsf T}W_{O,h}^{\mathsf T}.
\]

它们分别在

\[
W_Q' = G_Q^{\mathsf T}W_Q,\quad W_K'=G_Q^{-1}W_K
\]

和

\[
W_V'=G_V^{\mathsf T}W_V,\quad W_O'=W_O G_V^{-\mathsf T}
\]

下不变。本轮逐 layer 穷举 `6!=720` 个 head permutation，最小化 QK/VO combined invariant mismatch；
还解 Sylvester least-squares problem，把真实 endpoint delta 投影到 infinitesimal gauge tangent，报告
explained energy。该投影只用于否证，不生成 user state，也不是 migration mapper。

Native-query witness 固定使用每版模型自己的前 `256` 个非 PAD item embedding，加入默认 query
type/action 和 time-zero temporal embedding，经过真实 `in_proj -> layer0 RMSNorm -> Q projection`。只要
这 `256` 个合法 query 已经 span 一个 head 的全部 `32` 维，就足以否定该 head 存在对**全部** native
queries 都不可见的非零 key direction；无需读取 serving candidate panel。

## 3. Weight-update provenance：effective low rank 不等于 exact algebra

### 3.1 五边完整谱

| edge | exact numerical rank min/med/max | sum `r90` | sum `r99` | direct `Delta W` cost @r90 | @r99 | @exact factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v0->v1` | `180/187/191` | 132 | 510 | 2.176% | 8.406% | 92.599% |
| `v1->v2` | `183/188/192` | 162 | 613 | 2.670% | 10.104% | 93.127% |
| `v2->v3` | `185/188/192` | 174 | 633 | 2.868% | 10.433% | 93.275% |
| `v3->v4` | `185/188/192` | 174 | 629 | 2.868% | 10.368% | 93.061% |
| `v4->v5` | `184/188/192` | 157 | 588 | 2.588% | 9.692% | 92.962% |

factor cost 只计算

\[
C_{direct}=4NH\sum_m r_m
\]

这一组 `X Delta-W^T` direct terms，分母为 repository 的
`Exact-All=4,771,282,944 FLOPs/user`。它没有计算 input/embedding delta、`Delta X W_P`、RMSNorm、
QK、AV、ELU、out projection、gate、residual、state truncation 或 sidecar。因此 exact-rank 数字不是
完整算法成本，而是足以停止“把 endpoint delta exact factorize 后逐 primitive 推进”的单项成本。
因为 full-rank factor apply 会比一个 dense matmul更贵，合法 executor应逐 matrix 使用
`min(4NHr,2NH^2)`。只构造六层 K/V 时，前五层需要 Q/K/V/out/gate，末层只需 K/V，共 `27` 个 square
transforms；把它们全部切回 dense 仍为
`2,038,431,744 FLOPs = 42.7229% Exact-All`。这还没计算 QK、AV、ELU、gate pointwise 或 residual，
所以 no-go 不依赖故意使用低效的 full-rank factor。表中的 30-matrix数字保留为完整 block weight-geometry
audit，不能偷换成 cache-only executor cost。

观察到的 `r90=3--15` 很有诊断价值：相邻训练 update 的主要 Frobenius energy 确实集中。但它不能承担
论文机制，原因是：

- exact delta 仍接近 full rank；
- `rank@90/99` 丢弃的是 model parameter energy，不是 Current-reader functional null direction；
- token-dependent RMSNorm、ELU region crossing、SiLU/Hadamard gate 会把 low-rank direct term变成 dense
  state defect；
- 进一步对 `Delta X` 截 rank 就是 generic numerical compression。

这与已有 [release tangent preflight](release_tangent_propagation_preflight.md) 的边界一致：low-rank
`Delta W` 只让 direct parameter term 便宜，不会让 causal state propagation 自动便宜。

### 3.2 为什么 optimizer lineage 不能补上缺失的代数

训练合同明确使用 `fresh_AdamW_per_checkpoint`；endpoint checkpoint 没有保存 optimizer moments 或逐步
gradient。因而不存在可供 migration constructor replay 的 finite update tape。

即使未来保存训练 path，一次 linear-layer minibatch gradient可写成 outer-product sum，也不推出
最终 AdamW delta exact low rank：一阶/二阶 moments、coordinatewise division、weight decay 与几千个
step 的求和一般产生 full-rank endpoint，实际 `180--192` rank 已是直接反例。若反过来把训练约束成
LoRA/adapter update，确实能得到结构化 delta，但那是新的 migration-aware training contract，并与
[LoRA](https://arxiv.org/abs/2106.09685) / delta tuning 的核心机制重合；不适用于当前 v0..v5。

## 4. Attention gauge：唯一干净的 exact shortcut，但真实 release 不在其 orbit 上

### 4.1 它本来可以怎样成为 exact migration

若 Current 与 Parent 的 attention 参数只相差每-head Q/K、V/O gauge 和 head permutation，则无需
重算 historical contextualization：对 Parent cached K/V 做相应坐标 transport，Current reader 的
logits、activation 和 output 可保持精确不变。该 route：

- 不读 target Current KV；
- 有 machine-precision exact limit；
- 变换只由两版权重决定；
- per-token cost 可低于 Full。

所以 gauge 是本轮最接近“非 mapper、非 compression 的 exact finite-release algebra”的候选。

但它不能成为本论文的新方法。Transformer attention 的 Q/K inverse-transpose、V/O inverse gauge 和
head permutation 已被系统刻画；[Complete Characterization of Gauge Symmetries in Transformer
Architectures](https://neurips.cc/virtual/2025/136893) 给出了 canonical Transformer generic stratum 上的
最大 gauge group。用 symmetry 对齐模型参数也已经属于 model re-basin/canonicalization 方向，
[Git Re-Basin](https://arxiv.org/abs/2209.04836) 是直接 related-work boundary。

更决定性的是，当前 endpoints 不满足它。

### 4.2 实际五边结果

| edge | best head-permuted invariant mismatch, layer min/med/max | Q/K tangent explained | V/O tangent explained | best permutation |
| --- | ---: | ---: | ---: | ---: |
| `v0->v1` | `8.25/9.39/11.89%` | 5.66% | 6.06% | identity, 6/6 layers |
| `v1->v2` | `5.96/7.09/9.12%` | 5.45% | 7.51% | identity, 6/6 layers |
| `v2->v3` | `5.77/7.07/7.55%` | 5.13% | 6.04% | identity, 6/6 layers |
| `v3->v4` | `5.21/7.18/8.69%` | 5.93% | 6.26% | identity, 6/6 layers |
| `v4->v5` | `5.89/9.39/10.58%` | 5.76% | 5.82% | identity, 6/6 layers |

这里 mismatch 是 combined QK/VO invariant Frobenius distance，以两个 endpoint invariant energy 归一化。
它比 FP32 roundoff 大多个数量级；只要 invariant 不等，任何 GL head gauge 都不能把该层 Parent attention
变成 Current attention。identity 在 `30/30` layer-edge 都是最优 permutation，也符合 continued training
保持 head identity、而不是发生离散 head swap 的预期。

作为更宽的 residual-coordinate sanity check，固定前 `256` 个 item embedding 的 row-normalized Gram
matrix 在五边的相对漂移为：

```text
6.879% / 8.762% / 8.252% / 7.311% / 6.669%
```

任意单一 global orthogonal/signed-permutation coordinate change都会保持这个 Gram；实际 drift 因而也不
支持“整个 residual stream 只是换了坐标”的解释。

可以把每个 endpoint delta 分成“best gauge tangent + non-gauge residual”，但后者仍保留约 `93%--95%`
的 Q/K、V/O movement energy。迁移 residual 时仍需 Current causal state；这一步若拟合/压缩就回到 generic
method。故“gauge canonicalization + residual correction”只是已有 symmetry alignment 与另一个近似器的
组合，不满足用户要求的论文高度。

## 5. Parameter path、secant 与 commutator 为什么不能 finite-close

### 5.1 一阶 path 不是有限 endpoint

令 `theta(t)=theta_P+t Delta-theta`。严格恒等式是

\[
F(theta_C,x)-F(theta_P,x)
=\int_0^1 J_\theta F(theta(t),x)\,Delta\theta\,dt.
\]

Parent JVP 只取 `t=0` integrand。它即使 full rank，也只是一阶 tangent；Current endpoint 的 finite
difference还包含 RMSNorm、ELU、SiLU 和多层 composition 产生的全部高阶 mixed terms。linearized
network/JVP 本身已有 [Fast Adaptation with Linearized Neural
Networks](https://proceedings.mlr.press/v130/maddox21a.html) 等直接先例。把应用场景改成 cache migration
不会让一阶算子本身变新。

对单个 changed linear primitive已有精确 finite identity

\[
\Delta(XW)=\Delta X W_P+X_P\Delta W+\Delta X\Delta W.
\]

问题不在漏掉这一个 interaction，而在 `Delta X` 必须经过前面全部 Current nonlinear primitive 才能
知道。现有 [K/V finite interaction audit](paper_height_mechanism_audit.md) 也已显示，省略 finite
interaction 在一条 edge 会灾难性 over-correct；只保留 commutator/first-order term没有统一安全性。

### 5.2 Exact telescoping 没有删除计算图

把所有 changed primitive 任意排序，构造 hybrid endpoint
`theta^(0)=theta_P,...,theta^(M)=theta_C`，总有

\[
F(theta_C)-F(theta_P)
=\sum_{j=1}^M [F(theta^{(j)})-F(theta^{(j-1)})].
\]

这是 exact finite-release algebra，但每一项都需要 changed primitive 之后的 dependency closure。attention、
gate 和 residual 使这个 closure 延伸到后续所有层/历史位置；共享 Parent forward 不能提供 hybrid 的
Current upper state。逐项执行比 Exact Current 更贵，截断若干项则是 parameter locality / Taylor
approximation。

Baker--Campbell--Hausdorff 或 nested commutator 只有在高阶 commutator终止时才提供有限表达；当前 block
不是一串可交换线性流，而含 `RMSNorm -> QK -> ELU+1 -> AV -> out -> SiLU gate -> Hadamard -> residual`。
这些解析/分段解析非线性具有一般非零的任意阶导数。真实 checkpoint 也不存在一个训练-path operator
ledger来证明特殊 nilpotency。因此“secant/commutator”最多是 oracle expansion，不能成为 `<20%` exact
constructor。

## 6. Native query symmetry：推荐请求共享不产生 exact low-dimensional quotient

一个 head 的 key 只通过 `q k_i^T` 被 reader 看见。若所有未来 native queries 落在 proper subspace
`S subset R^d`，则 `K` 在 `S^perp` 上的分量确实不可见；这是一个干净的 query quotient：

\[
K \sim K+D\quad\Longleftrightarrow\quad qD^T=0,\ \forall q\in S.
\]

model-only witness 的结果却是：五个 Current endpoint、每版六个 head，全为 `rank=32/32`；共 `30/30`
edge-head witness 满秩。各 witness 的 `sigma_min/sigma_max` 最小/中位/最大为
`3.89e-5 / 5.39e-5 / 1.65e-4`。因此：

- exact annihilator 为零；
- recommendation candidate sharing 不意味着 head feature directions 存在 exact nullspace；
- 用 top singular directions 忽略小 singular values 仍可作为 approximate query compression，但不再是
  invariant，而且会依赖 query distribution / tolerance。

上层 query 还会经 Parent cache response、gate 和 residual 变成 history-dependent；layer-0 已经没有
exact nullspace，不能期待上层反而提供一个对所有未来请求安全的固定 quotient。对一个有限 candidate
vocabulary做每-user response lookup 在形式上 exact，但状态为 `O(|items|H)`、依赖 candidate，不是 compact
persistent migration object。

## 7. Attention aggregation closure：为什么 legacy 特例也不能救场

legacy activation 为

\[
\phi(z)=\begin{cases}1+z,&z\ge 0\\e^z,&z<0.\end{cases}
\]

若固定所有 query-key sign 且只保留 positive branch，response可由

\[
B=\sum_i v_i,\qquad M=\sum_i k_i^{\mathsf T}v_i,
\qquad r(q)=B+qM
\]

精确读取。但 `e^{qk^T}` 的 Taylor expansion包含所有阶 `q/k` monomial；对一般连续 q/k 没有固定有限维
feature map。因此 native negative branch 要么保留逐 token K/V，要么使用 kernel approximation / sampled
features。后者正是 linear/kernel attention approximation，不是新的 release invariant。

而且这不只是理论担忧。[all-history affine falsifier](all_history_affine_response_preflight.md) 已用 full
Exact K/V 构造该 representation oracle，五边 recovery 为

```text
-0.0569 / 0.7333 / -13.0716 / 0.9195 / -1.9074
```

所以失败源于 representation，不是 estimator。即使它成功，`sum phi(k)^T v` 也与
[Transformers are RNNs](https://proceedings.mlr.press/v119/katharopoulos20a.html) 和
[Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html)
的 associative fast-weight state同构，不能承担 novelty。

## 8. 当前 source interface 的最后一道成本门

排除 symmetry、parameter structure、query nullspace 和 finite moments 后，exact route 只剩：从 exact
Parent source恢复足够的 Parent execution state，再逐层推进 finite Current defect。

[Parent-anchored delta scan](parent_anchored_delta_scan_preflight.md) 已给出：

| optimistic mandatory source work | Exact-All |
| --- | ---: |
| K-only checkpoint decode + historical Q/gate | 25.3173% |
| stable joint K/V checkpoint decode + historical Q/gate | 34.8113% |

二者都尚未计算 Current input/state delta、attention response、output projection、nonlinearity、defect
compression 或 sidecar。另一个 migration-ready tape 方案虽然删除 decode，却要求额外 `26` 个 `N x H`
source fields（约 `19.5 MiB/user`，相对现有 K/V 增加 `216.7%`），而 Current native first-five-layer
QK+AV floor 已为 `42.2367%`。因此 source tape 也不能让当前模型进入 `<20%`。

## 9. Constraint-complete classification

下面不是对所有计算机程序的 formal lower bound，而是对当前 computation graph 的“省计算出口”分类。

| structural exit | exact 时需要什么 | 当前 endpoint/interface 证据 | 若放松 exact 后退化成 | 裁决 |
| --- | --- | --- | --- | --- |
| function-preserving symmetry | Parent/Current 在相同 gauge orbit | invariant mismatch `5.21%--11.89%`；gauge只解释约 6% movement | gauge alignment + residual mapper/compressor | NO-GO |
| structured endpoint update | sparse/low-rank/commuting `Delta theta`，且 state closure | exact `Delta W` rank `180--192`；无 optimizer path | LoRA/delta tuning/JVP/truncated secant | NO-GO |
| native-query quotient | future query span存在固定 annihilator | layer-0 native witness `30/30` heads full rank | query-aware KV compression/probe fitting | NO-GO |
| finite reader sufficient state | attention kernel有有限 exact feature map且递推闭合 | ELU negative exponential + SiLU/RMS；full-Exact affine oracle失败 | linear/kernel attention moments | NO-GO |
| cached causal separator | source保存足够 Parent execution checkpoints | K/V-only decode floor `25.3%--34.8%`；full tape过大且 native floor `42.2%` | generic reduced replay or new architecture/training | NO-GO |

这张表也解释为什么“再组合两个已有组件”不能产生论文方法：每个组合都必须有一个新的结构出口来删除
Current causal work。若没有，它只是把近似误差从 KV、parameter、moment、query 或 residual 中的一个坐标
系换到另一个坐标系。

## 10. 任何未来 release-algebra 候选必须通过的 matched controls

### M0. Exactness sanity

- `theta_C=theta_P` 时 correction bitwise zero；
- synthetic pure Q/K + V/O gauge、含 head permutation的 release 必须 machine-precision恢复；
- full structural rank/no truncation 必须逐层恢复 Exact Current，而不是只恢复 final score；
- constructor API 不接 Current Exact state、candidate/query panel或 label。

### M1. Version-essential control

同 cost 比较：

1. proposed Parent+Current algebra；
2. Current-only reduced replay；
3. 删除 Parent-specific term；
4. 用相同 Frobenius norm、相同 effective-rank spectrum 的 random non-gauge `Delta W` 替换真实 release
   delta。

只有真实 endpoint coupling稳定优于 2/3，并在 4 上消失，才能说明收益来自 finite release structure，
而不是普通 compression capacity。

### M2. Gauge reduction test

分别报告 identity frame、best head permutation、exact synthetic gauge transport、actual endpoint best-gauge
residual。若方法在 pure-gauge case做的只是已知解析变换，novelty 为零；若 actual gain 全由 residual mapper
产生，也不能把整体称为 gauge migration。

### M3. Secant/curvature control

沿 `theta_P+alpha Delta theta` 固定 `alpha={1/4,1/2,1}`，比较 Parent JVP、midpoint JVP、固定阶
quadrature 与 finite endpoint。只有一阶误差按 `alpha^2` 缩放但在 `alpha=1` 仍足够小，才能保留 tangent
作为 workload observation；它仍需与 linearized-network prior art区分。parameter `rank@90/99/full` 与
state-rank truncation必须正交 ablate，不能把两种误差相消当机制。

### M4. Native-query quotient control

用全部 model-native query span的 rank/nullspace 作主判定；低维 artificial probe span只作 positive
sanity。若 quotient 只对 frozen odd-32 candidates有效，它是 target-panel compression，不是 persistent
state。

### M5. Prior-art reduction test

把“release”“recommendation”两个名词从算法描述中删掉：

- 若只剩 `Q/K inverse gauge + V/O gauge`，归入 attention symmetry/model re-basin；
- 若只剩 truncated `Delta W` + JVP，归入 LoRA/delta tuning/linearized network；
- 若只剩 `sum phi(k)^T v`，归入 linear attention/fast weights；
- 若只剩 source-to-target state relation，归入 cross-model KV mapping/model stitching；
- 若只剩 query-weighted error objective，归入 functional KV compression。

只有删掉应用名词后仍留下一个以前没有的 finite-version Transformer identity，并且 M1 证明它对真实
Parent/Current coupling必需，才允许进入论文 Design。

## 11. 对 Insight 2 / Design 1 的直接含义

当前最诚实、也最有信息量的结论不是给 gauge、secant 或 low-rank update 起新名字，而是：

> **有限请求上的 functional compactness，并不推出 Parent KV 中存在一个可由有限版本代数低成本生成的
> Current functional quotient。真实 release 同时离开 attention gauge orbit、具有 full-rank endpoint
> update、暴露 full-span native queries，并通过非有限闭合的 nonlinear reader传播；所以 compact
> response 的“存在”与 compact correction 的“可构造性”必须分开。**

这是 Transformer-specific 的接口否证，但它本身还不是 Design 1。对当前 v0..v5，若继续坚持
K/V-only、无新训练、`<20%` 和 paper-level novelty，则应保持 **no Design frozen**，不能把 numerically
strong 的 single-r8、PRO、gauge+residual、parameter rank 或 moments 重新包装成新方法。

重新打开 paper-worthy Design 只有两类诚实变化：

1. 新的 model-system interface 在 cache production 时形成真正的 causal separator，使 later reader
   对旧 prefix 的依赖按结构穿过一个 bounded state；
2. release training 本身受结构约束，使真实 endpoint update落在可证明的 symmetry/low-rank/closed
   operator family。

二者都需要新的 architecture/training contract，不是当前六个 checkpoint 上的“再做一个映射”。是否
采用其中之一应由单独 prior-art 与资源审计决定；本文件不替它们预先冻结 Design。
