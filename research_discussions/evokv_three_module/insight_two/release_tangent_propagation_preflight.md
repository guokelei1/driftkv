# Release tangent propagation：机制、接口与成本预检

日期：2026-09-02  
状态：**只读推演；未实现、未运行用户实验、不是 Design 1，也不授权 GPU execution**

## 1. 裁决摘要

`v0->v1` 的模型级观察——多数 `192 x 192` block parameter delta 的 90% Frobenius-energy rank
约为 `3--11`，99% rank 约为 `7--52`——值得开启一个新的 oracle diagnostic，但它还不能直接推出
低成本 cache migration。

最关键的区分是：

> 低秩 `Delta W` 只让直接参数项 `X Delta W` 变便宜；它不会自动让状态传播项
> `Delta X W_parent`、attention Jacobian、normalization 或 gate 变便宜。

因此本路线只有在另外两个条件成立时才可能进入 `0--20%`：

1. Parent cache 除 K/V 外还提供足以重放局部 Jacobian 的 source execution state；
2. 跨历史位置的 state defect `Delta X in R^(N x H)` 在每个非线性边界后仍能被很小的 rank `s`
   近似。

当前两点都没有证据。特别是，只有现有 K/V 接口时，恢复每层 hidden/Q/gate 所需的三个 dense
`H x H` transforms 下界已经约为 Exact-All 的 `28.48%`，尚未开始 attention 或 delta propagation。
所以 **KV-only 的 release tangent action 当前不可行**。

在一个明显更重的 Parent execution tape 假设下，使用最乐观的 `r<=11`、state rank `s=8`、legacy
activation-region prefix moments 和三次低秩压缩/层，算术下界约为 Exact-All 的 `18.34%`。这个数字
只说明“存在一个很窄的可研究窗口”，不是预算通过：它尚未计入完整 sign/certificate、QR/SVD 常数、
metadata、I/O 和 kernel inefficiency，而且 source tape 本身会使每用户持久状态扩大到当前 K/V 的约
三倍。

本路线最小可辩护的论文命题不是“parameter delta 低秩”，而是：

> 相邻 release 的模型差分能否沿缓存的 Parent Transformer trajectory 做一个 delta-only execution，
> 在不拟合 Current KV/score 的条件下形成全历史 functional defect，并在 full-rank/no-truncation
> 极限恢复原 Current computation graph？

这条命题尚未成立，首先必须通过 full-rank tangent ceiling 与 state-defect rank oracle。

## 2. 两个必须分开的算法对象

设 `theta_C = theta_P + Delta theta`，历史 trajectory 为 `X_l(theta)`，待迁移 K/V 为
`F(theta,x)`。

### 2.1 一阶 JVP：机制 oracle

沿直线 release path `theta(t)=theta_P+t Delta theta`，定义：

\[
\dot X_l = \left.\frac{dX_l(theta(t))}{dt}\right|_{t=0}.
\]

一层的一阶传播为：

\[
\dot X_{l+1}
=J_X f_l(X_l^P,theta_P)\dot X_l
+J_theta f_l(X_l^P,theta_P)Delta theta_l.
\]

它可以由 forward-mode JVP 精确定义，不需要任何 Parent-to-Current state pairs。若所有 parameter
delta 使用 full rank、所有 state tangent 不截断，则它精确恢复
`J_theta F(theta_P,x) Delta theta`，但 **不等于** 有限 release difference
`F(theta_C,x)-F(theta_P,x)`；余项为二阶及以上。

完整 Current 的严格极限是路径积分：

\[
F(theta_C,x)-F(theta_P,x)
=\int_0^1J_theta F(theta(t),x)Delta theta\,dt.
\]

因此，full rank 只给“一阶 tangent exact”，还需要 homotopy step 数趋于无穷或高阶积分，才能得到
finite-version exact。这个边界必须明确，不能把 JVP 写成 Exact Current reconstruction。

### 2.2 有限差分 residual propagation：更强的 Design 候选

若 Parent trajectory 已缓存，实际更值得实现的是逐 primitive 的有限差分，而不是丢掉二阶项。对
线性层 `Y=XW`：

\[
Delta Y
=(Delta X)W_P+X_P(Delta W)+(Delta X)(Delta W).
\]

等价地可以用 `X_C Delta W` 合并后两项。normalization、attention activation 和 gate 直接计算
`op_C-op_P`，不做 Taylor 展开；只在 `Delta X` 的 token-by-feature matrix 上做 rank-`s` 数值截断。

若：

- `Delta W` factors 保留每个矩阵的完整实际 rank；
- state rank `s=H`，不做近似；
- attention backend 使用原生完整 kernel；
- Parent execution state 精确；

则有限差分 recurrence 可按 layer 归纳恢复 Exact Current trajectory。这是比一阶 JVP 更适合作为
算法 correctness test 的 exact-limit invariant。JVP 应作为“相邻 release 是否足够线性”的先导
diagnostic，而不是预先指定的最终实现。

## 3. 模型参数差分如何进入计算

每个 block matrix 使用只由两版参数决定的 deterministic SVD：

\[
Delta W_m \approx U_m Sigma_m V_m^T.
\]

对 `N x H` activation，full dense projection 为 `2NH^2` FLOPs；rank-`r` parameter term 需要：

\[
2NHr+2NrH=4NHr.
\]

这部分 factor 是 model-edge global object，可在 release 时对每条 edge 生成一次并由所有用户共享；
不能把它的 factorization wall time 隐藏，但不应把 SVD 重复计入每用户 FLOPs。

90%-energy rank 可以成为事前固定的 model-only axis，却不能被称为功能充分：Frobenius energy 没有
考虑某个 parameter direction 经用户 trajectory Jacobian 后的放大。99%-energy rank 也不自动满足
预算。对 Medium 30 个 block matrices，直接 parameter terms 的严格公式为：

\[
C_{parameter}=4NH\sum_{l,m}r_{l,m}.
\]

在后面的 `s=8` 乐观 envelope 中，其他项已占 `12.90%`，所以所有 30 个矩阵的 rank 总和必须不高于
约 `431`，即平均 rank 不高于约 `14.4`。`r<=11` 的 90% grid 可以进入预算预检；99% grid 中出现
`r=52`，必须先报告真实 rank sum，不能假定它仍可执行。

item/behavior embeddings、RMSNorm weights 和 input projection 另行处理。历史实际访问到的 embedding
rows 可以直接做两版差分；它们不能因不属于 `192 x 192` block matrix 而被漏算。

## 4. State defect 才是真正未知量

即使每个 `Delta W` 都是 rank 3，上一层产生的
`Delta X_l in R^(1024 x 192)` 也可能是 full rank。候选必须显式假设并验证：

\[
Delta X_l \approx A_l B_l^T,
\quad A_l\in R^{N x s},\ B_l\in R^{H x s},\ s\ll H.
\]

此时通过一个 full-rank Parent matrix 的传播成本为：

\[
C_{state-matmul}(s)=2sH^2+2NsH=2sH(N+H),
\]

而不是 `2NH^2`。但 RMSNorm Jacobian、token-dependent gates 和 elementwise attention derivatives 都
会使低秩 factor 变稠密。例如 RMSNorm JVP 包含一个逐 token 系数乘 Parent hidden row 的项；gating
包含两个 dense `N x H` activation 的 Hadamard product。参数低秩不能阻止这些 rank explosions。

一个合法数值实现可以在 norm output、attention output 和 block output 三处 materialize dense defect，
再用固定 rank、固定随机种子的 two-pass range finder 压回 rank `s`。这不是 target fitting，但它仍是
一个近似数值积分器，必须报告：

- exact defect 的 rank@90/rank@99；
- 每次 truncation 丢失的 energy；
- truncation 后 functional recovery，而不只报告 tensor cosine；
- QR/range-finder 的真实 FLOPs、workspace 与 deterministic reproducibility。

如果 state rank 需要 `s>=12`，后面的 two-arm cost envelope 已超过 `20%`；因此“parameter rank 很低”
不能替代 state-rank oracle。

## 5. Dense attention propagation

原生 full-attention JVP 不在预算内。一个 layer 的 native attention base 包含 QK 与 weighted-V 两个
matmul；其 tangent 还包含：

\[
\dot S=\dot QK^T+Q\dot K^T,
\qquad
\dot A=(phi'(S)\odot\dot S)V+phi(S)\dot V.
\]

后四个 dense pairwise matmuls在六层 Medium 上约为 Exact-All 的 `101.37%`；即使 Parent base
activation 已缓存，attention JVP 本身也远超 `20%`。

当前 legacy ELU+1/no-bias reader 提供一个架构特定的低成本 diagnostic。在固定 positive region 中：

\[
A_i^+=B_i+s q_iM_i,
\quad
B_i=\sum_{j\le i}m_jv_j,
\quad
M_i=\sum_{j\le i}m_jk_j^Tv_j.
\]

它的 tangent 为：

\[
\dot B_i=\sum_{j\le i}m_j\dot v_j,
\]

\[
\dot M_i=\sum_{j\le i}m_j
(\dot k_j^Tv_j+k_j^T\dot v_j),
\]

\[
\dot A_i^+=\dot B_i+s\dot q_iM_i+s q_i\dot M_i.
\]

这些 prefix states 可由 associative scan 构造，覆盖全部 1024 positions，不再使用 token coreset。
有限差分版本则分别构造 Parent/approximate-Current prefix moments，再做 response difference。

但这个 shortcut 有严格边界：

- mask crossing 的导数/有限差分没有被固定-region公式覆盖；
- ELU negative branch 不是 affine；
- faithful SiLU HSTU 和 softmax attention 没有相同的 exact moment；
- shared historical mask 目前只是待验证假设。

所以 native `O(N^2)` backend 必须保留为 exact-limit correctness oracle；moment backend 只是 Medium
可执行候选。若最终贡献只能依赖 ELU+1 affine identity，不能把它写成一般 Transformer 定理。

## 6. 所需 source-state interface

### 6.1 当前 KV-only 接口：不够

Parent K/V 都来自 normalized hidden 的 square projection。理论上可以在 `W_K^P` 可逆时由 K 做解析
反演，但它仍需每层一个 dense transform，而且 RMSNorm inverse 在 `eps` 很小时可能病态。随后从
reconstructed normalized hidden 形成 Parent Q 和 gate 又各需一个 dense transform。

Medium 每层一次 dense `H x H` transform across history 为 Exact-All 的 `9.494%`；K inverse、Q、gate
三项合计 `28.482%`。这还是不含 attention、parameter delta、state propagation 和 output projection
的算术下界。因此 analytic inverse 虽不属于 learned mapper，也不能挽救现有接口。

### 6.2 乐观的 Parent execution tape

一个 finite-difference delta pass 至少需要下列 per-user、per-layer Parent state：

1. pre-block residual hidden `X_l^P`；
2. Parent query projection `Q_l^P`；
3. Parent gate preactivation `G_l^P`；
4. Parent post-output-projection attention activation `O_l^P`；
5. 已有 Parent K/V；
6. raw item/behavior/time lineage，用于精确形成 Current input delta。

若不保存第 4 项，就需要从 Parent attention heads 再过一次 full output projection，或做不稳定的
gate division。若为了省去 Parent-side moment build/read再保存 pre-output attention heads，tape 还需
增加第五个 H-width field。

在 `L=6,N=1024,H=192` 下：

- 现有 FP32 K/V：`9.0 MiB/user`；
- 四个额外 H-width fields：`18.0 MiB/user`，即现有 K/V 的 `200%` 额外空间；
- K/V 加四-field tape：`27.0 MiB/user`；
- 再加 Parent attention heads：总计约 `31.5 MiB/user`。

这些 activation 必须在 Parent cache 原始生成时持久化；release 时重算 Parent trajectory 约等于再做
一次 Full，不能计作免费。量化或压缩 tape 是另一个尚未授权的研究问题，不能用未实现的压缩掩盖
I/O obstacle。

最终迁移输出不应持久化 dense approximate Current K/V。对当前 Medium，可把 all-history paired
response 编译为 38,016-scalar functional moments（现有正式 representation recovery 约 `0.995`）；
dense Current-like trajectory 只在 release constructor 中短暂存在。其他 Transformer 需要自己的
functional compiler。

## 7. Medium FLOP 可行性

Exact-All 的现有审计基准为 `4,771,282,944 FLOPs/user`。下表是一个**乐观算术下界**，假设：

- 30 个 block matrices 都用 rank `r=11`；
- state rank 固定 `s=8`；
- Parent four-field execution tape 已经存在；
- 每层在 norm/attention/block 三处做 two-pass rank-8 compression；
- Parent 与 Current 两臂都通过 activation-region associative moments build/read；
- 不做 token selection。

| component | FLOPs/user | Exact-All fraction |
| --- | ---: | ---: |
| exact Current input formation | 88,080,384 | 1.8461% |
| 30 low-rank parameter terms, `r=11` | 259,522,560 | 5.4393% |
| state factors through five Parent matrices/layer, `s=8` | 112,066,560 | 2.3488% |
| three two-pass state compressions/layer, `s=8` | 113,246,208 | 2.3735% |
| paired all-history moment build/read | 301,989,888 | 6.3293% |
| **optimistic subtotal** | **874,905,600** | **18.3369%** |

剩余 margin 只有 `1.6631%`，还需容纳 mask probes、normalization、activation、Hadamard gate、factor
concatenation/QR、certificate 和 metadata。它不能称为严格通过，只能允许一个 CPU/instrumentation
prototype继续审计。

如果 tape 再保存 Parent attention heads，只构造 Current-side moments，可将 moment项减半。此时
`s=16` 的乐观 subtotal 为 `19.8945%`，但 source state 又增加 4.5 MiB/user，且几乎没有 runtime
margin。相反，two-arm `s=12` 已是 `20.6980%`，`s=16` 为 `23.0592%`。

完整 cost 函数应写成 rank sums，而不能用“典型 rank”：

\[
C_{direct}=4NH\sum r_{l,m},
\]

\[
C_{state}=5L\cdot2sH(N+H),
\]

再加明确的 compression、attention reduction、input、pointwise、I/O 和 certificate。qualification
只能使用 observed model-edge rank sum 和固定 `s`，不能根据 quality outcome 选择 rank。

## 8. 最小 oracle diagnostic

第一轮不应直接实现系统。应冻结一个不读 confirmation 的 32-user、五-edge oracle，按以下顺序判定：

### O1. Full-rank tangent ceiling

用原生 full forward-mode JVP 计算
`J_theta F(theta_P,x) Delta theta`；不截断 parameter rank/state rank，不使用 moment shortcut。将
`KV_P + dot KV` 交给 Current reader，在 held-out panel 上测 functional recovery，同时报告：

\[
\frac{\|F_C-F_P-J_PDelta theta\|_2}{\|F_C-F_P\|_2}.
\]

若 full-rank JVP 的 edge-equal recovery 低于 `0.80`，一阶 release tangent family 立即停止；问题是
release curvature，不是 low-rank implementation。

### O2. Parameter-rank loss

在 state/JVP 仍保持 exact 的情况下，仅把 `Delta W` 换成 model-only `rank@90` 与 `rank@99` factors。
这一步隔离 Frobenius truncation 的功能损失。所有五条 edge 的 rank 分布必须先报告；当前只有
`v0->v1` 观察，不能外推。

### O3. State-defect rank

对 exact JVP 以及 exact finite difference 的 `Delta X_l/Delta N_l/Delta A_l/Delta K_l/Delta V_l`
分别计算 token-by-feature spectrum，并做事前固定 `s={4,8,12,16}` 的 oracle truncation。最重要的
gate 是 `s=8` 是否仍能达到至少 `0.80` functional recovery；因为 two-arm `s>=12` 已不满足当前
20% envelope。

### O4. Dense moment closure

使用 exact Parent tape 和 O2/O3 生成的 trajectory，分别比较：

1. native pairwise response；
2. teacher-forced activation-region moments；
3. closed historical prefix-moment rollout。

teacher-forced 成功、closed rollout 失败时，应判定 functional compiler 不具备因果闭包。不能只用
最终 query 的 `0.995` representation ceiling替代 historical rollout。

### O5. Source-interface audit

分别记录 KV-only reconstruction、four-field tape 和 five-field tape 的 bytes、reads 与 release FLOPs。
只有 quality gate 和 compute/I/O gate同时通过，才允许把 finite-difference residual propagation写成
prospective action。

最小实现 correctness tests：

- `theta_C=theta_P` 时所有 delta state 为零；
- full parameter rank、`s=H`、native attention 时逐层 K/V 与 Exact Current 一致；
- JVP 与中心有限差分在 `epsilon -> 0` 时一致；
- 改变 future history suffix 不影响 earlier delta state；
- constructor API 不接收 Current Exact K/V 或 score；
- Parent tape bitwise unchanged。

## 9. 它是不是 mapper？

在下面的严格实现中，它不是 mapper：

- factors 只来自 `W_C-W_P`，不来自 Parent/Current KV pairs；
- coefficients 由原 Transformer arithmetic、JVP 或 finite difference产生；
- 不最小化 target-KV、readout 或 score loss；
- full-rank/no-truncation 有 computation-graph exact limit；
- user-specific输入只有 raw history 与 Parent execution state。

以下任一变化会把它推回 mapper 或 fitted translator：

- 学习 `KV_P -> KV_C`、hidden decoder 或 layer transfer matrix；
- 用 Exact Current users 训练 state basis、rank scheduler 或 correction coefficients；
- 用 held-out score 选择 parameter/state ranks；
- 在 source tape 缺失时引入 learned activation reconstruction。

但“不是 mapper”不等于“有论文创新”。从参数更新中提取低秩 factors 与低秩适配的核心代数已有
[LoRA](https://arxiv.org/abs/2106.09685)；用 JVP 线性化网络也不是新操作，已有工作明确使用 implicit
JVP/VJP 做 linearized-network adaptation
([Fast Adaptation with Linearized Neural Networks](https://proceedings.mlr.press/v130/maddox21a.html))。
跨模型 cache reuse 也已有 activated-LoRA serving
([aLoRA KV reuse](https://arxiv.org/abs/2512.17910))，而直接拟合跨模型 KV 的路线则以
[Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893) 为明确反例边界。

所以不能把“SVD model delta + JVP”本身写成 Design contribution。只有下列完整现象同时成立，才可能
达到论文高度：

1. release parameter delta 低秩；
2. induced **dense user-state defect** 在 Transformer nonlinear boundaries 后仍保持可截断的低秩；
3. delta-only execution 从 Parent tape 传播该 defect，而不物化完整 Current KV；
4. all-history functional compiler 保留 native query interaction；
5. exact-limit、质量、FLOPs 与 source I/O 同时闭合。

这时可贡献的是“persistent Transformer state 的 release-delta execution”，而不是 parameter low rank、
自动微分或 moment algebra中的任何单项。

## 10. 当前建议

这条路线值得进入 **oracle-only** 下一轮，优先级如下：

1. 先补全五条 edge 的 parameter-delta rank sums；
2. 先跑 full-rank JVP ceiling 和 exact state-defect spectra；
3. 只有 `s=8` 有明显功能充分性时，再实现 low-rank finite-difference residual pass；
4. 同时把 four/five-field Parent tape 当作显式系统代价，而不是免费输入；
5. 若 KV-only 接口必须保持不变，则立即判本路线不可执行，不继续调 rank。

因此当前最准确的状态是：**有清楚 exact-limit 和非-mapper语义的机制候选，但在现有 source-state
接口下不可行；在 richer tape 下只有一个尚未经证据支持的、非常窄的 `<20%` 乐观窗口。**
