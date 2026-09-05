# Release differential structure：全历史稠密支持与 release 子空间预检

日期：2026-09-02  
状态：**纯 oracle diagnostic；未运行 GPU、未建立合同、不是 Design 1**

## 1. 要回答的问题

Insight 1 已经说明，单个 token、局部 layer 或局部 K/V 的重要性不能稳定转化为低成本迁移。
这里测试一个与 locality 正交的可证伪命题：

> 相邻 release 造成的历史 K/V 差异是否遍布全历史 token，却主要落在一个由 release 参数变化
> 决定的小 feature subspace 中？

这个命题简称 **support-dense but release-subspace low-rank**。其中两个部分必须同时成立：

- `support-dense` 指 exact `Current - Parent` 差异在 token 位置上没有缩成少数 support；
- `release-subspace low-rank` 指这些位置共享少量 feature directions，而且这些 directions 能由两版模型的
  参数差分事先确定，不是从每个用户的 Current Exact cache 拟合出来。

即使命题成立，SVD 和低秩压缩本身也不是论文设计。它们只回答“差异结构在哪里”。真正可能进入
Design 1 的下一步，是从 dependency-free layer-0 `Delta KV` 取得 history-mode seed，让 Parent/Current
两条路径在同一个低维 history quotient 中耦合传播 release differential，最终只把低秩 signed
`Delta KV` 加到 exact Parent cache；不是保存一个由 Current Exact 拟合的低秩 cache。

## 2. 两级 oracle

对 edge `P -> C`、用户 `u`、layer `l`，只取该用户真实有效历史长度 `N_u`，定义

\[
D_l^u=[K_l^C-K_l^P\;|\;V_l^C-V_l^P]
\in\mathbb R^{N_u\times 2W}.
\]

PAD 行不得进入 SVD、support 或平均值。每个 oracle 都重建

\[
\widehat K_l=K_l^P+\widehat{\Delta K_l},\qquad
\widehat V_l=V_l^P+\widehat{\Delta V_l},
\]

然后才能通过 Current reader intervention 测 response、最终用户表示和 recommendation gap recovery。
只报告 tensor cosine 不足以通过结构门。

### 2.1 Oracle A：per-user joint `Delta[K,V]` truncated SVD

对 `D_l^u` 做一次 thin SVD：

\[
D_l^u=U\Sigma V^\top,
\qquad
\widehat D_l^u(r)=U_{:r}\Sigma_{:r}V_{:r}^\top,
\]

固定 rank grid 为 `1/2/4/8/16`。K/V 必须联合分解；分别给 K 和 V 一个 rank `r` 会把容量翻倍，
不能再称 joint rank `r`。同一矩阵只做一次 full SVD，各 rank 从相同 factors 截断。

这个 oracle 给出该用户该层最优 Frobenius rank ceiling，但 basis 和 coefficient 都读取 Current Exact。
因此它只能判断“共享 feature rank 是否小”，不能成为 migration action，也不能证明 subspace 在用户之间
共享。

实现同时显式返回 `U_:r` 这一 token/history-mode basis，供后续检查 layer-0 seed 是否能跨层维持同一
quotient。当前 per-layer independent oracle **没有**完成这个跨层 transfer test：上层各自低秩不等于
layer-0 history basis 足以支撑 coupled replay。

### 2.2 Oracle B：projection-weight-delta left subspace

PyTorch linear weight 的布局是 `[output,input]`。对同一层的 K projection，令

\[
\Delta W_K=W_K^C-W_K^P=U_K\Sigma_KV_K^\top,
\]

则 `U_K` 是 cache feature/output space 中的方向。V 同理。固定 per-K/V rank cap 为
`4/8/16/32`，并投影 exact user delta：

\[
\widehat{\Delta K}=\Delta K U_{K,r}U_{K,r}^\top,
\qquad
\widehat{\Delta V}=\Delta V U_{V,r}U_{V,r}^\top.
\]

这里 basis 只依赖两版对应 K/V projection weights，是 edge-global object；但是用户 coefficient
`Delta K U_K` 和 `Delta V U_V` 仍读取 Current Exact，所以这依然只是 ceiling。若 `Delta W` 的数值秩
小于请求 rank，必须截到真实支持秩。SVD 对零奇异值给出的任意 null-space completion 不属于 release
差分，不能用它提高 recovery。

参数子空间 ceiling 有一个重要的保守解释。实际

\[
\Delta K_l=X_l^C(W_K^C)^\top-X_l^P(W_K^P)^\top
\]

同时含直接 parameter term 和从下层传播来的 state term；faithful HSTU 还可能在 projection 后经过
SiLU。因此 ceiling 失败只说明“当前层 K/V weight-delta output subspace 不足”，不能单独否定沿完整
Transformer 参数路径递推 differential。ceiling 成功也仍不提供合法 coefficient constructor。

## 3. 三组不可混淆的统计量

### 3.1 Exact token support density

令每个历史位置的 exact energy 为 `e_i=||D_i||_2^2`。报告：

\[
\mathrm{PR}_{token}=\frac{(\sum_i e_i)^2}{N\sum_i e_i^2},
\]

以及覆盖 90% exact energy 所需的 token fraction、非零 token fraction。归一化 participation ratio 接近
1 表示许多 token 共同参与，接近 `1/N` 表示单点支持。这些量必须从 exact `D` 计算；从截断
coefficient 计算的 participation 只能叫 `captured-component participation`，不能证明原始 drift 稠密。

### 3.2 State reconstruction 与 reader recovery

每层、每 rank 报告 unclipped relative-L2 recovery、captured delta energy、residual norm 和 cosine。
随后把 `Parent + reconstructed delta` 注入 Current reader，在同一请求上报告：

- query-conditioned attention response recovery；
- post-block residual / final-user-representation recovery；
- Reuse-to-Current-Exact recommendation gap recovery。

聚合顺序固定为 user-equal、layer/head 按预注册规则、最后 edge-equal。不能删掉 small-gap 用户或只选择
有利 layer。ranking 与 next-item workload 若以后共同运行，必须分别报告，不能互相补偿。

### 3.3 Release parameter spectrum

每个 edge/layer 分别报告 `Delta W_K`、`Delta W_V` 和 stacked `[Delta W_K;Delta W_V]` 的 singular
values、stable rank、entropy effective rank、numerical rank、rank@90/95/99 energy，以及固定 grid 的
captured energy。参数 spectrum 是 model-only 背景证据；它不经过用户 trajectory，不能代替 Oracle B
的 causal ceiling。

## 4. 建议的 prospective falsification gate

以下门只用于审核下一轮 32-user / 5-edge diagnostic 是否值得立合同；本文件没有冻结它，也没有授权
运行。R16 是唯一主结构点，R32 只解释失败是否来自容量不足。

1. **Dense-support gate**：五条 edge 至少四条的 edge-level exact token participation ratio 不低于
   `0.50`，且覆盖 90% exact delta energy 至少需要 `0.50` 的有效历史 token；同一 edge 的
   nonzero-token fraction 不低于 `0.90`。
2. **User-optimal rank gate**：joint token-SVD R16 在至少四条 edge 上达到 edge-equal captured
   delta energy `>=0.90`，并在 Current-reader intervention 中达到 response recovery `>=0.80`。
3. **Release-derived gate**：parameter-left-subspace R16（per K/V）在至少四条 edge 上达到 captured
   delta energy `>=0.80` 和 reader-response recovery `>=0.75`；任何 edge 均需完整报告六层，且不能靠
   R32 替代主门。
4. **Decision gate**：只有前三项同时成立，才允许开发 layer-0-seeded、shared-history-quotient 的
   two-version differential propagation。Oracle coefficient 的高 recovery 本身绝不允许写成 Design
   accuracy，也不计入 `0--20%` frontier。

这些 threshold 是明确的可证伪预案，不是已有结果。正式合同若采用不同值，必须在读取任何 GPU 输出
前解释并冻结。未来 runner 还应对 user-edge paired recovery 做固定 seed bootstrap，并报告所有五条
edge；不能按结果选 edge。

门的不同结果对应不同裁决：

- dense support + token R16 高 + parameter R16 高：保留结构 Insight，进入 causal differential
  propagation；
- token R16 高但 parameter R16 低：只是 per-user low-rank compressibility，拒绝把它发展成 mapping；
- parameter spectrum 低但 Oracle B 低：传播 state term 或 nonlinear path 主导，不能从 `Delta W` 谱
  推断 cache 可迁移；
- exact support 不稠密：本命题被否证，应回到 dependency-closed support 分析，而不是修改 density
  指标；
- R16 失败、只有 R32 成功：只保留容量诊断。R32 coefficient storage 已接近 20% 上界，不能反向降低
  Design 门槛。

## 5. 计算与存储语义

实现采用仓库既有的 multiply/add FLOP 口径。Medium 是 `L=6,H=W=192,N=1024,heads=6`；成本函数
明确只覆盖 `W=H` 的当前配置。

两条路径都先物化完整 Current Exact K/V，所以每用户 generation floor 已经是 `100% Exact-All`：

- token-SVD 还要做 exact delta、每层一个 user-specific `N x 2W` thin SVD、`U_r Sigma_r` 和重建；
- parameter-subspace 还要做 exact coefficient projection 和重建；K/V weight-delta SVD 是每个 edge
  一次的 release-shared 成本，必须单独报告，不能在 one-user cost 中消失；
- SVD FLOPs 使用公开的 classical thin-SVD estimate
  `4*max(m,n)*min(m,n)^2 + 8/3*min(m,n)^3`。实际 `torch.linalg.svd` backend 可能不同，故该项是
  audit estimate，不冒充 wall-clock measurement；其他 subtraction/matmul/add 分项为固定计数。

静态 Medium audit 中，Exact-All 为 4,771,282,944 FLOPs/user。joint token-SVD R16 连同 Current Exact
为 `196.62%`；parameter-subspace R16 的 per-user coefficient projection/reconstruction 连同 Current
Exact 为 `103.26%`，若仅以一个用户承担 release-shared basis construction 则为 `115.14%`。R32 前者
不变，parameter ceiling 为 `106.43%`（one-user unamortized 为 `118.30%`）。这些是 diagnostic
成本，不是 0--20% candidate estimate。

增量 factor storage 不含已有 Parent K/V：

\[
S_{token}(r)=Lr(N+2W),
\qquad
S_{parameter,user}(r)=2LNr,
\qquad
S_{parameter,shared}(r)=2LWr.
\]

Medium R16 下，三者分别为 135,168、196,608 和 36,864 个 FP32 scalars；前两者分别是完整 Current
K/V 的 5.73% 和 8.33%。full singular spectra 是 transient diagnostic metadata，不在迁移 factor
storage 中。低存储不能改变 coefficient 来自 Current Exact、总生成计算超过 100%、因而两者均不是
Design 的事实。

rank grid 的探索执行会共享一次 full SVD，但每个 rank 的 factor scaling/projection/reconstruction 仍有
各自成本。`release_differential_oracle_cost()` 报告一个指定 rank 的成本，不把整张 grid 偷换成单点。

## 6. 实现与下一步边界

`scripts/insight_two/release_differential_structure.py` 提供无数据加载、无 launch side effect 的纯函数：

- joint exact `Delta[K,V]` SVD 与 Parent-plus-delta cache reconstruction；
- parameter-delta left subspace、数值支持秩约束和 exact-delta projection；
- exact/captured token participation、tensor recovery 与参数 spectrum；
- 每层 model spectrum 汇总；
- Medium-compatible oracle compute/storage accounting。

`tests/test_insight_two_release_differential_structure.py` 覆盖低秩 exact reconstruction、每层一次 SVD
复用、null-space completion 反例、参数支持方向、已知 spectrum 和 oracle 成本边界。测试只使用 CPU
toy tensors。

在新 prospective contract 之前，不接 checkpoint、不选 UID、不运行四卡。若结构门未来通过，下一份
设计文档必须回答：如何由可直接形成的 layer-0 `Delta KV` 固定 history quotient，只读 exact Parent
persistent state 和 edge-global parameter delta，在 Parent/Current 共享 quotient 中沿 causal layer path
耦合产生 user differential coefficients，并在 full-rank/no-truncation 极限恢复相应 finite-release
computation。最终 intervention 只能向 exact Parent cache 加 signed low-rank `Delta KV`。任何直接读取、
回归或聚类 Current Exact coefficient 的方案仍是 mapper，不构成 Design 1。
