# Source-certified reduced execution：absolute residual 与 finite-defect residual 双重否证

日期：2026-09-03  
状态：**严格 NO-GO / RETIRE；固定 UID1930、五 edge、held-out odd-32 的非正式机制否证；未读取 confirmation 或 label**

## 1. 裁决先行

本轮没有把 exact Parent 当作一个待映射到 Current 的 tensor target，而是把它当作已知正确的 source
Transformer trajectory。目标是检验：已经物化的 Parent K/V 能否在每个非终止 block 内部提供少量
exact source response，作为 reduced Current execution 的残差证书。

在观察结果之前固定了两个递进构造，均使用 `Parent rank4 + Current rank4`、同一 matrix-free input、
同一 range finder、同一 DEIM rows 和同一 paired-native serving reader，不做 rank、pivot、layer、lift 或
damping sweep：

1. **absolute-source residual closure**：在四个 test rows 上计算 exact Parent block update 与 reduced
   Parent update 的残差，将其插值后同时加入 Parent/Current 两臂；
2. **finite release-defect closure**：不搬运 absolute Parent residual，而在同一 rows 上构造 exact
   Parent-anchored 的 Current-minus-Parent block-update residual，只把该残差加入 Current/defect arm。

运行前 admission 冻结为：constructor `<20% Exact-All`；五边全部为正且 `>=.80`；edge mean 必须超过
两个 single-r8 controls 的较强者（约 `.9365`）；相对每个 single control 至少赢 `3/5` edge。

结果如下：

| method | v0→v1 | v1→v2 | v2→v3 | v3→v4 | v4→v5 | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| absolute-source residual closure | -.003 | .729 | .896 | .911 | .686 | **.644** |
| finite release-defect closure | .212 | .672 | .667 | .945 | .815 | **.662** |
| paired-r4/r4 native deletion control | .872 | .917 | .955 | .933 | .824 | **.900** |
| single-Current-r8 factor reader | .913 | .997 | .851 | .981 | .937 | **.936** |
| single-Current-r8 shared-`U0` | .861 | .914 | .985 | .947 | .975 | **.936** |

finite-defect 版本只有 `2/5` edge 达 `.80`，只在 `1/5` edge 胜 paired deletion，且对两个 single-r8
controls 都是 `0/5`。两个构造均明确 **RETIRE**；不扩用户、不建 formal contract、不读取 confirmation。

更重要的是，prior-art 审计也给出独立 NO-GO：本实现的“少量残差点 + trial-space 插值 + reduced
recurrence”是标准 sampled-residual hyper-reduction / DEIM 形态；“exact/approximate attention residual
作为 correction”又与 attention control-variate 文献直接相邻。即使数值通过，它也不能仅靠
`source-certified` 命名成为论文 Design。

## 2. 第一个可验证 invariant：Parent K/V 是 source response certificate

对第 `l` 层 Parent RMS-normalized state `N_l^P`，legacy checkpoint 的 K/V 线性投影满足

\[
[K_l^P,V_l^P] = N_l^P [W_{K,l}^{P\top},W_{V,l}^{P\top}].
\]

六个 frozen checkpoint 的 joint projection 均满列秩。因此对任意少量位置集合 `I_l`，可只读取这些
位置的 exact Parent K/V，并通过模型全局 pseudoinverse 恢复 exact `N_l^P[I_l]`。随后这些 source
queries 读取 exact causal Parent prefix，经过原生 activation、`W_O` 和 gate，得到 exact Parent block
update：

\[
U_l^{P,*}[I_l]=U_l^P(N_l^P;K_l^P,V_l^P)[I_l].
\]

这个 invariant 不需要 Parent hidden tape，不读取 Current Exact，不使用 candidate 或 label。CPU test
逐层证明 selected-row update 与完整 Parent native block 相同。

但它只证明“可以便宜地问 exact source 几个问题”，并不证明这些问题足以决定全历史 release defect。

## 3. Absolute-source residual closure 及其失败原因

令 `Q_l in R^{N x 4}` 是 reduced Parent normalized-state factors 的正交 trial space，`I_l` 是确定性
DEIM rows。第一条路径形成

\[
e_l[I_l]=U_l^{P,*}[I_l]-\widehat U_l^P[I_l],
\]

并通过

\[
E_l=Q_l(Q_l[I_l])^{-1}e_l[I_l]
\]

将 residual lift 到全部历史位置，然后同时更新两臂：

\[
\widehat X_{l+1}^P=\widehat X_l^P+\widehat U_l^P+E_l,
\]

\[
\widehat X_{l+1}^C=\widehat X_l^C+\widehat U_l^C+E_l.
\]

构造满足：

- test rows 上 residual 被插值到数值精度；
- `Parent=Current` 时两臂仍相同；
- full token rank、无截断时恢复 exact Parent/Current trajectory 与 exact Current reader；
- API 不接受 Current Exact state。

严格成本为：

```text
paired-native r4/r4 base             872,238,088
source decode/response/DEIM/lift      33,428,910
total                                905,666,998
fraction Exact-All                       18.9816%
```

成本通过但质量从 paired deletion 的 `.900` 降到 `.644`。这不是 DEIM square matrix 的普通 condition
failure：五边 25 个 certificate 的 condition number 约 `1.7--5.4`，selected residual 也确实被满足。
真正的问题是 `Q_l[I_l]` 的绝对行尺度很小，lifted full-history residual norm 通常约为 sampled residual
norm 的 `6--24x`；更根本地，一个 absolute source correction 不具有跨版本平移不变性。

虽然 `E_l` 在当前一步的两臂相减中消失，但下一层看到的是

\[
\Gamma_l(E_l)=
[F_l^C(\widehat X_l^C+E_l)-F_l^C(\widehat X_l^C)]
-[F_l^P(\widehat X_l^P+E_l)-F_l^P(\widehat X_l^P)].
\]

RMSNorm、query--key activation、gate 与两版不同参数使 `Gamma_l(E_l)` 一般不为零。也就是说，
**一个对 Parent 绝对执行正确的残差，不是 Current release defect 的合法共同 correction。**

## 4. Finite release-defect closure 仍然失败

第二条路径据此不再把 `E_l` 加到 Parent。它先恢复 exact source normalized row，再构造

\[
\widetilde N_l^C[I_l]
=N_l^{P,*}[I_l]
+(\widehat N_l^C-\widehat N_l^P)[I_l].
\]

同时定义一个 logical Current prefix：

\[
\widetilde K_l^C=K_l^{P,*}+(\widehat K_l^C-\widehat K_l^P),
\qquad
\widetilde V_l^C=V_l^{P,*}+(\widehat V_l^C-\widehat V_l^P).
\]

实现不是三条 response 相减：先把 exact Parent logits 与两臂 factorized key logits 相加，再执行一次
Current 原生 activation；同一 weights 随后读取 exact Parent value 与两臂 value defect，因此保留 logical
endpoint 的完整 finite K-by-V interaction。

在 test rows 上计算 source-anchored release update：

\[
d_l^*=
U_l^C(\widetilde N_l^C;\widetilde K_l^C,\widetilde V_l^C)
-U_l^{P,*},
\]

再减去普通 paired update difference，得到只属于 release equation 的 residual：

\[
c_l[I_l]=d_l^*[I_l]-(\widehat U_l^C-\widehat U_l^P)[I_l].
\]

仅将 `Q_l(Q_l[I_l])^{-1}c_l[I_l]` 加到 Current arm。该版本有两个比 absolute closure 更强的
correctness invariant：

- `Parent=Current` 时 `c_l=0`，不会把 source approximation error 伪装成 release change；
- full token rank 时 logical prefix、normalized row 与两条 update 都到达 exact endpoint，`c_l=0`，完整
  reader 仍精确。

额外 Current row response 全部计入后：

```text
paired-native r4/r4 base                 872,238,088
source-defect certificate                 56,852,910
total                                    929,090,998
fraction Exact-All                           19.4726%
```

它避免了 absolute closure 在 `v0->v1` 的负 recovery，并在 `v3->v4` 达 `.945`，说明“只修 release
equation”比搬运 absolute source error 更符合语义；但五边 mean 仍只有 `.662`。四个 pointwise residual
tests 对完整 distributed/query-dependent release defect 仍然不充分，插值的 exactness 不能变成最终
recommendation functional sufficiency。

## 5. Prior-art collision：为什么数值即使变好也不够

本轮可保留的 Transformer-specific 部分只有：joint Parent K/V 能在少量 causal rows 恢复 source
normalized query，并对完整 prefix执行 native attention response。其余主要算法骨架已有直接来源：

1. [DEIM 原始论文](https://epubs.siam.org/doi/10.1137/090766498) 已定义从少量 interpolation points
   评价 nonlinear term，再由 reduced basis 恢复全维 nonlinear dynamics；本实现的
   `Q(Q[I])^{-1}residual[I]` 正是该形态。
2. [POD-DEIM state-space error](https://epubs.siam.org/doi/10.1137/110822724) 已研究这种 sampled
   nonlinear approximation 如何进入 reduced trajectory error；不能把“层间 residual closure”本身写成
   新的 dynamics principle。
3. [Efficient Attention via Control Variates](https://arxiv.org/abs/2302.04542) 已从 exact/approximate
   attention control variates 出发减少 attention approximation gap；paired response subtraction 或
   exact source anchor 不能独立承担 novelty。
4. [HeadQ](https://arxiv.org/abs/2605.03562) 与 [OptR](https://arxiv.org/abs/2608.02691) 已明确把 K/V
   error 放在 query-visible score、attention-weighted value 和 post-attention output 空间中；“用 reader
   response 当 certificate”也不是新的宽泛 Insight。

因此本轮没有找到一个“Transformer-only causal test functional”：DEIM rows 来自 generic trial-space
algebra，而不是 attention 因果图特有的守恒量、边界条件或 exact quotient。它既在 matched controls 上
数值失败，也没有越过 prior-art gate。

## 6. 冻结的 NO-GO 与下一步边界

本轮冻结以下负面结论，后续不得通过改 rank、pivot、oversampling、damping 或 layer schedule 复活：

> **Exact Parent response 是合法且便宜的 source certificate，但少量 sampled source residual 不能作为
> 跨版本 Transformer trajectory 的迁移对象。Absolute residual 不可跨版本搬运；finite-defect
> residual 虽语义更正确，仍无法由 generic trial-space interpolation稳定扩展到全历史。**

这也把“source-certified”一词的边界说清楚：有 exact source test 不等于有 sufficient migration state。
下一候选如果仍采用 sampled residual，必须首先给出一个 DEIM 无法还原的 Transformer causal invariant，
例如一个由 attention composition 严格守恒/闭合、且直接控制 future reader response 的测试泛函；否则
它只是 generic hyper-reduction 在 KV migration 上的应用。

如果找不到这样的 invariant，正确结论不是调大 test count，而是：在现有 KV-only interface 与 `<20%`
预算下，当前证据只支持将 single-r8 保留为数值 baseline，尚无论文级 Design 1。模型—系统协同的新
state interface 也必须带来新的可执行信息源，不能只是多存一份供 DEIM sampling 的 Parent activation。

## 7. 实现与验证

- `scripts/insight_two/source_residual_closure.py`：joint-KV row decoder、exact causal source response、
  logical Current endpoint、两种 recurrence、full-rank/zero-release invariants 与两个 Medium ledger；
- `scripts/insight_two/run_source_residual_closure_preflight.py`：固定 UID1930、五 edge、四项 hard controls；
- `tests/test_insight_two_source_residual_closure.py`：selected-row exactness、DEIM interpolation、两种
  full-rank limit、zero-release 与成本。

Focused verification：`7 passed`；相关文件 `ruff check` 与 `py_compile` 通过。GPU runner 总时约
`33.3s`，未写 formal result、contract 或 seal。
