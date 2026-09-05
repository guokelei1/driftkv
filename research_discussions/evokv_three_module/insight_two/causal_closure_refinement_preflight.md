# Recursive functional-delta closure：机制精炼与非正式反例检查

日期：2026-09-02  
状态：**仅使用 discovery UID index 0 的非正式机制诊断；不是 prospective result，不得用于冻结数值结论**

## 1. 结论先行

目前最有论文意义的 Design 1 候选不是 moment、address clustering、KV mapping 或若干压缩技巧的
组合，而是下面这个 state-transition semantics：

> 完整 Parent cache 是精确 control path；少量 Current carrier 形成 query-native 的 paired response
> defect。已经 commit 的 defect 不只在未来请求中修正 Parent reader，还被后续历史 carrier 读取，
> 通过原 Transformer block 递归生成自己的 upper-layer continuation。

这条机制可以简称为 **recursive functional-delta closure**。命名不是贡献；真正需要实验支持的命题是：

~~~text
the migrated functional delta is simultaneously
  serving state for future recommendation queries, and
  causal prefix state for constructing its own higher-layer continuation.
~~~

一轮 1-UID、五条 edge 的非正式检查显示，当前严格 commit 的默认 recurrence 在完全相同 support、
mass、Parent base 和 Current model 下，五条 edge 都优于 independent carriers；平均 logit-gap recovery
从 `0.8399` 提高到 `0.8861`。这只是很小的机制预检，不能替代 32-user prospective canary，也尚未达到
预注册的 `+0.10` novelty-gain 门槛。

更重要的反例是：三个表面上“更精确”的修改——强制 prefix mass 守恒、把 exact Current layer-0
response 作为默认 seed、允许 current carrier 在同一步承接自己的历史 mass——都没有稳定提高五条
edge。这说明当前的 sparse self-consistency 和 conservative causal admission 不是应被随手修掉的
数值瑕疵。

## 2. 默认 recurrence 的严格语义

设最终选中的真实历史 carriers 按 source position 排序为
`c_1 < ... < c_R`，由 candidate-free layer-0 address rule 得到最终 assignment `a(j)`。对位置
`i=c_g` 的 carrier，当前实现只允许已经严格 commit 的中心 `h<g` 承接 prefix source mass：

~~~text
w_h(i) = | { j < i : a(j)=h } |,  h < g;
w_h(i) = 0,                         h >= g.
~~~

第 `l` 层 carrier 的 response 因而是：

~~~text
r_hat_l,i = A_l^C(q_hat_l,i ; Parent_<i)
            + sum_{h<g} w_h(i) [
                rho_l^C(q_hat_l,i, k_hat_l,h) v_hat_l,h
                - rho_l^C(q_hat_l,i, k_l,c_h^P) v_l,c_h^P
              ]
            + Current self response.
~~~

这里未被 admissible carrier 表示的 source mass **没有被删除**：它仍走完整 Parent base，只是尚未获得
Current-minus-Parent correction。随着 carrier 按时间 commit，可用 correction measure 单调增加；任何
future carrier 都不能影响 earlier carrier。这个定义比“每个 prefix 都必须重新分配到已有中心”更保守，
但它避免把一个稀疏代表的误差乘上全部尚无可靠代表的 prefix mass。

当 `R=N`、每个 source position 都是 unit-mass singleton 时，`a(j)=j`，上述递推按时间和层双重归纳
恢复 Exact Current cache。这个 full-support invariant 是算法语义的一部分，而不是质量假设。

## 3. 为什么这不是 mapper

Current carrier 的 upper-layer `K/V` 不是由 `Parent KV -> Current KV` 的 ridge、MLP 或其他外部函数
预测。carrier 的 hidden state 由 Current block 原生计算；它所读的 prefix 是完整 Parent response 加
此前 paired defects。相邻工作的边界应写得很窄：

- [Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893) 用 closed-form ridge mapper 拟合
  target KV；本候选不拟合 target tensor，递归对象是 reader response defect。
- [LESS](https://arxiv.org/abs/2402.09398) 用 learned low-rank recurrent cache 保存同模型被压缩 token
  的历史贡献；本候选的 recurrence 发生在一次 Parent-to-Current release transition 中，并生成自身的
  upper-layer version defect。
- [ResKV](https://arxiv.org/abs/2607.29591) 恢复同模型 eviction 后遗漏的 attention mass；本候选保留
  完整 Parent cache，paired negative arm 显式删除的是 Parent-version response approximation，再加入
  Current-version approximation。
- Linear-attention moments、control variates 和 landmark selection 都可以实现 ledger，但不能承担新意。
  如果删掉 recursive constructor 后质量不降，Design claim 应判失败。

因此最小可辩护的新意是 **functional defect 对 Transformer computation graph 的因果闭包**，不是
`B/M`、采样或 clustering。

## 4. 1-UID 五边机制预检

范围严格限制为 frozen discovery population 的 UID index 0；没有读取 `[512,3000)` confirmation，
没有训练、label fitting 或 candidate-conditioned construction。指标是 held-out odd-32 candidate 的
logit-gap recovery。

| edge | same-support independent | strict recursive closure | closure gain |
| --- | ---: | ---: | ---: |
| `v0->v1` | 0.838318 | 0.947386 | +0.109068 |
| `v1->v2` | 0.748344 | 0.786149 | +0.037805 |
| `v2->v3` | 0.869898 | 0.893227 | +0.023329 |
| `v3->v4` | 0.877508 | 0.893548 | +0.016040 |
| `v4->v5` | 0.865345 | 0.910091 | +0.044746 |
| edge mean | 0.839883 | 0.886080 | +0.046197 |

这个对照固定了 R64 support、final cluster mass、Parent base、Current weights 和 serving reader；唯一
变化是 earlier paired defect 是否参与 later carrier 的形成。因此它是当前最直接的机制信号，但
`n=1` 不能提供稳定性或显著性结论。

另一个必要的反例检查去掉 paired Parent negative arm，只让 earlier Current carrier 作为额外 memory
参与递推。在 exact-layer0 + prefix-reflow 的诊断中，五条 edge 的 recovery 分别只有
`0.465/0.535/0.365/0.748/0.326`，而 paired 版本为
`0.993/0.769/0.642/0.902/0.978`。这支持“version defect”而非“多放一些 memory token”的解释；
仍需 prospective population 复核。

## 5. 被反例否定的“显然改进”

### 5.1 Exact layer-0 + prefix mass reflow

Current layer-0 K/V 只依赖 raw embedding、norm 和 projection，因此可以合法地对全 history 精确生成。
一个自然修改是让 layer-0 carrier 读取 dense Current prefix，并把所有 `j<i` 动态重新分配给当时已经
出现的 centers，使每个非首 carrier 的 represented fraction 等于 1。

这个修改让 upper carrier tensors 更接近 Exact，却没有稳定改善推荐决策：

| edge | exact-seed independent | exact-seed + mass-reflow closure | gain |
| --- | ---: | ---: | ---: |
| `v0->v1` | 0.899678 | 0.993041 | +0.093363 |
| `v1->v2` | 0.827813 | 0.769009 | -0.058804 |
| `v2->v3` | 0.662364 | 0.641545 | -0.020819 |
| `v3->v4` | 0.892580 | 0.901757 | +0.009177 |
| `v4->v5` | 0.907747 | 0.977659 | +0.069912 |

同一个修改在 3 条 edge 很强、2 条 edge 为负，不能升级为 primary。它揭示了两个不同误差：

1. constructor 的 carrier-state error；
2. constructor state 与最终 sparse serving reader 的 measure mismatch。

前者变小不保证后者变小。强制把全部 prefix mass 交给很少的已有 centers 还会放大 quadrature error。

### 5.2 Current-center same-step admission

在一层内部，当前 carrier 的 `q/k/v` 在 attention response 前已经产生，所以从纯因果性看，可以让
`c_g` 立即代表 `j<c_g, a(j)=g` 的历史 mass，而不读取 future token。这个修改保留 full-support
invariant，也几乎没有新增 FLOPs。

但只在 exact dependency-free layer 0 启用时，五条 edge 相对 independent 的 gain 是
`+0.116/+0.054/-0.004/+0.016/+0.048`；它在 `v2->v3` 变为负，edge mean 与严格 commit 几乎相同。
允许所有层 same-step admission 也没有改善该反例。因此“本层 K/V 已可用”只证明合法性，不证明它是
可靠 quadrature atom；默认仍应保持 commit-before-read。

### 5.3 Chronological block carriers

固定 midpoint、block start/end，以及 block 内 layer-0 address medoid 都做了 1-UID 诊断。它们能让
previous block mass 天然 causal-complete，并明显降低 selector 成本，但结果对 edge 和代表位置非常
敏感；出现过低于 Reuse 的 recovery。它们不能替代全局 address support，也不能被包装成新方法。

### 5.4 Serving-side exact layer-0 correction

将 serving reader 的 layer-0 correction 换成 dense exact response，或合法的 full layer-0 affine
moment，几乎不改变 mass-reflow recurrence 在 `v1->v2`、`v2->v3` 的负 closure gain。因此失败来自
upper-layer constructor propagation，不只是最终 layer-0 injection mismatch。

## 6. 当前应冻结的算法候选

在 formal canary 给出相反证据之前，R64 primary 应保持以下结构，不把上述失败修改混入：

~~~text
raw history
  -> dependency-free Current layer-0 projection for legal address support
  -> carriers sorted by real source time
  -> full causal Parent-prefix read at every layer
  -> strictly earlier, prefix-valid paired defects only
  -> native Current block produces the next-layer carrier K/V
  -> commit carrier after its full block trajectory is complete
  -> persist Current carrier atoms + Parent indices/masses
  -> future query reads Parent base + paired delta through native attention
~~~

这里 address rule 是一个可替换 constructor component；moment 是一个可选 compiler；二者都不写进
核心贡献句。核心贡献句只写 recursive functional closure，并用 matched no-recursion ablation 决定它
是否成立。

## 7. 成本边界

按当前 Medium `6L/H192/N1024/R64` 审计，strict recursive R64 的 expected-position 成本为
Exact-All 的 `16.4849%`：其中 neural generation `13.2653%`，address selection `3.2196%`。即使
64 个 unique carriers 全落在成本最高的位置集合，审计上界也是 `19.4518%`，仍在 `20%` 内。
recursive signed-atom reads 本身是 `18,943,488` FLOPs/user，约为 Exact-All 的 `0.397%`。

最小持久 sidecar 只保存 Current carrier K/V、Parent source indices 与 mass；FP32 大小约为完整
Current KV 的 `6.2609%`。Parent arm 复用已经存在的 Parent cache，不应为方便 intervention 而在系统
实现中重复持久化。

这些是理论计数，不是 wall-clock 结论。正式结果仍必须使用每个 UID 的 observed carrier-position sum，
并单独报告 metadata、I/O 与 kernel efficiency。

## 8. 可证伪条件

以下顺序不能颠倒：

1. `P=C` 时 paired ledger 为零；`R=N` 时 native recurrence 恢复 Exact Current cache、stage response
   和 score。
2. R64 strict closure 与 independent 使用完全相同 support/mass；只有 recursive read 开关不同。
3. strict closure 必须在 prospective population 上稳定优于 independent；若 bootstrap interval 包含零，
   不能把 recursion 写成主要贡献。
4. 去掉 Parent negative arm 必须显著下降，否则它只是 extra-memory replay。
5. moment compiler 成功、native recurrence 失败时，只能得出 activation algebra 可压缩，不能宣称
   functional causal closure。
6. exact layer-0、mass reflow、same-step admission 与 chronological block variants 保留为事前定义的
   negative/mechanism ablations，不根据 32-user outcome 选择性升级。
7. 若 formal closure gain 很小，仍可保留 Insight 2 的 functional-boundary observation，但 Design 1
   必须继续探索；不得靠 mapper、调参或重新挑 support 来补出论文新意。

## 9. 对 Insight 2 的最小表述

如果 prospective 结果支持上面的机制，Insight 2 可以比“aggregation 后低维”更进一步，但不应写得
超过证据：

> 跨版本 token-state error 在 reader aggregation 后形成紧凑的 query-conditioned response defect；
> 更关键的是，这个 defect 可能具有 causal compositionality：它可以作为近似 Current prefix，驱动
> 后续历史事件通过原 Transformer block 生成上层 defect，而无需先恢复完整 Current token state。

这条表述同时满足 Transformer 特异性、推荐系统的持久用户状态价值和单 Parent-to-Current edge 边界。
它不声称所有架构共享同一个 moment，也不把一次 1-UID 预检写成结论。
