# Insight 2 preflight：先聚合、后迁移的 whole-history response operator

日期：2026-09-02
状态：**已停止（contextual-freeze 既有证据否证）；不得实现为 Design 1**

## 0. 事后裁决：contextual freeze 已足以在 preflight 停止本路线

本文初稿把 upper-layer contextual-state freeze 留作 D2 的首要可证伪门。现有单 UID
gauge/reprojection 观察已经给出五条 rolling edge 的近似 recovery：

~~~text
0.64 / 0.26 / -1.95 / 0.13 / 0.61
~~~

这不是“数值还需要调参”：只有两条 edge 呈现中等正 recovery，其余明显不足，且一条
edge 为灾难性负值。它直接违反本文 D2 要求的跨 edge 稳定性，也远低于 Design 所需的
`0.80`–`0.90` recovery 区间。因此不应再先做 D0/D1 来美化 weight geometry 或 shared
activation region；核心因果门已经失败。

这也必须诚实地重画 mapper 边界。本路线的 `T=(W_Y^P)^dagger W_Y^C` 即使不用 matched
target states 拟合，仍然是一个从 Parent joint-K/V 坐标到 Current projection 坐标的解析线性映射。
“先 aggregation、后 congruence transport”只改变了执行顺序，没有修复被 freeze 掉的 Current
contextual trajectory，也不会使这个 mapping 自动变成新的 Transformer 迁移机制。已有 recovery
说明这条边界不只是写作风险，而是实际的功能失败。

裁决：**本路线在 preflight 直接 RETIRE**。下文保留为机制推导和成本记录，不再表示
待启动实验，也不能作为“不是 mapper”的证据。

## 1. 为什么必须离开 token support

最新正式 32-user、五 edge canary 已经把证据分成了非常清楚的两部分：

- 使用完整 Current 历史状态形成的 positive affine bulk，在固定 history-item probes 下达到
  `0.9951` edge-equal recovery、`0.9873` minimum edge；`P=32` 仍为 `0.9949`。因此 compact
  reader-side representation 是强证据，不是 candidate-panel artifact；
- 同一 response difference 一旦由 `R=64/128` 个 token support 构造，甚至允许 exact-state carrier
  oracle 或 recursive causal closure，质量仍然失败。R64 exact carrier oracle 为 `-0.4532`，R128
  也只有 `0.3719`；合法 R64 recursive closure 为 `-0.1962`。

这说明当前瓶颈不是“carrier 的生成方式还不够聪明”，而是 stable-region version drift 的质量本身
分布在整个历史上。下一条路线不应继续改变采样、聚类、权重或 token selector，而应问：

> 在 recommendation queries 共享 activation region 时，能否先把**所有**历史状态归约成 reader
> 真正消费的算子，再在这个算子上执行一次 Parent→Current 迁移？

候选 Insight 2 因而进一步收紧为：

> 跨版本误差不是由少量重要 token 承担；它由全历史共同形成，但在共享 activation region 内，
> Transformer reader 对这段历史只暴露一个可结合的低阶 response operator。若相邻 release 的
> projection drift 在 Parent cache 的原生 K/V 坐标中可表达，则**版本迁移与全历史 aggregation 可以
> 交换顺序**：先归约 Parent 历史，再迁移 operator，而不翻译或重建逐 token Current K/V。

这里真正需要验证的新结构是“migration commutes with aggregation”，不是 outer product、线性映射或
moment tensor 本身。

## 2. 架构边界

本 preflight 只针对当前 Medium legacy reader：6 layers、hidden 192、6 heads、head width 32、
context 1024、`ELU+1`、无 relative-position bias。它不是原始 SiLU HSTU 或一般 softmax Transformer
的定理。

对一层一 head，在固定 positive region `P` 中：

~~~text
A(q) = B_P + s q M_P + N(q),
B_P = sum_{i in P} v_i,
M_P = sum_{i in P} k_i^T v_i.
~~~

`N(q)` 是 negative exponential branch。正式 canary 中它的 response fraction 较小，但不为零；
因此本方法仍是 positive-bulk migration，不能写成完整 attention 的精确等式。softmax 若要采用同一
原则，必须迁移 numerator/partition 二元状态，不能直接套用这里的 `B/M`。

## 3. 核心机制：在 Parent 原生 joint-K/V 坐标中迁移 response operator

### 3.1 每 head 的架构给定坐标，而不是拟合 mapper

设第 `l` 层 Parent normalized token state 为行向量 `z_i^P in R^H`。对 head `h`：

~~~text
k_i^P = z_i^P W_K^{P,h},
v_i^P = z_i^P W_V^{P,h},
y_i^P = [k_i^P, v_i^P] in R^{2d}.
~~~

将同一 head 的 Parent projection weights 拼成：

~~~text
W_Y^{P,h} = [W_K^{P,h}, W_V^{P,h}] in R^{H x 2d},
W_Y^{C,h} = [W_K^{C,h}, W_V^{C,h}] in R^{H x 2d}.
~~~

release-shared、完全由两版模型参数决定的坐标 transport 为：

~~~text
T_h = (W_Y^{P,h})^dagger W_Y^{C,h} in R^{2d x 2d}.
~~~

把 `T_h` 的 Current-key 与 Current-value 输出列记为 `T_K,h` 和 `T_V,h`。若 Current projection
columns 落在 Parent joint-K/V column span 中，则对**相同 contextual state**：

~~~text
k_i^{C|P-state} = y_i^P T_K,h,
v_i^{C|P-state} = y_i^P T_V,h.
~~~

这里没有训练、ridge calibration、target-KV pair 或用户标签。`2d=64` 由 block architecture 固定，
不是从 canary 调出来的 rank。更重要的是，执行路径不应 materialize 上面两条 tokenwise 结果；它们
只用于推导下面的 aggregate identity。

### 3.2 先归约，再做 congruence transport

对一个由 candidate-free history probes 决定的 head-wise shared region `P_h`，直接从完整 Parent
cache 归约：

~~~text
m_h = sum_{i in P_h} y_i^P,
G_h = sum_{i in P_h} (y_i^P)^T y_i^P.
~~~

所有 `N=1024` 个 active history event 都参加；没有 token selection、importance、cluster mass 或
coreset。Current-on-Parent-state 的 functional operator 可在归约以后得到：

~~~text
B_tilde_C,h = m_h T_V,h,
M_tilde_C,h = T_K,h^T G_h T_V,h.
~~~

Parent operator 直接是 `m_h/G_h` 的对应 block：

~~~text
B_P,h = m_h[value columns],
M_P,h = G_h[key columns, value columns].
~~~

最终 sidecar 是：

~~~text
Delta B_h = B_tilde_C,h - B_P,h,
Delta M_h = M_tilde_C,h - M_P,h.
~~~

关键交换律为：

~~~text
Reduce({y_i T}) = Transport_T(Reduce({y_i})).
~~~

也就是说，系统不执行 `Parent KV -> approximate Current KV -> attention`；它执行
`complete Parent empirical measure -> masked first/second-order operator -> release transport`。这条交换律
才是候选机制。若论文最后只剩 `yT`，该路线应被判为普通线性 cache mapping，不能声称新 Design。

### 3.3 为什么使用一个 shared Parent region

Current 和 Parent 分别建 mask 会要求每 head 做两次 masked Gram reduction，并使当前实现的 worst
cost 越过 20%。更重要的是，它失去“同一个 empirical measure 上比较两个 release”的干净语义。

primary 预先固定一个合法 shared region：

~~~text
chi_l,h,i = majority_p 1[q_l,h,p^C · k_l,h,i^P >= 0],  p in 8 history probes.
~~~

同一个 `chi` 同时用于 Parent operator 和 transported Current operator。probe identities 是等宽固定
history item，不读 serving candidates 或 labels。上一层已经构造的 `Delta B/Delta M` 注入 probe
reader 后再产生下一层 `q_l`，所以 operator construction 按 Transformer layer 顺序推进。

已有 canary 给出两条先验但不能替代新实验：Current/Parent held-out majority agreement 约
`99.45%`，而 full-history `P8/P32` representation recovery 分别为 `0.9951/0.9949`。新的 diagnostic
仍必须直接测 shared-Parent-region oracle；不能把 separate-region 结果偷换过来。

layer 0 不采用 contextual freeze：raw event 对 Current layer-0 K/V 是 dependency-free 的，可以完整
投影并归约。upper layers 才使用 joint-K/V aggregate transport。这样把唯一精确、合法的 Current
whole-history seed 留在算法内，而不展开 quadratic Current prefill。

## 4. 因果和 persistent-state 语义

这个候选处理一次确定的 Parent→Current edge：

1. Parent K/V 仍是完整 base path，bitwise 不修改；
2. release-time compiler 只读 raw history、Parent K/V、Parent/Current weights 和固定 history probes；
3. 每层写入 edge-specific signed response operator `(Delta B_l, Delta M_l)`；
4. future query 由 Current reader 原生地产生 `q_l`，读取 `Parent response + Delta B_l + s q_l Delta M_l`，
   再走原 block output、gate、residual 和下一层；
5. release 后 append 的 event 由 Current reader 生成 native K/V，不属于旧 segment 的 delta；
6. eviction 不能继续保留被删 history 的 mass。若需要 rolling，预先把同一 associative reduction 分成
   固定 chronological segments，删除整段 operator；不能只存一个不可逆 global moment。

这里的因果对象是“完整旧 segment 对未来 query 的 response operator”，不是伪造的 Current token
trajectory。除已经显式保留的 shared-region、joint-chart 与 negative-branch 近似外，最关键的新近似是
upper-layer **contextual state freeze**：它迁移 projection/readout drift，但没有假装
`z_l^P = z_l^C` 是恒等式。

## 5. 与已有工作的严格边界

### 5.1 Linear attention / fast weights

[Linear Transformer](https://proceedings.mlr.press/v119/katharopoulos20a.html) 利用 kernel feature map 与
矩阵乘法结合律，把同一模型的 attention 从二次复杂度改成线性复杂度。
[Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html) 已明确说明 additive
key-value outer products 与 fast-weight memory 的等价性。因此，`sum k outer v`、prefix scan、`qM`
都不能成为本工作的 novelty。

本候选不改变训练时 attention kernel，也不把同版本 Transformer 改造成 RNN。它研究的是两个 release
之间，已存在的 Parent joint-K/V empirical measure 如何在**不生成 target token state**的情况下，
被推送成一个 edge-specific signed response operator。可区分的 claim 只能是 cross-version
aggregate/transport commutativity 及其 persistent lineage。

### 5.2 Tensor Cache

[Tensor Cache](https://arxiv.org/abs/2605.22884) 把同版本、已 eviction 的 K/V 写入 outer-product L2
memory，并用 learned gate/write-rate 与 local softmax cache 融合；论文也明确承认 outer-product identity
本身是已知的。本候选处理仍 active 的完整 Parent history、没有 learned gate、没有 eviction-driven
write，并且 sidecar 是一次 Parent→Current release 的 signed operator difference。

### 5.3 Cross-model KV mapping

[Cross-Model KV Transfer](https://arxiv.org/abs/2608.03893) 从 matched source/target KV calibration pairs
拟合 per-head ridge mapper，并 materialize receiver KV。这里确实也存在一个小矩阵 `T_h`，所以不能用
换名来回避相似性。严格差异必须同时满足：

- `T_h` 只由公开的 projection weights 解析得到，不拟合 matched target states；
- 执行不输出任何 tokenwise target K/V；
- transport 在 complete-history masked reduction **之后**以 congruence 形式发生；
- 持久化对象由真实 Current query 读取，是 release-edge response operator，而非 receiver cache。

若上述四点在实现或写作中丢失，本路线就退化成一个简单 mapper，达不到论文要求。

## 6. Medium 理论成本

沿用仓库的 `6L/H192/6 heads/d32/N1024/P8` Exact-All denominator：

~~~text
Exact-All = 4,771,282,944 FLOPs/user.
~~~

primary 使用一个 shared Parent region、exact layer 0、五个 upper-layer joint-chart reductions：

| component | FLOPs/user | Exact-All |
| --- | ---: | ---: |
| P8 history-probe trace through Parent base | 56,168,448 | 1.177% |
| probe reads of the progressively built operators | 608,256 | 0.013% |
| exact Current layer-0 input + K/V + norm（KV-only 实现） | 240,058,368 | 5.031% |
| layer-0 shared mask + paired full-history moments | 41,736,384 | 0.875% |
| one upper layer: mask read + full `2d` Gram + congruence | 59,652,288 | 1.250% |
| five upper layers | 298,261,440 | 6.251% |
| **total, KV-only layer-0 implementation** | **636,832,896** | **13.347%** |

当前 `project_kv()` 还会执行并丢弃 Q projection；若不先消除这段 dead compute，再加
`75,497,472` FLOPs，honest total 为约 **14.930%**，仍在甜点区。release-shared pseudoinverse 与
`T_h` 构造不按 user 重复；即使单独报告，摊到 30,000 users 也不改变结论。

若 Current/Parent 使用两个独立 mask，KV-only 估计约为 **18.998%**，当前 dead-Q 实现约
`20.58%`。因此 separate-mask 只能做 representation oracle，不能在看过质量以后改成 primary。

持久化的 global `(Delta B, Delta M)` 仍为 `38,016` FP scalars/user，即完整 Current K/V 的
`1.6113%`。若 rolling 使用固定 8-segment ledgers，则为 `304,128` scalars，约 `12.8906%`；这部分
不能在系统结果中隐藏。每 edge 的 `T_h` 总共约 `147,456` scalars，但由全部用户共享。

上述是 prospective algebraic count，不是 runtime measurement。实现必须单独报告 masked Gram 的
FLOPs、Parent-cache bytes read、temporary `G_h` workspace、operator write bytes 和实际 latency。

## 7. 最小可证伪 diagnostic

只做 32 users × 五 edge；不进入 512 discovery，不训练模型，不调 rank。顺序如下。

### D0：release weight geometry（CPU、无用户标签）

每层每 head 计算：

~~~text
epsilon_W = ||W_Y^C - W_Y^P (W_Y^P)^dagger W_Y^C||_F / ||W_Y^C||_F,
condition(W_Y^P),
rank(W_Y^P).
~~~

只报告，不据此搜索 cross-head、邻层或可调 rank。`W_Y^P` 明显退化时，方法在执行前即失败。

### D1：shared-region representation ceiling

用 Exact Current cache 仅作 oracle，比较：

1. 已有 separate Current/Parent region full moments；
2. 新的 shared Parent-region full Exact moments。

若第 2 项 edge-equal recovery `<0.95` 或任一 edge `<0.90`，commuting shared-region 假设失败，停止。

### D2：先判 contextual freeze，而不是先美化 chart

diagnostic-only 地捕获 Parent 每层 normalized token state `z_l^P`，用 Current `W_K/W_V` 直接投影全部
1024 tokens，再构造 shared-region full moments。它是 `Current projection on Parent contextual state`
ceiling，不是合法 serving action。

若该 ceiling edge-equal `<0.80`、少于 4/5 edges 正向，说明主要误差来自 contextual state evolution，
而不是 projection drift；整个 aggregate-then-transport family 直接停止，不能追加 regression、mapper
或 residual predictor。

### D3：joint-chart approximation 与交换律 invariant

固定 `2d` joint chart，比较三条路径：

1. `z_l^P W_Y^C`：D2 full-hidden ceiling；
2. tokenwise `y_l^P T_h`：只用于量化 architecture-given chart loss；
3. `m_h/G_h -> T_h`：真正 aggregate-then-transport 实现。

第 3 条必须与第 2 条的 `B/M` 在 FP32 tolerance 内一致；否则 implementation invalid。第 3 条相对
D2 可用 recovery 的保留率需 `>=0.90`，且最终 edge-equal recovery `>=0.80`、至少 4/5 edges 正向，
才值得进入 discovery。`>=0.90` 为 stretch，不是 canary 必过线。

### D4：合法性与无隐藏 mapping

- constructor API 不接收 Current Exact cache、targets、labels 或 serving candidates；
- monkeypatch `current.compute_kv` 为 forbidden 后仍能完成合法路径；
- Parent cache bitwise 不变；
- 打乱 raw/cache suffix 不改变此前已提交的 chronological segment operator；
- tokenwise transported K/V 只允许出现在 D3 oracle，legal path 中不得 materialize；
- P8/P32 只做事前固定的 region sensitivity，不以 held-out quality 选择 probe count；
- 同一 `T_h` 必须由 weights 解析并跨全部用户复用。

## 8. 结果分叉和停止条件

| 结果 | 解释 | 后续 |
| --- | --- | --- |
| shared-region Exact 失败 | activation-region 交换前提不成立 | 停止本路线 |
| D2 失败 | upper-layer contextual drift 主导 | 停止；不要加 mapper |
| D2 通过、joint chart 失败 | Parent per-head K/V span 不足 | 记录 negative；不搜索 rank/grid |
| D3 达到 0.70–0.80 | 机制可能成立但未达 Design gate | 只做误差分解，不冻结 |
| D3 `>=0.80`、4/5 正向、成本 `<=20%` | 第一版 executable candidate | 冻结 prospective 512 contract |

这条路线最重要的科学风险正是 contextual freeze。它若失败，结论不是“再多加一个 correction
network”，而是当前只靠 Parent K/V 无法在 20% 内恢复 whole-history Current operator；Design 1 必须
转向 migration-aware model training 或承认更高 recomputation budget。

## 9. 可以写进论文的最小表述（仅在通过以后）

Insight 2 候选：

> 推荐请求共享的 activation region 使 distributed history state 暴露为一个 compact response
> operator；对相邻 release，projection transport 可以与 complete-history aggregation 交换，从而使
> migration 不必经过 tokenwise Current KV reconstruction。

Design 1 候选：

> 保留 Parent cache 作为 exact base，以固定 history probes 定义共享 reader region，对完整 Parent
> joint-K/V measure 做 associative reduction，再用 release weights 解析出的 congruence transport
> 生成 Current-version signed response operator；future query 通过原生 interaction 读取该 operator。

在 D1–D3 通过前，这两句只能作为 hypothesis，不能替换当前“functional representation supported,
constructor not admitted”的正式结论。
