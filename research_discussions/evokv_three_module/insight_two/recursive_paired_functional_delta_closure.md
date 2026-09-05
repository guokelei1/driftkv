# Design 1 机制审阅：recursive paired functional-delta causal closure

日期：2026-09-02  
状态：**算法与创新性审阅；toy implementation 已通过，population canary 尚未执行，不能写成冻结 Design 1**

## 1. 审阅结论

当前 activation-region 结果足以支持一个强 observation，但 `B/M moments`、version pairing、
control variate、landmark selection 中任何一项都不能单独成为论文设计。它们分别与 linear
attention/fast weights、control-variate attention、KV compression/coreset 有直接重合。

仍然可能达到 PRO-level novelty 的核心不是换一种采样或映射，而是下面这条新的计算闭包：

> 不恢复完整 Current token state；保留 Parent prefix 作为精确 base path，让少量真实历史
> carrier 在每一层读取此前已经形成的 query-conditioned version defect，并由此产生下一层的
> Current-side defect。迁移状态因此既是 serving correction，也是构造后续迁移状态的因果计算状态。

更紧凑的 Insight 2 候选是：

> Transformer 的跨版本误差不仅在 reader aggregation 后变得紧凑；它还可能具有
> **causal compositionality**：一个 prefix 的 reader-response defect 可以替代缺失的 Current prefix，
> 驱动较晚历史事件产生上层 defect，而不必先物化该 prefix 的逐 token Current K/V。

如果递归项在同 support、同权重下不能显著优于 Parent-conditioned carriers，这个命题就不成立；
剩下的方法只能被诚实地归类为 paired quadrature/control variate，而不应包装成新 Design。

## 2. 功能状态的严格定义

设 Parent/Current 为 `P/C`，历史为 `z_1...z_N`，第 `l` 层 Current reader 对一个 K/V 集合的
prefix response 写成：

~~~text
A_l^C(q; S) = sum_{(k,v,p) in S} rho_l^C(q, k, p_query, p) v.
~~~

`rho` 必须是 Current reader 的原生 interaction；Parent K/V 进入 base path 时也由 Current reader
读取。对 softmax，`A` 改写为 numerator/partition 二元状态，不能把 normalized context 直接相加。

对 source interval `I_g`，理想的 functional delta 是一个函数而非向量：

~~~text
D_l,g(q)
  = A_l^C(q; S_l,I_g^C) - A_l^C(q; S_l,I_g^P).
~~~

完整 Current prefix response 因而可以写成：

~~~text
A_l^C(q; S_l,<i^P) + sum_{g: I_g < i} D_l,g(q).
~~~

这只是恒等式，不是创新。设计问题是能否在不读取 Exact Current upper-layer K/V 的情况下，
递归构造一个稀疏近似 `D_hat`。

## 3. 可执行递归

### 3.1 Causal support contract

先将历史冻结为顺序不交叠的 intervals：

~~~text
I_1 < I_2 < ... < I_R,     union_g I_g = [1,N].
~~~

每个 interval 只选一个或固定少量真实 carrier `c_g in I_g`，并有事前定义的 quadrature weight
`w_g`。选择可以使用 raw history 和 dependency-free Current layer-0 features，但不能使用候选、label、
Exact Current upper-layer state 或 qualification result。

`D_hat_l,g` 只有在 `I_g` 完整结束后才 commit，并且只能被 `I_h, h>g` 的 carriers 读取。全局
Voronoi cluster mass 若跨越多个 future interval，不能直接放进递归 primary；否则一个 future token
的 mass 会影响更早 carrier。合法 address rule 必须嵌套在 chronological interval 内，或者显式维护
prefix-valid partial mass。

### 3.2 Dependency-free seed

Current layer-0 K/V 只依赖 raw event embedding、normalization 和 projection，可以对全部 `N` 个
位置精确计算而不展开 attention closure。第一层因而提供一个 dependency-free seed：

1. transient 地生成 full-history Current layer-0 K/V；
2. selector 与每个真实 carrier 的 layer-0 K/V 都直接读取这个精确 seed；
3. primary carrier reader 仍使用 `Parent base + earlier paired delta`，与最终 serving reader 保持同一个
   稀疏 functional state；完整 layer-0 Current prefix read 是“局部更精确但表示不一致”的消融；
4. 生成 layer-0 paired functional state 后，丢弃 full layer-0 transient tensors。

dependency-free seed 保证被选 carrier 的 layer-0 address 是真实 Current state；但若 constructor 用
dense Current prefix、serving 却用 sparse paired prefix，upper carrier 会在另一套 reader state 下生成。
正式实验必须把这种 locally exact path 与 self-consistent closure 分开报告。

### 3.3 Native paired-ledger recurrence

按 `g=1...R` 处理 carriers。对 carrier `c_g`，令：

~~~text
x_hat_0,c_g = Embed_C(z_c_g).
~~~

在第 `l` 层，由 `x_hat_l,c_g` 经 Current projection 得到
`q_hat_l,g, k_hat_l,g, v_hat_l,g`。primary path 在每一层都只允许：

~~~text
r_hat_l,g
  = A_l^C(q_hat_l,g; ParentCache_l,<c_g)
    + sum_{h<g} w_h [
        rho_l^C(q_hat_l,g, k_hat_l,h) v_hat_l,h
        - rho_l^C(q_hat_l,g, k_l,c_h^P) v_l,c_h^P
      ]
    + Current self response.

x_hat_l+1,c_g
  = CurrentBlockFinish_l(x_hat_l,c_g, r_hat_l,g).
~~~

处理完 carrier 后，才把它在每层的 paired atom 写入 ledger：

~~~text
D_hat_l,g(q)
  = w_g [
      rho_l^C(q, k_hat_l,g) v_hat_l,g
      - rho_l^C(q, k_l,c_g^P) v_l,c_g^P
    ].
~~~

这里没有 `P KV -> C KV` mapper。`k_hat/v_hat` 来自 Current block 的真实 forward，只是它所读的
prefix 被 `Parent base + earlier functional delta` 近似。上层 delta 因而由下层 delta 通过原模型
递归生成，而不是由一个外部函数预测。

### 3.4 Serving read

迁移完成后，真实 query 在每层执行：

~~~text
r_hat_l(q)
  = A_l^C(q; full active Parent cache)
    + sum_{active g} D_hat_l,g(q),
~~~

然后继续 Current gate、residual 和后续层。query 的系数由原生 `rho(q,k)` 当场产生；同一 sidecar
可以被 ranking candidates、next-item query 和随时间变化的请求读取。

新 append 由 corrected Current reader 产生 native Current K/V，不写入 `P->C` delta。Parent source
被 eviction 时，对应 interval delta 必须同步删除或耗减。若用 aggregate moments，则必须保留足以
删除的 block ledger；只存一个不可删除的全局 tensor 不构成 persistent migration object。

## 4. Exact-support invariant

这个递归比“若干技巧的组合”更强的地方，是它有一个可直接测试的 full-support invariant：

> 当 `R=N`、每个 interval 是 singleton、`w_g=1`，并使用原生 paired atoms 时，递归输出应与
> Exact Current K/V 和最终 score 数值一致。

证明是一个按时间和层的双重归纳。假设所有 `j<i` 的 Current atoms 已精确生成，则 Parent base 加
所有 paired replacements 正好等于 Current prefix response；Current block 因而生成位置 `i` 的精确
hidden/K/V。layer 0 的 dependency-free exact seed 给出归纳起点。

这个 invariant 不能证明稀疏版本有效，但能证明算法确实是 Current computation graph 的稀疏化，
而不是经验 mapping。若 `R=N` 不能恢复 Exact，先判 implementation invalid，不解释质量结果。

## 5. 两种 ledger realization

### 5.1 Native signed atoms：机制主版本

每个 carrier 保留 Current `k_hat/v_hat`、Parent source index、weight、position 和 interval lineage；
query 通过原生 kernel 同时读取正负 arm。它支持 legacy ELU+1、faithful SiLU HSTU 和带 position
bias 的 pointwise reader；softmax 版本分别累计 numerator 与 partition delta。

这是检验 causal closure 的 reference realization。`R=64` 时 Current-side carrier K/V 约为完整
1024-token K/V 的 `6.25%`，Parent arm 只需 index，不必重复存储 Parent tensor。

### 5.2 Activation-region moments：Medium 的可选 compiler

对当前 legacy `ELU+1`/no-bias checkpoints，在固定 positive region 内可将 atom ledger 编译为：

~~~text
Delta B_l = sum_g w_g (chi_C v_hat_C - chi_P v_P)
Delta M_l = sum_g w_g (chi_C k_hat_C outer v_hat_C
                       - chi_P k_P outer v_P).
~~~

read 为 `Delta B + s q Delta M`。这个 compiler 可把 sidecar 降到 `38,016` scalars/user，但它不是
设计创新，也没有 native-ledger 的 full-kernel exact invariant；negative branch 和 region exits 必须
单独报告。若需要 rolling deletion，可以保存少量 chronological segment moments，例如 8 段约为
完整 KV scalars 的 `12.9%`，并在 segment 首次 eviction 时保守停用整段。

论文主消融必须先验证 native recurrence，再检查 moment compiler 保留了多少收益。反过来只验证
`B/M`，会与 linear attention、fast weights 和 Tensor Cache 的边界重叠。

## 6. 为什么它目前只是“有创新潜力”

初步 related-work audit 的边界如下；这不是正式查新结论：

| 邻近方向 | 已有核心 | 本候选必须新增的不可替代部分 |
| --- | --- | --- |
| [Cross-Model KV Transfer](https://arxiv.org/abs/2608.03893)、[C2C](https://arxiv.org/abs/2510.03215) | ridge/MLP 或 neural projection 将 source KV 翻译、融合到 target KV | 不拟合 target KV；paired response defect 被递归读写并生成 upper-layer defect |
| [DroidSpeak](https://arxiv.org/abs/2411.02820) | 选择关键层重算 | 不依赖 layer locality；所有层通过小 functional ledger 闭包 |
| [ResKV](https://arxiv.org/abs/2607.29591) | 同版本 eviction 后重建遗漏的 attention numerator/denominator | 两个 release 的 paired replacement，并作为 migration constructor 的中间计算状态 |
| [Tensor Cache](https://arxiv.org/abs/2605.22884)、[Linear Transformer](https://proceedings.mlr.press/v119/katharopoulos20a.html) | outer-product/feature-map recurrent memory | moments 只作为 compiler；创新不能落在 `qM` 恒等式上 |
| [LESS](https://arxiv.org/abs/2402.09398) | 用 learned low-rank recurrent cache 保存同模型被 eviction 的 KV 贡献 | 本候选不学习 attention kernel；递归发生在 release defect 对 upper-layer carrier 的构造依赖中，而非仅对新 KV 做 memory update |
| [Efficient Attention via Control Variates](https://arxiv.org/abs/2302.04542) | 用 control variates 改善 attention approximation | Parent base + delta 不是贡献；贡献必须来自跨层、跨时间的 causal self-propagation |
| [CollectiveKV](https://arxiv.org/abs/2601.19178) | global shared pool + user-specific KV | 本候选不做 cross-user retrieval 或共享映射，state 绑定一次 Parent-to-Current edge 与 cache lineage |

因此可以声称的最小新意不是“把这些组件放在一起”，而是一个新的 state-transition semantics：

~~~text
functional delta is simultaneously
  (1) the migrated serving state, and
  (2) the causal prefix state that constructs its own higher-layer continuation.
~~~

只有实验显示第 `(2)` 项不可删除，论文创新链才成立。

## 7. 必须固定的 ablations

所有下列对照必须使用相同 UID、edge、carrier intervals、representatives、weights、probe/candidate split
和理论预算；不能让 selector 变化替 causal closure 取得收益。

1. **Reuse**：无 sidecar。
2. **Parent-conditioned carriers**：当前代码路径；每个 carrier 只读 Parent prefix，彼此独立。
3. **Paired, no recursive read**：构造 paired ledger，但 ledger 只给 serving query，不给后续 carrier。
4. **Recursive paired closure**：本文 recurrence；唯一主方法。
5. **Recursive Current-only**：去掉 matched Parent negative arm。若它同样好，收益可能只是额外
   memory token，而不是 version defect。
6. **Future-leaking oracle**：所有 carriers 读取 final/global ledger。它只能作为非法 ceiling；若合法
   recurrence 明显失败而该项成功，说明所需依赖不能因果闭合。
7. **Layer truncation**：只在前 `k=1/2/4` 层递归。用于确认收益不是已有浅层 selective recompute。
8. **Native atoms vs moment compiler**：区分 causal mechanism 与 legacy algebraic compression。
9. **Exact-state paired oracle vs recursive carriers**：固定 support，量化 constructor gap。
10. **Fixed chronological support vs within-interval address support**：selector 只是 constructor component；
    不允许把挑点变化写成主贡献。
11. **Fixed S4 offset / PRO-compatible rank-0**：证明 query-native operator 确实必要。
12. **Singleton `R=N`**：验证 exact-support invariant，不参与成本 frontier。

最关键的论文消融是 `2/3 -> 4`。建议预注册：在主预算点上，recursive closure 相对同 support 的
no-closure 至少提高 `0.10` edge-equal recovery，且至少 4/5 edges 方向一致；同时 user-level paired
bootstrap interval 排除零。若只提高几个百分点，可以作为优化，但不足以承担 Design 1 的核心新意。

## 8. 失败判据与 claim ladder

### 8.1 Implementation validity

任一项失败都不得解释 quality：

- `P=C` 时所有 paired ledger 为零，输出等于 Reuse/Exact；
- `R=N` native recurrence 恢复 Exact Current K/V、stage response 和 score（预注册 tolerance）；
- 改写 carrier 之后的 raw/cache suffix 不改变该 carrier 及此前 ledger；
- constructor API 不接收 Exact Current upper-layer cache，测试中 monkeypatch `compute_kv` 为 forbidden；
- Parent cache bitwise 不变，interval weights 守恒，无 candidate/label 输入；
- inclusive/exclusive self 和 source position bias 与原 reader 完全一致。

### 8.2 Mechanism admission

推荐先冻结 `R=32/64/128`，但 primary 只取完整核算后 `<=20%` 的最大点。至少需要：

- 主点 edge-equal recovery `>=0.80`，至少 4/5 edges 正向；`>=0.90` 为 stretch；
- 相对 Parent-conditioned/no-recursion 的 closure gain 达到上一节门槛；
- 相对同-support Exact-state paired oracle，recursive constructor 保留大部分可用增益；建议报告
  `(recursive - Reuse)/(oracle - Reuse)`，不能只报最终 recovery；
- 完整成本包含 full Current layer-0 `K` **和** `V` projection、selection、Parent reads、recursive
  signed-ledger reads、moment construction、certificate、writes 与 metadata。

对 unnormalized native kernel，一次 query 读取一个正负 paired atom 约为 `8H` attention FLOPs；
因此 `R` 个按序 carriers 在上面 `L-1` 层新增的递归读开销约为
`4(L-1)H R(R-1)`，另加 base Parent-prefix reads、Current projections 和 layer-0 exact seed。
该项通常不是主成本，但必须显式进入合同，不能因其是“sidecar read”而记为零。

若只有 `R=128` 质量好但 selector + recursion 总成本超过 `20%`，它仍是机制证据，不是 Design action。

### 8.3 Novelty failure

出现以下任一情况，应明确停止“新 Design”声称：

- recursive closure 对同 support 的 independent carriers 无稳定增益；
- 去掉 Parent negative arm 不降，说明它是 extra-memory method 而非 migration defect；
- 只有 future-leaking/global assignment 成功；
- 必须训练 ridge/MLP 或拟合 Exact target K/V/score 才成功；
- selector 的选择决定全部收益，而 random/fixed support 上 closure 无效；
- moment compiler 成功但 native recurrence/causal constructor 失败；
- cutover 成功但 append/eviction 后无法按 lineage 删除，或整体 recovery 低于预注册门；
- faithful SiLU HSTU 小控制中 causal-closure gain 消失，却仍把方法写成一般 Transformer Design。

### 8.4 Paper claim ladder

1. 目前可以写：legacy Medium reader 的 full functional difference 在 activation aggregation 后高度
   紧凑；absolute sampled construction 失败。
2. exact-support invariant + recursive canary 通过后可以写：functional defect 是一个可执行的 causal
   state-transition abstraction。
3. `<=20%`、population discovery、rolling lineage 通过后，才可以冻结 Design 1。
4. faithful SiLU HSTU/softmax control 只需验证“recursive defect closure 优于 independent carriers”；
   不要求复现 legacy `B/M` compiler。没有这一级时，论文应限定为当前 pointwise recommender reader。

## 9. 对现有代码的具体审阅

`scripts/insight_two/paired_region_delta.py` 保留 Exact-cache paired oracle、Parent-conditioned lower
bound 与完整成本账；`scripts/insight_two/causal_delta_closure.py` 已实现 recursive primary。后者按
source position 处理 carriers，让每个 carrier 读取 prefix-valid earlier paired atoms，并输出 native
signed memory。`replay_parent_conditioned_current_carriers()` 仍只是旧 lower bound，不能称 closure。

当前实现把 global assignment 改写为 prefix-valid partial mass，并禁止尚未到达的 future center 参与
earlier carrier；full mass 只在 cutover serving read 使用。正式裁决仍要报告 represented-prefix coverage，
因为合法不等于近似一定准确。还需注意：

- `project_full_current_layer0()` 当前通过 `_project()` 额外计算并丢弃 `Q`；成本已按实际 Q/K/V 计算，
  同时单列未来 KV-only 优化下界；
- address selection 与 recursive positive/negative ledger reads 已进入 total FLOPs；R64 的平均/最坏
  total fraction 约为 `16.48%/19.45%`，R128 约 `27.08%`，所以只有 R64 可作主预算点；
- `trace_history_item_region_queries()` 的 upper-layer probes 只读 Parent path，moment masks 不是
  recursive sidecar 的 fixed point；它可以是 frozen bootstrap，但必须和一次 sidecar-refreshed probe
  ablation 分开；
- full-history cluster masses 只用于最终 cutover read；constructor 只读取 prefix-valid partial mass；
- certificate 只能判断两个 estimator 是否一致，不能证明它们接近 Current Exact。必须报告
  coverage、false-admit 和 fallback 后 overall recovery。

因此，no-recursion helper 只作 matched baseline；新实现必须靠递归项本身通过 novelty gate。

## 10. 下一份实现合同只应回答一个问题

下一轮不要同时调 selector、moments、probe 数或 fallback。固定成本可承受的 `R=64` 和
candidate-free `P=8`，只正式裁决：

~~~text
exact Current layer-0 carrier seed
  + prefix-valid chronological paired ledger
  + later carriers read earlier ledger at every layer
  + the same ledger serves held-out real queries.
~~~

toy model 的 singleton exactness、suffix causality、no-Exact-access 与 Parent-cache immutability 已经
通过。随后在 frozen 32-user/five-edge canary 以 Parent-conditioned、recursive closure、dense-layer0
ablation 三臂比较；full moments、Exact-state carriers 与 affine compiler 只作解释。只有递归项本身
贡献显著，再讨论 512-user discovery 和 rolling transport。
