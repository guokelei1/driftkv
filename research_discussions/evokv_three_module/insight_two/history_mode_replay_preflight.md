# Insight 2 preflight：由 layer-0 版本差异锚定的 history-mode replay

日期：2026-09-03  
状态：**结构线索保留；single-arm 已降级为 baseline，双臂尚未通过机制准入；尚未形成 Insight 2 或 Design 1 结论**

> 2026-09-03 裁决：本文最初描述的 single-arm `Current reduced replay + shared U0 + Parent splice`
> 与 xKV 的跨层共享 token basis、已有 base-plus-residual cache reader 高度相邻，不能承担论文新意。
> 后续发现的 equal-resolution Parent/Current finite-release recurrence 有更强的机制信号。原完整成本
> `25.2952%`、KV-only 路径 `21.8226%`；随后 matrix-free input factor 将同语义 KV-only executor
> 降至 `18.3264%`。但 paired 单 UID recovery 仍低于 generic single-arm rank-8，因此本文后续章节
> 只作为形成过程和 baseline 定义保留；当前裁决以
> [related-work boundary](history_quotient_related_work_boundary_preflight.md) 与
> [matrix-free cost audit](matrix_free_paired_input_preflight.md) 为准。

## 1. 当前最值得验证的科学命题

Insight 1 已经说明，跨版本误差不能稳定地归因给少量 token、recent window 或单层；即使某些位置的
Exact splice 有效，其 dependency closure 也会迅速扩大。后续若继续换 token selector、cluster 或
sampler，最多得到另一个 locality heuristic，不会形成新的设计原则。

新的待证命题不再寻找少量位置，而是区分 **support** 与 **mode**：

> 跨版本状态差异可能覆盖几乎全部历史位置，却只沿少数个用户级 history modes 共同变化；这些
> dense modes 在 dependency-free 的 layer-0 projection 首次可见，并在后续 Transformer 层中保持
> 对功能差异足够的跨层坐标。换言之，version drift 可以是 support-dense、mode-compact。

这里的 mode 位于 token/history 轴：`U in R^(N x r)` 的每一列通常在全部 1,024 个历史位置上都有
非零系数。它不是选择 `r` 个 token，也不是把 hidden width 从 192 压到 8。若该命题成立，它与
Insight 1 构成直接逻辑关系：失败的是 coordinate locality，而不是全历史差异的低维组织结构。

一个更强、也更 Transformer-specific 的子命题是：

> 对 rank-`r` token state，RMSNorm 与 Q/K/V linear projection 不增加 token rank；任意 native
> attention kernel 的 value aggregation 也不增加单 head 的 value-side token rank。真正产生新
> token modes 的位置是 multi-head merge、pointwise gate/FFN 与 residual addition。因此，Current
> Transformer 可以只在这些 rank-expansion boundaries 重压缩，而不是重算并写回完整 Current KV。

这个子命题不依赖 legacy `ELU+1` 的 positive cone；softmax、SiLU attention 与 relative-position
bias 都可以改变 attention weights，但 `A V = (A U) C_V` 仍保持 value-side factor。当前实现先在
Medium legacy block 上验证执行语义，之后才讨论其他 backbone adapter。

## 2. 历史 baseline：生成 Current modes，替换 Parent 的同一坐标

该 baseline 不是 `Parent KV -> Current KV` 的 learned mapping，也不是保存一个整体低秩 Current cache。对每个
用户和单条 Parent→Current edge，它执行四步。

### 2.1 用完整历史做一次 reduced Current replay

从 raw item/action/time history 形成 Current input state。每层输入写成：

~~~text
X_l_hat = A_l C_l,       A_l in R^(N x r), C_l in R^(r x H).
~~~

`r=8` 的 primary 由 Medium 的 20% cost envelope 约束，不从 candidate quality 拟合。每层使用固定
Gaussian range finder、固定 oversampling 和固定 power-iteration count；所有用户、edge 和 layer
共享同一事前规则。

在该 factorization 上：

~~~text
RMSNorm(A C) = (D A) (C diag(gamma)),
Q = A C_Q, K = A C_K, V = A C_V,
A_attn V_h = (A_attn A) C_V,h.
~~~

attention logits、causal mask 和原模型 activation 都保留；不把原 attention 换成 linear attention。
多头 attention output 与 gate/residual 在 full token coordinates 中形成后，再进入下一次固定
compression。数值实现只把重压缩放在真正的 rank-expansion boundary。

### 2.2 只从合法 layer-0 差异形成 release basis

reduced replay 已产生 approximate Current layer-0 K/V，Parent layer-0 K/V 已经在 persistent cache 中：

~~~text
D_0_hat = [K_0_hat^C - K_0^P, V_0_hat^C - V_0^P].
~~~

对 `D_0_hat` 使用一次固定 one-pass range finder 得到 `U_0 in R^(N x 8)`。这一 construction 不读取
Current-Exact upper-layer K/V、candidate、label 或 future event。layer 0 的意义不是“第一层更重要”，
而是它是 raw event 到版本差异的最早 dependency-free formation boundary。

### 2.3 在所有层只编译同一组 signed mode coefficients

若 reduced replay 第 `l` 层的 factorized K/V 为 `A_l C_K,l` 与 `A_l C_V,l`，只写：

~~~text
Delta C_K,l = U_0^T A_l C_K,l - U_0^T K_l^P,
Delta C_V,l = U_0^T A_l C_V,l - U_0^T V_l^P.
~~~

逻辑 migrated state 是：

~~~text
K_l^M = K_l^P + U_0 Delta C_K,l,
V_l^M = V_l^P + U_0 Delta C_V,l.
~~~

所以 Design 做的是 **subspace replacement**：`span(U_0)` 中的 Parent coefficient 被 Current replay
替换，正交补中的 Parent information 原样保留。它不需要物化任何 upper-layer Current K/V。一个
`U_0` 被六层共享，Medium rank-8 sidecar 为：

~~~text
N r + 2 L r H = 26,624 FP scalars/user,
~~~

即 full Current KV 的 `1.1286%`。这比每层各存一套 rank-8 basis 的 `67,584` scalars / `2.8646%`
更紧凑；更重要的是，共享 basis 是待验证的 cross-layer release structure，而不是 storage trick。

### 2.4 Future query 原生读取 Parent + signed modes

reader 不需要展开 migrated cache。对一层一 head：

~~~text
q (K_P + U_0 Delta C_K)^T
  = q K_P^T + (q Delta C_K^T) U_0^T,

A(q) (V_P + U_0 Delta C_V)
  = A(q) V_P + (A(q) U_0) Delta C_V.
~~~

activation 作用在完成 key correction 后的 logits 上，因此 future candidate/query 仍由 Current
Transformer 自己决定系数；这不是 candidate-shared broadcast offset。append 的 Current event 使用
native Current K/V，令其 `U_0` row 为零；eviction 同步删除旧位置的 `U_0` row。该 state 有清楚的
lineage，而不是每次请求重新估计 correction。

## 3. 为什么这不是“低秩/SVD 就是创新”

以下内容明确不构成 novelty：

- activation/KV 有低 effective rank；
- 用 SVD、PCA、randomized range finder 或 low-rank factor 存 cache；
- 用 sequence projection 加速 attention；
- 用一个矩阵把 source KV 映射为 target KV；
- LoRA 式 low-rank parameter delta；
- 单独写出 `A V` 的结合律。

若最终方法只能描述为“rank-8 prefill”或“low-rank KV cache”，它不能成为 Design 1。当前可能有论文
意义的最小机制组合必须同时保留：

1. **release-specific formation**：basis 来自同一用户 layer-0 的 Parent→Current state defect；
2. **cross-layer persistence**：同一个 dense history basis 承载全部层的 signed version coefficients；
3. **Parent control state**：只替换 version-carrying modes，保留 Parent orthogonal complement；
4. **native Current dynamics**：coefficient 由 bounded Current Transformer replay 生成，不由 matched
   target KV、score 或 label 拟合；
5. **factorized reader and lineage**：future query 直接读取 Parent-plus-modes，并支持 append/eviction。

related-work audit 仍在进行。在 audit 完成前，不使用“首个”“从未有过”等表述。

## 4. 当前仅有的非正式数值信号

以下均来自 frozen population 的第一个 UID `1930`，只用于决定是否值得立 prospective canary；它们
不是 population evidence，也不能用于论文数字或 confirmation 调参。

### 4.1 结构 ceiling

- 每层独立 exact joint `Delta[K,V]` token-SVD rank 8，在五条 edge 上的 score recovery 约为
  `0.9989 / 0.9985 / 0.9928 / 0.9995 / 0.9999`；
- 只用 exact dependency-free layer-0 defect 的 rank-8 token basis 投影所有 upper-layer exact defect，
  recovery 约为 `0.9981 / 0.9732 / 0.9396 / 0.9302 / 0.8574`；
- Parent cache 自身的 token basis不能稳定承载 exact defect，因此正信号不是“任意低秩 basis 都行”。

这些都是 Current-Exact oracle，只支持提出 cross-layer basis hypothesis。

### 4.2 合法 semantic prototype

primary preflight 使用：Current replay rank 8、oversampling 4、每层一次固定 power iteration；layer-0
defect basis 使用同 rank/oversampling 的 one-pass range finder。它只读 raw history、Current weights 与
Parent cache。五条 edge 的 held-out odd-32 probability-gap recovery 为：

| edge | legal shared layer-0 mode splice |
| --- | ---: |
| `v0->v1` | 0.8610 |
| `v1->v2` | 0.9173 |
| `v2->v3` | 0.9852 |
| `v3->v4` | 0.9473 |
| `v4->v5` | 0.9753 |

同一预检中，直接使用 factorized Current cache 的 generic control 平均水平相近，但需要每层自己的
basis；而用 replay 自己的 per-layer basis 做 Parent splice 在部分 edge 明显更差。这个现象提示
layer-0 release basis 可能承担了真实 cross-version 语义，但单 UID 不足以裁决。

## 5. 在 formal canary 前必须完成的否证

### 5.1 稀疏执行等价

当前 CPU tests 已验证：

- factorized RMSNorm 与 dense RMSNorm 数值等价；
- factorized native attention 与相同 rank state 的 dense attention 数值等价；
- factorized replay 与 dense semantic replay 在 exact-SVD/fixed-range 两种 compressor 下等价；
- `Parent + U DeltaCore` reader 与先 materialize cache 再走 native reader 等价；
- mode splice 精确替换 basis coefficient，并保留 Parent orthogonal complement。

formal runner 还必须在 Medium checkpoint 上记录 max absolute cache/logit error；unit test 不能代替
真实形状 instrumentation。

### 5.2 严格成本

成本审计必须包含：full Current input projection、六次 fixed range finder、factorized Q/K/V/gate/output、
causal logits 与 weighted-value work、multi-head rank expansion、dense gate/Hadamard/residual、layer-0
defect range finder、六层 `U_0^T Parent K/V`、sidecar write，以及 reader 的额外 mode read。任何 full
`N x H` 或 `N x N` temporary 都不能因“最终 rank 很小”被隐藏。

只有完整 constructor `<=20% Exact-All` 才允许立 formal contract。当前 dense PyTorch prototype 的
wall time 不代表该 theoretical executor 已实现为高性能 kernel。

### 5.3 32-user / five-edge prospective gates

若成本通过，formal canary 事前冻结以下三条路径；不在结果出来以后更换 rank、seed、oversampling 或
iteration count：

1. `generic reduced Current cache`：相同 Current replay，不读 Parent coefficient；
2. `per-layer Current-mode splice`：每层自己的 replay basis，用于排除“任意 splice 都有效”；
3. `shared layer-0 defect-mode splice`：本文 primary。

同时保留两个 Current-Exact representation oracle：per-layer joint-delta rank 8 与 fixed layer-0-basis
rank 8。它们不进入 action frontier。

primary 的最低 launch gate：

- edge-equal user-equal recovery `>=0.80`；
- 至少 4/5 edge 为正，目标 4/5 edge `>=0.80`；
- 不低于 generic reduced-Current control 超过 0.05，或在相近 recovery 下给出明确的 shared-state/
  persistence 优势；
- sidecar、Parent immutability、labels absent、Current-Exact isolation 和 factorized-reader reconstruction
  全部通过；
- bootstrap uncertainty 与所有 edge 完整报告。

canary 不过就停止，不把 rank 改成 12、不增加 learned mapper，也不启动 512。canary 通过以后，才在
512 discovery users 上复核结构与 recovery，再做真实 append/eviction persistence；confirmation
`[512,3000)` 继续保持 unread。

## 6. 当时的条件性表述，以及当前失败出口

以下表述只记录最初的条件性假设，**当前不能作为论文 Insight 2**。即使 population evidence 同时支持
cross-layer exact oracle 和 legal shared-mode splice，若 constructor 仍是 single-arm low-rank replay，
它也没有越过 xKV-adjacent prior-art boundary。原条件句是：
Insight 2 写成一句不依赖新名词的话：

> Transformer recommender 的跨版本状态误差虽然分布在完整历史上，却沿少量在早期 projection
> 形成、并跨层保持的 dense history modes 演化；attention aggregation 保留这些 value-side modes，
> 而 mode expansion 主要发生在 gated residual boundary。

相应 Design 1 是：

> 用 bounded Current replay 只重物化这些 history modes，并把它们作为 signed per-layer coefficients
> 注入 Parent persistent state，而不重建完整 Current KV。

若 exact layer-0 basis ceiling 不稳定，说明“早期形成、跨层保持”不成立；可以保留 per-layer low-rank
观察，但它不足以成为该 Design。若 oracle 成立而 legal replay 失败，说明 mode structure 可表示却
不可在 20% 内生成。若 generic reduced Current cache 与 shared splice 完全等价且没有 persistence/
storage 优势，则研究贡献退化为已有低秩 prefill，必须停止而不能换名包装。
