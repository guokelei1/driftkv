# Insight 2 / Design 1：论文高度机制审计与 response-control 否证

日期：2026-09-03  
状态：**两条候选均未通过；当前没有可冻结的 Insight 2 / Design 1；单 UID 非正式 route elimination**

## 1. 裁决先行

本轮把 `single-arm Current rank-8` 作为硬对照，而不是把它藏在 related work 讨论里。其同一
UID/five-edge recovery 为：

```text
.8610 / .9173 / .9852 / .9473 / .9753, mean=.9372
```

matrix-free initial factor 加 terminal-KV specialization 后成本为
`853,836,992 FLOPs/user = 17.8953% Exact-All`。它仍是 xKV-adjacent generic compression，不能作为
论文 Design；但任何更复杂的版本机制至少必须给出它没有的稳定收益。

当前严格结论是：

1. **paired finite-release replay 不通过。** Paired r4/r4 在 S4 functional boundary 编译后从 K/V
   splice 的 `.8662` 提高到 `.8997`，成本为 `19.6603%`；但它只在 1/5 edge 胜过 single Current r8
   functional control，均值仍低 `3.72pp`，也低于 strongest r8 K/V control。它是 paired compression
   加已有 moment compiler 的组合，不是新 Design。
2. **common-projection native-response control 不通过。** 新的单臂 r8 机制在同一 per-layer Current
   history span 中同时表示 approximate Current 与 exact Parent，随后在 native query--key activation
   和 value aggregation **以后**做差。它成本上界 `19.4941%`，但五边 recovery
   `.2722/.9337/.8316/.9878/.9362`，均值 `.7923`；比同 runner 的 Current reduced cache `.9364` 和
   shared-layer0 splice `.9372` 都差。把差分从 K/V space 移到 response space 没有产生独立优势。

因此，当前不能把“paired”“early three layers”“rank handoff”“response moment”或“native response
control variate”中的任何一个写成 Insight 2 / Design 1。它们要么是已有数值组件的组合，要么被最强
generic control 支配。

## 2. 审计硬门

本轮只允许一个 Parent→Current edge；不训练、不读 `[512,3000)` confirmation、不读 label。运行只用
frozen discovery 的第一个 UID `1930`、held-out odd-32 candidates、Medium `v0..v5` 五条 edge。

候选必须同时满足：

- 完整 constructor `<20% Exact-All`；
- 至少达到约 `.80` recovery，目标 `.90`；
- 与 strongest single-arm r8 做同 runner、同 cost semantics 比较；
- 新收益来自 Transformer finite-version mechanism，而不是 SVD/range finder、mapper、rank schedule、
  probe tuning 或已有 moment compiler；
- Parent-specific component 删除后应出现稳定退化，否则该组件只是复杂化 generic replay。

旧 functional-boundary contract 锁定的是追加前的 research-plan hash。本轮 nonformal runner 只豁免该
living plan hash；dataset、population、candidate panel、Insight-1 evidence 与六个 checkpoint hash 均
严格重验通过。没有修改 frozen contract、seal 或 raw result。

## 3. Exact finite-release K/V response decomposition

为避免只凭输出 recovery 给 paired mechanism 写故事，本轮先在相同 Current query 上精确拆分一层
prefix response。记

\[
R_{ab}(q)=R(q;K^a,V^b),\qquad a,b\in\{P,C\}.
\]

定义

\[
D_K=R_{CP}-R_{PP},\quad
D_V=R_{PC}-R_{PP},
\]

\[
D_{KV}=R_{CC}-R_{CP}-R_{PC}+R_{PP}.
\]

则对模型原生 attention activation 有精确恒等式

\[
R_{CC}-R_{PP}=D_K+D_V+D_{KV}.
\]

`D_K` 是 routing/address change，`D_V` 是 value-content change，`D_KV` 是有限版本 K--V
interaction；这不是 Taylor 展开。实现的相对 L2 identity error 在各层 edge mean 约
`4e-8--8e-8`。

### 3.1 Coherent causal interventions

每个 intervention 在六层递归推进自己的 Current query trajectory，而不是把单层 tensor norm冒充最终
质量：

| path | v0→v1 | v1→v2 | v2→v3 | v3→v4 | v4→v5 | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K only | .850 | .557 | -.000 | .542 | .641 | .518 |
| V only | .624 | .471 | .880 | .633 | .207 | .563 |
| K + V, no finite interaction | **-1.107** | .847 | .902 | .917 | .888 | .489 |
| finite interaction only | -.817 | -.088 | -.098 | -.108 | .100 | -.202 |
| exact joint endpoint | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

这给出三个克制结论：

1. K 或 V 单侧都不构成稳定迁移对象；
2. 去掉 `D_KV` 在四条 edge 上仍可恢复 `.847--.917`，但在 `v0->v1` 灾难性过冲，因此不能把
   first-order/tangent-like K+V 加法冻结为统一机制；
3. `D_{KV}` 单独也不是 correction；它的作用是在某些 release 中抵消 direct terms。

### 3.2 不是稳定的 K/V cancellation law

沿 Current-Exact query path，六层的 edge-mean
`||D||/(||D_K||+||D_V||+||D_KV||)` 为：

```text
.754 / .695 / .701 / .718 / .638 / .660
```

但 `D_K,D_V` cosine 的 layer mean 多数为正（约 `.43--.85`），不是稳定的 key/value 相消。
`||D_KV||/||D||` 则随层大致从 `.130` 增至 `.348`，说明 finite interaction 会在 depth 中累积，但
其符号与必要性具有明显 edge heterogeneity。

所以可以保留一个**负面科学观察**：跨版本 K/V 不能被两个独立的一阶 correction 稳定替代；完整
finite endpoint interaction 是安全语义的一部分。但该恒等分解没有给出低成本 constructor，也没有
形成比 single-arm 更强的迁移对象，不能单独承担 Insight 2。

## 4. 候选一：paired finite-release functional defect

最强版本使用：

```text
Parent rank4 trajectory + Current rank4 trajectory
  -> paired S4 ELU+1 response moments
  -> exact Parent response + signed functional defect
```

五边 probability recovery 为：

```text
.8725 / .9212 / .9487 / .9335 / .8226, mean=.8997
```

它比 paired K/V splice 的 `.8662` 高 `3.35pp`，说明 aggregation 有时会过滤 trajectory error；但和
matched single Current r8 functional control 比较，逐边差为：

```text
-.0412 / -.0771 / +.0943 / -.0479 / -.1141
```

只胜 1/5 edge。matrix-free input 后 paired functional cost 为 `19.6603%`；它既没有质量优势，也没有
比 shared-layer0 K/V sidecar 更小的状态或更好的 append/eviction closure。

**裁决：RETIRE as Design。** Paired approximation-error cancellation 仍可作为待解释的局部现象，但
当前证据不能证明它生成 generic compression 所没有的 functional information。

## 5. 候选二：同一近似视野内的 native response defect

### 5.1 为什么值得单独测

state-space control variate 会先形成

\[
Z^M=Z^P+\widetilde Z^C-\widetilde Z^P,
\]

再让 nonlinear attention 读取它。对 K 的 query-dependent activation，这一般不等于

\[
R(q;Z^P)+R(q;\widetilde Z^C)-R(q;\widetilde Z^P).
\]

因此本轮固定一个不扫参的 r8 机制：Current reduced replay 每层给出 history span `U_l`；把
approximate Current 和 exact Parent 都投影到同一个 `U_l`，future query 使用模型原生 activation
分别读取两者，最后在 S4 response boundary 做差：

\[
R_l^{M}(q)=R_l(q;Z_l^P)
 +R_l(q;\Pi_l\widetilde Z_l^C)
 -R_l(q;\Pi_l Z_l^P).
\]

这条路径保留完整 K--V interaction，没有 candidate regression、Current Exact upper state、mapper 或
ELU affine approximation。它检验的是一个明确命题：**同一个 approximation view 是否让两个 release
的 native response error 比 tensor error 更相关。**

### 5.2 结果与成本

| method | v0→v1 | v1→v2 | v2→v3 | v3→v4 | v4→v5 | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current reduced cache | .913 | .999 | .852 | .981 | .937 | .936 |
| common-projection state splice | .392 | .987 | .834 | .999 | .889 | .820 |
| shared-layer0 state splice | .861 | .917 | .985 | .947 | .975 | **.937** |
| common-projection response defect | **.272** | .934 | .832 | .988 | .936 | **.792** |

保守 cost 从完整 single-r8 matrix-free/terminal-KV ledger 开始，不扣除本方法不需要的 `U0` work，再
额外计算六层 QR 与 Current/Parent 两臂 `U_l^T K/V`：

```text
930,118,850 FLOPs/user = 19.4941% Exact-All.
```

sidecar 为 `86,016` scalars/user。即使给它保守成本通过，质量仍被同一 trajectory 的 generic reduced
cache 和 shared-layer0 splice 大幅支配；`v0->v1` 还出现严重 response-level over-correction。
相对普通 Reuse reader，每个 query 还要在六层各做两次 rank-8 native factor read 与 signed add，
约 `2,435,328 FLOPs/query`；这高于 shared-layer0 K/V control 的一次 mode read，不能用 constructor
成本掩盖 serving overhead。

**裁决：RETIRE。** “先做 native response、后做 common-projection subtraction”没有得到
Transformer-specific cancellation advantage。继续换 basis、rank、probe 或 edge certificate 会退化为
query-aware compression/controller tuning，不满足论文高度要求。

## 6. 为什么 early-depth handoff 也不能晋级

`d=1 -> d=3` 的 paired recovery 从 `.528` 提升到 `.831`，说明 matched effect 需要穿过早期
attention--gate--residual blocks；但 fixed `Current4 -> Current8` upper handoff 只有 `.836`，比
single r8 低约 `10.1pp`。这最多是 layer profile，不是独立迁移原理；把 rank budget 在某层重新分配
属于数值 schedule。由于 Insight 1 已经排除把 layer locality 当主要迁移抽象，它不能靠“前三层”这一
观察重新包装成 Design。

## 7. 当前真正剩下的研究边界

已有合法 `<20%` 路线提供的 Current information，本质上都来自 reduced Current trajectory；Parent 的
作用无论放在 tensor subtraction、paired replay、S4 moments 还是 native response control，目前都没有
稳定超过这条 generic source。与此同时，更有机制含义的 Parent-anchored finite-difference scan 在
KV-only source interface 下，仅 mandatory source decode + historical Q/gate 下界就达 `25.3173%`；稳定
joint decode 下界为 `34.8113%`，尚未开始 Current defect computation。

这形成一个比“继续调 rank”更重要的分叉：

- **保持当前 KV-only source 与 `<20%` cap：** 尚无证据支持一个优于 generic compression 的新 Design；
- **追求真正的 finite-release delta execution：** 需要 release-ready Parent source state 或模型—系统
  协同接口，已超出当前 frozen Medium cache contract；
- **只追求数值达标：** single-arm r8 已经约 `.937`，但 related-work 边界决定它只能作为 baseline。

因此下一项最高信息量工作不应是 r6/r7、不同 probe、不同 handoff layer 或另一个 response mapper。
应先在纸面和 CPU cost 上冻结一个**新的 Current-information source**：它必须不是普通 reduced replay，
并能在现有 Parent persistent interface 下证明 `<20%`；若做不到，应明确把 Design 1 转向
migration-ready state co-design，而不是继续从当前数字拼装论文方法。

## 8. 实现与验证

- `scripts/insight_two/kv_response_coupling.py`：exact `D_K/D_V/D_KV` 分解与 coherent interventions；
- `scripts/insight_two/run_kv_response_coupling_preflight.py`：UID1930/five-edge fixed runner；
- `scripts/insight_two/common_projection_response.py`：same-span native-response defect 与成本；
- `scripts/insight_two/run_common_projection_response_preflight.py`：唯一 r8 route-elimination runner；
- `tests/test_insight_two_kv_response_coupling.py`；
- `tests/test_insight_two_common_projection_response.py`。

Focused verification：`6 passed`；相关文件 `ruff check` 通过。上述 runner 输出未写入 formal results、
未生成 contract/seal，也未读取 confirmation。
