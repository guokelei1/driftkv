# Defect-first coordinates 与 native-response finite-release correction 预检

日期：2026-09-03  
状态：**单 UID / 五 edge 非正式 route elimination；defect-first 与 paired native-response 均 NO-GO / RETIRE；不是 Insight 2 或 Design 1 冻结稿；未建立 formal contract，未授权读取 confirmation**

## 1. 这轮只问一个机制问题

普通 reduced replay 每层压缩的是绝对 Current state。若 Parent 的 dominant history modes 占据大部分
rank，它可能把真正决定跨版本变化的方向挤掉。本轮执行前固定一个不扫 rank 的反事实：把每层工作状态
拆成

\[
\widetilde B_l\quad\text{(Parent base, rank 2)},
\qquad
\widetilde D_l\quad\text{(Parent-to-Current defect, rank 4)},
\]

并令 Current block 的输入是两个 factors 的精确和，effective rank 为 6：

\[
\widetilde X_l^C=\widetilde B_l+\widetilde D_l.
\]

层间递推为

\[
\widetilde B_{l+1}
=\mathcal C_2\!\left(F_l^P(\widetilde B_l)\right),
\]

\[
\widetilde D_{l+1}
=\mathcal C_4\!\left(
F_l^C(\widetilde B_l+\widetilde D_l)
-F_l^P(\widetilde B_l)
\right).
\]

因此 Current absolute state 从不被一次 rank-6 operator 压缩。总 active block-input rank 是
`2 + (2+4) = 8`，与 `Parent4 + Current4` control 相同。这个固定分配来自“让 defect 拥有 Parent
两倍预算”的机制假说，不是观察五边以后调出的参数。

这项构造本身与 base-plus-delta、low-rank activation 和 residual cache prior art 有明显表面重叠，
特别是 ForkKV/MobiLoRA 一类 base/residual 表示。因此本轮的准入条件不是“代码能写成 base + defect”，
而是：它必须在 matched controls 下产生足够的 functional recovery；否则直接否决，不把 rank allocation
包装成创新。

## 2. primary intervention 为什么放在 native response 边界

为了不让另一个 P8 projection 或 affine moments 掩盖 trajectory 本身，本轮 primary 不先把 K/V defect
投影成 fixed state splice。对每个 Current query、每一层，直接计算

\[
R_l^{M}(q)
=R_l^C(q;K_l^{P,Exact},V_l^{P,Exact})
+\left[
R_l^C(q;\widetilde K_l^C,\widetilde V_l^C)
-R_l^C(q;\widetilde K_l^P,\widetilde V_l^P)
\right].
\]

三项都保留 Current reader 的原生 query--key interaction、`ELU+1` activation 和 value aggregation；
差分在 output projection 以前的 multi-head response 上形成，随后只执行一次 Current `W_O`、gate 和
residual update。它不是把 K correction 当作 attention response 的线性项，也没有用 candidate probe
拟合 correction。

full-rank、关闭截断时，`P_hat=P_exact`、`C_hat=C_exact`，所以上式逐层严格化为 Current-Exact
response。query residual 也因此按层归纳回到 Current Exact。CPU test 同时验证了 trajectory K/V、
post-block defect 与 reader 的这一 finite-release exact limit。

## 3. 固定 protocol 与 controls

- 数据：Yambda-500M Medium seed17，五个独立 edge `v0->v1 ... v4->v5`；
- 用户：仅 discovery 首个 UID `1930`；
- 请求：冻结 odd/held-out 32 candidates；没有读取 label；
- confirmation：未读取；
- Current Exact：只作为评价 target 计算 score，不进入任何 constructor 或 sidecar；
- 初始 factors：直接从 raw history 的 matrix-free HSTU input operator 形成；defect factor 使用
  `X_current-X_parent` operator，没有物化两份 dense `N x H` input；
- upper compression：固定 oversample 4、power 1、seed `17+layer`；
- terminal layer：只形成 RMSNorm 和 K/V，不计算无人消费的 attention/gate/post-block state；
- 不做 rank、depth、seed 或 injection grid。

同一 runner 的固定 controls 为：

1. `ordinary asymmetric P2/C6`：两份 absolute state 独立压缩，active rank、sidecar 与 primary 对齐；
2. `paired absolute P4/C4 native response`：已有 equal-resolution trajectories，但首次在完整原生
   query-conditioned response 上做差；
3. `single absolute C8 factor reader`：最强 generic compressed-Current control；
4. `paired P4/C4 P8 state splice`：已有表示路径，只用于判断 native functional boundary 是否改变结果，
   不进入 primary。

## 4. 五 edge 结果

表中每格是 `logit-gap recovery / probability-gap recovery`。defect-first 两行来自本 runner；已有
paired/single controls 引用专用 authority runner。两者使用同一 UID 与 frozen held-out panel。

| method | v0->v1 | v1->v2 | v2->v3 | v3->v4 | v4->v5 | probability mean | edges >= .80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **defect-first B2+D4 native response** | .569 / .554 | .696 / .690 | .541 / .549 | .808 / .821 | -.077 / -.076 | **.508** | **1/5** |
| ordinary absolute P2/C6 native response | -1.180 / -1.346 | -.962 / -1.125 | -3.823 / -4.040 | -.040 / -.044 | -2.236 / -2.507 | -1.812 | 0/5 |
| **paired absolute P4/C4 native response** | -- / .872 | -- / .921 | -- / .955 | -- / .933 | -- / .824 | **.901** | **5/5** |
| single absolute C8 factor reader | -- / .913 | -- / .999 | -- / .851 | -- / .981 | -- / .937 | **.936** | **5/5** |
| paired P4/C4 P8 state splice control | -- / .870 | -- / .902 | -- / .985 | -- / .868 | -- / .706 | .866 | 4/5 |

defect-first 与 ordinary P2/C6 两行是本 runner 的 raw `logit / probability` vector；已有 paired/single
controls 的 probability 数字与裁决以
[paired native-response preflight](paired_native_response_preflight.md) 的专用实现为准，避免把两个
implementation-specific ledger 或轻微数值差异并列成两份 authority。

### 4.1 defect-first 的硬否决

defect-first 比 matched `ordinary P2/C6` 大幅更好，说明先形成 release difference 确实避免了一部分
absolute-state mismatch；但 `.508` mean、只有 `1/5` edge 达到 `.80`，且最后一条 edge 比 Reuse 更差。
它不满足本轮 admission，也没有资格因为“坐标设计看起来合理”被保留为 Design。

一个与结果一致、但仍需诊断才能确认的解释是：rank-2 Parent base 太不准确；非线性的
`F_C(B+D)-F_P(B)` 虽把预算留给 defect，却是在失真的 base 周围传播差分。这里**不根据结果把 base
改成 rank 3/4，也不做 rank grid**。这一固定实例裁决为 `RETIRE`。

### 4.2 native response placement 有信号，但仍是负面边界

相同的 paired P4/C4 trajectories 从 P8 state splice 改到 native response difference 后，五边
probability recovery 变化为：

```text
+0.22 / +1.99 / -3.00 / +6.50 / +11.77 percentage points
```

专用 runner 报告平均提升 `+3.49` 点，最弱 edge 从 `.706` 提高到 `.824`，从 `4/5` 变成
`5/5 >= .80`。这不是换 trajectory、rank、seed 或 candidate，而只改变“在哪个 Transformer 功能边界
读取 paired defect”。尤其最后两个 edge 的跃升说明：把近似 K/V difference 先冻结为一个
query-independent state projection 会丢失对 nonlinear attention weight 与 value aggregation 的联合
影响；两条 trajectory 的共享误差在 **query-conditioned native response** 上相消得更完整。

这支持一个克制的机制观察：

> 跨版本 bounded trajectories 的差分未必在 K/V tensor 边界已经充分；当两版近似状态分别通过同一个
> Current query 的原生 attention interaction 与 activation 后，版本共享的 approximation error 才可能
> 在 reader response 上变成稳定、因果充分的 functional correction。

但是它不能提升为 Insight 2：response boundary 过滤了一部分 K/V-space 合成误差，却没有补回低秩
trajectory 从未保留的 Current information。

### 4.3 paired native-response 的硬否决：generic C8 control 更强

paired native response 的 authoritative probability mean `.9012` 已达到绝对探索门槛，五边全过
`.80`；但相同 factor sidecar 下，generic single Current-rank8 reader 的 mean 是 `.9364`，高
`3.53` 点，并且 release-time cost 更低。逐边 paired-minus-single 为：

```text
-4.09 / -7.72 / +10.34 / -4.79 / -11.36 percentage points
```

paired path 在 `v2->v3` 显著更强，说明它不是逐值等价于 single-arm compression；但它只在 `1/5`
edge 胜 generic reduced-cache control，总体质量更低、constructor 更贵。因此按照运行前 hard matched-
control gate：

- defect-first B2+D4：**RETIRE**；
- paired P4/C4 native response：**NO-GO / RETIRE，不扩 32-user canary**；
- Design 1：**不冻结**；
- single C8：必须始终同 runner 报告，不能因为它“没有论文味道”而删除。

## 5. 严格 release-time cost、sidecar 与 request cost

成本按 `N=1024,H=192,L=6,heads=6`、multiply-add=`2 FLOPs`、Exact-All
`4,771,282,944 FLOPs/user` 计算。两条新方法按 semantic executor 显式计入每个 nonterminal layer 的
dense residual/defect materialization，再执行固定 power-1 range finder；已有 controls 则引用各自专用
authority ledger。

| method | release FLOPs/user | Exact-All | persistent factor sidecar | result |
| --- | ---: | ---: | ---: | --- |
| defect-first B2+D4 | 880,621,524 | **18.4567%** | 67,584 FP32 = 264 KiB | cost PASS, quality FAIL |
| ordinary absolute P2/C6 | 887,589,512 | 18.6027% | 67,584 FP32 = 264 KiB | quality FAIL |
| paired absolute P4/C4 native | 872,238,088 | **18.2810%** | 67,584 FP32 = 264 KiB | authoritative dedicated ledger; matched-control FAIL |
| single absolute C8 | 853,836,992 | **17.8953%** | representation-dependent | authoritative strongest generic-control ledger |
| paired P4/C4 P8 state splice | 874,402,376 | 18.3264% | 26,624 FP32 = 104 KiB | authoritative representation control |

前两条新方法的数值是本模块按实际 dense residual/defect materialization 给出的保守 ledger；已有
paired/single controls 必须引用专用
[paired native-response preflight](paired_native_response_preflight.md)，其 matrix-free/fused executor 是
authority。本文件不再用自己的 absolute-replay reference cost 覆盖它。

native paired reader 保留 exact Parent Reuse response，并额外读取两份总 rank 8 的 factor caches。相对 Reuse
的两次 factorized prefix response 与 response add/sub 是 `1,218,816 FLOPs/request`，另有 `73,728`
次 native activation evaluation；这不包含 Reuse、query projection、self term、output projection、gate
和 residual 的共同工作。single controls 的 reader ledger 见 authority 文档。以上 request overhead
不能混入 release-time `0--20%` 指标，也不能在系统设计中忽略。

264 KiB sidecar 比先前 P8 state-splice 的 104 KiB 更大，但仍只占 dense six-layer K/V 的约 2.86%。
在 matched-control 已经失败后，不能再以 weakest-edge stability 为理由授权 population expansion。

## 6. novelty / no-go 边界

### 6.1 不能声称的新意

- base-plus-defect、anchor-plus-residual 或双 cache reader；
- low-rank K/V factors、history-axis compression、range finder 或 rank allocation；
- control variate、“两个近似相减”或 full-rank exactness 本身；
- per-user sidecar、后续请求复用或不物化 dense migrated cache。

这些分别与 ForkKV/MobiLoRA、xKV/ShadowKV 和标准 numerical control-variate 思想重叠。尤其本轮失败的
defect-first B2+D4 不能靠改名回避这种 collision。

### 6.2 本轮关闭的机制分支

本轮实际检验了下面四点的组合：

1. 对任意相邻 full-model release，使用两个真实参数端点逐层生成 matched bounded trajectories；
2. exact Parent persistent state 是不动的 serving control variate；
3. 两版近似误差不在 K/V tensor 上直接相减后冻结，而是分别通过同一个实时 Current query 的原生
   attention activation 与 finite KxV aggregation；
4. 只把得到的 query-conditioned response defect 注入 Current gate/residual，且 full-rank 时逐层精确
   回到 Current reader。

它在 dedicated matched-control gate 上失败，且仍可被描述为 paired low-rank replay 加标准
control-variate placement；因此本分支不再扫 rank、basis、probe 或 subtraction boundary。即使 prior-art
检索没有找到逐式相同的方法，数值上被 generic control 支配也已足够 NO-GO。

## 7. 下一步边界

defect-first 与 paired-native 均不进入 formal `32 users x 5 edges` canary，也没有 expansion authorization。
后续若继续寻找论文级 Design，必须引入新的 Current information source 或 migration-ready state interface，
不能继续改变同一 reduced trajectories 的 rank、basis、probe 或 subtraction placement。本轮没有读取
confirmation population。

## 8. 实现与验证

- `scripts/insight_two/defect_first_replay.py`：matrix-free release-defect operator、base/defect recurrence、
  terminal-KV replay、native response-difference reader、matched cost/sidecar ledger；
- `scripts/insight_two/run_defect_first_replay_preflight.py`：UID1930 / odd32 / 五 edge 固定 runner；
- `tests/test_insight_two_defect_first_replay.py`：operator identity、禁止 dense initial projection、recurrence
  exact limit、reader exact limit、sidecar 与 cost tests。

focused verification：`8 passed`；三份文件 `ruff check` 与 `py_compile` 通过。GPU runner 用时约
`21.3 s`，未写 formal result seal。
