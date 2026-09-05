# Probe-free all-history affine response invariant：结构性 falsifier

日期：2026-09-03

状态：**UID 1930 / 五 edge 单配置 falsifier 完成；representation oracle 已失败，路线停止；不是 Insight 2 或 Design 1**

## 1. 裁决先行

把 strongest single Current-r8 reduced replay 编译成无需 probe 的 all-history affine response state，
完整成本为 `18.5548% Exact-All`，确实进入甜点区；但功能结果灾难性失败：五 edge probability-gap
recovery 为

~~~text
-0.1082, 0.7168, -12.9945, 0.9249, -1.8267
~~~

只有 2/5 edge 正向、1/5 达到 `0.80`，edge mean 为 `-2.6575`。`v2->v3` 不只是没有修复，
而是把 probability gap 放大到 Reuse 的约 14 倍。

这不是 rank-8 replay 误差造成的。使用完整 Current Exact K/V 构造同一个 all-position affine state，
五 edge oracle 为

~~~text
-0.0569, 0.7333, -13.0716, 0.9195, -1.9074
~~~

edge mean `-2.6766`，与 executable 几乎相同。因此 representation 在 exact-state ceiling 已经被否决，
不允许继续搜索 rank、probe、mask、kernel feature 或 edge-specific fallback。

这个结果修正了对“negative-response fraction 仅约 4%”的过度解释：小的平均 response fraction
不构成跨版本 signed residual 的误差上界。将少量 negative-logit 位置错误地延拓到线性支路后，它们的
Current-minus-Parent 向量差可以很大，并沿后续 output projection、gate 和 residual 多层放大。

科学结论是：

> Legacy ELU+1 reader 的 functional compactness 不能退化成一个 query-independent、全历史的二阶
> affine invariant；activation-region information 虽然只涉及较小的 negative branch，却是跨版本
> response correction 的必要组成。

即使数值通过，这个状态也只是标准 linear-attention / fast-weight memory，不具备论文新意。本次则在
新颖性审查之前已经被 representation oracle 直接否决。

## 2. 固定协议与合法性

本预检只运行一个事前固定配置：

- frozen discovery 第一个 UID `1930`；
- 五条 `v0->v1,...,v4->v5` edge；
- 只在 frozen held-out odd-32 candidate panel 上评价；
- single Current replay 固定 `rank=8, oversample=4, power=1, seed=17`；
- initial factor 的理论 executor 固定为已经验证语义等价的 matrix-free operator；
- primary constructor 的 probe count 为 `0`，不接收 candidate/query、label、mask 或 future event；
- Current Exact 在所有 legal path 构造完成以后才生成，只用于 evaluation reference 和明确命名的
  full-Exact representation oracle；
- 不读 confirmation `[512,3000)`，不训练，不建立/修改合同与 result seal；
- device 固定 `cuda:1`，五 edge 非正式执行 `18.72s`。

旧 functional-boundary 合同中的 living `research_plan` hash 因本轮追加记录而变化；runner 显式记录
expected/actual hash，只允许这一项 drift。数据、population、candidate panel、prior evidence 和 v0--v5
checkpoint hash 均按旧合同逐项验证。

## 3. 无 probe functional state

对每个 layer/head，不做 activation-region mask，直接在全部 `N` 个历史位置形成

\[
B_h=\sum_{i=1}^{N}V_{hi},
\qquad
M_h=\sum_{i=1}^{N}K_{hi}^{\mathsf T}V_{hi}.
\]

reader correction 为

\[
\Delta r_h(q)=\Delta B_h+s_hq_h\Delta M_h,
\]

其中

\[
\Delta(B,M)=(\widetilde B^C,\widetilde M^C)-(B^P,M^P).
\]

Current reader 仍原生读取 exact Parent response，再在 S4 aggregated-context boundary 加上这项 signed
response，随后执行真实 output projection、gate 和 residual。它不是最终 score offset，也不 materialize
translated Current K/V。

对 factorized reduced layer

\[
K=LC_K,\qquad V=LC_V,
\]

可以不物化 K/V 而精确形成该 approximate trajectory 的 invariant：

\[
B=(\mathbf 1^{\mathsf T}L)C_V,
\qquad
M=C_K^{\mathsf T}(L^{\mathsf T}L)C_V.
\]

测试已证明 factorized 公式与 materialized dense cache 一致；在 full token rank 时，single-arm compiler
回到 full-Exact all-history affine oracle。这里“exact”只指对所选 affine invariant 的代数实现，不表示
该 invariant 等于 native ELU+1 response。

## 4. 为什么 4% negative response 仍不可删除

ELU+1 对 logit `z` 的真实权重是

\[
f(z)=
\begin{cases}
1+z,&z\ge 0,\\
e^z,&z<0.
\end{cases}
\]

all-history invariant 将所有负位置也当成 `1+z`。即使 negative branch 对单个 reader response 的平均
norm fraction 约为 4%，仍不能推出

\[
\lVert(r^C-r^P)-(r^C_{aff}-r^P_{aff})\rVert
\]

很小，原因包括：

1. fraction 是单边 response 统计，而迁移对象是两个版本的 signed difference；两个大向量相减后，小
   branch 可以占据主要 residual；
2. `1+z` 在 `z<-1` 时甚至变成负权重，而 `e^z` 始终为正；position fraction 小不等于向量误差小；
3. value direction 会发生抵消，small activation mass 不约束 vector norm 或 downstream score；
4. 每层 correction 会改变下一层 query，再经过 gate/residual 递推放大。

full-Exact oracle 与 rank-8 executable 的近乎同样失败直接排除了“提高 trajectory rank 即可修复”的解释。
`v2->v3` 的 top-10 overlap 降到 `0`，rank correlation 约 `-0.84`，也说明这不是 recovery denominator
很小造成的单一数值视觉问题。

## 5. 五边 matched controls

下表均为 UID `1930`、held-out odd-32 probability-gap recovery；Reuse 为 `0`，Current Exact 为 `1`。

| edge | single r8 all-affine | full Exact all-affine oracle | single r8 P8 moments | single r8 K/V splice |
| --- | ---: | ---: | ---: | ---: |
| `v0->v1` | -0.1082 | -0.0569 | 0.9137 | 0.8610 |
| `v1->v2` | 0.7168 | 0.7333 | 0.9983 | 0.9173 |
| `v2->v3` | -12.9945 | -13.0716 | 0.8544 | 0.9852 |
| `v3->v4` | 0.9249 | 0.9195 | 0.9814 | 0.9473 |
| `v4->v5` | -1.8267 | -1.9074 | 0.9367 | 0.9753 |
| **edge mean** | **-2.6575** | **-2.6766** | **0.9369** | **0.9372** |
| **minimum** | **-12.9945** | **-13.0716** | **0.8544** | **0.8610** |
| **positive edges** | **2/5** | **2/5** | **5/5** | **5/5** |
| **edges >=0.80** | **1/5** | **1/5** | **5/5** | **5/5** |

P8 和 K/V controls 在同一个 runner、同一 replay、同一 candidate panel 中都保持约 `0.937` mean，证明
checkpoint、history 或 rank-8 trajectory 没有失效。真正的分叉只有：P8 保留 query-conditioned sign
region，K/V reader保留完整 native activation，而 all-affine 抛弃了 sign region。

## 6. 严格 matrix-free 成本

Exact-All 分母保持 `4,771,282,944 FLOPs/user`。primary all-affine 的完整账本为：

| component | FLOPs/user |
| --- | ---: |
| single-r8 matrix-free/final-KV-only trajectory | 801,341,056 |
| factor-aware approximate Current `B/M` | 6,064,128 |
| dense exact Parent `B/M` | 77,856,768 |
| signed moment subtraction | 38,016 |
| **total** | **885,299,968** |
| **Exact-All fraction** | **18.5548%** |
| **`<=20%`** | **PASS** |

完整 controls：

| path | constructor / Exact-All | sidecar scalars | incremental reader FLOPs/query |
| --- | ---: | ---: | ---: |
| single r8 all-affine | **18.5548%** | 38,016 | 77,184 |
| full Exact all-affine oracle | 103.2644% | 38,016 | 77,184 |
| single r8 P8 moments | 20.2527% | 38,016 | 77,184 |
| single r8 K/V splice | **17.8953%** | 26,624 | 1,254,528 |

all-affine 不运行 P8 trace、不形成 mask。它仍显式支付 exact Parent full-history moment scan，不能因为
Parent cache 已存在而写成零。FP32 moment sidecar 为 `152,064 bytes/user`，即 full Current K/V 的
`1.6113%`。matrix-free input 另报告 `32,768` 次 sin/cos、`2,304` 个 Gaussian draws、`393,216`
embedding payload scalars 和 `3,072` 个 raw-history fields；`18.5548%` 不是 wall-time 比例。

成本通过但 representation oracle 失败，因此这条路线不进入更多用户。它也说明甜点区本身不是充分
条件：必须先有因果充分的 functional object。

## 7. Prior art 与新颖性审计

令

\[
\phi(k)=[1,k],\qquad \phi(q)=[1,sq].
\]

则 all-history state 可写成

\[
S=\sum_i\phi(k_i)^{\mathsf T}v_i,
\qquad r(q)=\phi(q)S.
\]

这正是标准的 unnormalised linear-attention / fast-weight outer-product memory，而不是新的 migration
object。[Transformers are RNNs](https://proceedings.mlr.press/v119/katharopoulos20a.html) 已系统使用
feature-map associative state；[Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html)
进一步明确了 outer-product fast-weight 解释。把 Current-minus-Parent 的两个这种状态相减，也只是
linearity，不产生新的 Transformer principle。

因此即使五边质量很高，这条路径最多也只能是 probe-free linear-attention baseline。当前 quality 已经
失败，更没有重新命名或扩展的理由。若继续用 polynomial/random-feature 近似 `e^z`，研究对象会变成
已有 kernel/linear attention approximation；若恢复 sign mask，则重新回到 query-conditioned P8/cone
路线。两者都不应通过扫 feature 数来挽救本 falsifier。

`B/M` 在代数上可按新 token 相加，但单一 global state 仍不能在不知道被删 row contribution 时支持
任意 eviction；segment/per-token ledgers 会引入先前已经审计的额外 storage。这个 associative property
也是 linear attention 的已有性质，不构成 recommendation migration novelty。

## 8. 停止条件与留下的研究边界

本轮同时满足两个独立停止条件：

1. full-Exact representation oracle 在 3/5 edge 为负，且出现 `-13.07` 的强反例；
2. proposed state 与 linear-attention fast-weight state 完全同构。

所以不建立 32-user prospective contract，不读 discovery/confirmation，不运行 rolling，不调 rank/probe/
feature map。可以持久化到 Insight 2 的边界只有：

> Query-dependent activation geometry 不是可删除的 estimator detail。一个 compact functional state 若要
> 忠实迁移 legacy HSTU release response，必须保留 native nonlinear reader 对历史证据的条件化作用；
> 仅保留全局一阶/二阶 affine moments 即使存储和计算都很低，也可能比 Reuse 严重得多。

这并不自动证明 P8 是最终 Design；P8 仍是已有 moment compiler 且成本/closure/novelty 已有独立问题。
它只排除了最简单的 probe-free response invariant。

## 9. 实现与复核

- `scripts/insight_two/all_history_affine_response.py`：probe-free factor/dense moments、合法/Exact compiler
  与成本账本；
- `scripts/insight_two/run_all_history_affine_response_preflight.py`：固定 UID 1930、odd32、五 edge、
  `cuda:1` falsifier；
- `tests/test_insight_two_all_history_affine_response.py`：factorized/dense identity、full-rank Exact limit、
  no-probe API 与 cost invariant。

~~~bash
PYTHONPATH=src:scripts pytest -q \
  tests/test_insight_two_all_history_affine_response.py
PYTHONPATH=src:scripts python \
  scripts/insight_two/run_all_history_affine_response_preflight.py \
  --device cuda:1
~~~
