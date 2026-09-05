# Paired reduced trajectory 在 S4 functional boundary 的非正式预检

日期：2026-09-03

状态：**UID 1930 / 五 edge 路线淘汰已完成；不建合同、不进入 32/512；不是 Insight 2 或 Design 1**

## 1. 裁决先行

把 `rank-4 Parent + rank-4 Current` 的 approximate finite-release trajectory 编译为 S4 signed
response moments，确实比把同一 trajectory 编译为 shared-`U0` K/V splice 更稳：五 edge
probability-gap recovery 从平均 `0.8662` 提升到 `0.8997`，最差 edge 从 `0.7059` 提升到
`0.8226`。这说明 recommendation reader 的 aggregation 可以滤掉一部分 token-state trajectory
error。

但是这条路线仍必须作为创新 Design 候选停止，理由来自机制、对照与新颖性，而不是用成本失败替代
科学裁决：

1. aggregation 相对 paired K/V 只平均增加 `0.0335`，在 `v2->v3` 反而下降 `0.0362`，不是稳定的
   trajectory-error cancellation amplification；
2. matched total-rank 的 single-arm Current rank-8 functional control 平均 `0.9369`，在 4/5 edge
   优于 paired functional，paired 平均低 `0.0372`；没有出现版本差分特有的优势；
3. dense-initial 账本为 `23.1565%`，但换成仓库已经独立验证语义等价的 matrix-free initial range
   finder 后为 `19.6603%`，所以完整成本 gate 实际通过；不能再用 over-budget 掩盖机制对照失败；
4. 方法本身只是把已有 paired low-rank replay 接到已有 legacy ELU+1 affine-moment compiler 后面。
   它没有产生独立的新闭包或新的 Transformer migration mechanism，不能因为数字接近 `0.90` 就写成
   论文 Design。

因此本预检支持的克制结论是：

> S4 aggregation 是一个有选择性的 trajectory-error filter，但本次证据不支持它把 paired
> finite-release replay 提升为独有的迁移方法。matrix-free executor 可以把它降到 20% 内，但不能
> 创造 paired-specific scientific advantage。

完整 Exact-state P8 moment oracle 平均 recovery `0.9990`，再次确认 S4 functional representation
本身近乎充分；失败发生在 constructor 和 paired-specific mechanism，而不是 functional boundary
不存在。

## 2. 固定问题与协议

本实验只回答一个淘汰问题：在不读取 Current Exact upper-layer state 的前提下，把 paired approximate
trajectory 直接归约到 S4 response space，是否会显著放大两个 release arm 的 approximation-error
cancellation？

协议在执行前固定为：

- frozen discovery 的第一个 UID：`1930`；
- 五条 `v0->v1,...,v4->v5` edge；
- 只读 frozen candidate panel 的 held-out odd 32；
- probe 是完整历史上 8 个等宽区间的 lower midpoint，对应已有 P8 history-probe protocol；
- probe identity、mask 和构造都不读 label、held-out candidate、future event 或 confirmation
  `[512,3000)`；
- paired arm：Parent/Current 均为 `rank=4, oversample=4, power=1, seed=17`；
- single-arm control：Current 为 `rank=8, oversample=4, power=1, seed=17`；
- paired K/V control 继续使用固定 `U0 rank=8, oversample=4, power=0, seed=1017`；
- device 固定为 `cuda:1`，只作为非正式资源记录；五 edge 总执行时间 `18.63s`。

没有搜索 rank、probe 数、mask、seed 或 edge-specific 规则。Current Exact cache 只在全部 legal
constructor 完成以后生成，用于评分 reference 和明确标注的 full-moment oracle。

旧 functional-boundary 合同中的 `research_plan` hash 因本轮持续追加 living plan 而失配；runner
没有改合同或放宽其他 seal。数据、population、candidate panel、prior evidence 和 v0--v5 checkpoint
hash 均逐项通过旧合同记录的校验，并在输出中显式记录 plan expected/actual hash。

## 3. S4 compiler 的精确语义

P8 history-item probes 通过 Current query encoder，并在每层沿 exact Parent cache 的 coherent Reuse
path 得到 query `q`。对每个 release arm、layer 和 head，P8 对历史位置的 QK 符号投票定义 majority
positive region。

若 reduced cache layer 为

\[
K=LC_K,\qquad V=LC_V,
\]

则不物化 approximate K/V 就能精确形成该 approximate trajectory 的 positive ELU+1 moments：

\[
B_h=\left(\sum_n m_{hn}L_n\right)C_{V,h},
\]

\[
M_h=C_{K,h}^{\mathsf T}
\left(L^{\mathsf T}\operatorname{diag}(m_h)L\right)C_{V,h}.
\]

对 paired path，持久化

\[
\Delta B_l=\widetilde B_l^C-\widetilde B_l^P,
\qquad
\Delta M_l=\widetilde M_l^C-\widetilde M_l^P.
\]

后续 Current reader 仍计算 exact Parent prefix response，并在 attention output projection、gate 和
residual 之前加入

\[
\Delta r_l(q)=\Delta B_l+s_lq\Delta M_l.
\]

所以 intervention 位置是 S4 aggregated context，而不是最终 score offset。P8 仅定义 activation
region；held-out odd-32 query 没有参与 constructor。

实现测试证明两项 correctness invariant：

1. factorized mask、`B` 和 `M` 与先物化 `LC_K/LC_V` 再做 dense moment 的结果一致；
2. full token rank 时，paired functional compiler 与 full Exact functional-moment oracle 一致到数值
   tolerance。

这些 invariant 只证明 compiler 正确，不证明 low-rank trajectory 或 Design 有效。

## 4. 五边结果

下表均为 UID `1930`、held-out odd-32 的 label-free probability-gap recovery；Reuse 归一化为 `0`，
Current Exact 归一化为 `1`。

| edge | paired r4/r4 K/V splice | paired r4/r4 S4 moments | single Current r8 S4 moments | full Exact S4 oracle |
| --- | ---: | ---: | ---: | ---: |
| `v0->v1` | 0.8703 | 0.8725 | 0.9137 | 0.9997 |
| `v1->v2` | 0.9015 | 0.9212 | 0.9983 | 0.9998 |
| `v2->v3` | 0.9849 | 0.9487 | 0.8544 | 0.9960 |
| `v3->v4` | 0.8685 | 0.9335 | 0.9814 | 1.0000 |
| `v4->v5` | 0.7059 | 0.8226 | 0.9367 | 0.9992 |
| **edge mean** | **0.8662** | **0.8997** | **0.9369** | **0.9990** |
| **minimum** | **0.7059** | **0.8226** | **0.8544** | **0.9960** |
| **edges >= 0.80** | **4/5** | **5/5** | **5/5** | **5/5** |

paired S4 相对 paired K/V 的逐边增益为：

~~~text
+0.0022, +0.0197, -0.0362, +0.0650, +0.1167
~~~

它在后两条难边上有明显正作用，尤其将 `v4->v5` 拉回 `0.80` 以上；但一条 edge 下降，前三条中
两条增益不足 `0.02`。因此可以写成 aggregation filter 的线索，不能写成“aggregation 稳定放大 paired
cancellation”。

paired S4 相对 single-arm S4 的逐边差为：

~~~text
-0.0412, -0.0771, +0.0943, -0.0479, -0.1141
~~~

paired 只在 `v2->v3` 胜出。这是本路线最重要的 negative control：把两臂差分移到 S4 并没有产生
single-arm compression 所没有的总体 functional advantage。

## 5. 完整成本，而不是只报 38,016 个 moments

沿用 Medium `6L/H192/6 heads/d32/N1024/P8` 与

~~~text
Exact-All = 4,771,282,944 FLOPs/user
~~~

的 frozen multiply-add=`2 FLOPs` 口径。

先给出与本次 dense semantic runner 直接对应的账本：

| component | paired r4/r4 S4 | single Current r8 S4 | full Exact S4 oracle |
| --- | ---: | ---: | ---: |
| bounded trajectory；final layer KV-only | 1,039,053,832 | 882,314,368 | 4,771,282,944 |
| P8 probe trace through exact Parent | 56,168,448 | 56,168,448 | 56,168,448 |
| mask + moment build | 9,603,072 | 108,767,232 | 194,568,192 |
| signed moment subtraction | 38,016 | 38,016 | 38,016 |
| **constructor total** | **1,104,863,368** | **1,047,288,064** | **5,022,057,600** |
| **Exact-All fraction** | **23.1565%** | **21.9498%** | **105.2559%** |
| **`<=20%`** | **FAIL** | **FAIL** | oracle |

paired factor-aware moment build 很便宜，但它无法消除 dense initial formation 的重复工作。这里没有
把 P8 probe 当作免费，也没有只按最终 rank 而忽略 oversampling、power iteration、nonlinear attention、
gate/residual boundary 或前五层 recompression。

实际 Python prototype 仍执行第六层完整 block output；表中已经给它采用更有利且合法的 KV-only
final-layer specialization，因为迁移只需要第六层 K/V。

### 5.1 Matrix-free initial factor 后的严格准入成本

仓库已经独立验证，初始 fixed range finder 可以从 raw item/behavior lookup、temporal feature 与公开
projection weight 通过 `X@R` / `X^T@Q` matrix-free application 形成，与 dense input 后再 range-find
语义等价。它不改变 rank、oversampling、power iteration、seed 或五边 quality，也不是 Insight 2 创新。

对 paired r4/r4，替换为：

\[
1,039,053,832
-2(88,489,984+12,951,382)
+2(18,033,494)
=872,238,088
\]

trajectory FLOPs；再完整加入 P8 probe、factor-aware masks/moments 和 subtraction：

\[
872,238,088+56,168,448+9,603,072+38,016
=938,047,624=19.6603\%\;Exact-All.
\]

对 single Current r8，对应 trajectory 为：

\[
882,314,368-(88,489,984+19,766,208)+27,282,880
=801,341,056,
\]

完整 functional constructor 为：

\[
801,341,056+56,168,448+108,767,232+38,016
=966,314,752=20.2527\%\;Exact-All.
\]

最终 cost sensitivity 是：

| path | dense initial | matrix-free initial | `<=20%` after matrix-free |
| --- | ---: | ---: | ---: |
| paired r4/r4 S4 moments | 23.1565% | **19.6603%** | **PASS** |
| single Current r8 S4 moments | 21.9498% | **20.2527%** | FAIL |
| paired r4/r4 K/V splice | 21.8226% | **18.3264%** | **PASS** |
| full Exact S4 oracle | 105.2559% | not applicable | oracle |

所以 paired functional 的预算阻塞已经解除。路线仍被否决，是因为 matched functional boundary 上它
只在 1/5 edge 超过 single-arm，并且 scientific object 仍是 compression 加已知 moment compiler 的组合。
不能把 `19.6603%` 的系统实现改进包装成新的 Transformer Insight。

每个方法的 S4 persistent state 为

\[
6\times 6\times(32+32^2)=38,016\ \text{FP scalars},
\]

即 full Current K/V 的 `1.6113%`；FP32 为 `152,064 bytes/user`。两套 majority masks 共
`73,728` logical transient bits，不是 persistent state。每个 future query 的 incremental moment read
为 `77,184 FLOPs`，明显低于 fixed-`U0` K/V splice 的约 `1,254,528 FLOPs/query`。这是 functional
interface 的系统优点；matrix-free 后 constructor 也已进入甜点区。但两者都不能自动形成论文新意。

single-arm 的 dense exact-Parent control moments 需要额外完整扫描 Parent K/V；这正是其
`108,767,232` mask/moment 项，未被 Parent cache “已经存在”这一事实隐藏为零。

`19.6603%` 是仓库统一的理论 FLOP 口径，不是 wall-time 比例。paired matrix-free input 还显式产生
`65,536` 次 sin/cos evaluation、`3,072` 个 Gaussian draws、`786,432` 个 embedding lookup payload
scalars 和 `6,144` 个 raw-history fields；两个 arm 的 P8 region 形成包含 `589,824` 次 sign comparison。
这些不塞进 matmul FLOP 分子，但已作为独立计数保留。预算 margin 只有约 `16.21M FLOPs/user`
（`0.3397` 个百分点），所以任何未来 kernel 若增加新的 dense pass 都必须重新审计，不能继续沿用
`19.6603%`。

## 6. Append / eviction closure：明确为否

response moments **不具有 single-arm 或 K/V sidecar 所没有的新 append/eviction closure**。现有
`38,016`-scalar sidecar 甚至比 token-aligned `U0` K/V correction 更缺少删除语义。

对 append-only，可以定义一个有限的 hybrid lineage：旧的 Parent rows 继续由 frozen signed moments
修正，新行为直接由 Current model 写 native K/V，因此新 rows 的 migration delta 为零，global old-history
moments 不更新。这一规则不读 future，也不会反向改写旧 rows，但它只给出合法执行定义，不给出质量
定理：append 后 layer query 可能离开 cutover P8 定义的 majority cone，原 moments 的 affine bulk 是否
仍充分只能做经验 persistence 检查。

对任意 eviction，当前 sidecar 不闭合。若 Parent base 删除历史 row `i`，必须同步从每层/head 的
`Delta B/Delta M` 中删除该 row 的 signed contribution；global sum 无法反推出该贡献。cutover 的两套
majority masks 也没有持久化为可删除 ledger。可选补救都会改变状态与成本：

- 保存每 token contribution 需要约 `N x 38,016 = 38,928,384` scalars/user，约为 full KV 的
  `16.5x`，不可接受；
- 保存 8 个 chronological segment moments 需要 `304,128` scalars，约为 full KV 的 `12.8906%`，
  只能支持对齐 segment 的删除，还需 boundary/mask metadata；
- eviction 时重放被删 token 或重建全局 moments 会重新引入 raw-history I/O 和 release/serving compute，
  不能称为现有 sidecar 的闭包。

此外，paired 与 single-arm functional control 使用完全相同的 moment state，所以即便增加 segment
ledger，也不存在 paired-specific closure。结论是：**不值得为本路线启动 rolling mechanism
experiment**。最多可以在其他 Design 已冻结后，把 append-only persistence 当作 functional-boundary
observation；在此之前不能用 rolling 数字挽救本次 matched control 失败。

## 7. 新颖性与 Insight 2 边界

本路线不应被命名为新的 migration algorithm。其组件分别是：

- standard randomized low-rank Current/Parent replay；
- two-approximation subtraction；
- legacy ELU+1 positive-region affine moments；
- exact Parent response 加一个 signed functional residual。

factor-aware `L^T diag(mask)L` 是正确且有用的 compiler optimization，但它只是已有 moment algebra 在
low-rank factors 上的结合律，不是新的 Transformer mechanism。当前实验也没有给 paired path 提供
single-arm 没有的稳定收益。若把这条组合直接写成 Design，reviewer 可以准确地将它归类为“compression
后接已有 associative summary”。

因此本轮对 Insight 2 的贡献只保留一条观察：

> 对同一 approximate release trajectory，reader-level aggregation 有时比 token-state splice 更能保留
> 推荐功能，尤其可能抑制某些 difficult-edge error；但这种抑制并不自动证明 paired release evolution，
> 也不保证优于普通 Current compression。

下一条真正值得进入 canary 的方法必须同时满足：

1. functional object 在 Transformer 递推中原生形成，而不是先支付完整 reduced-KV trajectory 再做
   moment 编译；
2. paired/version structure 相对 matched single-arm 有独立、稳定、可定位的机制增益；
3. 完整 constructor 在加入 probe、mask、build、write 后仍 `<=20%`；
4. 方法不是 mapper、moment、sampling 或 compression 的再组合。

在出现这样的结构以前，本路线不建立 prospective 32-user 合同，不读 512 discovery/confirmation，也
不通过调整 rank、P8/P32、mask 或 seed 追数值。

## 8. 实现与复现

- `scripts/insight_two/paired_functional_boundary.py`：factor-aware mask/moment identity、paired/single/full
  compiler 与静态成本账本；
- `scripts/insight_two/run_paired_functional_boundary_preflight.py`：固定 UID 1930、五 edge、P8/odd32、
  `cuda:1` runner；
- `tests/test_insight_two_paired_functional_boundary.py`：factorized/dense identity、full-rank Exact limit 和
  budget/cost invariant。

执行：

~~~bash
PYTHONPATH=src:scripts pytest -q tests/test_insight_two_paired_functional_boundary.py
PYTHONPATH=src:scripts python \
  scripts/insight_two/run_paired_functional_boundary_preflight.py \
  --device cuda:1
~~~
