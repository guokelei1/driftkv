# Matched Parent-cache differential preflight

日期：2026-09-03  
状态：**严格单 UID 非正式否证；shortcut RETIRE；不是 population evidence，不是 formal contract**

## 1. 裁决

把 exact Parent persistent K/V 事后做 matched rank-4 joint-`[K,V]` approximation，不能代替
rank-4 Parent Transformer replay。它虽然有一个约 `14.15% Exact-All` 的 prospective cost window，
但五条 edge 的 held-out probability-gap recovery 只有：

~~~text
0.3858 / 0.2153 / 0.3467 / 0.7608 / -0.3754
~~~

edge mean 为 `0.2666`，还低于直接用 approximate Current 减 exact Parent 的 control（`0.3553`）。
相同配置下，真正的 Parent/Current 两条逐层 replay 为：

~~~text
0.8701 / 0.9014 / 0.9850 / 0.8686 / 0.7059
~~~

因此本轮得到的机制结论不是“对 Parent cache 做一次对称压缩就能保留 control-variate gain”，而是更窄、
也更有辨识力的否证：

> paired 路径的正信号依赖两版状态经过各自 Transformer attention--gate--residual 链以后形成的
> **trajectory-matched approximation error**；对已经形成的 Parent K/V 做静态低秩近似，并不会生成与
> Current reduced trajectory 同源的 truncation bias。

这个 shortcut 只是 ordinary cache compression 与 residual splice 的组合，没有形成新的方法，立即
`RETIRE`。若以后要把 paired finite-release lead 压到 20% 以下，需要在双臂递推内部共享或消去计算，
不能把 Parent trajectory 换成一次 post-hoc cache factorization。

> 后续更新：matrix-free input 已在不替换 Parent trajectory 的前提下把 full paired KV-only 成本降至
> `18.3264%`。本 shortcut 的质量否证不变；当前主阻塞已经是 paired 未超过 single-arm，而不是成本。

## 2. 固定问题与执行边界

本轮只回答一个预先限定的问题：能否保留 paired rank-4/rank-4 的 approximation-error cancellation，
同时省掉 Parent block replay。没有搜索 rank、seed、edge、candidate 或 numerical schedule。

- frozen discovery 的第一个 UID：`1930`；
- 五条 Medium edge：`v0->v1` 至 `v4->v5`；
- 只读 odd-index held-out 32 candidates；没有读 anchor、label 或 confirmation `[512,3000)`；
- Current arm 固定 `rank=4, oversample=4, power=1, seed=17`；
- layer-0 defect basis 固定 `rank=8, oversample=4, power=0, seed=1017`；
- Current Exact K/V 只在评价端产生 Exact score，不进入任何 migration constructor；
- 没有训练、没有 formal contract、没有写 result seal。

对 exact Parent persistent cache 的每层

\[
Z_l^P=[K_l^P\mid V_l^P]\in\mathbb R^{N\times 2H}
\]

使用同一固定 randomized range-finder protocol 得到 rank-4 factor
`Z_l^{P,cache}`。Current reduced replay 给出 `Z_l^{C,replay}`。从 layer 0 的

\[
\widehat D_0=Z_0^{C,replay}-Z_0^{P,cache}
\]

构造 `U0`，最终只向 exact Parent base 加 signed cores：

\[
Z_l^M=Z_l^P+U_0U_0^\top
\left(Z_l^{C,replay}-Z_l^{P,cache}\right).
\]

持久化格式没有变：一个 `N x 8` basis 加六层 K/V cores，共 `26,624` scalars/user，约为完整
Current KV 的 `1.1285%`。这不是质量优势，只说明本轮公平地保留了同一个 serving interface。

## 3. 三条 matched path 的结果

下表全部是 UID `1930` 的 held-out odd-32 probability-gap recovery；它们只能用于路线筛选。

| edge | Current r4 - exact Parent | Current r4 - matched Parent-cache r4 | Current r4 - Parent-replay r4 |
| --- | ---: | ---: | ---: |
| `v0->v1` | 0.5109 | 0.3858 | 0.8701 |
| `v1->v2` | 0.2680 | 0.2153 | 0.9014 |
| `v2->v3` | 0.3742 | 0.3467 | 0.9850 |
| `v3->v4` | 0.7407 | 0.7608 | 0.8686 |
| `v4->v5` | -0.1173 | -0.3754 | 0.7059 |
| **edge mean** | **0.3553** | **0.2666** | **0.8662** |

matched Parent-cache 相对 exact-Parent subtraction 的逐边变化是：

~~~text
-0.1252 / -0.0527 / -0.0275 / +0.0201 / -0.2581
~~~

它只在一条 edge 有 `+0.02` 的微小改善，其余四条下降，且最后一条比 Reuse 更差。相对真正 paired
replay，它逐边少恢复：

~~~text
0.4843 / 0.6861 / 0.6383 / 0.1078 / 1.0813
~~~

这个差距太大，不能解释为再调一个 randomized seed、换一个 rank 或多一次 power iteration即可修复；
这些改动也会违反本轮先固定机制再否证的目的。

## 4. 为什么静态 matched cache 不会相消 trajectory error

令两版逐层 bounded replay 的 K/V error 为：

\[
E_l^m=Z_l^{m,replay}-Z_l^m,\qquad m\in\{P,C\}.
\]

paired replay 的 projected error 项是：

\[
\Pi_0(E_l^C-E_l^P).
\]

`E_l^P` 与 `E_l^C` 都经过 matched block-input compression、RMSNorm、native attention、multi-head
merge、gate、residual 和下一层 recompression；邻近 release 可能让这些误差保持相关。

本轮 shortcut 实际替换的是：

\[
E_l^{P,cache}=\operatorname{Compress}(Z_l^P)-Z_l^P.
\]

它只对已经形成的 K/V 做一次 joint matrix approximation，没有经历 Parent block 的 state recurrence。
即使 `E_l^{P,cache}` 与 `E_l^C` 使用相同 nominal rank、oversampling、power 和 Gaussian seed，它们也不是
同一计算边界产生的误差。数值结果说明：**matched numerical resolution 不等于 matched formation
trajectory**。

这也排除了一个较弱解释：paired 正信号并不是“给相减两边都加任意 rank-4 bias就自然会抵消”。如果
只是低秩维数对称或差分至多 rank 8 在起作用，static matched-cache path 应接近 two-replay path；实际
edge mean 相差 `0.5996`。

## 5. Prospective cost：便宜，但质量门已经失败

沿用现有 Medium ledger：`N=1024,H=192,L=6`，multiply-add=`2 FLOPs`，Exact-All 为
`4,771,282,944 FLOPs/user`。Current rank-4 arm 采用合法 final-layer KV-only specialization 后为
`519,526,916 FLOPs/user`。

对每层 `N x 2H` Parent joint-`[K,V]` 做 target-rank 4、sketch-width 8、`power=1` range finder：

\[
C_{P,cache/layer}
=8N(2H)s+2C_{QR}(N,s)+C_{truncate}(2H;s,4)
=25,571,158.
\]

六层为 `153,426,948`。再计入 factor-aware layer-0 `U0` builder `1,247,808` 和 upper-five signed-core
build `916,480`：

\[
C_{matched-cache}
=519,526,916+153,426,948+1,247,808+916,480
=675,118,152,
\]

即：

\[
14.1496\%\;\text{Exact-All}.
\]

所以这个 shortcut 在理论算术上进入 0--20% 甜点区，也比 paired KV-only 的 `21.8226%` 低很多；但
它的 edge-mean recovery 只有 `0.2666`，且 1/5 edge 为负。成本通过不能挽救机制/质量失败。真实实现
仍需单列多次 Parent-cache scan 的 bytes、QR/SVD workspace 和 wall time；因为路线已经在质量上
`RETIRE`，不再为它建立 production kernel 或 formal cost contract。

## 6. 实现与可复核性

- semantic module：`scripts/insight_two/matched_parent_cache_differential.py`；
- 非正式固定 runner：`scripts/insight_two/run_matched_parent_cache_differential_preflight.py`；
- unit tests：`tests/test_insight_two_matched_parent_cache_differential.py`。

测试覆盖：fixed joint-`[K,V]` operator 的确定性、full-token-rank exact limit、paired replay 确实减去
approximate Parent trajectory 而非 exact Parent，以及 shared-basis/per-layer-core sidecar 形状。

复核命令：

~~~bash
PYTHONPATH=src:scripts pytest -q tests/test_insight_two_matched_parent_cache_differential.py
ruff check scripts/insight_two/matched_parent_cache_differential.py \
  scripts/insight_two/run_matched_parent_cache_differential_preflight.py \
  tests/test_insight_two_matched_parent_cache_differential.py
PYTHONPATH=src:scripts python \
  scripts/insight_two/run_matched_parent_cache_differential_preflight.py \
  --device cuda:0
~~~

最终路线裁决：**RETIRE static matched Parent-cache differential；保留 paired finite-release trajectory
error cancellation 作为尚未冻结、且仍需低于 20% executor 的 scientific lead。**
