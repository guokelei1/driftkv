# Producer-state × reader-version commutator：正 oracle、负 Design

日期：2026-09-03  
状态：**ORACLE OBSERVATION PASS；DESIGN NO-GO / RETIRE**  
范围：固定 UID1930、五条 Medium edge、frozen held-out odd-32；无 label、无 confirmation、无 sweep

## 1. 问题与四条 exact path

对 reader 版本 (r\in\{P,C\}) 和 cache producer 版本
(p\in\{P,C\})，定义：

\[
F(r,p)=\operatorname{Reader}_{r}(\operatorname{KV}_{p}(x),q_r).
\]

每个 reader 使用自己的 candidate/query embedding、Q projection、gate、residual、final norm 和 score
head；只有 persistent history K/V 由 producer 指定。四个角分别是 Exact (F(C,C))、线上 Reuse
(F(C,P))、reverse cross path (F(P,C)) 和 Parent native path (F(P,P))。本轮固定检验：

\[
\widehat F(C,C)=F(C,P)+[F(P,C)-F(P,P)],
\]

其误差恰好是 reader update 与 producer-state update 的 mixed finite difference：

\[
\Omega=F(C,C)-F(C,P)-F(P,C)+F(P,P).
\]

这是 exact-state 结构诊断。(F(P,C)) 读取了不可用的 Exact Current K/V，所以从一开始就不是
可执行 migration action。

## 2. 结果：raw score 上确实近似交换

| edge | probability-gap recovery | score mixed/state L2 | 两个 state-effect cosine |
| --- | ---: | ---: | ---: |
| v0→v1 | .9657 | .0430 | .99930 |
| v1→v2 | .7651 | .2301 | .999997 |
| v2→v3 | .9333 | .0701 | .999863 |
| v3→v4 | .9427 | .0620 | 1.000000 |
| v4→v5 | .8087 | .1886 | .999969 |
| edge-equal mean | **.8831** | — | — |

按运行前的 oracle 门（mean ≥ .80 且至少 4/5 edge ≥ .80），**近交换结构通过**。这说明相邻
release 中，历史 producer state 对 score 的有限影响，在 Parent reader 与 Current reader 下几乎沿同一
方向；reader-version update 主要改变了该影响的幅值，而没有完全改变方向。

这是一条有用的 Transformer-reader 观察，但不能只看 raw score。去掉每个 candidate panel 的共享
分量后，score-space L2 recovery 为：

```text
-4.6972 / .8844 / .7127 / .8052 / .4020, mean=-.3786
```

只有 2/5 edge 达到 .80。v0→v1 的 raw `.9657` 尤其由 candidate-shared logit shift 主导；其排序相关
state effect 很小，两个 reader 的残余方向不再一致。因此 `.8831` 不能被写成“推荐决策 correction 已被
Parent reader 恢复 88.3%”。它首先是 calibration/common-mode 的交换性，candidate-dependent 部分并不
稳定。

## 3. S4 与 readout：结构存在，但不是单一稳定边界

四角路径也在每层 native aggregation 后计算同一个 Ω。raw S4 L2 recovery 很高：layer 0 的五边为
`.950/.953/.966/.984/.972`。candidate-centered layer 0 仍为
`.857/.915/.806/.930/.845`，layer 5 为 `.855/.873/.893/.916/.813`。这说明结果不只是 final score
head 的偶然抵消。

但中间层不单调：candidate-centered layer 2 在 v1→v2 为 `-.695`，layer 3 五边只有
`.234/.491/.736/.788/.546`。最终 readout 的 centered recovery 也只有
`.696/.801/.684/.855/.777`。也就是说，coherent Parent/Current query trajectories 在某些
aggregation 边界近似交换，并不意味着一个固定层 correction 注入后会在后续 nonlinear
gate/residual 中保持充分。

S4/readout 的跨 reader 加减还必须克制解释。当前六个模型是同一 lineage 连续训练，神经元坐标有自然
对应，因此数值诊断有意义；对于独立训练或不同 backbone，Parent/Current head/hidden basis 没有天然
身份映射。最终 task score 才是严格架构中立的共同坐标。故本轮不能把 raw S4 commutator 直接提升为
通用 migration interface。

## 4. 为什么不能导出 ≤20% 的 persistent Design

这个恒等式没有提供新的 Current information source：

1. (F(P,C)-F(P,P)) 的第一项仍需要完整 Exact Current upper-layer K/V；生成它已经等价于解决原问题，
   不可能因为换成 Parent reader 就落入 0–20% constructor 区间。
2. correction 是 (q_P)-conditioned。按公式在线执行需要每个 candidate 再跑一条 Parent reader 并在
   score 端相加，违反 user-level persistent migration object 与“无 per-candidate score mixing”边界。
3. 若在 release time 用少数 probe 编译该函数，方法就退化为已审计的 response fitting/mapping；若保存
   K/V response operator，又回到 signed/native-response state，而其 Current side 仍缺合法 constructor。
4. 用 rank-8 等 approximate Current cache 替换 Exact Current，只是“generic Current compression +
   Parent control variate”的组合；它没有新信息，也没有理由越过已更强、更便宜的 single-r8 control。

因此本轮的正确裁决是：**保留“adjacent release 的 producer-state effect 对 reader update 近似一阶
不变”作为 oracle observation；拒绝 commuted endpoint 作为 Design 1。** 不扩 32/512 用户，不读取
confirmation，也不围绕 edge-specific scale、centered correction 或 probe 调参。

若未来出现一种无需 Current Exact、能直接产生该 finite state effect 的新的 causal source interface，
本观察可以成为其理论依据；在此之前，它不能单独回答 migration constructor。

## 5. 实现与验证

- `scripts/insight_two/producer_reader_commutator.py`：四角 coherent trace、exact mixed finite difference；
- `scripts/insight_two/run_producer_reader_commutator_preflight.py`：固定 UID/five-edge runner 与 score/S4
  adjudication；
- `tests/test_insight_two_producer_reader_commutator.py`：native cross-cache equivalence、finite-difference
  identity、zero-commutator 与输入拒绝。

Focused verification：`4 passed`；相关文件通过 `ruff check` 与 `py_compile`。运行未写 formal result、
contract 或 seal。
