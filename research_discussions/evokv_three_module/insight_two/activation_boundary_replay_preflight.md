# Cross-version attention interaction boundary：严格 no-go

日期：2026-09-03  
状态：**UID 1930 / 五 edge 非正式机制预检完成；interaction-boundary route 未准入，不读取更多用户**

## 1. 裁决先行

本轮检查一个与 low-rank、mapper、probe 和 token selection 正交的信息源：Transformer 历史
self-attention interaction graph 在模型发布前后的**激活分支变化集合**。Medium legacy HSTU 的
attention 为 `ELU(z)+1`，所以 `z=0` 是真实计算边界：正分支为 `1+z`，负分支为 `exp(z)`。

实验得到一个有用但否定性的结论：

> 跨版本 attention graph 的拓扑大多保持不变，但主要功能误差并不集中在少数 crossing edges；大量
> pair 即使留在同一激活分支，连续变化的 query、key、value 仍会累积成主要 response defect。

Exact Parent/Current 历史 graph 在六层、五 edge 上平均 `87.32%` 的 causal pair 保持同分支，crossing
为 `12.68%`。然而只对 crossing pair 注入 Exact Current contribution 的 serving oracle，五边
probability-gap recovery 只有：

```text
.7778 / .0635 / .0342 / .0519 / .1276, mean=.2110
```

历史 response 层面也一致：crossing delta 单独只恢复平均 `12.55%` 的 response gap；unchanged-region
delta 的 norm 平均为 joint endpoint delta 的 `87.45%`。所以“graph change set 小”不能推出“只迁移
graph change set 即充分”。

一个完全 no-target 的递归 boundary replay 在同一 UID 上得到：

```text
.9546 / .9044 / .5213 / .9980 / .7865, mean=.8330
```

这个数值表明 Parent interaction 可以作为有信息的 trajectory control，但不能接纳为 Design：其最宽松
attention-only FLOP 下界为 `63.3550% Exact-All`。更决定性的是，即使假设 Parent graph 已免费持久化，
只发现 Current graph 的一次 causal QK contraction 已是 `21.1183% Exact-All`，尚未计算任何 AV、
projection、norm、gate、output 或 cache write。因此不存在满足当前 `<20%` 门的 exact boundary
constructor。

## 2. 固定协议

- frozen discovery 第一个 UID `1930`；
- Medium `v0->v1,...,v4->v5` 五条 edge；
- frozen held-out odd-32 candidate panel；
- 不读 label，不读 `[512,3000)` confirmation；
- 不训练，不扫 rank、threshold、layer、probe 或 token count；
- Exact Current history 只进入明确命名的 endpoint diagnostic 和 serving oracle；
- primary recursive replay 只接收 raw history、Parent persistent K/V 和 Current model；
- terminal layer 只生成 K/V，不执行对 cache 无用的历史 attention update；
- immutable dataset、population、candidate panel、prior evidence 和六个 checkpoint hash 全部重验；只记录
  living research-plan 的预期/当前 hash 差异，不修改 frozen contract。

运行固定在 `cuda:3`，五 edge 共 `19.19s`。结果保存在：

`results/yambda500m_medium_seed17/insight2_functional_boundary_v1/activation_boundary_replay_preflight/summary.json`

## 3. Exact interaction-graph decomposition

对 layer `l`、head `h` 的每个 causal pair `j<=i`，定义 endpoint logit

\[
z^P_{hij}=s\langle q^P_{hi},k^P_{hj}\rangle,
\qquad
z^C_{hij}=s\langle q^C_{hi},k^C_{hj}\rangle.
\]

若两者符号相同，pair 属于 `S`；否则属于 crossing set `X`。因为 `S` 与 `X` 是 causal graph 的严格
partition，attention response 的有限版本差满足：

\[
R^C-R^P=
\underbrace{\sum_{(i,j)\in S}
  \left[\phi(z^C_{ij})v^C_j-\phi(z^P_{ij})v^P_j\right]}_{\Delta R_S}
+
\underbrace{\sum_{(i,j)\in X}
  \left[\phi(z^C_{ij})v^C_j-\phi(z^P_{ij})v^P_j\right]}_{\Delta R_X}.
\]

实现对每层都验证该 identity，relative L2 error 约 `1e-7`。这里没有 Taylor 展开、SVD、拟合或采样。

五 edge 等权后的关键统计为：

| metric | mean |
| --- | ---: |
| endpoint activation-region agreement | .8732 |
| endpoint crossing fraction | .1268 |
| Current activation mass on crossings | .0403 |
| Current response norm on crossings / full | .0637 |
| activation-change L1 on crossings | .2182 |
| `||Delta R_X|| / ||Delta R||` | .3569 |
| `||Delta R_S|| / ||Delta R||` | .8745 |
| crossing-only response-gap recovery | .1255 |

两个 norm ratio 不要求相加为一，因为 `Delta R_S` 与 `Delta R_X` 是 signed vectors，会相消或相长。真正
具有因果含义的是 gap recovery：只修 crossing 后仍留下 `Delta R_S`，平均只消除 `12.55%` response
gap。

crossing 随深度由 layer 0 的 `6.31%` 上升到 layer 1--5 的约 `13.45%--14.40%`。这说明 release
perturbation 会沿 contextual trajectory 扩散；它没有坍缩成更稀疏的 upper-layer graph edit。

## 4. Serving causal intervention

对每个真实 Current recommendation query，在每层用同一个 evolving Current query 分别读取 Parent 与
Exact Current prefix，并按 matched-query sign partition 形成：

\[
R_{X\text{-only}}(q)=
\sum_{S(q)}\phi(qK^P)V^P+
\sum_{X(q)}\phi(qK^C)V^C.
\]

随后执行真实 Current output projection、gate、residual 和下一层 query。其逐边 recovery 为：

| method | v0→v1 | v1→v2 | v2→v3 | v3→v4 | v4→v5 | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crossing delta only | .778 | .063 | .034 | .052 | .128 | **.211** |
| unchanged delta only | -.072 | .910 | .970 | .961 | .853 | **.724** |

matched serving graph 的 crossing fraction 只有 `8.17%`，但 crossing-only 在 4/5 edge 几乎没有修复。
`v0->v1` 是反例：该 edge crossing 较大且 crossing-only 达 `.778`，unchanged-only 反而过冲；因此也
不能简单写成“所有 edge 都只需要 unchanged region”。稳定结论只能是：**interaction topology 不是
跨版本功能误差的充分坐标。**

## 5. No-target recursive boundary replay

为了排除“oracle 单层 partition 没有形成合法 trajectory”的解释，本轮实现了一个递归、无 Current
Exact target 的构造。Current raw-history trajectory 在每个 non-terminal layer 形成自己的
`q_hat,K_hat,V_hat`，再与 persistent Parent K/V 比较：

\[
\widehat R_l=
\sum_{S_l}\phi(\widehat q_lK_l^P)V_l^P+
\sum_{X_l}\phi(\widehat q_l\widehat K_l^C)\widehat V_l^C.
\]

`S_l/X_l` 完全由这两个可执行 arm 动态形成；`x_hat_{l+1}` 经 Current gate/residual 递归推进，最后
输出六层 K/V。它有以下语义：

- 不读取 Current Exact cache、hidden state 或 target response；
- full crossing 时回到完整 Current trajectory；
- Parent=Current 时 crossing 为空且精确回到原 cache；
- token-aligned causal mask 保持原始顺序，不做 token ranking 或 sampling。

其 matched-query crossing 平均只有 `6.76%`，五边 recovery 均值 `.8330`，3/5 edge 达到 `.80`，但
`v2->v3=.5213`、`v4->v5=.7865`。它既低于同 UID generic single Current-r8 control 的约 `.937`，也
无法满足成本门。

## 6. 为什么“crossing 稀疏”没有形成低成本 action

令 `N=1024,H=192,L_active=5`，每层每 head 的 causal pair 数为

\[
P=N(N+1)/2=524,800.
\]

一次全图 QK contraction 按 multiply-add=2 的冻结口径至少为

\[
2L_{active}PH=1,007,616,000\ \text{FLOPs}
=21.1183\%\ \text{Exact-All}.
\]

这已经假设 Parent graph、所有 projection、norm、gate、AV 和 writes 都免费。实际 no-target replay
至少要比较 `q_hat K^P` 与 `q_hat K_hat^C` 两张 graph，并做一次 mixed weighted-V reduction；同样宽松
的 floor 为：

\[
3,022,848,000\ \text{FLOPs}=63.3550\%\ \text{Exact-All}.
\]

把 Parent sign graph 预存也不能解决 Current graph 的发现成本。五个 active layer、六个 head 的 causal
bitmask 即使压到一 pair 一 bit，仍为 `1,968,000 bytes/user`，约为 FP32 Parent K/V 的 `20.85%`，
30,000 用户约 `55.0 GiB`；而 Current crossing set 随 release 和递归 trajectory 才产生，不能在 Parent
cache 创建时提前知道。

因此，按 crossing 数量支付 correction FLOPs 只在**预先免费知道 crossing ledger**时成立。生成这份
ledger 正是被忽略的 dependency closure；不能把 oracle sparsity 当成 executable sparsity。

## 7. 新颖性与泛化裁决

这条 route 还有两个独立的论文高度问题：

1. 若持久化 mask 并只运行 crossing edges，机制退化为 cross-version mask reuse / dynamic sparse
   attention；“mask 来自模型发布差异”本身不足以构成新的 functional migration principle。
2. 若用正分支 affine moments 跳过 same-region pairwise read，则重新回到已经审计的 legacy ELU cone
   moments；负分支和 query-dependent mask 仍需额外状态。继续加 kernel feature、rank 或 probe 会进入
  已有 approximation 组合，而不是新信息源。

更重要的是，`z=0` branch change 是 legacy ELU+1 特有边界；standard softmax Transformer 没有对应的
离散 activation graph。它可以作为 HSTU 实例中的机制 falsifier，不能承担架构中立 Insight 2。

**最终裁决：RETIRE as Design。** 不建立 32-user contract，不扩展用户，不调 crossing threshold，不做
稀疏 kernel。留下的科学边界是：

> 跨版本功能迁移必须表示 stable interaction region 内的连续 response deformation；仅迁移 attention
> graph 的拓扑 edit，即使 edit set 较小，也既不因果充分，也没有低成本 exact discovery closure。

这个结论支持继续寻找 aggregation/residual 上的功能差异，或改变 migration-ready source interface；
它不支持把 interaction mask、affine cone 或 sparse crossing replay 写成 Design 1。

## 8. 实现与验证

- `scripts/insight_two/activation_boundary_replay.py`：exact graph decomposition、serving causal oracle、
  no-target recursive replay 和静态成本下界；
- `scripts/insight_two/run_activation_boundary_replay_preflight.py`：固定 UID/five-edge runner；
- `tests/test_insight_two_activation_boundary_replay.py`：native cache equivalence、partition identity、
  Parent=Current exact limit 与 Medium cost invariant。

```bash
PYTHONPATH=src:scripts pytest -q \
  tests/test_insight_two_activation_boundary_replay.py
PYTHONPATH=src:scripts python \
  scripts/insight_two/run_activation_boundary_replay_preflight.py \
  --device cuda:3
```

Focused tests：`4 passed`；三份 Python 文件 `ruff check` / `ruff format` 通过。
