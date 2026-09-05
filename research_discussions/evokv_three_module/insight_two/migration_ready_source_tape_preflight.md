# Migration-ready source tape：exact-Parent finite-release execution 审计

日期：2026-09-03  
状态：**NO-GO；只做 model-only/静态审计，未读取 UID、未运行 GPU、不是 Insight 2 或 Design 1**

## 1. 裁决先行

本轮把 `parent_anchored_delta_scan` 的 source-interface 障碍向前推进一步：允许 Parent cache 在原始生成时
额外保存 Transformer execution cut，使 release constructor 不再从 K/V 反解 hidden/Q/gate，也不再近似
重放 Parent absolute trajectory，而只传播有限版本缺陷

\[
D_{l+1}=D_l+
\left\{F_l^C(X_l^P+D_l)-F_l^P(X_l^P)\right\}.
\]

这条 recurrence 有一个真正的正结果：它不是 `KV_P -> KV_C` mapping，也不是一阶 JVP；保留全部参数秩、
state rank 和原生 attention 时，可按层归纳恢复 finite Current computation graph。Parent trajectory 是精确
control variate，Current Exact state、candidate 和 label 都不进入 constructor。

但严格系统裁决仍是 **NO-GO**：

1. 可逆信息与低成本执行不是一回事。generic nonzero-gate 的代数 cut 已需 `21` 个 `N x H` field；避免
   gate 除法和病态 output inverse 的稳定 fixed-layout cut 需 `26` 个 field，即现有 Parent K/V 之外
   `19.5 MiB/user`、`+216.7%` persistent state。
2. 即使把 projection、normalization、defect truncation、gate、sidecar 和全部 I/O 都免费，前五个 block
   只形成一次 exact Current historical attention response 的 `QK + AV` 已需
   `2,015,232,000 FLOPs/user = 42.2367% Exact-All`。它在实际 finite-defect 工作开始计全之前已超过
   `20%` 两倍。
3. 唯一看似低于 `20%` 的算术窗口必须把 native attention 换成 legacy `ELU+1` positive-affine moments，
   再叠加 `rank@90 Delta W` 与 rank-8 state truncation。已有 full-Exact all-history affine oracle 已出现
   灾难性反例；query-dependent activation geometry 不能删除。因此这不是被本轮 exact invariant支持的
   可执行路径，也不是架构中立的 Transformer 设计。
4. strongest legal control 已用 `17.8953%` 达到 UID1930 五边 mean recovery `.937`，只需读现有
   `9 MiB` Parent K/V 并写约 `0.102 MiB` sidecar。stable tape 路线在尚未产生任何 quality 结果时就需
   至少读 `28.5 MiB` source state，且改变 cache producer contract。

因此按事前规则停止：**不做 UID1930 semantic prototype，不用 quality 数字掩盖 exact compute 与 source
I/O 的决定性失败。** 这条路线可作为 Discussion 中的 model--system boundary：若未来模型原生提供可结合
的 exact attention state，或预算/状态格式改变，可重新打开；当前 Medium v0--v5/KV-only 主线不接纳。

## 2. 科学对象：finite defect，不是两条 absolute replay

令 Parent block 的 pre-state、normalized state 和 residual update 分别为
`X_l^P,N_l^P,U_l^P`。Current state 写成 `X_l^C=X_l^P+D_l`。线性 primitive 的有限差分为

\[
\Delta Y
=D_NW_C^\top+N_P(W_C-W_P)^\top,
\]

其中第一项已经包含 `D_N Delta W^top`，所以不是删除二阶项的一阶 tangent。residual boundary 为

\[
D_{l+1}=D_l+U_l^C-U_l^P.
\]

对 attention 和 gate 也直接计算 endpoint difference，而不在 `theta_P` 做 Taylor 展开。若所有中间
defect 不截断且使用 native attention，则

```text
D_0 = X_0^C - X_0^P
X_l^P + D_l = X_l^C
finite primitive differences = Current primitive - Parent primitive
D_{l+1} = X_{l+1}^C - X_{l+1}^P
```

逐层成立。这是本路线唯一可称为 scientific invariant 的部分。低秩 parameter factor、randomized range
finder 和 tape 都只是执行组件，不能各自作为创新 claim。

## 3. 哪些 source coordinate 必须穿过 release boundary

当前 legacy block 为

\[
N=\operatorname{RMSNorm}(X),\quad
(Q,K,V)=NW_{Q,K,V}^\top,
\]

\[
A=\phi(QK^\top)V,\quad O=AW_O^\top,\quad
G=NW_G^\top,\quad U=O\odot\operatorname{SiLU}(G),\quad X'=X+U.
\]

已有 K/V 不足以在低成本下补出下列 cut：

| coordinate | 为什么 finite defect 需要它 | 若不保存 |
| --- | --- | --- |
| `X_0^P` 与前五层 `U_l^P` | 重建所有 exact Parent pre-block residual；直接给出 residual subtraction | 六个 K/V checkpoint decode，或重新跑 Parent |
| historical `Q_l^P` | 计算 key drift 对原 Parent query 的 response；K/V 不决定低成本 Q | 五个 full-width dense transforms；既有 joint-KV→Q map rank@90 为 69--75 |
| gate preactivation `G_l^P` | `SiLU(G+Delta G)-SiLU(G)` 依赖非线性 base coordinate | `U=O*SiLU(G)` 对 `(O,G)` 非单射 |
| pre-output attention `A_l^P` | output-weight drift 项 `A_P Delta W_O^T` | 从 `O_P` 反解；真实 `W_O` condition number 为 `970--37,149` |
| post-output attention `O_l^P` | gate drift 项 `O_P Delta SiLU(G)` | 由 `U/SiLU(G)` 恢复；gate=0 时不可识别、近零时不稳定 |
| existing `K_l^P,V_l^P` | attention base source 与最终 Parent-plus-defect storage | 失去当前 serving cache 本体 |

### 3.1 21-field 只是 generic algebraic minimum

若假定每个 gate element 均严格非零，可以存：

```text
X0                                1 field
U0..U4                            5 fields
Q0..Q4, G0..G4, A0..A4           15 fields
                                  ---------
                                  21 N x H fields
```

然后由 `O=U/SiLU(G)` 恢复 Parent post-output coordinate。这个表示在实数域 generic point 上信息充分，
但 gate=0 时 `U=0` 与任意 `O` 相容；给 gate 加 epsilon 或 clamp 会改变 finite computation。可增加
zero-exception bitmap 和对应 O values，但其最坏大小仍是完整 O。

### 3.2 26-field 才是稳定 fixed-layout floor

不依赖用户数据稀疏性、也不做近零 division 的固定接口必须再存五个 `O_l^P`：

```text
21-field algebraic cut + O0..O4 = 26 N x H fields.
```

`A` 与 `O` 不能稳定合并成一个 field。只存 `O` 时，直接 output-weight defect 需要
`O(W_O^{P,T})^{-1}Delta W_O^T`；六个 checkpoint 的前五层 `W_O` condition number 实测最小
`970.43`、中位 `1755.60`、最大 `37149.16`。只存 `A` 时又必须做五次 dense Parent output projection
才能得到 gate 所需的 `O`。所以稳定 cost audit 使用 26-field，而不是把 generic gate division 当免费接口。

## 4. Persistent bytes 与 release I/O

Medium 为 `N=1024,H=192,L=6`，checkpoint-native cache dtype 是 FP32。

| object | scalars/user | bytes/user | 相对 Parent K/V |
| --- | ---: | ---: | ---: |
| existing Parent K/V | 2,359,296 | 9.00 MiB | 100% |
| 21-field algebraic tape | 4,128,768 | 15.75 MiB | +175.0% |
| 26-field stable tape | 5,111,808 | 19.50 MiB | +216.7% |
| Parent K/V + stable tape | 7,471,104 | 28.50 MiB | 316.7% total |

在 30,000 users 上，stable tape 单独增加 `571.29 GiB`，K/V 加 tape 共 `834.96 GiB`。这是 cache
生成时的持续容量，不是一次性 constructor workspace。

若 practical path 以 state rank `s=8`、逐矩阵 `rank@90` 参数 defect 原样保存 K/V factors，五条 edge
的 sidecar 为 `0.673/0.714/0.751/0.724/0.705 MiB/user`。即使不计 raw history、model weights 与
temporary workspace，worst-edge release I/O 至少为：

```text
read Parent K/V + stable tape      28.500 MiB
write factorized K/V defect         0.751 MiB
                                      --------
                                     29.251 MiB/user
```

作为硬对照，single Current-r8 shared-`U0` 读取同一 `9 MiB` Parent K/V，写
`26,624 FP32 scalars = 0.102 MiB` sidecar；不需要提前改变 Parent cache producer。tape path 的 minimum
state I/O 约为其 `3.21x`。

FP16/BF16 tape 可以把字节减半，但也会把“exact Parent anchor”改成量化 anchor；在没有单独误差协议与
实验前，不能用它美化本轮 fixed-FP32 contract。

## 5. Native exact FLOP floor：attention 单项已为 42.24%

前五层必须形成后续 K/V 所依赖的 historical Current update；第六层只需终止 K/V。对 causal inclusive
history，单层有效 pair 数为

\[
P=N(N+1)/2=524,800.
\]

即使 Parent `A/O/U` 全部已在 tape 中，任意 native dense attention backend 至少还需形成 approximate
Current score 与 weighted value：

\[
C_{one\ layer}=2PH\;(QK)+2PH\;(AV)=4PH=403,046,400.
\]

五层合计

\[
C_{attention}=5\cdot4PH
=2,015,232,000
=42.2367\%\;C_{Exact-All}.
\]

这个 floor 特意不计：

- Current input defect；
- 六层 RMSNorm 与 defect compression；
- `Delta Q/K/V/G/O` 的 parameter/state terms；
- activation、mask、output projection、gate、residual；
- final K/V factors、sidecar build/read/write；
- tape I/O 与 kernel inefficiency。

所以 source tape 删除了 Parent replay，却没有删除 arbitrary Transformer attention 对 Current endpoint 的
二次历史 interaction。它不是一个可以靠 kernel 常数从 `42.24%` 压到 `20%` 的边缘失败。

## 6. 为什么 affine/rank envelope 不能挽救裁决

五条 checkpoint edge 的 model-only audit，只计前五层 `Q/K/V/O/G` 加末层 `K/V` 共 27 个 matrix，
Frobenius energy rank sum 为：

| edge | sum rank@90 | sum rank@95 | sum rank@99 |
| --- | ---: | ---: | ---: |
| v0->v1 | 121 | 191 | 466 |
| v1->v2 | 150 | 232 | 567 |
| v2->v3 | 161 | 243 | 582 |
| v3->v4 | 159 | 244 | 579 |
| v4->v5 | 146 | 228 | 544 |

这说明 direct parameter term 确实有低秩数值结构；worst-edge `rank@90` factor apply/reconstruction 为
`126,615,552 = 2.6537% Exact-All`。但它没有降低 native attention floor，也没有证明 nonlinear
state defect 保持 rank 8。

若进一步做三项未经本轮 invariant保证的替换：

1. 所有 matrix 用 `rank@90`，state 固定 rank 8；
2. 每个 nonlinear boundary 用 power-0 randomized truncation；
3. native activation 换成 legacy all-positive affine prefix moments；

则可以写出一个约 `14.52%` 的**不完整乐观 subtotal**：input defect `0.97%`、16 次 truncation
`3.37%`、projection defect `4.77%`、五层 paired affine build/read `5.42%`。它仍遗漏 pointwise、factor
bookkeeping、sidecar 与 activation-region certificate。

更重要的是，仓库已经用 full Current Exact 构造过 probe-free all-history affine oracle，其五边 recovery
为 `-.057/.733/-13.072/.920/-1.907`。失败来自 query-dependent negative/region geometry，而不是
constructor rank。因此这个 subtotal只证明“删除关键语义后算术可以变便宜”，不能授权 prototype，更
不能把 HSTU affine shortcut写成 Transformer-general design。

## 7. 与 single-C8 的硬比较

同一 nonformal UID1930/odd-32/five-edge，strongest generic control 为：

```text
single Current-r8 shared-layer0 K/V splice
recovery = .861/.917/.985/.947/.975, mean=.937
cost     = 853,836,992 FLOPs/user = 17.8953% Exact-All
sidecar  = 26,624 FP32 scalars = 0.102 MiB
```

source tape path 当前没有合法 `<20%` native executor、没有 quality 数值、persistent source 增长
`216.7%`、minimum state I/O 约为 control 的 `3.21x`。即使将来某个 legacy affine prototype达到 80%，
也仍需证明它的 scientific mechanism 与激活区域闭包，而不能仅凭“不是 mapper”越过这张对照表。

## 8. Contract 边界与最终结论

这条路线明确**超出当前 KV-only contract**：

- tape 必须在 Parent cache 原始生成时 instrument；
- 已存在的 legacy K/V 没有这些 coordinates；
- 对 v0--v5 重新跑 Parent Full 可以模拟 tape，却不能在部署成本中假装该历史工作已免费发生；
- 它是 prospective model--system co-design，不是当前 cache 的迁移算法。

source execution cut、activation checkpointing、low-rank weight delta 和 numerical truncation本身都不是足够
的新贡献。只有发现“某个 Transformer-native exact/controlled response state使 finite defect 在全历史
attention 后仍可结合且低于 20%”才可能重开。当前 observation 恰好相反：query-dependent activation
geometry 不可删除，native response floor 又远超 cap。

最终裁决：

> **exact-Parent-anchored finite-release recurrence 是正确且非 mapping 的科学语义；但当前 Transformer
> source coordinates 不能把它变成 `<20%` 的 native executor。migration-ready tape 以超过两倍 K/V 的
> persistent state 换掉 Parent replay，仍没有消除 Current attention 主成本，并被 single-C8 硬对照支配。
> 本路线停止，不运行 UID，不接纳为 Insight 2/Design 1。**

实现与复核：

- `scripts/insight_two/migration_ready_source_tape.py`；
- `tests/test_insight_two_migration_ready_source_tape.py`。

```bash
PYTHONPATH=src:scripts pytest -q \
  tests/test_insight_two_migration_ready_source_tape.py
ruff check scripts/insight_two/migration_ready_source_tape.py \
  tests/test_insight_two_migration_ready_source_tape.py
```

