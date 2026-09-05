# Parent-anchored delta scan：source-state 与严格成本预检

日期：2026-09-03  
状态：**接口信息充分；`0--20%` 成本门失败；未运行 UID/GPU，不是 Insight 2 或 Design 1**

## 1. 裁决摘要

本轮检验一个比 paired reduced replay 更直接的机制：不近似重放 Parent/Current 两条完整历史路径，而把
现有 exact Parent per-layer K/V 当作 Transformer trajectory checkpoint，只沿这些 checkpoint 传播
`Parent -> Current` 的 finite-release response/residual defect。它不是 `KV_P -> KV_C` mapping；在
full-rank、native-attention 极限下，本应逐 primitive 恢复 Current computation graph。

预检得到一个正结论和一个决定性负结论：

1. **信息上可逆。** legacy HSTU 的 joint K/V 加每 token/layer 一个 RMS denominator，确实足以无学习
   地恢复 exact Parent pre-block residual checkpoint。Medium 只需 `6,144` 个 scalar，即现有 Parent
   K/V scalar 数的 `0.2604%`。
2. **计算上不可行。** 稳定 joint decoder 单独已需
   `905,969,664 FLOPs/user = 18.9880% Exact-All`。更乐观但病态的 K-only inverse，再加前五个 active
   block 无法从 K/V 消除的 historical query 与 HSTU gate 两组 dense coordinates，下界已为
   `1,207,959,552 = 25.3173%`。稳定 joint 版本对应 `34.8113%`。两者都还完全没有计算 Current input/
   state defect、attention response difference、output projection、nonlinearity、compression 或 sidecar。

因此，**“exact Parent K/V + 极小 RMS metadata”不能支持本论文 20% cap 下的 parent-anchored delta
scan**。本轮不进入 UID 1930 semantic prototype，也不启动 GPU；这是预先成本门的正常停止，而不是
质量失败后改口。若以后允许更重 source tape 或模型—系统协同投影，这个机制可重新打开，但那不再是
当前 pretrained Medium/KV-only Design 1。

## 2. 候选机制及其论文意义

令第 `l` 层 Parent/Current pre-block state 为 `X_l^P,X_l^C`，finite-release defect 为

\[
D_l=X_l^C-X_l^P.
\]

理想 recurrence 不是预测 Current KV，而是沿 exact Parent checkpoint 计算

\[
D_{l+1}=D_l+
\{F_l^C(X_l^P+D_l)-F_l^P(X_l^P)\}.
\]

其中 `F_l` 包含 RMSNorm、query/key/value response、output projection、gate 和 residual update。
Parent block update可由相邻 exact checkpoint 的差得到；Current-minus-Parent attention response 则可在
legacy `ELU+1` 的固定 positive region 内写成 causal prefix moments。若 induced defect 在 token axis
保持 rank `s`，`delta-Q/K/V` 还可先与 Parent K/V moment 做 contraction，再沿全历史 scan，不必选择
“重要 token”。

这条路线若成立，科学对象会是 **Parent trajectory 上的 release-delta execution**：Parent exact state
作为 control variate，constructor 只推进版本缺陷，full-rank/no-truncation 有 computation-graph exact
limit。它在概念上不是 mapper、sampled replay 或两份低秩 cache 的简单相减，具有足够清楚的论文命题。
本预检否定的是当前 source interface 下的执行窗口，不是否定这个问题本身。

## 3. joint K/V + RMS scalar 的可逆性

legacy block 使用 RMSNorm：

\[
N_l=\Gamma_l X_l^P/\rho_l,
\qquad
\rho_l=\sqrt{\operatorname{mean}((X_l^P)^2)+\epsilon}.
\]

PyTorch projection convention 给出

\[
K_l=N_lW_{K,l}^{P\top},\qquad
V_l=N_lW_{V,l}^{P\top}.
\]

记

\[
B_l=[W_{K,l}^{P\top},W_{V,l}^{P\top}]\in\mathbb R^{H\times2H},
\qquad Z_l=[K_l,V_l].
\]

只要 `B_l` 满 row rank，

\[
N_l=Z_lB_l^+,
\qquad
X_l^P=(N_l\oslash\Gamma_l)\rho_l.
\]

`B_l^+` 只依赖 Parent model，可每个 release/layer 编译一次；`rho_l` 是每 token 一个 scalar。对
`L=6,N=1024,H=192`：

| source object | scalars/user | 相对 Parent K/V |
| --- | ---: | ---: |
| existing Parent K/V | 2,359,296 | 100% |
| RMS denominators | 6,144 | 0.2604% |

对六个 frozen Medium checkpoints 的全部 `36` 个 block 做 model-only audit：joint matrix 均为 algebraic
rank `192`，condition number 范围 `15.92--35.23`、中位数 `26.42`。作为对照，single-K condition
number 为 `684.72--6968.12`，single-V 为 `788.71--33334.80`。固定随机 normalized states 经 checkpoint
FP32 projection、再用 FP64 decoder 还原时，joint relative error 的中位数/最大值为
`3.11e-7/3.78e-7`；K-only 为 `6.00e-6/2.42e-5`。因此 joint K/V 的信息充分性是可靠正结果，不能继续
把 RMS scale ambiguity写成原则性障碍。

CPU exact-limit test 还直接验证了 synthetic Parent state 经 RMSNorm、K/V projection、joint decoder 和
stored denominator 后恢复原 pre-block state，误差在 FP64 `1e-10` tolerance 内。

## 4. 为什么 response moments 没有消除 dense source coordinates

在固定 positive region，单 head 的 Parent response 为

\[
R_i^P=B_i^P+q_i^PM_i^P,
\quad
B_i^P=\sum_{j\le i}v_j^P,
\quad
M_i^P=\sum_{j\le i}k_j^{P\top}v_j^P.
\]

finite difference 包含

\[
\Delta R_i=Delta B_i+Delta q_iM_i^P+q_i^P\Delta M_i+Delta q_i\Delta M_i.
\]

若 `Delta Q/K/V` 有 token rank `s`，部分项确实可降为 `O(Nsd)` associative scans。例如
`Delta q_i=sum_a A_{ia}b_a^q` 时，

\[
\sum_{j\le i}(\Delta q_i k_j^{P\top})v_j^P
=\sum_aA_{ia}\sum_{j\le i}(b_a^qk_j^{P\top})v_j^P.
\]

但这不消除两个 base-dependent coordinate：

1. `q_i^P Delta M_i` 中尤其是 Parent attention weights 对 `Delta V` 的作用，需要 historical Parent
   query，或等价的 full-rank query coordinate；
2. HSTU update
   `O_l * SiLU(G_l)` 的 finite difference 需要 Parent/Current gate coordinate。只知道相邻 checkpoint
   给出的 product `X_{l+1}^P-X_l^P`，不能唯一分解出 `O_l^P` 与 `G_l^P`。

从 joint K/V 到 Q/G 的 model-only exact maps 在上述 36 个 block 中全部 algebraic rank `192`。其
Frobenius `rank@90` 仍分别为 Q `69--75`、G `72--76`，`rank@99` 为 Q `129--133`、G `131--135`。
所以它们不是一个可用 rank-4/8 exact shortcut。截断这些 maps 会重新变成 generic projection
compression，而且没有 finite-release exact limit；它不能被包装成本机制。

negative ELU branch、activation-region crossing 和 output projection还会增加工作。本节即便把它们全部
免费处理，下面的 source-coordinate floor 也已经失败。

## 5. 严格 FLOP 下界

沿用 repository 口径：multiply-add 为 2 FLOPs，Medium
`N=1024,H=192,L=6`，Exact-All denominator 为 `4,771,282,944`。一个全历史 square transform 为

\[
C_H=2NH^2=75,497,472.
\]

六层 K/V 必须恢复六个 normalized/residual checkpoint；只有前五层需要执行 attention/gated residual，
最后一层可在 K/V 后停止。

| optimistic mandatory component | FLOPs/user | Exact-All |
| --- | ---: | ---: |
| K-only checkpoint decode, `6 C_H` | 452,984,832 | 9.4940% |
| stable joint checkpoint decode, `12 C_H` | 905,969,664 | 18.9880% |
| historical Q + gate, `2*5 C_H` | 754,974,720 | 15.8233% |
| **K-only optimistic subtotal** | **1,207,959,552** | **25.3173%** |
| **stable-joint optimistic subtotal** | **1,660,944,384** | **34.8113%** |

`K-only` 是刻意给候选的最有利下界：假定一个 condition number 高达约 `7e3` 的 square inverse仍完全
可接受。`stable-joint` 才对应本轮提出的 joint K/V checkpoint。两条 subtotal 均遗漏：

- Current input/state defect与各层 normalization；
- delta K/V/Q/G parameter/state terms；
- ELU+1 mask/sign、causal prefix build/read；
- response output projection、SiLU、Hadamard、residual；
- defect compression、functional sidecar build/read；
- 所有 metadata/KV I/O 和真实 kernel inefficiency。

因此这不是“实现还可继续优化”的 20% 附近估计，而是关键计算开始前已经越界的 lower bound。

## 6. strongest single-arm control

同一 UID 1930/odd-32/five-edge 的 legal single-arm Current rank-8 shared-layer0 splice，已有
probability-gap recovery：

```text
.8610 / .9173 / .9852 / .9473 / .9753, mean=.937
```

其 matched terminal-KV ledger 为 `934,810,304 = 19.5924%`；把初始 range finder换成已验证的
matrix-free 等价 executor 后为 `853,836,992 = 17.8953%`。它仍只是 xKV-adjacent generic control，
不是论文主方法；但任何新机制至少不能在**尚未做 attention delta**时就比它更贵。parent-anchored scan
当前最乐观 floor 已为 `25.3173%`，故在 compute 上被 control 严格支配，不值得用 UID quality 数值掩盖。

## 7. 重新打开路线需要什么

只有以下接口变化之一能删除本轮 floor：

1. source cache 在 Parent 生产时额外持久化 pre-block hidden、historical Q 和 gate/output tape；
2. 模型训练时约束 Q/G 可从 persistent coordinates 低成本精确形成；
3. 放宽 20% cap，允许一次 dense source decode 与 query/gate replay。

第一项不是“小 metadata”：六层 hidden 为 `1,179,648` scalars（Parent K/V 的 `50%`），前五层 Q+G
再为 `1,966,080` scalars（`83.33%`）。第二项属于新的 model-system co-design，需要重新训练，不适用于
当前 frozen v0..v5。第三项直接违反本轮 Design 1 admission。

所以当前裁决是：**保留“Parent trajectory 上传播 finite-release defect”作为有科学意义但 source-state
受限的否证；停止 KV+RMS parent-anchored scan，不把 checkpoint 可逆性包装成可执行 Design。**

实现与复核：

- `scripts/insight_two/parent_anchored_delta_scan.py`；
- `tests/test_insight_two_parent_anchored_delta_scan.py`。

~~~bash
PYTHONPATH=src:scripts pytest -q \
  tests/test_insight_two_parent_anchored_delta_scan.py
ruff check scripts/insight_two/parent_anchored_delta_scan.py \
  tests/test_insight_two_parent_anchored_delta_scan.py
~~~
