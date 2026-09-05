# Mode-space Current rematerialization：理论成本与新颖性 preflight

日期：2026-09-02  
状态：**成本/语义否证审计；未运行 GPU，不是已接纳 Design 1**

> 2026-09-03 后续更新：本文的 dense-input ledger 与 novelty 审计继续有效，但“`21.8226%` 是最低
> 算术成本”的结论已由 [matrix-free input preflight](matrix_free_paired_input_preflight.md) 取代。
> 在不改变 rank、seed、power 或 paired trajectory 语义的情况下，合法 KV-only 构造为
> `18.3264%`。该更新只解除成本阻塞；paired 仍未超过 single-arm control，因而未获 Design 准入。

## 1. 裁决先行

对 Medium `N=1024,H=192,L=6,heads=6,d=32,r=8` 的结论是：

1. **one-arm compressed-Current replacement A/C** 是 generic low-rank/xKV-adjacent control，
   不是最终迁移机制。冻结的 `rank=8, oversample=4, power=1` executor 为 `21.7025%`。
   只有删除 oversampling/power iteration、采用最激进 `q=0` 和 fused sketch 时才出现
   `19.6641%` 的窄窗口；不能用这条 post-hoc 更弱配置替换冻结点。
2. **当前真正候选 D** 对 Parent/Current 各做一条独立的 `rank=4, oversample=4, power=1`
   replay；两臂只共享 Gaussian seed/rules，不假装共享 data-dependent basis。再从两臂 approximate
   layer-0 `Delta[K,V]` 构造 `rank=8, oversample=4, power=0` 的固定 `U0`，把六层 signed cores
   加到 exact Parent cache。完整六 block ledger 是 `1,206,907,124 FLOPs/user = 25.2952%`。
3. 即使做一个合法但尚未在当前代码实现的 KV-only specialization——第六层只做 norm 和 K/V
   projections，不计算不再被消费的 Q/attention/gate/out——D 仍需
   `1,041,218,120 FLOPs/user = 21.8226%`。相对 20% 硬上限的**最小算术超额**仍为
   `86,961,531 FLOPs/user`，即 `1.8226` 个 Exact-All 百分点。
4. D 的现有单 UID 五 edge recovery 为 `0.870/0.901/0.985/0.869/0.706`：四条超过 `0.80`，
   但最后一条未过，且单 UID 不能替代 population evidence。质量信号值得保留；它不抵销成本失败。
5. 强迫 Parent/Current 两臂共享同一 rank-8 basis 的旧 B 下界为 `39.3090%`–`41.2550%`，且
   arm-specific RMSNorm 本身就会破坏该共享假设；D 正是为了移除这条不成立的前提。
6. 存储确实小，但不能挽救 compute 和 novelty。单臂替代式 factorized KV 只是 dense Current
   KV 的 `2.8646%`；固定 `U0` 的 signed sidecar 更只有 `1.1285%`。
7. **现在的 `mode_space_replay.py` 是 dense semantic prototype，不是上述低成本 executor。**
   它调用 full SVD、materialize 稠密状态并走 native dense block；不得用本文的 prospective
   kernel 成本为当前代码背书。

更重要的是，“每层把 token-by-feature activation 做低秩分解，在 factor 上执行 attention，再用
randomized range finding 重压缩”本身属于已有低秩推理/KV 压缩的技术范畴。它不是
mapper，但也不因此自动具有论文级创新性。

## 2. 审计对象和不可混淆的三种语义

### A. one-arm Current compressed replacement

只运行一条 Current 轨迹，将每层历史状态保持为

\[
X_l \approx C_lB_l,
\qquad C_l\in\mathbb R^{N\times r},\quad B_l\in\mathbb R^{r\times H}.
\]

最后持久化 approximate Current K/V factors，并让 reader 直接读它们。这是 compressed
rematerialization，不是“保留 Parent base 只加 migration sidecar”。

### B. Parent/Current shared-quotient 双臂 delta

两臂分别执行 Parent 和 Current，并假设每层能维持同一 token/history basis：

\[
X_l^P\approx C_lB_l^P,\qquad X_l^C\approx C_lB_l^C.
\]

然后持久化 `Parent + C_l(Delta R_K,Delta R_V)`。这才是对称的 coupled differential
replay，但 shared basis 必须被证明，不能从单臂 low-rank 反推。

### C. 单臂 Current replay + 固定 `U0` signed cores（generic/xKV-adjacent control）

先用 Current reduced replay 的 approximate layer-0 K/V 与 exact Parent layer-0 cache 形成 defect：

\[
\widehat D_0=[\widehat K_0^C-K_0^P\mid
\widehat V_0^C-V_0^P]\in\mathbb R^{N\times 2H}.
\]

用固定、label-free range finder 取 `U0 in R^{N x r}`。随后只运行一条 Current reduced
replay，但对每层写入

\[
E_{K,l}=U_0^\top(\widehat K_l^C-K_l^P),\qquad
E_{V,l}=U_0^\top(\widehat V_l^C-V_l^P).
\]

reader 消费

\[
K_l^{mig}=K_l^P+U_0E_{K,l},\qquad
V_l^{mig}=V_l^P+U_0E_{V,l}.
\]

C 不读 Current Exact upper-layer cache；但它需要 full-history Parent projections 来形成各层 signed
cores，也没有先验证明一个单臂 Current basis 能承载版本差异。它与已有 single-arm low-rank KV
compression 最接近，因此只保留为 control。

### D. 当前候选：independent low-rank two-arm release differential

Parent 与 Current 使用相同的、预注册的 numerical scheme 和 Gaussian seed，但分别从自己的 trajectory
形成 basis：

\[
\widehat X_l^P=C_l^PB_l^P,\qquad
\widehat X_l^C=C_l^CB_l^C,
\]

其中两臂均为 `target rank=4, oversample=4, power=1`。`C_l^P` 与 `C_l^C` **不要求相等**。
从 approximate layer-0 differential

\[
\widehat D_0=[\widehat K_0^C-\widehat K_0^P\mid
\widehat V_0^C-\widehat V_0^P]
\]

构造 `U0`（target rank 8、oversample 4、power 0），随后写入

\[
E_{K,l}=U_0^\top(\widehat K_l^C-\widehat K_l^P),\qquad
E_{V,l}=U_0^\top(\widehat V_l^C-\widehat V_l^P),
\]

并仍以 exact Parent cache 为 control variate：

\[
K_l^{mig}=K_l^P+U_0E_{K,l},\qquad
V_l^{mig}=V_l^P+U_0E_{V,l}.
\]

因为每条 approximate arm 的 K/V 共享一个 rank-4 left factor，两臂差分的 token rank 至多为 8；
layer-0 rank-8 `U0` 因而不是从任意 rank grid 调出的容量。但是“layer-0 differential span 能跨六层
保持功能充分”仍然只是待验证的 release-law hypothesis。

## 3. factorized legacy HSTU 执行代数

本审计只对应现有 Medium legacy block：`ELU+1`、无 relative-position bias、
`inner=H=heads*d`。它不是 softmax Transformer 或 faithful SiLU-HSTU 的直接成本。

### 3.1 RMSNorm 可以保持 factor，但不是免费

对 `X=CB`：

\[
G=BB^\top,\qquad
\rho_i=\operatorname{rsqrt}(c_iGc_i^\top/H+\epsilon).
\]

因此 normalized state 可写成

\[
\bar X=(\operatorname{diag}(\rho)C)(B\operatorname{diag}(w)).
\]

不需要为 norm 物化 `N x H`，但需要 `B B^T`、每行二次型、row scaling 和
`N` 次 `rsqrt`。

### 3.2 dense Current weights 不截断

权重仍是原生 Current `H x H` 矩阵。对 `Q/K/V/gate` 的每一个投影，只把
`B` 乘过权重：

\[
R_Q=\bar B W_Q,\quad R_K=\bar B W_K,\quad
R_V=\bar B W_V,\quad R_G=\bar B W_G.
\]

四个投影的总成本是 `8rH^2`，不是 `8NH^2`。这是 activation factorization，
不是 low-rank weight approximation。

### 3.3 ELU+1 之后仍然是全因果对

对 head `a`：

\[
S_a=R_{Q,a}R_{K,a}^\top,\qquad
L_a=(\bar C S_a)\bar C^\top,
\]

\[
A_{a,ij}=\mathbf 1[j\le i]\{\operatorname{ELU}(sL_{a,ij})+1\}.
\]

`S_a` 只是 `r x r`，但 pointwise ELU 不保持 rank。因此至少必须生成

\[
heads\cdot P,\qquad P=N(N+1)/2
\]

个 causal activation。低秩 Q/K 只把每个 pair 的 dot width 从 `d` 降为 `r`，没有把
`N^2` 变成 `N`。

V 一侧可以严格地保留 factor：

\[
Z_a=A_a\bar C,\qquad O_a=Z_aR_{V,a}.
\]

multi-head concatenate 后 token rank 最多是 `heads*r=48`。将其通过 dense `W_O` 时，
block-diagonal V basis 在每行只有 `d` 个非零元，所以成本是
`2*heads*r*d*H = 2rH^2`，而不是错误的 `2*(heads*r)*H^2`。但下一步 gate
要求一个 dense `N x H` attention output。

### 3.4 gate 是真正的 dense boundary

\[
U_l=O_l\odot\operatorname{SiLU}(\bar C R_G),\qquad
X_{l+1}=C_lB_l+U_l.
\]

SiLU 不能在 `C/B` 上分离执行。因此每层必须至少形成：

- dense gate preactivation，`N x H`；
- dense attention output，`N x H`；
- dense Hadamard result `U_l`，`N x H`。

可以不单独物化最后的 `X_{l+1}`，但只能通过把 residual factor 的贡献显式加到
range-finder 的每次 matrix application 中实现；这些乘法和加法已在成本里计入。

## 4. 固定 label-free 重压缩：不允许免费 SVD

对一个 dense `Z in R^{N x H}`，冻结一个不读 label/请求/分数的 Rademacher
`Omega in R^{H x r}` 及 seed。两个可执行版本是：

**`q=0` two-pass range finder**

~~~text
Y = Z Omega
Q = thin_qr(Y)
B = Q^T Z
return (Q, B)
~~~

\[
C_{RF,0}=4NHr+C_{QR}=6,422,187,
\]

**stabilized `q=1` subspace iteration**

~~~text
Y0 = Z Omega
Q0 = thin_qr(Y0)
T  = Z^T Q0
Y1 = Z T
Q1 = thin_qr(Y1)
B  = Q1^T Z
return (Q1, B)
~~~

\[
C_{RF,1}=8NHr+2C_{QR}=12,844,374,
\]

其中

\[
C_{QR}=\left\lceil2Nr^2-\frac23r^3\right\rceil=130,731.
\]

这两者都是近似；`q=0,r=8`没有 oversampling 或 power iteration，没有稳定的近似质量保证。
它只因为很便宜而出现在 optimistic floor，不得根据 quality 结果事后把 `q=1` 换成
`q=0`。这类 randomized range finder 本身也是经典数值线性代数，不是新机制
([Halko, Martinsson and Tropp, 2011](https://doi.org/10.1137/090771806))。

对 `Z=C_res B_res+U_dense` 做 fused `q=0/q=1` 时，不物化 residual sum，但每次
`Z`/`Z^T` application 都显式执行 dense `U` arm 和 factor residual arm。结果为：

\[
C_{post,0}=4NHr+4r^2(N+H)+r(N+H)+C_{QR}=6,743,211,
\]

\[
C_{post,1}=8NHr+8r^2(N+H)+2r(N+H)+2C_{QR}=13,486,422.
\]

### 4.1 冻结配置中的 oversampling 不能只算 rank 维

D 的两臂 replay 使用 target rank `a=4`、oversampling `p=4`，所以 sketch width 是
`s=a+p=8`；single-arm C control 使用 target rank `r=8,p=4`，所以 `s=12`。所有 range-finder
GEMM 和 QR 必须按 `s` 计，不能按最终 target rank 计完以后免费截断。

一个可执行的 target-rank 截断不使用 full `N x H` SVD。对 range finder 得到的
`Q in R^{N x s}, B=Q^T Z in R^{s x H}`：

~~~text
G = B B^T
(E, lambda) = symmetric_eigh(G)
C_new = Q E[:, :r]
B_new = E[:, :r]^T B
~~~

本审计给 small `s x s` symmetric eigensolver 一个显式的保守估计 `9s^3`，并计入 Gram、两次
factor rotation。于是

\[
C_{truncate}(H;s,r)=2Hs^2+9s^3+2Nsr+2rsH.
\]

对 D 的每条 `a=4,s=8,power=1` replay：

\[
C_{QR}(N,8)=130,731,
\]

\[
C_{init}=8NHs+2C_{QR}+C_{truncate}=12,951,382,
\]

\[
C_{post}=8NHs+8as(N+H)+2s(N+H)+2C_{QR}+C_{truncate}
=13,282,134.
\]

对 single-arm C 的 `r=8,s=12,power=1`，对应数值为：

\[
C_{init}=19,766,208,\qquad C_{post}=20,729,280.
\]

所以前文 `r=s=8` 的 `q=0/q=1` 数字只是无 oversampling sensitivity floor，不是冻结 D/C
配置的正式成本。

## 5. Exact-All 分母和逐项 FLOP 审计

仓库固定 multiply-add=`2 FLOPs` 口径下：

\[
P=524,800,
\]

\[
C_{Exact}=2N(2F)H+2NH^2
+L\{2N(5H^2)+4PH\}
=4,771,282,944.
\]

`20%` 硬上限是 `954,256,588.8 FLOPs/user`。这个 Exact 分母本身没有定价
norm/超越函数；本文仍把新方法的 scalar arithmetic 列出，并把 exp/reciprocal/rsqrt
单独报告，不用“不是 matmul FLOP”把它们隐藏。

### 5.1 输入与初始 factor

| component | formula | FLOPs/user |
| --- | ---: | ---: |
| temporal dense projection | `2*N*(2F)*H` | 12,582,912 |
| dense `in_proj` | `2*N*H^2` | 75,497,472 |
| item/behavior/time additions | `2*N*H` | 393,216 |
| temporal phase multiplies | `N*F` | 16,384 |
| initial `q=0` range finder | above | 6,422,187 |
| initial stabilized `q=1` | above | 12,844,374 |

另有 `2NF=32,768` 次 sin/cos evaluation，不并入 matmul FLOP 分子。真实输入会先物化
`N x H`；初始 `C/B` 绝不能假定为免费已存在。

### 5.2 每层（ideal factor-aware triangular kernel）

| component | formula | FLOPs/layer |
| --- | ---: | ---: |
| factor RMS arithmetic | explicit Gram/quadratic/scale | 183,808 |
| Q/K/V/gate basis projections | `8*r*H^2` | 2,359,296 |
| head cores + `C*S` | `2*h*r^2*d + 2*h*N*r^2` | 811,008 |
| native causal logits | `2*h*P*r` | 50,380,800 |
| logit scaling | `h*P` | 3,148,800 |
| ELU+1 arithmetic-equivalent | `h*P` | 3,148,800 |
| factorized `A@V`: first `A@C` | `2*h*P*r` | 50,380,800 |
| sparse head basis through `W_O` | `2*h*r*d*H` | 589,824 |
| attention output materialization | `2*N*(h*r)*H` | 18,874,368 |
| gate materialization | `2*N*r*H` | 3,145,728 |
| SiLU arithmetic (exp/reciprocal separate) | `3*N*H` | 589,824 |
| gate Hadamard | `N*H` | 196,608 |
| fused residual + stabilized `q=1` compression | above | 13,486,422 |
| **per-layer total, `q=1`** |  | **147,296,086** |

上表的 ELU 项只用一个明示 arithmetic-equivalent 保持可复算表格；真正的非线性工作量是
每层 `3,148,800` 个 ELU+1 element（negative branch 需 exp）、`196,608` 个 SiLU
element（exp 与 reciprocal）、`1,024` 个 rsqrt。

### 5.3 A 的总成本

| A executor | FLOPs/user | Exact-All | 裁决 |
| --- | ---: | ---: | --- |
| **frozen C control: `r=8,s=12,power=1`, fused** | **1,035,489,856** | **21.7025%** | **FAIL** |
| `q=1`, fused residual/sketch | 985,110,874 | 20.6467% | FAIL |
| `q=1`, materialize residual then compress | 1,001,312,602 | 20.9862% | FAIL |
| `q=0`, materialize residual then compress | 956,357,293 | 20.0440% | FAIL |
| `q=0`, fused residual/sketch | 938,229,421 | 19.6641% | 只是 optimistic arithmetic window |

后四行均为 `r=s=8` 的 sensitivity，而不是 oversample-4 冻结配置。最后一行只剩
`16,027,168` FLOPs（`0.3359% Exact`）margin。它依赖最弱的无-power/无
oversampling compression 和特制 fused kernel。若仅给每个 exp/reciprocal/rsqrt 记一个操作当量，
最坏 ELU negative-branch 计数会使其成为 `20.1096%`。因此它不是可以宣称的
strict end-to-end `<=20%` 结果，更不是当前代码的成本。

## 6. 不可隐藏的 `N^2` 和 `N x H` 对象

### 6.1 causal attention workspace

ideal triangular execution 每层仍需 `hP=3,148,800` 个 logits/activation scalars：

- FP16/BF16：`6,297,600 bytes`，约 6.01 MiB/layer；
- FP32：`12,595,200 bytes`，约 12.01 MiB/layer。

若 logits 和 activated `A` 不能 in-place/fused，workspace 再翻倍。可以用 streaming triangular
kernel 减少峰值内存，但不能减掉 `Theta(hPr)` 的 logits 和 `A@C` 算术。

当前 PyTorch `_aggregate` 先计算 full `h*N^2=6,291,456` 个 logits，再乘 causal mask。
即使新写 factor-aware 但仍使用 dense `N^2` kernel，QK 和 `A@C` 六层也会相对
triangular 版额外增加 `603,389,952` FLOPs；再加 future-position scale/activation 和 mask
work 只会更高。字面复用当前 `_aggregate` 还会先物化 full Q/K/V 并在 head width
`d=32` 上做 matmul，不等于本文的 rank-8 kernel。

### 6.2 nonlinear block workspace

以 FP32 计：

- 一个 `N x H` buffer 是 `196,608 scalars = 786,432 bytes`；
- `heads*r=48` 的 attention left factor 是 `49,152 scalars = 196,608 bytes`；
- input，gate，attention output 和 Hadamard result 在不同 kernel 融合策略下可能同时存在。

本文的 fused floor 只避免另外物化 `residual + update`，没有避免 gate/attention/Hadamard
的 dense boundary。range finder 要多次扫描该 boundary；若不保留 dense update，就必须每次重新
生成它，计算只会更贵。

## 7. Parent + delta 的三种诚实成本

### 7.1 D：independent rank-4 two-arm differential（当前候选）

令每条 arm 的 state target rank 为 `a=4`、sketch width 为 `s=8`。每条 arm 的完整账本是：

| one arm component | FLOPs/user |
| --- | ---: |
| raw model-specific input formation | 88,489,984 |
| initial `a=4,s=8,power=1` compression | 12,951,382 |
| six factorized block bodies | 421,237,248 |
| six fused post-block recompressions | 79,692,804 |
| **one arm total** | **602,371,418** |

其中一个完整 block body 的逐项成本是：

| component, `a=4` | FLOPs/layer |
| --- | ---: |
| factor RMS | 54,016 |
| factor Q/K/V/gate projections | 1,179,648 |
| head cores + `C*S` | 202,752 |
| causal logits | 25,190,400 |
| logit scale | 3,148,800 |
| ELU+1 arithmetic-equivalent | 3,148,800 |
| factorized `A@C` | 25,190,400 |
| sparse V/head basis through `W_O` | 294,912 |
| rank-24 attention materialization | 9,437,184 |
| gate materialization | 1,572,864 |
| SiLU arithmetic | 589,824 |
| Hadamard | 196,608 |
| **block body** | **70,206,208** |
| fused residual + `a=4,s=8,power=1` compression | **13,282,134** |

Parent raw input 必须以 Parent embedding/temporal/in-projection 重算；现有 Parent K/V cache 不包含
block-input hidden state。两臂使用同一个 Gaussian seed 只保证 numerical protocol 对称，不能共享任何
GEMM 或把 Parent arm 当作已存在。

layer-0 两臂 K/V 各自共享一个 rank-4 left factor，因此 joint differential 的 token rank 至多为 8。
对这个 factorized difference 做 `U0 target=8,s=12,power=0` 不必物化 `N x 2H`，但仍需显式执行：

| `U0` builder | FLOPs/user |
| --- | ---: |
| apply two-arm factor difference to Gaussian sketch | 282,624 |
| apply its transpose to `Q` | 274,944 |
| thin QR, `N x 12` | 293,760 |
| small Gram/eigh + truncate/rotate to rank 8 | 396,480 |
| **total** | **1,247,808** |

range finder 已经给出 layer-0 `U0^T Delta[K,V]`。对 upper five layers，每层从两个独立
rank-4 factors 形成 signed cores：

\[
2\{2N(8)(4)+4(8)(4)H\}+2(8)H=183,296\;FLOPs/layer.
\]

因此完整 D 为：

| D component | FLOPs/user |
| --- | ---: |
| two raw inputs | 176,979,968 |
| two initial compressions | 25,902,764 |
| 12 arm-layer block bodies | 842,474,496 |
| 12 arm-layer recompressions | 159,385,608 |
| layer-0 `U0` builder | 1,247,808 |
| five upper-layer signed-core builds | 916,480 |
| **total** | **1,206,907,124** |
| **Exact-All fraction** | **25.2952%** |

相对 `954,256,588.8` 的 20% cap，完整 D 超出 `252,650,535.2 FLOPs/user`，即 `5.2952`
个百分点。

一个合法的最小算术优化是：迁移只需六层 K/V，不需要最后一层 block output。因此前五层走完整
block/recompression，第六层每条 arm 只执行 factor RMS 与 K/V basis projections：

\[
C_{last,KV}=54,016+4aH^2=643,840\;FLOPs/arm.
\]

这一 KV-only specialization 的总成本仍为：

\[
C_{D,KV-only}=1,041,218,120
=21.8226\%\;Exact-All.
\]

它相对 20% cap 的**最小超额**为 `86,961,531.2 FLOPs/user`，即 `1.8226` 个百分点。该数字已经
乐观地假设 direct K/V basis projections、factor-aware triangular kernel、fused residual sketch，且不
给 exp/reciprocal/rsqrt 赋额外 FLOP 权重；因此不能再向下解释为测量误差。

D 的单 UID 五 edge recovery `0.870/0.901/0.985/0.869/0.706` 表明 release differential 可能比
single-arm compression 更有结构，但只够保留一个 scientific lead。它既不是 population result，也没有
通过五边 `>=0.8`，更不能用 recovery 调低已经冻结的两臂成本。

### 7.2 B：真双臂 shared-quotient 的旧乐观下界

假设 Parent/Current 两臂的 rank-8 token basis 可以免费共享，所有 arm-specific attention/gate
仍必须执行两次。对 `[X^P|X^C] in R^{N x 2H}` 做一次 joint compression，
最乐观成本为：

| B lower bound | FLOPs/user | Exact-All |
| --- | ---: | ---: |
| joint `q=0` | 1,875,543,725 | 39.3090% |
| joint stabilized `q=1` | 1,968,391,514 | 41.2550% |

这还不是一个完整 executable。即使 block-input 共享 `C_l`，两臂 RMSNorm 会形成

\[
\operatorname{diag}(\rho_l^P)C_l,
\qquad
\operatorname{diag}(\rho_l^C)C_l,
\]

它们一般不共享同一 column space。要么再做一次 joint normalized-state compression，要么把
delta rank 扩到最多 `2r`，要么强制共享 row scale。最后一种是新的未证明近似，不能
写成 algebraic identity。无论选哪个，都不会把 39% 降到 20%。

### 7.3 C：single-arm rank-8 control 与更便宜的 `U0` 变体

冻结的 C control 先做一条 `r=8,s=12,power=1` Current replay，成本 `1,035,489,856`。再从其
approximate layer-0 K/V 与 exact Parent layer-0 cache 构造 `U0`；factor-aware builder 为
`20,122,176` FLOPs。layer 0 core 由 builder 直接返回，upper five layers 的 Current-factor/Parent-dense
projection 共 `32,373,760` FLOPs。因此：

\[
C_C=1,087,985,792=22.8028\%\;Exact-All.
\]

它没有旧版 exact-Current-layer0 seed 的额外 full K/V projections，但仍不过预算，而且主体就是一条
single-arm compressed Current replay。它保留为 generic/xKV-adjacent control，不作为 D 的 novelty。

更便宜的 seed 仍不够，且功能语义不等价：

| C basis source | total FLOPs | Exact-All | 额外前提 |
| --- | ---: | ---: | --- |
| approximate replay `Delta[K,V]` | 1,087,985,792 | 22.8028% | frozen C control |
| approximate replay `Delta K` only | 1,081,683,328 | 22.6707% | layer-0 V core 仍需另算 |
| exact post-`in_proj` input defect | 1,173,060,224 | 24.5858% | 必须重算 Parent raw input |
| input defect, **Parent input 假定免费已持久化** | 1,084,570,240 | 22.7312% | 现有 state 不提供 |

K-only `U0` 不包含 value-defect geometry；input-defect `U0` 位于 layer-0 K/V projection 之前。它们
不能在看到五边 recovery 后替换 joint `Delta[K,V]`。特别是 Parent post-`in_proj` history 不在现有
persistent state 中，不能用 Parent cache 的存在把第二次 raw-input computation 写成零。

### 7.4 因果命名与 append 边界

whole-history token-mode projection 不是逐 token causal state transition。对一个 dense history `X`：

\[
\widehat X=QQ^\top X,
\qquad
\widehat x_i=\sum_{j=1}^{N}(QQ^\top)_{ij}x_j.
\]

一般存在 `j>i` 的非零系数，所以 cutover 历史中的早期 row 会混入同一 cutover 之前、但时间上更晚的
历史事件。这里没有读取 release 之后的 event、future request 或 label；对 cutover 后的 recommendation
query，全部 `N` 个事件都已经是过去。因此它仍是合法的 **release-time whole-history functional
compiler**。但它不等于 Current Transformer 的逐位置 causal rematerialization，也不能宣称保持了原生
prefix semantics。

append 时的唯一低成本定义是冻结旧 `U0`，把新 event 对应的 basis row 设为零，并由 Current reader
为新 event 写 native K/V。这样不会让新 event 反向改变旧 rows，但“旧 whole-history correction + new
native rows”是否仍保持功能充分不是代数定理，必须由 rolling Gate C 检查。若每次 append 后重算
whole-history basis，则所有旧 row/core 都可能改变，时间持久性和 release-time-only 成本同时失效。

因此论文中允许的名称是 **whole-history release differential compiler**；在没有新的 triangular/online
mode construction 及其成本以前，不得称它为 causal Current rematerialization。

## 8. 持久化字节与 reader 成本

### 8.1 A：factorized Current replacement

同层 K/V 可共享 normalized left factor `C_l`：

\[
S_A/layer=Nr+2rH=11,264\;scalars,
\]

\[
S_A=67,584\;scalars.
\]

| dtype | factorized Current KV | dense Current KV | ratio |
| --- | ---: | ---: | ---: |
| FP16/BF16 | 135,168 bytes | 4,718,592 bytes | 2.8646% |
| FP32 | 270,336 bytes | 9,437,184 bytes | 2.8646% |

若现有 cache API 为 K/V 各复制一份 left factor，FP32 是 `466,944 bytes (4.9479%)`。

对一个 future query、一层，factorized prefix attention 的 QK+AV 成本是

\[
heads\{4Nr+4rd\}=202,752\;FLOPs,
\]

为 dense K/V 对应 `4NH=786,432` 的 `25.7813%`。这不包含两者共有的 one-token
QKV/gate/out 和 activation。

### 8.2 C/D：fixed-`U0` signed sidecar

`U0` 只存一份，每层存 K/V signed cores：

\[
S_C=Nr+2LrH=26,624\;scalars.
\]

| dtype | incremental sidecar | full Current KV ratio |
| --- | ---: | ---: |
| FP16/BF16 | 53,248 bytes | 1.1285% |
| FP32 | 106,496 bytes | 1.1285% |

Parent base 仍然存在，不能把它从 serving I/O 删掉。C 的 release compiler 还会对六层 exact Parent
K/V 做一次完整投影读取，即 `4,718,592 bytes` (FP16) 或 `9,437,184 bytes` (FP32)。D 用 raw history
重放 approximate Parent arm 来形成 delta，不需要在 constructor 中扫描 exact Parent K/V，但 serving
reader 仍需读取完整 exact Parent base；这份 I/O 不能用小 sidecar 比例掩盖。

D construction 同时保留两个 rank-4 working states；每条 arm 的一个 state factor 是
`Na+aH=4,864 scalars`，两臂为 `9,728 scalars`。它们是 transient workspace，不计入 persistent
sidecar，也不能反过来被称为免费 Parent trajectory。

future reader 在原 Parent dense attention 上额外执行 factor delta read；每层约为
`202,752` factor FLOPs，再加 `6,336` 个 logit/response adds。即 prefix-attention 算术相对
Parent base 额外约 `26.6%`，而不是免费 injection。

## 9. 当前 prototype 的真实边界

`scripts/insight_two/mode_space_replay.py` 目前已经同时提供 exact-SVD diagnostic 与固定
range-finder semantic path，但两者都不是本审计假设的 fused production executor。exact-SVD 选项以
classical thin-SVD estimate：

\[
C_{svd}(N,H)=4NH^2+\frac83H^3=169,869,312,
\]

\[
C_{svd}(N,2H)=4N(2H)^2+\frac83(2H)^3=754,974,720.
\]

六层两类 SVD 合计约 `5,549,064,192 FLOPs`，已经是 Exact-All 的 `116.30%`；再加 dense Current
forward 后，该 exact-SVD semantic path 约为 `216.30% Exact-All`。

fixed-range path 避免了 full SVD，但当前实现仍有以下真实 dense work：

- `randomized_token_factors()` 接收和多次扫描已经物化的 dense `N x H` state；
- `_factorized_legacy_attention_impl()` 物化 full `N x N` logits/activation，不是 triangular fused
  kernel；
- `_finish()` 当前先形成 dense head output，再调用原生 dense `out_proj`，尚未实现本文按 sparse
  head basis 计价的 `2*heads*r*d*H` 路径；
- gate/residual 先形成 dense state，下一轮 range finder 没有把 residual factor application 融合进去；
- single-arm approximate layer-0 defect helper 还会 materialize Current K/V 和 dense defect。

当前仓库尚没有 D 的 two-model factorized constructor。因而 D 的 `25.2952%`/`21.8226%` 是一个
逐算子 prospective ledger，不是现有 wall-clock path 的 FLOP 标签。

因此现有 full-rank toy equality test 只能证明语义，不能证明 rank-4/8 质量、`<=20%` 或
production executability。

## 10. 它是不是新方法？

目前 D **不是简单的 Parent-to-Current learned mapper**：两臂都由 raw history 经各自 Transformer
arithmetic 产生，没有用 matched Current-Exact upper-layer KV 回归 translator。但是“不是 mapper”
不等于“有新颖性”。

目前的组成部件都有很强的已有边界：

- randomized range finding 是经典低秩矩阵近似，不是 Transformer 发现；
- [Linformer](https://arxiv.org/abs/2006.04768) 已经以 sequence/token 方向的低秩投影近似
  attention；
- [Eigen Attention](https://arxiv.org/abs/2408.05646) 和
  [Palu](https://arxiv.org/abs/2407.21118) 已经在 low-rank space 中计算 attention/压缩 KV；
- [xKV](https://arxiv.org/abs/2503.18893) 已经利用跨层 dominant singular-vector alignment，把多层
  KV cache 合并进 shared low-rank subspace；所以“跨层共享 token/KV basis”本身不能作为本工作新意；
- 与本候选更直接重叠的 [PuzzleKV](https://arxiv.org/abs/2608.23843) 已经把每个 cache
  region 做 low-rank decomposition，并直接对 dense/factorized pages 执行 attention；
- [Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893) 已经明确研究跨模型
  cache handoff，只是它的核心是 fitted linear mapping，与本候选的 raw-history Current replay 不同。

因此，以下表述都不足以成为 Design contribution：

- “我们发现 history activation 可以低秩”；
- “我们把 QKV 在 rank-8 factor 上计算”；
- “我们每层用 SVD/random projection 重压缩”；
- “我们将 Parent cache 在同一 basis 上替换一部分”；
- “Parent/Current 各压缩一次，然后相减”。

这些合起来仍然像“为了迁移场景组合已有低秩压缩与 cache splice”。

## 11. 不沦为组合所必须增加的论文贡献

只有在下面这个**新的、可证伪的 Transformer release law**成立时，mode space 才可能成为
论文机制，而不是 compression baseline：

> 相邻 release 的全历史状态差异虽然是 token-support dense，但两个独立 reduced trajectories 的
> 差异在逐层 contextualization 中保持一个由 approximate layer-0 release differential 决定的
> history quotient；该
> quotient 保留 Current reader 可见的 release gain，而其正交补可以继承 Parent 而不传播
> 有害的 cross-version error。

这个 claim 需要同时有四件目前不存在的东西：

1. **现象**：由两条 independent rank-4 trajectories 形成的 layer-0 `U0`，对 upper-layer release
   differential 与 Current-reader functional response 仍因果充分，而不只是 tensor energy。当前单 UID
   五边 `0.870/0.901/0.985/0.869/0.706` 是线索，不是结论。
2. **闭包**：一个不读 Current Exact upper-layer cache的 paired quotient evolution law，并解释两臂
   basis 不同、RMSNorm/gate dense boundary 以后，为何 fixed `U0` 仍承载 functional differential。
   “两臂各自重压缩再相减”只是算法流程，不是闭包律。
3. **迁移特异性**：在 matched storage/compute 下，`Parent + signed quotient delta` 必须显著优于
   “压缩 Current replacement cache”和普通 low-rank KV compression。否则跨版本只是应用场景，不是
   方法贡献。
4. **区别于 xKV/PuzzleKV 的 release law**：matched compute/storage 下，D 必须优于对 Parent/Current
   分别做 ordinary low-rank cache compression 后的朴素差分，并证明 exact Parent control variate 而非
   shared SVD 本身带来收益。

系统上还需要真正的 factor-aware causal-triangular kernel 和 signed-delta reader。它们可以是系统
贡献，但单独的 kernel fusion 仍不是 Insight 2。

## 12. 最小可证伪 diagnostic（不授权 GPU launch）

成本已经使 D 无法进入 `<=20%` 主 frontier。若只为判断上述结构 claim 是否值得将来
改变 budget 或做 model-system co-design，最小 oracle 应为 `32 users x 5 edges`，且顺序固定：

1. 严格执行两条 independent `rank4/os4/power1` arm，以 approximate layer-0 differential 的
   `rank8/os4/power0 U0` 构造 D；不读任何 Current Exact upper-layer cache。
2. 报告 D 在 32-user/five-edge 上的 reader response、final representation 与 recommendation-gap
   recovery；单 UID `0.870/0.901/0.985/0.869/0.706` 只作为冻结前线索，不参与调 rank。
3. 在评价端加入 exact upper-layer differential 投影到同一 `U0` 的 oracle ceiling，并与每层
   independent best rank-8 oracle 对比。fixed `U0` 低而 independent 高时，跨层 quotient 被否证。
4. 做 matched storage/compute controls：single-arm C、ordinary per-layer/xKV-like compression、
   两臂 compressed-cache 朴素差分，以及 exact Parent control-variate D。必须隔离 signed release
   differential 的特异增益。
5. rolling Gate C 固定旧 `U0`、给 append row 填零并让新 event 写 native Current K/V；分别报告
   1/8/32 次 append 后的功能保持。不得通过每次重算全局 basis 隐藏时间不持久。
6. 不得在看过结果后用 K-only/input-defect basis 替换 joint `Delta[K,V]` primary；它们只能
   作事先分开的容量对照。

事先 stop rule：若 fixed `U0,r=8` 的 exact-oracle functional recovery 在五条 edge 中不能至少
四条达到 `0.80`，立即停止 mode-space release-law 路线。即使通过，D 仍然需要新的
`<=20%` constructor；质量证据不能抵销已审计的 FLOPs。

## 13. 最终定位

最准确的定位是：

> mode-space replay 是一个有用的 structural diagnostic 和 strong compression baseline，但目前不是
> 可接纳的 Design 1。single-arm C 是 `22.8028%` 的 xKV-adjacent control；真正候选 D 的完整成本
> 是 `25.2952%`，即使合法跳过最后一层未被消费的 block output，最小值仍为 `21.8226%`，超出预算
> `86,961,531 FLOPs/user`。D 的单 UID 结果支持继续保留“independent reduced trajectories 的
> release differential 可能共享跨层功能 quotient”这一科学线索，但若没有 population/rolling
> 证据和新的低于 20% closure，它仍是已有低秩推理/KV 压缩技术在跨版本场景中的组合。
