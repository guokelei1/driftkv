# Paired release replay 的 matrix-free input preflight

日期：2026-09-03
状态：**解析成本通过；CPU 代数测试通过；未运行用户/GPU，未建立 formal contract**

## 1. 结论先行

paired finite-release trajectory 的最后一层 KV-only 版本原成本为
`1,041,218,120 FLOPs/user = 21.8226% Exact-All`，因而不能进入 `0--20%` frontier。这里找到的
省算不是删掉一条 Parent trajectory，也不是降低 rank、oversampling 或 power iteration，而是把两臂
**语义完全相同的初始 range finder 改写成 matrix-free operator evaluation**。

对每条 release arm，HSTU 历史输入为

\[
X=(E_{item}+E_{behavior}+\Phi W_t^\top)W_{in}^\top.
\]

固定 `target rank=4, sketch width=8, power=1` 的 range finder 只需要 `X@R` 和
`X^T@Q`，不要求先生成 temporal projection `Phi W_t^T` 或最终 `X`。逐项计入两次正向 operator、
两次转置 operator、两次 thin QR 和 small Gram/eigh/rotation 后，每臂 input-plus-initial-factor 成本从
`101,441,366` 降为 `18,033,494 FLOPs`。替换双臂相同部分后：

\[
C_{paired,mf}=874,402,376
=18.3264\%\;C_{Exact}.
\]

它比 20% cap 低 `79,854,212.8 FLOPs/user`，即 `1.6736` 个 Exact-All 百分点。因此，**在沿用此前
factor-aware triangular block、fused post-block sketch、最后一层 KV-only 等全部审计前提时，paired
候选第一次得到一个不改数值协议且严格低于 20% 的解析 executor**。

这不表示 Design 1 已经冻结。paired 的机制、population recovery、append persistence、真实 GPU kernel
和端到端 bytes 尚未通过 formal gate。本组件也不单独 claim novelty；它只是使“matched Parent/Current
trajectory error cancellation”这个研究机制进入可执行预算的系统实现。

## 2. 为什么这是等价改写，不是新的近似

记

\[
A=E_{item}+E_{behavior},\qquad
X=(A+\Phi W_t^\top)W_{in}^\top.
\]

对任意右操作数 `R`：

\[
XR=A(W_{in}^\top R)
  +\Phi\{W_t^\top(W_{in}^\top R)\}.
\]

对任意左操作数 `Q`：

\[
X^\top Q=W_{in}\{A^\top Q+W_t(\Phi^\top Q)\}.
\]

两式只是结合律，没有更换 sketch、basis、rank 或模型权重。冻结的 `power=1` range finder 仍执行：

~~~text
Y0 = X Omega                 # operator apply 1
Q0 = thin_qr(Y0)
T  = X^T Q0                 # operator apply 2
Y1 = X T                    # operator apply 3
Q1 = thin_qr(Y1)
B  = Q1^T X = (X^T Q1)^T   # operator apply 4
small Gram/eigh + rotate Q1,B to target rank 4
~~~

因此不能把它误写成 one-pass range finder，也不能漏掉最后一次 `X^T Q1`。CPU tests 同时比较了
`X@R`、batched `X@R`、`X^T@Q` 与 dense `embed_inputs` 的结果，并比较了 `power=0/1` 两种完整
factor reconstruction；当前 float32 容差内数值等价，**不是 bitwise equality**。结合律改变会带来最后
几位舍入差异，QR/SVD 的列符号在数学上也不唯一。测试还通过 monkeypatch 禁止调用
`temporal_enc.forward` 和 `in_proj.forward`，确认 matrix-free 路径没有暗中回退到两个 dense
`N x H` 中间量。

semantic-equivalence 路径使用与现有 dense prototype 相同的 `torch.linalg.svd(small_core)`。解析账本
对应可执行的 `small_core @ small_core^T -> symmetric eigh -> two rotations`，代码也实现并测试了
`gram_eigh` 分支；它与 small SVD 给出同一个 target-rank projection，但在重复/极接近奇异值时允许
不同的正交坐标和浮点舍入。正式 executor 必须在 contract 中冻结其中一种，而不能在观察 quality 后
切换。

## 3. 单臂逐项 FLOP ledger

沿用 Medium：`N=1024,H=192,F=16,target a=4,oversample=4,s=8,power=1`；multiply-add
计 `2 FLOPs`。item/behavior lookup 与超越函数单列，不藏进 matmul numerator。

### 3.1 一次 `X@R`

| 项 | 公式 | FLOPs |
| --- | ---: | ---: |
| `W_in^T R` | `2*H^2*s` | 589,824 |
| `A @ (...)` | `2*N*H*s` | 3,145,728 |
| `W_t^T (...)` | `2*(2F)*H*s` | 98,304 |
| `Phi @ (...)` | `2*N*(2F)*s` | 524,288 |
| 两个结果相加 | `N*s` | 8,192 |
| **一次 right apply** |  | **4,366,336** |

`power=1` 需要两次，合计 `8,732,672`。

### 3.2 一次 `X^T@Q`

| 项 | 公式 | FLOPs |
| --- | ---: | ---: |
| `A^T @ Q` | `2*N*H*s` | 3,145,728 |
| `Phi^T @ Q` | `2*N*(2F)*s` | 524,288 |
| `W_t @ (...)` | `2*H*(2F)*s` | 98,304 |
| base/time core 相加 | `H*s` | 1,536 |
| `W_in @ (...)` | `2*H^2*s` | 589,824 |
| **一次 transpose apply** |  | **4,359,680** |

`power=1` 的 power step 与 final core 各需要一次，合计 `8,719,360`。

### 3.3 形成 feature 与数值线性代数

| 项 | 公式 | FLOPs/arm |
| --- | ---: | ---: |
| item + behavior | `N*H` | 196,608 |
| temporal phase multiply | `N*F` | 16,384 |
| two right applies | above | 8,732,672 |
| two transpose applies | above | 8,719,360 |
| two thin QR | `2*ceil(2Ns^2-2s^3/3)` | 261,462 |
| Gram + `9s^3` eigh + two rotations | `2Hs^2+9s^3+2Nsa+2asH` | 107,008 |
| **per-arm total** |  | **18,033,494** |
| **two-arm total** |  | **36,066,988** |

这里没有把 `A` 视为免费：每臂仍读取并相加 item/behavior embedding。省掉的是原账本中每臂
`12,582,912` 的 dense temporal projection、`75,497,472` 的 dense `in_proj` 和随后对已经形成的
`X` 做四次 dense sketch application 的重复工作。matrix-free 四次 application 本身反而比
`4 * (2NHs)` 更贵；总节省来自避免先对全部 `H` 个输出通道做两次 dense projection。

## 4. 非 FLOP 工作与 I/O 单列

每臂还有：

- `2NF = 32,768` 次 sin/cos evaluation；paired 共 `65,536` 次；
- literal per-user 构造还会做 `F=16` 次 frequency exp；paired 为 `32` 次；该频率只依赖
  `(F,max_period)`，生产 executor 可在每个 model/release 加载时预缓存，但这里不把它静默删除；
- `H*s = 1,536` 个固定 Gaussian draws；paired 共 `3,072` 个；
- `2NH = 393,216` 个 embedding payload scalars 的 lookup，FP32 cold-read 为 `1.5 MiB/arm`；
- raw item ID、behavior ID、time delta 共 `3N = 3,072` 个 history fields；
- transient `A` 为 `NH = 196,608` scalars，`Phi` 为 `2NF = 32,768` scalars。

模型 weight bytes 沿用 Exact 与原 paired ledger 的 resident-weight 口径，不按每次小 GEMM 重复计入
per-user bytes。真实 kernel 若不能复用 resident weights，必须在 runtime roofline 中重新报告。

这个执行仍保留一个 `N x H` 的 `A=item+behavior`，所以它不是“完全不物化 history”。准确说法是：
**不物化 temporal `N x H` 和 post-`in_proj` `N x H`；embedding lookup/base 与 `N x 2F` phase
feature 仍然存在。**

## 5. paired KV-only 总账重算

Exact-All 分母保持：

\[
C_{Exact}=4,771,282,944,
\qquad 0.2C_{Exact}=954,256,588.8.
\]

旧 KV-only paired 总账中，two raw inputs 与 two initial compressions 分别为
`176,979,968` 和 `25,902,764`。其余项完全不变：

| component | FLOPs/user |
| --- | ---: |
| two matrix-free input + initial factors | 36,066,988 |
| first five block bodies, two arms | 702,062,080 |
| first five fused recompressions, two arms | 132,821,340 |
| final-layer norm + K/V only, two arms | 1,287,680 |
| factor-aware layer-0 `U0` builder | 1,247,808 |
| upper-five signed-core builds | 916,480 |
| **total** | **874,402,376** |
| **Exact-All fraction** | **18.3264%** |

等价地：

\[
1,041,218,120
-2(88,489,984+12,951,382)
+2(18,033,494)
=874,402,376.
\]

margin 为 `79,854,212.8 FLOPs/user`。此前 block ledger 已把 logit scale、ELU+1 arithmetic、SiLU
arithmetic、Hadamard、factor RMS、QR 和 small eig/rotation 明列；exp/reciprocal/rsqrt 与本节 sin/cos
继续按仓库统一口径单独报告。不能把这个 `18.3264%` 外推为 wall-time 比例。

## 6. 研究身份与下一门槛

matrix-free input 是经典 operator-form randomized range finding 在 HSTU feature formation 上的直接
代数实现。它没有发现新的 migration object，因此论文中的合理身份是 **paired finite-release
compiler 的 executor component**，不是 Insight 2 或独立 Design point。

它的重要作用是排除一个工程性 blocker：paired 的科学假说不再因为输入端重复物化而必然超过 20%。
接下来仍必须先做 prospective 32-user mechanism canary，并在同次运行中比较 paired、matched-compute
single-arm、independent seed 和 static Parent-cache controls。只有 approximation-error cancellation、
functional recovery、五边稳定性与合法 append protocol 都通过，才能把 paired recurrence 提升为
Design 1。当前不因为解析成本通过而读取 confirmation users 或启动 GPU。

实现与复核：

- `scripts/insight_two/matrix_free_input_range.py`；
- `tests/test_insight_two_matrix_free_input_range.py`。

~~~bash
PYTHONPATH=src:scripts pytest -q \
  tests/test_insight_two_matrix_free_input_range.py
ruff check scripts/insight_two/matrix_free_input_range.py \
  tests/test_insight_two_matrix_free_input_range.py
~~~
