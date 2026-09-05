# Insight 2 candidate：用户 attention cone 中的跨版本 response moments

日期：2026-09-02  
状态：**结构性新假设；只有单 UID 预检，尚未通过 population canary，不能写成论文结论**

## 1. 从两个失败中得到的新问题

固定 S4 vector 有很高同请求 ceiling，却不能跨请求持久；chronological 与 layer-0 address K/V coreset
也都在正式 canary 失败。这说明紧凑对象既不是广播 correction，也不是“选少量代表 token”。真正需要
解释的是：为什么 1,024 个历史 K/V 经过 Current reader 后会出现接近低秩的用户级 response。

当前 Medium HSTU 使用 pointwise、unnormalized `ELU(qK)+1` attention。对一层一 head：

~~~text
R(q) = sum_i [ELU(s q k_i) + 1] v_i.
~~~

它不是全局 linear attention；activation 作用在 dot product 之后。但在一个固定的 qK 正负符号域内，
正半空间 `P` 的 response 有精确恒等式：

~~~text
R_positive(q)
  = sum_{i in P} (s q k_i + 1) v_i
  = B_P + s q M_P,

B_P = sum_{i in P} v_i,
M_P = sum_{i in P} k_i outer v_i.
~~~

负半空间 response 为 `sum exp(s q k_i) v_i`。如果同一用户的多个 recommendation candidates 形成
稳定 query cone，且大量负 logit 已饱和，那么完整历史在该 query family 下就不是 1,024 个独立 K/V，
而是一个由 `B` 和 `M` 表示的用户级 response operator。

## 2. 待证伪 Insight

候选 Insight 2 现在收紧为：

> Recommendation queries for the same user occupy a stable attention cone. Inside that cone,
> HSTU's distributed cross-version KV error collapses exactly into signed response moments;
> only cone crossings and the exponentially suppressed negative branch remain non-affine.

它同时包含推荐与 Transformer 的结构：

- 推荐侧：同一历史被多个候选和后续请求反复读取，query 不是任意全空间点；
- Transformer 侧：query–key activation 的分支决定了历史 response 的函数形式；
- 跨版本侧：Current 与 Parent 可以拥有不同 positive set，因此迁移的是两套 moments 的 signed
  difference，而不是把 Parent K/V 映射成 Current K/V。

32 个 label-free anchor candidates 只用于定义每个 user/layer/head 的 majority cone；held-out 32 个
candidates 检验 cone sharing。Current 与 Parent 的 majority-positive mask 分开构造，避免把真正的
cross-version sign crossing 错当成噪声。

## 3. 新的 persistent object

对每层、每 head，持久化：

~~~text
Delta B = B_Current - B_Parent
Delta M = M_Current - M_Parent.
~~~

serving query 在 attention aggregate、gate/residual 之前读取：

~~~text
R_hat_Current(q)
  = R_Parent_complete(q) + Delta B + s q Delta M.
~~~

Medium 的 `L=6,H=6,D=32` 下，`Delta B/Delta M` 共 `38,016` scalars/user，仅为完整一份
Current K/V (`2*6*1024*192`) 的约 `1.61%`。它天然对不同 candidate 产生不同 correction，并且不需要
per-request regression coefficient。

单 UID、`v0->v1` 的非正式预检得到：

- held-out Current/Parent majority-sign agreement 在多数 layer 接近 `0.98–0.999`，最弱 Current
  layer 约 `0.930`；
- 完整 Exact-state signed moments 的 coherent recovery 为 `99.97%`；
- exact Parent moments 加 128 个 layer-0-address landmarks 的 oracle Current-moment estimate 为
  `95.93%`，64 个为 `65.28%`。

这些数值只用于决定立正式协议，不用于挑 edge、隐藏 grid 或声称机制已经成立。

## 4. Design 1 candidate：迁移 response moments，而非迁移 token state

如果 population oracle 通过，Design 1 的构造路径是：

1. 完整 Parent K/V 直接形成 Parent majority-cone moments；
2. 对 full raw history 独立计算 exact Current layer-0 K，用它在跨版本 address space 中分配固定预算；
3. 只对预算内的真实 event 做 sparse causal replay，得到 approximate Current upper-layer K/V；
4. 用 quadrature mass 累加 Current `B/M`，再减去 exact Parent `B/M`；
5. 丢弃 landmark K/V，只持久化 signed moments、version/cone metadata 与 lineage update state；
6. append 时对 `B/M` 做外积增量，eviction 时做对应减法；若 query 离开 cone，则触发预注册的
   boundary fallback，而不是拟合时间 scalar。

这里 address landmark 只是估计 Current additive moments 的数值手段，不是论文贡献本身。真正的方法是
把 nonlinear attention 在 recommendation query cone 内变成一个可更新、可相减、由真实 query 读取的
跨版本功能状态。

## 5. 为什么它不是已有方法的组合

Linear Transformer 通过预先选择可分离 kernel feature map，把模型本身改写成全局 recurrent state；
Fast Weight Programmer 说明 outer-product memory 与 linearized attention 的关系，参见
[Transformers are RNNs](https://arxiv.org/abs/2006.16236) 和
[Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174)。这里既不
替换 HSTU attention，也不假设全局 kernel factorization：`ELU(qK)+1` 仍保持原样，只在观测到的用户
query cone 内利用其**分段精确**的 affine 结构。

普通 KV compression/merging 试图近似同一版本的完整 context；cross-model KV transfer 则学习
source-to-target tensor map。本文候选保留完整 Parent response，只迁移 release update 在用户 reader
上的 signed functional moments。没有 learned token、ridge/MLP、candidate ID feature 或 output-score
loss。

因此，若最后只剩“用 address clustering 选 128 个 token”，这条路线仍应判失败。只有以下三点被
因果实验证实，才达到论文贡献门：

1. **Cone observation**：anchor-defined majority cone 在 held-out 与 rolling query 上稳定，full moments
   在五边保持高恢复；
2. **Moment constructor**：合法 sparse replay 在 `0%–20%` 内生成足够准确的 Current moments；
3. **State evolution**：`B/M` 能随 append/eviction 增减，并在 cone crossing 时有明确 fallback。

## 6. 正式证伪顺序

第一层 representation canary 固定 32 users、五 edge：

- `full signed moments` 必须 edge-equal recovery 至少 `0.90`，且最差 edge 至少 `0.80`；
- 全量报告 held-out sign agreement、negative response fraction 与 Current/Parent sign-crossing rate；
- 128-point address moment oracle 达到至少 `0.70`、4/5 edge 正向，才允许 512 discovery；
- chronological moment estimate 是同预算 control，不能只报告 address rule。

第二层才实现不读 Current Exact upper-layer state 的 sparse causal replay。若 full moments 不通过，
attention-cone Insight 被否决；若 full moments 通过而 sampled/executable constructor 失败，只保留新的
functional representation observation，不把它写成完成的 Design 1。
