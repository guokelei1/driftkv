# Insight 2 candidate：activation-region 中的跨版本 functional delta

日期：2026-09-02  
状态：**机制观察已过 32-user canary；constructor 与 Design 1 均未冻结**

## 1. 已经成立与尚未成立的部分

当前 Medium checkpoint 是仓库的 legacy unnormalized pointwise-attention 实例：`ELU+1`、无
relative-position bias；它不是原始 HSTU 论文中的 SiLU + relative-bias attention。因此，本页不把
下面的分段代数写成一般 HSTU 或一般 Transformer theorem。

对当前 reader 的一层一 head，在固定 positive region `P` 内：

~~~text
F_theta(q)
  = B_theta,P + s q M_theta,P + N_theta(q),

B_theta,P = sum_{i in P} v_theta,i,
M_theta,P = sum_{i in P} k_theta,i outer v_theta,i,
N_theta(q) = sum_{i not in P} exp(s q k_theta,i) v_theta,i.
~~~

正式 32-user、五 edge canary 显示：held-out recommendation query 对 anchor-majority region 的
Current/Parent agreement 均约为 `99.45%`；只迁移两版本 positive affine bulk 的完整 moments，
edge-equal recovery 为 `99.57%`，最差 edge 为 `98.92%`。Current/Parent negative-response fraction
仍约为 `3.76%/4.37%`，所以这是高恢复的 **exact affine bulk + nonlinear remainder**，不是完整
response 的精确 affine 等式。

同时，直接用 128 个 address samples 估计完整 Current moment 只有 `27.33%` recovery。因此已经
成立的是 functional representation observation；“少量样本可以构造它”尚未成立。

## 2. 候选 Insight 2

当前最准确、也最架构中立的表述是：

> 跨版本误差在 token/KV 空间中是分布式的，但 reader aggregation 后，**版本变化本身**比完整
> 历史状态更可压缩。对被测 pointwise-attention reader，同一用户请求共享稳定 activation geometry，
> 使 response difference 分解为共享 affine bulk 与较小的 query-dependent nonlinear remainder。

这个观察对 Design 的含义不是“把 KV 映射到 `B/M`”，而是改变 dependency closure：完整 Parent
state 保留为 control path，Current 计算只负责生成跨版本 functional delta；后续真实 query 通过模型
原有 interaction 读取 delta。

candidate bank 也不应成为 region 定义的隐藏输入。一个非正式 8-user 预检把 anchors 换成用户 raw
history 的 32 个固定 lower-midpoint item，完整 moment recovery 在五边仍约为 `99.6%–99.96%`。
这提示 region 可能由用户历史形成，而不是 candidate panel artifact；它必须在 prospective population
protocol 中重跑后才能成为证据。

## 3. 为什么下一步必须先做 version pairing

已失败的 sampled implementation 实际估计：

~~~text
sampled Current total - complete Parent total.
~~~

它再加回完整 Parent response 后，等价于用少量样本估计一个很大的 Current absolute total，未利用相邻
release 的共同部分。正确的诊断应在同一历史位置先形成 version delta：

~~~text
Delta B_hat
  = sum_{i in S} w_i [I_C,i v_C,i - I_P,i v_P,i],

Delta M_hat
  = sum_{i in S} w_i
      [I_C,i k_C,i outer v_C,i
       - I_P,i k_P,i outer v_P,i].
~~~

这里 Current 与 Parent 使用各自 region mask；不能强迫两版本共享一个 sign set。paired control
variate 是必要 baseline，但已有 control-variate、linear-attention 与 fast-weight 文献覆盖了这些单项，
它本身不是论文创新。

## 4. Design 1 candidate：在 functional space 闭合因果依赖

有论文意义的设计点必须是：**上层 Current carrier 不依赖 Exact Current prefix，而递归读取完整
Parent prefix 加已经构造的 lower-layer version-delta state。** 也就是把 token-state dependency closure
改写为 functional-delta dependency closure：

~~~text
exact Current layer-0 projection from raw history
  -> paired layer-0 delta ledger
  -> Current carrier query reads Parent prefix + causal delta ledger
  -> approximate Current layer-1 carrier K/V
  -> paired layer-1 delta ledger
  -> ...
  -> persistent all-layer reader-response delta
~~~

第一版合法下界使用真实 event carriers：对位置 `i`，Current token reader 只能读取 Parent prefix
`[0,i)`，不能读取 Exact Current upper-layer K/V。递归版还必须保证 carrier 在第 `l` 层只读取代表
`i` 之前 source mass 的 delta ledger；future carrier 不能代表 earlier token，否则会破坏历史状态的
causal semantics。

服务请求读取：

~~~text
complete mixed/Parent prefix response
  + query-read affine version-delta bulk
  + explicit nonlinear/boundary residual when admitted
  -> native output transform, gate, residual, and downstream layers.
~~~

这和普通 KV mapping、token compression 或 fast-weight cache 的边界是：它不重建 Current token state，
也不把完整历史改写成另一个同版本 cache；Parent response 是精确 control path，新增状态只描述一次
release update，并在 functional space 中递归承担上层 carrier 的依赖。

## 5. 失效检测与 lineage 语义

最终方案不能只在 cutover 有效：

- region certificate 必须由 label-free history probes、activation margin 或 nested delta-ledger disagreement
  产生；不读取 Current Exact target score。证书失败时使用 Reuse/Exact fallback，不能静默注入；
- release 后的新 event 已由 Current reader 生成 native K/V，不属于旧 Parent-to-Current delta；
- 旧 Parent segment eviction 必须同时耗减对应 delta mass。候选实现可按 chronological block 保存 ledger
  与 assignment metadata；整块停用或低频重建的误差、额外 bytes 和 compute 都必须实测；
- 若为更新而持久化 sampled Current/Parent atoms，其 storage 也必须计入，不能只报告最终 `B/M`。

这些接口尚未由实验闭合，因此当前不能称 executable migration framework。

## 6. 已得到的反例

非正式 8-user 预检已经排除了两个看似漂亮的捷径：

1. 只注入前 1/2 层完整 functional moments 的五边均值为 `-1.30/-0.98`；前四层才约为 `0.90`。
   因此只编译一两个 block 不足，而完整处理四层的 all-token linear work 又超过 `20%`；
2. 只迁移 Current/Parent majority-mask crossing positions，除首 edge 外 recovery 约只有 `0.04–0.08`；
   stable-region delta 反而约为 `0.84–0.96`。所以不能把 Design 简化为“只重算 crossing token”。

这两条结果说明核心难点是用少量 Current 计算恢复遍布全历史的 stable affine version delta，而不是
找到另一批“重要 token”。

## 7. 最小证伪阶梯

下一轮按以下顺序执行，任何一级失败都不进入下一级：

1. **Paired exact-state oracle**：candidate-free history probes；固定 `R=64/128`；同位置先做
   Current-minus-Parent 再聚合。若 `R128` 仍明显低于 `0.80`，停止 sampled moment constructor；
2. **Parent-conditioned carrier**：selection 只用 raw-history Current layer-0 projection 与 Parent state；
   upper K/V 只由 Current token over Parent prefix 生成。完整成本必须小于 `20%`；
3. **Functional causal closure**：carrier 逐层读取 causal delta ledger；必须优于第二级并通过 cache/
   time causality canary；
4. **Certificate and rolling**：冻结 cutover state，在真实 append/eviction trace 上测 fallback、mass
   depletion、恢复与额外 storage；
5. **Architecture boundary**：至少增加 faithful SiLU HSTU 小控制；标准 softmax 只检验 aggregation 后
   version error contraction，不套用本实例的 affine tensor。

只有第 3–4 级形成闭环，才能把它冻结成达到论文要求的 Design 1。第 1–2 级即使数值高，也只证明
paired functional delta 值得继续，并不构成最终贡献。

