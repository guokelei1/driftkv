# Insight 2 follow-up：attention address space 与跨版本 response defect

日期：2026-09-02  
状态：**由 chronological coreset 反例导出的待证伪假设；不是冻结 Insight，也不是最终 Design**

## 1. 这一轮真正要检验什么

已完成实验说明，跨版本误差在 token/time 坐标上没有可用的简单 locality；但它到达 S4 aggregated
context 后，同请求 correction 又高度集中。固定 response vector、少量时间 scalar 和按时间均匀的
signed landmarks 都不能保持这种恢复。共同缺失的不是一个更强 mapper，而是 Transformer reader
本身用于区分历史状态的坐标：**query–key attention address**。

对于 HSTU 当前层的 native kernel `rho_l(q,k)`，精确 response defect 是：

~~~text
Delta R_l(q)
  = sum_i [rho_l(q, k^C_li) v^C_li - rho_l(q, k^P_li) v^P_li].
~~~

chronological midpoint 只保证 position coverage，并不保证 `rho(q,k)` 的地址覆盖。当前 Medium
checkpoint 使用 `ELU(qK)+1` 且 raw qK 跨越较大的正负范围；两个时间接近的 token 可以位于不同
attention half-space，时间很远的 token 也可以被同一 query 以近似方式读取。因此这一轮的可证伪命题是：

> 跨版本 state error 在 token/time 坐标中是分布式的，但 Current reader 消费的 signed functional
> defect 可能在 attention address space 中形成紧凑、可由真实 query 读取的 quadrature support。

这比“S4 tensor 低秩”更具体：它预言 address-covering landmarks 应在同样 atom 数量下系统优于
chronological landmarks；如果不能，attention-address hypothesis 就被否决。

## 2. 固定的 oracle 对照

本轮不拟合 response、不读取 candidate、不读取 label，也不学习 virtual token。对每个 user-edge：

1. 从 full-history layer-0 Current/Parent K 构造每个位置的 paired address feature；
2. L2 normalize 后用 deterministic nested farthest-first 覆盖联合地址空间；
3. 对每个预算前缀做 Voronoi assignment，以 cluster mass 重加权被选位置的 Current/Parent V；
4. 所有层和 head 使用同一组真实历史位置，避免 layer-wise union 破坏可执行闭包；
5. held-out candidate 的真实 Current query 只经模型原生 attention kernel 读取 paired signed atoms；
6. `R=N` 必须重建 Current Exact，作为数值正确性检查。

选择只使用 layer-0 K，是因为 Current layer-0 projection 对每个 raw event 独立可算；Current upper-layer
Exact K/V 只用于本轮 oracle atom value，不能进入后续 estimator。若必须使用 upper-layer Exact address
才能选点，则该机制不具备合法 constructor。

chronological midpoint canary 已被封存为同协议的负对照。本轮只改变 landmark geometry 和对应
quadrature mass，不改变 reader、注入点、query panel 或评价指标。

## 3. 即使 oracle 通过，什么才算论文 Design

attention-space clustering 本身已有大量先例，不能作为 EvoKV 的创新声明。它在这里最多是一个
constructor component。最终 Design 只有同时具备以下因果链才有论文意义：

~~~text
complete Parent response as a Reuse control variate
  + compact signed Current-minus-Parent response defect
  + attention-address support selected without labels or target requests
  + sparse causal replay that propagates earlier version defects
  + native Current-query read before gate/residual
  + append/eviction transport tied to cache lineage
~~~

这里迁移的是两个 release 之间的 compatibility defect，不是压缩一份普通 KV，也不是把 Parent KV
映射成 Current KV。future query 的 correction coefficient 由 Transformer 原生 interaction 现场产生，
不是一个学习得到的 `query -> correction` 函数。每个 entry 对应真实历史事件、真实位置和明确的
positive-Current/negative-Parent replacement 语义。

若 oracle 通过，下一阶段才另立合同实现：full-history exact Current layer-0 address scan 选点，随后按
时间顺序让这些真实 event 读取完整 Parent prefix 与 earlier signed entries，执行一到两轮 sparse causal
replay，生成 approximate Current upper-layer atoms。负项引用既有 Parent cache；只持久化正项、索引、
mass 与 lineage metadata。

## 4. 论文贡献准入与反例

以下任一情况出现，都不能把候选写成 Design 1：

- 只有某个单 UID 或某条 release edge 有效；
- 需要用 held-out quality 挑 `R`、selector、head 或 layer；
- 只有任意 ridge/MLP/free virtual atoms 通过，而 native real-state atoms 失败；
- address oracle 通过，但合法 sparse causal replay 在 `0%–20%` 内不能保持至少约 `80%` recovery；
- cutover 有效，但 append/eviction 后 correction 快速坍塌；
- 去掉 signed replacement、Parent control variate 或 causal propagation 后效果不变。

反过来，支持完整机制至少需要三层证据：

1. **Observation**：address coverage 在同 storage 下显著胜过 time coverage，且五 edge 方向一致；
2. **Construction**：不读 Current Exact upper-layer state 的 causal replay 接近 oracle，并在甜点区过门；
3. **Persistence**：同一 signed state 被不同 candidate 与连续请求原生读取，随 lineage 更新后仍保持恢复。

因此当前方法的“论文味道”不来自一个复杂名称或更多参数，而来自一个 Transformer-specific 的误差
坐标发现，以及由这个坐标严格推出、可以被消融和证伪的 migration mechanism。
