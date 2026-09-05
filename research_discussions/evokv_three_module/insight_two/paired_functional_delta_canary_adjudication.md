# Paired functional-delta canary 裁决

日期：2026-09-02  
状态：**正式 canary 已裁决；sparse causal-closure Design 候选未准入**

## 结论

`recursive paired functional-delta closure` 有独立的因果语义，也在 matched support 上显示平均正增益，
但正式 population canary 证明它不能成为当前 Design 1：R64 closure recovery 为 `-0.1962`，paired
bootstrap 区间跨零。按预注册 gate 停止，不启动 512-user discovery，也不继续调整 carrier selector。

与此同时，full-history activation-region representation 的 P8 recovery 为 `0.9951`、最差 edge
`0.9873`。Exact-state R64 carrier oracle 却只有 `-0.4532`。最重要的科研结论因而不是“recursion
还需调参”，而是：

> 跨版本功能差异在 reader 输出处高度可压缩，但这种 output compactness 不意味着它在输入 token
> support 上稀疏；dense reduction 与 sparse support 是两个不同性质。

这条反例把下一轮设计空间限制为 whole-history functional construction。任何只更换 sampling、
clustering、distance、mass assignment 或 mapper 的方案均不再准入论文方法候选。

## 正式证据

- v1 contract SHA256：`605fc606d1e1ba0f78c17841a6580d807a97cf25d6775dbfedb6ee320a460e83`；
- v1 在首个 metric row 前发生 metadata finite-check instrumentation failure，failure record SHA256：
  `6a86f03550dff22c734d98539a61519e3ee4fd756339feb8bcbd79e484df02de`；
- execution-only v2 contract SHA256：
  `45d6d5affc2626f527547e74e2af76e41677d9c4da98b872f753abf4d50bc8b2`；
- 32 users × 5 edges；1,920 metric rows；160 diagnostic/correctness rows；
- P8 representation recovery `0.9951109461`，minimum edge `0.9872836266`；
- R64 Exact-state carrier oracle `-0.4532352753`；
- R64 independent `-0.3430553030`；recursive closure `-0.1962431751`；
- closure gain `+0.1468121279`，4/5 winning edges，95% CI
  `[-0.0060331220, 0.3834080392]`；
- observed maximum R64 total compute `17.5871%` Exact-All；budget gate pass；
- Design gate fail；confirmation、labels 和 quality join 均未读取。

正式 machine-readable 裁决位于：
`results/yambda500m_medium_seed17/insight2_functional_boundary_v1/diagnostic_paired_functional_delta_v2/canary/analysis/summary.json`。

## 对创新性的裁决

以下内容不构成 Design 1 创新：activation moments 本身、Parent control variate、paired subtraction、
token coreset、address clustering。recursive closure 原本可能构成独立机制，因为迁移 defect 同时作为
serving state 与自身 upper-layer continuation 的 causal prefix；正式结果没有达到机制和质量门，故不
保留为最终 Design。

下一候选必须同时满足：

1. 全历史覆盖，不以少量真实 token support 近似 dense response；
2. 由 Transformer 原生 query interaction 读取，而非输出映射或 score fitting；
3. 改变 release-time computation graph，并有可测试的 exact/limit invariant；
4. 完整生成成本落在 Exact-All 的 `0%–20%`；
5. 方法贡献即使暂时拿掉调参，也能用一条明确的科学命题表达。
