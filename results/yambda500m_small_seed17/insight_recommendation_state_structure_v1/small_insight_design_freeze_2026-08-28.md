# Small/seed17 Insight 与 Design 冻结记录

日期：2026-08-28  
状态：**Insight discovery closed；Small C32 Design frozen；Medium/Large training not authorized**

## 1. 冻结裁决

Small/seed17 已完成机制发现、实现打通和边界暴露。后续不再在同一五条 D14/E14 edge 上调整
probe、carrier、decay、coverage scale 或选择性删除反例，也不再开启 semantic GROUP、embedding
selector、history-token basis、PCA/SVD 或新的 operator catalog。

冻结的核心 Insight 是：

> parent-state 与 Current reader 的兼容性误差来自 history-side distributed error，在 HSTU 的
> query-dependent `activated(qK) · V` 聚合处形成，并在 AV 边界成为可由整个候选集合共同消费的
> candidate-shared correction；该 correction 能跨相邻真实请求保持方向稳定。

冻结的单一具体 Design 是 **C32 lightweight PRO**：每个用户在版本更新时由 Current-version
reader 对少量固定 probe/carrier 执行一次 compact read，生成一个 AV compatibility sidecar；随后
整份 candidate bank 复用该 sidecar。Reuse、Design 0 和 Exact-All 是对照路径，不是与 C32 串联的
第二套服务设计。未来若需要 label-free admission/No-op fallback，它是同一 Design 的安全边界，也不是
Design 0/Design 1 的两层机制。

## 2. 支撑冻结的完整证据链

| 证据 | 已观察结果 | 冻结解释 |
| --- | --- | --- |
| signed causal intervention | candidate-shared causal gate 通过，并由真实 exposed candidates 复核 | shared correction 不是候选采样假象 |
| reader-stage localization | correction 最早在 `activated(qK) · V` 聚合处稳定形成 | 注入边界应为 AV，不是任意 K/V token splice |
| request persistence | 相邻真实请求方向 cosine `0.966–0.983`；coverage-scaled recovery `60.6%–84.1%` | 可以按用户生成一次，而不是逐候选计算 |
| compact AV canary | 五条 edge 中 4/5 相对 Design 0 改善 | compact sidecar 具备进入真实质量验证的最低资格 |
| C32 rolling quality | 217,584 个真实请求；相对 Reuse 的 ROC-AUC 5/5 改善，log-loss 3/5 改善；edge 均值为 `+0.06641` AUC pp、`−7.65e-5` log-loss | 方法总体可行，但原事前严格双门仍如实为 FAIL |
| C32 theoretical compute | Exact-All release-time Full FLOPs 的 `9.14%`；不物化 translated prefix | 已形成约 10% 档的可执行设计点 |

原事前 quality gate 要求 AUC 与 log-loss 各至少 4/5 edge 不劣，因此 C32 不是 serving-qualified
结果；后验的“平均为正且过半 edge 正向”只能支持继续跨规模验证，不能改写原 gate。

## 3. 保留的反例与负结果

- `v3→v4` 不删除、不救边。该 edge 的 Full-only AUC 增益仅约 `0.046 pp`，Current Exact 的
  log-loss 也差于 Reuse，说明上游 release signal 本身很弱；它是边界证据，不足以推翻五边 AUC
  均正向的 C32 结论。
- 首个 matched-cost history-value basis 为 0/5，排除了“shared correction 是历史 V 向量的简单
  可加 basis”这一解释。
- 双 probe 在 5/5 edge 几乎完全一致，但 C32 对 Exact shared AV 的 absolute direction gate 只在
  cutover 2/5、rolling 0/5 通过；oracle amplitude-only 改进为 0/5+0/5。因此误差不能归结为 probe
  随机性或单纯幅值失准。
- C64 relative L2 对 C32 在 cutover/rolling 均为 5/5 改善，但 rolling absolute-direction gate 为
  0/5，且 C48/C64 不形成单调 precision axis。progressive PRO 增量不升级，正式设计仍为原 C32。

这些负结果属于 development evidence，必须与正结果一同保留；不得在 Medium 上重新展开已经排除的
结构分支。

## 4. 冻结后的研究边界

Small 的结论只覆盖 10k 用户、4L/H128/context512、training seed17 和五条 D14 edge。它证明了机制
存在和 C32 的最低 Design viability，没有证明跨 seed、30k/6L Medium、真实 GPU runtime、I/O 或
持久状态写入收益。

下一阶段只允许：

1. 先建立稳定的 Medium Full-only release environment；
2. 在独立、事前冻结的 Medium edge 上复核 signed candidate-shared、AV boundary 与 persistence；
3. 按 Medium 架构重新换算约 10%/20% 两个预算点，再验证冻结 PRO；
4. 质量通过后才做额外 seed 与 runtime qualification。

Small/seed17 本身不再承担 estimator 选择，也不因本冻结获得 serving、Medium training 或 runtime
授权。

## 5. 可复核入口与冻结哈希

- 完整专家讨论稿：`expert_discussion_summary.md`；
- C32 rolling quality 合同：
  `configs/contracts/yambda500m_small_hstu_native_pro_lazy_rolling_quality_v1.yaml`
  (`sha256=3735bac1baf18781d86dc4df44d51bd32f311045700c46711cdbd4e5356f1e72`)；
- C32 rolling summary：
  `results/yambda500m_small_seed17/insight_pro_lazy_reader_v1/rolling_quality_v1/formal/summary.json`
  (`sha256=1cee68f72aed083d1d0763eb8f6c2f00c67ca92a1a0105542071271c3a6f4abe`)；
- progressive decomposition adjudication：
  `results/yambda500m_small_seed17/insight_progressive_pro_v1/decomposition_v1/formal/adjudication.json`
  (`sha256=3e9463c9f17a57960569a9c53a0905b9fb76202dabd2a2e034ee03fed297665f`)；
- progressive frontier adjudication：
  `results/yambda500m_small_seed17/insight_progressive_pro_v1/frontier_v1/formal/adjudication.json`
  (`sha256=8a3fa358036e284982d323182fd77d0f5307af3e0d496cbf728c912bde7cb884`)。

一句话冻结：

> **停止 Small/seed17 的开放式 Insight 与 estimator 调整；保留 persistent candidate-shared AV
> compatibility correction 为核心 Insight，保留 C32 lightweight PRO 为唯一 Design，并把下一笔
> 研究资源用于 Medium 的 Full-only release stability 与随后独立的 scale qualification。**
