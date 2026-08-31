# Results registry

当前只保留与 HSTU-native motivation 复现相关的结果和必要输入。

## 保留结果

- data_audit/yambda500m_scale_v1/：Yambda-500M 人口与数据审计；
- `data/manifests/yambda500m_medium_hstu_native_d7_d14_v1/`（位于 data 目录）：已按 prospective
  Medium 合同物化的 `[0,300)` request manifest；包含 2,962,852 个因果去重请求，未计算质量指标，
  只授权数据准备和 focused canary；
- `yambda500m_medium_seed17/full_reuse_matrix_v1/smoke/`：GPU2/3 双 rank、global batch 32 的
  6L/H192/context1024 correctness canary；v0/v1 两步训练、Full-only raw seal 与 118-request
  adjacent-Reuse mechanics 已通过。它不构成 quality、release admission 或 formal 长训练结果；
- `yambda500m_medium_seed17/full_reuse_matrix_v1/cpu_runtime_v2_canary/`：D7 全部完成后、D14
  恢复前执行的 16-user raw-only CPU runtime canary；660 行 raw/seal 守恒且 hash 一致，固定 GPU2/3
  各 14 个互不重叠的 NUMA-local 物理核，不读取质量指标；
- `yambda500m_medium_seed17/full_reuse_matrix_v1/reuse_4gpu_runtime_v3_canary/`：剩余 D14 Reuse
  切换四卡前的 raw-only canary；GPU0/1/2/3、cohort32、query chunk256、每 rank 14 个本地物理核，
  共 1,000 请求/3,000 三路径行，四卡 peak reserved 6.7–7.3 GiB，未读取质量指标；
- `yambda500m_medium_seed17/full_reuse_matrix_v1/D7/forced_reuse_diagnostic_v1/`：用户在正式
  admission 已封存后要求补跑的 D7 全 20 格相邻 Reuse 诊断；该目录单独绑定 forced-diagnostic 合同，
  不改写 `D7/admission/`、正式 `D7/reuse/`、顶层 summary 或 serving/cache lineage；canary 与正式结果
  均保留 raw-first seal，完整矩阵不得选择性报告；
- `yambda500m_medium_seed17/full_reuse_matrix_v1/D14/v5_extension_v1/`：独立的 D14 v4→v5
  四卡扩展。v5 只训练完整 `[273,287)`；E3/E7 为完整窗口，`E14_partial` 明确包含不完整 day300，
  只能作诊断。`canary/` 不读取质量，正式 checkpoint、Full、Reuse 与 summary 均绑定独立合同，不改写
  原 D14 v1…v4 结果或 serving admission；
- `yambda500m_medium_seed17/full_reuse_matrix_v1/medium_scale_experiment_summary.md`：本轮 Medium
  seed17 的统一专家讨论稿；汇总模型/数据/训练、原始 D7/D14、D7 forced Reuse、D14 v5、统一百分比
  口径、运行成本、异常边、结论和下一步。Recovery 只使用同一 sealed 三路径 cohort 计算，不混用
  Full-only 与 rolling cohort；
- `yambda500m_medium_seed17/full_reuse_matrix_v1/structured_log_test_notice_2026-08-28.md`：记录并
  限定早期单元测试误写入 `pipeline.jsonl` 的 synthetic D7 edge1/2 事件；正式 seal/checkpoint/raw 未受
  影响，测试隔离已修复；
- yambda500m_small_foundation_canary_2026-08-24.md：foundation correctness canary；
- yambda500m_small_seed17/base.json：当前 Small foundation 输入；
- yambda500m_small_seed17/hstu_native_release_chain_v1/v0/：当前 D14/E14 使用的 parent checkpoint；
- yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3/：当前 HSTU-native recipe scan；
  train_1d、train_4d、train_7d 按当前保留决定保留，train_14d 及 D14 Full-only、One-hop
  Reuse、direct long-age 结果用于当前 motivation 复现；
- yambda500m_small_seed17/insight_recommendation_state_structure_v1/：固定 3,000 用户、五条
  v0..v5 边的 label-free recommendation-state observation；[专家讨论稿](yambda500m_small_seed17/insight_recommendation_state_structure_v1/expert_discussion_summary.md)
  将实验动机、协议、三个实验、结果、结论、反证边界与待讨论问题收束在一篇文档中；目录只保留
  compact population、state factorization、coreset、candidate-subspace 与 adjudication，不保留展开
  K/V/attention tensor；[Small Insight/Design 冻结记录](yambda500m_small_seed17/insight_recommendation_state_structure_v1/small_insight_design_freeze_2026-08-28.md)
  收口开放式探索，保留 C32 lightweight PRO，并明确 Medium 前不得继续用相同五边调 estimator；
- yambda500m_small_seed17/insight_candidate_shared_causal_v1/：signed、逐 head、四种 candidate
  width 的 3,000-user causal intervention，以及五边真实 exposed candidate raw-first 复核；正式
  causal gate 通过，但 shared/residual 仍是 diagnostic oracle；
- yambda500m_small_seed17/insight_evidence_measure_basis_v1/：唯一 matched-cost signed
  value-measure basis canary；五边 0/5 不弱于 Design 0，按合同停止，保留为机制负结果；
- yambda500m_small_seed17/insight_reader_compatibility_correction_v1/：按最新专家意见将 claim
  收紧为 candidate-shared reader compatibility correction，并事前冻结地定位其 HSTU 形成阶段、
  检查跨真实请求持久性；该目录中的 oracle 仍不是 action，机制能否解锁以最终 adjudication 为准；
- yambda500m_small_seed17/insight_av_broadcast_residual_v1/：两道 reader gate 通过后唯一执行的
  compact-probe AV sidecar；无标签 score canary 4/5 不弱于 Design 0，`v3→v4` 为保留反例；
  未启动 formal quality，未准入 action；
- yambda500m_small_seed17/insight_pro_lazy_reader_v1/：取消 per-position translated-prefix
  物化的 lightweight PRO 正确性/成本证据。`correctness_cost/` 保留 dimensionful AV absolute
  threshold 导致的 v1 失败；机制未变且换用下一批 32 用户的 `correctness_cost_v2/` 通过
  scale-aware 数值等价、零物化和理论成本门。32-carrier 为 Full 的 9.1%；`rolling_quality_v1/`
  保留五边 full-population raw-first 质量结果：AUC 5/5、log-loss 3/5、均值两项改善。总体 Design
  viability 为正，事前严格双门未过，未准入 serving action、额外 seed 或 runtime qualification；
- yambda500m_small_seed17/insight_progressive_pro_v1/：专家建议后的 label-free 增量。
  `decomposition_v1/formal/` 证明两条固定 probe 在 5/5 edge 几乎一致，但 C32 的 absolute direction
  与 amplitude-dominant 门均未通过，segment decay 仅 2/5；`frontier_v1/formal/` 完整报告
  C32/C48/C64 的 10.52%/14.54%/18.64% Full-FLOPs 轴。C64 relative L2 对 C32 在 cutover/rolling
  均 5/5 改善，但 absolute rolling direction 为 0/5 过门且 C48/C64 非单调，按事前规则不选择升级，
  不读取旧五边 label；
- checkpoint_cleanup_2026-08-24.md：已完成 checkpoint 清理范围的审计记录。

D14/E14 当前结果的核心汇总位于：

- hstu_native_rolling_recipe_matrix_v3/matrix_result.json；
- hstu_native_rolling_recipe_matrix_v3/d14_onehop_reuse_diagnostic_v1/；
- hstu_native_rolling_recipe_matrix_v3/d14_onehop_reuse_completion_v2/；
- hstu_native_rolling_recipe_matrix_v3/d14_direct_long_age_reuse_v1/。

## 已删除结果

旧 P7–P11、8L、archive、Yambda-50M audit、旧 Small fixed-endpoint diagnostics、
旧 evaluation 和 release_diagnostics 已删除。它们不再是当前文档、脚本或合同的输入。

## 存储规则

raw aggregate、seal、adjudication、summary 和必要 invalidation 记录属于可审计结果；
rank shard、progress marker、临时日志和重复中间产物不属于默认保留对象。新结果不得重新
建立按旧编号命名的结果族。
