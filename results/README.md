# Results registry

当前只保留与 HSTU-native motivation 复现相关的结果和必要输入。

## 保留结果

- data_audit/yambda500m_scale_v1/：Yambda-500M 人口与数据审计；
- yambda500m_small_foundation_canary_2026-08-24.md：foundation correctness canary；
- yambda500m_small_seed17/base.json：当前 Small foundation 输入；
- yambda500m_small_seed17/hstu_native_release_chain_v1/v0/：当前 D14/E14 使用的 parent checkpoint；
- yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3/：当前 HSTU-native recipe scan；
  train_1d、train_4d、train_7d 按当前保留决定保留，train_14d 及 D14 Full-only、One-hop
  Reuse、direct long-age 结果用于当前 motivation 复现；
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
