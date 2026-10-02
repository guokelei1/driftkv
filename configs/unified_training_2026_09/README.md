# Yambda 三规模训练配置

本目录保存已完成训练的执行配置。[18 个选定模型](../../docs/unified_training_2026_09/model_versions.md)
是当前使用入口；各配置中的合同和源码哈希描述原运行，已完成输出不得覆盖。

## Medium / Large

| 运行 | 配置 | 当前端点 |
| --- | --- | --- |
| Medium V2 | [四卡 global32 / CPU14](medium_v2_4gpu_b32_cpu14_execution.yaml) | epoch2 |
| Medium V3 | [四卡 global32 / CPU14](medium_v3_4gpu_b32_cpu14_execution.yaml) | epoch1 |
| Medium V4 | [四卡 global32 / CPU14](medium_v4_4gpu_b32_cpu14_execution.yaml) | epoch1 |
| Medium V5 | [四卡 global32 / CPU14](medium_v5_4gpu_b32_cpu14_execution.yaml) | epoch1 |
| Large V4 | [四卡 global64 / CPU14](large_v4_4gpu_b64_cpu14_execution.yaml) | epoch2 |
| Large V5 | [四卡 global64 / CPU14](large_v5_4gpu_b64_cpu14_execution.yaml) | epoch1；epoch2 评价保留 |

Medium V0/V1、Large V0–V3 复用历史模型，其原配置由模型清单引用。
本轮版本训练入口为 `scripts/unified_training/run_medium_v2_2epoch.py`，
显式传入 `--execution-config` 选择运行；名字沿用最初 Medium V2，
实际按合同读取 Medium/Large/Max 数据和端点配置。
本轮只运行[既定 E14](evaluation_e14_only.yaml)。

## Max

- [5B / 20万用户数据处理](max_data_preparation.yaml)。
- [batch80 资源探针](max_probe_b80_execution.yaml)：历史四卡、每卡20、12步检查。
- [V0](max_v0_prepared_execution.yaml)：从头训练一轮；入口
  `scripts/unified_training/run_max_v0.py`。
- [V1](max_v1_epochs12_4gpu_b80_cpu14_execution.yaml)：连续两轮，选 epoch1。
- [V2 epoch1](max_v2_1epoch_4gpu_b80_cpu14_execution.yaml)：当前 V2；
  [追加 epoch2](max_v2_epoch2_from_epoch1_4gpu_b80_cpu14_execution.yaml)
  是重置 AdamW/RNG 的已完成历史分支。
- [V3 from V2@2](max_v3_epochs12_4gpu_b80_cpu14_execution.yaml)：当前 V3 取其 epoch2；
  [V3 from V2@1](max_v3_from_v2e1_epochs12_4gpu_b80_cpu14_execution.yaml)
  是另一条已完成比较分支。
- [V4](max_v4_epochs12_4gpu_b80_cpu14_execution.yaml)、
  [V5](max_v5_epochs12_4gpu_b80_cpu14_execution.yaml)：均连续两轮，均选 epoch2。

Max 使用四卡 global80 / CPU14。V1–V5 复用上述版本训练入口；
具体训练来源、数值检查和原准入见[Max 结果](../../results/unified_training_2026_09/max/README.md)。

## 保存范围

最终配置、合同、canary 摘要、预算、原始评分与封存保留；
中间探针及优化器恢复 payload 已清理，当前不存在可直接续训的这些恢复点。
只保留模型清单中的实体权重，历史多端点评价记录不代表备选权重可用。
配置存在不代表新训练授权。
