# 脚本入口

此目录只保留当前数据处理、六层/十层训练评估、论文 Motivation 和两个 Insight 的流程。
从下表选择入口；不要按版本号猜测哪个脚本最新。正式运行仍需原合同、canary 和用户授权，
本索引不授权训练或重新生成封存结果。

日常开发先用已有数据和最小样本验证当前假设，复用工具，按实际路径补实现。
只检查会影响数值、数据因果、评价和运行成本的部分；不为小改动运行完整测试，
也不把通用异常处理、并发或恢复机制当作 idea 验证的前置条件。
小探针的配置与结果记录在现有实验上下文中；长作业再落实相应合同、资源和 canary。

## 当前流程

| 用途 | 入口 |
| --- | --- |
| 六层 Medium 训练与 Full/Reuse 矩阵 | run_yambda500m_medium_full_reuse_matrix.py |
| Medium V5 扩展 | run_yambda500m_medium_d14_v5_extension.py |
| Motivation 完整旧 producer 对比 | run_yambda500m_medium_d14_direct_long_age_reuse.py |
| 十层 Large 基础训练和 Full-only 队列 | run_yambda500m_large_qualification.py |
| Large V4 两 epoch 轨迹 | run_yambda500m_large_v3_v4_epoch_sweep.py |
| Large V5 从 V4@2epoch 继续训练 | run_yambda500m_large_v4e2_to_v5_epoch_sweep.py |
| Large 当前 V0–V5 模型路径检查 | validate_yambda500m_large_d14_canonical_chain.py |
| 论文重算成本表 | benchmark_release_cost.py |
| Insight 1 局部替换诊断 | [insight_one_locality/README.md](insight_one_locality/README.md) |
| Insight 2 响应修正和持续性诊断 | [insight_two/README.md](insight_two/README.md) |
| 六层 Design 四组件原型、校准与连续评价 | [design/README.md](design/README.md) |
| 只修改或生成论文图片 | [../figures/README.md](../figures/README.md) |

Medium 和 Large 的具体命令、GPU 设置、窗口、epoch、hash 和历史范围分别在
[Medium 训练记录](../docs/medium_scale_training_plan.md)和
[Large 训练记录](../docs/large_scale_training_and_qualification_plan.md)中保留。
Large 的 v4e2_vs_legacy_v5_full_only 是保留的历史对照，**不是当前 V5 训练入口**。

## 公共数据与评估工具

数据入口为 download_scale_datasets、prepare_yambda500m_scale_populations、
plan_yambda500m_streaming_windows、build_yambda500m_unified_scales。
manifest 由 build_yambda500m_foundation_manifests 和
build_yambda500m_hstu_native_matrix_manifest 构造。训练复用
train_yambda500m_foundation_fsdp。原来的 Small 默认合同仍是历史默认值；
当前 Medium/Large 应由上表 runner 传入对应合同，不能裸跑默认参数。

Full-only 使用 evaluate_yambda500m_release_candidates_raw 与
adjudicate_yambda500m_release_candidates。Reuse 使用
evaluate_yambda500m_hstu_native_onehop_reuse_raw 与
adjudicate_yambda500m_hstu_native_onehop_reuse，共享 evaluate_yambda500m_foundation_raw。
benchmark_yambda500m_history_cpu 和 canary_yambda500m_large_state_io 是支撑检查。

insight/ 现在只有六个共享函数模块，不再包含旧路线的实验启动脚本。
它们被当前 evaluator、Insight 2 或 Large 原资源 canary 引用，详见该目录 README。
旧 refinement 和 evidence-measure 的构造器、命令行选项及专属测试已经移除；
参数映射独立到 parameter_maps.py。保留 PRO 依赖不表示它是当前 EvoKV 方法。

## 已退出的入口

四层专属 rolling recipe、one-hop completion、旧 long-age reuse v1/v2/v3 及旧 Insight
候选启动脚本已删除；不要将它们与 Medium 的现行 runner 混用。
历史删除清单、恢复包与证据边界见 [结果索引](../results/README.md)。
