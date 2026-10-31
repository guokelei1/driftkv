# 相邻版本 Reuse 的保留范围

Medium、Large、Max各五条边，共15条，已于2026-09-25完成。
最终指标与口径见[结果索引](../../results/unified_reuse_2026_09/README.md)，
必要设置见[固定计划](../../configs/unified_reuse_2026_09/adjacent_e14.json)和
[模型版本清单](../../docs/unified_training_2026_09/model_versions.md)。

本目录只保留`common.py`，为当前Design及Motivation提供哈希、JSON读写、
固定模型/窗口和公共路径工具。原准备、探针、逐边队列与adjudication入口已删除；
原评分/请求分片和运行状态也已清理。15条边的原始最终summary保留，
历史来源哈希说明当时执行，不表示旧队列可以续跑。

新测量需要新的完整输入与实验设置；本目录没有等待执行的任务。
