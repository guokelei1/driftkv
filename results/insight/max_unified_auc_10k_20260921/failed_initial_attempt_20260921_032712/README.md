# Max AUC 首次执行失败记录

首次尝试于 2026-09-21 03:27 Asia/Shanghai 在 `evokv_max_auc_10k` 中启动，
已因初始校准显存不足而结束；没有产生完整评价分数。
[失败摘要](diagnostic/summary.json)记录 `status=failed`、68.312 秒及原错误，
[原配置](diagnostic/configuration.json)和[资源估计](resource_estimate.json)继续保留。

两条边完成了小型共享拟合，随后 32 用户 DroidSpeak profiling 的八候选块
展开为 256 份完整缓存并触发 OOM；先前八用户 canary 没覆盖该分配规模。
后续修复仅缩小候选块，保留 32 用户、16 候选和 136 个层区间。
修复检查与完成结果见[主记录](../README.md)。旧启动器与预览图已经退役，
本目录不再表示运行中任务。
