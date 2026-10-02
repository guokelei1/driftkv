# 相邻版本 Reuse 入口

Medium、Large、Max 各五条边，共 15 条，已于 2026-09-25 全部完成。
[结果与口径](../../results/unified_reuse_2026_09/README.md)、
[固定计划](../../configs/unified_reuse_2026_09/adjacent_e14.json)
和[模型版本清单](../../docs/unified_training_2026_09/model_versions.md)是使用入口。

## 执行阶段

从仓库根目录运行：

```bash
python scripts/unified_reuse_2026_09/run.py --prepare
python scripts/unified_reuse_2026_09/run.py --probe
```

`--prepare` 不占 GPU，核对既有 Full 分数、权重与请求身份并生成四卡分片。
`--probe` 在各规模 V4→V5 的小样本上检查数值、吞吐和显存。
已有记录位于 `results/unified_reuse_2026_09/<scale>/` 和 `probes/`。

正式入口是 `bash scripts/unified_reuse_2026_09/run_in_tmux.sh`：
逐边串行、每边 GPU 0–3 并行，只计算 Current 读取上一版缓存的 Reuse，
Parent/Current Full 使用已有封存结果。原始分数每 128 用户一片封存；
恢复时校验模型、请求、计划和源码绑定，跳过已完成分片。
输出包括 `status.json`、各边 `summary.json`、`runtime.log` 和退出码。

这是已完成实验的可复用入口；查看结果不需要重新执行准备、探针或正式队列。
新的长评测需要明确启动授权。
