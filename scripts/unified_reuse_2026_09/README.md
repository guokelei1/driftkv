# 相邻版本 Reuse 入口

从仓库根目录运行：

```bash
python scripts/unified_reuse_2026_09/run.py --prepare
python scripts/unified_reuse_2026_09/run.py --probe
```

`--prepare` 不占 GPU；`--probe` 对每个规模的 V4→V5 做四卡小样本和旧评测器
Reuse 分数核对。结果分别写入 `results/unified_reuse_2026_09/<scale>/` 的
`binding.json`、请求分片，以及 `probes/` 下的探针记录。

用户明确启动后才在 tmux 中执行 `bash scripts/unified_reuse_2026_09/run_in_tmux.sh`。
该命令逐边串行、每边四卡并行；任意时刻重启同一命令会校验并跳过已封存分片和
已完成的边。输出 `status.json`、每边 `summary.json`、`runtime.log` 和退出码。
不重新测 Parent/Current Full。
