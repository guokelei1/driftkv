# 历史探索代码

这些是保留的历史实验编排、诊断和变体。可复用的数值实现与公共依赖已迁到 `src/hstu_kvcache`。
历史输出中的部分输入、权重已在此前清理；保留代码不代表旧命令能直接恢复实验。

| 目录 | 内容 |
| --- | --- |
| `query_relay`、`cachetide_runtime` | CacheTide 原型诊断、评测与运行优化实验 |
| `code_structure`、`compact_read`、`partial_refresh` | 码结构、紧凑读取和部分刷新探索 |
| `system_study` | 旧服务系统、事件与成本研究 |
| `baseline_studies`、`max_baselines` | 较早的基线评测编排 |
| `read_correction`、`nonlinear_read`、`query_read`、`selective_recompute` | 读取校正与局部重算的历史入口 |
| `history`、`release_replay`、`reuse` | 原目录说明的历史留档，公共代码已提取 |

这里的旧 README 保留当时的叙述，过时命令和删除的结果链接请结合
[路径对照](../../docs/code_migration.json)及[迁移说明](../../docs/code_layout.md)阅读。
不自动恢复已停止或完成的队列。
