# RecFlow 脚本入口

4096用户A–F六模型链已完成。协议、质量失败和用户角色见
[开发计划](../../docs/recflow/plan.md)，指标见[结果索引](../../results/recflow/README.md)。
以下文件保留实际实现与依赖，不是重跑队列。

| 文件 | 保留用途 |
| --- | --- |
| `prepare_data.py` | 完整realshow预处理与因果审计 |
| `run_expanded_daily.py` | A–F执行编排；通过子进程调用训练、评价与随机基线 |
| `window_chain.py` | 分阶段完整epoch训练、真实模型/AdamW谱系与评价调度 |
| `distributed_probe.py` | 完整epoch训练公共函数、固定global batch与不等长尾部 |
| `development_probe.py` | 训练/评价共用数据、batch与生成评分；独立checkpoint评价 |
| `parallel_evaluate.py` | FP32 beam请求分片及全局request/user/day聚合 |
| `evaluation_pair.py` | 同面板Parent/Current独立评价比较 |
| `random_baseline.py` | 解析随机期望与随机策略分布 |
| `window_comparison.py` | 配对评价的同面板检查 |
| `daily_comparison.py` | 扩展链报告的读取、哈希与一致性检查 |
| `expanded_chain_report.py` | 完整前缀、配对结果、随机对照与optimizer谱系汇总 |

当前共11份Python文件。旧4096/512用户面板构建脚本及历史源码副本已清理；
当前原始/处理数据、冻结用户角色、设置和六个选定模型继续保留。
六个模型为A epoch3及B–F epoch1，固定配置为
[window_6l_expanded_u4096_seed17.json](../../configs/recflow/window_6l_expanded_u4096_seed17.json)。
`src/hstu_kvcache/recflow/`及相关数值、因果检查仍是上述调用链的依赖。

评价必须匹配保存的架构、catalogue、历史category设置和面板。
`evaluation_pair.py`默认`full_catalog`/`ndcg@50`；采样协议显式使用
`--primary-scope uniform_1000 --primary-metric ndcg@50`。
失败free确认保留原`ndcg@100`口径，beam/exact与采样结果不能混称。
随机分析读取具体evaluation目录，默认5000次、seed20260918。

训练保持完整eligible请求、顺序/目标哈希、global batch、尾部权重与真实optimizer步数；
父模型不能已拟合本阶段的新日期。多卡编号相对于`CUDA_VISIBLE_DEVICES`。
原日志进度与旧源码哈希不代表当前可继续执行；新训练遵守独立协议及授权边界。
