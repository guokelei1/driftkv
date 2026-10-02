# RecFlow 脚本入口

本目录是独立生成式开发轨道。当前协议、授权和下一研究问题见
[开发计划](../../docs/recflow/plan.md)，完整指标与失败见
[结果索引](../../results/recflow/README.md)。4096用户A–F扩展已完成，以下入口不是重跑指令。

## 数据、完整训练与评价

| 入口 | 用途与主要输入 |
| --- | --- |
| `prepare_data.py` | 完整realshow预处理与因果审计；输出本地 `data/processed/recflow_v1/` |
| `expanded_daily_panels.py` | 固定4096用户、A/日更新拟合池、完整D19–24及每日6144采样面板 |
| `run_expanded_daily.py` | `--config --output`指定扩展链；`--canary`检查tiny A/B，正式路径读取 `--canary-report` |
| `expanded_chain_report.py` | 汇总完整前缀、base/random及全部配对结果、epoch与实际optimizer谱系 |
| `window_chain.py` | `--config --phase --output`；后续阶段用 `--checkpoint`接续真实模型/AdamW |
| `distributed_probe.py` | 固定global batch的2/4-rank数值、尾部和资源检查；完整epoch训练公共函数 |
| `parallel_evaluate.py` | FP32 beam请求分片；恢复面板顺序并重算全局request/user/day聚合 |
| `evaluation_pair.py` | `--parent --current --output`比较同一面板上的独立评价 |
| `random_baseline.py` | `--runs --output`读取评价目录，计算解析随机期望和随机策略分布 |

`window_chain.py`支持 `--device`、`--eval-devices`及torchrun DDP，
日期和epoch从配置读取；`--canary`不能产生正式训练父端点。
每轮必须覆盖全部eligible请求，不重复填充尾部；保留顺序/目标hash、
模型与实际optimizer步数。父模型不能已拟合本阶段的新日期。
只有明确完整的checkpoint才能进入下一阶段，日志进度不是完成证明。
执行源码在链内按hash固定，不在活跃任务期间修改。

`evaluation_pair.py`默认 `full_catalog`/`ndcg@50`；当前采样协议须显式使用
`--primary-scope uniform_1000 --primary-metric ndcg@50`。
失败free确认用其原 `ndcg@100`配置，不因导航清理更改默认或历史主指标。
随机分析读取具体evaluation目录，不读取外层训练phase summary；
默认5000次、seed20260918，采样子面板使用自己的分母。
窗口汇总保留所有质量失败，不自动准入release。

多卡设备编号相对于 `CUDA_VISIBLE_DEVICES`。训练用NCCL，驻留rank等待用CPU Gloo，
避免等待collective阻塞同GPU的评价worker。保持global batch和不等长尾部权重；
worker显存不是驻留模型/optimizer存在时的整卡峰值。

## 当前保留的公共依赖

| 模块 | 当前用途 |
| --- | --- |
| `window_panels.py` | 固定用户、目录和请求采样常量；4096用户面板生成器复用 |
| `daily_window_panels.py` | 当前4096面板依赖的旧512用户输入面板准备；原NPZ及manifest一并保留 |
| `development_probe.py` | 当前训练与并行评价复用的数据、batch和生成评分原语 |
| `window_comparison.py` | 当前配对评价复用的同面板检查 |
| `daily_comparison.py` | 当前扩展链报告复用的读取、hash和一致性检查 |

这些模块与上表9个入口组成当前A–F执行链的14个脚本，旧命名不代表已退役。
全部 `src/hstu_kvcache/recflow/`、共享模型实现及RecFlow测试继续保留。

## 保留模型与执行来源

实体模型保留4096用户六层链的A epoch3及B–F epoch1，共六个端点。
旧分支、中间和canary权重已移除；对应配置、全部结果、失败、随机对照及
训练/谱系记录继续保留，旧权重不再是现存输入。
完整历史设置及协议选择见[决策表](../../docs/recflow/plan.md#已完成决策与失败边界)。

当前冻结配置为 `configs/recflow/window_6l_expanded_u4096_seed17.json`。
原运行22份执行源的哈希及3份精确历史副本见
[源码封存清单](../../results/recflow/source_snapshots/expanded_u4096_seed17_launch_2026_09_18/manifest.json)。
副本只作复现证据，不替换当前共享模块；完成记录不授权重新训练。

独立重评通过 `development_probe.py --evaluate-checkpoint PATH --eval-day DAY --output NEW_DIR`；
必须匹配保存的架构、catalogue和 `--history-categories`设置，
`--decoder exact --eval-precision fp32`选择bound-pruned exact生成。
原beam、exact与采样结果不可混称；历史限时/限请求训练示例已从导航移除。
数据因果、生成路径、OOV、随机门槛和正式phi边界统一由[计划](../../docs/recflow/plan.md)定义。
