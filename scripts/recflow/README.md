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

## 已完成开发分支的工具

| 入口 | 保留用途 |
| --- | --- |
| `window_panels.py` | 512用户完整拟合池及平衡三日面板 |
| `daily_window_panels.py` | 同512用户D19/D20拟合、D20/D21完整评价及独立768采样面板 |
| `daily_followup_panels.py` | D21拟合/D22评价；D22已使用，不是最终未见数据 |
| `run_daily_probe.py`、`daily_comparison.py` | 日更新1/3轮分支执行与四端点汇总 |
| `run_daily_lr_probe.py`、`lr_comparison.py` | 新增LR1e-4/3e-5与保留1e-3比较，核对复用的A/day20评价 |
| `metric_grid.py` | 三个LR、三个cutoff、三个scope、两天共54单元；保留全部结果 |
| `update_parameter_drift.py` | CPU参数漂移诊断，不代表KV兼容性 |
| `daily_confirmation.py` | 核对C→D、完整epoch和未来面板；`--root --settings`区分free确认与sampled探索 |
| `window_comparison.py` | 声明的A/B/C端点、相同未来面板、逐日和随机对照 |

配置位于 `configs/recflow/`；日更新、LR、free确认和sampled分支各自保留冻结配置。
已经完成的输出不得覆盖或因文档提供入口而重复启动。
完整历史设置及为何切换协议见[决策表](../../docs/recflow/plan.md#已完成决策与失败边界)。

## 其他公共原语与诊断

| 入口 | 用途 |
| --- | --- |
| `development_probe.py` | 历史bounded/resource probe、共享评分函数及独立checkpoint重评；不是当前完整epoch训练入口 |
| `check_decode.py` | beam与exhaustive结构化路径排名对照 |
| `check_history.py` | 同目标的完整/短/空历史条件loss，不等于检索质量 |
| `popularity_probe.py` | 同因果窗口的initial/cumulative/recent计数对照 |
| `supervision_coverage.py` | 重建旧pilot的seeded正例抽样与实际监督覆盖 |
| `initial_supervision_coverage.py` | 512/2048用户初始池的覆盖上界，不是模型改进 |

独立重评通过 `development_probe.py --evaluate-checkpoint PATH --eval-day DAY --output NEW_DIR`；
必须匹配保存的架构、catalogue和 `--history-categories`设置，
`--decoder exact --eval-precision fp32`选择bound-pruned exact生成。
原beam、exact与采样结果不可混称；历史限时/限请求训练示例已从导航移除。
数据因果、生成路径、OOV、随机门槛和正式phi边界统一由[计划](../../docs/recflow/plan.md)定义。
