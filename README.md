# EvoKV / CacheTide

研究模型发布后如何复用持久化 K/V，并用读取校正恢复推荐质量。

## 找代码

| 要找的内容 | 位置 |
| --- | --- |
| HSTU、注意力、原生 KV 与追加 | [src/hstu_kvcache/models](src/hstu_kvcache/models/) |
| Yambda 数据、人口、窗口和请求清单 | [data](src/hstu_kvcache/data/) |
| RecFlow 模型与反馈数据协议 | [recflow](src/hstu_kvcache/recflow/) |
| Yambda / RecFlow 训练 | [training](src/hstu_kvcache/training/) |
| 模型装载、历史轨迹、评测、成本及统计 | [evaluation](src/hstu_kvcache/evaluation/) |
| Layer、Tail、Deviation 等局部重算 | [methods/baselines](src/hstu_kvcache/methods/baselines/) |
| KV Map（Large / Max） | [methods/kv_map](src/hstu_kvcache/methods/kv_map/) |
| CacheTide 的 state、read、session、校准及内核 | [methods/cachetide](src/hstu_kvcache/methods/cachetide/README.md) |
| 较早的读取校正基线 | [methods/read_correction](src/hstu_kvcache/methods/read_correction/) |
| 共享文件记录、哈希工具 | [common](src/hstu_kvcache/common/) |

Reuse 使用已有 KV 直接读取，Rebuild 用当前模型重新构建 KV；其公共执行逻辑在
[evaluation/trajectory.py](src/hstu_kvcache/evaluation/trajectory.py) 和
[evaluation/baselines.py](src/hstu_kvcache/evaluation/baselines.py)，Figure 9 的实验策略在各自 `engine.py`。

## 运行入口

在仓库根目录执行：

```bash
python scripts/evokv.py --help
python scripts/evokv.py train-yambda --help
python scripts/evokv.py train-recflow --help
python scripts/evokv.py figure9max --help
```

安装开发环境可用 `python -m pip install -e '.[dev,data]'`。`scripts/` 只保留薄入口；
公共实现从 `hstu_kvcache` 导入。完整代码索引与迁移说明见 [docs/code_layout.md](docs/code_layout.md)。

## 找实验与结果

- [Figure 9 Large](experiments/figure9a/README.md)、[Figure 9 Max](experiments/figure9max/README.md)：各自保留配置、执行流程和结果。
- [第三章机制探索](experiments/ch3_motivation/code/)：变体、诊断和历史队列；其配置、输入和结果路径保持原样。
- [RecFlow foundation 流程](experiments/foundation_recflow/code/)：训练链编排和比较实验。
- [历史探索](experiments/archive/README.md)：旧系统研究、部分刷新、码结构等；按主题集中保留。
- [文档](docs/README.md)、[模型版本记录](docs/unified_training_2026_09/model_versions.md)、[测试](tests/README.md)。

数据、权重及结果继续放在 `data/`、`results/` 和各实验的 `runs/` 下。论文仓库在相邻的
`../paper`，论文图由该仓库管理。旧实验状态记录和冻结计划是历史证据，不是待启动任务。
