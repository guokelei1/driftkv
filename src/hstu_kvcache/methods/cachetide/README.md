# CacheTide 实现

这里集中保留既有 CacheTide 原型的数值实现。目录重构不表示已经验证它与最新论文设计的全部细节一致。

| 文件 | 用途 |
| --- | --- |
| `adapter.py` | 候选门控、各层校正器、发布映射的组合 |
| `structure.py` | 共享码与逐 head 私有码结构 |
| `state.py` | 原地发布转换与锚码状态 |
| `read.py`、`kernels/reader.py` | 融合校正读取及 Triton 内核 |
| `session.py` | 校正读取参与事件追加、FIFO、发布更新 |
| `precision.py` | 稳定的发布组合与锚码转换 |
| `prepare.py`、`features.py` | 物品特征与共享表准备 |
| `publication.py` | 发布参数、全物品表和序列化 |
| `calibration.py`、`calibration_cost.py` | 局部/联合校准及计算账 |
| `backend.py` | 推理配置、后端适配 |
| `cost.py`、`runtime_cost.py` | 方法和执行成本 |
| `compact/`、`refinement/`、`lifecycle/` | 既有紧凑表示、校准修正和状态管理组件 |

新的可复用方法实现放在这里；尚未确定的变体、扫描、绘图和队列放在
`experiments/<实验名>/`，经确认后再提取公共实现。历史 prototype 的评测与诊断脚本位于
`experiments/archive/query_relay`、`cachetide_runtime`、`compact_read` 等目录。
