# 保留的检查

保留当前数值、因果、指标、模型/数据和Motivation公共实现的检查。
已退休实验的专用测试随其代码删除；不增加镜像实现或生产边界测试。

| 范围 | 检查 |
| --- | --- |
| 新系统研究的事件因果、会话、单worker计时与预算成本 | [system_study_2026_10/](system_study_2026_10/) |
| 当前Implementation共享准备、融合读取、单缓冲转换与集成 | [implementation_2026_10/](implementation_2026_10/) |
| 新 Query 门控、编译递推、纠正写入与简单校准 | [query_relay_2026_10/](query_relay_2026_10/) |
| rank8共享code与Key/Value读取 | [test_lightweight_fullitem_kv.py](design_one/test_lightweight_fullitem_kv.py)、[test_lightweight.py](design_one/test_lightweight.py) |
| 当前局部拟合 | [test_lightweight_local_kv.py](design_one/test_lightweight_local_kv.py) |
| 继承flag范围与FIFO维护 | [test_support_mask.py](design_one/test_support_mask.py) |
| 生命周期与按需编码 | [test_lifecycle.py](design_three/test_lifecycle.py)、[test_lightweight_lazy.py](design_three/test_lightweight_lazy.py)、[test_lightweight_lifecycle.py](design_three/test_lightweight_lifecycle.py) |
| 第4.4节旧码/编码版本、部分与强制刷新、raw门控 | [test_session.py](partial_refresh_2026_10/test_session.py) |
| 原生追加、时间语义与attention | [test_append_only_kv.py](test_append_only_kv.py)、[test_state_transition.py](test_state_transition.py)、[test_yambda_incremental_time_contract.py](test_yambda_incremental_time_contract.py)、[test_attention_backends.py](test_attention_backends.py) |
| Motivation和数据/训练指标 | `selective_recompute/`、`read_correction*/`、其余保留模型与数据检查 |

CPU检查：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q
```

pytest使用importlib导入模式，以允许不同目录的同名测试模块。
2026-10-07继续移除退休分支专用检查，并将生命周期检查改为当前full-item KV方案。
CPU检查176通过、7项CUDA检查跳过；未启动GPU实验或基础训练。

2026-10-13部分刷新原型新增8项数值与因果检查；CPU全套184通过、7项CUDA跳过。
GPU全量刷新、融合误差、原生KV不变和理论成本账证据见[本轮报告](../results/partial_refresh_2026_10/report.md)。

第二轮新增support稀疏编码、非零固定归一化因子化与static/no-cache行为、跨用户准备计账
和三发布生命周期检查：[test_sparse.py](partial_refresh_2026_10/test_sparse.py)、
[test_factored_encoder.py](partial_refresh_2026_10/test_factored_encoder.py)、
[test_factored_session.py](partial_refresh_2026_10/test_factored_session.py)。全套CPU197通过、7CUDA跳过，
GPU完整质量和逐项算术账见[第二轮报告](../results/partial_refresh_2026_10/round2/report.md)。
