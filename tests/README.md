# 保留的检查

保留当前数值、因果、指标、模型/数据和Motivation公共实现的检查。
已退休实验的专用测试随其代码删除；不增加镜像实现或生产边界测试。

| 范围 | 检查 |
| --- | --- |
| rank8共享code与Key/Value读取 | [test_lightweight_fullitem_kv.py](design_one/test_lightweight_fullitem_kv.py)、[test_lightweight.py](design_one/test_lightweight.py) |
| 当前局部拟合 | [test_lightweight_local_kv.py](design_one/test_lightweight_local_kv.py) |
| 继承flag范围与FIFO维护 | [test_support_mask.py](design_one/test_support_mask.py) |
| 生命周期与按需编码 | [test_lifecycle.py](design_three/test_lifecycle.py)、[test_lightweight_lazy.py](design_three/test_lightweight_lazy.py)、[test_lightweight_lifecycle.py](design_three/test_lightweight_lifecycle.py) |
| 原生追加、时间语义与attention | [test_append_only_kv.py](test_append_only_kv.py)、[test_state_transition.py](test_state_transition.py)、[test_yambda_incremental_time_contract.py](test_yambda_incremental_time_contract.py)、[test_attention_backends.py](test_attention_backends.py) |
| Motivation和数据/训练指标 | `selective_recompute/`、`read_correction*/`、其余保留模型与数据检查 |

CPU检查：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q
```

pytest使用importlib导入模式，以允许不同目录的同名测试模块。
2026-10-07继续移除退休分支专用检查，并将生命周期检查改为当前full-item KV方案。
CPU检查176通过、7项CUDA检查跳过；未启动GPU实验或基础训练。
