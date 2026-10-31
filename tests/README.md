# 测试入口

测试按代码职责组织。本次目录重构迁移现有检查，没有新增测试用例。
此前授权删除的旧 Medium 系统研究、部分刷新专用测试保持删除。
另移除了一项只核对已删除历史 gap manifest 的旧计划测试；数值与因果检查保留。

- 根目录：HSTU、注意力、原生 KV、数据与训练协议。
- `methods/baselines/`：Layer、Tail、Deviation 等局部重算。
- `methods/cachetide/`：状态转换、候选门控、融合读取、持续会话。
- `methods/read_correction/`：读取校正基线。
- `evaluation/`：基线执行、断点与统计。
- `experiments/`：第三章探索与历史 Max 基线的现有检查。
- Figure 9 检查保留在 `experiments/figure9a/`、`experiments/figure9max/`。

按实际改动选择检查，例如：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_state_transition.py tests/methods/cachetide
CUDA_VISIBLE_DEVICES="" python -m pytest -q experiments/figure9a experiments/figure9max
```

`python -m pytest -q` 收集 `tests/` 下的全套检查；Figure 9 需显式提供目录。
纯 CPU 检查会跳过要求 CUDA/Triton 设备的项目。
