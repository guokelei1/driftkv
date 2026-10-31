# 命令入口

`evokv.py` 是薄分发器：在仓库根目录运行 `python scripts/evokv.py --help` 查看命令，
在命令后加 `--help` 查看原流程参数。训练、数据、评测及方法实现在 `src/hstu_kvcache`；
实验编排在 `experiments`。

常用命令：`train-yambda`、`train-recflow`、`evaluate-yambda`、`evaluate-recflow`、
`prepare-yambda`、`prepare-max`、`download`、`figure9a`、`figure9max`。

多卡训练仍可通过 torchrun 执行模块，例如：

```bash
torchrun --standalone --nproc-per-node=2 --module hstu_kvcache.training.recflow.train_feedback --help
```

以上多卡调用需要先安装项目（`python -m pip install -e .`），或设置 `PYTHONPATH=src`。
本目录不再存放方法实现或历史实验队列。见[代码索引](../docs/code_layout.md)。
