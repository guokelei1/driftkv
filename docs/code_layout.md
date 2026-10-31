# 代码布局与迁移

2026-10-27，用户确认评测结束后执行目录重构。主索引在[仓库 README](../README.md)。

```text
src/hstu_kvcache/
  models/                  HSTU、注意力、原生 KV
  data/                    数据和人口/窗口/请求准备
  recflow/                 RecFlow 模型、反馈协议
  training/                Yambda、RecFlow 训练
  evaluation/              历史、轨迹、评测、成本、统计
  methods/
    baselines/             Layer、Tail、Deviation 等
    kv_map/                Large / Max 非线性 KV Map
    cachetide/             状态、校正读、会话、校准、内核
    read_correction/       较早的读取校正基线
  adaptation/              通用历史读与评分
  common/                  文件与实验记录工具
experiments/
  figure9a/、figure9max/     各自配置、评测编排和结果
  ch3_motivation/code/      第三章机制探索代码
  foundation_recflow/code/ RecFlow 训练链编排
  archive/                 按主题归档的早期探索
scripts/evokv.py            常用流程的薄入口
```

`kv_map/large.py` 和 `max.py` 保留原来 10/16 层的实际计算逻辑及各自成本默认值，
本次没有强行合并两套已验证的维度约定。所有核心代码的导入均指向 `hstu_kvcache`；
`src` 不依赖 `scripts` 或 `experiments`。

## 常用定位

- Yambda 训练：`training/yambda_fsdp.py`；RecFlow 训练：`training/recflow/train_feedback.py`。
- Yambda 模型装载/评测：`evaluation/yambda.py`；RecFlow 评测：`evaluation/recflow/evaluate_feedback.py`。
- 历史数据：`evaluation/history.py`；原生持续轨迹：`evaluation/trajectory.py`。
- CacheTide：见[逐文件索引](../src/hstu_kvcache/methods/cachetide/README.md)。
- 原 `unified_reuse_2026_09/common.py` 的哈希和原子记录工具：`common/artifacts.py`。
- 旧探索目录和代码的逐文件对应：见 [code_migration.json](code_migration.json)。

## 已有结果如何追溯

重构前基线提交为 `91fff42`（`clean1`）。已有 `runs/`、结果、模型、人口、输入、
冻结配置及其中的源码路径/哈希保持原样；没有把新源码伪装成旧实验执行过的源码。
迁移只调整代码位置、导入、入口与代码路径记录，旧计划中的源码哈希不会自动更新。

因此不要在重构后直接续跑冻结的旧 run。新实验使用新输出目录并重新生成源绑定；
需要严格检查旧记录时，在单独 checkout 中使用该记录对应的提交/源码快照。
`code_migration.json` 用于寻找当前实现，不是旧哈希的替代品。
Figure 9 的新源绑定覆盖 `src/hstu_kvcache` 全部模块和本实验入口，以包含迁出的间接依赖。

## 后续放置规则

稳定的方法、训练/评测能力和被复用的辅助函数放 `src`。
一次探索的变体、配置扫描、队列与分析放 `experiments/<主题>/`，避免再按日期创建公共脚本包。
`tests/methods`、`tests/evaluation` 和 `tests/experiments` 对应各自的职责；本次没有新增测试用例。

## 本轮验证

现有测试共覆盖 1309 项（含两套 Figure 9），1275 项通过，34 项因 GPU 或历史资产条件跳过。
分组运行后修复了迁移引起的源码绑定问题，全部失败项已重跑通过；另移除一项依赖已删除
历史 gap manifest 的旧计划检查。没有新增测试用例，也没有重新启动 GPU 实验。
九个常用命令的 `--help` 均通过；核心包在仓库目录外仅设置 `PYTHONPATH=src` 也可导入。
数据、结果、冻结配置和运行记录没有 Git 改动。
