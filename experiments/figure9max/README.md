# Figure 9：Yambda-Max 对比方法

2026-10-27 目录迁移：KV Map 实现位于 `src/hstu_kvcache/methods/kv_map/max.py`，
成本位于同目录 `cost_max.py`，实验编排仍在本目录。两组评测均已结束。
本目录 `runs/` 的原始记录保持不变；下面旧输出目录的执行命令是历史记录，
迁移后新实验须指定新输出目录并重新生成源绑定，不直接恢复冻结的旧 run。
见[迁移说明](../../docs/code_layout.md)。


2026-10-26 用户已授权：检查配置、数值和速度后，直接启动四张 A40 的五边全量实验。准备与正式任务均通过独立 tmux 启动。正式任务于 2026-10-26 22:17（Asia/Shanghai）启动，会话 `figure9max-formal`，输出 `runs/formal`。

本目录从已完成的 `experiments/figure9a` 复制核心执行代码，调整为实际 Max 的 **16 层、320 维、10 头、历史上限 1024**。保留共享底层数值实现的引用；不复制模型、数据或 Large 结果。模型端点是 Max 自身的 V0–V5，六个已训练好的 foundation 模型不重新训练。

## 固定评测协议

沿用 Max 已冻结的 196,363 名 quality 用户及原 fit/validation 角色；每条边的拟合独立，只用发布前历史。五条独立相邻边为 V0→V1 至 V4→V5，窗口分别为 `[231,245)`、`[245,259)`、`[259,273)`、`[273,287)`、`[287,301)`。最后一条保留 `E14_partial` 标记。反馈请求数预期为 891,487／865,122／872,163／887,280／899,616，并由原始数据重新核对。

各方法从共同的旧版原生缓存初始化，在首次历史消费者之前修复一次，边内按当前模型因果追加和 FIFO 淘汰；没有跨版本继承校准权重或状态。只测对比方法和 Reuse/Rebuild，不运行 CacheTide。

每边 29 个配置，共 145 个方法／边组合：

- Reuse、Rebuild 各一个。
- Layer 使用 2／3／5／6／10／13 层，约覆盖原 Large 的层数比例。1-based 闭区间为：`1–2, 6–7, 11–12; 1–3, 6–8, 11–13; 1–5, 9–13; 1–6; 1–10; 1–13`，共 11 个。中间区间保留真实旧模型输入边界，并单列其字节成本。
- Tail、Deviation 各重算 64／128／256／512／768 个位置，短历史取实际长度与档位的较小值。
- KV map 六档 `(拟合用户, MLP 隐藏宽度)`：`(32,40), (64,64), (96,96), (128,128), (192,160), (256,184)`。每层每头的 K/V 分开拟合 `32→W→32` 残差 SiLU MLP，共 320 个网络。沿用 8 个完整 epoch、8192 行 minibatch、AdamW 和 seed17。所有容量先固定，再计算实际 FLOPs；不用测试 AUC 选择配置。

图上的横轴为完整新增解析 FLOPs／一次 Rebuild FLOPs，包含教师及源缓存准备、标准化、全部训练前后向、loss、AdamW、固定拟合诊断和全人口部署映射。运行秒数只用于排期。`planned_costs.json` 保留逐边成本账本，`speed_report.json` 保留四卡真实模型速度与数值检查。

## 执行与保存

四张 A40 按 UID 切分每条边。每卡同时维护全部方法；rank0 为该边拟合一次六档映射，再共享给其他卡。各边顺序执行，已完成边的 CPU 统计与下一边 GPU 评测重叠。使用已有 CUDA graphs、方法合批、共享 Layer 前缀和融合稀疏 attention；最终数值与独立 dense 参考逐请求核对。

每 30 分钟原子保存一次，保留最近两代压缩检查点，只存已完成游标及精确 AUC／配对 UID bootstrap 所需的分数、标签和 UID。每条边统计成功后清理这些临时数组，保留配置、来源指纹、映射权重和核心结果。完成时输出 `runs/formal/summary.json` 和 `results.csv`。1000 次配对 UID bootstrap 沿用原协议；单 seed 的拟合不代表训练随机性方差。

```bash
# CPU 输入、FLOPs 预检查（五边并行）
python experiments/figure9max/prepare.py

# 四卡有界检查；只用于数值和速度
python experiments/figure9max/run.py launch --mode benchmark \
  --output experiments/figure9max/runs/benchmark \
  --session figure9max-benchmark --users 256 --batches 1

# 已获用户授权；正式首次执行及中断恢复用同一入口
python experiments/figure9max/run.py launch --mode formal --authorize \
  --output experiments/figure9max/runs/formal --session figure9max-formal
```

恢复前确认原任务已经退出。恢复时校验配置、执行代码、角色、模型和输入指纹，复用已完成的边及检查点。运行期间不修改冻结的数值源码或配置。论文与 TOML 不在本次任务内。

## 已完成的启动前验证

11 项 CPU 核心检查通过。四卡各 256 个用户、覆盖边 1／2／3／5 的真实模型数值与标量检查点续跑通过，最大分数误差 9.54e-7；正式使用单用户、29 方法合批，评测峰值显存 18–22 GiB。额外四用户合批校验在 Max 上 OOM，已保留短日志并移除该额外校验；正式 user_batch 始终为 1。

旧 Max 输入清单的请求数包含 fit/validation 人口；本次以 quality 人口单独核对为 4,415,668 条，详情见 `feedback_coverage.json`。这只修正全量覆盖断言，不改变小规模校验的数值设置与输入。基准原始配置保存在 `runs/benchmark/configuration.json`，速度报告显式核对这项差异。

全量五边 CPU 事件账本与四卡实际训练的行数、步数、参数及 FLOPs 相符。六档 Map 五边平均成本约 6.31／10.01／14.95／19.93／25.05／29.02%，均在 5–30% 内。启动前排期中心约 18.1 小时，依据正式进度更新；`speed_report.json` 是启动前速度报告，实际启动信息见 `runs/formal/start_record.json`。
