# Figure 9 RecFlow 实验入口

**2026-10-30 02:01 用户已明确授权并启动正式全量评测。** tmux 为 `recflow-figure9d-formal`，输出在 `runs/formal_max`；当前运行情况见 [STATUS.md](STATUS.md)。以下准备记录保留原时间状态。

本目录准备 Figure 9d 的 RecFlow-Max 方法比较。正式全量评测需要另行显式启动；小规模程序不会自动派发后继任务。已存在的八个 Max 模型构成七条发布边，训练已经结束。本目录不训练推荐主模型。Figure 9c 需要对应的 RecFlow-Large 发布模型，不能把 Max 结果代作 Large。

2026-10-30 准备已完成：47 项 CPU 检查、2 项实际 CUDA Graph 小模型检查及真实 Max 的合批/独立执行对照通过。七边各固定 64 用户、29 配置、完整六档校准均已完成，共 203 点、2,934 请求、33,530,496 个方法候选分数。正负结果完整保留于 `runs/prepared_results/all_203_points.csv`。64 用户的七边 Rebuild−Reuse NDCG@10 区间均跨零，这些数据只用于工程验算和计时，不作为质量趋势的正式证据。

最终执行建议为 **eager、FP32/TF32 关闭、候选 chunk256、四卡用户分片**。CUDA Graph 通过数值检查，但实际 Max 样例读取和追加更慢，因此未选用；完整对照保留在 `runs/graph_profile_e4_uid246.json`。全量 30,000 用户×七边×29 点的四卡估算中心约 **26.9 小时**，建议预留 **34–38 小时**，不含资源被其他任务长期占用的等待。原始时间拆分与敏感性见 `runs/prepared_results/runtime_estimate.json`。默认输出约 12.4 GiB 用户结果外加校准权重和日志；当前无需清理原数据。

## 固定比较

沿用最新 `experiments/figure9max` 的 29 个自然配置：Reuse、Rebuild，11 个 Layer 区间，Tail/Deviation 各 64/128/256/512/768 条历史，六档 Alignment。Alignment 为各层各头 K/V 分别独立的残差 MLP，用户数/宽度分别为 32/40、64/64、96/96、128/128、192/160、256/184；固定八轮、8192 行 minibatch、AdamW。它只拟合发布前旧/新模型的 KV 对，不使用评测标签或质量选择 checkpoint。

代码适配真实 RecFlow 外部五字段特征编码、16 层/320 维/10 头、SiLU HSTU-reference 和相对位置偏置。没有套用旧 Yambda 的 ELU 算子或费用公式。历史上限始终为 **1024**。

## 时间与人口

七个目标评测日为 D29、D30、D31、D32、D33、D34、D36；源和目标 checkpoint 来自已接受的 `recflow_continuous_chain/chain_manifest.json`。每条边独立以源模型构造日初缓存、独立校准，修复结果不传给下一条边。

每个用户在第一项历史消费者之前修复一次（消费者可以是请求，也可以是行为追加）。随后每个方法保留自己的缓存，用当前模型追加真实行为，并逐事件执行 FIFO。候选只临时读取，不写入历史。完全静默用户保留在人口内，但不收费为一次修复。末请求之后的行为、无请求用户的行为仍维护。

同一时间戳的所有请求先读取之前的历史，然后才追加该时间戳的所有行为。保持原数据的曝光时间可见性假设：数据没有实际反馈到达时间，因此这不是对线上反馈延迟的验证。

质量人口为原 train 的固定 30,000 用户；这是同用户后续时间评测，不是未见用户泛化实验。校准固定取原 cache_fit 的 256 用户，原 validation/cache_validation 保持隔离。人口顺序使用固定随机种子，不按模型效果或未来活跃程度筛人。小规模评测使用该人口的固定前缀，静默用户也保留。

## 指标、费用与保留

每个请求评分完整候选集合。NDCG@10 为主指标，Recall@10 为辅助；同时保存 @5/@20。只有记录为正的候选获得增益；未知反馈保持 -1，不当作已知负反馈。无正反馈请求的 NDCG/Recall 未定义，保留覆盖计数，不填零进入平均。因此指标描述的是已观察正反馈的候选排序，而非未知候选的完整真实相关性。

按请求宏平均；配对 bootstrap 的采样单位是完整固定 UID 集合，含无请求用户。恢复率为 `(method−Reuse)/(Rebuild−Reuse)×100%`，不裁掉负数或超过 100% 的值；分母区间跨零则标记不稳定。跨边报告逐边恢复率的等权平均，全部七边保留，不按新结果重选边。旧 RecFlow 逐请求历史快照的 Reuse/Rebuild 分数不能充当本协议分母。

费用为独立方法的新增解析 FLOPs / 同人口触发一次 Rebuild 的 FLOPs。Alignment 包含教师源/目标缓存、归一化、实际 minibatch 的前反向和优化器，以及部署映射；Deviation 包含选择；非前缀 Layer 的旧隐藏边界字节单列。共同已有源缓存、普通评分/维护另列。实验中的合批或共用计算不会抵扣某个独立方法的费用。CUDA Graph 的末块候选补齐及逐事件额外 setup 单独入普通计算账，不算更多真实候选。

每用户原始输出保存所有请求的六项指标、Top20 的 ID/标签/分数和工作量/费用；小规模可用 `--full-scores` 保留全部分数。主结果、反例及失败日志都保留。中断可按用户恢复，不重复完成用户。输出锁和明文配置检查用于防止写错目录；**没有文件哈希或哈希保护**。

## 命令

在仓库根目录使用 `/home/gkl/miniconda3/bin/python`，环境为 `PYTHONPATH=src:.`。

```bash
PYTHONPATH=src:. /home/gkl/miniconda3/bin/python -m experiments.figure9recflow.run prepare
PYTHONPATH=src:. /home/gkl/miniconda3/bin/python -m pytest -q experiments/figure9recflow

# 小规模示例；诊断缩短的校准会明确标记 diagnostic_only。
PYTHONPATH=src:. /home/gkl/miniconda3/bin/python -m experiments.figure9recflow.run benchmark \
  --output experiments/figure9recflow/runs/example_benchmark \
  --edges 4 --gpus 0 1 2 3 --users 16 --fit-users 8 --fit-epochs 1
```

正式入口为 `run formal`，要求完整 30,000 用户和 256 用户/八轮校准。GPU 列表、候选 chunk 和 `--cuda-graphs` 是显式执行选项；CPU 预取默认每卡两个进程。正式启动前采用经过实际模型数值对照和计时的执行配置。不要把示例 benchmark 当正式结果；不要自动恢复任何旧实验队列。

## 文件

- `inputs.py`：固定人口、mmap、完整请求、严格时序。
- `cache_ops.py`：原生缓存和四类修复、持久追加、共享读取。
- `alignment.py` / `cost.py`：MLP 校准及独立方法算术账。
- `engine.py` / `graph_ops.py`：29 方法合批及固定形状 CUDA Graph。
- `run.py`：准备、校准、多卡分片、持久输出、汇总和显式恢复。
- `analysis.py`：配对 UID 区间、恢复率及跨边汇总。
- `gpu_check.py` / `profile_graph.py`：实际 Max 对照和预热速度检查。
- `estimate_costs.py`：全人口解析工作量和自然费用，纯 CPU。
- `runtime_estimate.py` / `export.py`：全人口耗时外推及完整 203 点导出。
- `runs/preparation/preparation.json`：完整窗口和覆盖账。

原论文、旧实验代码和旧结果不因本目录的运行而改写。
