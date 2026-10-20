# 本轮代码入口

工作定义和恢复状态在 [总计划](../../docs/system_study_2026_10/README.md) 与
[STATUS](../../docs/system_study_2026_10/STATUS.md)。旧会话、校准和推理核通过 import 复用，
这里只增加本轮需要的薄入口。

| 文件 | 用途 |
| --- | --- |
| `inventory.py` | 只读核查现存模型、候选、面板和角色；不加载全部模型或训练 |
| `events.py` | 合并真实跨用户事件，保持严格过去与版本顺序，分离到达时间 |
| `queueing.py` | 单 worker 到达、等待、执行与完成计时；行为标记不产生虚构 GPU 工作 |
| `prepare_trace.py` | 从已有32用户面板生成两种任务的事件摘要与恢复配置 |
| `model_replay.py` | 真实跨用户 GPU 会话，对照原逐用户路径的评分、缓存、游标与操作账 |
| `arrival_replay.py` | 使用真实 GPU 执行时间的有限到达回放，分别记录首推荐、热推荐与发布 |
| `phase_probe.py` | 对实际轨迹拆分历史推进、码转换和评分；比较已有行块参数，插桩耗时与到达测量分开 |
| `budgeted_rebuild.py` | 按发布前活跃度／估计重建成本选用户；首次推荐重建，后续保留独立状态 |
| `cost_screen.py` | 按真实历史、padding 和操作形状复现成本，再预筛读取／校准预算分支 |
| `scale_cost.py` | 拆分固定校准／共享准备与逐用户服务，求条件规模阈值和目录／副本／容量敏感性 |
| `workload_mix.py` | 核对30k真实原生操作形状与FLOPs，给时间模型提供工作量权重 |
| `model_group_probe.py` | Medium/Large/Max × V2/V5的原生算子计时；不推断跨模型纠正质量 |
| `batched_conversion_probe.py` | 同8个真实用户的逐个／合批反锚和物化，核对数值、存储与调用数 |
| `fused_conversion.py` | 两次Triton调用完成跨层原地码转换；运行时长度避免逐历史编译 |
| `batched_session.py`、`batched_session_probe.py` | 真实独立用户按形状合批，Rebuild同样优化，旧oracle核对与交错计时分开 |
| `fused_query_projection.py`、`fused_query_projection_kernel.py` | 将Query投影并入历史读取核，减少每层小算子调用，不改训练 |
| `shared_timing.py` | 实测发布级共享准备与搬运，并核对旧校准时间的包含范围 |
| `fit_size_v4.py`、`fit_size_evaluate.py` | 固定V3来源与开发评价，比较V4增量拟合人数；不用于跨模型纠正 |
| `fit_size_ranking.py` | 复用五档已有评分，核对同一候选集合内的排序与用户配对区间；不新增GPU实验 |
| `time_batch_shapes.py`、`time_model.py` | 枚举真实30k联合操作/合批形状，用实测阶段系数核算完整时间与分页敏感性 |
| `graph_compute.py`、`graph_compute_probe.py` | 双方同等固定形状图复用；动态码前缀作为GPU输入，真实会话另计staging/图空间/capture |
| `joint_batch_probe.py` | 同一V4原Adam八人批的等目标用户合批；只比较loss/梯度/执行时间，不更新冻结权重 |
| `joint_batch.py`、`joint_batch_adam_probe.py`、`joint_fit_comparison.py` | 保留原优化器批次、四步续写和选轮规则的joint合批；一次更新检查与完整V4执行对照，实验权重不替换候选 |
| `maintained_source_probe.py` | 已维护源状态的V4准备计时入口；重新计算全部教师与2名idle用户当前码，逐项计快照/搬运/保存，CPU准备已过，GPU尚待测量 |
| `compare_controls.py` | 将两个独立 GPU 入口的原生对照按完整任务身份匹配 |

从仓库根目录运行（输出目录必须尚不存在）：

```bash
CUDA_VISIBLE_DEVICES="" python scripts/system_study_2026_10/inventory.py \
  --output results/system_study_2026_10/inventory/initial_2026_10_19
CUDA_VISIBLE_DEVICES="" python scripts/system_study_2026_10/prepare_trace.py \
  --config configs/system_study_2026_10/service_smoke.json \
  --output results/system_study_2026_10/service/trace32
PYTHONPATH=scripts:src CUDA_VISIBLE_DEVICES="" python -m pytest tests/system_study_2026_10 -q
```

GPU 运行的实际命令、配置、PID及终态见 STATUS；已完成目录不能原地重跑。
早期到达 pilot 只使用四个用户、压缩到达时间与全版本共驻模型，
可检查真实队列和发布干扰，不能作为生产 p99、稳定吞吐或驻留容量结论。
大规模运行前记录该次实际输入、工作负载、预算和 GPU；不用本文件代替运行配置。
