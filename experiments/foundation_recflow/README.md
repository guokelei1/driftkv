# RecFlow：202 两卡先 Large、后 Max，V0–V5

2026-10-21 最新状态：**用户已授权启动，正式Large→Max完整链已在202两卡运行。**
使用独立tmux：socket `recflow202`、session `recflow-foundation`，无客户端连接时仍运行。
[当前运行目录](runs/chain_2gpu/attempt_1/)、[启动与恢复命令](runs/chain_2gpu/attempt_1/launch.json)、
[主日志](runs/chain_2gpu/attempt_1/launcher.log)。该安排替代先前只跑Large V0/V1的范围。
配置`training_authorized=true`已经冻结；运行期间不修改数值源码、设计／运行配置和数据。

## 一份配置，一个顺序入口

- [串行配置](configs/chain_large_max_2gpu_202.json)
- [入口脚本](../../scripts/recflow/run_foundation_chain.py)
- [按六版短测计算的预算](planning_chain_two_gpu.json)
- [模型／数据设计](../docs/recflow_model_design_2026_10_21.md)

202 使用 NVLink 直连的物理 GPU 0／1。顺序固定：

```text
Large：从头训练 V0 → D19 更新 V1 → D20 更新 V2 → … → D23 更新 V5
      → D20–24 五条相邻版本配对评价
Max：  独立从头训练 V0 → D19 更新 V1 → D20 更新 V2 → … → D23 更新 V5
      → D20–24 五条相邻版本配对评价 → 汇总并退出
```

Max 独立初始化；两档内部的模型、AdamW 两份矩和真实步数连续继承。
每个版本默认两轮，进入 V1 时学习率从 2e-4 改为 5e-5，V1–V5 保持 5e-5。
seed17，weight decay1e-4，gradient clip1.0。

只看计划、不启动进程：

```bash
python scripts/recflow/run_foundation_chain.py --plan
```

当前运行已由以下入口启动；不要重复执行。原进程确认退出后才使用`--resume`：

```bash
python scripts/recflow/run_foundation_chain.py
```

默认输出 `experiments/foundation_recflow/runs/chain_2gpu/attempt_1/`。
中断后同一入口加 `--resume`。启动时检查两张卡空闲及剩余磁盘，防止同一输出重复运行；
恢复时清理本次未提交临时文件。配置、来源、权重和数据绑定须一致；已完成训练端点恢复后
直接越过，完整评价分片复用。Large 失败会停止流程；Max 等 Large 的训练和评价成功后开始。
完成后生成顶层 `summary.json`／`report.md`，逐边结果和所有指标保留在各档 `evaluation/`。

## 断网后的查看与恢复

断开SSH／Codex或关闭本地客户端不会结束tmux中的训练。主进程和子任务日志写入磁盘，
不依赖当前终端；启动shell还忽略SIGHUP。接回会话：

```bash
tmux -L recflow202 attach -t recflow-foundation
```

查看主阶段用`attempt_1/progress.json`，当前训练步数用`attempt_1/large/training/progress.json`，
进入Max后对应`max/training/progress.json`。任务退出时`launcher.exit`记录退出码。
网络断开不需要重启；若进程确实退出，保留原配置／数据／源码，确认没有旧进程后，
在新的detached tmux会话运行同一入口并加`--resume`。

## 模型与数据

| 设置 | Large | Max |
|---|---:|---:|
| 层数／隐藏维度 | 10／320 | 16／320 |
| 独立视频 embedding | 2,200,000 | 3,500,000 |
| 稳定未知视频哈希行 | 65,536 | 65,536 |
| 总参数 | **730,211,201** | **1,149,285,121** |
| 相对对应 Yambda 参数量 | +1.81% | +0.97% |
| 全局请求 batch／每卡 microbatch | 96／48 | 80／20（累积两次） |
| 两轮六版 optimizer 步数 | 79,236 | 95,082 |
| 实际训练形状短测峰值分配／卡 | 18.17 GB | 22.95 GB |
| 实际训练形状短测峰值保留／卡 | 30.24 GB | 28.58 GB |
| 全15种形状及恢复的最大保留／卡 | **31.43 GB** | **29.12 GB** |
| 冻结输入 | `data/processed/recflow_feedback_large_v1` | `data/processed/recflow_feedback_max_v1` |

两档均为10头、每头32维、最多1024条真实历史，与对应实体 Yambda 使用相同 backbone
结构；视频表和层数扩大到各自档位。两档使用同一固定初始人口和数据窗口，用户角色
保持 development26,277、admission3,291、final3,210；根训练实际有非空前缀的用户为23,668。
Max 已知视频包含 Large 的全部已知视频，两个冻结映射与 CPU 规划的哈希一致。
源数据就在202，无需再从其他机器复制。原始数据和 Medium 生成式链未改。

| 版本 | 完整拟合窗口 | 请求／轮 | 曝光反馈／轮 | 轮数 |
|---|---|---:|---:|---:|
| V0 | D1–18 | 3,023,919 | 15,515,819 | 2 |
| V1 | D19 | 161,078 | 833,795 | 2 |
| V2 | D20 | 154,410 | 795,375 | 2 |
| V3 | D21 | 151,335 | 776,072 | 2 |
| V4 | D22 | 157,312 | 806,088 | 2 |
| V5 | D23 | 154,968 | 794,688 | 2 |

每档合计 7,606,044 次请求访问、39,043,674 条曝光反馈访问。增量版本只拟合新一天。
保留全部合法正负曝光和稳定哈希目标，严格排除当前整组同时间戳反馈进入自身历史。
未达到 effective-view 条件不能称显式 dislike。admission 的既定细分预留不参与根训练或 final 评价。

## 每轮永久 checkpoint 与小时恢复

每档独立保留：

```text
fresh → v01 → v02 → v11 → v12 → v21 → v22 → v31 → v32 → v41 → v42 → v51 → v52
```

每份包含模型、完整 AdamW、各 rank RNG、数据进度、架构、父端点和来源绑定。
每档12份永久端点，两档共24份；`v0`–`v5` 的主端点固定为对应第二轮，使用硬链接别名。
约每小时在完整 optimizer 步边界滚动保存，保留最近两份；永久 epoch 文件也可恢复。
完整写盘、SHA256 完成后才提交标记，未完成文件不会充当恢复点。

每份完整文件约 Large8.76 GB／Max13.79 GB。24份永久端点加两档各两份滚动恢复
约 **315.8 GB**，建议留 **350 GB**。现有磁盘空间充足；预留量含临时写盘余量。
本次双卡完整恢复文件实测保存时间为Large约24.5秒、Max约38.4秒；
两档均已在原进程退出后，由新进程恢复并重建空的CUDA Graph缓存，后续对照loss一致。
[显存／保存／恢复证据](runs/memory_readiness/summary.json)记录来源和限制。

## 速度与排期

沿用已经优化的请求共享前缀、长度分桶、Triton、BF16计算／FP32参数与Adam状态、
实际使用 embedding 梯度行通信、直接梯度累积、CUDA Graph 和 fused AdamW。
优化器仍逐步更新完整参数表。先前同 batch 预热短测的实现优化约3.37倍，
见[历史实现对比](runs/large/speed_final_batch96/summary.json)。

本次两档都在空闲的202 GPU0／1完成六版全部真实输入形状的短测：每种形状取一个
代表批次，预热后测六步，按完整 epoch 的形状频率外推。
[Large 原始短测](runs/large/profile_2gpu_v0_v5/summary.json)、
[Max 当前micro20短测](runs/max/profile_2gpu_v0_v5_micro20/summary.json)。

| 工作量，均每版两轮 | Large 纯训练估算 | Max 纯训练估算 |
|---|---:|---:|
| V0 初次训练 | 2.82小时 | 4.91小时 |
| V1 | 10.64分钟 | 18.34分钟 |
| V2 | 10.29分钟 | 17.74分钟 |
| V3 | 10.07分钟 | 17.38分钟 |
| V4 | 10.65分钟 | 18.34分钟 |
| V5 | 10.50分钟 | 18.09分钟 |
| 六版合计 | **3.69小时** | **6.40小时** |
| 含启动、保存、评价的排期 | **约4.5–5小时** | **约8–9小时** |

两档串行纯训练约10.09小时，加25%余量约12.62小时，整体安排 **约13–14小时**。
这些是短测外推。正式全程、保存及FP32评价尚未联合计时；不同 epoch 的 shuffle、
GPU负载和I/O会带来变化。保存／评价时间预算包含在余量中，无独立实测结论。

## 显存风险与长期运行准备

设备每卡实际总显存约48.31GB。Max每卡40的初版保留峰值约40.96GB，已调整成每卡20、
分两次累积再更新，保持全局80、完整数据和optimizer步数。完整形状速度短测仅增加约4.3%时间。
Large保持每卡48、全局96。

两档各自检查了全部5种历史长度×3种候选容量（最高1024／32）；实际固定训练和测试窗口
最大候选数分别20／12。所有尾batch仍填充到固定microbatch大小，图缓存最多15种。
初次运行和新进程恢复后，各进行了120步混合形状重放：缓存数稳定，稳态保留显存没有继续增长，
loss／梯度保持有限。最坏峰值保留显存为Large31.43GB、Max29.12GB／卡，另给CUDA上下文、
通信和长时分配波动留下十几GB余量。保存前后GPU显存未增加，完整文件哈希检查后才用于恢复。

启动前检查两张分配卡空闲、剩余checkpoint所需磁盘和临时文件余量；按当前空间约684.5GB，
剩余需求加余量约339.5GB。主机可用内存约973GiB，完整CPU checkpoint staging有足够空间。
运行中保存100步间隔的进度，定期和epoch结束检查loss／梯度，遇到非有限值停止且不提交坏恢复点。
每轮永久保存、小时滚动保存及新进程恢复共同限制中断后的重复工作量。

这些是有界显存／恢复检查，正式13–14小时训练刚启动；尚不能称已完成整夜稳定性验证。
详细源文件、峰值和恢复对照见[汇总](runs/memory_readiness/summary.json)。
另以development输入检查了单卡FP32评测的最长1024／候选32形状，Large每卡batch48、
Max每卡batch32的保留峰值分别4.40／5.77GB，见[评测显存短测](runs/evaluation_memory/summary.json)；
该检查未读取held-out质量指标。

## 配对质量报告

每档的 V0／V1 在同一 D20 比较，V1／V2 在 D21，依次至 V4／V5 在 D24。
每条边覆盖全部合法初始 final-role 曝光，父子使用相同因果历史；测试用户不按结果筛选。
报告 ROC-AUC、正负类 PR-AUC（AP）、LogLoss、Brier、用户宏 AUC、曝光请求内
NDCG@1/3/5、known／hashed OOV 分层，以及1000次配对 UID bootstrap 的 AUC 增益区间。
主指标预定 ROC-AUC，保留每个退步项。顶层报告给出逐边指标和等发布权重的平均差值。

这组结果检验底座版本更新收益；缓存纠正实验另行安排。
曝光请求内 NDCG 与旧 Medium 的 uniform1000 NDCG 保留各自口径。
根据此批结果挑选今后指标仍属探索，需要另一个窗口确认。

2026-10-22按用户后续生成／检索任务偏好补做了[Large常见指标全量核对](runs/chain_2gpu/attempt_1/large/sampled_retrieval_full/report.md)：每请求1000个固定均匀干扰视频及已知观测正例，全部94,809个合法请求含无正例／OOV零分。Recall@100五条点估计均正、三条超过1%，等发布平均+0.94667%（UID95% [0.32784,1.58641]%），新版为匹配随机的2.107–2.272倍；NDCG@50只有一条正向、平均-1.56585%，推翻1024请求小面板的较好判断。推荐Recall@100作当前候选覆盖指标，保留NDCG的排序退步。它是现有BCE反馈头的采样检索诊断，不是原生自回归／全目录生成；未重训或修改底座。补测已迁到2/3并完成，0/1原Max训练继续运行。

历史 V0/V1 四卡、双卡及大 batch 探针保留原配置／summary，不改写成当前完整链实测。
当前默认入口已经换成本文的两卡完整链；旧 `run_large_v0_v1.py` 仅保留早期准备入口。
