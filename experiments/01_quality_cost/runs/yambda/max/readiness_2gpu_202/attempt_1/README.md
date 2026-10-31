# Yambda-Max 双卡已就绪（2026-10-22）

本轮完成跑通、提速与排期预估，**完整人口质量及五边正式队列尚未启动**。用户已确认使用 **g202 物理 GPU 2/3（两张 A40）**；201 实际仅有 GPU0/1。GPU0/1 的 RecFlow 和 201 的 Large 队列均未改动。原 Large 执行源码 hash 前后相同。

## 可直接使用的设置与预估

- Max 实体为 **16L/H320/10头/上下文1024**；六个 V0–V5 checkpoint 均为最终正式完成状态，无须重训。
- 两卡按用户并行，同边各方法独立持有因果状态；统一合并全部原始分数后计算 AUC、恢复率及1000次配对用户 bootstrap。
- **append batch256、query batch16、选择器候选固定16**；CUDA Graph、相同原生写入共享、合批映射沿用；NUMA1；`PYTORCH_ALLOC_CONF=expandable_segments:True`。训练batch仍为模板8/4/8，优化器、epoch及目标未改。
- 测试人口 **196,363**；另有512 fit／16 validation，与测试分开。V0→V1 中89,121人贡献891,487条反馈。首次角色从 Max 20万人池独立建立，后续五边只复用身份；每边重新校准六点及选层。没有复用 Large 名单或跨边纠正权重。
- **每条边预计约8小时，建议排期8–9小时；五条顺序边约40–45小时。** 这是当前双卡离线评测进度预估，后续边尚未实测；GPU0/1上的RecFlow以及CPU/I/O共享仍在，不作论文受控时间或在线吞吐证据。
- 每卡4096人原子提交一块，预估约18–20分钟；中断最多重做各卡未提交的一块。已完成校准点可直接复用。启动前先排查同一队列／子进程，禁止重复启动。

## 实测依据

同一256人、24点输入，两卡均完成冷／热重放。以下取热重放较慢一张卡作为双卡完成时间；每卡128人。

| append | query | 双卡秒数 |
|---:|---:|---:|
|128|16|35.45|
|64|16|36.50|
|256|16|35.13|
|128|32|35.39|

四组速度接近，256/16约快0.9%；随后以真实六点 Max 权重扩大到 **2048人／9216条反馈／24点**。GPU2冷／热293.86/291.01秒，GPU3为268.11/265.84秒；双卡热重放完成时间291.01秒。样本两卡反馈量4863/4353不同；正式分块约447,841/443,646条反馈，更均衡。按正式两卡人数外推纯质量7.76小时，按反馈量归一外推7.53小时；再预留约20分钟给输入、校准、保存和全量统计，并给排期留余量。最初双卡六点校准实测296.03秒，V0→V1已保存，可在后续正式运行时复用。

实测张量峰值约21.12GiB/卡，PyTorch保留显存峰值31.02/30.92GiB；nvidia-smi观测峰值31.53/31.43GiB，每卡尚有约13.5GiB余量。该结果覆盖2048人的冷／热混合形状，尚非完整人口长运行。实际只需至多41个固定全窗图形状，当前见到39个；短历史走既有有界eager路径。

串行／合批真实分数最大logit差 **1.43e-6**，不同batch最大差9.54e-7。所选设置再次完成32人端到端链路：24点、1000次UID bootstrap、完整FLOPs、附加内存、CSV／LaTeX均已输出。换新进程后四块全部通过hash并跳过，0次GPU建图，分数和标记hash完全不变；已完成六点拟合也实际验证了跳过。最终核心推理AST与测速时相同。针对检查6 passed（含CUDA Graph权重生命周期），联合CPU检查9 passed/2 CUDA skipped，Ruff F/E9通过。

## 实验协议与证据

24点：Reuse/Rebuild；Layer 2/4/8/12层；Tail 10/20/40/80%；Deviation/Query 10/20/40/60%；Q-only fit128/256/512；History-conditioned fit64/128/256。层段只在发布前fit教师数据上选择。五条边均为独立相邻版本：重新建立各自旧版原生缓存，边内正常因果读写，跨边不继承修正状态。Query-only保持原生写入。暂缓CacheTide。

全量CPU新增费用账已保存：Layer2=13.21%、H256=70.88%×一次Rebuild，覆盖无反馈用户和末次评分之后的维护。CPU预检共同append形状使用128；新增费用的首次消费／查询形状不受append上限变化影响，正式两卡会按256重新计算完整native账。恢复率按全局AUC差值计算，保留负值；区间条件于本次固定权重。

| 边 | 合法测试用户 | 反馈人数 | 反馈条数 | 窗口 |
|---|---:|---:|---:|---|
|V0→V1|196,363|89,121|891,487|[231,245)，E14|
|V1→V2|196,363|89,406|865,122|[245,259)，E14|
|V2→V3|196,363|90,387|872,163|[259,273)，E14|
|V3→V4|196,363|91,667|887,280|[273,287)，E14|
|V4→V5|196,363|94,633|899,616|[287,301)，**E14_partial**|

最后一条包含部分源日300的70,628条反馈，保留全部可用数据。以上是输入覆盖，不是全量模型质量。小面板 AUC 只用于跑通链路，不能填入最终图。

主要证据：`summary.json`、`population_preflight.json`、`parallel_assignment_preflight.json`、`cpu_preflight.json`、`full_cost_preflight.json/csv`、`initial_calibration_wall.json`、`sweep_256/`、`extended_2048/`、`canary_final/{summary.json,report.md,table.csv,table.tex}`、`final_resume_validation.json`。`source_readiness_probe/`保留测速时入口，最终报告／计数／GPU限制改动后重新跑通，数值推理函数AST相同。

## 用户后续下令时的正式入口

仓库：`/home/gkl/work/evokv`。下例**尚未执行**；`--edges`编号1–5分别对应V0→V1至V4→V5，可只指定需要的边。

```bash
/home/gkl/miniconda3/bin/python scripts/max_baselines_2026_10/launch.py --edges 1 2 3 4 5
```

入口会核验GPU2/3空闲、原运行锁及冻结输入／代码／模型，自动创建独立tmux socket `yambdamax202`、session `max-baselines`，绑定GPU2/3及NUMA1，保存启动和恢复命令。全局队列目录为`experiments/01_quality_cost/runs/yambda/max/queue_2gpu_202/attempt_N/`，各边输出在本目录`edge_v{old}_v{new}/`。每边的源配置为`experiments/01_quality_cost/configs/baselines_yambda_max_v{old}_v{new}_2gpu_202.json`。

```bash
tmux -L yambdamax202 attach -t max-baselines
/home/gkl/miniconda3/bin/python scripts/max_baselines_2026_10/launch.py --resume-manifest <队列configuration.json>
```

只在原队列及GPU子进程退出后恢复，断开客户端无需重启。质量分块、每个校准点、配置／代码／checkpoint／请求及分数hash均保留。正式队列运行期间保持相关执行源码、配置及输入冻结。
