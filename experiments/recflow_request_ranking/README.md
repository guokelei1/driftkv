# RecFlow 完整日志候选的请求排序

本目录独立处理这次新训练；状态入口见 [STATUS.md](STATUS.md)，方案见
[configs/proposal.json](configs/proposal.json)，实测和保留的失败见 `reports/`、`runs/`。
不修改论文，不沿用旧任务的模型、词表、人口或队列。

十个实际 Max checkpoint 已训练完成，路径、父版SHA、数据量和阶段耗时见
[模型清单](reports/model_chain_inventory.json)。原定十九窗已独立核对，
[全部结果](runs/version_chain_verification/attempt_1/all_windows.csv) 保留全部正反；
未见用户确认有两条明确损失。追加的训练分组未来请求九窗也已全部完成，
六条复用下降、两条反向、一条区间跨零，独立位于
[runs/temporal_cohort/attempt_1](runs/temporal_cohort/attempt_1)。完整交付见 [RESULTS.md](RESULTS.md)，当前无GPU任务。

## 固定任务和模型

每次请求使用真实曝光历史的最近 **1024 事件**，对该请求的**全部已记录排序阶段候选**评分。
主指标为请求平均的 **observed-positive NDCG@10**，辅指标 Recall@10，另报 @5/@20。
这是日志候选池排序，不是全物品库检索，也不是无偏在线效果估计。
曝光且有效观看为 1，曝光无效为 0，未曝光为未知 -1；未知在 NDCG 中仅有零观测增益。
零正例请求保留计数，其 NDCG 不定义。完整候选不按模型词表过滤、不抽样、不补正例。
分数相同按原始 video ID 排序；区间使用配对 UID 聚类 bootstrap，保持 request-macro 估计量。

正式模型由用户确定为 **RecFlow Max**：16 层、hidden 320、10 heads × 32，完整宽度 320 的物品表。
3,500,000 显式物品 + 65,536 固定哈希桶，配作者、两级类别和时长特征，
新实现总参数 **1,154,235,993**，其中物品表 **1,140,971,840**。
旧规模依据是 `experiments/configs/recflow_models_2026_10_21.json`；
新任务采用 HSTU-reference SiLU、相对位置偏置和独立候选查询，不声称等同于旧 ELU+1 权重。
pilot 为 2 层/128 宽/4 heads/64 维物品表，只用于学习和工程诊断，不替代正式 Max。
所有版本都保留 1024 历史；内部容量 1025 是为了额外容纳一个不写入 cache 的候选查询。

每个请求只编码一次历史。候选分块读取同一 KV，块之间没有候选注意力；
查询激活重算减少训练内存，listwise softmax 分母仍覆盖全部候选。
历史包含过去曝光的行为和时间间隔；查询不含当前曝光、反馈、rank_index 或阶段标签。
input/attention dropout 均为零。训练 BF16；原生质量和缓存比较统一 FP32 参数/计算/KV，关闭 TF32。

## 数据与时间线

预处理目录：`data/processed/recflow_request_ranking_v1`。
14 个 all_stage 日文件及 37 天 realshow 均已核对；最终阶段目录为 **stage_v2**。
初次严格转换因原始阶段外曝光失败，其 `stage`、日志和审计保留，不能当作最终输入。

- 1,572,217,303 原始阶段行，1,307,553,332 保留的排序候选行，3,308,233 原始请求。
- 13,721,842 条曝光与 realshow 的键、特征、行为逐字段匹配，零不一致。
- 依据实际编码和全部阶段标签审计，成员为 `0 <= rank_index < 3000`；3000/8000 是早期阶段哨兵。
  这条规则来自实际数据核验，不是凭官方 README 猜出的定义。
- 8,659 个请求存在阶段外曝光，独立于 effective_view 和模型结果整请求隔离。
  首请求在隔离前确定，不以当天稍后的请求替代。
- 38,242,198 去重历史事件；D26 历史已满 1024 的居民 7,375 人，
  train/validation/development/confirmation 为 5,197/734/694/750。

北京时间自然日、UNIX 毫秒、左闭右开。历史严格排除当前用户的整个相同时间戳块。
同时间事件按固定键排序，不声称是真实观看先后。
此前曝光反馈采用 request-time 回放假设；数据未提供完整行为到达时钟。
输入特征来自各原始记录的当时元数据，不用未来记录回填。

| 用途 | 数据日 | 日期 |
| --- | --- | --- |
| 历史积累 | D1–11 | 1 月 13–23 日 |
| Foundation 曝光 BCE 预训练 | D12–23 | 1 月 24 日–2 月 4 日 |
| Foundation 完整候选对齐 | D24–26 | 2 月 5–7 日 |
| 九次逐日真实更新 M1…M9 | D27–35 | 2 月 8–16 日 |
| 每版下一整天评测 | D27–36 | 2 月 8–17 日 |
| 最后一条边额外时间窗口 | D37 | 2 月 18 日 |

UID 使用固定 SplitMix64 分成 70/10/10/10；不按后续每天活跃交集选人口。
词表只使用 foundation 时间内的 train UID，整个版本链保持映射不变。
新物品和长尾映射固定哈希桶，仍全部参评；显式表覆盖率与碰撞在 vocabulary manifest 中报告。
新 Max 词表在词表构建语料上的显式频次覆盖 93.41%，不等同于未来评测覆盖率。

## 训练量、预算和版本链

用户已授权本轮自行划分数据量。按**请求数**制定预算，不把原始候选行数当训练样本数。
当前长历史 train 人群共 1,398,220 warmup 请求、295,643 对齐请求，九天更新共 843,144 请求。
这些已是筛选后的用户/时间范围；不会为凑数据量训练全部原始日志。
若限制请求量，按用户×天分层、固定哈希选择嵌套子集；保留每个被选请求的完整候选和完整历史。
当前实际 Max 验证采用 `configs/max_budgeted.json`：262,144 warmup 请求、131,072 完整候选请求×最多两轮，
对齐 epoch1/epoch2 仅按 4096 条预发布验证请求的原生 NDCG 选定。
该子集触及 90.61% 的显式 embedding 行，保留全量方案 99.46% 的因果历史事件并集；
数据覆盖不等于模型质量，是否扩大样本仍由 Max 的验证结果决定。
当前两轮原生验证NDCG为0.030909/0.046171，选第二轮；配对UID增量区间[0.010431,0.020110]。
该结果支持继续实际日更新，但尚不能证明收敛。已在任何Max cache结果前记录
`runs/max_budgeted/attempt_1/foundation_data_decision.json`，暂保留当前请求量，为完整九次更新留预算。

目标是请求平均的曝光 BCE + λ × 完整候选多正例交叉熵。
λ=0/0.1 只按 D24–26 validation UID 的原生 NDCG 选，先选择再查看缓存结果。
listwise 对未知候选只有弱竞争监督，不能把它们称作真实负反馈。
AdamW，base LR 3e-4、update LR 1e-4，weight decay .01，clip 1，5% warmup，余弦衰减下限 5%。
后续版本继承父版参数和匹配的 AdamW 状态，重设当日学习率计划。

真实 Max 的 1/2/4 卡对照见 `runs/profiles/`。全局 batch 64、microbatch 16 时，
四卡完整候选训练 p95 0.883 秒/步，曝光训练 p95 0.357 秒/步；正式采用四卡。
完整模型/优化器保存约 34.71 秒，峰值已分配内存约 21.84 GiB/卡。
这是每阶段 40 个计时步的短测，最终预算还要计入验证、保存、启动及 25% 余量。
本次已完成的262144请求曝光预训练含初始化/保存实测25.05分钟；完整候选131072请求每轮约26分钟，
两轮训练含加载/保存实测52.82分钟，加验证后基模约1.5小时。25分钟不是完整基模的耗时，也不是全量数据一轮。
原冻结配置保留15小时硬限；用户最新约10小时目标在新的版本链入口收紧为基模与全链各最多10小时，
其中全链预算保守计入已有小模型/Max验证时间。未改动正在训练的配置或数值代码。
先前受控验证的小模型、Max 与补充评测合计上限三小时，已用131.93分钟完成，保守计入整个版本链预算。
当前数据量依据预发布原生验证和预算确定，完整链继续使用冻结配方。

`code/version_chain.py` 默认只做CPU准备，`--execute` 才会在当前Max完成、原生质量检查通过后继续。
它核验已有M0/M1/M2的真实父版、完整轮次、人口、源哈希和窗口，复用通过核验的已完成任务。
随后固定完成D27–35的九次更新，全部边照实报告，不按cache差异的正负中途选边或早停。
原生验证决定模型配方；确认集只在配方冻结且十个模型/九条开发边完成后打开。
当前九次更新加全部开发/确认评测预计5.59小时，已含加载保存和25%余量；
`runs/version_chain/attempt_1/preparation.json` 是准备记录。
2026-10-28 已根据预发布验证和M0下一整日原生结果审阅通过；
`run_after_pilot.py` / `runs/version_chain_queue/attempt_1` 已等待原有界Max自然结束并验收通过，
`runs/version_chain/attempt_1` 已实际复用通过核验的M0/M1并继续更新。
实时完成版本、当前进程与评测窗口见 STATUS.md 和该 run 的 progress.json；勿再次手动派发已有任务。
首条D28完整日开发边Rebuild/Reuse NDCG为0.038473/0.034265，差的UID95%区间[0.001922,0.006508]。
原首请求诊断区间仍跨零，128历史首请求点差为负；全部结果保留，不替换预先固定的完整日1024主协议。
上述为开发集记录；确认评测在配方冻结且十模型/九开发边完成后统一执行，最新确认进度见 STATUS.md。
第二条D29开发边Rebuild/Reuse为0.041130/0.040975，差区间[−0.000743,+0.001097]跨零，照实保留。
`verify_version_chain.py` 的CPU进程独立核对每个已完成窗口的全部候选/标签及NDCG/Recall，
结束时要求十模型、九开发窗口、九确认窗口和额外D37窗口全部齐全；输出在 `runs/version_chain_verification/attempt_1`。

多卡精确平均有效请求梯度；仅通信本步涉及的 item 梯度行，其他梯度正常同步。
物品表的梯度和 AdamW 更新仍为完整稠密语义，包括未访问行的历史动量和权重衰减，
没有替换成 SparseAdam。已实测其与完整梯度同步的一步参数、优化器状态一致。
日文件并行转换、Arrow 投影、mmap 共享历史、向量化 gather、spawn worker 和 pinned prefetch；
不展开“每个候选一份历史”，不在训练热路径做 Pandas 分组。

## 评测与后续接口

`RequestRanker.encode_history(batch)` 产生分层 KV；`read(batch, cache)` 接受同形状 KV 并独立评分。
小模型最初采用每名预定居民当日第一请求。其 1024 历史的两条复用差异在首请求和完整日均不明确。
在任何 Max 质量结果之前，`configs/max_budgeted.json` 将主评测统一为下一完整自然日的全部居民请求，
并从同一批分数中保留原始首请求诊断；确认用户在配方冻结且全部训练完成后单独评测。
每条请求均使用其自身严格在前的相同历史，构造版本快照：
父版原生、新版重建、同一新版读父版 KV 三路使用同一历史和完整候选。
新版原生不必优于父版；需要直接检验 Rebuild−Reuse 的 NDCG 及配对 UID 区间。
128 历史仅作同一模型的推理截断诊断，不能代替长/短历史重新训练的比较。
正式首边预先固定 M0→M1，其余八条全部报告；确认 UID 在配方冻结前不打开。
快照实验不等同于持续 serving；真实发布写入、追加和 FIFO 仍需单独接入和验证状态语义。

每次评测保存逐请求指标及全部候选原始 ID、标签、分数；分片无补齐重复，合并检查键集合。
checkpoint 原子提交并附 SHA，保存参数、优化器、scheduler、RNG、游标、实际父版本和累计预算。
profiling/resume-check 状态标记 `quality_use_permitted=false`，评测器拒绝作为质量证据。

检查入口：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q tests/experiments/recflow_request_ranking
ruff check experiments/recflow_request_ranking/code tests/experiments/recflow_request_ranking
```

训练/评测使用 `PYTHONPATH=src python -m torch.distributed.run --standalone --nproc_per_node=4 -m ...`；
每个已运行任务的精确命令、来源和输出在其 run 目录，不凭 README 重新启动已完成任务。

参考：[RecFlow 数据协议](https://arxiv.org/html/2410.20868v1)、
[HSTU](https://arxiv.org/html/2402.17152v1)、
[抽样推荐指标问题](https://arxiv.org/html/1912.02263v1)。
