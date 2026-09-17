# Max：数据与训练准备

**2026-09-18暂停点：固定当前V0及其10%用户E14评测，不启动全人口评测或V1–V5。** 权重、封存、数据/ID映射、抽样UID、原始分数、配置和脚本均保留原位置。后续按[计划](../../../docs/unified_training_2026_09/plan.md)逐版训练和评测。

2026-09-15：20万用户数据已处理，数据审计、四卡训练资源探针及评测流程检查通过。**V0已于2026-09-18 01:41完成训练与最终权重封存，退出码0；10%用户E14抽样AUC为0.653509，全人口尚未评测。**

后台会话：`evokv_max_v0_20260915`；[运行日志](seed17/v0_1epoch_4gpu_b80_cpu14/train.log)、[任务状态](seed17/v0_1epoch_4gpu_b80_cpu14/progress.json)、[启动前检查](preparation/v0_launch_preflight.json)。只训练V0；后续版本不自动启动。

## 数据与 ID

- 数据源：Yambda-5B；三个原始文件的完整 SHA256 与下载清单一致。
- 资格：day217 前至少一次 listen，共796,134名合格用户。复用历史 `evokv:yambda500m:medium:v1` 命名空间的稳定 SHA256 排序，选前200,000人；不使用未来标签选择人口。
- 保留原始 UID；与 Large 重叠20,113人，不要求覆盖 Large。UID 用于分组和历史查找，模型没有按用户建立的 embedding 表。
- 保留所选用户完整1,127,650,996条 listens、17,605,490条 likes、2,237,581条 dislikes；压缩数据约6.6 GiB。
- 词表仅取所选人口 `[0,217天)` 的 listened items，按原始 item ID 数值排序建立独立紧凑映射，共3,530,650项。跨规模不要求紧凑 ID 数值相等；使用各自映射与 checkpoint 绑定。

| ID 用途 | Max 范围 |
| --- | --- |
| Padding | 0 |
| Known item | 1–3,530,650 |
| 未知历史 item 的256个稳定哈希桶 | 3,530,651–3,530,906 |

新 Max 显式设置 `oov_bucket_start=K+1`，避免历史默认布局中 known/OOV 边界重叠。历史同时间戳按原始 item ID、behavior 排序，映射后保留该顺序，避免 OOV 哈希改变截断历史。两项均由 Max 数据元信息显式启用；Medium/Large 默认行为及已冻结资产不变。

监督沿用真实 like/dislike 二分类，按 `(uid,timestamp,raw_item_id)` 合并重复请求、排除冲突标签，历史严格早于目标时刻。未知目标留在审计中，排除训练和质量评价；未知历史映射到 OOV 桶。训练沿用用户等权 BCE，评价沿用 E14、同窗口 Parent/Current Full pooled ROC-AUC 与原准入规则。

| 窗口 | 半开天数范围 | Known 监督请求 | 未知目标占全部请求 |
| --- | --- | ---: | ---: |
| V0 | [0,217) | 13,250,545 | 0.38% |
| V1 | [217,231) | 991,453 | 4.45% |
| V2 | [231,245) | 912,098 | 9.38% |
| V3 | [245,259) | 884,178 | 13.41% |
| V4 | [259,273) | 893,856 | 14.82% |
| V5 | [273,287) | 908,323 | 16.36% |
| V5 E14，仅评价 | [287,301) | 920,937 | 17.31% |

V0 有155,497名用户产生有效 known 监督，其余选中用户仍保留历史。尾部 day300 数据不完整。固定词表使后期未知目标比例上升，AUC 结论限于 known-target 请求，不能描述为所有目标覆盖。

## 模型与资源

保留 **16L / H320 / 10 heads / context1024 / seed17**：每头32维，与 Large 保持宽度一致，通过深度与人口扩大规模。总参数 **1,138,203,521（约11.38亿）**；item embedding 为1,129,890,240，占99.27%。没有进一步增加 hidden size 的必要性证据。

四张 A40，FSDP full shard、BF16 compute、FP32 optimizer。每 rank history/Arrow CPU14线程、Arrow IO4、Torch/OMP4，使用互不重叠的 CPU 亲和范围。每次探针12个优化步，所有 rank 的所有 batch 均为1024长度；只比较资源，不读取质量。

| 全局 batch | 每卡 batch | 请求/秒 | 峰值 reserved MiB | 相对 Torch 可用显存余量 |
| --- | --- | ---: | ---: | ---: |
| 64 | 16 | 75.78 | 33,324 | 26.7% |
| **80，选定** | **20** | **86.29** | **36,672** | **19.4%** |
| 96 | 24 | 93.83 | 40,002 | 12.1% |

选80，较64吞吐高约14%，满足预设至少15%显存余量。96不满足余量，未继续运行128。评测小样本检查采用四卡、每卡64：59个请求、118行成对输出，同一权重的两份输出完全一致，峰值 reserved 14,000 MiB。该检查验证流程，不代表全人口评测吞吐或训练质量。

## 预算与入口

V0 准备配方：全新初始化，`[0,217)`，1 epoch，全局80，AdamW learning rate 2e-4、weight decay 1e-4，仅保存最终 checkpoint。预计约165,632步；按满长度探针外推，纯训练42.7小时，排期暂按43–52小时，评测另计。后续每个14天窗口约2.85–3.19小时/epoch；若 V0–V5 各1 epoch，合计约57.4小时纯训练。短探针外推有不确定性，实际运行应更新预计完成时间；后续版本 epoch 数在各次启动前确定。

- [数据配置](../../../configs/unified_training_2026_09/max_data_preparation.yaml)、[本地数据元信息](../../../data/processed/yambda5b_max_200k_v1/scales/max/dataset.json)、[请求清单](../../../data/manifests/yambda5b_max_200k_hstu_native_v1/manifest.json)。
- [数据审计](preparation/data_audit.json)、[资源选择](preparation/resource_selection.json)、[耗时预算](preparation/training_budget.json)、[准备完成记录](preparation/readiness.json)。
- [V0 待启动合同](../../../configs/contracts/yambda5b_max_v0_prepared_v1.yaml)、[执行配置](../../../configs/unified_training_2026_09/max_v0_prepared_execution.yaml)、[启动脚本](../../../scripts/unified_training/run_max_v0.py)。

`python scripts/unified_training/run_max_v0.py --print-command` 可检查命令。用户本次确认检查通过后启动，现已记录授权并更新合同哈希，以 detached tmux 启动且保留日志和退出状态；不自动训练整条链。

一次性探针权重已删除，保留资源摘要、日志、退出状态和评测原始封存，见 [清理记录](preparation/probe_cleanup.json)。数据审计通过；相关数据/历史/选择器测试7项、epoch测试2项通过。

## 故障恢复（2026-09-15补充）

用户要求为40余小时训练保留定期恢复点。V0待启动配方新增：第500步首次保存，随后每4000步保存，滚动保留最近两份完整恢复点。按当前吞吐，首次约8分钟，常规间隔约62分钟。最终模型仍只保留1 epoch端点；恢复点不作为release或质量候选。

恢复点在优化器步结束后保存各rank的本地参数分片、AdamW一阶/二阶动量和步数、CPU/CUDA/Python/NumPy随机状态，以及已完成优化步数与数据/配置绑定。重新读取相同数据顺序，从下一步继续，要求相同四卡布局、batch和代码。每个rank先写临时文件、flush/fsync、原子替换并校验SHA256；所有分片成功后才发布`complete.json`，随后淘汰更旧的完整恢复点。未完成的目录不参与自动恢复选择。

正式启动后若进程中断，可用 `python scripts/unified_training/run_max_v0.py --resume` 从最新完整恢复点继续；启动授权限制仍适用，续训会保留原日志并另写本次日志。V0现已启动。硬盘损坏或外部删除仍需另有备份，本机制主要应对进程退出、OOM和作业中断。

完整模型保存/加载开销及跨进程续训一致性见 [恢复探针](preparation/recovery_probe/summary.json)。每份含优化器的恢复点约13.66 GB，两份约27.32 GB；写新一份期间最多同时约41 GB，另有最终模型文件。探针数据为合成满长度输入，只检查恢复与成本，不读取AUC。

实测写盘并校验约17秒/份，按4000步间隔约占0.5%时间。合成数据的完整模型连续4步与第2步退出、另起进程续训到第4步对照：一个参数有1.86e-9浮点差异，其余参数逐位一致；优化器也存在微小差异，严格rtol=1e-6、atol=1e-8检查未通过。该探针不宣称严格等价，差异及初始失败日志保留在摘要中。另以**实际训练脚本和真实数据**完成第8步恢复到第12步，与连续12步的最终权重逐位一致，见[训练入口恢复检查](preparation/recovery_integration/summary.json)。小模型的参数与优化器跨进程恢复对照也逐位一致。

训练完成且最终权重通过检查、封存后，启动脚本记录恢复点摘要并清理恢复权重，最终仍保留正式端点；中断或失败时保留恢复点。

首次真实数据恢复检查发现reserved显存峰值40,968 MiB，实际张量占用未对应增加。恢复结束后增加一次`empty_cache()`释放初始化遗留的未使用缓存，重新跑完整训练入口对照：最终权重仍逐位一致，续训峰值恢复为36,672 MiB（19.4%余量）。最终检查见[恢复显存及一致性](preparation/recovery_integration_cache_release/summary.json)。各次一次性恢复权重已清理，摘要、失败诊断、日志和分片清单保留。

## V0完成记录（2026-09-18）

已完成165,632步、1 epoch，端到端耗时180,518.85秒，约50小时9分钟。最终权重4,552,853,783字节，SHA256为`83edf405eaf32d9413b2aefba05f921410eb51d583615289f20ac3a5c426d3ca`；本次状态检查重新计算哈希，与封存一致。训练与launcher退出码均为0，无中断续训；最终权重通过有限值检查后封存，滚动恢复权重已按计划清理，摘要仍保留。四卡已释放。

[最终权重](seed17/v0_1epoch_4gpu_b80_cpu14/checkpoint/checkpoint_100.pt)、[封存记录](seed17/v0_1epoch_4gpu_b80_cpu14/checkpoint/checkpoint.seal.json)、[完整训练结果](seed17/v0_1epoch_4gpu_b80_cpu14/checkpoint/train_result.json)。

完整运行（含最终checkpoint保存）的四卡最高reserved显存为41,476 MiB，约40.5 GiB，高于短探针的36,672 MiB；相对Torch可用45,490 MiB的余量约8.8%。最高allocated显存为24,718.48 MiB。没有OOM，但不能继续把短探针19.4%余量描述为整个运行的最坏情况余量。训练步时间中位数约1.023秒，对应78.17请求/秒；端到端耗时包含数据加载、保存和校验。

当前只确认训练完成；尚无正式V0 AUC或质量准入结论，尚未启动V1。

## V0 E14：10%用户抽样评测（2026-09-18）

用户授权先评测10%用户，再根据结果决定是否扩大。独立命名空间`evokv:max:v0:e14:uid10pct:seed17`对20万原始UID做SHA256排序，固定前20,000人；不按反馈、标签或模型分数挑人。评价V0训练结束后的[217,231)天，沿用Full、context1024、真实like/dislike和request-pooled ROC-AUC，unknown target仍排除。该窗口样本共101,545条请求，其中96,750条known-target请求、9,114名有效用户；4,795条unknown target排除，不补选用户。

[冻结配置与样本绑定](seed17/v0_e14_users10pct/configuration.json)、[短检查](seed17/v0_e14_users10pct/canary.pass.json)、[资源估计](seed17/v0_e14_users10pct/resource_estimate.json)。64用户/478请求资源canary通过，未读取质量；全样本预计11–15分钟，四GPU、每卡64。单模型评分复用现有release Full evaluator，原始分数封存后关联标签，AUC复用原实现。这是抽样基础质量诊断，不是全人口结果，也不做Parent/Current增益或release准入判定。更大范围尚未启动。

### 抽样结果

评测完成，实际耗时616.75秒（约10分17秒）。**Full pooled ROC-AUC = 0.6535090485**，log loss = 0.2818309273，Brier = 0.0764522825。固定抽样20,000人，窗口内有效9,114人、96,750条请求：like 88,538、dislike 8,212。所有分数有限，请求唯一性、样本归属、请求数、有效用户数与标签关联检查通过。四卡峰值reserved 21,316 MiB，退出码0。

结果位于[摘要](seed17/v0_e14_users10pct/summary.json)、[原指标实现输出](seed17/v0_e14_users10pct/evaluate/adjudication.json)、[原始分数封存](seed17/v0_e14_users10pct/evaluate/raw/raw.seal.json)。checkpoint和sample UID哈希记录在摘要，原始分数及标签保留本地。

该点估计符合用户期望的0.6–0.7基础AUC范围，可继续扩大样本验证；不是全人口质量结论，也不证明后续版本的相对增益。更大范围评测与V1均未启动。仅记录V0绝对质量，无父子模型准入比较。
