# Max：数据、固定模型与评价

Max V0–V5 训练和 Full 评价已完成，当前选定 epoch 为 **1、1、1、2、2、2**。
六份实体权重集中在 `seed17/checkpoints/v0..v5/checkpoint_100.pt`，
原选中入口保留相对符号链接。五条相邻边的 [Reuse 评价](../../unified_reuse_2026_09/README.md)
也已完成；[三规模版本清单](../../../docs/unified_training_2026_09/model_versions.md)
汇总各版窗口、batch、哈希和相对 AUC 提升。

## 选定模型与原运行

| 版本 | Epoch | 训练初始化 | 权重 | 原结果 |
| --- | ---: | --- | --- | --- |
| V0 | 1 | 全新初始化 | [V0](seed17/checkpoints/v0/checkpoint_100.pt) | [训练与封存](seed17/v0_1epoch_4gpu_b80_cpu14/checkpoint/train_result.json) |
| V1 | 1 | V0 | [V1](seed17/checkpoints/v1/checkpoint_100.pt) | [两个端点](seed17/v1_epochs12_4gpu_b80_cpu14/README.md) |
| V2 | 1 | V1@1 | [V2](seed17/checkpoints/v2/checkpoint_100.pt) | [epoch1](seed17/v2_1epoch_4gpu_b80_cpu14/README.md) |
| V3 | 2 | V2@2 | [V3](seed17/checkpoints/v3/checkpoint_100.pt) | [两个端点](seed17/v3_epochs12_4gpu_b80_cpu14/README.md) |
| V4 | 2 | V3@2 | [V4](seed17/checkpoints/v4/checkpoint_100.pt) | [两个端点](seed17/v4_epochs12_4gpu_b80_cpu14/README.md) |
| V5 | 2 | V4@2 | [V5](seed17/checkpoints/v5/checkpoint_100.pt) | [两个端点](seed17/v5_epochs12_4gpu_b80_cpu14/README.md) |

所选 V2@1 与 V3@2 在同一 E14 [259,273)、92,594 用户、893,856 请求上的
Full AUC 为 0.6961794233 → 0.7086984472，相对提升 **1.7982468%**。
[选择记录](seed17/selected_v2_v3_checkpoints.json)绑定精确端点和指标来源。

保留的比较分支包括 [V2 追加 epoch2](seed17/v2_epoch2_from_epoch1_4gpu_b80_cpu14/README.md)
和 [V3 from V2@1](seed17/v3_from_v2e1_epochs12_4gpu_b80_cpu14/README.md)。
前者从 V2@1 权重重置 AdamW/RNG 后再训一轮；后者的两个 V3 端点均未选用。
所有分支的原指标、准入和封存保留，未选中实体权重及优化器恢复 payload 已删除。
重放当前 V3 的原始训练需要先重建 V2@2；该历史来源的配置与结果仍在。

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

V0 有155,497名用户产生有效 known 监督，其余选中用户仍保留历史。固定词表使后期未知目标比例上升，AUC 结论限于 known-target 请求，不能描述为所有目标覆盖。

## 模型与资源

保留 **16L / H320 / 10 heads / context1024 / seed17**：每头32维，与 Large 保持宽度一致，通过深度与人口扩大规模。总参数 **1,138,203,521（约11.38亿）**；item embedding 为1,129,890,240，占99.27%。没有进一步增加 hidden size 的必要性证据。

四张 A40，FSDP full shard、BF16 compute、FP32 optimizer。每 rank history/Arrow CPU14线程、Arrow IO4、Torch/OMP4，使用互不重叠的 CPU 亲和范围。每次探针12个优化步，所有 rank 的所有 batch 均为1024长度；只比较资源，不读取质量。

| 全局 batch | 每卡 batch | 请求/秒 | 峰值 reserved MiB | 相对 Torch 可用显存余量 |
| --- | --- | ---: | ---: | ---: |
| 64 | 16 | 75.78 | 33,324 | 26.7% |
| **80，选定** | **20** | **86.29** | **36,672** | **19.4%** |
| 96 | 24 | 93.83 | 40,002 | 12.1% |

选80，较64吞吐高约14%，满足预设至少15%显存余量。96不满足余量，未继续运行128。评测小样本检查采用四卡、每卡64：59个请求、118行成对输出，同一权重的两份输出完全一致，峰值 reserved 14,000 MiB。该检查验证流程，不代表全人口评测吞吐或训练质量。

## 已完成的资源与恢复检查

V0 使用 AdamW learning rate 2e-4、weight decay 1e-4，接续版本 LR 为 5e-5。
V0 已完成 165,632 步，端到端 180,518.85 秒（约50小时9分），无中断续训，退出码0。
全程最高 reserved 为 41,476 MiB，相对当时 Torch 可用显存余量约8.8%；
最高 allocated 为24,718.48 MiB。短探针19.4%的余量不能当作全程最坏情况。
训练步时间中位数约1.023秒、78.17请求/秒；数据加载、保存和校验计入端到端耗时。

训练时采用第500步首次、随后每4000步滚动保存最近两份模型/AdamW/RNG恢复点。
[恢复探针](preparation/recovery_probe/summary.json)中每份约13.66 GB，
写盘校验约17秒；合成对照存在1.86e-9参数差异及优化器差异，严格断言失败保留。
[实际训练入口对照](preparation/recovery_integration/summary.json)从第8步恢复至第12步，
与连续12步的最终权重逐位一致。
恢复后释放未使用缓存使探针 reserved 从40,968降至36,672 MiB，
见[最终恢复检查](preparation/recovery_integration_cache_release/summary.json)。
这些是已完成检查，恢复 payload 已删除，不能直接从旧恢复点续跑。

V2 起使用 auto 后端，在支持条件下采用 Triton。
[后端对照](seed17/v2_1epoch_4gpu_b80_cpu14/backend_parity.json)
和后续各运行数值记录保留；通过的 BF16 训练/FP32 推理检查不等于所有 dtype
或完整训练轨迹逐位一致。Max V2 追加分支的停止、恢复及额外 FP32 反向失败详见
[关键记录](../../../docs/unified_training_2026_09/records.md)。

## V0 的独立 E14 抽样结果

V0 在 [217,231) 的初始诊断固定20,000个 UID（20万人口的10%），按独立 SHA256
顺序取样，不按标签或分数挑选。窗口内有效9,114用户、96,750 known-target请求，
包含88,538 like与8,212 dislike；4,795个未知目标请求排除，不补选用户。

Full pooled ROC-AUC **0.6535090485**，log loss **0.2818309273**，
Brier **0.0764522825**；耗时616.75秒，四卡峰值reserved21,316 MiB，退出码0。
这是该窗口的抽样质量结果。后续 V0→V1 完整配对评测采用另一个窗口 [231,245)，
V0 AUC 为0.656376，不能与前一个窗口混算更新收益。
[摘要](seed17/v0_e14_users10pct/summary.json)、
[原始分数封存](seed17/v0_e14_users10pct/evaluate/raw/raw.seal.json)
和[指标输出](seed17/v0_e14_users10pct/evaluate/adjudication.json)均保留。

## 配置与数据入口

- [数据配置](../../../configs/unified_training_2026_09/max_data_preparation.yaml)、
  [数据元信息](../../../data/processed/yambda5b_max_200k_v1/scales/max/dataset.json)、
  [请求清单](../../../data/manifests/yambda5b_max_200k_hstu_native_v1/manifest.json)。
- [数据审计](preparation/data_audit.json)、[资源选择](preparation/resource_selection.json)、
  [原始预算](preparation/training_budget.json)。
- [V0 合同](../../../configs/contracts/yambda5b_max_v0_prepared_v1.yaml)、
  [V0 执行配置](../../../configs/unified_training_2026_09/max_v0_prepared_execution.yaml)、
  [V0 入口](../../../scripts/unified_training/run_max_v0.py)。
- [V1–V5 执行配置](../../../configs/unified_training_2026_09/README.md)引用各次冻结合同。
  训练已结束；保存的入口不代表新的启动授权。
