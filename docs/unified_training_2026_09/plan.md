# Yambda 三规模训练设置

本页保留已完成训练的最终配方。18 个选定端点见[模型版本清单](model_versions.md)，
逐版 Full 结果见[结果索引](../../results/unified_training_2026_09/README.md)，
15 条相邻边的 [Reuse 评价](../../results/unified_reuse_2026_09/README.md)也已完成。
原启动预算和选择过程见[关键记录](records.md)及各运行配置；没有待自动执行的训练队列。

## 模型与数据

| 项目 | Medium | Large | Max |
| --- | --- | --- | --- |
| 数据源 | Yambda-500M | Yambda-500M | Yambda-5B |
| 固定用户数 | 30,000 | 79,681 | 200,000 |
| HSTU 层数 / hidden / heads | 6 / 192 / 6 | 10 / 320 / 10 | 16 / 320 / 10 |
| Context / seed | 1,024 / 17 | 1,024 / 17 | 1,024 / 17 |
| 所选人口完整 listens | 171,051,739 | 450,612,190 | 1,127,650,996 |
| 初始 known item 数 | 1,380,509 | 2,224,809 | 3,530,650 |
| 模型参数量 | 266,259,265 | 717,260,481 | 1,138,203,521 |

Medium/Large 复用 `data/processed/yambda500m_unified_v1/scales/`；
Max 使用 `data/processed/yambda5b_max_200k_v1/scales/max/`。
人口依据 day217 前活动和稳定 SHA256 顺序固定，词表仅使用初始截止前数据。
Max 从 796,134 名合格用户中选取 200,000 人，不要求包含 Large 人口。

训练目标为真实 like/dislike 二分类、用户等权 BCE。历史来自严格早于请求时刻的 listens；
未知目标保留在审计中，排除训练和质量评价；未知历史使用稳定 OOV 桶。
Max 的 OOV 显式从 K+1 开始，历史同时间戳顺序保留原始 item/behavior 排序。
各规模使用自己的 item 映射，不能仅凭紧凑 ID 交换数据与权重。

## 训练链路与最终端点

V0 训练区间为 [0,217)；V1–V5 依次使用 [217,231)、[231,245)、[245,259)、
[259,273)、[273,287)。每版评价其后的既定 E14 窗口；V5 为 [287,301)。

| 规模 | 复用的历史起点 | 本轮训练与选定 epoch | Global batch |
| --- | --- | --- | --- |
| Medium | V0、V1 | V2：2；V3–V5：1 | 32 |
| Large | V0–V3 | V4：2；V5：1（连续两轮中的第一轮） | 历史 96；本轮 64 |
| Max | 无，V0 从头训练 | V0–V2：1；V3–V5：2 | 80 |

本轮接续训练使用四卡 FSDP、BF16 compute、FP32 optimizer；
每 rank history/Arrow CPU14、Arrow IO4、Torch/OMP4，并独立绑核。
V0 学习率 2e-4，接续训练 5e-5；AdamW weight decay 1e-4。
新版本 fresh AdamW，同一连续多 epoch 训练不重置优化器。
Max V2 的额外 epoch 是从 epoch1 重新初始化 AdamW/RNG 的历史分支；精确训练来源见
[Max 结果](../../results/unified_training_2026_09/max/README.md)。

Large V5@1 是用户选定的实验模型，原四项准入未全部通过；其原判定保留。
Max V5@2 已选定。所有版本的实际 epoch、batch、权重哈希以
[版本清单](model_versions.md)及其引用的原配置为准。

## 评价和执行入口

- Full 比较在同一边、同一 E14 请求集上计算 Parent/Current pooled ROC-AUC；
  AUC 相对提升为 `Current / Parent - 1`。不跨窗口比较绝对 AUC。
- 训练端点评价与模型选择分别记录；全部已评测端点的指标和失败判定保留。
- Reuse 使用上一版缓存，按发布后的真实历史追加和淘汰；当前 15 条边见
  [结果表](../../results/unified_reuse_2026_09/README.md)。
- [执行配置](../../configs/unified_training_2026_09/README.md)引用冻结合同。
  历史 V0/V1 等入口仍需其原配置，不能用最新规模参数替换。
- 本轮是单 seed 开发训练与模型选择，不构成跨训练种子的重复证据。
  新长训练需要资源估计、相关 canary 和明确启动授权；保存的配方不触发重跑。
