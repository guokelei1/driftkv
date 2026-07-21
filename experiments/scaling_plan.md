# Scaling Plan: 迁移到 Attention 敏感的设置

## 问题诊断

当前 KV staleness 损失仅 0.8% MRR，原因：

1. **模型太小**（3层/128维/2.8M参数）：KV 误差经过 3 层不充分累积
2. **序列太短**（128）：attention 上下文有限，K,V 的区分度不够
3. **训练不充分**（5 epoch）：attention 没学到对 K,V 的高度依赖
4. **hidden state 余弦相似度 0.9984**：即使 KV drift 37%，hidden 方向几乎不变

但 attention 贡献本身不小（||attn||/||residual|| = 1.5-2x），elu(Q·K)*V 占 75-96%。问题在于 **误差累积不够 + 模型区分力不够**。

## 数据集决策

### 结论：继续用 KuaiRand-1K，不需要下载 27K

| 指标 | KuaiRand-1K（当前） | KuaiRand-27K |
|---|---|---|
| 用户数 | 1,000 | 27,285 |
| 人均交互 | 8,328（中位数），11,713（均值） | 类似 |
| 总交互 | 11.7M | 322M |
| 物品数 | 4.4M | 32M |
| 时间跨度 | 31天（ms级时间戳） | 同 |
| 磁盘 | 已有（1.2GB） | 需下载 46GB |

**理由**：
- 1K 用户 × 8K+ 交互/人 = 11.7M 交互，足够训练 15-25M 参数模型
- 流式 drift 估计的 probe 用户从 1K 中取 64-256 个已够（V3 实验已验证）
- 若 Phase 3 跨用户低秩分析需要更多用户，再下载 27K
- **当前瓶颈是模型规模和训练，不是数据量**

### 数据处理调整

| 参数 | 当前 | 调整后 | 理由 |
|---|---|---|---|
| max_items | 20,000 | 20,000（不变） | top-20K 覆盖 11.3%，embedding 表 20K×256=5M 参数，可控 |
| max_seq_len | 128 | **512** | 用户日均 259 交互，512≈2天历史，attention 上下文 4x |
| base_days | 14 | 14（不变） | 14天 base 足够 |
| stream_days | 17 | 17（不变） | 17天流式足够 |

## 模型 Scaling

| 参数 | 当前 | 目标 | 理由 |
|---|---|---|---|
| num_layers | 3 | **6** | 层数 2x → KV 误差累积路径 2x |
| hidden_size | 128 | **256** | 容量 2x |
| num_heads | 4 | **8** | 更细粒度 attention |
| head_dim | 32 | **64** | 每头更 expressive |
| max_seq_len | 128 | **512** | 上下文 4x |
| 总参数 | 2.8M | **~18M** | A40 单卡可训 |
| KV cache 大小/用户 | 3×128×128=49K | 6×512×256=786K | 16x → drift 更显著 |

## 训练调整

| 参数 | 当前 | 目标 |
|---|---|---|
| base epochs | 5 | **12** |
| base lr | 3e-4 | 3e-4 + warmup(500步) + cosine decay |
| stream steps/day | 40 | **100** |
| replay ratio | 0.5 | 0.5（不变） |
| batch size | 32 | 32（不变，序列变长后显存增加） |

预计 base 训练 ~30 分钟，流式 17天 ~20 分钟。

## 评测方案

### 主指标：MRR（最敏感）
- 当前已证实 MRR 能测出 0.8% 损失（hit@10 测不出）
- 目标：在 6 层模型上 MRR 损失 >3-5%

### 辅助指标
- hit@10, ndcg@10, recall@10（多正样本+1000负例）
- hidden state cosine similarity（连续诊断）
- attention contribution ratio ||attn||/||residual||（诊断）

### Staleness 梯度测试
- 1天、3天、5天、10天、17天 staleness
- 预期：损失随 staleness 天数单调增长

## 验证 Gating

1. **attention 贡献**：>1x residual（当前已满足）
2. **KV staleness MRR loss @17天**：>3%（当前 0.8%）
3. **损失随 staleness 单调增长**：day1 < day5 < day17

若 6 层仍不够：
- 加到 8 层
- 或测试去掉 attention 的 "+1" 项（ablation：`elu(Q·K)` 无 baseline）
- 或换 gating 策略（`none` → attention 贡献不被 gate 压缩）

## 执行步骤

1. 修改 HSTUConfig（6层/256维/8头/64dim/512seq）
2. 修改 StreamingDataPlan（max_seq_len=512）
3. 重跑 base 训练（12 epoch）→ θ₀
4. 重跑流式训练（17天 × 100步）→ θ₁..θ₁₇
5. 重跑必要性实验（stale vs fresh）→ 确认精度有意义
6. 重跑 KV 复用损失实验（累积 staleness）→ 确认损失 >3%
7. 若通过 → 进入 Phase 3 drift 估计

## 风险

- **R1**：6 层模型仍看不到 >3% 损失 → 加层或改架构
- **R2**：512 序列显存不够 → 降到 384 或用梯度累积
- **R3**：训练时间过长 → 减少 epoch 或用多卡
