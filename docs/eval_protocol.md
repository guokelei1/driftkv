# KuaiRand Reuse/Recompute 评测协议

本文件定义当前 θ1–θ8 基线中哪些数值有效且可比较。

## 1. 版本和时间语义

- 系统任一时刻只服务当前模型 θt。
- θs 只表示某用户前缀 K/V 的生成版本，`s < t`；它不是另一个同时在线的 serving model。
- θt 使用当天更新数据训练，评测正样本来自下一自然日。任何评测目标都不能进入 θt 的训练。
- θ0 是 14 天 base period 上训练的启动点，不进入 8×8 论文矩阵。

## 2. 两个端点

对同一用户、同一历史、同一当前模型 θt、同一正样本与同一候选集合：

- Recompute：用 θt 对完整有效历史重新前向，生成当前版本前缀 K/V。
- Reuse：保留 θs 生成的旧版本前缀 K/V；最新 token 与 query 仍由 θt 计算。

两端只有前缀 K/V 来源不同。不得改变模型、用户、历史长度、候选负样本或 query。padding 不计入有效历史与 K/V 容量。

## 3. 用户、序列和候选

- 使用 KuaiRand standard log 中满足历史与下一日正样本条件的用户，不只挑长用户。
- 序列由 base catalog 内的 engaged actions 构成，保留用户天然不等长历史，最大长度 512。
- 每个评测 query 使用 100 个候选：1 个真实下一日 engaged positive，加 99 个从 base-period exposure frequency 分布冻结采样且用户未见的 negatives。
- 候选随机种子固定；同一 query 的 Reuse 与 Recompute 必须逐项共享候选。
- 每个用户最多取冻结协议指定的 4 个目标。不得根据两端误差筛用户或目标。

## 4. 指标

主指标是 NDCG@5，同时报告 MRR 与 HR@5。每个矩阵格必须保存两端绝对值与相对差：

```text
relative_percent = 100 * (Recompute - Reuse) / Reuse
```

正值表示 Recompute 更准确，负值表示该有限评测集上 Reuse 更高。Full Recompute 是 cache fidelity 的正确参照，但不是有限候选排名质量的数学上界，因此负格不能被改写为 0 或删除。

## 5. 大模型有效性

- 模型容量按实际分配的参数 bytes 计算，不按文件稀疏大小、逻辑副本或手写常数计算。
- 当前大模型只有一份 23,396,297×512 的物理 embedding 空间并分片到 GPU0/GPU1；不得通过相同 embedding 副本伪造容量。
- `strided_hash_v0` 只扩展冷 catalog 行。所有 trace 可达 embedding、dense core 和 projection 必须与自然链相同，active-row 最大绝对误差必须为 0。
- 最终矩阵必须由两卡大模型实际前向生成。自然链矩阵仅用于验证函数保持，不能替代大模型执行记录。

## 6. 禁止项

- 不得人为旋转、缩放、平移或扰动旧 K/V。
- 不得对最终百分比乘常数或只展示有利指标。
- 不得在看过矩阵后修改候选集、用户集或目标集，再与旧协议混报。
- 不得把不同 protocol、seed、candidate count 或 target horizon 的格子拼成一个矩阵。
- 不得把调参使用过的同一训练 seed 当作独立统计重复。

## 7. 证据等级

当前结果是单 seed、冻结配置下的开发基线，标记为 `scientific_result=false` 和 `formal_result=false`。正式论文结果至少需要预先冻结协议、多个训练 seed、以 seed 为重复单位的统计，以及完整披露所有端点和负格。在此之前，当前矩阵只授权机制开发和系统实现，不授权最终泛化结论。
