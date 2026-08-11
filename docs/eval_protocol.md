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

## 6. 热 HBM 核心计时协议

推荐质量矩阵与系统时间必须分开测量。当前第一轮成本端点使用 `hot_hbm_hstu_core_only`：

- 当前模型、请求输入 embedding 和旧前缀 K/V 在计时前已经驻留对应 GPU 的 HBM；模型加载、checkpoint 读取、源 K/V 生成和 CPU→GPU 传输不计时。
- Reuse 计时包含“读取 resident prefix K/V、当前模型对固定两-token suffix 的 HSTU 前向、拼接并返回完整目标 K/V”；Recompute 包含“当前模型对完整序列的 HSTU 前向并返回完整目标 K/V”。不得让任一端省略各层 K/V 输出物化。
- embedding lookup、候选打分、NDCG/MRR/HR 计算和结果写盘不计时。它们不是本轮要隔离的 HSTU K/V 核心计算；后续端到端实验必须另立 protocol，不能与本结果混报。
- 版本间参数形状相同，不重复测 7 条相邻边。真实 θ8←θ7、32-request canary 只验证 stale-cache 路径；正式缩放只加载 θ8 一次。synthetic core 的 K/V 数值和随机权重不用于质量，只用于执行同一 HSTU kernel 与输出形状。
- full profile 扫描实际序列 32–2048、请求数 16–1024、4/8/16/24 层、H256/H512/H768/H1024，以及固定 512-token 输入下的 max-context 512–4096。44.666 GiB θ8 embedding shard 保持 resident，但 lookup 不计时。
- 单请求 latency 定义为每个 rank 一个请求、两卡并发后取 rank max；不得除以两后称作单请求 latency。批吞吐点使用 global batch 16；总时间为全部顺序 batches 的 CUDA 时间和，单请求值仅是吞吐摊销。
- 使用 CUDA event；计时前同步设备；每批交替方法顺序；双卡时间逐批取较慢 rank。保存全部 rank/batch/repeat 原始样本、重复中位数和分位数。
- 不设置 Recompute 必须慢于 Reuse 的结果门槛。短序列中完整 K/V 拼接可能使 Reuse 更慢，必须原样保留；break-even 是实验结果而不是预设。

人口级卡时只能从上述批吞吐点换算，不得从单请求 latency 外推。双卡 rank-max 结果换算 A40 卡时使用 `users × wall_ms_per_global_request × 2 / 3,600,000`；必须同时报告未折损的热内核下界和显式 sustained-efficiency 假设下的规划值。当前开发规划固定 70%，不得把该系数称为测量结果。容量表按 FP16 长期存储计算驻留用户，按 FP32 消费端 source+target 双缓冲与 10% 余量决定 admission batch；local batch 8 只是现有实测的保守校准点，不得称为已证明的最优或饱和 batch。若容量低于该校准点，必须披露 small-batch penalty。DRAM/SSD ingress、通信和排队仍不在该卡时内。

该协议刻意回答热 cache、长历史下的核心计算上界空间，不代表自然请求长度分布、cache 从 DRAM/SSD 搬入、排队、通信、候选检索和打分在内的端到端延迟。任何后续端到端数字都必须明确增加了哪些阶段。

## 7. 禁止项

- 不得人为旋转、缩放、平移或扰动旧 K/V。
- 不得对最终百分比乘常数或只展示有利指标。
- 不得在看过矩阵后修改候选集、用户集或目标集，再与旧协议混报。
- 不得把不同 protocol、seed、candidate count 或 target horizon 的格子拼成一个矩阵。
- 不得把调参使用过的同一训练 seed 当作独立统计重复。

## 8. 证据等级

当前结果是单 seed、冻结配置下的开发基线，标记为 `scientific_result=false` 和 `formal_result=false`。正式论文结果至少需要预先冻结协议、多个训练 seed、以 seed 为重复单位的统计，以及完整披露所有端点和负格。在此之前，当前矩阵只授权机制开发和系统实现，不授权最终泛化结论。

## 9. D1 单链递归协议

D1 方法不再输出另一张经过筛选的 8×8。执行从 θ0 下精确 cache 开始，沿 θ0→θ1→…→θ8 形成一条部署轨迹；每一边的方法输出必须成为下一边唯一的 cache 输入，θ0 后精确复位次数必须为 0。θ0→θ1 单独标记为 bootstrap 诊断，θ1→θ2 至 θ7→θ8 是当前 8×8 支撑的主证据边。

每条边在完全相同的 query 上报告 Full Recompute、Recursive Reuse 和 Recursive Method 三个绝对端点；edge-local exact-source Reuse 只能作为诊断。主指标仍为 NDCG@5，并同时报告 MRR 与 HR@5。只有 `Full Recompute - Recursive Reuse > 0` 时才报告 `(Recursive Method - Recursive Reuse) / (Full Recompute - Recursive Reuse)`；其余情况写 `NA` 并保留绝对值，禁止裁剪 recovery。

方法拟合、调参和动作选择不得读取 qualification 正样本、候选标签或已实现的逐 query 推荐收益。用于拟合和开发选择的记录必须与 qualification query 分离；无标签 transport 与线上动作规则不能读取推荐标签，独立 development split 的候选级聚合质量可以用于选择超参数。最终候选按整条递归轨迹冻结，不能逐边事后拼接。每条边同时记录有效 token 动作、K/V bytes、lineage、由实际 tensor shape 推导的理论操作数和单独实测的 GPU 时间。理论 token/FLOP 比例不能替代 GPU 计时。
