# KuaiRand 44.67 GiB θ1–θ8 基线复现

本说明保证在 checkpoint 丢失后，只依赖仓库、三份原始 KuaiRand 文件和冻结环境即可沿同一路径重建当前基线。

## 1. 事实来源

机器清单：

```text
configs/evokv_root_cause/kuairand_large_baseline_registry_20260811_v0.json
```

唯一编排入口：

```text
scripts/run_evokv_kuairand_large_baseline_rebuild.sh
```

清单冻结了 37 个直接依赖的数据、配置和代码文件及其 SHA-256，同时记录 θ0、自然链、大模型链和参考矩阵。不要使用其他历史 shell 脚本拼装这条链。

## 2. 当前有效产物

| 产物 | 路径 | 规模 |
|---|---|---:|
| θ0 | `results/root_cause_campaign/kuairand_latest_item_query_fullusers_h512_l8_20260810_v10/checkpoints/seed_53117/theta0.pt` | 6.03 GB |
| 自然链 | `checkpoints/evokv_kuairand_latest_query_medium_theta9_half_e3_lineage_v22/` | θ1–θ9，约 25 GB |
| 选定大模型链 | `checkpoints/evokv_kuairand_latest_query_large_capacity_lift_theta1_theta8_v37/` | θ1–θ8，384,616,642,208 bytes |
| 完整大模型推理 | `results/root_cause_campaign/kuairand_latest_query_large_capacity_lift_theta1_theta8_20260811_v37/lineage_theta1_theta8/result.json` | 28 个格的绝对端点 |
| 矩阵 | `results/root_cause_campaign/kuairand_latest_query_large_capacity_lift_theta1_theta8_20260811_v37/theta1_theta8_matrix.json` | NDCG@5/MRR/HR@5 |

自然链配置包含 θ9，是为了保持其完整、可 resume 的历史边界；选定大模型只消费并生成 θ1–θ8。

## 3. 环境与资源

- Python 3.13.12、PyTorch 2.12.1、CUDA 13.1。
- GPU0/GPU1 各一张 NVIDIA A40；GPU2/GPU3 不使用。
- 全新重建建议至少预留 450 GiB 磁盘。最终大模型 checkpoints 本身约 358.2 GiB（十进制为 384.6 GB）。
- 参考机器上的阶段耗时约为：θ0 35 分钟、自然链 29 分钟、容量 lift 8 分钟、大模型全矩阵推理 61 分钟。含 I/O 和检查应预留约 2.5–3 小时。

安装：

```bash
pip install -e .
```

## 4. 先验证，不重训

轻量验证会检查所有冻结源文件、θ0 哈希、checkpoint manifest、payload 大小、结果绑定和 8×8 数值，但不会读取 384.6 GB payload 的全部内容：

```bash
scripts/run_evokv_kuairand_large_baseline_rebuild.sh verify
```

需要做存储介质完整性检查时，再执行全 payload SHA-256；它不使用 GPU，但会顺序读取全部 checkpoint：

```bash
scripts/run_evokv_kuairand_large_baseline_rebuild.sh verify-full
```

## 5. 安全续跑

若有效前缀仍在，只缺少后续版本或最终评测，使用：

```bash
scripts/run_evokv_kuairand_large_baseline_rebuild.sh resume
```

脚本会依次检查 θ0、自然链、容量 lift、大模型 lineage 和矩阵；完整阶段直接跳过。它不会覆盖一个已存在但不一致的 θ0 边界，此时应先查明原因，或明确选择全新重建。

## 6. 从零重建

`fresh` 会删除且只删除以下五个可再生产路径：

```text
results/root_cause_campaign/kuairand_latest_item_query_fullusers_h512_l8_20260810_v10
checkpoints/evokv_kuairand_latest_query_medium_theta9_half_e3_lineage_v22
results/root_cause_campaign/kuairand_latest_query_medium_theta9_half_e3_lineage_20260810_v22
checkpoints/evokv_kuairand_latest_query_large_capacity_lift_theta1_theta8_v37
results/root_cause_campaign/kuairand_latest_query_large_capacity_lift_theta1_theta8_20260811_v37
```

确认磁盘和 GPU 后执行：

```bash
EVO_KV_CONFIRM_FRESH=delete-kuairand-large-baseline-v0 \
  scripts/run_evokv_kuairand_large_baseline_rebuild.sh fresh
```

该操作不可从脚本内撤销；当前模型仍有效时不要运行。日志保存在 `results/baseline_rebuild_logs/kuairand_large_theta1_theta8_20260811_v0/`，不位于删除集合中。

## 7. 重建阶段

### A. θ0

配置：

```text
configs/evokv_root_cause/kuairand_latest_item_query_fullusers_h512_l8_20260810_v10.json
```

使用 04-08 至 04-21 的 base period、8L/H512、video/author 和 seed 53117 训练。底层入口还会产生一个 θ1 诊断模型，但后续自然链只绑定 θ0，θ1 诊断结果不属于选定链。

### B. 自然日流式链

配置：

```text
configs/evokv_root_cause/kuairand_latest_query_medium_theta9_half_e3_lineage_20260810_v22.json
```

在 GPU0 上从 θ0 顺序训练 θ1–θ9。每个版本只使用其 update date，在下一自然日评测；配置内的 fixed schedule 固定每一版 epochs、dense/KV/embedding/projection 学习率和 20,000 样本上限。前两个 embedding checkpoint 为 full，后续为可递归恢复的 sparse delta。

### C. 两卡容量 lift

配置：

```text
configs/evokv_root_cause/kuairand_latest_query_large_capacity_lift_theta1_theta8_20260811_v37.json
```

`strided_hash_v0` 将 embedding 物理容量扩为 8 倍，并将 θ1–θ8 写成两卡完整 checkpoint。它不是复制 8 份相同 embedding；trace 可达行保持原值，新增行是独立的冷 catalog 容量。每版都要求 active-row 最大绝对误差为 0 和单卡容量溢出为真。

### D. 大模型实际评测

GPU0/GPU1 对 θ1–θ8 重新执行完整 lineage 前向，再由 `scripts/render_evokv_kuairand_capacity_matrix.py` 生成三项指标矩阵。不能用自然链表格替代本阶段。

## 8. 预期终点

验证器要求：

- θ0 SHA-256 为 `1a6ef08445b7dfd8fffdb49ef02ea97559eb6fb69a2ebe4f060d3cdc9146b294`；
- 每个大模型为 47,960,055,552 parameter bytes，8 个 checkpoint 都完整；
- NDCG@5 相邻边 7/7 为正，均值 `+6.006973698%`；
- NDCG@5 全部格 26/28 为正，均值 `+8.851582823%`；
- 参考矩阵的所有 NDCG@5、MRR、HR@5 格在 `1e-6` 容差内重现；
- 大模型 active-row lift 误差为 0，矩阵 JSON 与 lineage SHA-256 绑定。

重新运行产生的 manifest 可能因运行元数据而不与参考 manifest 字节相同；验证器把参考 hash 匹配作为审计信息，同时以配置、payload 自描述哈希、模型几何和最终数值判断重建是否有效。若要核对每个 payload 的自描述哈希，使用 `verify-full`。

## 9. 复现热 HBM 核心耗时

大模型链验证通过后，可运行一次真实 θ8←θ7 stale-cache 校验：

```bash
scripts/run_evokv_kuairand_large_hotkv_timing.sh canary
```

不要重复运行 7 条同形状版本边。正式成本实验只加载 θ8 一次，先运行包含最大形状的 22 点 canary：

```bash
scripts/run_evokv_kuairand_large_hotkv_scaling.sh canary
```

再运行 40 点正式扫描：

```bash
scripts/run_evokv_kuairand_large_hotkv_scaling.sh full
```

冻结配置为 `configs/evokv_d1/development/kuairand_large_hotkv_scaling_20260811_v0.json`。完整结果写入 `results/design1/kuairand_large_hotkv_scaling_20260811_v0/full/result.json`，每个点的双 rank 原始样本写入同目录 `points/`。参考完整结果 SHA-256 为 `e2c0e13a59db9ee098edb7300a053c921d834012f0219856023e9e94171f6fac`；参考 GPU 阶段含未计时设置共 159.4 秒，最大每卡 allocated bytes 为 29,795,053,568。

该脚本只使用 GPU0/GPU1，执行前验证基线清单、θ8 manifest、配置/实现哈希和两卡空闲状态；40 个点可逐点安全复用。它不训练或保存新 checkpoint，synthetic 权重只授权计时，不授权推荐质量结论。具体包含与排除项以 [评测协议](eval_protocol.md) 第 6 节为准。

40 点结果存在后，可无 GPU 地复现 1000 万用户 A40 卡时规划：

```bash
python scripts/estimate_evokv_a40_population_card_hours.py \
  --config configs/evokv_d1/development/a40_population_card_hours_20260811_v0.json
```

结果写入 `results/design1/kuairand_a40_population_card_hours_20260811_v0/result.json`。估算器验证源结果和自身实现哈希，固定 FP16 长期 cache、FP32 source+target 双缓冲、10% 容量余量、local batch 8 与 70% sustained-efficiency 规划系数。精简后的 C2/C4/C6/C7 来自实测点，C8 是明确标记的分解外推；两类不得混写成同等级实测证据。
