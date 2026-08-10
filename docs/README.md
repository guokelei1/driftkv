# 文档索引

当前文档只回答三个问题：我们在研究什么、什么结果有效、模型文件丢失后如何重建。

1. [08_core_insights_and_roadmap.md](08_core_insights_and_roadmap.md)：权威研究状态、已选基线和下一步。
2. [eval_protocol.md](eval_protocol.md)：Reuse/Recompute 的严格语义、数据边界和指标口径。
3. [BASELINE_REPRODUCTION.md](BASELINE_REPRODUCTION.md)：从原始数据重建 θ0、自然模型链、44.67 GiB 大模型链和 8×8 矩阵。

机器可读事实来源是：

- `configs/evokv_root_cause/kuairand_large_baseline_registry_20260811_v0.json`
- `scripts/run_evokv_kuairand_large_baseline_rebuild.sh`
- `scripts/verify_evokv_kuairand_large_baseline.py`

若文字与机器清单冲突，以 registry 和验证结果为准，并同步修正文档。旧的 QK/QB 搜索、人工 K/V 坐标变换、两层模型和早期 D1/D2/D3 路线均不是当前基线，相关长文档不再保留。
