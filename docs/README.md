# 文档索引

当前文档只回答五个问题：我们在研究什么、motivation 证据是什么、什么结果有效、如何进入 D1 单链实验、模型文件丢失后如何重建。

1. [08_core_insights_and_roadmap.md](08_core_insights_and_roadmap.md)：权威研究状态、已选基线和下一步。
2. [09_motivation_observations.md](09_motivation_observations.md)：8×8 质量矩阵与人口级卡时构成的精简 motivation。
3. [10_design1_recursive_route.md](10_design1_recursive_route.md)：从 θ0 精确初态到 θ8 的单链递归 D1 契约、输出和阶段边界。
4. [eval_protocol.md](eval_protocol.md)：Reuse/Recompute 与 D1 单链的严格语义、数据边界和指标口径。
5. [BASELINE_REPRODUCTION.md](BASELINE_REPRODUCTION.md)：从原始数据重建 θ0、自然模型链、44.67 GiB 大模型链和 8×8 矩阵。

机器可读事实来源是：

- `configs/evokv_root_cause/kuairand_large_baseline_registry_20260811_v0.json`
- `scripts/run_evokv_kuairand_large_baseline_rebuild.sh`
- `scripts/verify_evokv_kuairand_large_baseline.py`
- `configs/evokv_d1/development/kuairand_recursive_chain_design_v0.json`
- `scripts/preflight_evokv_kuairand_recursive_d1.py`
- `results/root_cause_campaign/kuairand_latest_query_large_capacity_lift_theta1_theta8_20260811_v37/theta1_theta8_matrix.json`
- `results/design1/kuairand_a40_population_card_hours_20260811_v0/result.json`

若文字与机器清单冲突，以 registry 和验证结果为准，并同步修正文档。旧的 QK/QB 搜索、人工 K/V 坐标变换、两层模型和早期 D1/D2/D3 路线均不是当前基线，相关长文档不再保留。
