# Phase 0 - V1-V4 前置验证总结

> 路线图 Phase 0 gating：V1 通过 + V3/V4 至少一个通过 -> 进入 Phase 1。

## 结果

| 验证项 | 结论 | 关键数据 |
|---|---|---|
| V1 工业流式训练频率 | **PASS** | 工业主流小时级在线学习 |
| V2 per-user JVP vs 全量重算 | 确认动机 | JVP = 3.2~6.4x 前向成本 |
| V3 跨用户 J·Δθ 可共享性 | **PASS** | 漂移范数 4 特征预测 rel_mae=7% |
| V4 旧 KV 精度衰减 | **PASS** | 小更新精度差<1.5%，大更新差15.7% |

## 后续演进

Phase 0 在 3 层小模型上完成概念验证。后续 scale 到 6 层/256 维/512 序列 relu 模型，发现 hit@10 对 KV staleness 不敏感（top-10 被热门 item 主导），改用 **Spearman rank correlation** 作为主指标后，KV 复用损失清晰可见（1-Spearman 从 0% 增长到 14.6%）。

最终完整 motivation 实验见 `results/streaming/complete_motivation.json`，证明三件事：
1. 流式训练必要性（frozen 50% vs fresh 96% hit@10）
2. KV 复用导致排名打乱（Spearman 1.0 -> 0.85）
3. Δθ -> KV drift -> 排名损失 信号链（Spearman 相关 0.93-0.98）
