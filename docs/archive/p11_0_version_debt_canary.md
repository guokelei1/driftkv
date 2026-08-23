# Archived: P11.0 Version-Debt Canary

P11 首次区分两种容易混淆的 age-2 状态：

- `DirectAge2Diagnostic`：在 edge2 当前前缀上直接用 θ0 重算，只是诊断，不是真实线上 lineage；
- `RecursiveMixed`：edge1 发布时只物化一次 θ0 state，随后在 θ1 服务期严格逐事件 append/evict，最终由 θ2 读取。这才是连续 No-op 后的真实状态债务。

M1 seed17 的 32-user canary 共递归处理 8,062 个中间事件，CurrentExact self difference 为零，所有时间与最终长度门通过：

| Lineage | Mean MSE | Mean Bernoulli JS | P95 JS |
|---|---:|---:|---:|
| One-hop θ1 | 0.006707 | 0.0001884 | 0.0004214 |
| Direct θ0 diagnostic | 0.008604 | 0.0002344 | 0.0007027 |
| Recursive θ0→θ1 mixed | 0.007211 | 0.0001950 | 0.0006133 |

Recursive mixed 相对 one-hop 的 mean JS 约高 3.5%，而 direct-age2 高约 24%。这说明简单地把 θ0 直接应用于 edge2 prefix 会夸大真实 recursive debt；真实债务在 canary 中较小但未消失。

该结果只授权实现 batched recursive full matrix，不是正式 version-debt 结论，也未使用 θ3 blind edge。
