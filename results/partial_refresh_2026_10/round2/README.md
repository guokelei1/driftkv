# 第二轮：刷新／维护编码节省至少30%，收益保留区间下界>90%

**已完成。** [最终报告](report.md)、[固定30k成本](factored_cost/summary.json)、[冻结设置](frozen_candidates.json)、[代码及复现](../../../scripts/partial_refresh_2026_10/README.md)。用户指定分母为第4.4节码刷新／维护计算；完整新增计算单列。长度分组只是候选。

| 最终方案 | 4worker维护FLOPs节省 | 3360用户收益保留UID95% | 完整新增 / 一次Rebuild | 证据 |
| --- | ---: | ---: | ---: | --- |
| 批内物品项复用，无持久物品表 | 35.10282% | [99.99995,100.00008]% | 16.80301% | [NoCache](nocache3360/quality/summary.json) |
| 静态4096物品表，全量码 | 49.94358% | [99.99994,100.00009]% | 16.11196% | [Static4k](static4096_3360/quality/summary.json) |
| 静态16384物品表＋raw旧码部分复用 | 61.13082% | [93.35104,97.46633]% | 15.82191% | [组合](static16384_strict3360/quality/summary.json) |

每点完整覆盖同一3360用户、69,361请求、三发布和闲置用户；1000次配对UID bootstrap。前两点rho1，重新生成全部所需行码，节省来自原编码函数的等价计算复用，不应写成激进少刷行数已达到30%。第三点的0/.875/.75旧码策略来自首轮冻结验证，分项归因见报告。benchmark此前已有质量探索，不是新盲确认。静态4096点每发布每worker增加4.5MiB表，另有0.282MiBfolded参数；离线原型保留3发布对象，不能声称总运行表峰值只有4.5MiB。

| 其他路径 | 结论 | 目录 |
| --- | --- | --- |
| support清除后跳过层／行编码 | GPU与原masked数值相等；fixed30k纯编码仅省6.59%，含mask维护省6.75%，不达标 | sparse_canary8/，sparse_cost/ |
| 首读长度>768，尾部(.5,.25) | 保留81.84% [75.99,89.13]%，不达标 | length769_tail50_25_512/ |
| 首读长度>768，尾部(.25,.25) | 保留77.65% [67.70,84.97]%，不达标 | length769_tail25_25_512/ |
| 仅打满1024，尾部(.25,.25) | 保留78.02% [69.73,86.14]%，不达标 | fullcap_tail25_25_512/ |
| 旧rank8码能量／新旧投影变化优先 | 保留71.29%／71.40%，不达标 | priority_energy25_512/，priority_drift25_512/ |
| 等价分解GPU canary | 8用户数值检查，最大原完整logit差4.77e−7；不作质量人口证据 | factored_static4096_canary8/，factored_nocache_canary8/ |
| static4k/static16k/组合小样本 | 512用户探索；被完整质量汇总取代，保留实际运行来源 | factored_static4096_full_512/，factored_static16384_full_512/，factored_static16384_raw_strict_512/ |
| 原4096人口机制诊断 | 短历史保护仍不足目标，MSE与AUC不一致；批内独立item约63.6%提供降本来源 | diagnostics/analysis.json，diagnostics/notes.md |
| 后续生命周期范围诊断（2026-10-14） | 3360用户3545请求(5.11%)无直接旧行但仍有高层继承依赖；无全层清除后的请求，仅为因果范围元数据证据 | [lifecycle_scope_metadata.json](diagnostics/lifecycle_scope_metadata.json) |
| 后续表示坐标反例 | Q/K同时旋转不改变注意力函数，但旧K直接复用失配；解析query变换恢复得分。仅为CPU代数反例，不代表实际模型主因或AUC收益 | [coordinate_rotation_counterexample.json](diagnostics/coordinate_rotation_counterexample.json) |

512用户候选用100次bootstrap，只用于探索。独立完整成本是30k/393521真实事件，包含首次缺失、V4/V5强制刷新及追加；不是对质量样本直接放大。NoCache还依赖批内物品重复率，低重复数据不保证30%节省；原方法在纯新用户上的回归没有被等价实现修复。

item_tables/只用原256fit用户在各次发布前的最新1024行为，按频率选共享物品IDs；prepared表和所有miss都收费。静态miss不admit，不按未来频率选表，因此成本不依赖user-major回放顺序。没有运行LRU质量/服务实验；其实现留作探索入口，不用离线命中率冒充因果服务账。4部署副本重复准备的成本与GPU研究前缀加余量5个执行来源的费用分别保留。

configs/记录评估前固定的候选；frozen_candidates.json记录完整回放前冻结的数值设置，storage_mode后补元数据明确引用原freeze哈希、与既有launch一致。每次运行保留configuration/summary/scores/users，父目录logs/记录过程。源run的policy是launch fallback；策略在首读覆盖它，最终actual_rho_by_release及每request rho是实际执行值。

三份3360汇总分别核对211680、211680、241920项scalar计数；源分片也各自验证。CPU全套197通过、7CUDA跳过，见父目录logs/round2_cpu_suite.log。所有本轮GPU任务退出；没有基础模型训练、论文编译或旧数据/评分/权重改写。
