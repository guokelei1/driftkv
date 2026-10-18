# 共享紧凑码结构探索

用户于 2026-10-14 授权探索 D1/D2 的码结构。D3 独立问题暂不改变。
实现与命令见 [源码索引](../../scripts/code_structure_2026_10/README.md)。

本轮已完成。推荐继续研究共享12＋私有6：固定4096用户、51,731请求上恢复89.4308%，
完整新增FLOPs18.6369%；所有结构总用户码存储与原方案相同。完整比较、区间、
查询开销及22.1375%源重建敏感性见[结果报告](report.md)和[汇总](summary.json)。
这不是独立确认，也不替换旧方法的历史结果或仓库默认入口。

| 证据 | 入口 | 含义 |
| --- | --- | --- |
| 三版本、三结构 CPU 检查 | [structure_checks.json](structure_checks.json) | 相同总存储、旧函数包含与合成表达能力，不代表真实质量 |
| fixed30k 完整成本预筛 | [prescreen.json](prescreen.json) | 继承 local32、替代 joint16、静态4k/4worker；最终用实际新校准费用替换 |
| GPU canary | [canary/group2/v5/calibration.json](canary/group2/v5/calibration.json) | 4 fit /2 validation，4 epoch，仅执行检查 |
| 两 head 共享16 | group2/calibration/{v3,v4,v5}/ | 同总存储，每 head 可访问16维；正式开发校准 |
| 共享6＋私有7 | shared6_private7/calibration/ | 同总存储，每 head 可访问13维；正式开发校准 |
| 共享12＋私有6 | shared12_private6/calibration/ | 同总存储，每 head 可访问18维；完整开发校准与回放已完成 |
| 三结构完整4096结果 | 各结构 quality4096/ | 全部三发布、UID区间、原始分数和成员/权重绑定检查 |
| 连接干预 | shared12_private6/ablation512_both/ | 同512用户切除后恢复88.5254%，配对差异区间跨零 |
| 共享存储 | [memory_summary.json](memory_summary.json) | adapter、校准buffers、副本和静态表分列，不混入用户码 |
| 运行过程 | logs/ | 每个任务独立日志；退出结果由任务进程和最终JSON确认 |

所有校准基础模型冻结，复用同一 local32 checkpoint，原 fit/validation 角色不变。
恢复与 80% 的比较使用 equal-release AUC gain / once-Rebuild–Reuse AUC gap；
保留三发布、所有选定用户（含无请求用户）和负结果。
原始静态4k基线也使用同一生成优化，不能把实现加速归为共享码的效果。
旧 K/V 若需重建用于校准，其敏感性成本另外报告；开发结果不是盲确认。
