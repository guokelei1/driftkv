# 2026-09-20：保留的摘要输入诊断

历史开发记录；以下设置和结果仅指本次实验。旧执行入口及预览图已退役，
原始评分、配置、失败和封存来源继续保留。当前三规模 Motivation 见
[实验索引](../../../docs/motivation_observations.md)。

本组完成后，用户明确 Insight 只检验共享规则与用户独特信息，不在这里检验摘要。
随后完成 [2560 用户共享读取诊断](../query_read_6l_2560_20260920/README.md)，
中间的 [128 用户信息诊断](../user_information_6l_20260920/README.md)也保留。本组不作为
新实验的候选配置；原结果全部保留，不按结果筛除方法、版本或用户。

当时设置：统一 Medium 6L/H192/6heads/context1024/seed17，V1→V2、V2→V3，
固定256校准及128评价开发UID。两共享规则均有实际q/r，区别仅为是否加入K/V均值
PCA32条件；每用户教师参照使用32anchor、32留出候选。故本组不隔离“有无用户信息”。

| 方法 | V1→V2 恢复率% | V2→V3 恢复率% |
| --- | ---: | ---: |
| 每用户教师offset+rank1 | 99.1208 | 99.4992 |
| 共享q/r，无显式摘要 | 12.1706 | 34.7059 |
| 共享s/q/r | 27.6341 | 24.6633 |

主量是先按用户计算概率MAE的未裁剪恢复率，再用户等权；具体配置与全部逐用户结果
在 [diagnostic/summary.json](diagnostic/summary.json)、`diagnostic/per_user.csv`和两份
本地NPZ中。绝对MAE、pooled恢复率及尾部结果均保留。第一次canary因MKL多线程
LU错误停止（没有质量结果），单线程同方程修复后的canary2及其较差质量结果也保留。

`source_snapshot/`保留这次执行的runner、原表生成器和原表，匹配原configuration中的
执行源hash。随后runner增加了用户信息探针，旧结果的执行版本以快照为准；公共原语的现存范围见 [Design 接口](../../../scripts/design/README.md)。
历史封存的512用户oracle诊断继续保留。
