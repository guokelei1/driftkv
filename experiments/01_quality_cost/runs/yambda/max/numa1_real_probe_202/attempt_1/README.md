# 真实Max双卡NUMA1小面板

2026-10-23通过：V1/V2实体Max模型、原历史加载／hash验证、同边六点校准读取、全部24策略，16名固定抽样用户、每卡8人，保留原append256/query16。独立32 GiB cgroup上限、swap0、NUMA1 CPU及严格内存绑定；真实主机内存峰值21.560 GiB，144次采样中cgroup匿名N0最大0字节，max/oom/oom_kill均0。数值核对与各卡执行结果见worker_0.json、worker_1.json；NUMA／内存总表numa_summary.json，完整命令launch.json。

随后用户授权的全量队列已使用相同NUMA1策略和64 GiB总硬上限恢复，记录见../../queue_v1_v3_gpu01_202/attempt_1/numa1_resume_1/launch.json。本小面板是执行验证，不是额外全量质量证据。
