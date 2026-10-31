# Yambda-Max V1→V2、V2→V3，g202 GPU0/1

**用户于2026-10-23 10:40主动停止任务。** 队列与GPU0/1质量worker已全部退出，两张卡显存已释放；V2→V3不会自动开始。V1→V2已保存34块／139,264人及六个校准点，原分数、配置、模型及权重保留，可按原launch.json恢复。停止证据见../../queue_v1_v3_gpu01_202/attempt_1/stop_completed.json。等待用户恢复指令；下文启动状态与完成时间为历史记录。

用户于2026-10-23授权依次启动这两条独立相邻边，每条边同时使用物理GPU0/1、NUMA0；不占用GPU2/3。复用已完成V0→V1的数值执行源码，新的顺序启动器为scripts/launch_max_v1_v3_gpu01_2026_10.py。

每边196,363名固定合法质量／完整费用用户、512 fit／16 validation、24个自然点，append256/query16、选择器候选固定16。仅沿用Max固定UID角色与该边请求；每条边从对应旧模型原生缓存开始，重新选连续层段并拟合全部六点，未复制其他边的纠正权重、层profile、缓存或质量分数。

校准只用发布前历史与教师输出；重算／映射在首次历史消费前执行一次，Query-only只修正查询读取。费用覆盖全用户全部事件，包括无反馈及最后评分后的维护；最终合并原始分数计算AUC、未截断恢复率和1000次配对UID bootstrap，生成CSV／LaTeX／报告／附加内存账。

每卡4096人原子保存分数与标记；每个校准点分别保存。断线不影响tmux；中断后仅在原队列和子进程退出后，使用queue_v1_v3_gpu01_202/attempt_1/launch.json中的resume_command，验证冻结配置、源码、模型、权重与分块hash后接续。

首轮V0→V1实测8小时54分，当前两条边预留18–20小时；是共享机器的排期估计，不是受控论文计时。只启动V1→V2及V2→V3，不重跑V0→V1、不排其他边或CacheTide。输入覆盖和模型核查见preflight.json。

实际启动：2026-10-23 04:31（上海）。队列主PID1508077，首边调度PID1508137，拟合worker PID1508199/1508200。tmux socket/session为yambdamax202/max-baselines-v1-v3；启动、接入、恢复命令在../../queue_v1_v3_gpu01_202/attempt_1/launch.json。预计10月23日22:30至10月24日00:30完成，仅为进度估计。

最新NUMA要求（2026-10-23）：后续GPU0/1必须NUMA1 CPU／内存绑定并施加独立总内存硬上限。旧launch.json.resume_command会强制NUMA0，禁止直接用；改用scripts/resume_max_v1_v3_numa1_2026_10.py --host-memory-gib <上限>。4GiB受限微型双卡试验通过（峰值1.663GiB、受检页全部NUMA1、测试组匿名N0=0）；全量保持停止，真实模型小用户峰值验证待后续授权。证据见../../numa1_gpu01_probe_202/attempt_1/README.md。

2026-10-23已按用户要求恢复：GPU0/1、CPU/业务内存NUMA1，整组64GiB硬上限、swap0、独立tmux＋systemd。真实模型16用户试验峰值21.56GiB；全量启动峰值46.28GiB，监测匿名N0=0；34个旧块已恢复跳过。当前记录../../queue_v1_v3_gpu01_202/attempt_1/numa1_resume_1/launch.json；此前停止状态及NUMA0入口均为历史。每边execution_placement_override.json保留硬件放置迁移来源，数值配置与原分数不改。
