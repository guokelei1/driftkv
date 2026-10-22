# Yambda-Max V0→V1，g202 GPU0/1

用户于2026-10-22明确授权开始本条边双卡全量评测，不使用GPU2/3。实际执行绑定GPU0/1、NUMA0；原RecFlow链已完成，启动前GPU空闲，未终止任何训练。

196,363名固定Max测试用户，24点，891,487条反馈。Layer2/4/8/12，其余沿用准备期自然点；每卡4096人原子提交，1000次用户配对bootstrap，保留完整事件费用。使用已完成的同边Max六点校准与层profiling；其原GPU2/3来源保留在calibration_origin.json，数值权重、角色和请求不改。新执行不占用GPU2/3。

代码隔离在scripts/max_baselines_gpu01_2026_10，仅调整卡号、NUMA、输出、运行锁和授权范围；推理函数／类AST与已验证实现相同。配置：experiments/01_quality_cost/configs/baselines_yambda_max_v0_v1_gpu01_202.json。只授权本条边，未排其他release或CacheTide。正式输出在edge_v0_v1/。

预计8–9小时，为此前同型号A40双卡真实样本外推，首个4096人块后可更新；不作受控论文时间。来源权重、原请求、模型和执行源码的hash随configuration.json保留。中断后只在原队列和子进程全部退出后使用队列launch.json中的resume_command；断开客户端无需重启。

实际启动：2026-10-22 14:07:10（上海），tmux socket/session为yambdamax202/max-baselines。启动与恢复命令保存在../../queue_v0_v1_gpu01_202/attempt_1/launch.json；14:14核对两张卡质量进程均在运行、GPU2/3空闲，尚未提交首块。预计当日22:10–23:10完成，仅为进度估计。
