# RecFlow Figure 9d 正式评测

2026-10-30 02:01:18（Asia/Shanghai），用户明确要求启动，已进入 tmux 后台运行。

- 会话：`recflow-figure9d-formal`
- 启动记录：`runs/formal_max/launch.json`（协调器 PID 1734332）
- 当前边与实际子进程：`runs/formal_max/live.json`
- 首边：V0→V1，D29；四 worker 初始 PID 1734407/1734408/1734409/1734410。
- 配置：固定 30,000 用户，七条边，全部 29 个配置；历史 1024，完整候选，NDCG@10/Recall@10。
- 资源：四张 A40 0/1/2/3；chunk256、eager FP32、TF32 关闭；每卡两个 CPU 预取进程。
- 每边独立校准六档 Alignment，固定 cache_fit 前缀最大 256 用户、八轮；不重新训练推荐主模型。

按小规模实测，中心预估约 27 小时：10 月 31 日约 05:00 完成。建议预算 34–38 小时，即 10 月 31 日约 12:00–16:00。实际进度以 worker 输出为准。

查看终端：`tmux attach -t recflow-figure9d-formal`。协调器日志在 `runs/formal_max/controller.log`，逐卡日志为 `edgeN_rankR.log`。逐用户结果提交后更新该边的 `progress_rankR.json`；每边完成生成 `summary.json`，七边完成生成根 `summary.json` 和 `exit.json`。

全量只运行这一份。运行期间不要修改数值源码、明文配置或输入，不要重复派发 worker。全部正负结果保留；失败日志保留，未提交用户可续跑。没有文件哈希保护；原论文、模型和旧实验队列不改动。结束后不派发额外实验。

准备阶段的报告与 `prepared_not_started` 历史计划保持原样；本次新授权记录在 `runs/formal_max/authorization.json`。当前实际执行模式为 `formal`。
