# Implementation 实验索引

本轮独立验证当前 Design 的等价工程优化。旧质量结论与边界仍以 `../query_relay_2026_10/report.md` 为准，不通过更换任务重新宣称质量提升。论文技术稿位于 `/home/gkl/work/paper/new5.md`，四小节只写设计/实现；这里另存实验与过程。

主要产物：

- `registration.json`：实现目标、冻结输入和验证范围。
- `reader/`：融合读取数值、资源和形状验证。
- `prepare/`：共享码准备及目录编译验证。
- `state/`：分块原地反锚/物化验证。
- `integration/`：冻结权重下真实因果轨迹的参考/优化对照。
- `population_confirmation1024/index.json`：既有完整1024用户复验的双分片、命令及合并索引。
- `cost/summary.json`、`summary.md`：完整理论成本、共享参数字节及原反馈失败边界。
- `paper/`：替换前技术稿、四节草稿及最终文件绑定。
- `final_tests.log`：新旧111项检查的真实日志。
- `report.md`、`progress.json`、`completion_evidence.json`：由 `make_report.py` 核验并汇总的结果、边界和过程索引。

组件实验、32用户双协议三版完整集成及既有1024用户的774,096评分工程复验均已完成；共同UID统计保留全部22名idle用户，评分差处于数值容差内，不作为新的盲确认或质量提升。最终状态和文件哈希见 `completion_evidence.json`，完整人口统计及退出状态见人口索引。所有本轮GPU任务和CPU汇总均已退出。

共享准备最终证据为 `prepare/final_full_catalogue/summary.json`，前期子集与无前代检查版本保留作过程记录。融合读取最终证据为 `reader/report.json`；全Query矩阵核的旧版本保存在 `reader/matrix_only_prior/`，Q1补齐浪费暴露后新增向量核。状态转换最终源码检查为 `state/gpu2_final.json`；更长重复行压力证据的构造差异在 `state/summary.json` 中说明，不当真实质量结果。

FLOPs、逻辑载荷、临时张量峰值、指令补齐和组件计时分别报告；没有生产服务吞吐/p99结论。所有GPU任务只使用当时空闲的卡，不操作其他人的进程，默认两张卡进行最终复验。
