# 分块原地码转换

入口是 `scripts/implementation_2026_10/state.py` 的 `InplaceReleasePrefix.wrap(prefix, row_chunk=128, feature_fn=...)`。`materialize` / `anchorize` 与原 `StableReleasePrefix` 签名一致，直接接入已有同步 `RelaySession`，返回传入的同一 `LightweightValueState`，保留 `codes.data_ptr()`。只支持当前 A / xi=1、共享12＋6头各私有6的48维码；训练继续使用原可微路径。

实现按层、按历史块重算固定特征 η，计算 `Qη+D`，再计算 `P a+offset` 或 `P^-1(c-offset)`，直到本块所有输出完成才覆盖输入。同一块 η 只计算一次，没有额外的持久逐行特征。原生 K/V、物品ID与发布表保持只读；raw model ID与子目录索引分别传入。几何、别名和完整ID范围在首个写入之前检查。若上游已经准备实际小块逆，包装器直接复用。

`summary.json` 是结果索引；`gpu2_final.json` 绑定最终代码、冻结V5权重、完整目录发布与已有真实校准源。真实非零混合源486120有359行，chunk128与参考逐位一致；重复到1024行的数值压力差异约2e-6。9项CPU检查还覆盖多个块宽、零出生后缀、FIFO、空窗口、raw/subset物品索引和跨3次发布的旧Session接入。

临时空间压力见 `gpu2.json`：真实1024行窗口中参考反锚/物化的峰值新增为4.5/5.625 MiB，chunk128为0.516/0.469 MiB。重复到32768行时，参考为144/180 MiB，而chunk128保持相同峰值。后两个长形状重复真实行，不属于质量用户或有效服务窗口证据。该压力运行早于包装器复用已有逆的构造优化；转换计算未改，最终构造和准备回调由 `gpu2_final.json` 再验。

同一浮点表达式因矩阵形状不同会选择不同计算路径。全零出生码反锚后再物化存在offset抵消，相对误差不能单独解释；32768行压力的最大绝对差约8.6e-6，完整相对值与输出RMS保留在结果中。不宣称所有形状逐位等价或无限版本精度稳定。

每层每行的两次小块矩阵乘与两次向量加为1536 FLOPs，固定η工作另外计入；分块不减少数学FLOPs。完整账继续沿用原六层每行93888 FLOPs的保守准备口径，不利用一个向量加的计数差异降低已有成本。测量仅报告转换调用的峰值新增分配，不是整套服务峰值、加载吞吐或p99；原Session的追加/FIFO仍可能短暂分配新张量。

运行示例：

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/implementation_2026_10/benchmark_state.py --source-uid 486120 --prepared-features --lengths 359 1024 --chunks 128 --output results/implementation_2026_10/state/gpu2_final.json
```

已有输出受覆盖保护。两次早期runner错误及修复原因保留在 `summary.json` 和对应日志中；它们未触发训练或参数调整。
