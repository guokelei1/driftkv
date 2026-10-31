# CacheTide Implementation 探索

本目录实现当前冻结 `A_itemgate6_rms_roll4` 的等价工程优化，不改原生模型、码宽度、校准权重、门控语义或行为/推荐协议。旧 `query_relay_2026_10` 源码与实验产物保持原样。

范围：共享码准备、融合动态读取、分块原地码转换，以及系统配置与集成。训练继续使用可微参考路径；优化服务路径只用于无梯度读取与状态准备。

| 入口 | 实现与边界 |
| --- | --- |
| `prepare.py` | 固定φ与当代D的全目录、各层合并物品投影；非线性位置、原生K/V特征及发布递推不变 |
| `reader.py`、`reader_kernel.py` | 直接读取共享/私有码，融合打分、ELU+1与V/Z聚合；Q≤4走FP32向量规约，较大Q走TF32x3矩阵分块 |
| `state.py` | `InplaceReleasePrefix.wrap(prefix,row_chunk=128,feature_fn=...)`；在同一持久码缓冲内分块反锚/物化 |
| `backend.py` | 显式`ImplementationConfig`与`InferenceAdapter`；一次操作RMS、各层门控打包、发布级decoder准备，不隐式fallback |
| `integration.py` | 预登记32UID、两协议、三发布、参考/仅读取/全优化完整轨迹对照 |
| `evaluate_population.py`、`population_analyze.py` | 全部已揭晓1024用户工程复验及共同UID合并统计，不作为新盲确认 |
| `cost.py` | 只CPU的完整理论账：仅扣跨层重复RMS，其余沿用冻结保守账，另报派生参数字节 |

推理接入继续使用旧`RelaySession`：加载并冻结原adapter，建立`InferenceAdapter(adapter,config)`，以`configure_prefix(...,feature_fn=partial(fixed_features_from_bank,bank=bank))`绑定共享发布表，再将两者传给`session.publish`。该接入不修改旧模型或校准源码；同步会话中的推荐与原生写入调用同一历史读取回调。候选自身不写入缓存，FIFO和出生零码保持原协议。

当前优化覆盖FP32存储、六头/头宽32、共享12＋私有6的48维码，每头读取共享3＋私有3，以及无相对位置偏置的原生ELU+1。Triton和原地状态转换严格用于无梯度推理；其他形状/训练路径应显式使用原参考，不作静默后端替换。`row_chunk=128`、目录块16384、候选批16为本轮执行配置，不是新的学习参数。

新结果、命令、配置、源文件和输入绑定、正负结果均写入 `/home/gkl/work/evokv/results/implementation_2026_10/`。设计文档最终写入 `/home/gkl/work/paper/new5.md`，只描述已落地的设计与实现，不放实验结果。

实验保留真实模型形状与独立合成压力形状，检查 Query 与行为读取、零码后缀、因果/FIFO、共享/私有坐标、反锚/物化，以及完整轨迹评分偏差。理论 FLOPs、访存模型、临时空间和算子数分别报告，不将融合读取等同于 FLOPs 降低。矩阵核TF32x3的多pass和tile补齐不当作逻辑FLOPs或DRAM实测；原生未归一化读取可很大，数值同时报告绝对值、相对值和最终评分。不重新训练或重新筛选质量用户。

检查命令（先检查GPU空闲，显式选择设备）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests/query_relay_2026_10 tests/implementation_2026_10 -q
python scripts/implementation_2026_10/run_prepare.py --help
python scripts/implementation_2026_10/run_reader.py --help
python scripts/implementation_2026_10/benchmark_state.py --help
python scripts/implementation_2026_10/evaluate_population.py --help
```

本轮实际命令、输入/源码绑定、失败过程和范围见[结果索引](/home/gkl/work/evokv/results/implementation_2026_10/README.md)。论文技术稿[第五章](/home/gkl/work/paper/new5.md)仅描述四节设计与实现，不放实验结果。

GPU 集群共享使用：每个任务先检查空闲卡，只操作自己的进程；默认先使用两张空闲卡，按需要扩展。任何额外微基准耗时只作工程诊断，不作为原20%/80%质量与FLOPs目标的替代。
