# 共享历史数据与指标工具

此目录只保留当前Design仍调用的历史装载、面板核对和配对AUC工具。
旧Design 1各轮训练、扫描和GPU执行入口已退休；重要设置与结果保留在
[历史迭代](../../docs/design/iterations.md)及[结果索引](../../results/README.md)。
当前方法入口是[local-KV校准](../design_system/calibrate_lightweight_local_kv.py)与
[继承flag范围评测](../design_system/evaluate_lightweight_support_mask.py)。
