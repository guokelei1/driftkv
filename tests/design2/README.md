# Design 2 局部检查

test_geometry.py对照显式联合特征、原fit_joint方程与直接求逆参考，
检查平均目标、Cholesky和长度/mask尺度。test_report.py对照全部正负对的并列加权AUC，
检查同UID多场景不会增加重复权重。test_budget_audit.py检查预算前缀、UID去重、
条件置换的分组/快照关系和三角求解FLOPs。
test_scale_calibration.py检查非负回归的内点/边界解、独立截距对照，以及只用UID元数据的稀疏合并。
test_spectral.py对照精确三角求解，检查一般SPD、退化ridge谱、未展开方向和零输入的二次型上下界。
test_conditional.py以不同源/query维数核查精确状态收缩及数值区间，另检查负点必须回退与候选数成本边界。
真实重建闭环使用run_lifecycle.py --canary的原8UID检查，直接比较真实KV重建、追加/淘汰及下一发布失效；
两个备选复用该canary，并逐元素核对未修改的Reuse/Design1/Exact参考分支，不另造重复的模型测试套件。
按改动选择对应文件，例如 PYTHONPATH=src:scripts pytest -q tests/design2/test_budget_audit.py。
只保留会影响结论的局部检查，不运行全模型测试套件。接口缺口见
[设计审查](../../docs/design2/review.md)。
