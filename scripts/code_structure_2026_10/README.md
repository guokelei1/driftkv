# 紧凑码结构探索

这一轮只改变 D1 的码表示及 D2 的联合校准，不添加 D3 模块。
基线是每 head 8 维、每层每历史行共 48 个 FP32 数，不是 8 字节。

候选保留相同总码存储：每两 head 共享 16 维；全局共享 6 维加每 head 私有 7 维；
全局共享 12 维加每 head 私有 6 维。共享部分输入其覆盖 head 的原生 K/V 与完整 Current
item embedding，decoder 可使用所有可见共享维度。每个候选精确包含旧 head-local 函数，
初始化时跨 head 连接为零，再通过原教师目标训练；增加连接后的真实质量需要实验判断。

保留单份原生 K/V、原生写入、按需初始化、追加/FIFO 和因果标记。
复用原 local32 检查点，替换 joint16；模型冻结，256 fit /16 validation 用户与原角色相同。
评测以原固定 4096 开发用户和三发布窗口为整体，不挑选有利用户或发布。
恢复率是等发布平均 AUC 增益除以 once-Rebuild 相对 Reuse 的增益。
成本另用既有 fixed30k 真实事件账，计入继承 local32、新 joint16、全部编码/追加/读取和
4 worker 独立静态 4k item 投影表准备；报告源 K/V 重建敏感性和原生 query 开销。

先 CPU 函数包含与表示能力检查，再 GPU canary，再做固定预算的开发筛选。
所有新结果写入 `results/code_structure_2026_10/`；旧权重和结果只读。
开发筛选和固定面板结果仍是探索证据，不能称为独立确认。

基本命令（其他结构/版本替换参数，输出必须是新目录）：

```bash
python scripts/code_structure_2026_10/verify_structure.py --output results/code_structure_2026_10/structure_checks.json
python scripts/code_structure_2026_10/verify_serving.py
python scripts/code_structure_2026_10/calibrate.py --layout shared6_private7 --target v3 --gpu 0 --output results/code_structure_2026_10/shared6_private7/calibration/v3
python scripts/code_structure_2026_10/cost.py --layout shared6_private7 --calibration-root results/code_structure_2026_10/shared6_private7/calibration --output results/code_structure_2026_10/shared6_private7/cost.json
python scripts/code_structure_2026_10/evaluate.py --calibrations results/code_structure_2026_10/shared6_private7/calibration --users 512 --gpu 0 --output results/code_structure_2026_10/shared6_private7/quality512
python scripts/code_structure_2026_10/verify_replay.py --quality results/code_structure_2026_10/shared6_private7/quality512
```

继续其余用户用 `--users 0 --start-user 512 --shards 2 --shard-index 0/1`，
避免重复前512用户。`aggregate.py --inputs`合并首512与两个互斥余下分片，
UID bootstrap 同时覆盖全部选定用户和三发布；原始字段 `partial_logit` 在本轮只作
分析接口兼容，指新结构完整行码评分，所有行正常生成，不是部分刷新。
