# 本轮统一训练配置

Medium最终V0–V5已固定，见[模型链索引](../../results/unified_training_2026_09/medium/README.md)。V0/V1为历史复用；以下为本轮保留版本的执行设置。

- [V2：2 epochs、四卡global32、CPU14](medium_v2_4gpu_b32_cpu14_execution.yaml)
- [V3：1 epoch、四卡global32、CPU14](medium_v3_4gpu_b32_cpu14_execution.yaml)
- [V4：1 epoch、四卡global32、CPU14](medium_v4_4gpu_b32_cpu14_execution.yaml)
- [V5：1 epoch、四卡global32、CPU14](medium_v5_4gpu_b32_cpu14_execution.yaml)
- [仅E14评测](evaluation_e14_only.yaml)

各执行配置引用`configs/contracts/`中的对应冻结合同。入口`scripts/unified_training/run_medium_v2_2epoch.py`必须显式传入`--execution-config`选择保留配置；现已支持按合同读取Medium/Large数据和最终epoch端点。历史运行记录中的代码哈希描述当时版本，已完成输出不得覆盖。

- [Large V4：2 epochs、四卡global64、CPU14](large_v4_4gpu_b64_cpu14_execution.yaml)：从固定V3训练，仅保留最终2 epochs checkpoint并评估E14。
- [Large V5：保存epochs 1/2、四卡global64、CPU14](large_v5_4gpu_b64_cpu14_execution.yaml)：从本轮达标V4连续训练，两个端点在同一E14上分别评测；保留day300不完整的评价范围标记。

被替代的执行配置和任务已按用户要求清理。原合同、最终运行配置、轻量canary通过证明及资源估计保留；探针payload已删除。目录存在不代表新训练授权。

## Max 准备

- [5B/20万用户数据处理](max_data_preparation.yaml)：人口资格、稳定选择、独立词表和显式OOV布局。
- [选定batch80资源探针](max_probe_b80_execution.yaml)：四卡每卡20，仅12步canary。
- [V0执行配置](max_v0_prepared_execution.yaml)：1 epoch、四卡global80、CPU14；V0已完成并封存。
- V0恢复策略：第500步首存、以后每4000步保存，最近两份完整恢复点；包含AdamW及随机状态，同四卡配置以`run_max_v0.py --resume`续训。
- [资源结果及预算](../../results/unified_training_2026_09/max/README.md)：数据审计通过，V0已启动。

- [Max V1：连续2epochs、四卡global80、CPU14](max_v1_epochs12_4gpu_b80_cpu14_execution.yaml)：2026-09-19用户授权启动，分别保留epochs1/2，在同一完整E14上与固定V0比较。
