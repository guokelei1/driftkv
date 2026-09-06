# 配置与合同

[contracts/](contracts/) 保存冻结的执行与证据边界。文件完成或路线退休后仍不可改写，
也不因脚本清理而删除。README 仅是导航，不是执行授权。

现有 Medium 六层、Large 十层模型及评估来自对应冻结合同；历史 Small 和各类 Insight 合同仍作为审计记录。
保留的工具入口见 [scripts/README.md](../scripts/README.md)，本阶段开发见
[design/plan.md](../docs/design/plan.md)，技术定义见 [experimental_design.md](../docs/experimental_design.md)。

当前 EvoKV 方法为 Cross-Version Cache Adaptation，尚没有已运行的 translator 或连续迁移合同。
2026-09-06 用户已撤销笼统 target-KV fitting 禁令；小规模摘要/共享 Translator 校准按计划
记录配置、划分、监督和资源，不因 K/V-derived 目标另设合同或审批。旧 sealed 合同中的禁止训练/
拟合条款仅约束原实验，原文保留。长训练和正式人口评价需要相应前瞻协议、资源估计、
focused canary 与用户启动。日常小规模探索只记录复现所需的配置，优先使用现有配置入口，
不为每次尝试新增冻结合同或通用校验框架；进入正式评价前再冻结实际采用的设置。
theta3 和 RecFlow 原有隔离与授权边界不变。
