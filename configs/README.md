# 配置与合同

[contracts/](contracts/) 保存冻结的执行与证据边界。文件完成或路线退休后仍不可改写，
也不因脚本清理而删除。README 仅是导航，不是执行授权。

现有 Medium 六层、Large 十层模型及评估来自对应冻结合同；历史 Small 和各类 Insight 合同仍作为审计记录。
保留的工具入口见 [scripts/README.md](../scripts/README.md)，下一阶段方案草稿见
[experimental_design.md](../docs/experimental_design.md)。

当前 EvoKV 方法为 Cross-Version Cache Adaptation，尚没有已运行的 translator 或连续迁移合同。
长训练、calibration 的新监督协议和正式人口评价需要相应前瞻协议、资源估计、
focused canary 与用户启动。日常小规模探索只记录复现所需的配置，优先使用现有配置入口，
不为每次尝试新增冻结合同或通用校验框架；进入正式评价前再冻结实际采用的设置。
theta3 和 RecFlow 原有隔离与授权边界不变。
