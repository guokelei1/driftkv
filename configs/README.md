# Configs

当前 37D 路线不复活旧实验配置。Yambda-50M 的 timestamp-corrected development contract 位于 `contracts/yambda50m_v2.yaml`。CC 的 P5 seen-aware 合同位于 `contracts/cc_p5_seenmix_v1.yaml`（failed gate，不得改判）；P6 identifiability 合同位于 `contracts/cc_p6_identifiability_v1.yaml`（分支 B / next-item No-Go）。P7 的 Yambda multi-regime 审计边界位于 `contracts/p7_yambda_stateful_suite_v1.yaml`，P8 的冻结 release-chain 合同位于 `contracts/f_release_chain_contract_v1.yaml`。P8 已给出 development H/S 证据，但仍不是正式论文 qualification；当前只进入 P9 tomography 与 action-space qualification。

旧的 D1/D2/D3、foundation、root-cause 和 CohortKV 配置已清理。后续实验必须围绕当前 contract、candidate manifest 和 lineage 重新建立。
