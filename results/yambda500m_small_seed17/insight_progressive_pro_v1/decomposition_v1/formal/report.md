# Progressive PRO 五边无标签 error decomposition

状态：sealed raw 已通过完整性复核；未读取行为 label。

## 冻结规则裁决

- C32 对 Exact shared AV 的稳定方向门：cutover 2/5，rolling 0/5，FAIL。
- amplitude-dominant 门：cutover 0/5，rolling 0/5，FAIL。
- 双固定 probe 一致性：5/5，PASS。
- segment decay 相对 global decay：2/5 edge 不差，最终冻结 `global` decay。
- 第二 component：probe disagreement 为 0/5 edge，最终冻结 1 个 component。

## 逐边核心量

| phase | edge | method | median cosine | median norm ratio | median relative L2 | oracle-amplitude reduction |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| cutover | v0_to_v1 | dual_probe | 0.9175 | 0.5615 | 0.5945 | 20.7% |
| cutover | v1_to_v2 | dual_probe | 0.8792 | 0.5949 | 0.5906 | 17.4% |
| cutover | v2_to_v3 | dual_probe | 0.8838 | 0.7877 | 0.5336 | 11.9% |
| cutover | v3_to_v4 | dual_probe | 0.9167 | 0.4780 | 0.6466 | 25.0% |
| cutover | v4_to_v5 | dual_probe | 0.8479 | 0.5940 | 0.7064 | 11.6% |
| rolling | v0_to_v1 | dual_probe_global_decay | 0.7823 | 0.5768 | 0.6940 | 21.9% |
| rolling | v1_to_v2 | dual_probe_global_decay | 0.6747 | 0.5454 | 0.7325 | 12.1% |
| rolling | v2_to_v3 | dual_probe_global_decay | 0.7477 | 0.6818 | 0.7275 | 13.7% |
| rolling | v3_to_v4 | dual_probe_global_decay | 0.8365 | 0.4119 | 0.7210 | 25.0% |
| rolling | v4_to_v5 | dual_probe_global_decay | 0.5805 | 0.5827 | 0.8194 | 9.3% |

## 结论

两条 probe 的 correction 几乎相同，因此单 probe 偶然性不是当前主要误差源。C32 对 Exact 的方向门和纯幅值解释均未通过；old/recent segment decay 也未达到 4/5 选择门。按事前协议不调阈值、不追加第二 component，下一步只在同一 PRO 内测 C32/C48/C64 carrier fidelity 轴，保留 global decay。

score aggregate 保存在 `score_aggregate.csv`；它只衡量无标签的 Current/Exact-shared 距离，不构成 AUC、log-loss 或 serving admission。
