# P8 Development Result Summary

更新日期：2026-08-21。P8 的冻结合同为
[`f_release_chain_contract_v1`](../configs/contracts/f_release_chain_contract_v1.yaml)。本页只总结
development 结论；它不是论文 qualification，也不授权 controller。

## 结果

F 是唯一进入 release chain 的主长期状态 workload：M0-F 为单任务主模型，M1-F 为共享多任务
companion；seed 17/37/71 均完整保留。R0、R1 edge1、R1 edge2、R2 的所有 update checkpoint
都通过既有 release admission。

| Release | M0-F S: Current Full vs Reuse JS | M1-F S: Current Full vs Reuse JS | 解释 |
| --- | ---: | ---: | --- |
| R0 output-only | <= 4.44e-15 (all conditions) | <= 4.44e-15 (all conditions) | cache producer 未变，No-op 合法 |
| R1 edge1 | 9.07e-5 [5.64e-5, 1.53e-4] | 5.38e-5 [2.40e-5, 9.16e-5] | routine update 产生稳定 S |
| R1 edge2 | 3.61e-4 [3.83e-5, 9.95e-4] | 2.67e-5 [1.28e-5, 4.44e-5] | 第二条 routine edge 复现 |
| R2 encoder refresh | 4.02e-4 [1.01e-5, 6.23e-4] | 1.35e-3 [4.79e-5, 3.75e-3] | refresh 下陈旧性更强，尤其 M1 |

方括号为 equal-seed hierarchical 95% CI。每个 R1/R2 cell 的 H 与 S 均通过冻结的
target-free edge-staleness candidate 门：H CI 高于 P7 numeric floor、S CI 高于 R0 floor、至少
两个 seed 的 S 高于地板、所有 seed admission 通过，且质量 companion 已完整报告。

最强的 quality 证据位于 M1-F R2。相对 Reuse Parent KV，Current Full 的改善为：

- log loss：+0.00327，CI [0.00073, 0.00672]；
- ROC-AUC：+0.01331，CI [0.00229, 0.03143]；
- dislike PR-AUC：+0.01773，CI [0.00017, 0.04630]。

因此 P8 支持的 development 表述是：长期 F state 的版本兼容性取决于 release semantics；
output-only release 可以 No-op，cache-producing update 会暴露状态债务，refresh 可同时造成
target-free fidelity 和实际 F quality 损失。

## 强制限制

- Current Full 是当前模型的执行语义参考，不是未来用户质量的理论上界。
- R1 的 S 虽稳定非零，但大多数 F quality companion 的 CI 不稳定穿过零；不能说每条 routine
  edge 都有稳定质量损失。
- rare-dislike 的 `dislike-only log loss` 继续作为强制 companion。R2 的 Full-vs-Recent
  在 M0-F/M1-F 上恶化，M1-F R2 的 Full-vs-Reuse 也没有稳定正向 CI；不得隐藏或事后变更 P8 gate。
- N/R 仍是 M1 共享状态对照，不升级为主长期状态 workload；P7 的 H qualification 不变。
- 所有数字均为 development evidence；P8 合同要求完成后停止，不能自动启动 tomography 或 controller。

原始结果：
[`R0`](../results/p8/r0_control/adjudication_v1.json)、
[`R1 edge1`](../results/p8/r1_edge1/hs_adjudication_v1.json)、
[`R1 edge2`](../results/p8/r1_edge2/hs_adjudication_v1.json)、
[`R2`](../results/p8/r2/hs_adjudication_v1.json)。
