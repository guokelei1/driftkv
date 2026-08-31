# Yambda-500M Medium 全轮实验总结（专家讨论稿）

更新时间：2026-08-29  
状态：**本轮 Medium seed17 训练、Full-only、D7/D14 adjacent one-hop Reuse 以及 D14 v5 扩展均已完成并封存；Medium PRO 尚未启动。**

## 1. 一句话结论

本轮总体是好消息，但结论不是“Reuse 已经足够好”，而是：

> **30k-user、6L/H192 Medium 已形成比 Small 更稳定的模型更新环境，并再次确认 persistent Parent K/V 会以显著且 edge-dependent 的方式阻止 New 模型收益兑现。D14 明显比 D7 更适合作为下一阶段的主研究环境；新增 v4→v5 在完整 E3/E7 上取得 +3.14%/+2.83% 的相对 AUC 提升，Reuse 仅保留其中 48.79%/52.34%，是目前最干净的 Medium 动机证据之一。**

完整结果同时暴露了边界：D7 的更新信号和 Reuse recovery 都很不稳定；D14 也不是每条 edge 都稳定，尤其 v3→v4 的 Reuse 在 E3/E7 低于 Old，E14 仅保留 12.84%。因此 Medium 已经足以支持继续验证 frozen PRO，但还不能宣称机制已经跨 scale 通过，更不能据此准入 serving action。

## 2. 本轮实际完成了什么

| 项目 | 已完成范围 | 状态 |
| --- | --- | --- |
| 共享 foundation | v0，训练 `[0,217)` | 完成并封存 |
| D7 checkpoint chain | v1…v10，每版增量 7 天 | 10/10 完成 |
| D14 checkpoint chain | v1…v4，每版增量 14 天 | 4/4 完成 |
| D14 v5 扩展 | v4→v5，训练 `[273,287)` | 完成并独立封存 |
| 原始 Full-only matrix | D7 20 格 + D14 12 格 | 32/32 完成 |
| 原始 D14 Reuse | 4 edges × E3/E7/E14 | 12/12 完成 |
| D7 forced-Reuse diagnostic | 10 edges × E3/E7 | 20/20 完成 |
| D14 v5 Full + Reuse | E3、E7、E14_partial | 3+3 完成 |

最终共保留 16 个正式 checkpoint：共享 v0、D7 v1…v10、D14 v1…v4，以及独立扩展的 D14 v5。

## 3. 冻结实验设置

| 项目 | Medium 设置 |
| --- | --- |
| 数据 | Yambda-500M unified Medium population |
| 固定人口 | 30,000 users |
| Seed | 17 |
| 模型 | HSTU-native CC，6 layers，hidden 192，6 heads |
| Context | 1024 |
| 参数量 | 266,259,265 |
| Item vocabulary | 1,380,509 known items + 256 stable OOV buckets |
| 训练目标 | F-only binary cross-entropy |
| Foundation | `[0,217)`，1 pass |
| 增量训练 | direct-parent warm start，fresh AdamW，1 pass |
| Foundation LR | 2e-4 |
| Update LR | 5e-5 |
| D7 | 10 个 7-day update，评测 E3/E7 |
| D14 | 5 个 14-day update；v1…v4 评测 E3/E7/E14，v5 评测 E3/E7/E14_partial |
| Reuse | adjacent one-hop；cutover 前 Parent cache，cutover 后由 Current append |
| 禁止项 | recursive Reuse、long-age Reuse、future-label scheduling、serving promotion |

基础 manifest 使用完整 `[0,300)` 数据。v5 扩展为了复现名义 `[287,301)` 窗口，单独物化了包含 day300 partial tail 的 manifest；day300 只有 12,962 条原始 feedback row，最后事件位于当日第 79,995 秒，所以该窗口只能称为 `E14_partial`。

## 4. 指标口径

每个结果都来自同一 sealed raw 中严格对齐的三条路径：

- **Old**：Parent 模型及其自身 rolling cache；
- **New**：Current 模型及其完整 Current rolling cache；
- **Reuse**：Current 模型读取 Parent 在 cutover 前生成的 cache，之后由 Current append。

本文只在同一三路径 cohort 内计算比例：

```text
New AUC vs Old (%)
  = (AUC_New - AUC_Old) / AUC_Old × 100

Reuse AUC retained (%)
  = (AUC_Reuse - AUC_Old) / (AUC_New - AUC_Old) × 100

New loss reduction (%)
  = (Loss_Old - Loss_New) / Loss_Old × 100

Reuse loss retained (%)
  = (Loss_Old - Loss_Reuse) / (Loss_Old - Loss_New) × 100
```

解释规则：

- recovery 为 100%：Reuse 完整保留 Old→New 收益；
- 0%：Reuse 回到 Old；
- 负数：Reuse 比 Old 更差；
- 大于 100%：Reuse 在该指标上超过 New，通常要结合很小的 Old→New 分母谨慎解读；
- New 没有优于 Old 时，AUC recovery 记为 `N/A`，不把负分母包装成“恢复”。

原顶层 `summary.md/json` 的 Full-only Old/New 与 rolling Reuse 来自不同执行对象，适合分别审计，但不应用于精确 recovery 混算。本文的 recovery 均取各 Reuse adjudication 内的同 cohort `three_path_summary`；这也是后续讨论应使用的统一口径。

## 5. D7 全部结果

D7 的正式 Full-only admission 在第一条 v0→v1 上失败，因此整个 accepted diagnostic lineage 被锁住。后来按用户要求执行的 20 格 Reuse 全部属于 **forced diagnostic**：它补齐观察，但没有修改 admission seal、serving parent 或 cache lineage。

| Edge | Window | Requests | New AUC vs Old | Reuse AUC retained | New loss reduction | Reuse loss retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0→v1 | E3 | 30,413 | +3.69% | +87.37% | +1.09% | +92.94% |
| v0→v1 | E7 | 72,053 | +4.45% | +78.96% | +1.17% | +81.89% |
| v1→v2 | E3 | 30,064 | +0.25% | -61.19% | +0.34% | +65.07% |
| v1→v2 | E7 | 68,627 | +0.03% | -530.05% | +0.21% | +74.22% |
| v2→v3 | E3 | 29,987 | +0.90% | +50.81% | -0.20% | N/A |
| v2→v3 | E7 | 68,541 | +0.75% | +50.65% | +0.05% | +116.95% |
| v3→v4 | E3 | 27,510 | +0.27% | -148.64% | -0.21% | N/A |
| v3→v4 | E7 | 64,727 | +0.12% | -293.89% | -0.04% | N/A |
| v4→v5 | E3 | 27,000 | +1.36% | -42.48% | +0.78% | +37.77% |
| v4→v5 | E7 | 66,320 | +1.32% | +3.02% | +0.59% | +53.06% |
| v5→v6 | E3 | 26,926 | -0.42% | N/A | +0.59% | +81.39% |
| v5→v6 | E7 | 65,254 | +0.24% | +243.78% | +0.38% | +106.62% |
| v6→v7 | E3 | 26,643 | +3.75% | +21.29% | +0.89% | +36.11% |
| v6→v7 | E7 | 66,529 | +3.72% | +21.89% | +0.17% | -8.76% |
| v7→v8 | E3 | 29,242 | -0.51% | N/A | +0.19% | +26.93% |
| v7→v8 | E7 | 67,842 | -0.07% | N/A | +0.71% | +75.72% |
| v8→v9 | E3 | 27,306 | +0.58% | -9.38% | +0.28% | +34.72% |
| v8→v9 | E7 | 65,995 | +0.41% | -39.19% | +0.35% | +42.54% |
| v9→v10 | E3 | 31,388 | +1.22% | -5.89% | +1.05% | +11.61% |
| v9→v10 | E7 | 72,074 | +2.28% | +37.54% | +1.32% | +45.70% |

D7 的直接读法：

- New AUC 在 20 格中 17 格为正，说明 7-day update 不是完全无效；
- 但在这 17 个正增益格中，Reuse 只有 9 格保留正 AUC 收益；
- positive-New 格子的 Reuse recovery 中位数仅 3.02%，且大量极端正负值来自接近零的 Old→New 分母；
- v0→v1 很强，但随后不形成稳定、单调的 release chain；
- 因此 D7 适合作为噪声/边界诊断，不适合成为本轮主 qualification recipe。

## 6. D14 v0→v4 全部结果

D14 的 v0→v1、v1→v2、v2→v3、v3→v4 均通过事前 Full-only primary-horizon admission，并完成全部 12 个 adjacent-Reuse cell。

| Edge | Window | Requests | New AUC vs Old | Reuse AUC retained | New loss reduction | Reuse loss retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0→v1 | E3 | 30,064 | +3.13% | +61.19% | +1.69% | +75.50% |
| v0→v1 | E7 | 68,627 | +3.75% | +68.21% | +1.55% | +77.92% |
| v0→v1 | E14 | 137,168 | +3.71% | +78.04% | +1.43% | +81.99% |
| v1→v2 | E3 | 27,510 | -0.37% | N/A | +0.35% | +19.97% |
| v1→v2 | E7 | 64,727 | +0.16% | -10.83% | +0.37% | +49.07% |
| v1→v2 | E14 | 131,047 | +0.20% | +26.95% | +0.28% | +52.84% |
| v2→v3 | E3 | 26,926 | +2.07% | +33.63% | +1.26% | +71.58% |
| v2→v3 | E7 | 65,254 | +1.90% | +43.67% | +0.76% | +47.18% |
| v2→v3 | E14 | 131,783 | +2.05% | +45.55% | +0.76% | +55.09% |
| v3→v4 | E3 | 29,242 | +1.42% | -2.29% | +0.55% | -40.48% |
| v3→v4 | E7 | 67,842 | +1.56% | -2.74% | +0.42% | -12.00% |
| v3→v4 | E14 | 133,837 | +2.46% | +12.84% | +0.65% | +9.22% |

D14 的 edge-level 读法：

- **v0→v1：** 最强的早期结果，New AUC +3.13%～+3.75%，Reuse 保留 61.19%～78.04%；说明 Reuse 有价值，但仍丢失 22%～39% 的模型升级收益。
- **v1→v2：** release signal 很弱，E3 甚至为负；这里的 recovery 对小分母高度敏感，不适合作为机制成败的主证据。
- **v2→v3：** New AUC 稳定提升约 1.90%～2.07%，Reuse 只保留 33.63%～45.55%；这是清晰的 compatibility debt。
- **v3→v4：** New AUC 提升并不弱，但 Reuse 在 E3/E7 低于 Old，E14 也只保留 12.84%；这是不能删除的核心反例和 safety/admission 动机。

## 7. D14 v4→v5 扩展结果

v5 使用完整 `[273,287)` 训练窗口。E3/E7 是完整评测；`E14_partial` 包含不完整 day300，只能作为方向性诊断。

| Edge | Window | Requests | New AUC vs Old | Reuse AUC retained | New loss reduction | Reuse loss retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v4→v5 | E3 | 31,388 | +3.14% | +48.79% | +0.83% | +61.68% |
| v4→v5 | E7 | 72,074 | +2.83% | +52.34% | +1.11% | +62.36% |
| v4→v5 | E14_partial | 138,734 | +1.59% | +46.74% | +0.33% | +30.66% |

v5 是本轮很重要的新增证据：

- 完整 E3/E7 上 New AUC 分别提升 3.14% 和 2.83%，不是弱 release edge；
- Reuse 只兑现约一半 AUC 收益，E3/E7 recovery 为 48.79%/52.34%；
- loss 也只保留 61.68%/62.36%；
- 因此这里同时具备“New 明确更好”和“Reuse 明确阻碍收益兑现”，比弱边更适合后续验证 frozen compatibility correction；
- `E14_partial` 的方向与 E3/E7 一致，但不能进入完整 E14 统计或 qualification gate。

## 8. 聚合对比

下面均为 cell 等权的描述性汇总，不替代逐 edge 报告：

| Result family | Cells | New AUC 正增益 | 正增益中 Reuse 仍为正 | New AUC 相对提升均值 | positive-New recovery 中位数 | New loss 改善 | 其中 Reuse loss 保留为正 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D7 forced diagnostic | 20 | 17/20 | 9/17 | +1.22% | +3.02% | 17/20 | 16/17 |
| D14 v0→v4 | 12 | 11/12 | 8/11 | +1.84% | +33.63% | 12/12 | 10/12 |
| D14 v4→v5 | 3 | 3/3 | 3/3 | +2.52% | +48.79% | 3/3 | 3/3 |

这张表支持两个判断：

1. **D14 比 D7 更稳定。** D14 的 New gain 更大，Reuse recovery 的中位数也明显更可解释；D7 大量 edge 的分母接近零，导致 recovery 极端波动。
2. **规模增大没有消除 compatibility debt。** Medium 上 New 模型可以稳定变好，但 Parent cache 仍经常只保留一部分收益，甚至把结果拉回 Old 以下。

## 9. Admission 与证据性质

必须区分三类结果：

1. **原始 Full-only admission evidence**
   - D7：v0→v1 未通过 primary E7 gate，后续 candidate 不进入 accepted diagnostic lineage；
   - D14 v0→v4：四条 edge 均通过事前 gate，Reuse 正式解锁。
2. **D7 forced-Reuse diagnostic**
   - 按用户要求补齐所有 20 格；
   - bypass 的只是“是否执行 Reuse”的锁，不是 release admission；
   - 不修改原 admission seal、serving parent 或 cache lineage。
3. **D14 v5 extension**
   - v5 训练、E3/E7 Full/Reuse 均完整；
   - 因没有完整 E14，未形成可与原 D14 primary gate 等价的正式 admission；
   - `E14_partial` 永远不能作为完整 E14 qualification。

## 10. 实际运行成本与并行设置

### 10.1 训练

| 阶段 | 拓扑 | 实测时间 |
| --- | --- | ---: |
| Shared v0 `[0,217)` | GPU2/3，2 ranks，global batch 32 | 219.48 min |
| D7 v1…v10 | GPU2/3，2 ranks，global batch 32 | 153.89 min，总计；15.39 min/version |
| D14 v1…v4 | GPU2/3，2 ranks，global batch 32 | 97.80 min，总计；24.45 min/version |
| D14 v5 | GPU0/1/2/3，4 ranks，global batch 32 | 19.58 min |

训练全程 1 epoch/pass，未 early stop，未按结果挑 checkpoint。

### 10.2 评测

| 阶段 | 实测时间/设置 |
| --- | --- |
| 原 D7 Full-only 20 格 | 225.85 min 总计；双卡原 runtime |
| 原 D14 Full-only 12 格 | 103.58 min 总计；双卡 + 28 physical CPU workers |
| D7 forced Reuse 20 格 | 251.79 min 总计；四卡；E3 约 6.7–7.5 min，E7 约 17.8–19.2 min |
| D14 四卡 Reuse | cohort32/rank，query chunk256/rank；E3 约 7.6 min，E7 约 19.9 min，E14 约 44.3 min |
| v5 Full | E3 2.84 min，E7 5.24 min，E14_partial 8.49 min；batch128/rank |
| v5 Reuse | E3 7.88 min，E7 19.92 min，E14_partial 44.87 min |

四卡评测使用 GPU0/1/2/3，每 rank 14 个互不重叠的物理 CPU 核，共 56 核。Reuse 的 cohort32/query256 已接近安全显存上限：原 D14 正式矩阵最坏 rank 的 peak reserved 达 44,950 MiB，而 A40 总显存为 46,068 MiB，因此没有继续扩大 batch。

## 11. 本轮可以得出的结论

### 11.1 已支持

- Medium 30k/6L 环境可以产生真实且较稳定的 Full/New 提升；Small 不是唯一有效环境。
- Persistent Parent K/V 会阻止 Current 模型收益完全兑现，且影响显著依赖 release edge。
- D14 是比 D7 更合适的主实验 recipe。
- v3→v4 证明 compatibility correction 必须有 label-free safety/admission 或 No-op fallback；不能假设每条 edge 自动受益。
- v4→v5 提供新的强 release edge：New 明确提升，而 Reuse 只保留约一半收益，值得用于 frozen Design 的后续验证。

### 11.2 尚未支持

- 尚未在 Medium 上复核 candidate-shared signed correction、AV boundary 与跨请求 persistence。
- 尚未运行 Medium 的 frozen C32 PRO，也未比较约 10%/20% Exact FLOPs 两个预算点。
- 尚无额外 training seed；seed17 仍是唯一统计重复单位。
- 当前 wall-clock 是研究 evaluator 的执行记录，不等价于 serving GPU compute/I/O/state-write 收益。
- 不能用 D7 forced diagnostic 或 v5 `E14_partial` 形成正式 release qualification。

## 12. 建议的下一阶段

1. 冻结本轮全部 Medium 结果，不再按这些 edge 调 release recipe、probe、carrier 或 scale。
2. 以 D14 为主环境，只复核 Small 已发现的三个核心 Insight gate：candidate-shared signed correction、AV 形成边界、跨真实请求 persistence。
3. 若 Insight gate 仍成立，直接验证冻结 PRO 的约 10% 与 20% Exact-FLOPs 两个预算点；完整报告所有事前指定 edge，不选择性删除 v3→v4。
4. v4→v5 的 E3/E7 可作为强动机/机制复核 evidence；`E14_partial` 只做一致性参考。
5. 真实质量通过后，再做额外 seed 和 serving runtime qualification。

建议交给专家讨论的核心问题是：

> **Medium 已确认 D14 上存在稳定但 edge-dependent 的 cache compatibility debt；下一步是否应直接在冻结 D14 edges 上复核 Insight 并运行 frozen PRO，还是先增加一个全新的 Medium seed 作为 prospective qualification？**

## 13. 证据索引

- 基础合同：[`configs/contracts/yambda500m_medium_hstu_native_d7_d14_full_reuse_v1.yaml`](../../../configs/contracts/yambda500m_medium_hstu_native_d7_d14_full_reuse_v1.yaml)
- 原始执行/admission 合同：[`configs/contracts/yambda500m_medium_hstu_native_d7_d14_execution_admission_v1.yaml`](../../../configs/contracts/yambda500m_medium_hstu_native_d7_d14_execution_admission_v1.yaml)
- 四卡 Reuse runtime：[`configs/contracts/yambda500m_medium_hstu_native_d14_reuse_4gpu_runtime_v3.yaml`](../../../configs/contracts/yambda500m_medium_hstu_native_d14_reuse_4gpu_runtime_v3.yaml)
- D7 forced diagnostic 合同：[`configs/contracts/yambda500m_medium_hstu_native_d7_forced_reuse_diagnostic_v1.yaml`](../../../configs/contracts/yambda500m_medium_hstu_native_d7_forced_reuse_diagnostic_v1.yaml)
- D14 v5 合同：[`configs/contracts/yambda500m_medium_hstu_native_d14_v5_extension_v1.yaml`](../../../configs/contracts/yambda500m_medium_hstu_native_d14_v5_extension_v1.yaml)
- 原始矩阵 summary：[`summary.md`](summary.md)
- D7 forced diagnostic summary：[`D7/forced_reuse_diagnostic_v1/summary.md`](D7/forced_reuse_diagnostic_v1/summary.md)
- D14 v5 summary：[`D14/v5_extension_v1/summary.md`](D14/v5_extension_v1/summary.md)
- 训练/执行方案：[`docs/medium_scale_training_plan.md`](../../../docs/medium_scale_training_plan.md)

所有原始分数均先于 label join 封存；各目录内的 `raw.seal.json`、`adjudication.json`、checkpoint seal 和合同 hash 是最终可审计依据。本讨论稿不替代这些 seal。
