# 核心 Motivation 与 Observation

更新日期：2026-08-25

本文是当前论文最详细的结果文档。它只记录已经观察到的 HSTU-native motivation 和版本化状态 observation；没有观察到的系统设计效果、最终 EvoKV action 或 scheduler 结论不写成已验证结果。

## 1. 当前结论

在 Yambda-500M Small 的 HSTU-native foundation 上，模型更新带来的发布收益与 persistent KV 的跨版本兼容性是两个不同问题：

- Current Full 在多条连续版本边上优于 Parent Full；
- 直接复用父版本 prefix KV 会使当前模型的 rolling 质量低于 Current Exact Rolling；
- 在四条常规正收益边上，One-hop Reuse 侵蚀了 25.5%–47.9% 的模型发布 AUC 收益；
- 直接 Reuse 的损失随 producer version age 增长，在 long-age direct matrix 中呈严格单调关系；
- 这支持“版本化 persistent state 会阻碍新模型收益兑现”这一 motivation；
- 这还没有证明 recursive lineage debt、最终 migration policy 或 scheduler 应该采用哪种具体设计。

## 2. 当前结果的实验对象

- 数据：Yambda-500M，Explicit Feedback；
- 时间线：约 300 天；
- foundation：Day 0–217；
- 当前 Small：固定 UID hash 人口，约 10,000 用户；
- 模型：HSTU-native，4L/H128/context512；
- seed：17；
- item mapping：foundation cutoff 前固定，未来 item 使用 256 个稳定 OOV bucket；
- 版本链：v0 → v1 → v2 → v3 → v4 → v5；
- 核心切片：每次 update D=14 天，发布后观察 E=14 天；
- 评测：同一用户、同一 causal history、同一 query/target/candidate、同一当前模型与 readout。

## 3. 对照定义

### 3.1 Full-only release gain

Parent Full 和 Current Full 都在同一未来请求上完整重算：

~~~text
Release gain = Current Full - Parent Full
~~~

该差值只回答新模型是否比父模型好。它不读取 Reuse、JS、KV distance、release debt 或 scheduler 输出，因此不会用 compatibility 结果决定模型 admission。

### 3.2 Rolling reuse harm

Current Exact Rolling 和 One-hop Reuse Rolling 共享同一 rolling 执行语义：

- Current Exact Rolling：cutover prefix 由当前模型重算，随后当前模型 append；
- One-hop Reuse Rolling：cutover prefix 使用父版本 KV，随后当前模型 append；
- 两条路径使用相同的 query、target、candidate、eviction 和 append 规则。

~~~text
Reuse harm = Current Exact Rolling - One-hop Reuse Rolling
~~~

对 log-loss 使用相反方向：

~~~text
Reuse loss = LogLoss(One-hop Reuse) - LogLoss(Current Exact)
~~~

## 4. D=14、E=14 核心 AUC 结果

所有 AUC 差值单位为 percentage points。

| 版本边 | Parent Full | Current Full | 发布收益 | Current Exact Rolling | One-hop Reuse | Reuse 损失 | 侵蚀比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 → v1 | 0.669198 | 0.681196 | +1.199879 | 0.681769 | 0.678709 | +0.306014 | 25.5% |
| v1 → v2 | 0.663884 | 0.670146 | +0.626216 | 0.670493 | 0.668496 | +0.199662 | 31.9% |
| v2 → v3 | 0.611481 | 0.614582 | +0.310144 | 0.615001 | 0.613514 | +0.148693 | 47.9% |
| v3 → v4 | 0.619490 | 0.619953 | +0.046331 | 0.620994 | 0.619054 | +0.193984 | 418.7%* |
| v4 → v5 | 0.549289 | 0.551495 | +0.220611 | 0.551858 | 0.551221 | +0.063736 | 28.9% |

* v3→v4 的绝对 Reuse 损失真实存在，但 Full-only 发布收益很小，导致百分比分母放大。它是高风险尾部案例，不是 headline 平均效果。

四条常规正收益边的侵蚀比例为 25.5%–47.9%。这组结果同时展示了：

1. 新模型可以确实变好；
2. 父 KV 可以阻碍新模型兑现收益；
3. 侵蚀比例不是固定常数；
4. 不能因为某一条边不 harmful 就否定整个问题，也不能因为某一条边很大就声称所有 release 都有害。

## 5. Companion 质量与表示结果

| 版本边 | Full-only PR-AUC Parent → Current | Rolling PR-AUC Current → Reuse | Reuse − Current event log-loss | 用户等权 log-loss 差 | Bernoulli JS |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0 → v1 | 0.193987 → 0.206501 | 0.206536 → 0.204757 | +0.000502 | +0.000089 | 1.69e-05 |
| v1 → v2 | 0.186811 → 0.194637 | 0.194268 → 0.192455 | +0.000274 | +0.000110 | 3.18e-06 |
| v2 → v3 | 0.152918 → 0.153150 | 0.153849 → 0.153327 | +0.000195 | −0.000041 | 2.31e-06 |
| v3 → v4 | 0.159239 → 0.160395 | 0.159553 → 0.159566 | −0.000080 | −0.000095 | 2.27e-05 |
| v4 → v5 | 0.133822 → 0.135537 | 0.134760 → 0.134218 | +0.000042 | +0.000005 | 2.31e-06 |

AUC 的损害不需要在所有 loss 口径上同幅出现。它可能集中在排序边界、部分用户或部分请求，因此主结论使用配对 AUC release gain 与 rolling Reuse harm，同时保留 PR-AUC、log-loss、Brier、用户等权差异和 JS 作为 companion。

## 6. 版本年龄 observation

在 direct long-age matrix 中，producer 版本越旧，Current 与 Reuse 的质量差异越大。D=14 的 long-age 结果显示：

- one-hop 是最小年龄的直接父版本对照；
- 更老 producer 的 Reuse 损失随版本年龄严格单调增长；
- v4←v0 的 ROC-AUC 损失约为 one-hop 的 3.6 倍；
- 这支持 version age 是 persistent-state compatibility 的重要维度。

该结论是 direct Reuse age observation，不是 recursive lineage debt 的证明。Recursive lineage 需要单独的真实 append/eviction chain 和预注册质量评测。

## 7. 当前 observation 的系统含义

目前可以形成的证据链是：

~~~text
模型发布收益存在
  -> 父版本 KV 与当前模型不完全兼容
  -> 直接 Reuse 造成任务质量损失
  -> 损失依赖 release edge 和 producer age
  -> 需要研究状态风险结构与预算化 state evolution
~~~

因此后续最值得研究的不是先把旧 prototype 的 action/scheduler 固定下来，而是定位 risk structure：

- 哪类用户的长期兴趣状态更容易失效；
- recent/old history 或兴趣片段是否有不同风险；
- item/embedding drift 和 OOV 是否放大 mismatch；
- 哪些 layer/head/readout 对跨版本状态最敏感；
- append、eviction 和 cutover 后时间长度如何稀释旧状态影响；
- 这些维度是否具有可用于 target-free profiler 的稳定信号。

## 8. 结论边界与反例

可以写：

> 在约 300 天真实交互时间线上，使用约 72% 时间跨度建立初始模型，再以约 4.7% 时间跨度进行更新并观察紧随其后的约 4.7% 时间跨度。多个相邻版本边显示，新模型的 AUC 发布收益会被直接复用父版本 KV 稳定侵蚀；四条常规正收益边的侵蚀比例为 25.5%–47.9%。

不可以写：

- 所有模型更新都必然有害；
- 418.7% 是典型效果；
- One-hop 结果已经证明 recursive cache debt；
- 当前结果已经证明某个最终 migration policy；
- 当前结果已经证明 scheduler、partial action 或 executor 的线上收益；
- HSTU-native Small 结果已经代表 M/L 或 RecFlow。

完整 CSV、PR-AUC 和 log-loss companion 以本文件中的固定表为准；结果源文件仍保存在 results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3/ 下。

## 9. 复现入口

当前结果对应的合同和代码入口：

- configs/contracts/yambda500m_small_hstu_native_rolling_recipe_matrix_v3.yaml
- configs/contracts/yambda500m_small_hstu_native_d14_onehop_reuse_diagnostic_v1.yaml
- configs/contracts/yambda500m_small_hstu_native_d14_onehop_reuse_completion_v2.yaml
- scripts/run_yambda500m_hstu_native_rolling_recipe_matrix_v3.py
- scripts/run_yambda500m_hstu_native_d14_onehop_reuse.py
- scripts/run_yambda500m_hstu_native_d14_onehop_reuse_completion_v2.py
- scripts/run_yambda500m_hstu_native_d14_direct_long_age_reuse.py
- scripts/summarize_yambda500m_hstu_native_d14_auc_coverage.py

这些入口只负责复现固定合同和结果，不授权根据结果新增 edge、调 recipe 或改变最终系统设计。

