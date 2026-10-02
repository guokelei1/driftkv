# 历史 128 用户诊断：候选特征与用户读取信息

历史开发记录；以下设置和结果仅指本次实验。旧执行入口及预览图已退役，
原始评分、配置、失败和封存来源继续保留。当前三规模 Motivation 见
[实验索引](../../../docs/motivation_observations.md)。

2026-09-20 完成并保留。随后完成
[256 校准／2560 评价的查询与响应诊断](../query_read_6l_2560_20260920/README.md)，
本组全部原始结果、失败、成本记录及[原表快照](source_snapshot/README.md)保持原样。
当时的论文表展示两种共享探针在两条预选发布边上的对照；
每用户教师参照已经执行，其全部结果继续保留在本记录和 raw 中，不进入论文主表。
读取响应差的存在由同查询下的定义说明，不用近 99% 的教师诊断代替可服务性证据。

本实验支持有限的跨用户预测信号：加入实际用户读取响应后，两条边的平均概率误差均
低于 Reuse，但逐用户平均恢复率只有 1.90%／28.45%，仍有明显负尾。它不验证当前
完整 Design、不检验摘要必要性，也不给出推荐 AUC 或连续状态维护的结论。

## 固定协议与实现

- [预先固定的配置](../../../configs/insight/user_information_6l_20260920.json)：
  当前 consolidated Medium 链 V1→V2、V2→V3，cutover 为 day245／259；
  seed17、6L/H192/6heads/context1024，实际算子为 legacy ELU+1。
  模型由当前链清单解析，核验权重／映射 hash，并要求既有 Full-only admission 解锁 Reuse。
- 256 个校准 UID、128 个不重叠的评价 UID，均取自既有开发划分；评价用户此前参与过
  其他开发，未见仅指没有参加本次共享拟合。没有读取 final／confirmation 人口。
- 每个发布点只使用严格早于 cutover 的 1024 条真实历史。64 个无标签候选由 recent16、
  old-only16 与发布前 popularity bank 补足；校准和评价分别用完整组的历史建立候选库，
  再切 canary／batch。历史中未出现的候选不是观测负例。
- 两种共享规则各用校准组每人 16 个候选，逐层拟合 FP64 线性 ridge=.01（含截距惩罚），
  同时预测六个 heads 的 192 维响应差。标准化和全部共享参数只来自校准组。
  监督是本分支实际 incoming query 下的 `(teacher response - native response)/1024`；
  已拟合的下层修正进入下一层查询，避免混用另一条前向路径的教师差。
- `shared_candidate` 只输入 Current 候选 item embedding。同一个候选的修正跨用户相同，
  不向修正器输入 UID、时间、历史或高层 query。底层原生读取仍使用用户自己的历史。
  `shared_user_response` 再输入该层实际原生历史响应除以 1024；修正生成不接收评价用户教师。
  这个响应包含当前用户的读取上下文，不能把其贡献进一步独立归因为静态画像或某种摘要。
- 所有方法在评价组相同的 32 个奇数索引候选上计量（索引从 0 开始）。已执行的
  `per_user_rank1` 另用该评价用户自己的 32 个偶数索引教师响应拟合 offset+rank1；
  它仅是表达能力参照，不是共享方法或严格上界。

原共享原语 `user_information_probe.py` 与执行入口 `run_shared_read_probe.py`
已退役；执行版本及输入绑定以保存的 `diagnostic/configuration.json` 为准。
发布相关的共享线性映射并非新的数学机制；按头的线性响应变换可以与 Value 映射等价，
跨头混合通常不能直接交换到各头聚合前。本诊断未比较同预算 K/V translation 的优劣。

## 全部结果与指标

对每个 UID，令 `e_m` 为 32 候选上 `|sigmoid(logit_m)-sigmoid(logit_Exact)|` 的均值。
冻结主指标 `U = mean_UID(1-e_m/e_Reuse)`；辅助指标
`P = 1-mean_UID(e_m)/mean_UID(e_Reuse)`。以下 U/P 均乘以 100，不裁剪负值。
两者权重不同，不能用较大的 P 替代较弱的主指标 U，也不把两个发布点当独立训练重复。

| 方法 | V1→V2 U (%) | V1→V2 P (%) | V2→V3 U (%) | V2→V3 P (%) |
| --- | ---: | ---: | ---: | ---: |
| Reuse | 0.00 | 0.00 | 0.00 | 0.00 |
| Exact | 100.00 | 100.00 | 100.00 | 100.00 |
| 每用户教师参照（仅保留记录） | 99.12 | 99.72 | 99.50 | 99.83 |
| Shared, item only | -436.12 | -140.40 | -129.25 | 10.15 |
| Shared + user response | 1.90 | 64.42 | 28.45 | 77.98 |

| 边／方法 | 概率 MAE | UID 恢复率中位数 (%) | UID 恢复率 p10 (%) | 不优于 Reuse 的 UID |
| --- | ---: | ---: | ---: | ---: |
| V1→V2 Reuse | 0.009322044 | 0.00 | 0.00 | 128/128 |
| V1→V2 每用户教师 | 0.000026199 | 99.68 | 98.16 | 0/128 |
| V1→V2 item only | 0.022410224 | -57.89 | -769.51 | 91/128 |
| V1→V2 + user response | 0.003317219 | 74.04 | -209.04 | 26/128 |
| V2→V3 Reuse | 0.007714179 | 0.00 | 0.00 | 128/128 |
| V2→V3 每用户教师 | 0.000012825 | 99.82 | 99.01 | 0/128 |
| V2→V3 item only | 0.006931437 | 16.08 | -389.43 | 55/128 |
| V2→V3 + user response | 0.001698305 | 79.57 | -72.53 | 21/128 |

Exact 的概率 MAE 为 0。最小逐用户 Reuse MAE 分别为 `7.95687e-5`、`1.78137e-4`，
没有触发近零分母停止条件；小分母仍会放大负尾。加入用户响应的最差恢复率分别为
-1657.11%／-1649.44%，全部保留。观察只涉及这批开发用户的概率保真度，不按版本解释
收益差异，不把当前线性输入对照提升为所有共享修正都必须使用某个特征的必要性证明。

## 成本口径

[成本分解](analysis/cost.md)和[机器可读记录](analysis/cost.json)按保留执行器的张量形状
估算标量运算，每个乘加计 2，特殊函数计 1。它不是 GPU profiler 或人口性能测量。
六层、1024 历史、每用户一次 32 候选读取下：

| 共享探针 | 每发布校准 K | 每用户额外读取 | 额外读取／Exact cache-only 重建 |
| --- | ---: | ---: | ---: |
| item only | 3.9677 T | 14.340 M | 0.342% |
| + user response | 3.9824 T | 29.270 M | 0.697% |

分母是依赖闭合的 cache-only 重建 `4.1971 G/user`，不是更大的 eager 完整 prefill。
普通原生 32 候选读取约 `228.530 M/user`，两条服务路径都需要，单列而不计入额外费用。
校准 K 计入 Parent／Current 两份前缀、逐层响应获取、已拟合低层修正和 ridge 拟合；
评价用户的 Exact 与逐用户 oracle 仅为研究测量，不是共享预测需要的服务输入。

只有假定 30,000 用户均保留 1024 事件、每人只读取 32 候选一次时，
`(K+U*C_read)/(U*C_rebuild)` 对响应分支为 **3.860%**（item-only 为 3.493%）。
这是条件算术外推，不是 30,000 用户质量结果、人口 FLOPs 实测或端到端加速。
重复读取会继续累计修正成本；不计数据 I/O、内存流量、传输、调度和硬件精度权重。

## 检查、资源与证据

[32/8 canary](canary/canary.pass.json)通过 CPU 正规方程与实际读取 callback 对照，
同 item 跨用户修正一致性及候选分支对 native 输入的独立性检查。完整运行两条边的
native reader／Exact replacement 与模型原生端点最大差不超过 `2.38419e-7`。
CPU ridge 使用单线程，避免当前 PyTorch 2.12.1／MKL 多线程 LU 的已复现问题；
模型为 FP32、求解为 FP64，TF32 关闭，`EVOKV_ATTENTION_BACKEND=torch`。

[资源估计](resource_estimate.json)在完整运行前固定为单 A40、5 分钟、约 10 GiB。
实际记录 history 载入 43.33 秒，两边分别 13.08／12.35 秒，峰值 9.87 GiB；
这是小诊断的运行开销，不作服务性能比较。没有 backbone 训练或长作业。

- [configuration.json](diagnostic/configuration.json)：完整设置与执行依赖 hash；
  [summary.json](diagnostic/summary.json)：所有方法、端点检查、计时和模型来源。
- [metrics.csv](diagnostic/metrics.csv)、[per_user.csv](diagnostic/per_user.csv)：所有汇总及未裁剪 UID 结果。
- `diagnostic/v1_to_v2.npz`、`diagnostic/v2_to_v3.npz`：UID、候选、delta、全部原始 logits，
  包括 Reuse／Exact／每用户教师及两个共享分支；本地 raw 按存储规则保留。
- `canary/` 保留全部早期结果与 `runtime.log`；`diagnostic/runtime.log` 保留完整执行日志。
- [历史表生成器与原表](source_snapshot/README.md)只读并独立重算 raw 中的 U/P。
  当时的表展示两个共享分支、两条边，不显示已执行的每用户教师行。

## 保留证据与旧实验边界

旧诊断和成本分析启动器已退役。保留的[原表生成器](source_snapshot/shared_read_insight.py)
可只读本组 raw 重算原表；它生成的是本组历史表，不是当前 Q-v5/H-v4 图。

前一[摘要诊断](../shared_read_6l_20260920/diagnostic/summary.json)及其
[旧表／源码快照](../shared_read_6l_20260920/source_snapshot/README.md)完整保留。
按当时对研究问题的澄清，该组 Insight 不展示摘要，不把该分支当择优候选，也不抹掉其混合结果。
历史逐用户 95.34%／99.46% 和动态 93.39% 等结果见
[历史 Insight 2](../../../scripts/insight_two/README.md)；它们不进入本次正文论证，
原 raw、analysis_v2、失败结果与 invalidation 均维持原范围。
