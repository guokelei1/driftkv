# 3.3 描述性验证：候选校正强度与逐 head 偏差方向

2026-10-24。A、B 两项均已完整执行并通过完成检查，无后继 GPU 队列。只用物理 GPU1；未训练新模块、未改变 foundation、未修改或编译论文。实验输出：[运行目录](../ch3_motivation/runs/candidate_strength_direction/attempt_1/)。A 是有明显标签/上下文混杂的偏好分散观察；B 支持偏差的绝对低维，但不支持偏差普遍比 native KV 更集中。

## 问题与固定对象

本轮回答两个独立问题：A，同一份校正的偏好强度是否随候选不同；B，每层、每个 head 内的 rebuilt−inherited 偏差是否集中在少量方向。它们分别提供候选参与校正、紧凑行表示的设计线索，不预设二者都成立，不证明具体 gate 或低秩模块的质量。

- 固定 Yambda-Large（10 层，10 head，每 head 32 维）的两条相邻完整 release 边：V1→V2（edge2，day245）及 V2→V3（edge3，day259）。不按新结果挑边、层或 head。
- 使用与 3.2 同源的 canonical 模型、历史、请求、release 日期和 eligible 75,807 人池；保留发布前严格时间截断、末尾 1,024 行、同模型原生前缀计算。
- 冻结人口 8,929 人。原固定 7,000 人在仅看事件支持量时，edge3 合格候选仅覆盖 494 名 UID，低于用户规定的 512 人。因此在计算任何新 λ 损失或谱之前，与已存在的 sensitivity20261024 人口取并集，新增 1,929 人。包含原 3.2 main2k；与 fit、validation、development、confirmation 角色均零交集。
- [plan.json](../ch3_motivation/runs/candidate_strength_direction/attempt_1/plan.json) SHA256：`f9ea39c4129143ecc99b93b3234f18e655021160bbd02836bb7d8909231ee5e8`；人口 SHA256：`b5fbca0709565be130e847f0e78e360b78d4c29dbf50737a8e00e08091767afa`。
- [population_expansion.json](../ch3_motivation/runs/candidate_strength_direction/attempt_1/population_expansion.json)、[support_coverage.md](../ch3_motivation/runs/candidate_strength_direction/attempt_1/support_coverage.md) 保存扩展理由、来源和候选支持。初始 7,000 人检查保留在 `preflight_7000/`，没有覆盖旧结果。

## A：同一校正的候选偏好强度

固定每条边已完成的 `linear256_rows` 映射，不重新拟合。λ 网格预先确定为 0、0.25、0.5、0.75、1、1.25、1.5、1.75、2。

在 release 切换时算一次 `Δ = map(old KV) − old KV`。之后只维护一条共同的 Reuse 轨迹，由 current 模型执行相同的因果追加、FIFO 淘汰及 gap 处理。每次评分只临时将尚存的初始 inherited 行替换成 `old KV + λΔ`；发布后新增行完全相同，读取不写回。这样候选的输入上下文不会随 λ 分叉。λ0 与 Reuse 原生缓存完全一致，λ1 直接使用映射值；inherited 行清空后九档输出逐位相同。

这项 read-only 干预与之前“映射后缓存持续参与行为写入”的方法质量实验不同，不能把这里 λ1 的结果当作旧持久映射方法的成绩，也不复用旧强度实验的分数。

主候选集合在评分前按至少 20 个反馈事件、10 个不同 UID 固定。edge2 有 45 个候选、913 个 UID、1,455 个事件；edge3 有 38 个候选、798 个 UID、1,149 个事件。两边完整反馈人口为 4,032／4,073 名 UID。最低支持不足的候选不会被纳入稳定偏好结论。

对每个候选计算各 λ 下平均二元 logloss，再取网格最小值。损失范围不超过 `1e−6 nat` 记为 flat，不强行指定 λ。非 flat 的近最小并列集合完整保留；展示代表值时取最接近 1 的点，再取较小值。报告所有候选的事件数、UID 数及正负标签数。

稳定性检查为 200 次整 UID 联合 bootstrap（seed17，同一抽样权重用于九档 λ），以及预先固定的 `(UID,timestamp)` 哈希上下文二分。bootstrap 至少 190 次有效、且落在原偏好 ±0.25 内的比例至少 70%，才标记描述性稳定；上下文二分每半至少 10 个事件、5 个 UID，并报告两半 λ 差异。保留失败、flat 及支持不足项。另报告 `inherited_rows>0` 的状态敏感性，主结果仍含全部反馈。

不会报告逐候选事后选 λ 的总体 AUC、恢复率或收益。跨候选的历史、用户、标签分布仍可能不同，即使偏好分散，也不等于排除了这些混杂，更不证明 gate 有效。

额外校准规模检查未找到真正同配方的 16／64／256 拟合系列：现有可比 raw-KV ridge 只有 100／256 人，旧配置中的 16 是 validation 人数。详见 [existing_map_inventory.md](../ch3_motivation/runs/candidate_strength_direction/attempt_1/existing_map_inventory.md)。不伪造三点曲线，不为这一可选检查训练新模块。

## B：逐层、逐 head 的 KV 偏差方向

在相同 cutoff 对全部 8,929 人重新计算 source-native 与 current-native 缓存；不使用 A 的校正缓存，不添加发布后事件。分别对 source K/V、rebuilt K/V、ΔK/ΔV 的全部行建立 100 个独立的 32 列矩阵，仅合并行与用户，绝不混合 head。

同长度前缀批处理（batch8，不补 padding）以减少模型调用。对 native FP32 值先提升为 FP64，再相减并流式累积各 head 的 `XᵀX` 和列和；Gram 的特征值等于 SVD 奇异值平方，无需保留整个人口的行矩阵。主口径不去均值，敏感性口径按每个 head 的所有行减去一个全局均值；不按 UID 各自中心化。

输出每个 head 的完整 r1…32 能量曲线、r4/6/8 能量、80%／90%／95% 所需维数，再汇总逐层和全 head 的中位数、q10/q90。分位带代表 head 间差异，不是置信区间。K/V 分开展示仅为检查各自几何，不是拆开评测两类修正方法。

完整计算之前，用实际满长及自然短长的两个 batch8 验证批处理与逐用户结果，并以显式 FP64 矩阵直接 SVD 核对 Gram 算法。数值检查、失败和修订证据均保留在 `B/`。不计算低秩方案的质量恢复率，不将谱能量占比写成准确率或任务收益。

## 执行与结果

### A 的完整结果

两条边均完成全部 8,929 名居民：分别 40,217／39,504 个真实反馈事件、4,032／4,073 名反馈 UID、361,953／355,536 个九档方法分数。每边 140 个用户块提交；纯评分阶段约 9.2／9.1 分钟，共享 GPU 条件下测得，不作为受控路径性能结论。

| 主候选集合 | V1→V2 | V2→V3 |
|---|---:|---:|
| 合格候选数／覆盖 UID | 45／913 | 38／798 |
| 近零 λ（0、0.25） | 19（42.2%） | 13（34.2%） |
| 中间 λ（0.5） | 1（2.2%） | 0 |
| 近一 λ（0.75、1、1.25） | 5（11.1%） | 0 |
| 强校正 λ（1.5、1.75、2） | 20（44.4%） | 25（65.8%） |
| flat／跨组并列不明 | 0／0 | 0／0 |
| 满足 bootstrap 稳定规则／全部候选 | 29／45 | 20／38 |
| 上下文二分 λ 相差≤0.25／可比较候选 | 14／36 | 12／29 |

严格 λ>1 的比例另为 23/45（51.1%）及 25/38（65.8%）；它与上表近一组中的 λ1.25 有重叠，不能与互斥分类直接相加。

仅看尚有 inherited 行的事件，仍有 42／37 个合格候选、865／773 名 UID；近零 18／13 个、强校正 19／24 个，bootstrap 稳定 27/42／19/37，上下文拆分一致 14/33／12/26。该状态敏感性保留主图中的分散现象。

**可支持的观察：** 在这两条固定边、现有映射和实际反馈条件下，不同候选集合的观测损失偏好并不集中于同一强度：弱校正和强校正都出现。它提供研究请求条件如何影响校正程度的探索线索；尚未分离候选身份的独立作用。

**尚不能支持的推断：** 拆分后偏好一致的只有约四成可比较候选，不能把所有网格最小点当作稳定候选属性。主集合正例为 1,324/1,455 和 1,021/1,149，用户、历史、时间及标签比例仍可能混杂；多数候选样本仅几十条，λ0/λ2 边界偏好也不等于找到了网格外的最优点。不能据此宣称 gate 必要、有效或能提升多少 AUC。

独立 [标签与上下文复核](../ch3_motivation/runs/candidate_strength_direction/attempt_1/A/audit_label_context.md) 进一步发现：有正例的 45/45、38/38 个候选，其正例平均 loss 在 λ2 相比 λ0 都变差；有负例的 42/43、33/35 个候选，其负例平均 loss 则改善。近零偏好组的正例占比为 95.4%／93.9%，强校正组为 86.0%／86.5%。因此直方图两端分化很可能包含标签比例和共同 logit 偏移的影响，不能直接归因于候选特异需求。

合格候选中，在同一 UID×timestamp 内同时出现至少两个候选的用户仅 47／31 人，其中含混合标签的仅 5／1 人，不足 512 人作同上下文的候选特异性证据。bootstrap 与上下文拆分都满足的候选为 14／11 个，覆盖 431／331 名 UID；它们只是描述性子集，不能事后替换主人口。固定 λ1−λ0 的逐候选平均 loss 差在两边分别为 [−0.01649,+0.00683]、[−0.02195,+0.00434] nat；完整效应量分布见 [effects_and_stability.md](../ch3_motivation/runs/candidate_strength_direction/attempt_1/A/figures/effects_and_stability.md)。这里比较固定两档，没有按候选挑最优后的整体收益。

![A：同一校正的候选观测偏好及稳定性](../ch3_motivation/runs/candidate_strength_direction/attempt_1/A/figures/candidate_strength_groups.png)

逐候选事件数、正负标签数、九档损失、近最小并列集合、bootstrap 和上下文拆分全部保留在 [edge2/item_analysis](../ch3_motivation/runs/candidate_strength_direction/attempt_1/A/full/edge_2/item_analysis/) 与 [edge3/item_analysis](../ch3_motivation/runs/candidate_strength_direction/attempt_1/A/full/edge_3/item_analysis/)；不存在逐候选选 λ 后的整体质量成绩。

### B 的执行说明

原绝对缓存 allclose 数值门在实际 batch8 对照中失败，证据保留在 `B/diagnostic16/`。进一步固定同一满长/短长输入检查发现，FP32 批处理和单 UID 缓存最大元素差约 `1.23e−4／1.10e−4`，但六类矩阵逐 head 相对 RMS 最大仅 `3.01e−6／5.55e−6`，所有去均值/不去均值能量曲线最大差 `1.17e−7／1.89e−7`，80%／90%／95% 的目标维数全部不变。强制 Torch、同公式 FP64 RMSNorm 的独立对照最大差 `1.07e−14／1.24e−14`。

因此本项谱分析的门改为逐 head 相对 RMS <1e−5、谱能量曲线差 <1e−5、目标维数不变；原绝对失败继续披露，而非把 cache_atol 盲目放大。生产模型的 FP32 数学未改，Gram 对显式 FP64 SVD 的原严格门未改。新 `diagnostic16_v2` 两边均通过，并绑定数值证据和新版源 SHA `d7f347fff99dc612b0cc2e007a7d8336f52cb47b92231df5271f6c090ee3dcd2` 后执行完整人口。

B 首次 full 的 edge2 在模型加载时遭遇同 GPU 的其他用户进程占用约 70.85 GiB 而 OOM，未产生谱结果；原日志、exit 及 launch 记录保留在 `B/full/edge_2/failures/resource_oom_1/`。该进程退出、资源释放后，以原源代码/输入/人口/配置重跑 edge2，未改模型或筛掉任何人。edge3 的计算正常完成。

### B 的完整结果

每边全部 8,929 人、100 个 layer/head 都保留。edge2／3 分别合并 7,577,143／7,718,811 行，每个 head 使用同一批真实历史行；各有 1,200 条逐 head 记录（6 类矩阵×100 head×2 种中心化）。实际批计算为 1,723／1,709 批；无补零 padding。完成计算约 44.0／66.5 秒，不含首次模型加载及失败加载，且为共享 GPU 条件，不能解释为受控端到端性能。

下表为全部 100 head 的中位数，能量占比为百分比；维数指达到 80%／90%／95% 能量所需 r 的中位数。

| 边 | 口径 | 偏差 | r4 | r6 | r8 | 维数80/90/95 |
|---|---|---|---:|---:|---:|---|
| V1→V2 | 不去均值 | ΔK | 97.28 | 98.43 | 99.03 | 1/2/3 |
| V1→V2 | 不去均值 | ΔV | 97.02 | 98.38 | 99.05 | 1/2/3 |
| V2→V3 | 不去均值 | ΔK | 98.10 | 98.81 | 99.22 | 1/2/2 |
| V2→V3 | 不去均值 | ΔV | 97.68 | 98.64 | 99.09 | 1/2/3 |
| V1→V2 | 去全局均值 | ΔK | 97.45 | 98.52 | 99.03 | 1/2/3 |
| V1→V2 | 去全局均值 | ΔV | 97.05 | 98.39 | 99.04 | 1/2/3 |
| V2→V3 | 去全局均值 | ΔK | 97.88 | 98.66 | 99.13 | 1/2/2 |
| V2→V3 | 去全局均值 | ΔV | 97.73 | 98.61 | 99.04 | 1/2/3 |

不去均值 Δ 的 r6 跨 head q10—q90 为 edge2 K [96.13,99.74]%、V [96.00,99.65]%；edge3 K [95.81,99.76]%、V [95.01,99.74]%。去均值后仍集中，说明该绝对低维现象不只是一个全局共同偏移。

**关键反例：原生 KV 本身也很低维。** 不去均值下 source/rebuilt 的 r6 中位数约 98.51%—98.78%；不能只报 Δ 的约 98% 而暗示它独有这种结构。逐 head 与两个 native 参照配对比较如下，分母均 100；其余为介于两参照之间的 head。

| 边 | 口径 | K：Δ 的 r6 高于两参照／低于两参照 | V：高于／低于 |
|---|---|---|---|
| V1→V2 | 不去均值 | 8／71 | 8／75 |
| V2→V3 | 不去均值 | 41／50 | 34／59 |
| V1→V2 | 去全局均值 | 26／43 | 21／48 |
| V2→V3 | 去全局均值 | 48／47 | 48／49 |

因此，“偏差在各 head 内可由少量方向描述”的绝对几何观察得到支持；“偏差普遍比原生 KV 更集中”的更强说法不成立，尤其第一条边多数 head 方向相反。按用户原设计中的比较标准，B 不能单独升级为修正状态相对原生 KV 具有普遍独特压缩优势的证据。它可以启发紧凑表示，但既不证明具体每行坐标数，也不证明低秩近似保存了 AUC。

![B：全部 head 的 r6 能量，未选择层或 head](../ch3_motivation/runs/candidate_strength_direction/attempt_1/B/figures/all_heads_r6_uncentered.png)

完整逐 head、逐层曲线和两个 native 参照均在 [B/figures](../ch3_motivation/runs/candidate_strength_direction/attempt_1/B/figures/)；[edge2](../ch3_motivation/runs/candidate_strength_direction/attempt_1/B/full/edge_2/per_head.csv)、[edge3](../ch3_motivation/runs/candidate_strength_direction/attempt_1/B/full/edge_3/per_head.csv) CSV 含所有 r1…32、r80/90/95；[analysis_summary.md](../ch3_motivation/runs/candidate_strength_direction/attempt_1/B/analysis_summary.md) 含 r4/6/8 的完整配对正反计数与维数分布。

[同 head 配对差值图](../ch3_motivation/runs/candidate_strength_direction/attempt_1/B/figures/paired_r6_differences_uncentered.png) 展示 Δ 相对 source/current 的 r6 能量差，正负点全部保留，避免绝对能量均接近 98% 时掩盖参照差异。去均值版本亦保留；两个逐层 PDF 各 40 页，覆盖全部边×K/V×层×head。

### 完成检查与论文边界

[completion_audit.json](../ch3_motivation/runs/candidate_strength_direction/attempt_1/completion_audit.json) 已通过：A 的 280 个原始分数块/manifest 哈希、9 档事件键/标签、完整人口、原映射权重/数值源，以及 B 全部 100 head、两种中心化、32 档累计能量的完整性及单调性均检查。A 在 inherited 行清空后的 2,056／2,324 个事件九档分数逐位相同；旧 7,000 人的 Reuse 分数与新 λ0 在原 `atol1e−6+rtol1e−5` 内一致，最大绝对差 `9.54e−7／1.19e−6`，没有重定义评分门。

论文 `main.tex` SHA256 仍为 `bf98cd0c85374e9bc9fbc01fc321e88374e20fd83131f88b33f21f0bed04d2e1`，未修改、未编译。本轮两项均完整执行，不能把“已完成验证”写成“预设结论均已证明”：A 仍有明显标签/上下文混杂；B 支持绝对低维，不支持偏差普遍比原生 KV 更低维。后续如何组织 motivation 应以这些边界为准。

最终派生图表、源绑定与本页本地链接的检查另存 `derived_artifact_audit.json`；本轮没有待执行的 GPU 后继。
