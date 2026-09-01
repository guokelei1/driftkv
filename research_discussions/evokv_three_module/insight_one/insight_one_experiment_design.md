# EvoKV Insight 1：Medium KV 局部重算诊断实验设计

日期：2026-09-01  
状态：待讨论、待合同冻结，尚未授权运行

## 1. 实验要回答的问题

这一实验只回答一个问题：在一次 Parent→Current 模型更新中，如果我们已经知道 Parent K/V 和 Current Exact K/V，那么只把少量层、少量离散 token 或一段连续历史替换成 Current Exact 状态，能否恢复大部分 Reuse–Current Exact 功能差距。它不是三种完整迁移系统的最终性能比较，也不试图证明推荐 K/V 中不存在任何局部结构；它要测量的是三类常见局部重算思路在一个对它们较为有利的理想化条件下，能够形成怎样的理论重算量—恢复率曲线。

实验采用 Exact-KV splice：先分别生成同一用户历史在 Parent 和 Current 模型下的完整 K/V，再将选中区域的 Parent K/V 替换为 Current Exact K/V。这样可以排除 selector 实现不够好、局部状态算得不够准等因素，直接观察所选坐标本身最多能带来多大恢复。需要强调的是，这种 splice 是结构诊断干预，不是 dependency-closed 的可执行重算动作；横轴也只是被替换的 K/V 数量比例，不是真实 FLOPs、GPU 时间或迁移 I/O。如果这种理想化实验仍然需要覆盖大部分状态才能获得较高恢复，那么真实执行只会更贵。

## 2. 固定实验对象

实验使用已经完成并封存的 Medium D14 模型链：Yambda-500M Medium 固定人口为 30,000 users，模型为 HSTU-native CC，6 layers、hidden size 192、6 heads、context 1024、seed 17。五条相邻 release edges 为 $v_0\!\rightarrow v_1$、$v_1\!\rightarrow v_2$、$v_2\!\rightarrow v_3$、$v_3\!\rightarrow v_4$ 和 $v_4\!\rightarrow v_5$，cutover day 分别为 231、245、259、273 和 287。五条 edge 使用完全相同的配置矩阵，任何一条都不能因为结果不利而删除。

主实验固定 3,000 名用户，并在五条 edge 上使用同一组 UID。用户只按已有的 label-free `selector_rank` 和 UID 选择，要求在最早的 day-231 cutover 前已经拥有至少 1024 次历史交互。当前只读人口审计显示，30,000 名 Medium 用户中有 21,200 名满足这一条件，因此可以稳定取得 3,000 名全长历史用户。每条 edge 使用该用户在 cutover 严格之前的最后 1024 个事件，位置 0 表示最旧事件，位置 1023 表示最新事件；核心实验不包含 cutover 后 append，从而只测量发布瞬间的状态版本失配。

每个用户在每条 edge 上固定 64 个无标签 candidate probes，沿用 Small 结构实验的候选语义：16 个来自较新的历史区域，16 个来自较旧的历史区域，其余 32 个由仅使用 cutover 前历史构造的全局已知 item bank 确定；不足或重复位置按照事先冻结的 item 顺序确定性补齐。候选只作为 Current reader 的功能探针，不被解释为曝光负例，也不读取 request label。候选集合在同一 edge 的所有配置之间完全相同，query timestamp 固定为该 edge 的 cutover。

## 3. 统一干预、成本与恢复率

对 edge $e$ 和用户 $u$，记同一 1024-event prefix 在 Parent 和 Current 下生成的状态分别为 $C^P_{e,u}$ 和 $C^C_{e,u}$。对一个二维 layer–position mask $M$，hybrid state 定义为

\[
K^M=M\odot K^C+(1-M)\odot K^P,\qquad
V^M=M\odot V^C+(1-M)\odot V^P.
\]

每次选择都同时替换 K 和 V，并覆盖相应位置的全部 attention heads。Layer 实验选择某些层中的全部 1024 个位置；token 和 window 实验选择某些历史位置，并在全部 6 层中替换这些位置。除 mask 外，Parent/Current checkpoint、用户历史、candidate probes、Current reader、query time、推理精度和 batching 语义均保持不变。

理论 KV 重算比例按被选中的 token–layer 单元计量：

\[
\mathrm{Cost}(M)=\frac{|M|}{6\times1024}.
\]

因为 K/V、head 数和每个 head 的宽度在所有单元中相同，这一定义与按 K/V 标量数量计算得到相同比例。它没有计入 selector 发现开销，也没有计入为了真实生成上层 K/V 所需的依赖传播，因此只能称为 theoretical KV coverage，不能称为真实计算成本。

主功能距离固定为 Current 模型输出概率相对 Current Exact 的平均绝对差。记 $p^M_{u,c}$、$p^P_{u,c}$ 和 $p^C_{u,c}$ 分别为 Current reader 在 hybrid、Reuse 和 Current Exact 状态上的预测概率，则

\[
G_e(M)=\frac{1}{|U||Q|}\sum_{u\in U,c\in Q_u}|p^M_{u,c}-p^C_{u,c}|,
\qquad
R_e(M)=1-\frac{G_e(M)}{G_e(P)}.
\]

Reuse 的 recovery 为 0，Current Exact 为 1。Recovery 不做裁剪，因此一个干预若使结果更差可以小于 0，若偶然越过 Exact 也可以大于 1。聚合采用 ratio of means：先在一条 edge 的全部用户和 candidates 上计算平均 gap，再做比值，避免被大量接近零的单用户分母放大。Bernoulli JS、平均绝对 logit gap、top-1 agreement 和 rank correlation 只作为不改变结论的稳健性指标；AUC 与 log-loss 不参与这一 Insight 的 selector 选择和主结论。

## 4. Layer family

层编号在论文中使用 1–6，代码中对应 0–5。所有候选都完整执行并保留原始结果。除用户草稿中的组合外，三层预算增加 `[4,5,6]`，因为原草稿的 `[1,2,3]` 与 `[3,4,5]` 没有真正覆盖尾部三层；加入这一组才能与“头部—中部—尾部”的设计动机一致。

| 更新层数 | 冻结层组合 | 配置数 | 理论 KV coverage |
| ---: | --- | ---: | ---: |
| 1 | `[1]`, `[2]`, `[3]`, `[4]`, `[5]`, `[6]` | 6 | 16.67% |
| 2 | `[1,2]`, `[3,4]`, `[5,6]` | 3 | 33.33% |
| 3 | `[1,2,3]`, `[3,4,5]`, `[4,5,6]` | 3 | 50.00% |
| 4 | `[1,2,3,4]`, `[3,4,5,6]` | 2 | 66.67% |

因此 Layer family 每条 edge 共 14 个配置。单层的六个结果以及其他预算下的所有组合都要分别落盘。论文主图在每个预算上使用 recovery 最高的 best-observed 配置，以给层局部性一个乐观比较；同时在支持表中报告平均值、最小值、最大值和 winner layer ID。这样既保留“逐层全测后看平均稳定性”的信息，又不让主结论建立在某个恰好较差的层选择上。五层不测，六层由 Exact-All 的 100% anchor 表示；如果 66.67% 的最高测量点仍未达到 80% recovery，只能报告“达到 80% 需要超过 66.67% coverage”，不能插值或虚构一个 L80。

## 5. Window family

窗口实验在完整 1024-event prefix 上进行。每个窗口都跨全部 6 层同时替换 K/V。128 和 256 宽度分别检查最新窗口及向前移动的两个相邻窗口；512 和 768 只检查最新窗口，避免大预算下继续膨胀配置数量。

| 窗口宽度 | 冻结区间，采用半开区间 `[start, stop)` | 配置数 | 理论 KV coverage |
| ---: | --- | ---: | ---: |
| 128 | `[896,1024)`, `[768,896)`, `[640,768)` | 3 | 12.50% |
| 256 | `[768,1024)`, `[512,768)`, `[256,512)` | 3 | 25.00% |
| 512 | `[512,1024)` | 1 | 50.00% |
| 768 | `[256,1024)` | 1 | 75.00% |

Window family 每条 edge 共 8 个配置。需要注意，128/256/512/768 在 1024 context 下的准确比例是 12.5%/25%/50%/75%，不能在图中标成 10%/20%/40%/80%。每个宽度在主图中同样报告冻结窗口中的 best-observed recovery，并在支持表中列出每个实际窗口的结果。512 和 768 只有 tail 配置，因此相应预算不存在窗口位置搜索优势。

## 6. Token family

Token family 的 mask 是 per-user、candidate-shared 的历史位置集合：同一用户的 64 个 candidates 共用一个 token mask，mask 再广播到全部 6 层和全部 heads。这样测量的是“少量离散交互位置是否能够承担大部分跨版本恢复”，而不是为每个候选单独选择一套状态。四个预算固定为 10%、20%、40% 和 80%，在 1024 个位置上分别选择 102、205、410 和 819 个 token；图中使用实际覆盖率 9.96%、20.02%、40.04% 和 79.98%。

候选 selector 保持简单，但同时包含传统 attention importance、HSTU 实际 read contribution 和对局部方法更有利的 Current-aware 诊断信号。所有分数都在读取 recovery 结果前冻结，按分数降序选取固定数量的位置；同分时先选择较新的位置，再按位置编号确定。不同 layer/head/candidate 的分数在聚合前只做各自历史维度上的 L1 归一化，以免 HSTU 非 softmax attention 的尺度差异让某一层或某一 head 无意中支配排序；这一归一化只用于生成 mask，不改变 reader 的真实计算。

| Selector | 冻结定义 | 是否读取 Current Exact |
| --- | --- | --- |
| `ATTN_MASS` | Current query 读取 Parent K 时，各 candidate、layer、head 对每个历史位置产生的 pointwise attention weight 之和 | 否 |
| `READ_NORM` | 上述 Parent read 中每个位置的加权 value contribution $\lVert a_tV^P_t\rVert_2$ 之和 | 否 |
| `PERSISTENCE` | 一个位置进入各 candidate–layer–head attention top-10% 的频率；频率相同时以 `ATTN_MASS` 排序 | 否 |
| `KV_DRIFT` | 每个位置跨层聚合的 $\lVert K^C-K^P\rVert_2+\lVert V^C-V^P\rVert_2$ | 是，诊断性 |
| `READ_DELTA` | 在 Reuse 与 Current Exact reader 轨迹上，每个位置的加权 value contribution 差异范数，跨 candidate、layer、head 聚合 | 是，诊断性 oracle |

各预算的冻结候选数量如下：

| Token budget | 位置数 | 使用的 selector | 配置数 |
| ---: | ---: | --- | ---: |
| 10% | 102 | `ATTN_MASS`, `READ_NORM`, `PERSISTENCE`, `KV_DRIFT`, `READ_DELTA` | 5 |
| 20% | 205 | `ATTN_MASS`, `READ_NORM`, `KV_DRIFT`, `READ_DELTA` | 4 |
| 40% | 410 | `READ_NORM`, `READ_DELTA` | 2 |
| 80% | 819 | `READ_DELTA` | 1 |

Token family 因此每条 edge 共 12 个配置。低预算配置更多，是有意把有限的 selector 搜索能力放在最可能出现强稀疏局部性的区域，并使局部方法在论文所关心的低成本区域获得更有利的机会；配置集合一旦冻结，不得因为观察到某条 edge 的结果后再加入新 selector。`KV_DRIFT` 和 `READ_DELTA` 使用 Current Exact 信息，只能作为乐观的结构诊断，不能被描述为可部署的 token selection policy。若这两种 Current-aware selector 仍不能在低覆盖率下取得高恢复，负面观察会更有说服力；若它们形成明显跃升，则应保留 token-local 路线，继续研究如何以可执行信号逼近它们。

## 7. 实验总量

每条 release edge 包含 14 个 layer、8 个 window 和 12 个 token 配置，共 34 个局部干预。再加上 Reuse 与 Current Exact 两个 anchor，每条 edge 需要评估 36 条状态路径。五条 edge 合计为 170 个局部配置—edge 单元，或 180 条包含 anchor 的 edge-path；这些不是 180 次模型训练，也不要求重复生成 180 份完整 cache。

Parent K/V 和 Current Exact K/V 对每个 user-edge 只生成一次，五种 token selector 的位置分数也各自只计算一次，同一 selector 在不同预算下通过 top-k 产生 mask。Hybrid cache 应当在 batch 内按 mask 临时构造并立即评分，不持久保存 34 份完整 K/V。按照 3,000 users、64 candidates、5 edges 和每条 edge 36 条路径计算，正式阶段共有 34,560,000 个 candidate-path score；实际 GPU 时间、显存峰值和 batch 大小由 canary 测量，不能从这一数量直接推断。

## 8. 执行顺序与变量控制

正式运行前先建立新的 prospective Medium Insight 合同，冻结六个 checkpoint 的路径与哈希、五条 cutover、3,000 UID 列表及其哈希、每条 edge 的 candidate bank、全部 34 个配置 ID、token selector 公式、metric、aggregation、输出目录和失败处理。现有 Medium Full/Reuse 合同中的 `medium_pro_or_insight_probe` 明确为 prohibited，因此本文档本身不构成执行授权。

实现完成后只先运行 32-user、全部五条 edge 的 focused canary。Canary 只检查正确性和资源，不用于修改候选集合：空 mask 必须在数值容差内复现 Reuse，全 mask 必须复现 Current Exact；选中位置的 K/V 必须等于 Current Exact，未选位置必须等于 Parent；K 与 V 必须成对替换；token mask 的 cardinality、candidate-shared 性质和确定性排序必须逐项验证；所有模型必须处于 eval/inference mode，且任何 label 都不得进入 raw generation。Canary 通过后，根据测得的显存和 wall-clock 给出正式资源估计，再由用户明确授权正式运行。

正式执行按 edge 串行。每条 edge 先加载 Parent 与 Current checkpoint，读取相同的 3,000 条 cutover prefix，按 batch 生成 Parent/Current cache 和两个 anchor score；随后一次性生成 selector signals 与 masks，依次构造 34 个 hybrid state 并由同一个 Current reader 对同一 64-candidate panel 评分。Raw score、mask 元数据、selector winner 和运行日志在任何可选 label join 之前封存并哈希；本 Insight 的主分析直接在无标签 raw 上完成。所有 edge 都完成并通过完整性检查后才能生成汇总图，不能边运行边根据前几条 edge 调整配置。

## 9. 主图与结论口径

主图采用 2×3 small multiples：前五个 panel 分别对应五条 adjacent edge，第六个 panel 为 edge-equal aggregate。每个 panel 只画 layer、token 和 window 三条 best-observed 曲线，并加入 `(0,0)` 的 Reuse anchor 与 `(1,1)` 的 Current Exact anchor。每个预算点必须记录实际 winner config；支持材料同时给出所有 34 个原始配置以及同预算下的 mean/min/max，避免 best-observed 曲线掩盖选择稳定性。

三类 family 的预算网格并不完全对齐，因此正文应按真实横坐标报告曲线和各自的低预算点，不应把 12.5%、16.67% 和 20% 写成同一个“共同预算”，也不应依靠插值制造对齐结果。80% recovery 的 crossing 只从已测点判断：如果某个 family 在最高非 Exact 点仍未达到 80%，就报告其测量下界，而不是从该点连向 Exact anchor 后估算阈值。Edge aggregate 先在每条 edge、每个 family 和预算上得到 best-observed recovery，再对五条 edge 等权平均；同时保留一个“全 edge 共用同一 selector/config”的辅助结果，用来区分局部性是否存在与 selector 是否跨版本稳定。

最终结论由曲线决定，而不是事先规定某个 10% 或 20% 点必须通过固定阈值。如果某一 family 在较小 coverage 下出现稳定、明显的恢复跃升，Insight 1 就不能写成局部重算失效，而应继续研究该局部结构的可识别性和 dependency-closed 实现。只有当三类 best-observed frontier 在五条 edge 上都主要随覆盖范围渐进上升、较高恢复普遍需要较大范围的 K/V replacement 时，正文才可以得出当前限定结论：现有 LLM-inspired 的层、离散 token 和连续窗口局部重算，在 persistent Transformer recommendation state 上没有形成有利的理论重算量—恢复率权衡。

## 10. 本设计冻结前仍需完成的事项

这份设计已经确定实验对象、三类配置网格、主 metric、主图和解释边界。真正运行前仍需把它转写成机器可读合同，并完成三项只涉及实现与资源的工作：冻结 Medium 3,000-user population 和五个 64-candidate panels；实现并测试通用 layer–position exact splice 与五个 token selector；用 32-user canary 测出安全 batch、显存、临时状态量和预计 makespan。上述内容不得使用 recovery 或未来 label 调参，正式配置在看到结果后不得扩展。
