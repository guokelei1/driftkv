# 3.2 读取修正探针

## 当前状态与结果

实验已完成。当前 Motivation 选用 **Q-v5 非线性 query 修正**
（`cross_phi_joint8`，C128/256/512）与 **H-v4 非线性全历史映射**
（`map_all`，C64/128/256），每种方法覆盖全部 15 条相邻边，各 45 个实测点。

- [当前结果与来源](../../results/read_correction_2026_09/motivation_final/README.md)
- [90 点总图与复现命令](../../figures/out/read_correction_2026_09/motivation_final/README.md)
- [四种局部重算对照](../../results/selective_recompute_2026_09/README.md)
- [结构示意图与生成器](../../figures/README.md)

本页记录实际方法和共同评测口径。两种探针用于观察质量—成本关系；
当前论文 Design 的摘要、发布期翻译、个性化修正和连续状态更新仍需实现与验证，
见[设计入口](../paper_design.md)。已完成的队列不作为重新启动实验的授权。

## 研究问题

1. 共享 query 修正能否以较少新增计算恢复部分推荐质量？
2. correction 显式处理用户历史后能否获得更高恢复，其实际计算代价是多少？

原生推荐路径本来就读取用户历史，上层 q 也可能携带下层历史响应。
这里区分的是 **correction 的额外输入与执行**。两种探针的结构和监督目标不同，
其差值不能全部归因于严格单因素的“加入历史”操作。

## 选定的方法

模型、embedding 和 score head 冻结；每个 scale/edge 单独拟合跨用户共享参数。

| 方法 | 拟合与读取 |
| --- | --- |
| Q-v5 | 每层拼接跨 head 的 `q` 与 `ELU(q)`，用校准组标准化特征、岭回归初始化，再进行最多 8 轮最终输出教师蒸馏。推理得到修正率，乘当前历史长度 N，加到原生历史读响应。 |
| H-v4 | 对每个历史位置拼接全部 head 的 K/V，先拟合仿射映射，再拟合 `2D→D/2→2D` 的 SiLU 残差。每次请求临时映射完整历史，再由该分支实际 q 按原生注意力读取；不增加 Q 补偿层。 |

Q 对原始 q 是非线性的；`ELU(q)` 是选定的特征基，不是原生注意力的精确分解。
Q 的 correction 不额外读取 K/V 特征或原生响应。
逐层初始化使用同一个实际 q 下的 Full 与继承缓存读差；最终蒸馏包含用户内
中心化分数差和用户均值误差，独立 16 名验证用户选择 epoch，包括初值。

H 使用 pure Parent 发布前缓存与同位置 Current Full 作监督；ridge 为 0.01，
每用户最多 128 个历史位置、16 个均匀已知目录 query，非线性拟合 6 轮。
目标包含 token K/V 残差和同实际 q 的读取误差，每层由独立 16 名验证用户选择 epoch。
`map_all` 映射全部保留位置；仅旧 producer 前缀映射是历史诊断变体，不是当前曲线。

两者均只改变当前请求的读取。H 的临时映射不写回持久 K/V，后续真实事件仍按
未修正的 Current 原生路径追加。结构图将 H 写为“原读取＋读取变化”是等价表达；
实际执行直接读取映射后的 K/V，不额外执行一次残差读取来凑成本。

## 用户与时间

- Medium/Large/Max 的 V0–V5，共 15 条相邻边；每边沿用四基线固定的 3,000 名用户
  及全部请求，共 401,187 请求，1024 历史容量。
- 面板按既有 Full–Reuse 排序差异选择，是条件化开发人群。Q/H 也用于开发探索，
  不将其均值解释为总体用户估计或独立测试结果。
- 校准与验证 UID 排除同规模五个评价/数值检查面板的并集；仅使用发布前合法历史，
  不用评价反馈拟合。拟合人数是嵌套预算；其后的 16 人为该预算验证组，
  因此不同预算不共享同一个验证组。
- 每条边发布时由 Parent 构建历史，此后由 Current 真实追加和淘汰；
  同时间先查询再追加。各边独立初始化，不建立连续多发布适配证据。
- 保留原 Full、Reuse 请求和分数。缓存对齐的映射 item 排序与原 Full collator
  的同时事件顺序差异按原数值对照记录，不通过重生成 Full 隐去。

## 质量、成本与展示

`Recovery = (AUC_method − AUC_Reuse) / (AUC_Full − AUC_Reuse)`

`Relative cost = extra_FLOPs_over_Reuse / (Full_FLOPs − Reuse_FLOPs)`

横轴 Reuse=0%、Full=100%。分子包含每个点完整的教师/cache 准备、拟合、验证
以及额外请求期计算；复用已拟合文件不令校准成本变成零。分母沿用四基线真实请求
工作量，并扣除 rolling Reuse 的 Current append。解析 FLOPs 不代表实测延迟。

总体先逐边归一化，再对全部 15 边等权平均；分规模各五边等权。
全部预算、负恢复、超过 100% 的点和非单调曲线保留，不逐边择优。
H 三档的推理部分相同，校准人数改变训练成本；Q 三档同理。

当前总体 Q 的成本为 1.49/2.79/5.43%，恢复为 11.96/33.35/37.32%；
H 的成本为 72.11/74.22/78.44%，恢复为 69.37/70.95/71.69%。
这些结果展示本组简单探针的恢复潜力与质量—成本取舍，不代表 Q 的能力上限、
历史处理的最低成本，也不支持“同成本胜过全部四个重算基线”。

## 代码与保留的探索记录

| 内容 | 入口 |
| --- | --- |
| Q 数值实现 / 拟合 | `src/hstu_kvcache/read_correction_v5/query_only/`、`scripts/read_correction_v5/query_only/` |
| H 数值实现 / 拟合 | `src/hstu_kvcache/read_correction_v4/`、`scripts/read_correction_v4/` |
| H C64/C256 补齐 | `scripts/read_correction_motivation/` |
| 公共数据、reader、成本 | [共享工具说明](../../scripts/read_correction_2026_09/README.md) |
| 数值检查 | `tests/read_correction/`、`tests/read_correction_v4/`、`tests/read_correction_v5/` |

[结果索引](../../results/read_correction_2026_09/README.md)保留 v1–v5 的实际设置、
全部结果和失败记录。旧队列入口已退休，历史封存哈希不改写成当前源码哈希。
[Max 诊断](../../results/read_correction_2026_09/v5/max_diagnostics/README.md)
仅用于解释内部研究现象，不构成某个规模专用的 Motivation 或 Design 规则。
