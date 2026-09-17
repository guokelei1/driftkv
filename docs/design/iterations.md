# 六层适配迭代记录

当前 Design 已完成设计，尚未实现和验证。本文件保留历史探索的发现和限制，供后续实现参考，不代表当前 Design 的验证结论。本轮 Medium／Large 模型链已按[统一训练记录](../unified_training_2026_09/README.md)整理，适配实验仍待重做；旧结果不替代新验证。

## 已知结论

- HSTU-native 读出在冻结六层开发模型上有质量信号；证据见 [native 基础质量报告](../../results/design/analysis/native_base_quality4091_01_report/report.md)。
- 查询留出、native 输入和解码闭环控制仍是判断机制是否有效的必要诊断，分别见 [查询留出](../../results/design/analysis/mechanism_query_holdout192_report_01/report.md)、[native 输入](../../results/design/analysis/mechanism_native_input192_report_01/report.md) 和 [解码闭环](../../results/design/analysis/mechanism_decoder_closure192_report_01/report.md)。
- 开发点的计算组成已单独测量，见 [FLOPs 报告](../../results/design/analysis/native_flops_01/report.md)。高成本且恢复弱的变体没有进入后续范围。
- 兼容性与模型准入分开处理：候选版本不满足 H/S 时，服务父版本和缓存谱系保持不变。
- 只有实际跨发布、混合生产者、持续追加和淘汰的状态寿命才能支持连续适配主张。

## 当前约束

开发校准可以使用摘要或 K/V 派生监督，但必须与最终评价隔离，并报告教师访问及其成本。评价使用冻结协议和所有训练种子；不得以规模结果回调工作负载、版本边、指标或方法接口。

当前开发阶段已收束到 [计划](plan.md) 中的四个组件。任何新的规模训练先满足合同、canary、资源估计和明确启动要求。

## 2026-09-16：三个对比方案的前期设计

按用户要求建立 [baselines 入口](../../src/hstu_kvcache/baselines/README.md) 及三个独立方法目录，本轮仅调研、目录和文档，无实现、模型加载或实验。

- LR 借鉴 DroidSpeak 的连续区间选择、入口 hidden 缓存和真实部分执行；记录 HSTU 层间依赖、跨发布可变入口的附加存储及继承误差。
- TR 复用已有 `hybrid_tail_refresh` 的设计基础，明确旧前缀条件下的尾部重放，不能称为 target Full 的精确尾部替换。
- KT 借鉴 Heo 等人 arXiv:2608.03893v1 的 top-k 源层、跨 head 特征和 K/V 独立 centered ridge；HSTU 无 RoPE，转换直接作用于实际缓存，完整计入校准和人口重写成本。

后续可先在 CPU 上用真实冻结六层模型的小输入检查，再进行独立开发选择及连续生命周期评价。当前文档不修改旧封存动作／结果，不表示三个 baseline 或当前 EvoKV Design 已经实现或有效。

### 同日：按用户缩小范围实现基本原语

用户进一步要求正确性优先，不调性能、不搭最终实验流水线。已在三个目录新增 `core.py` 与公开接口：LR 捕获真实入口 E、区间重算、append／evict 和简单区间评分；TR 薄封装尾部重放；KT 单源层探针、跨 head centered ridge 及稳定旧快照映射。复用既有模型算子，未修改 backbone 或训练代码。

CPU focused checks 共 11 项通过（LR 4、TR 3、KT 4），覆盖数值参考及状态依赖。另使用既有冻结 Medium V0/V1 权重（`full_reuse_matrix_v1/shared_v0/checkpoint_100.pt` 与 `D14/checkpoints/v1/checkpoint_100.pt`；V0→V1 的旧 Full-only 准入记录已允许 Reuse）作一次数值探针：6L/H192/6heads，CPU FP32、4 线程、seed17 合成输入；2×32 行拟合 k=2、ridge=0.01 的 mapper，再检查 1×1024 历史上的完整重算端点、局部保留、映射、淘汰后 native append 与只读 query，均通过。源层排名为 in-sample；没有读取真实用户数据／标签，不是质量评价。

该探针进程峰值 RSS 约 3.46 GiB；加载两版权重各约 1.7–1.8 秒，单次 1024 历史完整构建约 0.2 秒，仅用于判断 CPU 开发资源，不作为性能收益证据。所有运行隐藏 GPU；未启动训练、正式人口评价或 GPU 任务。原语输入暂限无 padding 的等长 batch，完整 producer 谱系、数据划分和正式比较协议仍由后续调用方接入。这不代表当前 EvoKV Design 已实现或验证。

### 同日：补齐 Insight 1 功能对比接口（初版，模型绑定已由下节修正）

按用户要求补齐未来 GPU 评价所需接口，本轮仍只在 CPU 开发。新增
[competitor_probe.py](../../scripts/design/competitor_probe.py)：三个可执行迁移从同一 Parent
快照出发，统一用 Current 原生候选评分；Current Exact 只作控制。
[独立入口](../../scripts/design/run_insight1_competitors.py) 支持显式设备、配置、校准 UID
及分批执行，输出原始分数、CSV 指标、输入／模型来源和校准设置。

入口只读复用旧 Insight 1 的冻结 1024-event／64-candidate 面板及依赖 hash，
不验证已删除的旧 design，也不修改旧 sealed 实验。现阶段支持普通 D14 准入的
V0→V1 至 V3→V4 四条单边；V4→V5 的 partial-tail 语义尚未接入。
KT fitting／可选选层 UID 彼此隔离，并排除全部旧 3000 UID；拟合在 CPU 完成，
校准 teacher 规模单列。配置选择和最终未见划分仍须在正式评价前确定。

本次 8 项 CPU focused checks 通过：三方案统一评分、旧状态不变、与旧数学指标一致、
负恢复／近零分母、真实小型六层模型的 2+1 分批校准与评分输出、UID 隔离和
临时文件中的准入绑定。`--help` 及新增代码静态检查通过。
未读取真实评价面板／用户或新 checkpoint，未启动 GPU。恢复率采用全用户概率缺口
均值之比，保留所有配置；K/V 更新比例不是 FLOPs。该接口只支持单边功能差异诊断，
没有给出推荐质量、加速或连续适配结论。

### 同日：纠正模型绑定，使用本轮 6L／10L 链

用户指出新 6L／10L 模型链早已整理；初版入口误用了旧 Insight 1 的六层 checkpoint
合同和固定人口，不能将其称为已接好本轮 Motivation 2（原 Insight 1）。现已改为从
本轮 [Medium 清单](../../results/unified_training_2026_09/medium/seed17/checkpoints/chain.manifest.json)
和 [Large 完整清单](../../results/unified_training_2026_09/large/seed17/checkpoints/chain.v0_v5.manifest.json)
解析版本、父子哈希、层数、宽度、词表、数据绑定与既有准入。

入口增加 `--scale medium|large`、`--describe` 和显式三组 UID 的 `--uids`。
静态历史长度取自当前模型；64 候选沿用旧构造规则，但以本次指定的整组评价用户及
相应规模的词表生成，不再绑定旧 3000 人口，也不随评分 batch／max-users 改变候选库。
模型权重、数据映射及 UID 归属在实际运行前核对。旧契约绑定 helper/test 已由当前清单
解析检查替代；旧封存实验和权重保持原样。

CPU 合成检查覆盖数据因果性、候选构造与旧规则一致、Large 词表、当前链解析、10L
三方法端点、动态配置和完整入口。两套真实清单的十条边均通过仅元数据解析，CLI 的
Medium V1→V2、Large V3→V4 `--describe` 也通过；没有读取权重或用户 Parquet。
本轮 Medium 五条边与 Large 前四条边具备已有准入；Large V5@1 保留用户暂定与原门限
未通过状态，不自动换用 V5@2。未启动 GPU 或真实 Motivation 2 评价。
