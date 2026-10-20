# 跨规模、版本与历史条件：最小验证矩阵

2026-10-19 只读核对。已有模型足以开展跨组原生验证；当前 correction 的质量证据仍主要来自 Medium，不能把“权重存在”写成“方法已支持”。本页不启动训练、不改变既有角色。

用户已明确：不同模型分别选择拟合用户并训练自己的纠正参数，不研究跨模型权重迁移。
本页矩阵是可用条件清单，不是必须跑完的队列；当前先完成Medium最终规模与拟合设置，
其他模型只在支撑论文核心主张时加入，不能为增加实验数量自动展开全部组合。

## 已有资产与实际可测范围

| 组 | 已保留模型 | 当前能直接做什么 | 各组独立方法实验还缺什么 |
| --- | --- | --- | --- |
| Medium，Yambda-500M，30k 用户 | V0–V5，6层/192维/6头 | 各边原生 Reuse/Rebuild；已有 V2→V3→V4→V5 correction 连续质量与两协议结果 | 早期 V0→V1 需要在该截点重新建立起点校准，不能使用后来 V3 的编码器或物品特征 |
| Large，Yambda-500M，79,681 用户 | V0–V5，10层/320维/10头 | 原生重建、增量写入、查询及理论操作账 | 72维平坦码布局、读取/维护接口适配；独立起点编码器、固定特征表和逐发布校准 |
| Max，Yambda-5B，200k 用户 | V0–V5，16层/320维/10头 | 同上，另覆盖更深网络和更大目录 | 同上；不得复用 Large 的物品映射或校准权重 |
| RecFlow，4096 开发用户 | A epoch3、B–F epoch1，6层/192维 | 已有六模型生成式推荐训练/评价链 | reference-HSTU、SiLU、位置偏置及生成式 query/评分适配；尚无当前 correction 或缓存兼容性结论 |

24 个权重实体路径均存在。本轮未重新散列全部大文件；正式装载仍核对封存哈希。三规模同时改变模型、人口与目录，能说明不同配置下的适用性，不能作为只改变层数的控制实验；当前选定链也只有seed17。Yambda 均为 head_dim32、context1024、legacy/ELU+1；RecFlow 的最大长度1027包含生成 token，不能直接套用 Yambda 服务入口。[模型版本与端点选择](../unified_training_2026_09/model_versions.md)、[RecFlow 链及配置](../../results/recflow/development/window_6l_expanded_u4096_seed17/chain_summary.json)。

当前实现不仅内核限定6头：`adapter._flat` 识别 `(6,8)`，`ReleasePrefix.zero_anchors` 固定48维。共享12＋每头私有6在10头时为72维，并非10×rank8的80维。因此需要将状态接口改为显式总码宽，再验证读取、追加、锚码转换等价；这属于形状适配，不等于质量迁移。现有 V3 校准还依赖 [Structured 起点权重](../../results/code_structure_2026_10/shared12_private6/calibration/v3/calibration.pt) 与 `origin_v3.pt` 固定物品表，不能只修改构造参数后装入旧权重。[适配入口](../../scripts/query_relay_2026_10/adapter.py)、[校准依赖](../../scripts/query_relay_2026_10/calibrate.py)。

## 固定矩阵与选择规则

建议先固定 **三规模 × 三个版本位置**：早期 V0→V1（D231）、中期 V2→V3（D259）、晚期 V4→V5（D287），共9组。不按更新收益或 Full/Reuse 差距删组。每组以发布前历史长度分为1–256、257–767、768–1024，再按发布前72小时是否有行为分层；0历史单列。固定 UID 哈希顺序取样，保留发布后无请求用户及空分层。活跃度与请求率属于工作负载维度，不是新的方法模块。

三条验证线可并行，但结论分开：

- **原生形状线**：每规模先做 V2/V5 两个端点、固定候选池中的真实短/长历史，测重建与 query16；随后9组覆盖重建、追加、查询。它回答实现能否覆盖不同模型形状，不回答 correction 质量。
- **已有方法线**：复用 Medium 连续三发布、两种请求协议的已保存分数，并按事前分层描述；不重跑已完成总体评价。当前 V5 是自身持续演化状态，不能冒充独立 V4→V5 单跳试验。
- **新增方法线**：Large/Max 先各做 V2→V3 起点小校准，再沿自己的状态接续 V4/V5；早期组另建因果起点。单跳试验验证局部修正，连续链验证 D3，不能互相替代。先固定 fit/validation/development 身份与完整费用口径，再执行；不依据这轮结果换用户或省略失败边。

工程开发面板可先用每非空分层最多8人，保持小规模。它不提供总体优势或独立确认；后续确认人口需另行冻结。拟合与评测排除已有角色，不能因旧权重被清理就把旧校准用户重新称作未见用户。

## 数据、角色与形状 probe 入口

Yambda 权重统一为 `results/unified_training_2026_09/{medium,large,max}/seed17/checkpoints/v{0..5}/checkpoint_100.pt`。Medium/Large 的 dataset 位于 `data/processed/yambda500m_unified_v1/scales/{scale}/dataset.json`；Max 为 `data/processed/yambda5b_max_200k_v1/scales/max/dataset.json`。映射和用户表在 dataset 同目录；通过 `shared_listens_glob` 读取历史。**dataset 的旧 `model` 字段不可靠，使用 checkpoint config。** known_vocab_size 分别为1,380,509／2,224,809／3,530,650，均另加256个OOV桶。

S 线已收到固定工程候选池：Large/Max 的 `users.parquet` 中首次行为早于D231、UID最小的16人；Medium复用原32人。按各端点严格历史的真实长度选短/长样本，不读评分。Medium原分层基于D259，因此其V2探针只是复用工程样本，不能称V2发布前冻结的质量人口。V2/V5 截止分别为D245／D287；使用 [半连接历史装载器](../../scripts/design_one/prepare_history.py)，不能用D259截断历史包倒推D245，也不能跨规模复用映射后的 item ID。Large/Max未发现保留的映射历史包，需要从原数据装载；同一scale一次取D245前1024行及后续区间即可供两端点裁切。

角色来源：Medium 的 [现开发面板](../../results/design_system_2026_10/phase2/representative4096/panel/users.json)、[当前冻结记录](../../results/query_relay_2026_10/probes/A_itemgate6_rms_roll4/freeze.json)及 [确认排除记录](../../results/query_relay_2026_10/confirmation_item/medium_unused_actual_roles.json)；三规模旧角色还在 `results/selective_recompute_2026_09/panels/{scale}/{edge}/users.json` 与 `results/read_correction_2026_09/v2/panels/{scale}/{edge}/calibration_users.json`。新方法面板需合并已有校准/验证排除，不能直接把旧 evaluation 字段视为全局未见。RecFlow 的角色在 `data/processed/recflow_v1/population.npz`；其 development/admission/final 保持不变。

## 训练成本能说明什么

Medium V2–V5 的 `checkpoints/v*/train_result.json` 保留4卡、batch32、步数和同步中位步时。以 [V3](../../results/unified_training_2026_09/medium/seed17/checkpoints/v3/train_result.json) 为例，4096步×0.171秒≈701秒只是稳态计算时间近似，不能替代训练 wall time、GPU-seconds 或能耗；启动、数据、检查点与评估不在该乘积里。Large/Max 的现存封存记录能核对模型、epoch、batch与多卡配置，但未找到同口径完整训练计时，不能从目录时间或评价耗时补造。

[RecFlow chain_summary](../../results/recflow/development/window_6l_expanded_u4096_seed17/chain_summary.json) 保留 A–F 训练循环共12,919.5秒，其中 A三轮12,360.8秒，日更新89.6–151.6秒；[原配置](../../configs/recflow/window_6l_expanded_u4096_seed17.json)记录四rank执行。这不是全链11小时38分的端到端时长，也不是各GPU忙时之和。当前 relay 三次 `diagnostics.elapsed_seconds` 为72.07／74.44／75.81秒，但不包含先前 Structured 起点训练和外部源轨迹准备；[V3校准成本](../../results/query_relay_2026_10/probes/A_itemgate6_rms_roll4/calibration/v3/calibration.json)已把这些费用另列，不应只拿这三段计时与基础训练相除。

Large V5 原端点准入未全部通过，Large/Max 某些晚期历史报告标记 day300 不完整；保留这些边界，后续以实际输入覆盖重新声明窗口。原 Full/Reuse 原始评分多已清理，历史汇总可引用但不能假定仍可逐请求匹配。新测量需保留同请求身份、实际执行账，维护费用与总服务费用分列。

本轮检查完成；原生多形状 probe 已有真实模型与数据入口。跨规模 correction 的布局适配、起点校准是后续工作，不阻塞现有 Medium 机制、强基线和服务回放线。
