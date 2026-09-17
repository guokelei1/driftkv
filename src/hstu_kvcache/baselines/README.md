# 三个缓存兼容性对比方案

日期：2026-09-16。状态：**基本原语及单边评价接口已通过 CPU 检查；正式质量与性能尚未评价。**

本目录承载三个独立的 HSTU 对比方案。按用户最新范围，第一版只实现缓存变换、必要状态和简单选层，优先验证正确性，不搭最终实验流水线或调性能。`paper_design.md` 保留原论文笔记，各目录 README 保留适配设计与当前接口。

| 目录／简称 | 方案 | 保留的方法思想 |
| --- | --- | --- |
| [layer_recompute/](layer_recompute/README.md)／LR | 按层重算 | DroidSpeak 的离线层区间选择、入口隐藏状态缓存和部分重算 |
| [tail_recompute/](tail_recompute/README.md)／TR | 尾部 Token 重算 | 复用旧前缀，用当前模型重放最后 n 个真实历史事件 |
| [kv_translate/](kv_translate/README.md)／KT | 直接 K/V 翻译 | Heo 等人的跨层选择、跨 head 特征和闭式 ridge 映射 |

LR、KT 分别称为相应论文思想的 **HSTU 适配实现**，不称原系统复现。TR 是简单的尾部策略，不冒充 CacheBlend 等工作的 token 选择算法。对比的是三种可实际执行的缓存处理方式，最终是否有效须重新实验。

原论文核心设计另存为 [DroidSpeak Design 笔记](layer_recompute/paper_design.md) 和 [闭式 K/V 映射 Design 笔记](kv_translate/paper_design.md)，只保留方法流程与必要公式，不含背景和实验结果。

## 代码与文档放置

三个子目录分别维护方法本身的选择、变换和特有状态；共享 HSTU、缓存追加继续复用 [models/](../models/README.md)。本目录没有通用插件框架或独立训练体系；单边对比编排放在 [scripts/design/](../../../scripts/design/README.md)，不进入模型模块。

本文是对比方案的实现设计，不替代 [论文设计](../../../docs/paper_design.md)、[实验设计](../../../docs/experimental_design.md) 和论文正文。现有冻结动作、sealed motivation 结果不因新目录而改变；这三个方案须在自己的开发阶段确定配置，不能直接追加到旧规模 frontier。

## 第一版接口与输入范围

- LR：`capture_state`、`recompute_interval`、`append`、`retain_latest` 保存和更新实际入口 hidden；`profile_intervals` 用调用方给出的评分函数比较连续区间。
- TR：`recompute_tail` 直接重放尾部，`n=0` 为保留，`n>=N` 为整段重建。
- KT：`fit` 接收已对齐 source／target K/V，完成源层选择与 centered ridge；返回 mapper 的 `apply` 转换整份输入缓存。

当前只接收没有 padding 的等长 batch，或由调用方逐用户调用。原型已在 CPU 检查；模型、事件和缓存须由调用方放在一致设备，KT 拟合在 CPU 完成。所有有效缓存行都参加拟合，原始事件与缓存的语义对齐由调用方保证。方法原语不自动读取数据、划分 UID、接受模型发布或记录完整 producer 谱系；LR 仅维护计算必需的 E/KV 对应状态。原语的数值／状态正确性与下述完整实验协议是不同完成度。

CPU 验证入口：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest -q tests/test_baseline_*.py
```

## 当前 Motivation 2 功能探针

[competitor_probe.py](../../../scripts/design/competitor_probe.py) 提供三个组合接口：
`evaluate_batch` 接收已对齐的历史、候选和已拟合 KT，从同一 Parent 状态独立迁移并评分；
`path_records` 描述各配置；`summarize_scores` 汇总与 Current Exact 的功能差异。
Exact 缓存只用于评价锚点，不传给 LR、TR 或 KT。

[run_insight1_competitors.py](../../../scripts/design/run_insight1_competitors.py) 负责 Motivation 2
（原 Insight 1）的单边、单设备执行，默认 CPU。`--scale medium/large` 选择本轮已有的
六层／十层模型链，`--edge-index 0..4` 选择 V0→V1 至 V4→V5；数据、词表、history length
与 cutover 由当前 chain、训练配置和 contract 确定。模型已经存在，接口连接不需要等待训练。
`--describe` 可查看来源和准入状态，不加载权重或用户历史。

Medium 五边的普通 gates 均通过，Large 前四边可执行；Large V5 使用用户暂定的 `V5@1`，
其 gates 仍为 false，因此 V4→V5 目前只可查看描述，不能默认解锁评价或自动替换为 `V5@2`。
旧 sealed 结果保持独立，新评价只向新目录输出。

执行时用 `--uids` 显式指定 JSON 中的 `evaluation`、`fit`、可选 `selection` 三组，组内
唯一、组间互斥，并来自所选 scale 的人口。当前不会自动使用旧 3000 人口。独立校准与评价
划分仍需确定；启用 KT 时只用 `fit` 拟合，`selection` 用于独立源层排序检查。
CPU 拟合与 `--device` 指定的模型、缓存和评分执行分开。

64-candidate 面板由全部 `evaluation` 用户的发布前历史统一构造，使用当前词表的
recent／old-only／novel-bank 策略；`--max-users`、`--eval-offset` 和 batch 大小不改变
共同面板 bank。默认 LR 区间根据模型层数生成首层、最后两层与全部层，十层时为
`0:0,8:9,0:9`。输入格式与未执行的 Medium／Large 命令示例见
[脚本说明](../../../scripts/design/README.md#当前-motivation-2-对比入口)。

本探针的恢复率是 `1 - mean(abs(p_method-p_exact)) / mean(abs(p_reuse-p_exact))`，
不是逐用户比值的均值。所有配置行均保留，近零分母标记未定义，不在评价面板上选 winner。
`kv_updated_fraction` 表示 K/V 覆盖比例，不表示 FLOPs。该接口只检查单边功能差异，
不报告推荐 AUC，也不能证明连续状态效果。当前只有 CPU 合成检查，未启动 GPU 或真实面板评价；
正式评价仍需固定配置与划分。

## 后续完整比较的共同执行语义

1. 当前对比入口面向本轮六层 Medium 和十层 Large，使用各自当前模型链与数据。加载时以具体 checkpoint 的配置、item mapping 和时间语义为准；每条边须由 Parent/Current Full-only admission 解锁后再评价缓存兼容性。
2. 三个方法和 Reuse／Exact 使用相同的因果历史、请求、版本发布及评分头。发布时只处理当时保留的前缀；候选 query 是瞬态评分输入，不进入持久历史。
3. 第一版统一在每次获准发布时执行一次同步迁移。随后使用 Current 的完整原生 query 路径，以及各方法自身缓存上的 native append／evict。TR 不默认在每个请求前重复重算，KT 不默认每次读取再映射。
4. 各分支从共同的初始 Parent 缓存出发，此后保留自己的真实状态。再次发布处理上一轮的实际输出，包含混合 producer、近似迁移、新增行的继承误差及淘汰影响。重新初始化的单边检查只证明单边行为。
5. Exact 对照在发布时从相同的保留原始历史重建，随后也按既定流程追加。这与“当前模型从最初一直在线运行”的滚动轨迹不同；两种 reference 不混称。

`HSTUKVCache` 的 K/V 布局为 `[layers, batch, sequence, kv_width]`，其中 `kv_width = heads × head_dim`，没有原始事件、producer 或层输入 hidden。现有六层历史资产使用 legacy／ELU+1、无 RoPE；“HSTU-native CC 评分”不表示 K/V 已经采用 `hstu_reference` 的 SiLU 形式。新 checkpoint 必须按其实际算子读取缓存，不能硬套旧设置。

状态只增加方法实际需要的元数据：事件顺序及时间、写入来源、迁移目标和近似处理记录。LR 还须表达同一 token 不同层的来源；现有单 token 一个 producer 标签不足以表达它。KT 的“已映射到 Current 空间”也不表示该行是 Current Exact。

## 开发选择与评价分离

- **Fitting**：KT 使用发布前可获得历史上的 source／target 成对 K/V，源层探针拟合／排序使用 fitting 内的 UID 子划分；LR 使用捕获入口 hidden 的真实执行状态。教师访问和源状态回放均计费。
- **Development**：独立 UID 上选择 LR 区间、TR 的 n、KT 的 k／ridge，并检查质量。所有开发选择与最终评价分开；线上不按目标标签选动作。
- **Final evaluation**：先冻结用户划分、发布边、候选配置、选择规则和成本口径，再报告所有既定版本与训练种子。新一轮精确 UID 划分尚待实验协议确定，不能把历史已看用户重新称为全未见。

第一版每条发布边共享一个配置，不按测试用户单独优化。可使用发布前开发数据衡量 task quality 或与 Full 的输出差异，但必须标明用途；最终评价的 AUC、最优替换层等不反哺配置。

质量至少报告逐发布 AUC 与 log-loss、相对 Reuse 的改善，以及相对 Exact−Reuse 缺口的恢复。只有对照差值可解释时才报告恢复比例；近零或反号的分母单列，不裁剪负收益或超过 100% 的结果。K/V 重建误差是诊断量，不能替代推荐质量。

## 成本必须包含什么

| 阶段 | 共同口径及特有成本 |
| --- | --- |
| 共享准备 | source／teacher 生成、校准、层搜索、ridge 求解、必要状态回放 |
| 首次建立状态 | 公共 Parent 构建单列；LR 的附加 hidden 捕获、写入和存储单独收费 |
| 每次发布 | 真正执行的 block／映射、原始历史与 KV 读取、结果写回、临时工作内存 |
| 后续服务 | 原生 query／append／evict，以及 LR 附加 hidden 维护 |
| 系统成本 | CPU 与 GPU 时间分别计量；主机／设备搬运、峰值及持久内存、发布停顿与请求延迟 |

可报告“相对 Reuse 的增量”，同时保留完整生命周期成本；不混用只含在线矩阵乘法的 KT 成本与含教师和人口重写的其他方法成本。共用教师可在实际开发中复用，但每个方法独立部署需要的完整准备费应在对比中明确归属。理论 MAC/FLOPs、实际设备耗时与 I/O 是不同账本。

## CPU 检查与后续范围

第一版用小型模型和可解析矩阵检查公式、重算依赖与状态更新；当前入口另检查六层／十层配置与来源连接、数据划分和批次汇总。真实模型已存在，后续可用所选链权重作 CPU 数值探针。使用设备无关的 PyTorch 实现，不运行 GPU 任务。当前阶段无需开发质量搜索、完整人口回放或成本分析系统。

入口与数据连接的 CPU 检查：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest -q tests/test_insight1_competitor*.py tests/test_competitor_*.py
```

真实权重检查保留模型结构，用小批合成事件输入验证同一计算路径；这不是用户质量数据。设 `CUDA_VISIBLE_DEVICES=""`，限制 CPU 线程；无需复制或裁剪 checkpoint 的模型结构，缩小的是输入量。

先测短前向／迁移，再估计扩大输入的资源。CPU 检查覆盖公式、时间边界和状态传递；GPU 可用后再做精度／设备 canary，质量和性能评价另行开展。本文不授权长训练或正式人口运行。

## 调研来源

- [DroidSpeak，USENIX NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan)：对应 LR 文档中的论文依据和 HSTU 改动。
- [Cross-Model KV Cache Transfer，arXiv:2608.03893v1](https://arxiv.org/abs/2608.03893v1)：对应 KT 文档中的公式、层选择和适配边界。
