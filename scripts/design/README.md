# 适配探索与对比脚本

当前 Design 已完成设计，尚未实现和验证。本目录保留历史 native read-adaptation 数据生成链，供后续复用，不代表当前 Design 的实现或验证。下述校准和结果均指历史探索。

旧 runner 的 query-affine 校准已退休；仅保留通过 `--evaluation-from` 读取既有权重的评价路径，不再尝试导入已删除的拟合模块。

这些历史运行读取冻结的
Medium 模型、manifest 与已有事件；长训练仍由仓库根目录的 Medium/Large 入口负责。

`data.py`、`run.py`、`diagnose_query_holdout.py`、`diagnose_summary_objective.py` 和
`diagnose_decoder_closure.py`提供当前校准所需的场景、面板、坐标和共享执行原语。
`diagnose_native_input.py`、`diagnose_native_coverage.py`、`fit_native_ablation.py` 与
`native_service.py`生成 native C 和两个必要消融。

`evaluate_native_base.py`、`run_native_base.py`、`report_native_base.py`、
`report_native_coverage.py`、`report_native_flops.py`、`audit_native_residual.py` 与
`audit_native_response_scale.py`产生当前的质量、覆盖、残余和计算证据。
`diagnose_constant_affine.py` 与 `finalize_base_method.py`只读取冻结结果，生成论文
Design 1 的机制图表输入和定稿索引。

历史 v3--v15、PRO、宽源及已删除运行的入口不再保留。当前结果范围见
`results/design/analysis/`，论文叙述以 `/home/gkl/work/paper/main.tex` 为准。

## 当前 Motivation 2 对比入口

[competitor_probe.py](competitor_probe.py) 将三个对比方案接入 Motivation 2（原 Insight 1）的功能指标：
`evaluate_batch` 从同一 Parent 快照分别执行 LR、TR、已拟合 KT，并用 Current 原生路径评分；
`path_records` 记录配置及 K/V 更新比例；`summarize_scores` 汇总概率／logit 差异、
Bernoulli JS、top-1 一致率、top-10 重叠和排序相关性。Current Exact 只作为评价锚点，
不会向迁移方法提供其缓存。方法接口见 [baselines/](../../src/hstu_kvcache/baselines/README.md)。

[run_insight1_competitors.py](run_insight1_competitors.py) 是独立的单边、单设备入口，默认 CPU。
`--scale medium/large` 分别选择本轮已存在的六层／十层模型链；`--edge-index 0..4`
对应 V0→V1 至 V4→V5。模型已完成，不需要等待训练结束才能连接这些接口。
[competitor_models.py](competitor_models.py) 从当前 chain、训练配置及 contract 确定具体
checkpoint、数据、词表、历史长度与 cutover，并检查所选边的准入状态；实际执行核验依赖 hash。
旧 sealed 记录保持独立，新结果写入新的目录。

Medium 五条相邻边均已通过普通准入；Large 前四条边可执行。Large V5 保留用户暂定的
`V5@1`，其普通 gates 仍为 false：V4→V5 可用 `--describe` 查看来源与状态，实际评价不会
默认通过，也不会自动改选 `V5@2`。

`--describe` 只读取模型链与数据来源的元数据，不加载权重、用户历史或 CUDA 设备，例如：

```bash
PYTHONPATH=src:scripts python scripts/design/run_insight1_competitors.py --scale medium --edge-index 1 --describe
PYTHONPATH=src:scripts python scripts/design/run_insight1_competitors.py --scale large --edge-index 3 --describe
```

实际执行必须显式提供 `--uids` JSON，格式为
`{"evaluation": [101, 102], "fit": [201, 202], "selection": [301, 302]}`
（仅示意格式，须为每个 scale 确定足够的实际用户）。三组内 UID 唯一、组间互斥，来自所选
scale 的人口；启用 KT 时 `fit` 非空，`selection` 可省略。不会自动采用旧 3000 用户。

[competitor_data.py](competitor_data.py) 提取严格早于 cutover 的完整历史窗口，静态窗口
第一个 delta 为零，其余为真实相邻事件间隔。候选策略保留 recent 16、old-only 16 和
novel bank 补足至 64 的规则，使用当前 scale 的 known-item 范围，排除 OOV 候选。
bank 一次性使用 JSON 中**全部 evaluation 用户**的发布前历史构造；`--max-users` 和
`--eval-offset` 只改变实际评分子集，batch 大小也不改变面板。若 bank 不足以填满候选则报错，
不从未来行为补齐。

下面是**未执行的未来命令示例**。独立校准与评价划分仍需确定；正式评价前固定配置与划分，
再以小探针估计资源。当前未启动 GPU 或真实评价。ridge 拟合在 CPU 完成，`--device`
控制模型、缓存变换和评分所在设备。

```bash
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/design/run_insight1_competitors.py \
  --scale medium --edge-index 1 --device cuda:0 --max-users 32 --batch-size 2 \
  --uids /path/medium_uid_split.json \
  --map-ks 1,2 --ridge 0.01 \
  --output results/baselines/motivation2/medium_edge1_probe

PYTHONPATH=src:scripts OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/design/run_insight1_competitors.py \
  --scale large --edge-index 3 --device cuda:0 --max-users 32 --batch-size 2 \
  --uids /path/large_uid_split.json --layer-intervals 0:0,8:9,0:9 \
  --map-ks 1,2 --ridge 0.01 \
  --output results/baselines/motivation2/large_edge3_probe
```

默认层区间随层数生成：首层、最后两层、全部层。Medium 为 `0:0,4:5,0:5`，Large 为
`0:0,8:9,0:9`；编号从零开始、两端包含。默认尾长度为 `0,N/8,N/4,N`，N 来自模型 context。

结果包括 `scores.npz`（评分 UID／全部面板人口 UID／路径／原始分数／候选）、`metrics.csv`
和 `summary.json`（配置、来源 hash、面板人口规模、校准用户、teacher 规模、选层结果与指标）。首版会在每次启动时重新拟合
共享 mapper；没有 mapper 文件管理或多卡调度。`--map-ks ''` 可只检查 LR/TR。
`selection` 省略时源层探针标为 `in_sample`；它仍仅使用拟合用户，不使用评价用户。
输入用户是否已被历史开发查看仍需照实记录，正式未见评价需明确划分。

CPU 接口检查（不加载真实面板和模型）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest -q tests/test_insight1_competitor*.py tests/test_competitor_*.py
```

层区间采用从 0 开始、两端包含的编号。概率 gap 恢复率为
`1 - mean(abs(p_method - p_exact)) / mean(abs(p_reuse - p_exact))`，对全部用户和候选
先求均值再求比值；不是逐用户恢复率的均值。Reuse gap 近零时保留行并标记恢复率未定义，
所有候选配置均保留，不根据评价结果选 winner。`kv_updated_fraction` 只表示缓存覆盖比例，
不代表 FLOPs 或实测加速。这是单边功能差异探针，既不是推荐 AUC 评价，也不证明连续缓存生命周期效果。
