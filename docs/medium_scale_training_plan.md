# Yambda-500M Medium 训练与评估记录

记录始于：2026-08-28；状态整理：2026-09-05。

状态：**共享 v0、D7 v1…v10、D14 v1…v5 共 16 个正式 checkpoint 已完成；基础 32 个 Full-only、D14 v1…v4 的 12 个 Reuse、D7 forced diagnostic 的 20 个 Reuse，以及 v5 的 E3/E7/E14 Full+Reuse 均已封存；D14/E14 direct Reuse 三角已完成 15/15 格。旧 PRO 路线未在 Medium 启动。**

本文保留训练配方的选择依据、最初资源估计和实际执行变更，不是当前待办或重新启动队列。
第 2–8 节解释当时的准备过程，第 9 节记录已完成的执行及扩展。
论文当前结论见 [motivation_observations.md](motivation_observations.md)。

## 1. 这份记录说明什么

Medium 准备阶段将原 Small 流程通用化，完成了合同、数据 manifest、canary 和训练评估队列。
最初的 Full-only 阶段先检查 30k-user、6L 模型的更新改善；相邻与跨版本 Reuse 后来由独立合同补齐。
该阶段不是对旧 Small C32 estimator 的继续调参，也不是当前 EvoKV 方法的效果验证。

旧 Small Insight/Design 文档已随废弃结果清理，其原文在清理包中，不再引用失效的活动结果路径。
保留范围与恢复限制见 [结果索引](../results/README.md)。

## 2. 为什么沿用 day217，而不是改到 day150

最终选择 **foundation `[0,217)`、stream `[217,300)`**，并只比较 D7 与 D14。

现有 Medium population 与 compact item mapping 都由 day217 之前的数据冻结。只把 cutoff 改成 day150
并不是简单缩短基础训练：现有 30k Medium 用户中有 3,346 人在 day150 前没有 listen history；若按原
SHA-256 规则重新选择 day150 population，只与当前 population 重合 26,654 人，即会更换约 11.2%
用户，同时必须重建 item mapping。这样会把 population/mapping 变化与模型规模变化混在一起。

沿用 day217 有三个好处：

- 直接复用已审计的 30k 固定 UID population 和 1,380,509-item foundation mapping；
- 与 Small 的 foundation boundary 一致，scale comparison 更干净；
- 83 个基础 stream days 支持 10 条 D7 edge 和前 4 条 D14/E14 edge；随后以同一 population/mapping
  增补 v5，并按统一 E14 口径保留 `[287,301)` 的实际观测覆盖。

基础对称矩阵使用半开区间 `[0,300)`。后续 v5 扩展使用 `[287,301)` 作为 E14，并以实际日期范围、
请求数和 source coverage 记录尾部差异；展示名称不再另立一档。

## 3. 从 Small 迁移到 Medium 的历史改造

### 3.1 继承的训练语义

当时沿用的 Small 训练语义如下，已写入 Medium 合同，以隔离 scale 变量：

- F-only HSTU-native binary objective；所有合格真实请求均进入训练；
- 用户等权：每个窗口内按用户请求数取逆权重，再做全局归一化；
- 严格时间因果：只取 query timestamp 之前的 listen，同 timestamp 原子化，query 本身不写回 history；
- 一个 foundation pass；每个增量版本一个 pass；
- foundation LR `2e-4`，增量版本 LR `5e-5`，AdamW、weight decay `1e-4`；
- 每个增量版本从 direct parent 权重 warm-start，但创建全新的 AdamW state；
- seed17 为首个 Medium seed；不做 early stopping、label-driven checkpoint selection 或 per-edge 调参；
- 只保留 whole-pass final checkpoint；raw Full scores 先封存，再进行 label join 与 metric adjudication；
- 物理 GPU2/3 上一次只运行一个双 rank FSDP `FULL_SHARD` job，bf16 compute、fp32
  reduction/optimizer；GPU0/1 未参与这一基础阶段。后续四卡 Reuse 和 v5 扩展见第 9 节。

LR 与 batch size 已写入 Medium 基础及执行合同。当时的规则要求：若 canary 表明数值不稳定，
只能在读取 formal quality 之前整体更换 recipe 并创建新合同，不能按 edge 调整。

### 3.2 已完成的参数化与历史加载改造

早期脚本曾写死 Small 的 dataset、人口、vocabulary、4L/H128/4 heads 和 context512。
现行 manifest builder、trainer 和 evaluator 已能从合同、dataset manifest 与 checkpoint 读取
对应配置，并被 Medium/Large runner 复用。部分工具仍保留 Small 默认参数，不能因此裸跑默认入口；
应由对应模型 runner 传入已封存配置，见 [脚本索引](../scripts/README.md)。

训练器与评估器已复用支持时间范围和前缀长度限制的 history loader，相关严格 prior、
同 timestamp 原子性和 1024-context 行为由测试覆盖。这些基础改造不是下一阶段 Design 的待办。

## 4. 已封存的 Medium 配置

| 项目 | Medium 值 | 当前性质 |
| --- | ---: | --- |
| population | 30,000 fixed UIDs | 已由 unified scale contract 定义 |
| foundation/stream | `[0,217)` / `[217,300)` | 基础矩阵已冻结；v5 尾段扩展另见第 9.4 节 |
| known item mapping | 1,380,509 items，固定于 foundation | 已物化并有 hash |
| OOV | stable 256 buckets | 已冻结 |
| architecture | 6 layers、hidden192、context1024 | 已由 unified scale contract 定义 |
| attention heads | 6（head dim 32） | 已冻结 |
| query schema | 4 behaviors、3 query types、query type id 2、1 query action | 已冻结 |
| seed | 17 | 已完成的训练 seed |
| training pass | foundation 1；每个 update 1 | 已冻结并执行 |

按 `num_items=1,380,509+256`、6L/H192/6 heads 计算，模型约有 **266,259,265** 个参数；一份只含
FP32 model weights 的 checkpoint 理论下限约 **0.992 GiB**。该数字不包含序列 activation、FSDP
通信 buffer、gradient 和 Adam state。

## 5. Full-only recipe matrix

一个共享 v0 在 `[0,217)` 上训练完成后，D7 和 D14 从同一个 v0 分成两条相互独立的 direct-parent
candidate chain。下面的请求数是 2026-08-28 使用固定 30k population、known-item join、严格先验
listen 和 `(uid,timestamp,item)` 去重得到的 **label-free planning upper bound**；正式 manifest 按
事前规则排除 conflicting feedback group，因此最终 eligible count 应查看已完成的 checkpoint seal，
不能将下面的历史规划上限当作正式计数。

基础期规划上限为 2,005,790 个请求组、23,347 个有请求用户。

### 5.1 D7：10 个增量版本

每条 edge 在 cutover 后报告 E3 和 E7；最后一条 v9→v10 在 day287 cutover，E7 于 day294 结束。

| Candidate | 训练窗口 | 请求组上限 | 用户数 |
| --- | --- | ---: | ---: |
| v1 | `[217,224)` | 78,085 | 10,720 |
| v2 | `[224,231)` | 72,116 | 10,537 |
| v3 | `[231,238)` | 68,680 | 10,375 |
| v4 | `[238,245)` | 68,608 | 10,471 |
| v5 | `[245,252)` | 64,782 | 10,295 |
| v6 | `[252,259)` | 66,368 | 10,434 |
| v7 | `[259,266)` | 65,309 | 10,278 |
| v8 | `[266,273)` | 66,568 | 10,470 |
| v9 | `[273,280)` | 67,882 | 10,527 |
| v10 | `[280,287)` | 66,042 | 10,543 |

D7 update 总计约 684,440 request-passes。

### 5.2 D14：基础4个增量版本，随后扩展至v5

每条 edge 在 cutover 后报告 E3、E7 和 E14；基础矩阵最后一条 v3→v4 在 day273 cutover，E14 于
day287 结束。后续独立合同又训练 v5 `[273,287)`，并将 `[287,301)` 统一报告为 E14。

| Candidate | 训练窗口 | 请求组上限 | 用户数 |
| --- | --- | ---: | ---: |
| v1 | `[217,231)` | 150,201 | 13,994 |
| v2 | `[231,245)` | 137,288 | 13,823 |
| v3 | `[245,259)` | 131,150 | 13,676 |
| v4 | `[259,273)` | 131,877 | 13,813 |

D14 update 总计约 550,516 request-passes。

### 5.3 Full-only 报告和 recipe 裁决

每个 `(edge,horizon)` 必须完整报告 Parent Full 与 Current Full 的 ROC-AUC、dislike PR-AUC、log-loss
和 Brier，以及 Current−Parent delta；edge 等权汇总，不能只展示正 edge。D7/D14 的 recipe acceptance
rule、bootstrap 单位和 failure policy 必须在正式训练前写入不可变合同，不能看完矩阵再发明门槛。

当时的 Full-only scan 阶段只用 Full-only 结果判断 release training signal 是否足够稳定，不读取 Reuse/PRO。recipe
scan 中的 v1…vn 只是 direct-parent candidate chain，不自动成为 serving lineage。如果某 candidate 在
后续正式 admission 中被拒绝，serving parent 与 cache lineage 保持不变；其 descendant 不能被挑出来
接到已接受 parent 上，必须按已接受 parent 重新训练。

## 6. 原始阶段划分与后续去向

M0–M5 是当时的准备与执行顺序，相应合同、代码、canary、模型和裁决现已保留；
后续 Reuse 扩展见第 9 节。以下历史阶段不是新的启动清单。

1. **M0：合同与数据 seal。** 新建 Medium D7/D14 prospective contract，冻结 population/mapping hash、
   `[0,300)` complete-day boundary、全部 train/eval windows、metrics、failure policy、seed 和资源上限。
   现有 `yambda500m_streaming_windows_v1` 只允许规划，不能直接当训练授权。
2. **M1：通用化代码与 CPU correctness。** 参数化 manifest builder、trainer、Full-only evaluator 和
   rolling runner；增加 30k selector、1024 causal prefix、direct-parent、raw-first seal 及 rejected-lineage
   测试。不得复制一套 Medium model module。
3. **M2：focused GPU2/3 双卡 canary。** 固定 global batch 32，即 16/rank；验证 6L/H192
   checkpoint round-trip、direct-parent warm start、Full-only raw seal 与 adjacent-Reuse mechanics。
   不比较三卡，也不读取 canary quality 来选择执行配置。
4. **M3：共享 v0 foundation。** canary 通过且用户显式 launch 后，才训练唯一一份 `[0,217)` v0。
   完成 fixed checkpoint 与 Full-only sanity；不能把固定 endpoint 自动叫作 accepted release。
5. **M4：D7/D14 Full-only scan。** 两条 branch 串行执行；每个 candidate 训练完成后先产生并 seal raw
   Full score，再 join label；整个阶段禁止 Reuse、PRO 和 cache 指标。
6. **M5：冻结 Medium release recipe。** 只依据事前 Full-only rule 裁决 D7/D14。若两条 recipe 均不
   稳定，先修 training recipe 并创建新的 prospective evidence；不能用 PRO 掩盖上游问题。
7. **M6：已退出主线的 PRO qualification 设想。** 当时拟在 accepted Medium edges 上复核
   candidate-shared correction、AV boundary、persistence 和约 10%/20% 两个 PRO 预算点。
   这不是当前 EvoKV 的后续计划；Medium PRO 未启动，不能将后来的 Insight 诊断视为该方案已经完成。

当前 theta3 仍受 blind boundary 保护。M0 必须一次性冻结第三个及以后 candidate 的 data、release、
admission、metric 和 failure contract；在该合同和显式 launch 之前，不训练或读取任何 theta3 结果。

## 7. 基础矩阵的历史规划预算

以下为 v5 扩展之前的初步估计，不包含后续新增任务，也不是当前剩余工作量。

共享 foundation 加两条 update branch 共约 **3,240,746** 个 label-free planning request-passes，产生
1 个 v0、10 个 D7 candidate 和 4 个 D14 candidate，共 15 份 final checkpoint。按每份 0.992 GiB
纯 FP32 weights 估算，必要 checkpoint 约 14.9 GiB；执行时建议至少预留 30 GiB 临时余量，完成 seal
后不保留 optimizer state、partial checkpoint 或重复 rank shard。

当时双卡固定 batch 16/rank（global batch 32），上限请求数对应约：

| 阶段 | Optimizer steps |
| --- | ---: |
| foundation | 62,681 |
| D7 全链 | 21,393 |
| D14 全链 | 17,206 |
| 合计 | 101,280 |

因此双卡改变的是每个 rank 承担的 batch 和 wall-clock，不改变上述 optimizer step 数或统计 batch。
按满 context 粗略计算，Medium 的 token-linear block FLOPs/request 约为 Small 的 6.75 倍，
attention term 约为 9 倍；同时 foundation 请求上限约为 Small 已训 642,483 请求的 3.12 倍。因此完整
Medium foundation 可能达到 Small foundation request-FLOPs 的约 21–28 倍，实际值必须由真实历史
长度分布和 canary step time校准，本文不承诺 wall-clock 时间。

## 8. 已完成的基础代码准备

原准备清单对应的基础改造已完成：

- manifest builder 已按合同与 dataset manifest 构造人口和窗口；
- trainer 已支持对应 dataset、vocabulary、模型配置、context 和 history loading；
- Full/Reuse evaluator 已读取 checkpoint 和模型运行配置；
- Medium runner 已保留串行队列、resume audit、raw seal 和完整矩阵报告；
- Medium canary、资源记录和执行合同已封存。

原 v0 和增量版本已完成，不应按这份历史清单重新训练。任何新的长训练仍需独立合同、
资源估算、passing canary 和用户明确启动；本文不提供新授权。

## 9. 已完成执行、保留入口与扩展记录

以下命令保留用于解释已有产物的来源，不是需要重新执行的步骤。
基础队列始于 2026-08-28；后续扩展各由独立合同记录。

当前基础合同、GPU2/3 双卡执行/admission 补充合同、通用化 trainer/evaluator、数据 manifest 和可恢复
runner 已实现。统一入口是：

```bash
PYTHONPATH=src python scripts/run_yambda500m_medium_full_reuse_matrix.py --mode plan
```

它固定执行 15 个 formal checkpoint 和 32 个 Full-only 评测 cell。formal 顺序是先完成共享 v0、
D7 v1…v10、D14 v1…v4 的所有 checkpoint；随后每条 edge 先产生 Old Full / New Full raw seal，
在 primary horizon（D7/E7、D14/E14）形成 release-eligibility seal。只有进入连续 accepted diagnostic
lineage 的 edge 才解锁 adjacent one-hop Reuse；拒绝 edge 及其既有 candidate descendants 仍完整报告
Full-only，但不构造 Reuse。三种报告对象是：

- `parent_exact_rolling`：Old/Parent 模型及其自身 rolling cache；
- `current_exact_rolling`：New/Current 模型及其完整 Current cache；
- `one_hop_reuse_rolling`：Current 模型读取紧邻 Parent 在 cutover 生成的 cache，之后由 Current append。

基础队列不执行 recursive 或 long-age Reuse；后者已由第 9.5 节的独立诊断补齐。
基础 `summary.md/json` 并列给出三条路径的 ROC-AUC、log-loss，
以及 Reuse 相对 Old→New 的 AUC/log-loss gain retention。

CPU 数据准备已经完成；可重复验证但不会覆盖已有 seal：

```bash
PYTHONPATH=src python scripts/run_yambda500m_medium_full_reuse_matrix.py --mode prepare
```

正式运行前的双卡 smoke 使用物理 GPU2/3、真实 6L/H192/context1024、batch 16/rank（global 32），
分别训练 v0/v1 两步，并对一个 D7/E3 小 cohort 顺序验证 Full-only raw-first 与 Reuse mechanics：

```bash
PYTHONPATH=src python scripts/run_yambda500m_medium_full_reuse_matrix.py --mode smoke
```

GPU2/3 各至少空闲 40,000 MiB 时 smoke 才会启动；GPU0/1 不检查、不占用。2026-08-28 canary 已通过：
v0 峰值 reserved 显存为 7.3/6.8 GiB，v1 为 8.7/9.0 GiB；118 个请求的 Full-only 和三路径 Reuse
小流程均完成。该小 cohort 的 quality 不作解释。OOM 会保留日志并停止，不会在 formal 中静默降低
batch size。当时用户显式启动基础长任务的命令为：

```bash
PYTHONPATH=src python scripts/run_yambda500m_medium_full_reuse_matrix.py \
  --mode formal --acknowledge-long-run RUN_MEDIUM_D7_D14
```

所有阶段均可重复执行并跳过 hash 验证通过的完整产物。结构化总日志位于
`results/yambda500m_medium_seed17/full_reuse_matrix_v1/logs/pipeline.jsonl`，每步 stdout/stderr 有独立
日志，`pipeline_state.json` 保存最近状态。若需要人工分开运行，可使用相同 acknowledgement 的
`--mode train` 和 `--mode evaluate`；遇到 partial checkpoint/raw 目录时 runner 会停止等待审计，
不会覆盖或跳过失败 parent。

### 9.1 D7 完成后的 D14 CPU runtime 冻结

2026-08-28 的实际运行中，15 个 checkpoint 全部完成，D7 的 E3/E7 共 20 个 Full-only cell 也全部
完成。D7 使用原始的每 rank 4 个 history threads。其后在第一个 D14 raw 结果形成前停止旧 evaluator，
单独冻结 `yambda500m_medium_hstu_native_d14_cpu_runtime_v2.yaml`，只改变 D14 的执行并行度，不改变
数据窗口、checkpoint、world size、batch、指标或 admission。

D14 固定使用 GPU2/3 和与两张卡同属 NUMA1 的 28 个独立物理核：rank0 绑定 CPU 28–41，rank1 绑定
CPU 42–55；每 rank 为 14 个 history/Arrow CPU threads、4 个 Arrow I/O threads、4 个 Torch/OMP
threads。这里没有再扩到 40 个 history workers，因为 NUMA1 只有 28 个物理核；继续增加会使用同核
超线程或跨 NUMA 访存，不能当作 40 个独立 worker，且更容易争抢内存带宽。

正式恢复前已用 D14 v0→v1/E14、每 rank 最多 16 用户完成 raw-only canary：660 行 raw 与 seal 行数
一致，SHA-256 一致，未读取 label/quality，且 raw seal 记录了完整线程数与 affinity。恢复后的首个正式
D14 evaluator 实测每 rank 约使用 12.2 个 CPU 核，说明新的 14-core affinity 与并行扫描已实际生效。

### 9.2 剩余 D14 Reuse 的四卡与高 batch runtime

完成 `v0→v1` 的 E3/E7/E14 双卡 Reuse 后，执行遥测显示 rolling evaluator 仍受较小 user cohort 和
两 rank 串行 fallback 限制。用户于 2026-08-28 明确授权：保留已完成结果，从下一条 edge 开始只将
未完成 D14 Reuse 切换为 GPU0/1/2/3 四 rank。新 runtime 由
`yambda500m_medium_hstu_native_d14_reuse_4gpu_runtime_v3.yaml` 事前冻结：cohort 16→32、query chunk
128→256，每 rank 14 个与 GPU 同 NUMA 的独立物理核，共使用全部 56 个物理核；不改变人口、窗口、
checkpoint、三路径定义、label、metric、admission 或 lineage。

正式续跑前以 `v1→v2/E3`、每 rank 最多 64 用户执行 raw-only canary。四 rank 共 1,000 请求、3,000
行，严格满足每请求 Parent/Current/Reuse 三路径与 raw hash/行数守恒；四卡 peak reserved memory 为
6.7–7.3 GiB，明显低于 40 GiB 门槛，未读取 canary quality。当时通过同一个 `--mode evaluate`
完成续跑，跳过双卡完成产物，对剩余九个 D14 Reuse cell 使用四卡 runtime；这些 cell 现已完成。

### 9.3 D7 全矩阵 forced-Reuse 诊断补跑

正式 D7 admission 已经封存且不得重写。用户于 2026-08-29 进一步要求：不把 admission gate 当作执行
门，补齐 D7 十条 candidate edge 在 E3/E7 上的全部 20 个相邻 one-hop Reuse 诊断。该要求由独立合同
`yambda500m_medium_hstu_native_d7_forced_reuse_diagnostic_v1.yaml` 承接，并冻结全部 D7 admission seal
和 v0…v10 checkpoint seal 的 hash。其“bypass”只允许执行 separate diagnostic，不把任何 edge 改成
accepted release，也不修改正式 `D7/reuse/` 或顶层 summary。

运行沿用 D14 已完整实测的 GPU0/1/2/3 四 rank、cohort 32/rank、query chunk 256/rank 和 14 个独立
物理 CPU 核/rank。没有继续放大 batch：D14 正式矩阵的最坏 rank 已达到 44,950 MiB peak reserved，
而 A40 总显存为 46,068 MiB，现值已是有完整矩阵证据支持的最大安全档。正式补跑前，以历史最长的
`v9→v10/E7`、最多 64 用户/rank 做 raw-only canary，不读取质量；通过后串行运行 20 格。

```bash
PYTHONPATH=src python scripts/run_yambda500m_medium_full_reuse_matrix.py \
  --mode d7-forced-reuse-canary

PYTHONPATH=src python scripts/run_yambda500m_medium_full_reuse_matrix.py \
  --mode d7-forced-reuse \
  --acknowledge-long-run RUN_MEDIUM_D7_FORCED_REUSE
```

输出位于 `results/yambda500m_medium_seed17/full_reuse_matrix_v1/D7/forced_reuse_diagnostic_v1/`；每格
仍先 seal 三路径 raw，再 join label，并在独立 cell seal 中绑定 forced contract、原 admission seal、raw
和 adjudication hash。失败或 OOM 停止串行队列，不自动降 batch。

### 9.4 D14 v4→v5 单边扩展

用户于 2026-08-29 授权补齐 D14 v5。v5 的增量训练窗口 `[273,287)` 完整且直接继承 sealed v4；其
E3 `[287,290)`、E7 `[287,294)` 与 E14 `[287,301)` 均按统一 horizon 名称报告。day300 的实际原始
feedback row 为 12,962 条，最后时间为当日第 79,995 秒；这一覆盖差异通过日期范围和请求数保留，
不再改变 E14 的展示名称。

独立入口 `run_yambda500m_medium_d14_v5_extension.py` 先构建 `[217,301)` 扩展 manifest，再做四卡
raw-only canary，最后串行执行 v5 training、三个 Full-only 和三个 adjacent-Reuse cell。训练保持
global batch 32（8/rank）；Full 为 batch128/rank；Reuse 沿用 cohort32/query256；四 rank 各绑定 14
个独立物理 CPU 核。canary 在最长 `E14` 上同时覆盖 Full 与 Reuse，不读取质量。

```bash
PYTHONPATH=src python scripts/run_yambda500m_medium_d14_v5_extension.py --mode prepare
PYTHONPATH=src python scripts/run_yambda500m_medium_d14_v5_extension.py --mode canary
PYTHONPATH=src python scripts/run_yambda500m_medium_d14_v5_extension.py \
  --mode formal --acknowledge-long-run RUN_MEDIUM_D14_V5_EXTENSION
```

结果位于 `results/yambda500m_medium_seed17/full_reuse_matrix_v1/D14/v5_extension_v1/`，不会修改原
D14 v1…v4 checkpoint、admission、raw seal 或 summary。

### 9.5 D14/E14 跨版本 direct Reuse 补齐

Medium Motivation-1 的版本年龄矩阵现已完成。它复用五个已封存的相邻格子，并补齐
10个非相邻格子：v0→v2，v0/v1→v3，v0/v1/v2→v4，以及 v0/v1/v2/v3→v5。每格均为
direct long-age Reuse：指定 producer 直接物化完整 cutover 前缀，Current 读取该 K/V 后追加全部
post-cutover 事件；禁止递归串联历史 Reuse。

统一只测 D14/E14。v2/v3/v4 使用基础 manifest，v5 使用覆盖 `[287,301)` 的扩展 manifest；所有结果
显示为 E14，同时在结构化结果中保留精确日期范围与请求数。新增格只计算同一次运行内的 Current
Exact 与 Direct Reuse 两条路径，不计算 Old，也不把历史运行的 New 拼接进主比较。跨运行 Current
漂移只记录、不作为停止条件；完整报告全部预定义格子。

运行固定为 GPU0/1/2/3 四 rank、cohort32/rank、query chunk256/rank、每 rank 14 个互不重叠的物理
CPU 核。先运行最长跨度 v0→v5 的16-user/rank raw-only canary；正式队列串行、可续跑，每格 raw
先封存后才 join label，完成一格即更新三角矩阵汇总。

```bash
PYTHONPATH=src python scripts/run_yambda500m_medium_d14_direct_long_age_reuse.py \
  --mode preflight

PYTHONPATH=src python scripts/run_yambda500m_medium_d14_direct_long_age_reuse.py \
  --mode canary

PYTHONPATH=src python scripts/run_yambda500m_medium_d14_direct_long_age_reuse.py \
  --mode formal \
  --acknowledge-long-run RUN_MEDIUM_D14_DIRECT_LONG_AGE_REUSE
```

结果位于 `results/yambda500m_medium_seed17/full_reuse_matrix_v1/D14/direct_long_age_reuse_v1/`。
已完成的 `summary.json`/`summary.md` 包含10个新增非相邻格子和5个引用的相邻格子，共15格；该诊断不修改
已有 release admission、serving parent 或 cache lineage。

已完成流程小结：

> 已复用 day217 的 30k Medium population 与 mapping，完成 6L/H192/context1024 Full-only v0、
> D7×10 和 D14×5 candidate chain，再依独立合同补齐相邻及跨版本 direct Reuse。
> 这些是现有模型与论文证据，不是当前 Design 的待执行任务。
