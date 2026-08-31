# Yambda-500M Medium Full-only 训练推进方案

日期：2026-08-28  
状态：**共享 v0、D7 v1…v10、D14 v1…v5 共 16 个 checkpoint 已完成；基础 32 个 Full-only、D14 v1…v4 的 12 个 Reuse、D7 forced diagnostic 的 20 个 Reuse，以及 v5 的 E3/E7/E14_partial Full+Reuse 均已封存；PRO 未启动**

## 1. 这份方案解决什么

本方案把既有 Small 训练流程迁移到已经定义的 Medium scale，用于下一步编写通用化代码、生成合同、
组织 canary 和串行训练队列。第一阶段只回答上游问题：30k-user、6L Medium 能否形成稳定、连续的
Full-only model-update gain。此时不运行 Reuse、Design 0、PRO 或 cache compatibility 指标。

Medium 不是为了继续调 Small C32 estimator，而是一次新的 scale environment qualification。Small
已经冻结的 Insight/Design 见
`results/yambda500m_small_seed17/insight_recommendation_state_structure_v1/small_insight_design_freeze_2026-08-28.md`。

## 2. 为什么沿用 day217，而不是改到 day150

最终选择 **foundation `[0,217)`、stream `[217,300)`**，并只比较 D7 与 D14。

现有 Medium population 与 compact item mapping 都由 day217 之前的数据冻结。只把 cutoff 改成 day150
并不是简单缩短基础训练：现有 30k Medium 用户中有 3,346 人在 day150 前没有 listen history；若按原
SHA-256 规则重新选择 day150 population，只与当前 population 重合 26,654 人，即会更换约 11.2%
用户，同时必须重建 item mapping。这样会把 population/mapping 变化与模型规模变化混在一起。

沿用 day217 有三个好处：

- 直接复用已审计的 30k 固定 UID population 和 1,380,509-item foundation mapping；
- 与 Small 的 foundation boundary 一致，scale comparison 更干净；
- 83 个完整 stream days 足以支持 10 条 D7 edge 和 4 条具有完整 E14 的 D14 edge。

完整数据只到半开区间 `[0,300)`；day300 是 partial tail，正式训练和评测均排除。D14 若训练第 5 个
增量版本，其 E14 会延伸到 day301，因此不进入对称的正式 recipe matrix。

## 3. Small 流程中应继承与不应照搬的部分

### 3.1 继承的训练语义

Small 当前训练语义如下，Medium 默认保持一致，以隔离 scale 变量：

- F-only HSTU-native binary objective；所有合格真实请求均进入训练；
- 用户等权：每个窗口内按用户请求数取逆权重，再做全局归一化；
- 严格时间因果：只取 query timestamp 之前的 listen，同 timestamp 原子化，query 本身不写回 history；
- 一个 foundation pass；每个增量版本一个 pass；
- foundation 建议起点 LR `2e-4`，增量版本 LR `5e-5`，AdamW、weight decay `1e-4`；
- 每个增量版本从 direct parent 权重 warm-start，但创建全新的 AdamW state；
- seed17 为首个 Medium seed；不做 early stopping、label-driven checkpoint selection 或 per-edge 调参；
- 只保留 whole-pass final checkpoint；raw Full scores 先封存，再进行 label join 与 metric adjudication；
- 物理 GPU2/3 上一次只运行一个双 rank FSDP `FULL_SHARD` job，bf16 compute、fp32
  reduction/optimizer；GPU0/1 不参与本轮 Medium 执行。

LR 与 batch size 仍须写入新的 prospective Medium contract。若 canary 表明数值不稳定，只能在读取
formal quality 之前整体更换 recipe 并创建新合同，不能按 edge 调整。

### 3.2 不能机械复制的 Small 硬编码

当前脚本仍直接写死 Small 的 dataset、`selector_rank<=10000`、781,678 known items、4L/H128/4 heads、
context512、checkpoint status 和输出路径。Medium 实现必须把这些值从合同和 dataset manifest 读取，
复用现有数据/model primitives，不复制一套完整 pipeline。

尤其要处理 history memory：Small trainer 会为当前 rank 的用户加载完整时间域 listen，再在 collator 中
做 causal slice。Medium 30k/context1024 下应改为 window-bounded 或 partition-streamed history index，
同时通过测试证明严格 prior、同 timestamp 原子性和最大 1024 history 不变；不能靠加载 future events
后再假定内存足够。

## 4. Medium 冻结定义与建议实现值

| 项目 | Medium 值 | 当前性质 |
| --- | ---: | --- |
| population | 30,000 fixed UIDs | 已由 unified scale contract 定义 |
| foundation/stream | `[0,217)` / `[217,300)` | 沿用已有 time boundary；本方案选择 |
| known item mapping | 1,380,509 items，固定于 foundation | 已物化并有 hash |
| OOV | stable 256 buckets | 继承 Small recipe，待 Medium 合同冻结 |
| architecture | 6 layers、hidden192、context1024 | 已由 unified scale contract 定义 |
| attention heads | 6（head dim 32） | 建议实现值，待 Medium 合同冻结 |
| query schema | 4 behaviors、3 query types、query type id 2、1 query action | 继承 Small，待合同冻结 |
| seed | 17 | 首个 scale seed 建议，待合同冻结 |
| training pass | foundation 1；每个 update 1 | 继承 Small，待合同冻结 |

按 `num_items=1,380,509+256`、6L/H192/6 heads 计算，模型约有 **266,259,265** 个参数；一份只含
FP32 model weights 的 checkpoint 理论下限约 **0.992 GiB**。该数字不包含序列 activation、FSDP
通信 buffer、gradient 和 Adam state。

## 5. Full-only recipe matrix

一个共享 v0 在 `[0,217)` 上训练完成后，D7 和 D14 从同一个 v0 分成两条相互独立的 direct-parent
candidate chain。下面的请求数是 2026-08-28 使用固定 30k population、known-item join、严格先验
listen 和 `(uid,timestamp,item)` 去重得到的 **label-free planning upper bound**；正式 manifest 还会按
事前规则排除 conflicting feedback group，因此 checkpoint 中的最终 eligible count 以未来 seal 为准。

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

### 5.2 D14：4 个增量版本

每条 edge 在 cutover 后报告 E3、E7 和 E14；最后一条 v3→v4 在 day273 cutover，E14 于 day287
结束。`[287,300)` 保留，不为了凑第五条 edge 使用不完整 E14。

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

本阶段只允许用 Full-only 结果判断 release training signal 是否足够稳定，绝不读取 Reuse/PRO。recipe
scan 中的 v1…vn 只是 direct-parent candidate chain，不自动成为 serving lineage。如果某 candidate 在
后续正式 admission 中被拒绝，serving parent 与 cache lineage 保持不变；其 descendant 不能被挑出来
接到已接受 parent 上，必须按已接受 parent 重新训练。

## 6. 分阶段执行顺序

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
7. **M6：后续 scale qualification（本方案不授权）。** 在 accepted Medium edges 上只复核
   candidate-shared signed correction、AV boundary 和跨请求 persistence；随后按 Medium FLOPs 重算约
   10%/20% 两个 PRO 预算点。通过真实质量后，才做额外 seed 与 runtime。

当前 theta3 仍受 blind boundary 保护。M0 必须一次性冻结第三个及以后 candidate 的 data、release、
admission、metric 和 failure contract；在该合同和显式 launch 之前，不训练或读取任何 theta3 结果。

## 7. 初步资源预算

共享 foundation 加两条 update branch 共约 **3,240,746** 个 label-free planning request-passes，产生
1 个 v0、10 个 D7 candidate 和 4 个 D14 candidate，共 15 份 final checkpoint。按每份 0.992 GiB
纯 FP32 weights 估算，必要 checkpoint 约 14.9 GiB；执行时建议至少预留 30 GiB 临时余量，完成 seal
后不保留 optimizer state、partial checkpoint 或重复 rank shard。

当前双卡固定 batch 16/rank（global batch 32），上限请求数对应约：

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

## 8. 下一步代码清单与启动门

后续实现应按以下顺序改造，而不是立即长训：

- 将 `scripts/build_yambda500m_hstu_native_matrix_manifest.py` 的 population、窗口和 dataset 输入参数化；
- 将 `scripts/train_yambda500m_foundation_fsdp.py` 的 dataset、known vocab、模型 config、context、状态名
  与 history loading 参数化；
- 将 `scripts/evaluate_yambda500m_foundation_raw.py` 和
  `scripts/evaluate_yambda500m_release_candidates_raw.py` 中的 Small/context512/vocab 硬编码改为从
  checkpoint 与合同读取；
- 把现有 rolling runner 抽成 contract-driven runner，保留 single-job serial queue、resume audit、raw
  seal 和完整矩阵报告；
- 补充 Medium canary contract、资源记录和失败清理规则。

长训练启动前必须同时具备：prospective Medium contract、精确资源估算、focused canary PASS，以及
用户对具体 launch command 的明确授权。当前本文只允许进入代码与 canary 准备，不允许启动 v0 或
任何 release candidate。

## 9. 已实现的一键入口（2026-08-28）

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

不执行 recursive 或 long-age Reuse。最终 `summary.md/json` 并列给出三条路径的 ROC-AUC、log-loss，
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
batch size。用户显式启动全部长任务的命令为：

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
6.7–7.3 GiB，明显低于 40 GiB 门槛，未读取 canary quality。当前正式续跑入口仍为同一个 `--mode
evaluate`，runner 会保留并跳过双卡完成产物，只对剩余九个 D14 Reuse cell 使用四卡 runtime。

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
E3 `[287,290)` 与 E7 `[287,294)` 也完整。名义 E14 为 `[287,301)`，但 day300 只有 12,962 条原始
feedback row，最后时间为当日第 79,995 秒，不能冒充完整日。因此新合同把它命名为 `E14_partial`，
保留全量 partial-tail 诊断但禁止 qualification/serving admission。

独立入口 `run_yambda500m_medium_d14_v5_extension.py` 先构建 `[217,301)` 扩展 manifest，再做四卡
raw-only canary，最后串行执行 v5 training、三个 Full-only 和三个 adjacent-Reuse cell。训练保持
global batch 32（8/rank）；Full 为 batch128/rank；Reuse 沿用 cohort32/query256；四 rank 各绑定 14
个独立物理 CPU 核。canary 在最长 `E14_partial` 上同时覆盖 Full 与 Reuse，不读取质量。

```bash
PYTHONPATH=src python scripts/run_yambda500m_medium_d14_v5_extension.py --mode prepare
PYTHONPATH=src python scripts/run_yambda500m_medium_d14_v5_extension.py --mode canary
PYTHONPATH=src python scripts/run_yambda500m_medium_d14_v5_extension.py \
  --mode formal --acknowledge-long-run RUN_MEDIUM_D14_V5_EXTENSION
```

结果位于 `results/yambda500m_medium_seed17/full_reuse_matrix_v1/D14/v5_extension_v1/`，不会修改原
D14 v1…v4 checkpoint、admission、raw seal 或 summary。

一句话执行口径：

> **复用 day217 的 30k Medium population 与 mapping，先训练一份 6L/H192/context1024 Full-only v0，
> 再从同一 v0 串行扫描 D7×10 和 D14×4 candidate chain；全程先验证模型更新本身，冻结稳定 recipe
> 后才解锁 Reuse/PRO，且任何长训都需新合同、canary 和用户再次明确授权。**
