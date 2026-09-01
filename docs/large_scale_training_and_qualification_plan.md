# Yambda-500M Large 训练与验证执行记录

日期：2026-08-30  
状态：**Large D14 当前工作序列已冻结为原 V0–V3、V4@2.0、direct-child V5@2.0；五条相邻边 aggregate AUC 全正**

## 1. 冻结结论

Large 主点为 `10L / H320 / 10 heads / C1024`，seed 17，F-only，已知 catalog
2,224,809，加 256 个 OOV buckets。它是一个约 **717.26M total parameters** 的
embedding-dominated recommender，其中 contextual blocks 约 **5.12M parameters**。按全人口、满
context、BF16 K/V 计算，persistent state 上限约 **972.7 GiB**。

本轮不运行 `12L/H320`，也不根据 Large AUC 在架构间选择。只有主点出现预注册的 OOM、非有限训练、
显存/吞吐、checkpoint 或 state-I/O 物理失败时，才依次考虑 `8L/H320`、`8L/H256`。focused canary
已经通过，因此当前没有触发 fallback。

## 2. 数据与 release matrix

Large population 是全部 79,681 名 lineage-eligible 用户。冻结 manifest 的真实计数为：

| Block | 请求 | 已知 target 请求 | 已知用户 |
| --- | ---: | ---: | ---: |
| Foundation `[0,217)` | 5,324,569 | 5,301,016 | 62,035 |
| Matrix `[217,301)` | 2,506,473 | 2,171,324 | 58,675 |

去重/冲突审计共移除 68,098 条完全重复行，并排除 12,366 个冲突请求。准备阶段没有计算 label metric。

版本矩阵固定如下：

| Branch | 训练增量 | Update/edge | 评测窗口 | 完整性 |
| --- | ---: | ---: | --- | --- |
| D7 | 7 天 | 10 | E7 | 10 条均完整 |
| D14 | 14 天 | 5 | E7、E14 | 5 条 E7 完整；前 4 条 E14 完整 |
| D14 v4→v5 | 14 天 | 第 5 条 | E14 | `[287,301)`；按统一 E14 口径报告，并保留实际请求数 |

shared v0 只训练一次；D7 和 D14 分别从 v0 开始沿 direct-parent candidate chain 训练。每个 checkpoint
只跑一个完整 pass，fresh AdamW，foundation LR `2e-4`，update LR `5e-5`。总计保留 16 个正式
checkpoint：shared v0、D7 v1–v10、D14 v1–v5。

## 3. 评测协议

原合同要求全部 20 个 Full-only cell 先完成 raw seal、label join 和独立 admission seal，之后才开始任何
Reuse/PRO label join。实际停止时已裁决 18/20：D7/E7 十格与 D14 前四条 edge 的 E7/E14；D14
v4→v5 E7 只留下 sealed raw，E14 未运行。模型 admission 与 cache compatibility 分开报告；即使某条 candidate edge 未过
严格模型门，也只在 admission seal 之后运行预冻结的 adjacent diagnostic，且不改变 serving parent 或
cache lineage。

- Full-only 保持原矩阵：D7 的 10 edges × E7，以及 D14 的 5 edges × E7/E14；
- 正式 Reuse/PRO 不再运行；此前缩减为 D14/E14 五格的计划已在首个正式 Reuse cell 前取消；
- D14 第五条的 14-day cell 与其余版本统一命名为 `E14`；实际日期范围与请求数仍完整记录；
- 不运行 D3/E3，不运行 recursive/cross-version Reuse；
- 所有 raw score 都在 label join 前封存，所有预冻结 cell 全量报告。

Large PRO 固定为 `repair_width=256, carriers=64, represented_mass=4`，即与 C1024 保持 1/4 repair
和 1/16 carrier 比例；理论计算为 Exact-All 的 **8.919%**，sidecar 为每用户 3,200 个 FP32 scalar
（12,800 bytes）。它只用于 D14，不从 Large quality 调 carrier、probe、scale 或 estimator。

## 4. focused canary 实测

四张 NVIDIA A40 和 56 个互斥物理 CPU core 已完成真实 10L/H320 canary。选择只读取显存、吞吐、
利用率、checkpoint 和 I/O，不读取 AUC、loss 等质量指标。

| 项目 | 选择/结果 |
| --- | --- |
| 训练 | global batch 96，24/rank；146.07 request/s |
| 训练 GPU | active utilization 76.88%；v0 peak reserved 18.28 GiB/rank |
| Parent restore + D14 v1 | 20 steps 通过；peak reserved 30.41 GiB/rank |
| Full | batch 64/rank；active utilization 56.0%；peak reserved 16.38 GiB/rank |
| Reuse + C64 PRO | cohort 12/rank，query chunk 128 |
| 延长 Reuse 计算段 | 1,259 requests；active utilization 53.79%；peak reserved 13.38 GiB/rank |
| Full state I/O | 32 users、Parent+Current、10×1024×320 K/V；checksum 全通过 |
| NVMe | 800 MiB 样本；聚合 write/read 约 8.28/7.72 GiB/s |

状态 I/O canary 的大 tensor 只用于物理校验，checksum、shape、bytes 和时延封存后已删除冗余样本；
checkpoint、raw seal、运行日志和摘要保留。短 Reuse sizing 段的 1 秒采样不足 50%，因此在不改变
cohort/query 的前提下延长到 48 users/rank 复核，最终通过原定 50% 门，没有降低门槛。

正式 v0 首次全人口装载还暴露并修复了一处 Small canary 无法覆盖的 CPU 复杂度问题：历史列已经按
UID 排序，旧 `FoundationHistoryIndex.from_columns` 却为每名用户重新扫描完整事件列。当前实现按
连续 UID 边界线性切片，保持同一时间/item 次序和 prefix 语义；专项等价测试与全仓测试通过。真实
Large v0 从启动到 GPU compute 约 9–10 分钟，进入计算后四卡瞬时利用率 97–100%。

## 5. 执行与恢复

唯一入口是：

```bash
# 只读状态
python scripts/run_yambda500m_large_qualification.py --mode status

# CPU manifest（已完成，可哈希校验后跳过）
python scripts/run_yambda500m_large_qualification.py --mode prepare --threads 56

# label-free focused canary（已完成，可校验摘要后跳过）
python scripts/run_yambda500m_large_qualification.py --mode resource-canary

# 正式长队列
python scripts/run_yambda500m_large_qualification.py \
  --mode formal \
  --acknowledge-long-run RUN_LARGE_D7_D14_10L_H320
```

正式 runner 每次只运行一个四 rank job，当前按 `shared v0 → D7 checkpoints → D14 checkpoints → all
Full/admission → stop` 排队。完整 artifact 会核对 seal/hash 后跳过；存在不完整目录时停止审计，
不会覆盖或悄悄续写。结构化进度位于
`results/yambda500m_large_seed17/qualification_v1/pipeline_state.json`，最终汇总写入同目录的
`summary.json` 和 `summary.md`。

每个训练 checkpoint 另有原子更新的 `progress.json`；正式 Large 每 100 step 记录完成比例、近期/
累计 rank0 loss、中位 step time、剩余 ETA 和 peak reserved memory，避免长训练只能靠进程存活猜测进度。

## 6. 证据边界

资源 canary 只证明主架构、四卡 FSDP、checkpoint save/restore、Full、Reuse+C64 PRO 与小段真实状态
I/O 可执行；它不是质量证据。正式结果的训练 seed 才是 repeat unit，Large 是冻结机制的 prospective
scale qualification，不是新的 PRO development set，也不授权 serving promotion。

权威合同：

- `configs/contracts/yambda500m_large_hstu_native_d7_d14_full_reuse_pro_v1.yaml`；
- `configs/contracts/yambda500m_large_hstu_native_d7_d14_execution_v1.yaml`；
- `configs/contracts/yambda500m_large_reuse_scope_d14_e14_only_v1.yaml`（2026-08-31 Reuse 范围缩减修订）。
- `configs/contracts/yambda500m_large_full_only_stop_v1.yaml`（当前有效：Full-only 后结束，正式 Reuse/PRO 为零格）。

## 7. 已完成 Full-only 结果

以下均为 Current 相对 Parent 的比例变化；loss reduction 为正表示更好。D7/E7 十条中 8 条 AUC
为正，严格四门通过 5 条：

| Edge | AUC 相对变化 | loss reduction | 严格门 |
| --- | ---: | ---: | --- |
| v0→v1 | +5.272% | +2.164% | PASS |
| v1→v2 | +0.460% | -0.020% | FAIL |
| v2→v3 | +0.909% | +0.540% | PASS |
| v3→v4 | +0.964% | +0.291% | FAIL |
| v4→v5 | +1.280% | +0.240% | FAIL |
| v5→v6 | +1.468% | +1.714% | PASS |
| v6→v7 | -1.718% | -1.839% | FAIL |
| v7→v8 | +1.425% | +1.423% | PASS |
| v8→v9 | +1.059% | +0.452% | PASS |
| v9→v10 | -0.172% | -0.237% | FAIL |

D14 前三条在 E7/E14 都稳定为正，v3→v4 在两个 horizon 都退化：

| Edge | E7 AUC | E7 loss reduction | E14 AUC | E14 loss reduction | 严格门 |
| --- | ---: | ---: | ---: | ---: | --- |
| v0→v1 | +3.761% | +1.554% | +3.396% | +1.656% | PASS |
| v1→v2 | +2.281% | +1.050% | +2.072% | +1.016% | PASS |
| v2→v3 | +2.896% | +1.412% | +2.750% | +1.392% | PASS |
| v3→v4 | -0.455% | -0.050% | -0.576% | -0.123% | FAIL |

因此 Large 证明了 release training 大多数时候有效，但不是一条全正的稳定 release chain；D14
v3→v4 是真实的模型 endpoint/admission 问题，不应归因于 Reuse 或 PRO。

## 8. D14 v3→v4 endpoint-strength 专项

完整 E14 的一轮训练 v3→v4 已观察到 AUC 和 log-loss 同时退化。该结果出现后才定义的
`yambda500m_large_v3_v4_epoch_sweep_v1.yaml` 是 post-hoc development，不能冒充独立 Large
qualification。它从原封存 v3 出发，在同一 `[259,273)` 数据和同一条连续 AdamW 轨迹上，于
0.5/1.0/1.5/2.0 epoch 保存四个 checkpoint；随后在完整 E14 `[273,287)` 上一次性报告 parent 与
全部四个 Current。禁止 Reuse、PRO、early stopping 和按结果隐藏 endpoint。

入口为 `scripts/run_yambda500m_large_v3_v4_epoch_sweep.py`。focused canary 只验证 20-step 训练、
四卡 checkpoint 和五模型联合 Full 推理的正确性/显存，不读取质量；正式任务必须在 canary 通过后由
用户使用独立 acknowledgement 启动。

## 9. 当前 canonical D14 V0–V5

后续代码与讨论只使用
`configs/contracts/yambda500m_large_d14_canonical_v0_v5_v1.yaml` 和
`results/yambda500m_large_seed17/canonical_D14_v0_v5_v1/chain.json` 解析当前序列：

| Version | 训练窗口 | Epoch | 相对直接 parent AUC |
| --- | --- | ---: | ---: |
| V0 | `[0,217)` | 1.0 | — |
| V1 | `[217,231)` | 1.0 | +3.396% |
| V2 | `[231,245)` | 1.0 | +2.072% |
| V3 | `[245,259)` | 1.0 | +2.750% |
| V4 | `[259,273)` | **2.0** | +0.855% |
| V5 | `[273,287)` | **2.0** | +4.826%（E14） |

V4/V5 的 parent checkpoint hash 已分别核对为 V3 和 canonical V4，六个版本形成一条闭合的直接父子
lineage。当前 development release rule 按用户决定采用 aggregate ROC-AUC 严格为正；loss、Brier 和
user-cluster bootstrap 继续完整报告，但不作为这条工作序列的否决门。

这个 rule 只解释已经完成的 post-hoc working lineage。未来新 edge 若要形成独立 qualification，必须在
读取其质量前冻结两 epoch recipe，不能对同一 qualification labels 反复训练到 AUC 变正。旧 0.5/1.0/
1.5 endpoint、原 V4@1.0 与 legacy V5 已从当前入口排除，其封存结果和负证据不删除、不改写。
