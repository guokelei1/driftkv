# 本轮准备资产清单

Read-only preparation inventory, not training, quality evaluation, or checkpoint integrity verification.

状态：`ready_for_preparation`。只读元数据与文件大小，未加载模型、未使用 GPU；已记录哈希未重新验证。

当前候选：`A_itemgate6_rms_roll4`；fit/val=64/16，local/joint=8/4，读取共享/私有=3/3，续写=4，门控=item。

| 面板 | evaluation | 保留 fit / validation | 角色文件 |
|---|---:|---:|---|
| development | 4096 | 256 / 16 | `/home/gkl/work/evokv/results/design_system_2026_10/phase2/representative4096/panel/users.json` |
| calibration | 4096 | 256 / 16 | `/home/gkl/work/evokv/results/design_two_2026_10/benchmark_panel/users.json` |
| revealed_layer_confirmation | 1024 | 0 / 0 | `/home/gkl/work/evokv/results/query_relay_2026_10/confirmation/panel/users.json` |
| revealed_item_confirmation | 1024 | 0 / 0 | `/home/gkl/work/evokv/results/query_relay_2026_10/confirmation_item/panel/users.json` |

两个确认组均已揭晓；不能用于新候选调参后再次宣称独立确认。面板保留角色数不等于当前实际训练人数。

| 基础模型 | 存在 / 已登记 | 权重字节（仅 stat） |
|---|---:|---:|
| medium | 6 / 6 | 6,390,330,716 |
| large | 6 / 6 | 17,214,409,636 |
| max | 6 / 6 | 27,317,129,142 |
| recflow | 6 / 6 | 14,146,071,720 |

已完成证据不列入重跑队列：
- item_development4096: `/home/gkl/work/evokv/results/query_relay_2026_10/probes/A_itemgate6_rms_roll4/population4096/merged_actual/summary.json`（存在：True）
- item_feedback4096: `/home/gkl/work/evokv/results/query_relay_2026_10/probes/A_itemgate6_rms_roll4/feedback4096_actual/merged/summary.json`（存在：True）
- item_confirmation1024: `/home/gkl/work/evokv/results/query_relay_2026_10/confirmation_item/behavior1024/summary.json`（存在：True）
- layer_confirmation1024: `/home/gkl/work/evokv/results/query_relay_2026_10/confirmation/behavior1024/summary.json`（存在：True）
- static_writer_development4096: `/home/gkl/work/evokv/results/query_relay_2026_10/comparisons/item_vs_constant4096/summary.json`（存在：True）
- static_writer_confirmation1024: `/home/gkl/work/evokv/results/query_relay_2026_10/comparisons/item_vs_constant_confirmation1024/summary.json`（存在：True）
- item_code_content128: `/home/gkl/work/evokv/results/query_relay_2026_10/probes/A_itemgate6_rms_roll4/code_use128_v3/summary.json`（存在：True）
- optimized_engineering_recheck1024: `/home/gkl/work/evokv/results/implementation_2026_10/population_confirmation1024/merged/summary.json`（存在：True）
- engineered_cost: `/home/gkl/work/evokv/results/implementation_2026_10/cost/summary.json`（存在：True）
- recflow_foundation_chain: `/home/gkl/work/evokv/results/recflow/development/window_6l_expanded_u4096_seed17/chain_summary.json`（存在：True）

待适配：
- medium: Current reference and H6/D32 optimized paths exist; protocol/shape changes need matched controls.
- large_max: Foundation endpoints exist. Prepare scale-specific panels and origin calibration; current H6 optimized reader is not a ready-made backend for H10 models.
- recflow: Foundation A-F endpoints exist; no relay cache result yet. SiLU/reference block/relative bias and three-token joint scoring require adapter and teacher integration.
- service_replay: Current integration is synchronous per-user numerical replay; arrival/queue/resident-capacity service measurements remain to implement.
- budgeted_rebuild: Native rebuild primitives exist; persistent budgeted selection under the same event stream remains to implement.

必需资产缺失：0。历史清理后的 raw 路径不是恢复任务。
