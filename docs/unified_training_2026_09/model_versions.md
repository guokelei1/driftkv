# Yambda 三规模模型版本

本页记录当前选用的 seed17、context1024 模型端点。Medium 和 Large 使用 Yambda-500M，
分别为 6L/H192、30,000 用户和 10L/H320、79,681 用户；Max 使用 Yambda-5B，
为 16L/H320、200,000 用户。天数均为左闭右开的训练区间；AUC 提升在该版训练后的
同一个 E14 窗口比较所列相邻版本，计算式为 `当前 AUC / 上版 AUC - 1`。
表中 SHA256 为前 12 位，完整哈希见[Medium 清单](../../results/unified_training_2026_09/medium/seed17/checkpoints/chain.manifest.json)、
[Large 清单](../../results/unified_training_2026_09/large/seed17/checkpoints/chain.v0_v5.manifest.json)
及 Max 各端点的封存文件。三个规模的选定权重均集中在各自的
`seed17/checkpoints/v0..v5/` 目录；训练过程元数据仍可在原运行记录中查阅。

## Medium

| 版本 | 训练天数 | Epoch / 全局 batch | 选用权重（SHA256 前缀） | 相对上版 AUC |
| --- | --- | --- | --- | ---: |
| V0 | [0,217) | 1 / 32 | [4a51957fe7c7](../../results/unified_training_2026_09/medium/seed17/checkpoints/v0/checkpoint_100.pt) | 起点 |
| V1 | [217,231) | 1 / 32 | [72fdc2011f5e](../../results/unified_training_2026_09/medium/seed17/checkpoints/v1/checkpoint_100.pt) | +3.726%¹ |
| V2 | [231,245) | 2 / 32 | [8d4fa1ddad3d](../../results/unified_training_2026_09/medium/seed17/checkpoints/v2/checkpoint_100.pt) | +1.138% |
| V3 | [245,259) | 1 / 32 | [5581d722ae22](../../results/unified_training_2026_09/medium/seed17/checkpoints/v3/checkpoint_100.pt) | +2.369% |
| V4 | [259,273) | 1 / 32 | [c1a41440ce46](../../results/unified_training_2026_09/medium/seed17/checkpoints/v4/checkpoint_100.pt) | +3.095% |
| V5 | [273,287) | 1 / 32 | [c82fba0c9a95](../../results/unified_training_2026_09/medium/seed17/checkpoints/v5/checkpoint_100.pt) | +3.283% |

¹ V0→V1 为复用的历史 D14 E14 结果；V2–V5 是本轮逐版评测。
[历史 V0→V1 指标](../../results/yambda500m_medium_seed17/full_reuse_matrix_v1/summary.json)，
[本轮 V2–V5 指标](../../results/unified_training_2026_09/medium/README.md)。

## Large

| 版本 | 训练天数 | Epoch / 全局 batch | 选用权重（SHA256 前缀） | 相对上版 AUC |
| --- | --- | --- | --- | ---: |
| V0 | [0,217) | 1 / 96 | [e3fb0f569399](../../results/unified_training_2026_09/large/seed17/checkpoints/v0/checkpoint_100.pt) | 起点 |
| V1 | [217,231) | 1 / 96 | [4918c9425b57](../../results/unified_training_2026_09/large/seed17/checkpoints/v1/checkpoint_100.pt) | +3.396% |
| V2 | [231,245) | 1 / 96 | [7f1b9a31ca80](../../results/unified_training_2026_09/large/seed17/checkpoints/v2/checkpoint_100.pt) | +2.072% |
| V3 | [245,259) | 1 / 96 | [2cd79ab15739](../../results/unified_training_2026_09/large/seed17/checkpoints/v3/checkpoint_100.pt) | +2.750% |
| V4 | [259,273) | 2 / 64 | [8df05079a7b8](../../results/unified_training_2026_09/large/seed17/checkpoints/v4/checkpoint_100.pt) | +1.372% |
| V5 | [273,287) | 1 / 64 | [34c37ebe9da8](../../results/unified_training_2026_09/large/seed17/checkpoints/v5/checkpoint_100.pt) | +1.378%² |

² V5 第 1 轮是用户选定端点；AUC 提升为正，但原四项准入未全部通过。
[Large 逐版指标及备选端点](../../results/unified_training_2026_09/large/README.md)。

## Max

| 版本 | 训练天数 | Epoch / 全局 batch | 选用权重（SHA256 前缀） | 相对上版 AUC |
| --- | --- | --- | --- | ---: |
| V0 | [0,217) | 1 / 80 | [83edf405eaf3](../../results/unified_training_2026_09/max/seed17/checkpoints/v0/checkpoint_100.pt) | 起点 |
| V1 | [217,231) | 1 / 80 | [d36b7368732f](../../results/unified_training_2026_09/max/seed17/checkpoints/v1/checkpoint_100.pt) | +5.417% |
| V2 | [231,245) | 1 / 80 | [9cce41b4d150](../../results/unified_training_2026_09/max/seed17/checkpoints/v2/checkpoint_100.pt) | +1.246% |
| V3 | [245,259) | 2 / 80 | [5a696d2e8e23](../../results/unified_training_2026_09/max/seed17/checkpoints/v3/checkpoint_100.pt) | +1.798%³ |
| V4 | [259,273) | 2 / 80 | [e1b15e8a5b92](../../results/unified_training_2026_09/max/seed17/checkpoints/v4/checkpoint_100.pt) | +4.217% |
| V5 | [273,287) | 2 / 80 | [fc10cfd1ba68](../../results/unified_training_2026_09/max/seed17/checkpoints/v5/checkpoint_100.pt) | +2.173%⁴ |

³ V2/V3 选用端点的同窗口 AUC 比较，见[选定清单](../../results/unified_training_2026_09/max/seed17/selected_v2_v3_checkpoints.json)。
⁴ 用户已选定通过原四项准入的 V5 第 2 轮端点；第 1 轮 AUC 相对 V4 为 −0.150%。
[Max V5 两轮结果与封存](../../results/unified_training_2026_09/max/seed17/v5_epochs12_4gpu_b80_cpu14/README.md)。
