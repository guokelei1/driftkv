# Large D14 canonical V0–V5 chain

状态：**当前唯一工作序列**。本目录不复制约 2.67 GiB/checkpoint 的模型文件；`chain.json` 和
canonical contract 提供唯一的路径、hash、训练窗口、epoch 与父子关系。

| Version | 训练窗口 | Epoch | 当前 checkpoint |
| --- | --- | ---: | --- |
| V0 | `[0,217)` | 1.0 | `qualification_v1/shared_v0/checkpoint_100.pt` |
| V1 | `[217,231)` | 1.0 | `qualification_v1/D14/checkpoints/v1/checkpoint_100.pt` |
| V2 | `[231,245)` | 1.0 | `qualification_v1/D14/checkpoints/v2/checkpoint_100.pt` |
| V3 | `[245,259)` | 1.0 | `qualification_v1/D14/checkpoints/v3/checkpoint_100.pt` |
| V4 | `[259,273)` | **2.0** | `v3_v4_epoch_sweep_v1/checkpoints/checkpoint_epoch_2.pt` |
| V5 | `[273,287)` | **2.0** | `v4e2_to_v5_epoch_sweep_v1/checkpoints/checkpoint_epoch_2.pt` |

当前 development release rule 是 aggregate ROC-AUC 必须相对直接 parent 为正；其他指标仍完整报告，
但不再用旧四门否定本工作序列。五条 canonical edge 的 AUC 相对变化依次为
`+3.396% / +2.072% / +2.750% / +0.855% / +4.826%`。

该序列是在读取 endpoint quality 后整理出的 post-hoc working lineage，不是独立 qualification。
未来若复用两 epoch recipe，必须在读取新 edge quality 前冻结，不能继续在同一 qualification labels 上
“训练到变正”为止。

0.5/1.0/1.5 endpoint、原始 V4@1.0 与 legacy V5 不属于当前序列；它们的 summary、raw seal、
adjudication、hash 和负结果仍作为历史开发证据保留。
