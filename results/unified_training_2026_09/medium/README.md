# Medium：固定 V0–V5 模型链

2026-09-15 按用户要求固定。统一入口为 `seed17/checkpoints/v0` 至 `v5`；[机器可读清单](seed17/checkpoints/chain.manifest.json)记录完整 SHA-256、父模型哈希、原始来源和评测证据。

6L/H192、6 heads、context1024、30,000用户、seed17，Yambda-500M。V0/V1复用历史D14起始模型；V2–V5使用本轮最终达标模型。

| 版本 | Parent | 训练窗口（天） | Epoch | 全局 batch | GPU 数 | Checkpoint |
| --- | --- | --- | ---: | ---: | ---: | --- |
| V0 | — | [0,217) | 1 | 32 | 2 | [权重](seed17/checkpoints/v0/checkpoint_100.pt) |
| V1 | v0 | [217,231) | 1 | 32 | 2 | [权重](seed17/checkpoints/v1/checkpoint_100.pt) |
| V2 | v1 | [231,245) | 2 | 32 | 4 | [权重](seed17/checkpoints/v2/checkpoint_100.pt) |
| V3 | v2 | [245,259) | 1 | 32 | 4 | [权重](seed17/checkpoints/v3/checkpoint_100.pt) |
| V4 | v3 | [259,273) | 1 | 32 | 4 | [权重](seed17/checkpoints/v4/checkpoint_100.pt) |
| V5 | v4 | [273,287) | 1 | 32 | 4 | [权重](seed17/checkpoints/v5/checkpoint_100.pt) |

## E14 父子质量

各行在该行的同一未来窗口比较父子模型；不同窗口的绝对AUC不能直接比较。下表只写相对提升。

| 版本边 | E14 窗口 | Parent AUC | Current AUC | 相对提升 |
| --- | --- | ---: | ---: | ---: |
| [v1 → v2](seed17/v2_2epoch_4gpu_b32_cpu14/README.md) | [245,259) | 0.639586 | 0.646865 | +1.138% |
| [v2 → v3](seed17/v3_1epoch_4gpu_b32_cpu14/README.md) | [259,273) | 0.643808 | 0.659057 | +2.369% |
| [v3 → v4](seed17/v4_1epoch_4gpu_b32_cpu14/README.md) | [273,287) | 0.640726 | 0.660556 | +3.095% |
| [v4 → v5](seed17/v5_1epoch_4gpu_b32_cpu14/README.md) | [287,301) | 0.660402 | 0.682083 | +3.283% |

四条新训练边均通过原四项准入指标和相对AUC >1%目标。V0/V1为历史复用，此处不声称重新评测了V0→V1。

## 训练与文件约定

- V2–V5：4卡，每卡8/global32，CPU history14/Arrow14/IO4/Torch4、独立绑核；fresh AdamW、LR=5e-5、weight decay=1e-4、用户等权BCE。V2为2 epochs，其余为1 epoch。
- 主指标为 pooled request ROC-AUC，真实like/dislike反馈，仅Full-only E14；保留原始分数、封存、adjudication、最终训练日志和配置。
- V2–V5实体checkpoint已移入统一目录；原运行目录的`checkpoint`为相对符号链接，原合同和payload内路径仍有效。原seal内容不变。
- 本轮被替代的任务、失败记录和探针payload按用户要求删除，无本地备份；最终运行的轻量canary通过证明和预算随配置保留，其引用的探针payload已清理。
- 本链为单seed开发选择结果，不构成独立最终评价或已验证的缓存适配结果；尚未执行Reuse。固定模型链不修改历史服务推广标记。
- 其他规模及历史模型原件未改动。
