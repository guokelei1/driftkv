# EvoKV

本仓库研究流式更新推荐模型中，旧版本 HSTU 前缀 K/V cache 的复用精度与全量重算成本之间的空间。系统任一时刻只服务一个当前模型；矩阵中的旧版本编号仅表示 cache 的来源版本，不表示同时在线多个模型。

## 当前状态

2026-08-11 已选定第一轮 KuaiRand 大模型开发基线：

- 连续模型版本为 θ1–θ8，θ0 仅作训练启动点；
- HSTU 为 8 层、H512、8 heads，video/author 双字段；
- 每个大模型参数量为 47,960,055,552 bytes（44.666 GiB），单张 A40 无法容纳；
- 使用 1 个真实正样本与 99 个冻结负样本评测；
- NDCG@5 的 28 个 Recompute-over-Reuse 格子中 26 个为正，相邻 7/7 为正；相邻平均为 `+6.007%`，全部格平均为 `+8.852%`；
- 结果没有人工修改 K/V、没有指标倍乘，也没有删掉负格；目前仅是单 seed 开发证据，不是论文正式结果。

完整矩阵见 [当前结果](results/root_cause_campaign/kuairand_latest_query_large_capacity_lift_theta1_theta8_20260811_v37/theta1_theta8_matrix.md)。

## 唯一入口

验证当前产物：

```bash
scripts/run_evokv_kuairand_large_baseline_rebuild.sh verify
```

安全续跑缺失后缀：

```bash
scripts/run_evokv_kuairand_large_baseline_rebuild.sh resume
```

从原始 KuaiRand 文件全新重建的命令、空间要求和删除边界见 [基线复现说明](docs/BASELINE_REPRODUCTION.md)。

## 文档

- [文档索引](docs/README.md)
- [当前研究状态与路线](docs/08_core_insights_and_roadmap.md)
- [有效评测协议](docs/eval_protocol.md)
- [基线复现说明](docs/BASELINE_REPRODUCTION.md)

旧探索文档已经删除。历史脚本和历史结果目录不构成当前事实来源；只有上述文档、冻结 registry 和验证脚本共同定义当前基线。
