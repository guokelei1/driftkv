# HSTU KV Cache Version Migration

本仓库研究持续训练产生的 HSTU 模型版本变化：

```text
theta_old + old prefix cache -> theta_current + version-stale K/V
```

完整历史重算能够得到 current-model K/V，但成本高。当前方法利用缓存的旧
`Norm(x)`，为同一个 source/target version cohort 编译共享迁移程序，以更低成本生成接近
current-model 的 K/V。

权威研究状态见
[docs/08_core_insights_and_roadmap.md](docs/08_core_insights_and_roadmap.md)，实验可比性见
[docs/eval_protocol.md](docs/eval_protocol.md)，完整文档索引见
[docs/README.md](docs/README.md)。

## 当前三层设计

1. **Cohort migration compiler**
   - 以 old/current model pair 和隔离的无标签样本生成、认证并发布共享 migration program；
   - 当前 fast path 将 `fresh - cheap` K/V residual 编译进旧 `Norm(x)` 到目标 K/V 的一个
     affine projection；
   - version 用于编译、认证和合批，不用于预测某个用户或版本能否安全 reuse。
2. **Capsule-to-K/V operator**
   - fused affine、bias、valid-length handling 和 K/V split；
   - 支持连续或 jagged 输入以及目标 K/V direct write；
   - cohort page compaction 在当前长序列 KuaiRand trace 上只有约 1% host-boundary 改善，
     因此保留为条件式布局机制，不作为已成立的主要贡献。
3. **Destination-oriented out-of-core engine**
   - host-staged backend 以有界 transform/publication wave 执行 K/V 更新；
   - 支持单/多 GPU、host-staged publication 和 target-GPU HBM direct publication；
   - 通过统一 transaction 将结果发布到 HBM、DRAM、POSIX filesystem 或 remote-object
     backend，只有完整 record coverage 后才提交 target-version manifest。

薄层 Update Coordinator 只负责解析 job spec，并把已发布 program、capsule shards、devices
和 destination 交给上述三层。它是系统串联入口，不是第四项贡献，也不负责 reuse 判断、
训练、在线请求调度或自动 destination placement。

当前系统不依赖合成的用户请求 arrival、热度、routing 或训练/推理共置假设。训练只负责提供
模型版本；更新引擎接收固定 cache/capsule cohort 和显式 destination。filesystem 与 remote
目前是接口和正确性实现，不是 SSD 或网络性能结果。当前输入 capsules 仍由调用方整体准备；
bounded waves 尚不等于 source-side 全量状态已经流式化。

## 当前证据边界

- 大规模 KuaiRand 主设置为 16 层、hidden/K/V width 512、最大长度 2,048，模型约
  0.181B 参数。
- 4+12 训练和 66 个 stale-cache/current-model pair 的 motivation 已完成；结果支持
  fixed reuse window 不是稳定质量规则，但不支持“每隔固定版本必然出现 cliff”。
- verified compiler 在独立 fit/selection/certificate/final 用户角色下发布
  theta0/theta4/theta10 到 theta11 的 compiled full-affine programs。当前结果仍是 adaptive
  seed-0 evidence，需要冻结后复现。
- 两卡 real-capsule v2 在相同 pinned-host 输入/输出边界下达到 1.951x 的 1→2 GPU scaling，
  并相对 independently pipelined BF16 full recomputation 快 11.22x。
- v3 HBM/host endpoint 结果证明 destination 会改变数据移动边界，但不同 endpoint 的时间不能
  被解释成算子 speedup。
- 冻结 mixed-version trace 的四卡补测中，host-staged migration、direct-HBM migration 和
  host-staged BF16 exact 分别达到 3.275x、3.331x 和 3.592x 的 1→4 scaling；四卡共同
  host boundary 上 compiled migration 仍比 BF16 exact 快 10.39x。
- destination-v4 已完成代码与事务正确性闭环，尚未完成 source-side streaming、全 cohort
  HBM/DRAM 同 endpoint 性能结果以及物理 SSD/remote 实验。

## 代码布局

```text
src/hstu_kvcache/
  data/          KuaiRand、MovieLens 与 ordered-exposure 数据
  models/        模块化 HSTU 与一等 K/V 输出
  streaming/     leak-free next-item 训练与模型版本工具
  migration/     compiler、算子、多 GPU runtime 与 destination backends
scripts/
  train_kuairand_long_context.py
  evaluate_kuairand_long_context_motivation.py
  evaluate_kuairand_long_context_sync_design.py
  benchmark_kuairand_two_gpu_migration_system.py
  benchmark_kuairand_cohort_jagged_system.py
  benchmark_kuairand_four_gpu_scaling_system.py
  run_streamkv_update_coordinator.py
  validate_streamkv_destination_runtime.py
experiments/
  motivation/
  migration/
  system/
```

## 开发与轻量验证

```bash
pip install -e .
pytest
ruff check src tests scripts

python scripts/validate_streamkv_destination_runtime.py --destination dram
python scripts/validate_streamkv_destination_runtime.py \
  --destination filesystem --root /path/on/a/filesystem
python scripts/validate_streamkv_destination_runtime.py --destination remote
CUDA_VISIBLE_DEVICES=0 python scripts/validate_streamkv_destination_runtime.py \
  --destination hbm --devices cuda:0

python scripts/run_streamkv_update_coordinator.py --print-template
```

这些 destination validation 命令只运行小张量正确性路径，不训练模型，也不是性能实验。新
实验必须使用独立 protocol，并保证 compiled migration 与 full recomputation 具有相同的
source、destination、dtype、layout、durability 和 manifest timing boundary。
Coordinator 的 `--print-template` 和默认 plan-only 输出也只是架构接口，不是实验产物。
