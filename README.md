# HSTU KV Cache Version Migration

本仓库研究一个明确的模型版本问题：HSTU 在流式训练后从 `theta_old` 更新到
`theta_current`，由旧参数生成的历史 prefix K/V 不再与当前模型一致。全量重算能够恢复
fresh 结果，但需要重新执行完整历史前向；本项目尝试利用 HSTU 内部结构，以更低成本直接
迁移旧缓存。

当前状态是一个经过协议修复、完成逐轴与数据/模型组合规模验证的 proof of concept，已经形成
`motivation -> structural observation -> minimal method -> preliminary evaluation` 闭环，尚不是
完整论文。文档入口见 [docs/README.md](docs/README.md)，唯一研究路线图见
[docs/08_core_insights_and_roadmap.md](docs/08_core_insights_and_roadmap.md)。

## 当前结论

- 单次模型更新造成的平均 stale-cache 损失较小，多版本累计后损失明显增大；四个训练种子下，
  fresh 相对 stale 的 Best Rank 平均改善从 one-step 的 `4.15` 增至累计场景的 `63.39`。
- 六层控制实验中，流式训练的 full compute 在 `theta-5` 相对 frozen 改善 `484.34` Best
  Rank；full reuse 保留其中约 82.4%，cache maintenance 恢复剩余 `85.32` Rank 和约 30.8%
  的 NDCG@100 流式训练收益。
- 当前方法把迁移拆成两种计算：cheap 路径用当前 `Wk/Wv` 投影缓存的旧 `Norm(x)`；full
  路径从缓存的 split hidden 开始，执行当前模型的连续深层 block。
- 去掉 suffix 末层无法影响 K/V 的 attention/gate/residual 后，六层小模型中 cheap-all 的
  实测 GPU 时间约为优化后 full recompute 的 `0.187x`，suffix-5 约为 `0.767x`。21 个连续
  区间的 discovery 与 held-out 验证没有显示任意非 suffix 区间可稳定优于最深 suffix。
- 固定算子后，成本—质量曲线在 KuaiRand 的长度 32/64/128 和 3/6/9 层模型上均保留。形状
  基准中，序列长度从 128 增至 512 时 cheap/full 从 `0.189` 降至 `0.058`，但 suffix-5
  仍约为 `0.8x` full。MovieLens 两次更新的维护缺口很小且不稳定，因此尚不能声称跨数据集
  泛化。
- 本地 KuaiRand-1K 并未被“完全使用”：旧 top-5k 只保留标准日志的 3.67%。四 seed 的
  top-5k/top-20k × 6L/12L 组合验证仍保持正 maintenance gap；进一步在 top-50k、长度 512
  下用连续 chunks 覆盖完整 retained base history，使每轮有效 base target 从 230,945 增到
  620,958。此时 full maintenance 的 Best Rank 增益为 `885.56 [460.24, 1310.88]`，cheap
  以 `0.058x` full 成本恢复 54.6%，suffix-4 以 `0.613x` 成本恢复 76.2%。top-50k 仍只覆盖
  13.35% 标准日志，不能表述为 full KuaiRand。
- 旧的“为每个用户估计 KV drift/JVP，再做 reuse/migrate/recompute”路线已退出主线：原始
  用户级 KV 范数与实际质量收益没有可用相关性，估计本身也不具备成本优势。相关实现、旧协议
  结果和旧文档已从当前工作树移除。

## 当前方法

对第 `l` 层，旧状态中额外保存 `Norm_old(x_l)`。cheap refresh 直接计算：

```text
K_l = Wk_current(Norm_old(x_l))
V_l = Wv_current(Norm_old(x_l))
```

它修复投影参数变化，但不修复跨层 hidden propagation。`cheap + suffix-N` 对浅层执行上述
refresh，在 suffix 内传播 current hidden，并让末层只执行 current `Norm + Wk/Wv`；该末层
的 block output 不会被任何后续 prefix K/V 消费。`N = number_of_layers` 与当前模型完整
K/V 重算数值一致。

## 下一阶段

KuaiRand 内部的数据量、序列长度和模型规模探索暂时冻结。下一项跨数据集验证已选定 Taobao
UserBehavior，但仓库中尚无 Taobao 实验结果。首先只做数据审计和小规模
`frozen / full reuse / full compute` motivation gate；确认跨多个真实时间窗口存在可辨识的
cache-maintenance gap 后，才复用已经冻结的 cheap 与比例 suffix 配置做四 seed 验证。不会为了
追求正结果反复调整时间切分、行为标签或层区间。并行的系统任务是测量额外状态移动与端到端成本。

## 目录

```text
src/hstu_kvcache/
  data/          KuaiRand/ML1m 数据与流式日期计划
  models/        模块化简化 HSTU 与一等 K/V 输出
  streaming/     next-item 流式训练与版本 checkpoint 工具
  migration/     layerwise state capture 与 cache migration 算子
scripts/
  motivation_validity.py
  summarize_validity.py
  layerwise_validity.py
  summarize_layerwise_validity.py
  interval_oracle.py
  summarize_interval_validation.py
  streaming_value_control.py
  summarize_streaming_value_control.py
  operator_cost_scaling.py
  scaling_validity.py
  movielens_scaling.py
  summarize_scaling.py
  kuairand_data_coverage.py
  summarize_kuairand_factorial.py
  summarize_kuairand_data_utilization.py
experiments/validity/
  README.md
  LAYERWISE_METHOD.md
  INTERVAL_ORACLE.md
  STREAMING_VALUE_CONTROL.md
experiments/scaling/
  SCALING_V1.md
  KUAIRAND_FACTORIAL_V1.md
  KUAIRAND_DATA_UTILIZATION_V1.md
results/validity/       当前有效结果
checkpoints/validity/   与当前结果配套的 checkpoint
results/scaling/        固定算子的逐轴规模结果
checkpoints/scaling/    深度与 MovieLens 配套 checkpoint
```

## 开发与核验

```bash
pip install -e .
pytest
ruff check src tests scripts
```

实验脚本的参数以 `--help` 和 [docs/eval_protocol.md](docs/eval_protocol.md) 为准。新实验必须
使用新的 protocol 名称，不能跨 protocol 混合汇总。
