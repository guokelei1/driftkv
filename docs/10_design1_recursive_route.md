# Design 1：KuaiRand 单链递归 K/V 维护路线

更新日期：2026-08-11。

## 1. 当前任务

θ1–θ8 的 44.666 GiB 大模型和 8×8 Reuse/Recompute 矩阵已经冻结。8×8 的用途是证明存在质量空间，不是 D1 方法的输出格式。D1 接下来只执行一条线上轨迹：模型逐日从 θ0 更新到 θ8，cache 也沿同一条轨迹持续维护。

初始状态是 θ0 下精确计算的用户前缀 K/V。执行 θ0→θ1 后，D1 产生的 K/V 必须直接成为 θ1→θ2 的输入；以后每条边同理。θ0 之后禁止插入未报告的完整重算，也不能每条边都从精确旧 cache 重新开始。否则测到的是七个独立 one-hop 实验，不是长期部署中的递归误差。

这里有一个必须明确的证据边界：θ0 是 5.620 GiB 自然 bootstrap，θ1–θ8 才是当前选定的 44.666 GiB 大模型。θ0→θ1 会执行并报告，但它没有进入现有 8×8，暂时只作为 bootstrap 诊断；当前 D1 主证据是 θ1→θ2 至 θ7→θ8 七条连续边，且 θ1 的输入仍来自前述 bootstrap 输出而不是隐藏复位。

机器契约是 `configs/evokv_d1/development/kuairand_recursive_chain_design_v0.json`。

## 2. 一条链具体比较什么

每个版本 θt 的同一批 query 同时产生三个必需端点：

- Full Recompute：当前模型 θt 对完整有效历史重算，是 cache fidelity 参照。
- Recursive Reuse：从同一个 θ0 精确初态出发，不做 D1 修复，只携带旧 cache 和共同的新增行为。
- Recursive Method：从同一个 θ0 初态出发，逐边应用当前 D1 候选，上一边输出是下一边唯一输入。

另保留 `edge_local_reuse_from_exact_source` 作为诊断，它对应 8×8 的相邻边，但不能替代 Recursive Reuse。所有端点必须共享当前模型、用户、天然不等长历史、query、正样本和 99 个冻结负样本。

主指标仍是 NDCG@5，同时报告 MRR 与 HR@5。每条边保存三个端点的绝对值，并在 `Full Recompute > Recursive Reuse` 时计算：

```text
gap_recovery =
    (Recursive Method - Recursive Reuse)
    / (Full Recompute - Recursive Reuse)
```

分母不为正时 recovery 写 `NA`，不能裁剪、取绝对值或删边。最终结果是一张按 θ0→θ1、θ1→θ2、…、θ7→θ8 排列的表和一个 θ8 累计总结，不再为方法制作或挑选另一张 8×8。

## 3. 每条边必须记录的量

质量之外，每条边至少记录：

- lineage：源/目标版本、源 cache 的真实来历、递归深度、是否确实消费上一边输出、精确复位次数；
- 语义工作量：用户数、有效 prefix tokens、原样携带、迁移、精确重算、recent replay 和 append 的 tokens；
- 数据量：K/V 读写 bytes、方法参数 bytes、临时峰值 bytes；
- 理论计算量：由实际 tensor shape 和算法推导的操作数、同请求 Full Recompute 操作数及二者比例；
- 实测成本：GPU 时间单独记录，不能用 token 比例或手写常数冒充。

有效 token 才进入语义工作量，padding 不计。理论操作数必须随具体候选实现计算，因此当前契约只规定字段，不预先编造统一 FLOP 常数。

## 4. 设计空间与公平边界

第一轮可以探索三类基本部件及其组合：旧 K/V 到新 K/V 的轻量 transport、部分用户或 token 的精确重算、只重放近期行为。具体是线性、低秩、分层还是别的形式尚未冻结；当前目标是用小规模验证找出真实 Pareto frontier，而不是预先认定某个旧方法。

方法拟合和超参数选择只能使用与冻结 qualification query 分离的记录。transport 拟合和线上逐记录动作可以使用无标签的配对 K/V、模型参数差异、用户长度和 cache lineage，不能读取推荐标签或每个 query 已实现的收益；独立 development split 上的候选级聚合质量可以用于选择超参数。qualification 正样本、候选标签和逐 query NDCG/MRR/HR 始终不可用于拟合、选动作或调参。所有尝试都要留下紧凑摘要，包括失败候选，最终按整条递归轨迹选择，不能为每条边事后拼一个最好参数。

## 5. 执行顺序

1. Foundation preflight：核对 θ0、θ1–θ8、8×8 hash、两卡几何和 cache shape。
2. Single-edge canary：先跑 θ0→θ1 与 θ1→θ2 的小型、数据隔离验证，解决实现与 lineage 问题。
3. Recursive prefix：不复位地继续到 θ3，确认误差和工作量能够递归传播。
4. Full chain：冻结一个候选后一次执行 θ0→θ8，生成完整逐边表和累计结果。
5. Cost validation：语义与质量冻结后，再测 GPU 时间、峰值显存和传输；不能为了时间结果更好而改变质量协议。

阶段 2 或 3 的结果会决定下一轮候选，因此实验在真实结果边界停下分析，不预先把所有搜索串成一个长任务。当前只允许 GPU0/GPU1 上的一个 two-rank job。

## 6. 当前 readiness

进入 D1 设计与 canary 所需的 foundation 已具备：θ0、八个大模型 checkpoint、自然日 transition、冻结候选和 8×8 端点均有机器绑定；模型重建路径也已保留。尚未完成的是 D1 方法本身、独立 fit/development 用户清单和单链 runner——这正是下一阶段要实现的内容，不是 baseline 缺失。

运行以下命令可重新检查 readiness：

```bash
python scripts/preflight_evokv_kuairand_recursive_d1.py
```

该命令只读，不训练、不生成 K/V，也不修改 checkpoint。
