# 导师汇报稿：流式训练下 HSTU KV Cache 的版本迁移

> 当前定位：这是一条已经通过当前 KuaiRand 数据/模型规模 gate 的早期研究路径，而不是一篇
> 已经完成的论文。我们已经有了
> `motivation → structural observation → minimal design → scaled preliminary evaluation` 的闭环；
> 当前重点已经从“值不值得继续”转向端到端系统证据、自然混合版本与跨数据集泛化。

---

## 0. 汇报摘要

生成式推荐模型依赖 KV Cache 复用用户历史，但推荐模型又会持续进行流式训练。模型从
$\theta_t$ 更新到 $\theta_{t+1}$ 后，即使用户历史 $x_u$ 不变，旧缓存
$F(\theta_t,x_u)$ 也不再等于新模型应产生的 $F(\theta_{t+1},x_u)$。全量重算可以恢复
一致性，却会重新支付所有用户历史的完整前向成本。

我们的初步结果支持七个判断：

1. **流式训练有价值，cache inconsistency 会侵蚀其中一部分，而且具有时间尺度。** 单次小
   更新造成的平均损失较小；缓存跨多个模型版本后会出现可恢复损失，而且一部分损失可能在
   某次更新集中出现。因此系统既不能永远复用，也不必每次更新都全量重算；单看 cache age
   也不足以稳定判断何时重算。
2. **把 KV 当黑盒、按用户预测是否重算，目前没有证据支持。** 用户级相对 KV drift 与
   实际排序收益几乎不相关，基于原始 drift/JVP 的用户选择不是当前主线。
3. **打开 HSTU 的层结构后，存在低于完整前向的迁移路径。** 使用新模型的 $W_k/W_v$
   对缓存的 $\operatorname{Norm}(x)$ 做 cheap projection refresh，再对一段深层连续后缀执行
   current block，可以形成可测的计算—质量曲线。
4. **这条曲线不是六层单点现象，但完整方法曲线尚未跨数据集泛化。** 固定 optimized suffix
   后，KuaiRand 的长度 32/64/128 和深度 3/6/9 都保留了中间 Pareto 点；MovieLens 的两次
   更新只产生很小且不稳定的维护缺口。
5. **旧实验明显低估了 KuaiRand 数据利用率；补全 retained history 后，问题与曲线反而更清楚。**
   top-50k、长度 512 的 chunked 训练将每轮有效 base target 从 23.1 万增至 62.1 万。四个 seed
   下，theta-5 maintenance Best Rank 从 latest-only 的 82.68 增至 885.56，NDCG 区间也排除
   0；cheap 仅需 0.058x full，suffix-4 以 0.613x 成本恢复 76.2% Rank gap。
6. **三份真实曝光数据已经给出一致的 pre-design motivation，cheap 方法开始跨数据集，
   suffix 泛化仍是开放问题。**
   KuaiRand、fixed-horizon Tenrec QB、top-5k Tenrec QK 在四个 seed 上都表现为：流式训练
   长期有价值，旧 cache 仍保留大部分价值，而旧版本 cache 留下额外可恢复 gap。原始
   BestRank 数字不能跨 catalog 比较；用“被 stale cache 损失的流式训练收益比例”归一化后，
   三者 coarse endpoint 分别为 17.6%、31.5%、27.6%，最大仅为最小的 1.79 倍。细粒度固定
   endpoint 仍为 17.7%、35.0%、14.1%，并出现局部跃升。在 aligned QK 上，cheap 以
   0.194x full 成本获得 9.17 `[6.71, 11.63]` BestRank 增益，恢复 70.6% 的 mean full gap；
   aligned QB 的 partial method 则在 seed 0 失败。suffix 曲线仍未跨数据集复现，ZhihuRec
   保留为负面边界。
7. **真正适合方法展开的是“中等失效 + 高重算成本”，而不是刻意把 reuse 做坏。**
   四 seed 的 KuaiRand top-50k/all-chunks 同时满足这两点：stale cache 损失 23.1% 的流式
   训练收益、仍保留 76.9%，而长度 512 下 cheap 只需 0.058x full。预先冻结的长上下文
   Tenrec 筛选虽然把 cheap/full 降到约 0.11-0.12，却只有 1.1%-6.0% staleness tax，因此
   已按规则停止。它说明长序列会放大重算成本，但不会自动放大版本不一致的质量损失。

在当前六层小模型、四个训练种子的实验中，cheap refresh 约使用优化后完整 KV 重算的 19%
GPU 计算；随着 full suffix 加深，质量总体向 fresh 靠近。重算最深五层约使用 77% 计算，在
两个可辨识的累计缓存年龄上，其 Best Rank 和 NDCG@100 与 full 的配对差距均未被当前
实验区分出来。这不是最终性能结论，但足以说明该方向已经越过“是否值得继续”的最小门槛。
进一步的形状基准显示，序列长度从 128 增至 512 时 cheap/full 从 0.189 降至 0.058，说明
轻量路径在长上下文下的相对优势会扩大；高质量 suffix-5 仍接近 0.8x full，是下一阶段必须
面对的成本边界。

更完整的 KuaiRand 使用进一步强化了这个判断，但也校正了“已经用了很多数据”的错觉。本地
标准日志有 1171 万行，原 top-5k 只保留 3.67%；top-50k 也只保留 13.35%。因此新的结果是
更强的数据规模证据，而不是 full KuaiRand 或工业规模结论。

跨数据集结果把问题主张与方法主张分开了。版本失效的 motivation 已在 KuaiRand、QB、QK
三份曝光数据上形成一致证据；但 QB/QK 来自同一 Tenrec collection，且 Tenrec 只有用户内
顺序、没有全局日历时间，因此不能把它写成三个独立来源的自然时间漂移复现。方法方面，
cheap projection refresh 不仅在原 QB 协议上有正收益，现在也在 aligned QK 的四 seed 上以
约 19% 成本恢复 70.6% mean BestRank gap；但 aligned QB seed-0 只恢复 5%。更重要的是，
“加深 suffix 就稳定逼近 full”目前仍只在 KuaiRand 上得到强支持。逐层 stale K/V 误差上升
说明 deep suffix 合理，却不能保证任务指标随计算量单调改善。

新的固定终点实验还把“随时间积累”推进了一步：在相同当前模型、相同用户历史和相同目标
下，只改变 cache 的模型版本。KuaiRand 与 QB 各有一个跨 4/4 seed 同向的局部恶化点；QK
每个 seed 都有明显局部跃升，但位置在 age 5、8、9、10 之间移动。这说明固定周期仍应作为
baseline，却不能被当成跨更新稳定的质量控制器；更合理的下一步是 model-version cohort
级的 update-aware trigger，而不是恢复昂贵的 per-user drift/JVP 路线。

---

## 1. 问题：模型版本变化使派生缓存整体失效

### 1.1 与普通行为追加不同的失效源

HSTU 类生成式推荐模型将用户历史编码为逐层 K/V Cache。新行为到达时，服务端只需处理
新增 token，并读取历史 K/V，而不必再次执行整段历史。

已有缓存复用通常隐含一个前提：产生缓存和消费缓存的是同一组模型参数。推荐系统却需要
周期性吸收新交互数据，形成参数序列

$$
\theta_0 \rightarrow \theta_1 \rightarrow \cdots \rightarrow \theta_t.
$$

用户 $u$ 在版本 $v$ 下的派生缓存记为

$$
C_v(u)=F(\theta_v,x_u).
$$

当模型更新到 $\theta_{v+1}$ 后，即使 $x_u$ 完全不变，通常也有

$$
C_v(u)\neq F(\theta_{v+1},x_u).
$$

这与“用户新增行为导致输入变化”不同。尾部行为追加不会使已有 causal prefix 自动失效，
但参数变化会改变所有历史位置上的投影、归一化和逐层 hidden propagation。本文研究的是
这一**模型版本维度的缓存失效**。

| 场景 | 用户历史 $x$ | 模型参数 $\theta$ | 核心缓存问题 |
|---|---|---|---|
| 增量行为 serving | 尾部追加 | 固定 | 为新 token 追加 K/V |
| 固定模型缓存压缩 | 固定或追加 | 固定 | 减少容量与读取成本 |
| **本文：模型版本更新** | 可以完全不变 | **变化** | 旧派生 K/V 与当前模型不一致 |

### 1.2 真正的研究问题

两个平凡端点分别是：

- 永久复用旧缓存：成本最低，但误差随版本年龄累积；
- 每次模型更新后全量重算：结果最新，但对所有用户重新执行历史前向。

因此核心问题不是“KV 是否会失效”，而是：

> 能否利用 HSTU 的内部计算结构，将旧版本 KV 迁移到新版本，并在显著低于完整历史前向
> 的成本下恢复足够多的推荐质量？

这个问题把缓存维护从二元的 reuse/recompute 变成了一个有结构的 migration 问题。

---

## 2. Motivation：修复协议后，问题仍然成立

### 2.1 Validity-first 实验

早期实验存在同位置重构目标、padding 位置取值和 full-history stale forward 等问题，不能作为
可靠证据。当前 motivation 使用修复后的 `validity_v1_incremental_prefix_cache`：

- 训练目标是 hidden $t$ 预测 item $t+1$；
- 只使用当前流式日期中的 engaged target；
- vocabulary 只在基础训练期拟合，避免未来信息泄漏；
- padding 不参与 last hidden 或 K/V；
- 评估使用旧版本 prefix cache，并用当前模型处理最新行为 token；
- fresh baseline 是当前模型对完整历史的前向，并在 5,000 个物品上做全目录排序。

实验使用 KuaiRand-1K、14 个基础日期、五个三日流式窗口。Motivation 模型是三层简化 HSTU
（908,160 参数、序列长度 128），每个训练种子评估 300 个用户，共四个独立训练种子。
统计单位是训练种子，而不是把同一模型下的用户错误地当作独立重复实验。

### 2.2 单步 staleness 较弱，累计 cache age 明显

下表报告 fresh recompute 相对 stale reuse 的平均增益；Best/Mean Rank 为排名降低的数量，
其余指标为绝对提升。每个数先在 seed 内跨五个窗口平均，再在四个 seed 间统计。

| 缓存版本关系 | Best Rank | Mean Rank | NDCG@100 | Hit@100 |
|---|---:|---:|---:|---:|
| 上一版本 $\theta_{t-1}\rightarrow\theta_t$ | +4.15 | +6.26 | +0.00086 | +0.0053 |
| 固定旧版本 $\theta_0\rightarrow\theta_t$ | **+63.39** | **+109.12** | **+0.00527** | **+0.0415** |

累计 Best Rank 增益的 seed-level 95% t interval 为 `[41.57, 85.20]`。在五个缓存年龄上，
累计参数距离与 Best Rank 增益的 Spearman 相关系数在四个 seed 上平均为 $0.975$。

这组结果支持的是一个比“旧缓存总是有害”更细的结论：

- 一次小更新的影响通常较弱，后期窗口甚至可能出现方向反转，不能声称每次更新都必须重算；
- 多版本累计后，staleness 损失稳定出现，永远复用同样不成立；
- cache age 因而提供了 migration 发挥作用的区间。

### 2.3 一个决定方法转向的负结果

我们原先考虑估计每个用户的

$$
\left\|F(\theta_{t+1},x_u)-F(\theta_t,x_u)\right\|
$$

并据此选择哪些用户重算。但在 20 个 seed-window 单元中，用户级相对 KV drift 与实际
rank-utility gain 的 Spearman 相关系数均值只有 $0.020$，95% 区间为
`[-0.012, 0.052]`，与随机选择基本一致。

同时，per-user JVP 的成本约为一次反向传播，本身不比直接 KV 前向便宜。因此当前证据不支持
“先为每个用户估 drift，再决定 reuse/recompute”的主线。这个负结果缩小了方法空间：方法应
直接产生低成本、可消费的新缓存，而不是先支付一笔昂贵估计成本。

### 2.4 流式训练价值链控制

为了避免“模型本身并不需要流式更新”的替代解释，我们在同一六层 validity 协议下比较：

- `frozen`：$\theta_0$ 模型、与其一致的 cache 和 scoring head；
- `full reuse`：当前 $\theta_t$ 模型消费 $\theta_0$ prefix cache；
- `full compute`：当前 $\theta_t$ 模型使用当前 prefix cache。

下表给出四个训练种子的均值，所有数均为正向 Best Rank 改善：

| Cache age | full compute 相对 frozen | full reuse 相对 frozen | full compute 相对 full reuse |
|---|---:|---:|---:|
| $\theta_0\rightarrow\theta_1$ | 111.43 | 105.77 | 5.66 |
| $\theta_0\rightarrow\theta_3$ | 237.97 | 196.29 | **41.68** |
| $\theta_0\rightarrow\theta_5$ | 484.34 | 399.02 | **85.32** |

这说明流式训练本身带来大且可重复的收益，旧 cache 并不会使收益全部消失，而是保留大部分
收益并留下随 cache age 增长的可恢复缺口。在 $\theta_5$，cache maintenance 占流式训练总
Best Rank 收益的 17.6%，占 NDCG@100 收益的 30.8%；二者 seed-level 区间均排除 0。
MRR 并非始终改善，因此本文应将主张限定在 full-catalog Best Rank 与 NDCG，而不是声称所有
ranking metric 同向变化。

### 2.5 固定当前模型后，cache age 仍会出现平台、跃升与反转

为了不把 cache age 与评测日期、用户组成混在一起，我们固定最终 $\theta_T$、最终评测
用户与历史，只用 $\theta_{T-1},\ldots,\theta_0$ 分别生成同一 prefix 的 K/V。跨数据集不再
比较 BestRank 的绝对差，而比较

$$
\text{staleness tax}=
\frac{\text{full compute}-\text{reuse}}
     {\text{full compute}-\text{frozen}},
$$

即 stale cache 损失了多少比例的流式训练收益。

| 数据集 | coarse endpoint tax | fine endpoint tax | fine 中首次达到 10% 的 age |
|---|---:|---:|---|
| KuaiRand | 17.6% `[11.5%, 23.7%]` | 17.7% `[11.8%, 23.6%]` | 14 / 14 / 14 / 14 |
| Tenrec QB | 31.5% `[22.5%, 40.4%]` | 35.0% `[26.3%, 43.7%]` | 2 / 1 / 5 / 4 |
| Tenrec QK | 27.6% `[16.8%, 38.5%]` | 14.1% `[5.6%, 22.7%]` | 1 / 10 / 5 / 8 |

在 fine curve 中，KuaiRand 的 age 13→14 与 QB 的 age 6→7 分别一次增加 6.3 和 9.4 个
百分点，且均为 4/4 seed 同向。QK 的最大单步增加在每个 seed 都存在，但分别落在
8→9、9→10、4→5、7→8；同一个 age 上无法得到窄区间。三个数据集的每条曲线也都包含
非单调反转。

因此现在可以把 observation 表述得更具体：旧 cache 不是匀速老化，它可以先处于低损失
平台，再因某次模型更新局部恶化；坏点的位置还会随更新轨迹变化。现有结果不能证明任何
经过单独调参的 fixed window 都失败——KuaiRand 在当前设置下反而很稳定——但足以说明
age-only window 不是通用、可校准的质量状态。这里的最大跃升统计仍是 exploratory，下一份
独立数据必须冻结定义后再验证。

### 2.6 机会区间：主结果足够宽，但不能把两条轴强行绑在每个数据集上

方法需要的不是一个退化到不可用的 reuse 端点，而是一个内部工作区间：

1. `full compute - frozen > 0`，说明流式训练值得做；
2. stale reuse 保留相当一部分收益，说明不能简单退回 frozen；
3. `full compute - reuse` 稳定且中等，给 migration 留出质量恢复空间；
4. full history recompute 足够贵，给 partial migration 留出计算空间。

当前最完整的同一设置证据来自 KuaiRand top-50k/all-chunks：full compute、reuse 和
maintenance 的 BestRank 改善分别为 3837.67、2952.11 和 885.56，即 reuse 保留 76.9%，
staleness tax 为 23.1%；长度 512 下 cheap/full 为 0.058。这已经是一个适合优化的中间点，
不需要继续人为放大失效。

我们同时预先冻结了 QB top-50k 和 QK top-20k 的 256-raw-exposure 长上下文筛选。实际保留
历史中位数为 244 和 206；对应 batch-32 full/cheap 为
`4.504/0.512 ms` 和 `3.578/0.439 ms`。但在 `1e-4/2e-4/4e-4` 三档流式学习率下，QB 的
oldest-cache tax 只有 5.6%/1.1%/2.5%，QK 为 3.6%/6.0%/3.7%，且五个 cell 的 NDCG
maintenance 非正。因此这两个 cell 停在 motivation screen，不运行方法，也不扩 seed。

汇报时应明确区分：KuaiRand 提供当前最强的联合 quality-cost opportunity；aligned QB/QK
提供跨数据集的版本失效复现；长上下文 QB/QK 则是“成本变大不等于失效比例变大”的负面
边界。这样比为得到漂亮方法结果而继续调数据更可信。

---

## 3. Design：把版本漂移分成直接投影变化与跨层状态传播

### 3.1 HSTU K/V 的结构机会

对第 $l$ 层，可将 K/V 写成

$$
K_l=W^K_l\operatorname{Norm}_l(x_l),\qquad
V_l=W^V_l\operatorname{Norm}_l(x_l).
$$

模型更新后，K/V 的变化来自两部分：

1. **直接参数变化**：$W^K_l,W^V_l$ 发生变化；
2. **间接状态变化**：前面各层参数改变，使输入该层的 $x_l$ 发生变化并逐层累积。

完整前向同时修复两部分，但成本最高。只要缓存旧版本的 normalized state，就可以用当前
$W^K_l,W^V_l$ 修复第一部分，而不执行该层 attention、gate 和 residual。这就是 cheap layer：

$$
\widetilde K_l=W^{K,t+1}_l\operatorname{Norm}^{t}(x^t_l),\qquad
\widetilde V_l=W^{V,t+1}_l\operatorname{Norm}^{t}(x^t_l).
$$

它不是精确 fresh，因为没有修复 $x_l$ 的跨层传播误差，但比完全复用多吸收了新投影参数。

### 3.2 当前最小设计：cheap prefix + full suffix

对于 $L$ 层模型，选定分界点 $s=L-N$：

- 第 $1$ 到 $s$ 层使用 cached `Norm(x)` 与当前 `Wk/Wv` 做 cheap refresh；
- 从缓存的 $x_{s+1}$ 开始，在最深连续区域内传播 current hidden；区域末层只执行 current
  `Norm + Wk/Wv`，因为它的 block output 不会影响任何后续 prefix K/V；
- $N=0$ 是 cheap-all，$N=L$ 从当前 input embedding 开始，严格等于 full recompute。

这里的 `top-N` 更准确地说是 **deepest suffix-N**，不是按用户或按动态分数选择的任意 N 层。
选择深层后缀是一个受实验启发的初始假设：当前六层模型中，stale K/V 相对误差随深度稳定
增大，而且连续后缀能够让更新后的 hidden 在相邻层之间传播。它还不是最优性结论。

该设计的学术重点并不在“固定取后 N 层”这个启发式本身，而在于：

> 利用 K/V 生成过程的可分解结构，把一次完整历史前向拆成 projection refresh 与
> state-propagating block recompute，从而暴露连续的成本—质量操作空间。

---

## 4. Preliminary Evaluation：已经出现可重复的成本—质量曲线

### 4.1 实验设置

为了获得足够的层数分辨率，我们在不扩大数据的前提下使用六层简化 HSTU：hidden size 96、
四个 head、head dimension 24、序列长度 128、5,000 个物品，共 770,496 参数。仍然使用
300 个用户、全目录 engaged-positive 排序和四个训练种子。

方法实验聚焦累计 cache age：$\theta_0\rightarrow\theta_1/\theta_3/\theta_5$。迁移时间使用
CUDA event 测量 GPU-resident batch，并归一化到同一 prefix 的 full K/V recompute。质量恢复率为

$$
\text{Recovery}(M)=
\frac{M_{\text{method}}-M_{\text{reuse}}}
     {M_{\text{fresh}}-M_{\text{reuse}}},
$$

其中 rank 指标先统一成“降低越多越好”的 gain。短年龄 $\theta_0\rightarrow\theta_1$ 中
fresh 本身收益很小，比例容易被噪声放大，因此下表只呈现两个可辨识年龄。

| 配置 | 时间 / full | 额外状态 / KV | $\theta_0\rightarrow\theta_3$ Rank / NDCG 恢复 | $\theta_0\rightarrow\theta_5$ Rank / NDCG 恢复 |
|---|---:|---:|---:|---:|
| cheap all | **0.187×** | 50.0% | 50.8% / 24.3% | 72.9% / 67.1% |
| cheap + suffix-2 | 0.372× | 41.7% | 65.3% / 54.0% | 84.6% / 67.8% |
| cheap + suffix-4 | 0.635× | 25.0% | 87.5% / 64.7% | 101.3% / 90.5% |
| cheap + suffix-5 | **0.767×** | 16.7% | **88.8% / 84.7%** | **103.0% / 99.7%** |
| full recompute | 1.000× | 0.0% | 100% / 100% | 100% / 100% |

在这张小 gap 表里，超过 100% 的方法与 full 配对差异包含 0，不能解释为近似方法优于
fresh。作为绝对量参照，full 在
$\theta_3$ 和 $\theta_5$ 上分别比 reuse 降低 41.68 和 85.32 个 Best Rank；对应的
NDCG@100 提升为 0.00344 和 0.00693。

### 4.2 当前可以读出的三个结构性结果

**第一，cheap refresh 确实是一个有效而非平凡的端点。** 它在约 19% 的 GPU 计算下，随
cache age 增大可恢复约 51%–73% 的 Best Rank 收益。说明新投影参数本身承载了可利用的
版本更新信息。

**第二，执行更多连续层能够补回 cheap 未处理的 hidden propagation。** suffix-2 相对
cheap 的额外 Best Rank 收益在 $\theta_3$ 为 6.02，95% 配对区间 `[1.54, 10.49]`；在
$\theta_5$ 为 9.95，区间 `[4.53, 15.36]`。这说明层间状态传播并非可以完全忽略。

**第三，高质量端已经接近 full，但节省仍有限。** suffix-5 在两个年龄上的 Best Rank 和
NDCG@100 与 full 的配对差距区间都包含 0，同时使用约 77% 计算。这只能说明当前实验尚未
区分二者，并不是统计等价证明；它也表明若目标是几乎无损，当前实现只节省约 23%。

另一个重要负结果是 suffix-1：质量几乎不变。其结构原因已经被用于优化算子——最后一层
只执行 `Norm + Wk/Wv`，不再执行无法影响后续 prefix K/V 的 attention/gate/residual。
优化后 suffix-1 从旧实现的 0.329× 降至 0.229×，所有 suffix 深度与旧实现的 K/V 最大误差
均为 0。进一步的 21 区间搜索及 held-out seeds 1–3 验证显示，中间或早期区间没有稳定优于
同成本最深 suffix：它们在 Best Rank 上持续更差，偶发的 NDCG 优势不跨年龄与 seed 稳定。

### 4.3 首轮逐轴规模验证

在不改变 optimized suffix 的前提下，我们补了 sequence length、batch size、模型深度和
update magnitude 四个轴。最重要的结果不是某个单点更高，而是原来的曲线没有随规模变化
立即消失：

| 模型深度 | cheap 成本 / Rank 恢复 | 约 1/3 suffix 成本 / 恢复 | 约 2/3 suffix 成本 / 恢复 |
|---:|---:|---:|---:|
| 3 | 0.202 / 66.7% | 0.277 / 67.0% | 0.554 / 94.6% |
| 6 | 0.187 / 72.9% | 0.373 / 84.6% | 0.636 / 101.3% |
| 9 | 0.181 / 70.8% | 0.399 / 87.4% | 0.668 / 95.3% |

在这组首轮小模型 cell 中，超过 100% 的方法与 full 配对差异无法区分，因此不能据此宣称
优于 fresh；可以判断的是结构分解在 3/6/9 层都形成了可用中间点。
沿真实 $\theta_0\rightarrow\theta_5$ 方向做受控参数插值时，stale K/V error 随更新幅度从
0.206 单调增至 0.656；Best Rank 缺口却在插值 0.75 处达到峰值，说明参数距离只能作为
版本严重度特征，不能直接当作质量收益预测器。

MovieLens 的结果必须主动作为边界讲清楚。optimized full K/V 仍严格等于 fresh，增量 parity
也小于 $4.8\times10^{-6}$，说明代码路径成立；但两次更新后的 full maintenance Best Rank
均值只有 1.48，seed 区间为 `[-6.37, 9.33]`，NDCG 也无法区分。因此仅就该 MovieLens gate，
证据支持“算子可以迁移到第二数据格式”，不支持“问题和质量收益已经跨数据集复现”。

### 4.4 数据/模型组合规模与 KuaiRand 利用率修复

随后我们固定同一个算子，做了 top-5k/length-128 与 top-20k/length-256、6L/H96 与
12L/H192 的四 cell 组合验证。四个 cell 的 full maintenance Best Rank 与 NDCG@100 均为
正；组合最大 cell 中，full 绝对 GPU 时间是原 baseline 的 9.1 倍，而 cheap/full 降至 0.099。
这说明轻路径的相对计算优势会随模型与上下文共同扩大。

但数据审计发现，扩大 catalog 仍不等于真正使用更多训练记录：旧 iterator 每个用户每轮只取
最后一段。我们因此增加了不改变既有默认协议的 `all_chunks` 模式，用 stride 511 的长度 512
chunk 覆盖 top-50k retained base history，并保持流式 target 数完全相同。

| top-50k 训练方式 | 每轮有效 base target | full compute over frozen Rank | maintenance Rank / NDCG@100 |
|---|---:|---:|---:|
| latest only | 230,945 | 2,064.19 | 82.68 / 0.00109 |
| all chunks | **620,958** | **3,837.67** | **885.56 / 0.00250** |

all-chunks 的 maintenance Rank seed 区间为 `[460.24, 1310.88]`，NDCG@100 区间为
`[0.00169, 0.00330]`。full compute 相对 frozen 在四个 seed 都大幅为正；累计参数距离反而从
latest 的 0.326 略降到 0.308。因此更强 gap 不是简单由更大 parameter norm 或无价值的流式
训练造成的，而是说明训练数据覆盖会实质改变 cache version inconsistency 的强度。

在这个更可辨识的 operating point 上，方法曲线也更保守、更可信：cheap、suffix-2、
suffix-4、suffix-5 分别以 0.058/0.248/0.613/0.796 的 full 成本恢复
54.6%/58.9%/76.2%/84.1% Best Rank gap。高质量端仍然昂贵，不能用小 gap 下偶然超过 100%
的 recovery 掩盖这一边界。

一个额外的单-seed bridge 把同一 top-50k/all-chunks 数据扩到 12L/H192：full maintenance
为 659.04 Best Rank，cheap、约 1/3 suffix、约 2/3 suffix 分别以 0.054/0.312/0.653 成本
恢复 61.1%/63.4%/82.9%。它没有暴露组合失效，但只能作为连接两个已重复轴的描述性 gate，
不能代替跨 seed 结论。

组合规模还暴露了一个应主动报告的现象：在 top-20k/12L cell 中，one-third suffix 的 Best
Rank 配对增益比 full 高 25.84，区间 `[7.69, 44.00]`，但 NDCG 配对差异接近 0 且区间跨 0。
这说明 full recompute 是版本一致性与 cache fidelity 的 oracle，却不是任意 ranking metric 的
数学上界。后续必须同时报告 fidelity、多项 task metric 和 method-vs-full 配对差异，不能把
所有 recovery>100% 简单写成噪声，也不能据此声称 partial 全面优于 full。

### 4.5 真实曝光跨数据集复现

后续数据均保留真实负曝光并复用同一个 next-positive-item 目标。首轮统一 top-50k 协议暴露
两个测量问题：QB 的 base-only cohort 随窗口从 4,049 个正样本用户降到 2,552 个，cache age
与用户组成变化混在一起；QK 的 50k item embedding 在 5k 用户下形成稀疏、方差很大的
maintenance 分母。因此在不看标签、不看 seed 结果的前提下，各改变一个数据轴：

- QB 保留 top-50k，但只纳入原始曝光长度覆盖完整 `64+6x8` horizon 的用户；六个窗口因此
  都稳定在约 3.7 万条保留曝光。这个设置条件化于未来活跃可用性，但不按未来正反馈选人。
- QK 改为 top-5k，匹配 KuaiRand 的 dense-catalog operating point；模型参数从 509 万降为
  77 万。它提高了测量稳定性，但不能据此断言 catalog size 是唯一因果因素。

四 seed 同口径结果为：

| 数据集 | theta-5 full over frozen | theta-5 full reuse over frozen | theta-5 maintenance | age/gap Spearman |
|---|---:|---:|---:|---:|
| KuaiRand | 484.34 `[462.15, 506.54]` | 399.02 `[370.79, 427.26]` | 85.32 `[53.74, 116.91]` | 0.925 `[0.845, 1.000]` |
| Tenrec QB fixed horizon | 94.70 `[70.49, 118.90]` | 64.38 `[54.41, 74.35]` | 30.31 `[14.08, 46.55]` | 0.600 `[0.005, 1.000]` |
| Tenrec QK top-5k | 47.34 `[29.20, 65.47]` | 34.34 `[20.55, 48.13]` | 13.00 `[5.70, 20.30]` | 0.700 `[0.475, 0.925]` |

BestRank 绝对值不能跨 catalog 比大小；这里比较的是符号、价值分解和年龄关系。三个数据集
现在都支持“流式训练必要—旧 KV 部分保留—老缓存形成可恢复 gap”。QB 的 gap 出现更晚，
QK 的相邻窗口不严格单调，但跨五个年龄的 seed-level 关系稳定为正。

固定最终模型和评测窗口后，coarse BestRank staleness tax 分别为
17.6%/31.5%/27.6%，把 raw full-compute gain 的 10.2 倍跨度缩小到 1.79 倍；MeanRank tax
跨度为 1.60 倍。Fine BestRank tax 的最大/最小为 2.48 倍，并进一步暴露了平台后的局部跃升。
这才是适合跨数据集讲效应量的主表；上面的 raw BestRank 只保留为各数据集内部的直观量纲。

aligned method gate 已经完成。QB seed-0 的 full gap 为 21.95，而 cheap 只恢复 1.09，
suffix-2/4/5 为 -0.33/-2.07/+1.16，且所有 partial NDCG 都下降，因此没有扩 seed。QK
seed-0 中 cheap 以 0.197x 成本恢复 89.3% BestRank，所有更深 suffix 的主指标反而更低；
只扩展 cheap 后，四 seed 结果为：

| 配置 | 成本 / full | BestRank gain | Mean recovery | NDCG@100 gain |
|---|---:|---:|---:|---:|
| QK cheap | 0.194 `[0.187, 0.200]` | **9.17 `[6.71, 11.63]`** | **70.6%** | 0.00125 `[-0.00091, 0.00341]` |
| QK full | 1.000 | **13.00 `[5.70, 20.30]`** | 100% | 0.00227 `[-0.00086, 0.00541]` |

cheap 与 full 的 paired BestRank 差为 -3.83 `[-8.85, 1.19]`，当前未区分但不构成等价证明。
因此现在可以讲 projection refresh 这个最轻量结构在 QK 上跨数据集成立，不能讲完整 suffix
曲线已经泛化。ZhihuRec 的 theta-5 maintenance 仍为负，作为负面边界保留。

---

## 5. 当前工作的潜在学术价值

### 5.1 一个边界清楚的问题轴

本工作研究的不是输入追加、缓存压缩或固定模型下的 serving，而是**持续训练造成的模型版本
变化如何破坏派生 KV Cache**。这一版本维问题具有清楚的数学对象、系统代价和 fresh oracle。
它与已有工作的最终新颖性边界仍需通过完整 related-work 调研确认，但已经足以构成独立问题。

### 5.2 从黑盒缓存判断转向白盒结构迁移

早期路线试图预测“谁需要重算”，容易陷入估计本身不比重算便宜的问题。当前设计直接利用
HSTU 的 `Norm → K/V projection → block propagation` 结构，产生新的可服务缓存。这里真正
值得发展的抽象是 **structure-aware cache migration**，而不是某个固定的 suffix 数字。

### 5.3 版本风险与层级迁移共同形成二维决策空间

Motivation 表明单步 staleness 通常较弱、累计后可能局部跃升，而且 age 本身并不是可校准的
质量状态；method 表明迁移深度控制成本与质量。因此系统可以围绕两个变量设计：

- 版本维：old/current model pair 经历了哪些更新、当前兼容性风险有多大；
- 结构维：哪些层只 refresh projection，哪些连续层需要传播新 hidden。

这比“更新后全部缓存立即失效”或“只挑部分用户重算”提供了更细的研究空间。

### 5.4 负结果同样提供研究判断力

原始 KV norm drift 不能预测用户收益、suffix-1 的 full block 大部分无效，这两个结果分别排除
了昂贵的 per-user estimator 和一个低效的层级配置。当前路线是被实验逐步收窄出来的，而不是
先写一个复杂方法再寻找现象配合。

---

## 6. 当前能讲什么，不能讲什么

### 可以讲

1. 我们已经修复了足以影响结论的训练和 serving 评估问题，核心现象在四个 seed 上仍成立。
2. 模型版本 staleness 具有随 cache age 增长的质量影响，同时单步影响较小，存在迁移空间。
3. 三个数据集的 BestRank staleness tax 已处在 2.5 倍以内；局部跃升与越界年龄变化说明
   age-only window 不是通用的质量控制器，值得测试 model-version cohort 级触发。
4. HSTU 的 K/V 计算可以被拆成 cheap projection refresh 与部分 state propagation。
5. 最小设计的曲线已经在长度与 3/6/9 层上保留，值得进入混合版本和系统成本评测。
6. 在 top-50k complete-base-chunk 协议下，流式训练价值、maintenance gap 与固定 suffix 曲线
   都在四个 seed 上成立，路线已通过当前 KuaiRand 数据规模 gate。
7. 在 KuaiRand、fixed-horizon QB、top-5k QK 上，pre-design motivation 的三个环节均在四个
   seed 上成立；差异主要体现在 gap 出现的时间与幅度，而不是逻辑方向。

### 不能讲

1. 不能声称当前方法已经达到论文严谨度或优于现有系统；related-work 边界尚未完整核验。
2. 不能把小模型的相对 GPU kernel 时间直接外推到工业用户规模或端到端 serving latency。
3. 不能声称 deepest suffix 在所有规模上最优；连续区间 oracle 只在六层模型完成，更深模型
   仅验证了预先固定的 proportional suffix，没有重新搜索任意区间。
4. 不能忽略额外状态：cheap/suffix 方法需要保存 normalized state 和 split hidden，且尚未测量
   host-device transfer、缓存读取、allocator 与 admission 开销。
5. 不能把“与 full 差异不显著”表述成严格等价；当前只有四个训练种子。
6. 不能声称完整 suffix 曲线已经跨数据集泛化；新的 QB/QK 结果只冻结了 motivation，方法
   尚未在 aligned cohort 上重跑，ZhihuRec 也没有形成正的累计 gap。
7. 不能声称已经使用完整 KuaiRand：top-50k 仍只覆盖标准日志 13.35%，KuaiRand-27K 未在本地，
   random-exposure log 也没有混入训练。
8. 不能把 full 当作所有 task metric 的质量上界；它严格定义的是当前版本的一致性结果。
9. 不能把 QB/QK 写成两个独立数据来源，也不能把 Tenrec 用户内顺序写成真实全局日历时间；
   QB fixed-horizon 结果还条件化于用户能覆盖完整活跃 horizon。
10. 不能声称任意 fixed-window reuse 都必然失败；当前只证明 age 不是跨数据集、跨更新轨迹
    都可校准的质量状态。Fine QK 的 MeanRank、rank-utility 与 NDCG 区间仍包含 0。

---

## 7. 下一步最小研究计划

terminal 优化、连续区间 gate、流式训练价值链、逐轴规模、数据/模型组合规模、top-50k
chunked-data 以及 aligned method gate 均已完成。QK cheap 通过四-seed gate，QB 与 aligned
suffix 均已按规则停止，当前不继续投入任意层动态选择或继续盲目扩大 catalog。下一步是：

1. **进入系统成本。** 测量额外状态读取、host-device transfer、allocator、端到端 latency、
   吞吐和显存；profiling 确认后再做 `Wk/Wv` kernel fusion。
2. **把自然 cache-version 分布纳入评估。** 当前 theta-0→theta-5 是可控 stress test；下一步
   应在可辨识的 KuaiRand/QB/QK 设置上比较 cohort batching、periodic full、age threshold
   与 update-aware trigger 的等质量/等成本策略。
3. **动态的是维护时机，不是重新搜索任意层。** 先冻结 deepest suffix 候选，用小型
   held-out probe 为每个 model-version cohort 选择 reuse/cheap/suffix/full；除非新证据显示
   最优区间稳定移动，否则不恢复任意层搜索，更不做 per-user JVP 决策。

---

## 8. 建议的导师汇报顺序

1. **一张图讲问题：** 模型持续更新，旧版本用户 KV 在输入不变时也整体失效。
2. **一张图讲 motivation：** 单步通常较弱，累计后会出现平台、局部跃升与反转；age 只能
   粗排风险，说明 reuse 与 recompute 之间既有迁移空间，也需要 update-aware 触发。
3. **一张结构图讲 design：** 将 K/V 变化拆成 projection change 与 hidden propagation，得到
   cheap refresh + full interval。
4. **一张 Pareto 图讲结果：** 横轴 measured compute，纵轴 quality recovery，只标 cheap、
   suffix-2/4/5 与 full。
5. **一张小表讲规模：** 3/6/9 层都保留曲线；top-50k chunked 让有效 base target 增至
   62.1 万，并得到 0.058x cheap 与 0.613x suffix-4 的更强四-seed结果。
6. **最后主动讲边界：** 三数据集一致的是 motivation；方法上只有 cheap refresh 已在
   aligned QK 形成四-seed 正结果，完整 suffix 曲线仍只在 KuaiRand 强。QB/QK 来自同一
   collection，Tenrec 无全局日历时间，QB 还使用完整活跃 horizon 条件。ZhihuRec 为负，
   系统数据移动尚未计入。请导师重点判断结构化迁移抽象、混合 cache-age 系统实验，以及
   是否应把 “cheap refresh” 而非固定 suffix 深度作为更稳健的跨数据集贡献。

汇报时最核心的一句话可以是：

> 我们并不是预测哪些旧缓存已经坏掉，而是利用 HSTU 生成 K/V 的层级结构，以不同计算预算
> 直接把旧版本缓存迁移到新版本；当前结果表明版本损失会累积但并非匀速，固定年龄无法稳定
> 标定风险，同时结构化迁移已经形成一条初步的成本—质量前沿。
