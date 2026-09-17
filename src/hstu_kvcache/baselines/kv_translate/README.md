# KT：跨层闭式 K/V 翻译

日期：2026-09-16。状态：源层选择、闭式拟合和直接 KV 映射已实现，通过 CPU 数值检查；正式质量／成本评价未开展。共同执行与评价口径见 [上级说明](../README.md)。

[原论文核心 Design 笔记](paper_design.md) 单独保存闭式 mapper 的方法与公式；本文记录本仓库的 HSTU 适配设计。

## 当前接口

实现位于 [core.py](core.py)，直接接收已对齐、无 padding 的同构 K/V：

```python
from hstu_kvcache.baselines.kv_translate import fit

mapper = fit(source_cache, target_cache, num_heads=6, k=2, ridge=0.01)
translated = mapper.apply(live_cache)
```

`fit_affine_ridge` 可独立检查公式；`select_source_layers` 返回源层集合、逐源／目标层分数和排名口径。`fit` 可额外传 `selection_source`／`selection_target`，用这对缓存给源层探针打分；不传时明确记录为 `in_sample`，传入时记录为 `provided_validation`，但不自动保证 UID 留出。

拟合和 OLS 探针在输入设备上用 FP64 求解，再把参数转回源缓存 dtype；`apply` 让参数跟随输入缓存的设备和 dtype。它从同一旧快照生成全部目标层，不原地覆写输入。第一版只验证 CPU 路径，未验证 GPU 求解器／混合精度。

缓存采集、UID 划分和版本轨迹由调用方管理。当前 [CPU 测试](../../../../tests/test_baseline_kv_translate.py) 覆盖独立最小二乘参考、相关特征、跨层跨 head、K/V 独立性和两次映射的状态传递。以下连续校准和完整成本等内容仍是后续实验设计。

## 论文核验与采用范围

已核验 arXiv ID、题名及作者：Taekyung Heo 等，*Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse*，arXiv:2608.03893v1，2026-08-04。[论文元数据](https://arxiv.org/abs/2608.03893v1)，[正文 §3](https://arxiv.org/html/2608.03893v1#S3)。

本次未找到可确认的作者官方代码，设计依据是论文正文和附录，不把第三方独立实现当成作者代码。

采用三项核心：按目标层选择 top-k 源层；拼接所选层的全部源 heads；对 K/V 分开做带 bias 的闭式 ridge 映射。论文中 per-head 指输出参数按目标 head 独立，**不是只能读取同编号源 head**。K 不与 V 混合作输入，两者独立求解。依据为 §3.1–3.2、Eq.(3)–(5)。

论文 §3.3 先去掉源 K 的 RoPE、在内容空间映射、再加目标 RoPE。当前 HSTU 无此旋转，因此这一步不适用；时间编码是模型输入的一部分，不能类比 RoPE 从缓存中直接减掉。论文的 LLM benchmark、MLP 扩展和异构模型尺寸不进入首版。

以下校准划分、连续生命周期和具体开发候选是我们的适配设计，不能归因于原论文。尤其论文 §4.1 的部分 benchmark 参与 k 选择，本仓库改为独立 development 选择、final evaluation 留出。

## 映射对象与形式

令层数 `L=6`，KV heads 数为 H，head dimension 为 d，`D=H×d`。参数按获准发布边 `e: p→c` 共享，推荐 backbone 冻结。每个目标层 l 有一个源层集合 `S_l`，大小 k。

对同一个用户、同一个真实 token i，拼接旧缓存中所选层的全部 heads：

\[
X_K^{l}[i]=\mathop{\mathrm{concat}}_{s\in S_l}K_{\mathrm{in}}^{s}[i]
\in\mathbb R^{kD},\qquad
X_V^{l}[i]=\mathop{\mathrm{concat}}_{s\in S_l}V_{\mathrm{in}}^{s}[i].
\]

对每个目标 head h：

\[
\widehat K^{l,h}[i]=X_K^l[i]W_K^{l,h}+b_K^{l,h},\qquad
\widehat V^{l,h}[i]=X_V^l[i]W_V^{l,h}+b_V^{l,h},
\]

其中 `W` 的形状为 `[kD,d]`，bias 为 `[d]`。同一目标层所有 heads 共享源层集合，但无参数共享约束；实现可把所有目标 heads 的输出拼成 D 维，一次多输出线性求解／矩阵乘法，代数上等价，不退化为 head 内的小矩阵。

同层 `K_old W+b` 可作最简单的诊断对照，但正式设计必须包含源层选择和跨 head 输入，否则丢掉了论文的重要结构。

## 成对数据、源层选择和求解

`paired_calibration` 只读取发布时已可获得的 fitting 用户历史。source 与 target 必须在相同 UID、事件位置、因果前缀、item mapping 和时间输入上对齐，屏蔽 padding，不混入候选 query 的临时 K/V。

单边实现 canary 可用干净 Parent Full／Current Full 配对。连续主线应以 calibration 用户上本方法实际回放至发布前的缓存为 X、同一保留历史的 Current Full 为 Y，涵盖已有映射与 native append 的混合输入；不得每轮把实际 X 偷换成干净 Parent Full。这个 lineage-aware 校准是 HSTU 生命周期适配，不是原文默认设置；记录源轨迹的准备成本。

版本按因果顺序推进：后续边的 fitting X 由已冻结的先前 mapper／配置回放得到。若开发中修改早期 k、λ 或 mapper，受影响的后续 fitting／development 轨迹必须重新生成，后续映射重新拟合；不能沿用旧轨迹并声称评估的是新配置的连续行为。后续发布结果不用于回调早期已冻结策略。

`source_selector` 的初版步骤：

1. 在 fitting UID 的一个子集拟合逐 `(源层,目标层,对应head,K/V)` 的单源仿射探针；对 fitting 内另一个 UID 子集计算 R²。按 UID 划分，避免同用户相邻 token 同时出现在探针拟合与检查侧。
2. 目标层对每个源层取 K/V、head 的平均 R²，选 top-k，并由该目标层所有 K/V heads 共用集合。单源对应 head 探针与最终跨 head mapper 的输入维度不同；论文 §2.3 的结构探针和 §3 的实际 mapper 也须分开理解。
3. 用 fitting 用户的成对数据拟合最终多源 ridge，随后仅在独立 development 用户上选 k／ridge 等参数。后续最小 k 候选为 `{1,2,4,6}`；先实现 k=1 与 k=2 检查增量价值，再决定完整开发表。首版不做用户级或 token 级选层。

论文采用 centered ridge：对某个目标层的一类缓存，收集 M 个有效 token 后，令 `X_c=X-mean(X)`、`Y_c=Y-mean(Y)`，求解

\[
(X_c^\top X_c+\lambda I)W=X_c^\top Y_c,
\qquad b=\bar Y-\bar XW.
\]

首个公式对照沿用论文 `λ=0.01` 和未按 M 归一的 Gram 定义，不对 bias 正则化；不把这个数值称为已适合 HSTU 的最优值。若开发时改用平均损失、特征标准化或其他 λ，必须记录，不能在改变样本量／尺度后仍暗示正则强度完全相同。实现用线性方程求解，不显式形成逆矩阵；很小的求解参考可用 FP64。

原论文 §3.1 的校准是 500×1024 token 序列再 stride-4 采样，不直接照搬该数据量或 LLM fitting 耗时。HSTU 第一轮只用极小因果开发样本估计成本与数值，再决定 calibration 规模。

## 为什么成本可能低，以及必须支付什么

发布后执行的是 token-wise 仿射映射，不重跑历史 embedding、query、attention 或 gate；在固定 L、k、D 下，转换随历史长度 N 线性增长。它仍会扫描并重写整个保留 K/V，不是常数大小的摘要更新。

对相同宽度的六层模型，映射参数主项为 `2LkD²`，单用户转换 MAC 主项为 `2NLkD²`，bias 另计。以既有 `D=192,L=6` 为例：

| k | 斜率参数数目 | FP32 斜率存储 | N=1024 时转换 MAC |
| --- | ---: | ---: | ---: |
| 1 | 442,368 | 1.69 MiB | 452,984,832 |
| 2 | 884,736 | 3.38 MiB | 905,969,664 |
| 6 | 2,654,208 | 10.125 MiB | 2,717,908,992 |

另有 `2LD=2304` 个 bias。这是本配置的公式计算，不是测得的加速；k 增大时可能不再比重算便宜，需开发成本表确认。不能误用只允许同 head 输入的 `2LkHd²` 公式低估本方案工作。

若 `F=kD`，每个目标层、每类 K/V 的拟合包括 Gram `O(MF²)`、交叉项 `O(MFD)`、分解 `O(F³)` 和多输出回代 `O(F²D)`。同层所有目标 heads 共用一次 Gram 和分解，K 与 V 不共用 Gram。小样本先直接求解；样本扩大后才考虑分块累计充分统计量，不先建设分布式拟合系统。

每用户至少读取所需源 K/V，并写出 `2LNDs` 字节目标 K/V，s 为每元素字节数；多目标层重复读取、拼接临时张量、CPU／GPU 搬运和峰值内存照实计量。源／teacher 计算、top-k 探针、ridge 拟合也属于完整成本，闭式不表示没有校准费用。

## 一次发布与多次发布

1. 发布前完成该版本边的 source selector 和 mapper，记录拟合／开发 UID、模型配置、k、λ、精度和监督来源。
2. 读取用户**发布前**真实缓存作为稳定输入。所有目标层都从同一旧快照取特征，再安装映射结果；不能先原地覆写某层再给后续目标层当“源层”。后续可按 token 分块控制内存，但块内必须先保留所需全部源层。
3. 所有保留层、所有有效 token 都映射一次。它们的表示目标变为 c，但原写入来源和近似 lineage 不因此消失；首版不按原 producer 拆成多套 mapper。
4. Current query 直接消费映射 K/V；后续真实新增 token 由 Current 在此状态上 native append，不额外套 mapper。淘汰随正常窗口执行。
5. 下一次发布用当时实际缓存再映射，包含已经映射过的旧行及后来追加的新行。保留多轮误差，不从最早 native source 重建后直达新目标，也不保留一份免费的 exact-source 旁路。

论文 §4.6 的 multi-turn handoff 是参考动机；本仓库仍须验证发布间真实 appends、evictions、长期旧状态及继承误差。单边高 R² 不证明多发布推荐质量好。

## 计划模块与最小验证

| 计划职责 | 内容 |
| --- | --- |
| 成对数据 | 当前由调用方提供；没有数据集采集／UID 划分器 |
| 源层选择 | 已有 `select_source_layers`，单源 OLS 探针和逐目标层 top-k |
| 闭式 mapper | 已有 `fit_affine_ridge`／`fit`，K/V 独立 centered ridge 与跨 head 多输出 |
| 发布转换 | 已有 `KVTranslator.apply` 的快照变换；模型准入、安装及完整 lineage／成本由后续调用方负责 |

输出复用 [HSTUKVCache](../../models/kv_cache.py)，评分复用 [HSTU.score_cc_reuse](../../models/hstu.py)。现有 [adaptation/translator.py](../../adaptation/translator.py) 的摘要／响应译者不是本方法的逐 token KV mapper，不能直接换名视为已经实现。

后续 CPU 检查先覆盖：已知仿射数据的 centered ridge 对照；合成跨层／跨 head 依赖能被指定特征表达；各目标层输出均来自旧快照；真实同版本缓存的近 identity 行为及一次 native append。**λ>0 的拟合有收缩，不能把“同版本 ridge 必须逐元素等于 identity”写成错误断言**；精确恒等控制用显式 identity 权重。

再用一条两次发布、append、evict、query 的小轨迹检查实际输入传递。大范围质量、最优 k、正则强度和 GPU 节省仍是待验证问题；不把线性映射天然有效或天然廉价当作设计前提。
