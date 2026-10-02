# KT：保留的跨层闭式 K/V 映射原语

本模块提供源层选择、闭式 ridge 拟合和逐 token K/V 映射。
它是历史对比及后续研究可复用的原语，**不属于当前四种局部重算基线**，
也不等于当前论文 Design 的共享翻译与状态更新。
当前实验入口见[上级说明](../README.md)；旧三规模比较证据见
[历史来源账本](../../../../figures/out/three_scale_auc_preview/cost_ledger.json)。

## 接口与输入

实现位于 [core.py](core.py)，输入为事件位置已对齐、无 padding 的同构 K/V：

```python
from hstu_kvcache.baselines.kv_translate import fit

mapper = fit(source_cache, target_cache, num_heads=6, k=2, ridge=0.01)
translated = mapper.apply(live_cache)
```

`fit_affine_ridge` 提供独立公式接口；`select_source_layers` 返回每个目标层
的源层集合、评分与排名口径。`fit` 可传入 `selection_source/selection_target`
作选择验证数据；缺省是 `in_sample`，传入是 `provided_validation`。
这只记录输入方式，不自动保证 UID 留出，数据因果与划分由调用方负责。

拟合与 OLS 探针在输入设备上用 FP64 求解，参数再转回缓存 dtype；
`apply` 让参数跟随输入设备与 dtype，全部目标层均从同一旧快照产生，
不原地覆盖后再充当其他目标层的输入。

## 映射与源层选择

对每个目标层 l，选 k 个源层，将每个有效历史位置 i 的全部源 head 拼接：

- `X_K[i] = concat(K_source[s][i] for s in S_l)`，维数 kD；
- `X_V[i] = concat(V_source[s][i] for s in S_l)`；
- `K_hat[l,h][i] = X_K[i] W_K[l,h] + b_K[l,h]`，V 同理。

K 和 V 独立拟合；单个目标 head 可以读取全部所选源 head，
不是仅同编号 head 映射。同一目标层所有输出 head 可合并成一次多输出求解。

源层选择使用单源仿射探针的平均 R²，再取每个目标层 top-k。
历史适配方案在 fitting UID 内划分拟合/选择组，最终 mapper 再用校准数据拟合；
实际旧实验的具体 UID 与 k 以各自结果设置为准，不以接口说明补写过去实验。

Centered ridge 求解：

```text
Xc = X - mean(X); Yc = Y - mean(Y)
(Xc.T @ Xc + lambda*I) W = Xc.T @ Yc
b = mean(Y) - mean(X) @ W
```

不惩罚 bias，不显式计算逆矩阵。这里 Gram 没有按样本数归一，
因此不能将 lambda=0.01 与平均损失定义下的同名值视为相同正则强度。
lambda>0 会收缩，即使同版本拟合也不保证逐元素 identity；
精确恒等控制应使用显式恒等权重。

## 工作量与状态边界

固定 L、k、宽度 D，斜率参数主项为 `2LkD²`，
单用户 N 个历史位置的转换 MAC 主项为 `2NLkD²`，另计 bias。

| 六层、D192 | 斜率参数 | FP32 斜率存储 | N1024 转换 MAC |
| --- | ---: | ---: | ---: |
| k=1 | 442,368 | 1.69 MiB | 452,984,832 |
| k=2 | 884,736 | 3.38 MiB | 905,969,664 |
| k=6 | 2,654,208 | 10.125 MiB | 2,717,908,992 |

另有 2LD=2304 个 bias。表中是公式计算，不是测得的加速。
以 F=kD，拟合包含 Gram O(MF²)、交叉项 O(MFD)、分解 O(F³)、
回代 O(F²D)；K/V 不共用 Gram。完整实验还应计入教师与源缓存、
源层选择、拟合、扫描与写回、临时拼接和数据搬运。

`apply` 只做快照转换，不提供发布准入、服务安装或完整 producer 谱系。
若后续用于连续发布，输入必须是本方法真实保留、追加和淘汰后的状态，
而不能每条边重新构造干净 Parent 缓存；现存单边结果不证明该连续行为。

## 来源与数值检查

[来源笔记](paper_design.md)记录 *Cross-Model KV Cache Transfer in LLM Families:
A Closed-Form Linear Mapping for Prefill Reuse* 的方法及原公式。
HSTU 无 RoPE，不能将其时间输入当作可从缓存减去的旋转编码；
这里不宣称复刻原 LLM 系统。

[公式与状态测试](../../../../tests/test_baseline_kv_translate.py)覆盖独立最小二乘参考、
相关特征、跨层/跨 head、K/V 独立及多次映射的输入传递。
[adaptation/translator.py](../../adaptation/translator.py) 是另一个历史摘要/响应接口，
不可与此逐 token mapper 混称为已实现当前 Design。
