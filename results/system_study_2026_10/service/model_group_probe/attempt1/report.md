# 原生模型分组时间探针

本轮完成 Medium、Large、Max 三条真实权重链的 V2/V5 六个端点，共 86 个算子形状，每项预热 2 次、测量 5 次。使用一张 A40（GPU0），同卡顺序执行，完成后已释放。这里只建立原生操作的实测时间系数，不比较 CacheTide 与重建的总体优劣。

## 输入与执行范围

- 模型实际结构来自 checkpoint：Medium 6层/192宽/6头，Large 10层/320宽/10头，Max 16层/320宽/10头；均 head32、context1024、legacy ELU+1。不使用 dataset.json 中过期的模型档位字段。
- 各规模使用自己的原始行为与 item mapping；历史严格截止 D245（V2）或 D287（V5），不读取未来评分标签。Medium 复用原32人的工程池；该池最早按D259分层，因此V2只是回溯算子采样，不是发布前锁定的质量人口。Large/Max固定最小UID前16人，要求首次行为早于D231。
- 每个端点只按发布前保留长度选最短/最长两人，先固定选择，再查看本版是否有16条后续行为；不足则跳过追加，不换用户。两端点的短/长长度分别为 Medium 13/1024、43/1024；Large 92/1024、92/1024；Max 21/1024、33/1024。
- B=1使用一条实际历史；B=8是这条历史平铺八份的合成算子批，不能称八个独立用户或已实现跨用户调度。候选为本地词表有效ID 1–16，仅是性能输入，不是next-item质量请求。
- 重建使用已有 compute_kv，追加使用 append_with_rolling_band；满1024前缀沿用Triton，短前缀沿用原Torch策略。Query分别测通用Torch和已有零码Triton融合核；H10零码核在本轮独立核对，不代表H10纠正器已适配。

## 满1024行的实测系数

以下为5次同步API墙钟的中位数，单位ms。每格为 **B1 / B8整批**，B8没有除以8。输入和模型已在GPU；包含CPU调用、算子发起与等待GPU完成，不含历史SQL、模型加载、输入搬运、队列、码准备或校准。

| 模型 | 端点 | 重建 | Query16融合 | Append16 |
| --- | --- | ---: | ---: | ---: |
| medium | v2 | 2.82 / 6.56 | 2.36 / 2.48 | 2.31 / 2.71 |
| medium | v5 | 2.33 / 6.57 | 2.31 / 2.51 | 2.34 / 3.09 |
| large | v2 | 3.63 / 18.25 | 3.57 / 3.75 | 3.60 / 4.61 |
| large | v5 | 4.81 / 19.10 | 3.56 / 3.72 | 3.63 / 4.60 |
| max | v2 | 5.43 / 29.29 | 7.18 / 5.58 | 13.04 / 7.13 |
| max | v5 | 7.08 / 29.00 | 5.39 / 7.56 | 5.35 / 7.03 |

满长B1的分析FLOPs（GFLOPs）如下，B8近似按8倍计；各单元精确整数在summary中。重建/追加沿用原执行形状模型，计入已有Triton tile补齐；query列为有用算术，不计融合核K64补齐及TF32x3机器指令展开。因此有效FLOPs与时间分别报告，不用GPU峰值TFLOPs换算时长。

| 模型 | 重建 | Query16 | Append16 |
| --- | ---: | ---: | ---: |
| medium | 5.0120 | 0.1143 | 0.2024 |
| large | 18.1011 | 0.3827 | 0.6275 |
| max | 28.8222 | 0.6101 | 1.0018 |

B8满长重建的峰值增量分配约150/410/650MiB（Medium/Large/Max），包含新输出和临时量；持久K/V输出分别为72/200/320MiB。它不是整卡峰值，也不包含已常驻模型；模型参数约0.99/2.67/4.24GiB。每次调用的before/after/peak均保留，query融合的中间量更小，但本轮不宣称所有形状更快。

短历史的完整结果也保留。例如Medium V2只有13行，B1重建仍约2.49ms，而满1024行约2.82ms；FLOPs较少未等比减少调用时间。批量执行对每条历史的摊销有明显影响，所以不能用一个统一“秒/FLOP”系数覆盖长短历史与B1/B8。

## 正确性和限制

- 零码融合query对同输入Torch参考的最大logit差为7.15e-7。B8对B1复制结果的K/V最大绝对差5.34e-5，均通过atol=3e-5、rtol=3e-5的逐元素检查；它是浮点容差一致，不是bitwise一致。
- 12个用户/端点样本中5个在所加载本版区间不足16条后续行为，因此只测其重建/query。不会把旧行为复制成未来追加。V5历史只加载至D288，是否有16条仅指这一天内的可用性。
- 五次连续重复不是交错多轮负载试验。部分小算子样本波动明显，例如Max V2 B1 Append16为8.12–14.17ms；保留每次墙钟、CUDA event时间与范围，不挑最小值当总体结果。GPU独占，但宿主机负载/时钟未锁定，版本间小差异不作统计结论。
- input_prepare_h2d_seconds记录每个B形状的一次前缀/query输入准备，在同批各操作行复写，仅作上下文，不能相加；append输入准备未单独计时。历史加载、模型加载、每项预热各自单列。
- 单次只加载一个模型，没有加载发布表或corrector。Large/Max没有本轮已训练的纠正器，所以没有跨模型恢复率、方法胜负或总体20%时间预算的结论。

## 如何用于目标规模

需要把发布周期的真实操作账按模型、版本、历史长度、候选宽度、追加宽度和实际批大小分组，以调用数乘对应实测API系数，再加入单独实测的共享准备、码转换和纠正读写。已有30k账中满1024行覆盖约95.6%的query调用，但短/长两例不能等权平均作为人口。旧账的query32需按实际服务query16拆成两次调用；FLOPs近线性，发起成本不会自动抵消。

B8系数只有在实际批处理路径能组成该批、并计入组批等待和搬运时才能用于总时间预测。上述加权先得到执行时间估计；真实到达下的墙钟完成时间还受排队、重叠和发布屏障影响，最终仍需规模回放验证。当前全局服务逐用户处理，因此B1是直接适用起点，B8是批量路径潜力。

## 证据与复现

- [summary.json](summary.json)：86个单元、逐次计时/内存、FLOPs及配对检查。
- [selection.json](selection.json)、[history_loads.json](history_loads.json)、[model_loads.json](model_loads.json)：用户选择、独立映射入口、真实结构与准备时间。
- [configuration.json](configuration.json)、[input_bindings.json](input_bindings.json)、[run.log](run.log)：设置、源码/权重绑定及各组完成状态。
- 代码：scripts/system_study_2026_10/model_group_probe.py；配置：configs/system_study_2026_10/model_group_probe.json；针对性CPU选择检查2项通过。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/system_study_2026_10/model_group_probe.py \
  --config configs/system_study_2026_10/model_group_probe.json \
  --output <new-output-directory> --gpu 0
```
