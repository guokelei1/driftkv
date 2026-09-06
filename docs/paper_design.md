# EvoKV：Cross-Version Cache Adaptation

更新日期：2026-09-06

当前论文正文位于 [paper/main.tex](../../paper/main.tex)。本文件只保留实现和实验需要的
设计映射，不再复制一份完整论文稿。当前方法尚未实现或取得方法实验结果。

## 问题与设计依据

模型通过使用新鲜历史状态的 Full-only admission 后，旧 producer 生成的 K/V 仍可能损失
一部分模型更新收益。EvoKV 关注缓存适配，不改变上游模型准入，也不通过迁移结果挑选发布版本。

Insight 1 的低覆盖局部替换不足以稳定恢复主要差距，因此保留逐事件 K/V。
Insight 2 显示历史聚合处可用紧凑响应修正改善预测一致性，但冻结的发布时 offset
不能持续应对新请求和历史变化。这两项观察引出写入预计算、共享版本转换和按请求读取修正。
95.34%/99.46% 等 oracle 数字属于诊断，不是 EvoKV 方法效果。

## 与论文第 4 章的对应关系

| 小节 | 职责 |
| --- | --- |
| 4.1 System Overview | 普通缓存提供详细历史；附加路径预测并应用版本变化 |
| 4.2 Cache Summarization | 随普通 K/V 写入 source summary，固定 segment/slot 对应关系 |
| 4.3 Release-Time Translation | 冻结推荐模型，在独立 calibration users 上学习共享转换 |
| 4.4 Read-Time Correction | 同一 query 读取 source/translated 表示，将差加入历史聚合 |
| 4.5 State Maintenance Across Releases | 处理追加、淘汰、混合 producer、目标切换与版本退出 |

参考配置每段最多 64 个事件、每层两个 slots。边界对齐的 1,024-event 历史有 32 slots；
未满段与提前封段的额外容量另计。writer 使用实际存储的 K/V 累加，淘汰时扣除同一贡献。
摘要不会添加普通 K/V 本来没有的信息，其价值是提前准备发布时可快速处理的输入。

translator 使用同用户待迁移状态的跨层、跨段上下文，输出 K/V 残差。
事件数量、时间、位置及有效性由维护路径保留，不由网络预测。
训练同时约束目标表示和 reader 响应差；query 沿逐层修正的实际执行路径产生。
共享方法使用 calibration 教师，在独立用户上评价；逐用户目标拟合可作标明用途与成本的诊断/对照，
不能同时称为这些用户上的未见泛化。用户已撤销笼统的 target-KV fitting 禁令，
摘要重建和其他 K/V-derived 监督按完整流水线的质量、成本与可执行性选择。

reader 在原模型聚合位置使用修正。若 attention 跨历史位置归一化，需要同时修正加权值和
归一化总权重，不能直接相加独立归一化的 segment 输出。
具体 HSTU 算子映射属于实现和实验协议，不定义论文的一般概念。

## 持续状态管理

每段保留真实 producer 与原始 source 表示。新版本始终从该 source 直接转换，
不把上一轮预测继续作为下一轮输入。这里避免的是预测摘要的串联，不代表新追加 K/V
不再继承修正上下文的误差；完整连续轨迹仍需验证。

Translator 实际读取的旧 source 内容变化后，先刷新同用户剩余旧段的输出再做修正读取，
因为转换使用跨段上下文。仅追加未进入译者输入的 current 段不触发旧段重翻译。
请求只使用模型、source revision 和 translated revision 匹配的状态；
未完成刷新或未覆盖 producer 的用户走 Reuse，并计入覆盖率和质量报告。
被拒绝的模型不改变当前服务版本，过期目标的迟到结果不提交。

## 成本、资产与下一步

一次发布的计算预算目标为同人口 Exact-All 的 20% 以内，包含教师、转换网络训练、
必要的初次摘要构建和全人口转换。在线维护、读取、刷新、I/O 和存储另行计量，并在固定
服务时段内比较总成本。32/1024 配置约 6.25% 只指额外 QK/值聚合算术，不是延迟预测。
旧 0.60% 翻译和约 7% 存储数字不再作为当前成本结论。

已具备六层/十层模型、Full/Reuse 评估与诊断 reader；摘要 writer、translator、
生产读取路径和连续发布执行器仍待实现。六层四组件原型、小规模共享校准及连续状态开发按
[design/plan.md](design/plan.md)推进，技术定义见 [experimental_design.md](experimental_design.md)。
首版须检验低计算、实质恢复的方向；长训练与正式人口评价保留资源、canary 和明确启动要求。
