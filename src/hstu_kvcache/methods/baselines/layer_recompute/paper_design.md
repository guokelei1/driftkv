# DroidSpeak：原论文核心 Design 笔记

整理日期：2026-09-16。中文概述，非逐字翻译；只保留设计，不含背景、实验或本仓库的适配方案。

来源：Yuhan Liu 等，*DroidSpeak: KV Cache Sharing Across Fine-tuned Model Variants*，NSDI 2026。[正式 PDF](https://www.usenix.org/system/files/nsdi26-liu-yuhan.pdf)，§4.1–4.4（印刷 pp.324–326）、Appendix C（p.338）。本仓库方案另见 [README](README.md)。

## 状态与执行边界（§4.1，Fig.6–7）

Sender 提供历史 K/V 和层输入 E；receiver 重算选定的连续层区间，复用区间外 K/V。由复用切到重算时，需要整段历史的入口 E，不能仅靠当前 token 的 hidden。连续区间减少入口切换、E 传输及边界误差；重算仍可能继承 sender E 的误差。

## 离线选配置（§4.2）

每个模型对在任务相关数据上 profile 连续区间，记录重算层数与输出质量，保留 Pareto 配置。根据可接受质量下降选较省计算的运行点。论文给出 O(L²) profiling，并允许多层成组降低搜索成本；没有规定遍历顺序、贪心算法或平局规则。

## 在线执行（§4.3–4.4，Fig.8–9）

1. 从离线配置选择重算区间。
2. 优先获取区间入口 E，开始 receiver 部分 prefill。
3. 同时传输复用层的 K/V，避免传输将被重算的 K/V。
4. 重算与加载完成后，用组合缓存继续 receiver 推理。

公开接口为 `store(Cache, context, LLM)`、`fetch(context, LLM, layer_id)` 和 `partial_prefill(recompute_config, context)`；按层存取 E/KV。通信与计算使用不同 CUDA streams，使 K/V 加载与重算重叠；入口 E 是开始重算的依赖。

## 运行点调整（Appendix C）

有排队请求时，选择达到质量目标的最少重算配置；无排队时，根据 token 数预测 prefill 延迟，选择满足 SLO 的最高重算比例。预测参数来自 profiling，不在线重新搜索层集合。

原文假设入口 E 可获取，但未展开动态更换入口时的多 E 存储管理；不能把单入口存储量解释成所有配置的总成本。
