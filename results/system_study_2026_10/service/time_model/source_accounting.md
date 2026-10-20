# 来源状态复用与完整S的口径

当前时间预测保留的是**实际离线复现来源状态**的保守工作流。数学主账则采用 `source_available=true`，将来源重建列为敏感性；两者不是完全相同的部署情景，不能用同一个“完整成本”标签掩盖这一差别。此审计不修改已保存的fair128负结果，也不从S直接删除历史计时。

## 哪些来源工作可以被线上维护覆盖

逐一核对候选 `calibration/v3,v4,v5/configuration.json` 中的用户集合：三版使用相同64名fit与16名validation用户，这80人全部位于真实Medium 30k人口中。`source_v4` 与 `source_v5` 的配置均为behavior stride16、每请求16候选，与当前主轨迹一致；fit/validation仍不进入质量评价人口。

`capture.py` 为每个用户重新构造V2原生历史，然后重放目标版本之前的全部发布、行为及请求边界。候选评分不改变持久状态，因此捕获程序只执行请求前的状态准备，不重复评分。V4来源包含V3期演化；V5来源再次从V2起点演化到V5之前，包含此前两期。若同一部署已按该路径维护这80人的状态，这些来源历史replay不是每次校准必须重做的工作。

但可复用的对象须正确：训练输入是切换前的原生K/V和**上一版current码**。compiled服务应在旧版drain完成后、反锚前取样；已转成anchor的缓冲区不能直接充当previous_codes，需要使用旧发布参数恢复current。原捕获采用逐步显式更新，当前服务采用compiled路径，二者有代数对应但浮点实现不同；不能据此宣称新快照与旧pack逐位相同。

## 不能被来源复用免掉的工作

每次校准仍需要当前模型重新构建Fresh teacher、候选教师输出、当前物品特征、原生K/V及码的快照、必要搬运和输出保存。抽样若在交接后发生，还要付必要的码格式转换。

原 `source_snapshot()` 会无条件调用 `prepare_read()`。对上一发布期完全无行为、无推荐请求的校准用户，线上服务可能尚未物化旧当前码，取样本身需要做这一步。用保留历史与 `trigger_indices` 检查得到：

| 目标校准版本 | 上一发布期没有行为或请求的fit/val用户 |
|---|---|
| V4 | 832000、106980 |
| V5 | 207010、832000 |

因此“已有线上状态”不等于80人的样本准备成本为零。现有source pack已包含这些强制物化后的状态；直接读取pack计时若不补此项，只能声称“已准备样本的消费时间”。

进一步核对发现，原 `record_native_capture(source_available=True)` 将source额外FLOPs整体移入敏感性，其中也包含这2名idle用户仅因校准抽样而发生的码物化。完整live-source数学账应加回这部分必要更新与物品查表，不应把所有来源操作都一并删除。新 `maintained_source_probe.py` 已显式保留 `required_snapshot_operations`，目前仅通过CPU检查，尚无GPU计时；此前数学结果不据此静默重写。

## 历史计时为什么不能直接减掉88.52秒

| 来源包 | 整体elapsed | 教师重建FLOPs | 来源原生轨迹FLOPs | 来源纠正读取FLOPs |
|---|---:|---:|---:|---:|
| source_v4 | 33.801780秒 | 397,056,092,838 | 996,525,430,262 | 22,607,436,924 |
| source_v5 | 54.716555秒 | 394,649,943,660 | 1,709,649,108,434 | 48,130,281,792 |

这些FLOPs在原记录中可分，来源码更新另外由 `priced_source_capture()` 对保留形状计价。但elapsed从命令入口持续到source pack保存和hash完成，混合了模型/历史加载、来源replay、教师重建、物品特征、快照与CPU搬运；没有分别记录上述工作的秒数。不能按FLOPs比例分摊时间，也不能把33.80+54.72秒全部删掉。

后续live-source测量应直接消费已维护的来源，实际重建教师并执行样本准备，同时单列必要的额外物化；它可以作为新情景替代离线replay的对应部分。原S、只替V4校准的S和live-source的S需要分别保留，不能修改旧结果来形成优势。

## 可复核入口

- 人口：[medium_users.parquet](/home/gkl/work/evokv/data/manifests/yambda500m_scale_v1/medium_users.parquet)；角色分别见候选各版 `calibration/*/configuration.json`。
- [V4来源记录](/home/gkl/work/evokv/results/query_relay_2026_10/probes/A_itemgate6_rms_roll4/source_v4/summary.json)、[V5来源记录](/home/gkl/work/evokv/results/query_relay_2026_10/probes/A_itemgate6_rms_roll4/source_v5/summary.json)。
- [来源捕获流程](/home/gkl/work/evokv/scripts/query_relay_2026_10/capture.py:64)、[快照语义](/home/gkl/work/evokv/scripts/query_relay_2026_10/session.py:277)、[来源数学计价](/home/gkl/work/evokv/scripts/query_relay_2026_10/calibration_cost.py:193)。
- [历史S分项](/home/gkl/work/evokv/results/system_study_2026_10/service/shared_timing/attempt1/calibration_timing.json)、[保留的fair128完整预测](/home/gkl/work/evokv/results/system_study_2026_10/service/time_model/fair128_comparison/attempt1/summary.md)。

本次仅做CPU集合与轨迹边界检查，没有新增GPU实验或重新评价质量。
