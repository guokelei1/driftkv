# 新 Query correction 与跨版本码接续

本目录独立保留当前探索的配置、结果和过程。当前冻结Current item门控A的完整4096人与另一组未参与调参的1024名新用户均支持主任务质量/成本与动态门控收益；独立静态的开发及新用户配对、原反馈总体边界和冷用户分组均完成。早期layer-query A未支持动态排序收益，作为完整对照保留。请求频率、教师误差与内存边界分别报告。

| 路径 | 内容与用途 |
|---|---|
| `cost/event_ledger.json`、`cost/prefilter.json` | 原反馈请求协议的完整成本预筛，含纠正写入 |
| `workloads/registration.json` | 在测量新质量前登记的行为触发频率与候选数 |
| `workloads/cost_sensitivity.json` | 六种新工作负载的成本敏感性；历史校准成本仅作参考 |
| `canary/gate_v3/` | 首次运行因错误选择仅含评估用户的历史 panel 而中止；无权重或质量结果 |
| `canary/gate_v3_retry/` | 4 fit＋2 validation 的 V3 GPU 门控／执行检查，不作为质量证据 |
| `probes/B_r18_v3/` | 原反馈4096人V3：约70.79%恢复，完整18维读取写入成本过高 |
| `probes/B_r6_v3/`、`B_balanced6_rms_roll4/` | 固定原生摘要的低读宽分支；128人首版约82%–84%，动态门控未改善AUC |
| `probes/C_balanced6_rms_roll4/` | 非线性递推表达能力对照；不能直接合成冷用户跨代更新 |
| `probes/A_origin6_rms_roll4/` | 早期layer-query可合成对照；三版校准/全目录发布齐全，恢复率/成本通过但动态排序收益不利 |
| `probes/A_origin6_rms_roll4/fullsamehistory/merged/` | 全4096人同历史只读门控消融；centered评分MSE相对Rebuild增量参考改善28.2959%，95%[20.2788,37.2931]；pooled/pair AUC均显著不利，不构成排序或优于KV映射的证明 |
| `probes/A_origin6_rms_roll4/same_history128/` | 早期128人响应诊断与16个validation用户发布点精确Current教师诊断；后者centered评分MSE略退化，与完整增量轨迹参考分开保留 |
| `probes/B_origin6_rms_roll4/` | 额外原生摘要乘性交互分支；128人三版约95.86%，主成本20.0515%，未选 |
| `probes/A_localorigin6_rms_roll4/` | 低成本初始化分支：原反馈4096人三版40.27%恢复，19.96%成本；质量失败 |
| `cost/current_candidate_budget.json`、`.md` | 真实校准stage与全目录编译、固定30k服务账本和工作负载边界 |
| `progress.json` | 全部已完成质量探针的简洁过程索引 |
| `probes/A_origin6_rms_roll4/freeze.json` | 早期A权重/发布绑定与设置冻结；完整开发人口和两类旧确认已完成，不再用于新Item调参 |
| `probes/A_origin6_rms_roll4/population4096/merged/` | 完整开发人口：99.1171%恢复，UID95%[96.6558,101.7015]%；动态门控比fit均值门控略差 |
| `confirmation/panel/verification.json`、`confirmation/behavior1024/` | 排除全部旧实验角色的新用户迁移确认：106.3237%恢复；使用Medium模型与固定映射 |
| `confirmation/feedback1024/` | 新用户原反馈确认：参考增益不稳定，80%恢复未获支持；不能沿用新任务结论 |
| `probes/A_firstgate6_rms_roll4/` | 第一层query门控：三版128人94.3450%，与主A持平，动态收益不确定 |
| `probes/query_components_v3/` | 仅fit/validation pre-cut的Query组成及时间间隔诊断；公共type/action主导q0，不是质量实验 |
| `probes/A_itemgate6_rms_roll4/` | 当前冻结候选：Current物品表示门控、A码递推；正式compiled actual全4096完成，直接AUC增益+0.0105650，动态排序收益为正 |
| `probes/A_itemgate6_rms_roll4/same_history128_v3/`、`code_use128_v3/` | 同集合均值门控、零码/fit全局均值码控制：排序均有正收益；centered教师误差仍较差，不能宣称更准确还原教师 |
| `probes/A_constantgate6_rms_roll4/` | 独立训练恒1门控读/写控制：自己的来源与三版轨迹；正式4096恢复78.52%，95%[76.35,80.76] |
| `comparisons/item_vs_constant128/` | 同UID与请求/原生控组配对，动态减独立静态AUC为+0.01152463，95%[0.00843262,0.01585175]；仅开发小样本 |
| `comparisons/item_vs_constant4096/` | 全4096共同UID配对：AUC差+0.00914646，pair-AUC差+0.07638396，区间均为正；相对Rebuild的centered评分MSE仍较差 |
| `cost/item_and_static_actual_budget.json`、`.md` | 当前两冻结分支的完整实际校准/全目录发布成本，含实际求逆和item lookup流量 |
| `confirmation_item/panel/verification.json` | 另一组新1024人，排除全部开发/旧确认/保留校准角色；仅发布前信息分层选择 |
| `confirmation_item/behavior1024/` | 冻结当前Item的新用户迁移确认：方法减Rebuild AUC+0.00854451，95%[0.00767799,0.00956600]，动态减fit均值也为正 |
| `probes/A_itemgate6_rms_roll4/population4096/merged_actual/cold_boundaries.json` | 首次任意consumer的缺席发布数与读取长度分组；有限三版描述点值，无分组CI |
| `probes/A_itemgate6_rms_roll4/feedback4096_actual/merged/` | 完整原反馈4096：恢复65.89%，80%非劣未获支持，完整成本23.7671%；主任务结果不可替代该负边界 |
| `confirmation_item/static1024_restart.json`、`static1024_interrupted_reboot/` | 原静态确认因服务器重启中断的留档；只补齐尚未完成的冻结静态评分 |
| `confirmation_item/static1024/`、`comparisons/item_vs_constant_confirmation1024/` | 冻结静态确认完整1024已合并；动态减独立静态AUC +0.00887098，共同UID95%为正；排序与centered教师误差分别保留 |
| `formal_aggregation/final_completion_evidence.json`、`.md` | 最终完成证据、范围、角色/冻结/成本绑定、负边界及恢复后70项CPU检查 |
| `numerical_stress/` | 显式递推/合成、长连乘与发布级实际逆矩阵的数值边界；并非128次真实发布质量 |
| `report.md` | 当前机制、证据、负结果与有效范围 |
| `/home/gkl/work/paper/new4.md`、`paper/new4_before_query_relay.md` | 已验证机制的纯Design稿及替换前备份；设计稿不放实验结果 |

原反馈请求来自实际 feedback manifest。行为触发任务重新构造 next-item 正例与固定负采样候选，并计算新的原生控制，单独报告结果。成本分母始终为每用户每发布首次实际推荐请求重建一次 K/V，随后正常增量服务；所有用户即使没有推荐请求，其行为写入纠正成本仍要计入。

总体恢复率由各发布 AUC 差的等权均值计算；不平均各发布恢复百分比。校准、质量与部署成本人口分别说明。预训练的初始化、校准来源重建及完整 Query/写入开销不得隐藏。开发候选冻结之后才可进行独立确认。
