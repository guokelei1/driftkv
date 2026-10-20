# 部署人口、拟合人口与评价角色

本页只清点真实文件与已有角色，不读取未来质量结果，也不启动训练。人口数来自各数据集的 `users.parquet` 去重UID，不能把实际存在的用户数量写成已经完成的方法评价数量。

| 条件 | 实际部署用户池 | 当前方法质量评价入口 | 当前拟合/验证 |
|---|---:|---|---|
| Medium | 30,000 | 固定4,096开发用户；历史fit/val排除后的可评价池为28,720 | 每次发布64 fit＋16 validation；原冻结fit池256 |
| Large | 79,681 | 尚无当前Large corrector评价；旧单边3,000面板属于开发材料 | 尚未定义新的跨版corrector校准面板 |
| Max | 200,000 | 尚无当前Max corrector评价；旧单边3,000面板属于开发材料 | 尚未定义新的跨版corrector校准面板 |

Medium用户池是Large的子集，Large不是Max的子集；Max使用另一个5B数据配置。Large/Max现有V2/V5测量只覆盖原生算子，不能据此声称已做其完整纠正实验。

Medium的28,720来自30,000减去历史D1/D2三次发布fit/validation角色并集1,280，而非只减当前64＋16。固定4,096属于开发评价，不是未经观察的确认组。两份已完成1,024确认都选自Large-only人口并使用Medium模型和映射，不是Large模型实验，也不能重新包装成新方案的盲测。

按最后保留的严格角色排除快照，再排除已揭晓的item-confirm1,024，剩余候选UID数为Medium **0**、Large **39,064**、Max **141,842**。这是该严格规则下的角色集合余量；其中规则排除了以前使用过的各种人口，因此“Medium无严格新用户”不等于28,720开发用户不能评价。Large/Max余量尚未成为冻结面板，还需按各自映射和发布前历史确认可用性，不能直接当质量分母。

## 最小拟合规模探索

先固定Medium的原V3来源、原origin、16名验证用户及同一开发评价组。实际V4的fit64严格等于原冻结fit256的有序前缀，因此16、32、64、128、256使用嵌套身份；64原结果复用。授权后的本轮V4单边16/32/128/256探索均已完成，逐点训练计量见 `results/system_study_2026_10/calibration/fit_size_v4/combined_curve/training.json`。这不是五套新的连续发布链，当前主时间账仍使用原64配方。

原 `source_v4` 只覆盖64 fit＋16 validation，共80用户。本轮按同一冻结V3方法额外因果捕获192名原fit身份，形成144/272行的128/256来源包；原80行未重收，证据位于 `results/system_study_2026_10/calibration/fit_size_v4/source_extension_attempt1/`。不能换验证用户，也不能从评价池挑“好训”的人补入。固定4,096之外的新评价角色仍须独立冻结。

固定local8/joint4、batch8时，增加fit人数也增加优化步数；这首先是“原配方人数—开销—质量”曲线，不是纯数据多样性的因果实验，更不能据此证明64已经饱和。只在需要解释明显差异时，补一个固定更新步数的对照，避免同时展开人数、epoch、目标函数的大搜索。

当前64/30,000约0.213%，只是已试的数值点，并不是应按人口固定比例拟合的证据。部署用户数增大而模型、目录和分布不变时，可以先检验固定64；模型容量、物品域或用户分布变化时，校准需求可能增加。后续Large/Max应各自先建立固定角色和origin，再从64开始按同样步骤检验，不能向一亿用户推出无证据的常数或比例结论。

## 真实入口

- Medium人口：`data/processed/yambda500m_unified_v1/scales/medium/users.parquet`；Large人口：`data/processed/yambda500m_unified_v1/scales/large/users.parquet`；Max人口：`data/processed/yambda5b_max_200k_v1/scales/max/users.parquet`。
- 当前有序fit256/validation16：`results/design_two_2026_10/benchmark_panel/users.json`；当前开发4,096：`results/design_system_2026_10/phase2/representative4096/panel/users.json`。
- 28,720与历史角色排除依据：`results/design_system_2026_10/phase2/representative4096/panel/selection.json`，并已读取其六个原始排除来源核对并集1,280。
- 真实V4 fit64/val16：`results/query_relay_2026_10/probes/A_itemgate6_rms_roll4/calibration/v4/calibration.json`；现成来源80用户：同候选 `source_v4/summary.json`。
- 严格排除快照：`results/query_relay_2026_10/confirmation_item/panel/excluded_roles.json`；补排该目录 `users.json` 的已揭晓1,024。
- Large/Max旧单边开发角色：`results/selective_recompute_2026_09/panels/{large,max}/*/users.json`；旧read-correction保留拟合身份：`results/read_correction_2026_09/v2/panels/{large,max}/*/calibration_users.json`。这些旧角色不自动成为新实验队列。
