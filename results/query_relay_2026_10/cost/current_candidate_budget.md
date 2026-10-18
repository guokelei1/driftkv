# Query relay development budget audit

Theoretical FLOPs on actual fixed30k causal event shapes and observed calibration operations; no GPU timing or quality claim.

Primary: 16行为触发一次推荐、每轮16候选；4副本独立准备。原feedback和已注册频率敏感性保留各自的Full/Rebuild分母。

| Candidate | observed calibration TF | 16/16 added % | original feedback added % | 16/16 incl. source % | status |
| --- | ---: | ---: | ---: | ---: | --- |
| A_origin6_rms_roll4 | 14.761656 | 17.37908 | 23.76709 | 18.61370 | 校准三版完成 |
| B_origin6_rms_roll4 | 17.685661 | 20.05150 | 28.64039 | 21.30216 | 校准三版完成；未发布项按显式编译几何预测 |
| A_localorigin6_rms_roll4 | 7.090454 | 15.29241 | 19.96271 | 16.02738 | 校准三版完成；未发布项按显式编译几何预测 |

缺少校准版时，上表仅为已观察校准加完整服务开销的下界，不能称完整候选结果。重建源K/V的敏感性已含原始origin一次，不重复叠加。

- A_origin6_rms_roll4：主口径额外读开销占native完整query 9.079%；全局共享状态每副本 3.182 GB，4副本 12.726 GB，另加逐行48维码（native K/V的12.5%）。
  native历史＋self注意力的额外query开销为 13.499%。整个建模服务 Full=1007.020553 TF，Relay=703.283523 TF；净省 303.737030 TF，即Full服务的 30.162%。若源状态需重建，Relay=707.822324 TF，净省 299.198228 TF（29.711%）。
- B_origin6_rms_roll4：主口径额外读开销占native完整query 9.079%；全局共享状态每副本 6.363 GB，4副本 25.451 GB，另加逐行48维码（native K/V的12.5%）。
  native历史＋self注意力的额外query开销为 13.499%。整个建模服务 Full=1007.020553 TF，Relay=713.108045 TF；净省 293.912508 TF，即Full服务的 29.186%。若源状态需重建，Relay=717.705805 TF，净省 289.314747 TF（28.730%）。
- A_localorigin6_rms_roll4：主口径额外读开销占native完整query 9.079%；全局共享状态每副本 3.182 GB，4副本 12.726 GB，另加逐行48维码（native K/V的12.5%）。
  native历史＋self注意力的额外query开销为 13.499%。整个建模服务 Full=1007.020553 TF，Relay=695.612321 TF；净省 311.408232 TF，即Full服务的 30.924%。若源状态需重建，Relay=698.314286 TF，净省 308.706266 TF（30.655%）。

20%口径是全部新增成本/KV重建，不能解释成整个模型服务的FLOPs比例。整个建模服务包含三版期间真实历史追加、完整query及首次推荐KV重建；两条路径共同继承初始V2缓存，不含基础模型训练、检索基础设施或实际运行时间。

所有数值均为理论FLOPs；query增幅与总重建分母分别报告。全局D表和immutable bank不属于逐行12.5%码占比。20%是成本边界，恢复率和多版本有效性仍需独立质量结果。
