# Query relay development budget audit

Theoretical FLOPs on actual fixed30k causal event shapes and observed calibration operations; no GPU timing or quality claim.

Primary: 16行为触发一次推荐、每轮16候选；4副本独立准备。原feedback和已注册频率敏感性保留各自的Full/Rebuild分母。

| Candidate | observed calibration TF | 16/16 added % | original feedback added % | 16/16 incl. source % | status |
| --- | ---: | ---: | ---: | ---: | --- |
| A_itemgate6_rms_roll4 | 14.761656 | 17.37908 | 23.76709 | 18.61370 | 校准三版完成 |
| A_constantgate6_rms_roll4 | 14.761441 | 17.37903 | 23.76699 | 18.61365 | 校准三版完成 |

缺少校准版时，上表仅为已观察校准加完整服务开销的下界，不能称完整候选结果。重建源K/V的敏感性已含原始origin一次，不重复叠加。

- A_itemgate6_rms_roll4：主口径额外读开销占native完整query 9.079%；全局共享状态每副本 3.182 GB，4副本 12.726 GB，另加逐行48维码（native K/V的12.5%）。
  当前发布的实际小块逆矩阵另存8,640B/副本；FP64发布级求逆按LU算法估计48,384 FLOPs/发布/准备副本，四副本三版合计580,608 FLOPs。每行状态和矩阵乘计算量不增加。
  Current物品门控另有 41.760120 GB瞬时重复lookup载荷：每候选/追加行为768B，包含空纠正前缀的实际lookup，六层共用一次输入。仅披露字节，没有新增投影FLOPs或持久特征；不能推断实际延迟免费。
  native历史＋self注意力的额外query开销为 13.499%。整个建模服务 Full=1007.020553 TF，Relay=703.283523 TF；净省 303.737029 TF，即Full服务的 30.162%。若源状态需重建，Relay=707.822325 TF，净省 299.198228 TF（29.711%）。
- A_constantgate6_rms_roll4：主口径额外读开销占native完整query 9.079%；全局共享状态每副本 3.182 GB，4副本 12.726 GB，另加逐行48维码（native K/V的12.5%）。
  当前发布的实际小块逆矩阵另存8,640B/副本；FP64发布级求逆按LU算法估计48,384 FLOPs/发布/准备副本，四副本三版合计580,608 FLOPs。每行状态和矩阵乘计算量不增加。
  native历史＋self注意力的额外query开销为 13.499%。整个建模服务 Full=1007.020553 TF，Relay=703.283308 TF；净省 303.737245 TF，即Full服务的 30.162%。若源状态需重建，Relay=707.822109 TF，净省 299.198443 TF（29.711%）。

20%口径是全部新增成本/KV重建，不能解释成整个模型服务的FLOPs比例。整个建模服务包含三版期间真实历史追加、完整query及首次推荐KV重建；两条路径共同继承初始V2缓存，不含基础模型训练、检索基础设施或实际运行时间。

所有数值均为理论FLOPs；query增幅与总重建分母分别报告。全局D表和immutable bank不属于逐行12.5%码占比。20%是成本边界，恢复率和多版本有效性仍需独立质量结果。
