# 历史资料边界

当前入口是 [../current_route.md](../current_route.md)，完整新路线见 [../newset.md](../newset.md)。历史表示可追溯但不再定义当前研究路线，不表示所有内容都没有价值。

## 2026-08-17 已清理内容

- 旧路线文档、评测协议、实验记录和论文写作资料。
- 全部实验结果、checkpoint、运行日志、缓存和生成的 processed data。
- TenRec、MovieLens 等不再服务当前路线的数据副本。

这些内容不再提供复现入口，也不再作为研究结论来源。

## 不得复活的路线

事后 score 混合或缩放、逐边选择 schedule、用 qualification 用户的 exact target K/V 拟合自由 mapper、删除负边后拼接矩阵、把 system smoke 当作正式性能结果。这些记录保留的意义是说明为什么停止，而不是提供可重新选择的候选。

上述描述只针对 37D 重建前的旧路线。此后仓库已经形成 Yambda、P7、P8、P9 的新 contracts、scripts、tests 与 development results；它们受[当前路线](../current_route.md)约束，不属于旧清理范围。

清理区 `/data/gkl/.evokv_cleanup_20260817` 是项目外历史位置，不是当前实验输入。不得为了寻找有利数字而恢复其中的旧结果。
