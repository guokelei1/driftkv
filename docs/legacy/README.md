# 历史资料边界

当前入口是 [../current_route.md](../current_route.md)，完整新路线见 [../newset.md](../newset.md)。历史表示可追溯但不再定义当前研究路线，不表示所有内容都没有价值。

## 已清理内容

- 旧路线文档、评测协议、实验记录和论文写作资料。
- 全部实验结果、checkpoint、运行日志、缓存和生成的 processed data。
- TenRec、MovieLens 等不再服务当前路线的数据副本。

这些内容不再提供复现入口，也不再作为研究结论来源。

## 不得复活的路线

事后 score 混合或缩放、逐边选择 schedule、用 qualification 用户的 exact target K/V 拟合自由 mapper、删除负边后拼接矩阵、把 system smoke 当作正式性能结果。这些记录保留的意义是说明为什么停止，而不是提供可重新选择的候选。

当前仅保留源码、测试、配置、路线文档和少量 KuaiRand 原始数据。清理区位于项目外的 `/data/gkl/.evokv_cleanup_20260817`，待确认无误后释放。
