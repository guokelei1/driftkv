# 6.5 合并适用边界图（模拟，当前 Figure 11）

原 Figure 11、12 合并为一张单栏图，左右各一幅子图；原 Figure 12 不再保留第二个子图。导出高宽比约 1:2.3，矢量文字，图内无模拟水印。出处在 caption、文件名及元数据。

- 左：直接读取 `e3_release_hold/data.json`，调用原 `cost_trajectory`，质量与累计费用数值保持不变。曲线不是预算前沿，稀疏点标签 0N／0.5N／1N／2N 表示同一轨迹的追加进度。横轴累计附加 FLOPs／一次起始 Rebuild，纵轴匹配检查点的恢复率。完整初始化计一次，无下一发布交接。
- 右：直接复用 `e7_applicability` 的既有合成系数和 `costs` 公式，N=1024、每版追加20次，改变每用户每版候选读取总次数。横轴是读取数，不再是每次行为的候选数；传给原公式时除以20。纵轴为净省下的 FLOPs／一次 Rebuild 预算，和旧图的总服务 FLOPs 分母不同，不能互换百分比。共同原生工作同时扣除，所有附加工作仍计入。初始化分摊、交接、人口和目录使用原来源的明确假设。

右图四条线为 Yambda-Large／Max、RecFlow-Large／Max；图例缩写 Y/R 和 L/M 在正文解释。右图是条件成本账推导，没有新增质量或真实请求分布证据，不画人口覆盖率。两图使用不同服务过程，正文明确区分，不把右图当作左图的外推。

`data.json` 仅索引原数据、定义切片和版式；`model_ledger.json` 保留派生曲线。所有数据明确为模拟，没有 GPU、训练或真实结果变更。

```bash
python eval_figure/e9_boundaries/plot.py
```

输出 `fig_6_5_boundaries_simulated.{png,pdf}`；论文资产为 `fig_eval_boundaries_simulated.*`。原两图的独立资产保留为历史，不再插入正文或列入批量导出。
