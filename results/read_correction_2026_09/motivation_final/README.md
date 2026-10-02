# Motivation：当前非线性 Q/H 结果

**已完成：15 条相邻边，Q 45 点、H 45 点，共 90 个实测点。**

- **Q-v5**：`cross_phi_joint8`，C128/256/512；
  来源为 `../v5/population_run/evaluation/`。
- **H-v4**：`map_all`，C64/128/256；
  C128 来自 `../v4/population_run/evaluation/`，
  C64/C256 来自本目录 `evaluation/`。

[总图](../../../figures/out/read_correction_2026_09/motivation_final/overview.png) ·
[PDF](../../../figures/out/read_correction_2026_09/motivation_final/overview.pdf) ·
[90 点数据](../../../figures/out/read_correction_2026_09/motivation_final/points.csv) ·
[均值](../../../figures/out/read_correction_2026_09/motivation_final/equal_edge_means.csv)

## 当前结果

全部 15 边等权平均；沿用各边固定 3,000 用户、全部请求、Full/Reuse 及成本分母。

| 探针 | 校准人数 | 成本 | 恢复率 |
| --- | ---: | ---: | ---: |
| Q-v5 | 128 | 1.49% | 11.96% |
| Q-v5 | 256 | 2.79% | 33.35% |
| Q-v5 | 512 | 5.43% | 37.32% |
| H-v4 | 64 | 72.11% | 69.37% |
| H-v4 | 128 | 74.22% | 70.95% |
| H-v4 | 256 | 78.44% | 71.69% |

恢复率为 `(方法 AUC−Reuse AUC)/(Full AUC−Reuse AUC)`，先逐边计算再平均。
成本为 `(完整校准＋额外推理 FLOPs)/(Full 历史重算−Reuse 追加 FLOPs)`。
H 请求期历史处理均值为分母的 69.68%，占三档总成本约 89%–97%。
这些是解析计算量，不是延迟比例。

本组探针显示低成本共享修正可恢复部分质量，直接处理完整历史可在更高成本下
获得更高总体恢复。逐边可能下降或超过 Full，所有点均保留。两种探针的结构与
监督不同，不能当作严格的单因素消融；结论也不代表 Q 的能力上限或历史处理必然昂贵。
面板是按已有 Full/Reuse 结果选择的开发人群，不能外推为无偏总体估计。

方法与共同协议见[实验说明](../../../docs/design/read_correction_probe_2026_09.md)；
Q 逐规模结果见 [v5](../v5/README.md)，H 基础实现与比较见 [v4](../v4/README.md)。

## 图表与复现

总图包含总体、三规模均值及 15 条边，共 19 个子图。
[图表入口](../../../figures/out/read_correction_2026_09/motivation_final/README.md)
提供只读取已有结果的复现命令；生成器校验评分哈希、配对 Full/Reuse 及成本分母。

`figures/out/read_correction_2026_09/v5/` 另保留 135 个完整探索对照点，
包括旧 Q-v2 和便宜的 H-v1。不同方法均保留其真实费用，未通过坐标平移或删点
构造当前曲线。六方法原 400 点比较使用旧 Q-v2，见[历史比较](analysis/README.md)；
不能将其中 Q-v2 的比较数字当作当前 Q-v5。

## H 预算补齐的执行记录

本目录原任务只为固定 H-v4 补齐 C64/C256，共 30 点，C128 直接引用已有 15 点。
2026-09-29 21:20（北京时间）完成，耗时 6,244.74 秒（104 分 05 秒），
driver 与绘图退出码均为 0。新增两预算共 802,374 条评分，每预算 401,187 请求，
无 OOM 或 batch 缩减。

H 唯一预算变化为 `calibration_users`；pure Parent 校准、每用户 16 个
uniform-known query、最多 128 历史位置、ridge 0.01、6 轮非线性拟合及 seed17 不变。
预约名单前 C 人拟合、其后 16 人验证，均与评价面板分离；不同 C 的验证组随之变化。
每点计入完整教师、拟合、验证及推理费用。

- 入口：`scripts/read_correction_motivation/calibrate.py`，
  包装原 v4 fitter；评测复用
  `scripts/read_correction_v4/evaluate_full.py --variants map_all`。
- 计划及完成记录：`plan.json`、`canary.pass.json`、`complete.json`、
  `driver.exit`；阶段记录在 `runtime/c*/<scale>/<edge>/`。
- 拟合参数：`calibration/c{64,256}/<scale>/<edge>/`；
  完整评分：`evaluation/c{64,256}/<scale>/<edge>/`；
  小样本检查在 `canary/`，不混入正式曲线。
- 四卡每卡一项任务，70% 分配上限；每 256 用户一个评分分片。
  全量 Full/Reuse/identity 最大偏差为 1.19e−6 / 1.55e−6 / 1.43e−6，
  峰值 allocated/reserved 为 18.45/30.15 GiB。
- 事前 C64 Medium 与 C256 Max 检查分别拟合约 25/200 秒，评价 999/945 请求，
  最大对照偏差 9.54e−7，reserved 28.04 GiB；原 90–135 分钟估计见
  `resource_estimate.json`。

该轮原汇总为 Q-v2 60 点＋H-v4 45 点。旧 Q 四预算的恢复为
6.88/12.79/15.42/15.28%，成本为 1.42/2.83/5.67/11.33%；
[原 105 点输入与比较](analysis/README.md)保留原数据和来源哈希。
当前图改用 Q-v5 45 点，H 的 45 点保持原结果。

旧探索调度入口已退休；已存结果和封存哈希保留，不宣称原队列能以当前源码
直接通过全部历史哈希续跑。本页记录完成实验，不授权重复运行。
