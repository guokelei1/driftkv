# 当前论文图表

保留当前Motivation与最新Design的成图、点表和生成器。
旧预览和未选设计版本的位图/PDF与源码副本已清理；历史数值表及来源manifest保留。

| 图 | 成图与数据 | 生成器 |
| --- | --- | --- |
| 最新三部分Design结构 | [PNG](out/design_components/system_architecture.png)、[PDF](out/design_components/system_architecture.pdf)、[manifest](out/design_components/manifest.json) | [schematics.py](src/design_components/schematics.py) |
| 当前84.81%／18.44%开发证据 | [PNG](out/design_system_development/development_quality_cost.png)、[点表](out/design_system_development/points.json) | [design_system_development.py](src/design_system_development.py) |
| Motivation六方法图 | [输出](out/motivation_six_methods/) | [motivation_six_methods.py](src/motivation_six_methods.py) |
| 当前Q-v5/H-v4，90个点 | [成图与点表](out/read_correction_2026_09/motivation_final/README.md) | [read_correction_motivation_2026_09.py](src/read_correction_motivation_2026_09.py) |
| 四种局部重算 | [输出](out/selective_recompute_2026_09/) | [selective_recompute_2026_09.py](src/selective_recompute_2026_09.py) |
| 两类Motivation方法结构，选定v6 | [PNG](out/motivation_branches_2026_09_30/v6/motivation_branches.png) | [draw_v6.py](src/motivation_branches_2026_09_30/draw_v6.py) |

生成器只读已有证据，不启动模型实验。Q/H的当前90点原始评分与汇总保持，
四重算及六方法图保留所有正负点；显示规则不改变统计。
最新Design使用既有依赖flag限定row代码，主恢复点并非独立确认。

## 最新总架构图的阅读顺序

2026-10-07根据用户对核心Design图的反馈，改为围绕同一份原生缓存的机制结构。
上方是离线目标与共享映射；主体是K/V、窄编码与依赖flag的逐行对齐状态，
以及查询投影、同权重双聚合、聚合后Value解码与Current层的接入；
下方以进入的整行状态和code退出表现追加、继承依赖与生命周期。
矩阵、编码条带、投影的收缩/展开、加权汇聚和模型层切片都有实际含义。
主图展示各机制之间的协作；局部拟合过程、完整算子与退出判定细节留给小节图。

蓝色为查询与聚合读取，橙色为编码与投影，紫色为离线拟合及发布的共享参数，
青色为依赖与状态维护。灰色原生K/V仅保留一份；叠层轮廓代表不同层。
所有行、向量尺寸及权重均为示意，不表示测量结果或真实维度。

图为7.7 × 5.4英寸，最小字号9pt，导出400dpi PNG及矢量PDF/SVG。
PDF/PNG同步至`/home/gkl/work/paper/pic/fig_system_architecture.*`，
图注与Description同步至论文`main.tex`；未编译论文，也未改变方法或实验。

图表数值与方法范围见[结果索引](../results/README.md)，论文尚未编译。
历史点表和manifest仍能查阅；清理过的旧预览不再作为可打开的图片入口。
