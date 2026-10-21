# CacheTide 实验图表预览

当前正文以 `paper/main.tex` 为准。2026-10-21 的组织为：6.1 设置，6.2 核心实验，6.3 机制与优化来源，6.4 链路时间与持久存储，6.5 适用范围。

正式范围为 **Yambda / RecFlow × Large / Max**；Medium 仅作开发。这里全部是用户授权的模拟版式数据，不是实测证据；不运行训练或 GPU 实验。RecFlow 正式反馈任务、模型和人口仍须确定，旧 NDCG 不能改作 AUC。

## 正文图表分工

| 小节 | 资产 | 职责 |
| --- | --- | --- |
| 6.2 | `e1_quality_cost`，双栏1×4 | 四配置质量—完整成本前沿，逐版标记和均值曲线 |
| 6.2 | `e3_lifecycle`，双栏1×4 | 连续五次发布的恢复率与完整附加成本 |
| 6.3 | `e8_mechanism`，单栏左右两图 | Design：读取结构／门控与校准／追加监督 |
| 6.3 | `e6_implementation`，单栏左右两图 | Implementation：投影复用／合批与融合／请求规模分发 |
| 6.4 | `e2_time_breakdown`，2×2独立时间轴＋存储小表 | 相同资源下完整阶段时间；原生K/V、新增持久状态及比例 |
| 6.5 | `e9_boundaries`，单栏左右两图 | 原 Figure 11／12 合并：固定版本续写与读取次数的条件成本边界 |

当前正文六幅实验图、两张实验表（质量锚点、持久存储），含动机共11幅图、3张表。6.3恰好四幅子图，组成两张单栏图，不再插独立机制表。图号由 LaTeX 自动生成。

原 `e4_read_structure`、`e5_calibration`、`e2_time_scale` 是历史布局；`e3_release_hold`、`e7_applicability` 仍是合并图的数据和公式来源，但不单独导出为正文图。保留原始模拟输入。6.3局部分析用代表性 Yambda-Large，核心四配置证据仍由6.2提供，不表示跨模型纠正迁移。

## 生成与导出

```bash
python eval_figure/e8_mechanism/plot.py
python eval_figure/e6_implementation/plot.py
python eval_figure/export_to_paper.py
# 全量重画和图册，仅在需要更新所有预览时运行
python eval_figure/render_all.py
```

各目录保存模拟 JSON、脚本、说明和 PNG/PDF。脚本不读真实 `results/`，不自动编译论文。存储表由 `e2_time_breakdown/storage_table.py` 单独生成，正文原生表须同步；它不是显存峰值图。

## 视觉与证据约束

- 双栏图表注最多一行，单栏最多两行，解释放正文；模拟标识在 caption、文件名、脚本及元数据，图内不加水印。
- 6.2 两图都约7英寸宽；两图均继续压低高度。其余主图约3.35英寸单栏；Figure 9 和合并后的 Figure 11 导出高宽比约1:2.3。Figure 9 图例分属左右子图；Figure 10 为单栏2×2，模型名在上、Rebuild／CacheTide在下，Large和Max分别使用线性时间范围，薄分隔线保留分层。
- 核心方法蓝、参考深灰／浅灰；对照另配形状与线型，黑白可辨。机制四图各自有明确问题，不能只为丰富类型而加图。
- 质量、附加读取 FLOPs、完整附加成本、路径时间和持久字节是不同口径。共享准备、校准、维护及读写搬运计入完整账，不将局部加速比相乘或冒充生产吞吐。
- 不生成假误差棒。基础模型固定；正式训练重复仅针对纠正模块及关键学习基线，各模型组独立校准。

`make_overview.py` 生成独立布局图册，不是论文编译结果；页面中的图注也不用于投稿排版。
