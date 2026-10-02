# 当前 Query／History 修正图

- [当前非线性 Q/H 总图](motivation_final/README.md)：Q-v5 C128/256/512、
  H-v4 C64/128/256，15 条边，共 90 点。含总体和三规模均值、全部逐边曲线。
- [探索对照总图](v5/overview_15panels.png)／[PDF](v5/overview_15panels.pdf)：
  全部 135 点，保留 Q-v2 与低成本 H-v1 历史对照；
  [数据表](v5/overview_15panels.csv)／[来源](v5/manifest.json)。

```bash
python figures/src/read_correction_motivation_2026_09.py --require-complete
python figures/src/read_correction_v5_2026_09.py --require-complete
```

两套图只读取保留结果。生成器核对原始评分哈希、配对控制及计算量分母；
原始结果、封存记录和历史比较输入见
[结果索引](../../../results/read_correction_2026_09/README.md)。
