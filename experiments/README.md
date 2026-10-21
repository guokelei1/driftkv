# CacheTide 正式实验

这里集中保存正式实验的计划、配置、运行记录和结果。源码继续放在 `scripts/`、`src/`；已有数据、底座权重和历史结果保留原路径，通过配置记录其来源。模拟图继续放在 `eval_figure/`，不混入正式实测。

恢复工作先读[STATUS](STATUS.md)，再读[准备度与推进报告](docs/experiment_readiness_2026_10_21.md)。评测取舍的文献及历史依据见[论证笔记](docs/evaluation_rationale.md)。论文章节以[当前main.tex](../../paper/main.tex)为准，以下图号对应2026-10-21源码；目录编号是工作索引，不是旧E1–E7编号。

## 实验分组

| 目录 | 对应论文 | 要回答的问题 | GPU安排 |
|---|---|---|---|
| [01_quality_cost](01_quality_cost/README.md) | Figure 6、Table 2 | 相对强基线，用多少完整FLOPs恢复多少质量？ | 质量／FLOPs可共享 |
| [02_continuous_releases](02_continuous_releases/README.md) | Figure 7 | 同一默认方案连续五次发布能否保持收益？ | 可共享；复用01默认轨迹 |
| [03_read_structure](03_read_structure/README.md) | Figure 8(a) | 动态门控与宽存窄读分别是否值得？ | 质量／FLOPs可共享 |
| [04_calibration](04_calibration/README.md) | Figure 8(b) | 局部、联合及短尾校准如何影响当前和后续质量？ | 质量／FLOPs可共享；秒数另测 |
| [05_preparation](05_preparation/README.md) | Figure 9(a) | 目录复用和跨层合批如何降低准备时间？ | 正式计时需受控 |
| [06_fused_reads](06_fused_reads/README.md) | Figure 9(b) | 融合与请求规模分发各贡献多少？ | 正式计时需受控 |
| [07_execution_storage](07_execution_storage/README.md) | Figure 10、Table 3 | 完整时间花在哪里，引入多少持久存储？ | 时间受控；字节主要CPU核算 |
| [08_boundaries](08_boundaries/README.md) | Figure 11(a)(b) | 长期不发布、读取量增加时，收益在何处减弱？ | 质量／FLOPs可共享；公式CPU核算 |

正式四配置为 **Yambda／RecFlow × Large／Max**；Medium仅开发。各模型组独立拟合，真实反馈为主任务，next-item保留历史身份。基础模型不为多seed重复训练。RecFlow反馈目标、两条最终链与正式人口仍待落实；预期曲线不作为通过标准。

## 配置与输出放法

每个实验目录已有 `README.md`、`configs/`、`runs/`。说明文件记录实验目的和依赖；后两个目录目前只有占位文件，不表示已配置或已运行。

```text
experiments/<实验目录>/
  README.md
  configs/                              # 实际采用的实验配置与必要输入清单
  runs/<dataset>/<scale>/<variant>/
    seed_<seed>/attempt_<编号>/          # 实际运行后创建
      configuration.yaml                 # 本次解析后的配置及输入路径
      summary.json                       # 指标、原始聚合量、完整费用、完成状态
      ...                                # 必要分数、计账、计时与日志
```

这是存储约定，不要求先搭一套通用运行框架。非学习／确定性实验可省略seed层。实际配置须注明模型、任务、角色／请求、时间窗口、预算、方法及来源状态；运行记录注明命令、设备、共享情况、计时有效性与输出身份。失败保留独立attempt，不覆盖成功结果。

`runs/`内原始输出默认本地保留；Git跟踪说明、`configuration.yaml/json`和`summary.json`，延续仓库不提交大份分数、权重与运行日志的约定。关键汇总之外需要保留的小型证据，再按实际文件明确加入规则。

共同角色／请求manifest首次确定后保留一份，其他实验引用；初始模型与数据不复制进各组。02引用01默认轨迹，07和08按需引用其权重、状态及账本；派生结果记录输入路径，不复制大份cache和分数。各方法／预算／seed仍须维护自己的学生演化状态。

先共享完成质量与FLOPs，再冻结输入安排正式计时。共享运行耗时仅作进度记录。完整规则见[报告第7节](docs/experiment_readiness_2026_10_21.md#gpu调度质量flops可共享正式计时单独安排)。当前没有新增训练、GPU实验或论文编译。
