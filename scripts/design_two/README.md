# 持续轨迹与校准公共依赖

当前保留固定用户/历史装载、原生持续轨迹、教师采集和事件计数依赖。
旧formation、学习强度、coordination、interaction及十项诊断的专用运行入口已清理。
这些探索的正负结论仍在[开发计划](../../docs/design/plan.md)和各原路径的summary/analysis。

[capture.py](capture.py)的教师采集函数从旧入口原样抽出；
[common.py](common.py)保留当前模型装载、设备设置、成本及来源记录，
不再为使用小工具而导入旧校准器。当前实际入口见[脚本索引](../README.md)。
