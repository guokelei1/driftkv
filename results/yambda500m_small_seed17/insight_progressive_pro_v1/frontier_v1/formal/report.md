# Progressive PRO C32/C48/C64 无标签 fidelity frontier

状态：正式 raw/seal 完整，未读取行为 label；冻结 progressive 选择门未通过。

## 成本与五边平均 fidelity

| point | Full FLOPs | cutover relative L2 | rolling relative L2 |
| --- | ---: | ---: | ---: |
| C32 | 10.52% | 0.60851 | 0.73331 |
| C48 | 14.54% | 0.58437 | 0.69700 |
| C64 | 18.64% | 0.58541 | 0.70121 |

## 冻结规则裁决

- C64 relative L2 相对 C32：cutover 5/5、rolling 5/5，fidelity improvement gate PASS。
- C64 absolute direction cosine >=0.90：cutover 3/5、rolling 0/5，direction gate FAIL。
- C48 是否位于 C32/C64 之间：cutover=False、rolling=False，monotonic frontier gate FAIL。
- C64 无标签 score gap 不差于 C32：cutover 5/5、rolling 4/5；仅作诊断。

## 结论

增加 carrier 确实比 C32 更接近 Exact，但不是一个单调且达到绝对方向门的 precision axis：C48 的五边平均 relative L2 略低于 C64，而 rolling C64 的五条边均未达到 0.90 方向门。按事前合同不能在看到结果后改选 C48，因此不冻结 progressive upgrade，不启动旧五边质量重测；当前正式设计仍是已经取得 AUC 5/5 正向的 C32 lightweight PRO。

这不是核心 Insight 的反证，而是对本次增量的边界：仅靠更多同类 carriers、第二个近乎等价的 probe 和 scalar decay，尚不足以形成一个可靠的自校准升级。
