# 一次Adam更新核对

原seed17第一8名fit用户；使用既有V4参数的两个临时副本，原逐人执行与新same-N模块各反传、clip及Adam一步，未保存候选权重。

参数最大差 7.45058e-08，梯度相对L2 7.04128e-07。逐参数与Adam一、二阶moment原始误差见equivalence.json。

logical ledger仍为8次static、8次rollout、32个逐时序事件、8份反传；实际执行合为1次group forward/backward与4次批事件。原8人Adam边界未变。

这不代表完整训练后的权重逐位一致或最佳epoch必然相同；浮点归约顺序仍可能影响接近的验证值。未测完整S。
