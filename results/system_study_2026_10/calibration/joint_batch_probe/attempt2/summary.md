# V4 单个既定 optimizer 批的合批验证

One original V4 optimizer batch at retained finalV4 parameters; no optimizer step, new fit, quality claim or complete-S estimate

8人全部N=1024；同原batch8边界，逐人loss/8反传对比合批均值反传；四步续写保持时间顺序。权重未更新，backbone无梯度。原始loss/logits及逐参数梯度误差见equivalence.json。

|阶段|逐人中位秒|合批中位秒|逐人/合批|
|---|---:|---:|---:|
|forward_backward|1.804178|0.229260|7.870|
|validation_forward|0.772151|0.102416|7.539|

Warm interleaved complete forward/backward (or no-grad same objective); includes same-N packing, finite checks and zero_grad; excludes loading, initial prepare, diagnostics, optimizer step, checkpoint I/O and complete calibration.

Grouped mode retains eight users' differentiable rollout activations together; serial releases each graph after its backward. Peak includes temporary concatenations.

这是同一小批执行方式的可行性验证；不能直接将速度倍数用于完整S、全部层校准或source捕获，也未重新训练发布权重。
