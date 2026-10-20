# 追加统计开销：CPU定位

使用已完成graph32的1324个真实append批、2173个用户追加，七次交错回放CPU账与metadata。没有运行GPU或重放完整服务。

| CPU执行 | Relay | Native/Rebuild |
| --- | ---: | ---: |
| 原dense mask计数＋账与metadata | 160.55ms | 88.75ms |
| 闭式计数，保留完整账与metadata | 61.33ms | 35.98ms |
| 仅标量状态＋共同producer数组proxy | 27.41ms | 26.40ms |

闭式同时移除Native原本无用的mask构造，净缩小两边CPU差距约46.45ms，不能把Relay单边99.22ms全部当方法相对收益。原32人同步阶段中append scatter为759.27/412.87ms，差346.40ms；这个量级说明实验统计能解释一部分开销，但不足以解释全部差距。此CPU代理不含真实KV/code复制、kernel、同步或调度，不能直接从正式服务时间或30k预测扣除。

原recorder=None只关掉PhaseRecorder的CUDA同步和阶段记录；BatchedSessions仍计算_metadata/_read_geometry、追加self.calls、多个Counter和_record_memory。新改动仅将mask cardinality改为闭式整数求和，所有Counter、metadata、边界检查及KV/code/cursor/版本更新保留。_record_memory只读取Tensor形状/字节元信息，本身不读取GPU值。

CPU检查覆盖N=0..1024、全部coded前缀、width=1/2/4/8/16/32与inclusive/exclusive（6309900组），以及全部真实append几何；2 tests通过。CPU回放中的旧、新完整账和metadata逐项相同。它尚不代替GPU完整会话的数值/状态验证，后续128应继续比较分数、K/V、码和操作账。
