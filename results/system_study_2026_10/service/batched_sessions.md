# 真实用户合批服务验证

本轮沿冻结的 Medium V2→V3→V4→V5 轨迹，把已验证的单用户服务接成同版本、同历史长度的真实用户批。推荐仍每16次行为触发一轮16候选；每用户行为顺序、严格过去读取、发布屏障及末尾推进均保持不变。组批只重排相互独立用户，因而这些数字衡量饱和 bulk 工作量，不衡量到达驱动的排队延迟或生产吞吐。

## 实现和对照

- `batched_session.py` 同时合批码物化/反锚、原生重建、历史追加和推荐评分。批中每行来自不同真实用户；记录用户、事件、真实历史长度、批大小与有效码前缀。CPU 输入整理、H2D、K/V 和码打包、结果拆分及评分 D2H 均进入完整服务计时。
- Relay 保留原生 K/V 和单份码；Rebuild 每用户/发布首次推荐请求重建一次，此后正常原生增量服务。两者使用相同分组策略。推荐评分的原生读取都走无 code 融合核；初版原生追加仍走旧 HSTU 后端，下面单独说明公平修正。
- 相同长度但有效码前缀不同的用户可合批：组内读取最长前缀，其余用户读取到的额外位置为已有零码。个人前缀和维护账不变，额外匹配/聚合算术单列；未将其隐藏在原逻辑 FLOPs 账里。
- 最终执行路径使用融合码转换与融合 Query 投影。码转换及 masked 读取的长度/存储步长作为运行时参数，避免每个历史长度都重新编译。共享加载、初始 V2 K/V 构建和首次后端预热单列。

数值对照始终是原 Inplace 转换和原读取 adapter 的逐用户路径。除了所有请求评分，还核对每次发布和末尾的 K/V、code、cursor、生产版本及原始操作账；这用于检查实现等价，不是质量提升证据。

## 32 用户工程过程

| 执行路径 | Relay batch≤8 完整秒数，两次 | Rebuild batch≤8 完整秒数，两次 | 范围 |
| --- | ---: | ---: | --- |
| Inplace1024、严格相同码前缀 | 14.571 / 15.488 | 8.206 / 8.521 | 初始真实用户合批 |
| 融合转换、严格相同码前缀 | 13.601 / 13.567 | 8.996 / 9.012 | 保留首次 serial 冷编译异常值 |
| 融合转换、零尾合批 | 7.858 / 7.845 | 11.164 / 11.188 | 两方法先后运行，不能当严格速度比 |
| 最终融合路径、零尾合批 | 13.821 / 13.707 | 12.847 / 12.875 | 两方法预热后交错运行 |

每条完成路径均有27,200个评分与每方法128个边界检查。最终路径评分最大绝对差不超过约1e-6。零尾合批将本32用户的 Relay 评分批次数从1250降至837，与 Rebuild 相同；它额外执行1.1452 GFLOPs。该过程揭示了组批碎片和小算子调用开销，但不同运行时段宿主机负载没有严格控制，表中跨行秒数不能全部解释为单项优化收益。最终32用户交错比较中 Relay 仍稍慢于 Rebuild，保留此结果。

`batched_session32_projection/attempt1` 在新后端预热约291秒仍未结束时，经确认后停止：masked 核将依赖历史长度的 mask stride 特化，形成大量 JIT 变体。日志和中断原因保留。修复后 `attempt2` 的 serial 后端预热29.39秒结束；该数字包含完整服务执行，并非纯编译秒数。更早融合转换分支的首轮 serial28.89秒也保留为冷执行，不混入热态速度比。

## 有限 GPU 状态空间与传输

`batched_session32_fused/attempt1/state_transfer.json` 使用同32用户实际终态，独立测量 pinned CPU backing 的打包→H2D→D2H→拆分，逐元素检查往返一致。N=1024时，单用户 Relay/native 单向载荷为10.125/9 MiB；真实8用户批为81/72 MiB。完整非重叠往返墙钟中位数分别为单用户1.900/1.653 ms、8用户19.934/17.906 ms。GPU分配计入H2D，pinned staging分配另列。

上述搬运是独立系数，不是主服务已经实现了 CPU backing，也不代表线上命中率。目标人口时间需结合实际分组/驻留窗口，明确何时迁入、何时写回，不能假定30k状态全部常驻，也不能把全人口无限组批率和小驻留窗口的搬运成本混合使用。

## 固定128窗口与原生追加的公平修正

名单来自原评价 manifest 顺序前128，并与 fit-size 趋势名单逐项绑定。该面板产生5,800轮、92,800候选评分；Relay评分平均4.85人/批，76.69%的用户评分调用落入8人批；追加平均3.11人/批。新增零尾算术为6.08869 GFLOPs。

`batched_session128_final/attempt1` 的 Relay 已通过512个边界和全部评分；Rebuild全部评分与原serial参考的最大差为9.54e-7，但末尾drain的一个状态张量有2/1,179,648元素超出既定3e-5绝对/相对标准，最大绝对差5.998e-5。该次失败日志未记录UID，因此UID未知；没有静默放宽标准，也没有形成完整计时结果。旧reader为验证新历史形状产生的JIT属于参考验证成本，需与实际后端首次准备分开。

源码复核还发现，初版native追加对短前缀使用Torch、满窗口使用旧HSTU的BM32矩阵核，而Relay使用BM16/小宽度向量融合核。这样测得的追加差异不能归因于避免重建。`batched_session128_fair/attempt1` 将native追加接入与Relay同架构的无code融合读取：保留相同投影、滚动因果mask和输出变换，不创建或保存code/gate。其参考是原native batch8路径，仍使用既定数值标准，并为失败日志增加UID/边界与相对范数。该公平修正现已完成：新native与旧native batch8的512个状态边界全部通过原标准，K/V最大绝对差分别为5.25e-6/4.77e-6，相对L2误差不超过8.65e-8；92,800个评分最大差7.15e-7。Relay对同一既定批执行路径的512边界和评分逐项一致，独立旧读取实现的验证依据仍是前面的32用户以及首次128运行的Relay检查。

| 公平128窗口，方法交错 | 第一次完整服务 | 第二次完整服务 |
| --- | ---: | ---: |
| Relay | 26.568 s | 26.419 s |
| Rebuild，同架构无code追加 | 22.927 s | 22.878 s |

该窗口中Relay平均慢约15.7%，没有证明完整服务时间获益。分阶段诊断显示，Rebuild的重建compute合计0.578秒，Relay物化与反锚compute合计0.068秒；但Relay/native的追加compute为14.614/13.285秒、追加scatter为2.382/1.006秒，评分compute为6.411/5.775秒。码准备已经很小，额外读写和状态维护开销仍超过该窗口节省的重建时间。这些分阶段值带额外同步，仅用于定位；表中的总服务时间来自无插桩执行。

每次初始V2缓存构建约0.62秒，两方均单列；峰值GPU分配约Relay12.22GB、native12.05GB，包含共驻的多版本模型和共享产物。27.28秒装载时间也包括所有基础模型，不能直接当作Relay独有的发布准备S。完整人口结论需要另外合入真实共享构建/校准S、实际30k分组形状和明确的状态搬运策略，不能将本128均值直接乘人口。

## 码维护与计算图诊断

追加码维护的单项复核使用同32用户、同进程反序交错的旧／新 Relay 与公平 Rebuild。原维护为每用户新建零尾、拼接、截断并连续化；新核一次复制保留行并写零尾，各用户仍有独立存储，全零满窗口直接复用。`batched_session32_code_append/attempt2` 的旧 Relay 为11.657/11.708秒，新 Relay 为11.585/11.613秒，平均只减少0.715%；Rebuild为10.352/10.424秒，新Relay仍慢11.66%。27,200评分和每方法128个边界与原批路径逐项相同。该变化减少小调用和拷贝，不改变有效FLOPs，收益较小，不据此扩大实验或宣称完整服务获益。`attempt1` 在正式计时前停止以加入同进程旧实现对照，加载及已完成检查保留。

共同模型路径的许多小调用也可压缩。`batched_session32_graph/attempt1` 在两方法上同等缓存固定形状 CUDA Graph：仅N=1024、B=1或8，覆盖16候选评分、宽度1/16/32追加和完整重建；其他形状仍eager。每次先复制当前输入，码前缀长度从GPU标量读取；图输出复制到用户独立存储后才可复用。图因此保存计算流程，不保存用户状态或固定码前缀。

同进程两轮反序比较中，Relay的eager/graph均值为11.876/8.608秒，Rebuild为10.708/7.740秒；双方改善约28%，Relay仍慢11.21%。27,200评分及每方法128个边界逐项一致，正式计时没有新增capture。当前50图的准备耗时为Relay2.288秒/native0.556秒，额外live allocation约1.405/1.136GB；这些是共享准备与工作空间，未混入热态秒数，也未当作完整driver/reserved显存。新模型发布需重新capture，已有同形状JIT可以复用。

分阶段结果显示，当前主要剩余计算来自未覆盖的满窗口B=2–7（Relay/native2.731/2.469秒）、满窗口追加宽度2/4/8（1.241/1.113秒）和短历史（2.785/2.660秒）。已覆盖图的compute仅0.655/0.616秒。诊断值带同步，不与完整墙钟直接相加解释速度比。逐调用backend位于`*_calls.json`；本次Parquet首行schema遗漏该字段，`coverage_analysis.json`按方法/顺序核对operation、UID、版本和长度后还原，后续记录已修复。

据此，`batched_session32_graph_full/attempt1` 只把满窗口图扩到实际B=1–8及追加宽度1/2/4/8/16/32，短历史仍不填充、不改计算。两轮完整服务为Relay graph5.416/5.410秒、eager12.026/12.115秒；native graph4.861/4.860秒、eager10.920/10.848秒。双方约55%改善，Relay仍慢11.37%，没有改变这一小面板上的相对结论。数值逐项相同，计时新增capture仍为0。

扩覆盖将图数增至226，准备成本Relay6.358秒/native2.556秒，额外live allocation6.848/5.928GB。该成本随实际形状和版本数量增长；同时保留两方法的三版本图只为同进程对照，不是所需服务部署。过程设置70%进程显存预算。未来更大面板须按实际单方法、单发布驻留预算处理图空间，不能假定无限保留；32面板时长也不能直接代入30k工作量。

固定128的后续入口为`batched_session128_graph.json`：每次只保留一种方法的图，先完成该方法预热，再用独立新会话计时，两方法交错重复。完整warm轨迹秒数与其中graph准备秒数分列，不把整段warm都计成capture。`batched_session128_graph/attempt1` 在加载开始约9秒时因外部任务已占用全部GPU而停止本进程，尚无正式测量；等待空闲后另建attempt继续，已有128公平eager结果不受影响。

后续单方法图缓存每次释放会检查旧Python owner已销毁，并记录allocator的allocated/reserved前后值。按32的live增量和128实际图shape比例估计，128单方法峰值约Relay25.53GiB/native22.42GiB；这不是reserved显存保证，70%进程预算保持不变。外部任务以短episode间隙重启，须观察持续空闲再恢复，不能在间隙抢占。

为降低该风险，尚未运行完成的128配置已启用`graph_share_history_inputs`。同步执行时，不同图仅共享具有相同clone布局/形状/dtype/device的K/V、code输入；private pool、输出、ID/delta/count仍独立。每次执行及新图捕获前都重新填入当前输入，用户持久状态不引用该共享区。按128实际形状，显式历史输入预计减少Relay7.9596GiB/native6.2051GiB；未计private pool变化，尚不是GPU实测节省。默认及所有已完成32配置仍关闭该开关。CPU检查通过，最终GPU数值/计时验证仍待资源空闲。

从仓库根目录恢复，先确认所选物理卡持续空闲；下例使用GPU0，并保留失败attempt1：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u scripts/system_study_2026_10/batched_session_probe.py --config configs/system_study_2026_10/batched_session128_graph.json --output results/system_study_2026_10/service/batched_session128_graph/attempt2 --gpu 0
```

当前没有自有GPU进程或自动排队进程；attempt2未创建，单方法128计时尚未验证。

待运行版本另将scatter中仅用于操作账的dense布尔mask及求和改为精确整数闭式，保留Counters和全部状态/metadata对照；两方法都避免构造该无用mask。其几何等价由CPU检查覆盖，既有GPU结果未重写。最终CPU组合检查为8通过、1 CUDA跳过；共享输入及闭式计数的128完整GPU验证仍未完成。

## 证据入口

配置位于 `configs/system_study_2026_10/batched_session*.json`。对应目录保留 `configuration.json` 的输入与源码哈希、`summary.json`、逐批 `*_calls.json`、`phases.parquet`、分数和进程日志。`phases.parquet` 额外同步拆分 pack/H2D、compute、scatter，不能替代无插桩完整墙钟；主总时间使用 `time_units.service_wall_seconds`。

- 初始合批：`batched_session32/attempt1`
- 融合转换及真实状态搬运：`batched_session32_fused/attempt1`
- 零尾合批：`batched_session32_coalesce/attempt1`
- 最终32用户交错验证：`batched_session32_projection/attempt2`
- 固定128首次验证（含原生状态容差失败）：`batched_session128_final/attempt1`
- 追加后端公平修正后的128窗口：`batched_session128_fair/attempt1`
- 同进程码追加维护单项对照：`batched_session32_code_append/attempt2`
- 两方法同等固定形状图对照：`batched_session32_graph/attempt1`
- 满窗口实际批大小／追加宽度图对照：`batched_session32_graph_full/attempt1`

所有路径复用既有冻结权重，没有重训；多个模型版本/共享表在 GPU 同驻，因此不主张已实现生产容量管理。本轮没有 Large/Max 纠正器效果结论，已有跨模型测量只覆盖原生算子成本。
