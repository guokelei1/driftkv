# Max V2 完整训练 step 的性能剖析

2026-09-19 用户授权实际测量瓶颈，并允许自行决定暂停或等待。选择让正在进行的
Max V2 训练完整结束，仅暂缓等待训练的外层调度，在训练与原定 E14 评测之间插入
短测量，结束或失败后恢复调度。没有修改当前训练／评测源码、合同或 checkpoint。

## 测量设置（执行前固定）

- 同一个已保留 Max V1 epoch1，16L/H320/10heads/context1024，完整353万词表；
  四张A40、global80、每卡20，CPU布局和原训练保持一致。
- [fixture](fixture/configuration.json)复用现有因果历史、训练请求排序、rank分配和用户权重，
  取 V2 [231,245) 原训练窗口每rank前12batch，共960请求。它是诊断样本，不代表全epoch。
  全部batch宽度1024，实际长度分组数1–9；没有读取 E14 评价结果。
- 三个固定设置：原 PyTorch＋默认 AdamW、Triton auto＋默认 AdamW、
  Triton auto＋fused AdamW。各自从相同parent和fresh optimizer出发、seed17。
- 先2步有限值 canary；主测每设置预热4步、计时10步、另采集2步 profiler。
  不输出候选权重、不作质量选择、不改变正式训练配方。
- 正常计时包括CPU已组装batch的H2D、原FoundationForward／loss／backward／FSDP／AdamW，
  不包含一次性加载和原CPU collate。fixture单独测量CPU collate，以免混淆口径。
- GPU阶段event不插入阶段间同步；profiler单独运行，输出原始trace及GPU区间并集／交叠。
  算子工作时间与嵌套CPU范围不能直接相加当作墙钟百分比。
- 两次短进程预计3–8分钟，保守预留15–20分钟；等待训练不占用额外GPU。
  使用detached tmux；调度器在成功、失败、超时或TERM退出时恢复原runner。

保留测量入口：`scripts/unified_training/prepare_max_profile_fixture.py`、
`scripts/unified_training/profile_max_training.py`。
当时使用的 `run_max_profile_after_training.py` 一次性调度入口已于2026-09-22清理，
删除前源码快照见[结果索引](../../README.md#2026-09-22-入口清理)，本次操作与测量证据保留。
状态／命令／退出记录在 `probe/progress.json`、`probe/runtime.json` 和 `queue.log`。
主测的紧凑证据写入 `probe/measurement/summary.json`；原始NPZ／trace／日志留本地忽略。

CPU fixture准备已完成，耗时108.9秒、峰值RSS1.83GiB。CPU collate均值0.282ms，
不含H2D；rank0前240请求和权重与原训练器直接输出完全一致，所有fixture哈希核对通过。
## 已完成的四卡测量

正式 V2 连续完成11402步并保存最终checkpoint后，于23:04:12开始诊断；canary
耗时23.65秒、主测125.51秒，均成功。23:06:41恢复原runner，原定Full E14评测
已于23:07:01启动。仅外层调度等待，训练worker没有暂停／恢复或修改配方。
从观察到训练退出至恢复调度约149.22秒；外层runtime的train墙钟会包含这段人为等待，
训练器的step计时不包含它。不要将这段诊断延迟算作正常训练成本。

[完整计时及数值分析](analysis/README.md)、[紧凑分析数据](analysis/summary.json)、
[原始测量摘要](probe/measurement/summary.json)保留所有三组和四个rank。

| 同一Max V1、真实V2 batch、四卡global80 | step中位数 | step均值 | 峰值allocated显存／卡 |
| --- | ---: | ---: | ---: |
| PyTorch＋默认AdamW | 943.16ms | 963.70ms | 24490MiB |
| Triton auto＋默认AdamW | 759.47ms | 804.86ms | 12454MiB |
| Triton auto＋fused AdamW | 690.45ms | 709.53ms | 12454MiB |

当前Triton使完整step中位耗时下降19.48%（吞吐1.242倍），重现此前独立12步
对照的20.47%耗时下降。六层backbone的数倍加速不应外推为完整Max训练的数倍加速。
这里每组只计时10步，输入包含不同真实长度组；均值和逐步结果同时保留。

## 已定位的成本

- profiler中前缀前向的GPU工作由约156.0降至28.3ms/步，约5.5倍；这是嵌套
  prefix计算范围，既不含整步成本，也不等于含FSDP的forward墙钟。
- 每步仍有一次参数all-gather和一次梯度reduce-scatter；观测到all-gather约
  204–206ms，trace内没有与本rank计算重叠。模型11.382亿参数中，商品表占99.27%。
- 长度分组会重复查同一大embedding表，反向反复生成、清零和累加稠密表梯度。
  请求数均衡并未均衡这种工作。第一个profile batch中rank0有9组、rank3仅1组；
  rank3提前约306ms进入reduce-scatter，等待后与rank0一起完成。因此不能把全部
  NCCL kernel时间称作纯网络传输。两个profile step中的embedding反向及梯度累加
  GPU工作，rank0合计237.22ms（118.61ms/步），rank3合计57.68ms（28.84ms/步）。
  这是独立profiler范围，不能除以干净计时当作严格墙钟占比。
  详见[归因证据](attribution/summary.json)及
  [复算脚本](../../../scripts/unified_training/analyze_max_profile_attribution.py)。
- 默认AdamW阶段约43.46ms，fused约15.65ms，直接节省27.81ms，约为原auto整步
  的3.7%。三组短测中auto→fused整步中位数下降9.09%，但其他阶段也有波动，不能
  将全部9.09%归给优化器或承诺它是长期稳定收益。
- 此fixture的CPU collate中位数0.263ms，H2D不足1ms；它们不是这些已预备batch的
  主瓶颈。这不排除正式运行的一次性数据／模型加载成本。

[启动观察](startup/README.md)另保留了22:50:23的只读快照：V2启动后新增3060个
匹配Max H10/BF16/D32的HSTU内核，即1020组前向／两种反向；955个前向是按真实
context长度分别专门化的单query变体。缓存增长与早期慢步对应，支持编译预热成本。
缓存写入数不能换算成精确编译秒数；首次新缓存前约30.5分钟仍是未分解启动成本。

## 数值及下一步边界

auto默认组两步canary的梯度和loss有限；三组计时与profile步骤的loss均有限，
初始80个logit完全一致。14次更新后，auto默认
与fused的全局加权loss序列最大差1.08e-4，参数抽样最大差3.46e-6；这些是短程和
抽样证据，不证明全参数逐位相等或长期质量等价。执行源码起止hash一致，旧V1严格
加载成功；诊断没有保存候选权重或读取E14质量结果。

由实测支持的后续优先级是：先统一候选embedding lookup，再按长度索引vectors，
减少每组重复的大表梯度；减少单query按每种长度编译；再衡量fused AdamW及FSDP
通信组织。现有candidate_item_vectors接口可支持第一个小改动，但本轮尚未实施或
测量该改动的收益。当前正在运行的正式评测源码保持原样。
