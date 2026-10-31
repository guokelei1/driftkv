# GPU0/1 + NUMA1 小规模内存绑定测试

完成时间：2026-10-23T11:09:22.660715+08:00。两卡测试均退出0；主机内存峰值1.663 GiB，测试进程组硬上限4 GiB，禁用swap，memory.events的max/oom/oom_kill均为0。微型HSTU为2L/H64/4头、64上下文、2048 item，随机权重，每份checkpoint约0.69 MiB；未加载完整Max模型或人口。每卡PyTorch保留显存峰值24 MiB（不含CUDA驱动上下文）。

启动命令完整保存在launch.json。测试从Python启动前使用numactl --cpunodebind=1 --membind=1，CUDA_VISIBLE_DEVICES=0,1；父进程与两个GPU子进程均读取get_mempolicy确认MPOL_BIND、节点掩码2，CPU affinity仅NUMA1。外层独立systemd用户service设置MemoryMax=4G、MemoryHigh=3G、MemorySwapMax=0、OOMPolicy=kill、RuntimeMaxSec=120。

直接查询测试缓冲的每个物理页：普通CPU张量、CUDA pinned张量、Arrow/Parquet输入缓冲、checkpoint保存CPU张量、使用现有load_model读取后的CPU张量、新文件mmap读取、GPU回传CPU结果，全部位于NUMA1；checkpoint恢复前后GPU分数完全一致。33次测试组cgroup采样的anon N0最大值为0字节。逐卡证据在rank_0/summary.json、rank_1/summary.json及numa_maps.txt；总表及组内memory.peak/numa_stat在summary.json，完整采样在memory_samples.json。service退出末尾的systemd摘要显示512K，属于退出汇总时的值，不能代替运行中实际读取的memory.peak；这里采用后者1.663 GiB。

## 后续恢复入口

已准备scripts/resume_max_v1_v3_numa1_2026_10.py。调用时必须显式给出--host-memory-gib，统一限制主队列及所有子进程；独立tmux启动systemd用户service，先绑定NUMA1，再启动Python，进入数值源码前核查NUMA策略及cgroup硬上限。新入口使用旧队列的--run-manifest直接接续，从而不进入旧--resume-manifest生成的NUMA0 launch.sh。原始绑定、校准权重、数值源码及分块不变。

新入口的NUMA/cgroup检查及exec命令在一个1 GiB受限服务里经过CPU拦截检查：通过，未执行队列；Ruff F/E9通过。此次全量队列仍保持interrupted，V1→V2已保存34块，V2→V3尚未开始。正式模型／196k人口加载峰值尚未在NUMA1验证，4 GiB只用于本小试；后续应先做受限的真实模型小用户加载，确定正式总内存上限后再恢复。

## 保证范围

新分配的业务私有内存由严格MPOL_BIND:1控制；不使用preferred或允许跨节点回退的interleave。已存在的共享库／文件页缓存可能仍映射在NUMA0，绑定不会自动迁移已有物理页；内核／驱动元数据也不能声称全部在NUMA1。本次不改系统NUMA balancing、其他程序策略，不迁移／丢弃别人的页缓存。NUMA0匿名内存0是测试cgroup的监测值，不是全机空闲内存变化归因。

依据：[Linux NUMA memory policy](https://www.kernel.org/doc/html/latest/admin-guide/mm/numa_memory_policy.html)、[systemd255资源限制](https://www.freedesktop.org/software/systemd/man/255/systemd.resource-control.html)。GPU0/1与NUMA0物理直连，改用NUMA1后跨节点传输速度需重新测量；小试不承诺原来的全量耗时。
