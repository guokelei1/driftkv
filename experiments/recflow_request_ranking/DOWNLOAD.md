# 本轮需要下载的数据

从 [RecFlow 官方下载盘](https://rec.ustc.edu.cn/share/f8e5adc0-2e57-11ef-bea5-3b4cac9d110e)
取得 `all_stage` 中 **2024-02-05 至 2024-02-18，共 14 天**的原始 Feather 文件。
[官方仓库](https://github.com/RecFlow-ICLR/RecFlow)给出了目录和字段说明。

如果下载盘可按日选择，只下这 14 个文件。**如果约 40 GB 是单个不可拆分的
`all_stage` 压缩包，需要下载整个包；不用下载网盘中的其他数据包。**
40 GB 是用户看到的大小，本目录没有独立核实服务端文件大小或是否支持按日下载。

解压后的目标位置：`/home/gkl/work/evokv/data/raw/recflow/all_stage/`。
只有压缩包时，放到 `/home/gkl/work/evokv/data/raw/recflow/downloads/` 即可，先不解压。
如果这台机器磁盘不足，可以放在另一台机器，保留完整路径后再安排一次搬运。

```text
all_stage/
  2024-02-05.feather
  2024-02-06.feather
  2024-02-07.feather
  2024-02-08.feather
  2024-02-09.feather
  2024-02-10.feather
  2024-02-11.feather
  2024-02-12.feather
  2024-02-13.feather
  2024-02-14.feather
  2024-02-15.feather
  2024-02-16.feather
  2024-02-17.feather
  2024-02-18.feather
```

请保留全部原始列、行、文件名和 ID，不要预先按 `rank_pos/rank_neg` 筛选、
截成 top-10/50、删掉未曝光候选或重编号。`rank_neg` 等标签本身可能只标记
抽样子集；完整候选的筛选要结合实际 `rank_index` 编码审核。

37 天 `realshow.tar.gz` 本机已有，不用重复下载。当前不需要 `request_id_dict`、
`ubm_seq_request_id_dict`、`seq_effective_50_dict`、作者预制的 `rank_test.feather`
或检索测试集。本实验从原始曝光构建长历史，从每日阶段日志构建连续版本的候选集。

文件到齐后，在仓库根目录运行：

```bash
python -m experiments.recflow_request_ranking.code.preflight
```

该命令只检查文件头、数据覆盖、现有历史长度和硬件，不启动训练。
文件头通过也不等于候选协议通过：还需检查真实行的阶段编码、候选完整性、
曝光关联、时间单位和字段可用时间。

若拿到压缩包，后续执行者使用下面的入口（实际包名按下载文件填写）：

```bash
python -m experiments.recflow_request_ranking.code.ingest_archive \
  --archive data/raw/recflow/downloads/all_stage.tar.gz \
  --report experiments/recflow_request_ranking/reports/extraction_01.json
```

入口保留原包，仅写出所需 14 天，默认保留至少 20 GiB 剩余空间。它不会覆盖已有不同内容，
可以验证同内容已完成文件后续接；不展开整个 tar、不解出前 23 天。gzip tar 本身仍需顺序扫描。
等待下载工具明确完成后再执行；不要把 `.partial` 或仍在增长的文件当作完整数据。
