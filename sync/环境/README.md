# 202 实验环境记录与 201 迁移参考

根据 202 在 2026-10-21 的实际环境整理。硬件不同不妨碍运行准确性实验；先跑单卡流程确认依赖和数据，再按需要适配多卡。两台机器的耗时不直接比较。

## 202 当前环境

| 项目 | 实测 |
| --- | --- |
| 系统 | Ubuntu 24.04.2，Linux 6.8.0-31，x86_64 |
| CPU / 内存 | Intel Xeon Gold 5420+，112 logical CPU；约 1 TiB RAM |
| GPU | 4 × NVIDIA A40，compute capability 8.6，每卡 46,068 MiB |
| NVIDIA 驱动 | 590.48.01 |
| Conda | Miniconda3，Conda 26.1.1；安装在 `/home/gkl/miniconda3`，当前用 `base` 环境 |
| Python | 3.13.12 |
| PyTorch | 2.12.1+cu130；CUDA runtime 13.0、cuDNN 9.20.0、NCCL 2.29.7 |
| GPU kernel / 编译器 | Triton 3.7.1；系统 nvcc 是 CUDA 12.0；GCC 9.5.0 |
| 主要包 | NumPy 2.4.3、Pandas 3.0.1、SciPy 1.17.1、PyArrow 24.0.0、DuckDB（Yambda 数据读取）、pytest 9.1.1 |

完整快照见 [`202_snapshot.txt`](202_snapshot.txt)。项目基础依赖见 [`pyproject.toml`](../../pyproject.toml)：Python ≥3.10、torch ≥2.3、numpy ≥1.26、pandas ≥2.0、scipy ≥1.11、jsonschema ≥4.18；`data` extra 安装 pyarrow，`dev` extra 安装 pytest/ruff。DuckDB 被 Yambda 历史加载入口使用，但目前没有列入 `pyproject.toml`。当前基准不需要 torchrec 或 fbgemm_gpu。

### 201 建立环境

202 用的是 Miniconda 发行版提供的 Conda，不是 Anaconda，也没有专门的 `evokv` 环境。201 可以使用本机已有的 Conda/Miniconda/Mamba，建议新建隔离环境：

```bash
conda create -n evokv python=3.13 pip -y
conda activate evokv
python -m pip install torch==2.12.1
python -m pip install -e '.[data,dev]'
python -m pip install duckdb
```

安装后检查 torch 实际选到的 CUDA build 是否能被 201 的驱动使用：

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]'
nvidia-smi
```

202 的 PyTorch runtime 是 CUDA 13.0，但系统 nvcc 是 CUDA 12.0；当前流程使用 PyTorch/Triton，不要求本地 nvcc 编译扩展。201 应安装与本机驱动兼容的 PyTorch CUDA wheel。不要照抄 202 的 Miniconda 路径或 GPU 规格。

## 需要留意的历史配置硬编码

- **卡数和 rank 数：** `scripts/recflow/run_expanded_daily.py` 写死可见设备 `0,1,2,3` 和 `torchrun --nproc-per-node=4`；Yambda 正式原始评价入口 `scripts/evaluate_yambda500m_foundation_raw.py` 也要求 `world == 4`；不少历史训练 execution 配置写 `world_size: 4`。两张卡不能直接照此启动。
- **GPU 型号、显存和互联：** 一些配置明确写 A40、48 GB 或 NVLink；已有显存峰值和吞吐数字依赖这些硬件。201 的 A80 需要重新确认显存、互联和可用精度，不沿用 A40 性能结论。
- **每卡 batch 与总 batch：** 配置可能写 per-rank batch、global batch、尾批处理和梯度累积。改 world size 时要重点核对 global batch 和优化器更新语义，否则会改变实验本身。
- **设备编号：** `CUDA_VISIBLE_DEVICES` 会重排卡号；例如设为 `2,3` 后，进程内设备对应 `cuda:0, cuda:1`。`--gpu`/`LOCAL_RANK` 通常是可见设备编号，不是物理编号。
- **显存上限：** `scripts/design_two/common.py` 的设备初始化把单进程显存上限设为可见 GPU 的 70%；在不同显存容量上，需确认它是否适合该任务。
- **绝对路径：** 一些旧 JSON 和少数旧脚本写有 `/home/gkl/work/evokv/...`。201 上确认数据、checkpoint 和输出目录；本 smoke 用命令行相对路径。新脚本通常从自身位置找仓库根目录，但旧入口不一定。
- **CPU 线程：** 有些命令固定 `OMP_NUM_THREADS`、`MKL_NUM_THREADS` 或配置 CPU worker 数；按 201 的 CPU 资源核对即可。

这些主要影响启动、资源预算和时间比较；遇到不匹配时优先改 201 本地副本，并保留原始配置。需要改多卡训练时再做一次小 DDP 数值 canary；本页单卡 smoke 不验证 DDP。

## 单卡环境基准：Yambda Medium

载入现有 Yambda Medium v2（6L_H192）模型，从 Medium 处理数据读一个用户的真实历史，在单张 GPU 上运行 FP32 HSTU KV 构建和一个候选的打分。它检查 CUDA、checkpoint、Parquet/DuckDB 和模型前向，不训练、不修改权重、不测论文质量或运行时间。

需要 `sync/202` 中 **D007**（Yambda 500M 处理数据）和 **Y001**（Medium v0–v5 模型；命令只读取 v2 checkpoint）。复制后保留仓库相对路径。在 201 仓库根目录运行，把 GPU 物理号 `0` 换成分配给自己的空闲卡：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python sync/环境/yambda_smoke.py \
  --dataset data/processed/yambda500m_unified_v1/scales/medium/dataset.json \
  --checkpoint results/unified_training_2026_09/medium/seed17/checkpoints/v2/checkpoint_100.pt \
  --device cuda:0
```

成功时会打印 `passed_yambda_single_gpu_smoke`，且 score/readout 都为有限值。该 smoke 只验证环境能加载现有模型和真实 Yambda 历史；不构成正式质量结果，也不验证两卡 A80 的 DDP 或两机时间对比。
