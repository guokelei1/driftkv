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

完整快照见 [`202_snapshot.txt`](202_snapshot.txt)。项目基础依赖见 [`pyproject.toml`](../../pyproject.toml)：Python ≥3.10、torch ≥2.3、numpy ≥1.26、pandas ≥2.0、scipy ≥1.11、jsonschema ≥4.18；`data` extra 安装 PyArrow、DuckDB、PyYAML、scikit-learn，覆盖数据读取和当前 AUC 评价入口；`dev` extra 安装 pytest/ruff。当前基准不需要 torchrec 或 fbgemm_gpu。

### 201 建立环境

202 用的是 Miniconda 发行版提供的 Conda，不是 Anaconda，也没有专门的 `evokv` 环境。201 在 2026-10-21 实测没有可用的 Conda，已采用用户目录的 uv 和仓库 `.venv`，无需 sudo。日常运行先在仓库根目录执行：

```bash
source .venv/bin/activate
```

该环境与其他用户及系统 Python 隔离。若使用 Conda，以下是另一种建立方式：

```bash
conda create -n evokv python=3.13 pip -y
conda activate evokv
python -m pip install torch==2.12.1
python -m pip install -e '.[data,dev]'
```

安装后检查 torch 实际选到的 CUDA build 是否能被 201 的驱动使用：

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]'
nvidia-smi
```

202 的 PyTorch runtime 是 CUDA 13.0，但系统 nvcc 是 CUDA 12.0；当前流程使用 PyTorch/Triton，不要求本地 nvcc 编译扩展。201 应安装与本机驱动兼容的 PyTorch CUDA wheel。不要照抄 202 的 Miniconda 路径或 GPU 规格。

### 201 实际配置与复建

201 已安装 Python 3.13.16、PyTorch 2.12.1+cu130、Triton 3.7.1、cuDNN 9.20.0 和 NCCL 2.29.7；主要数据/数值包按 202 快照固定版本。Python patch 版本与 202 的 3.13.12 不同。硬件及软件记录见 [`201_snapshot.txt`](201_snapshot.txt)，完整依赖固定在 [`201_requirements.txt`](201_requirements.txt)，不含仓库自身的绝对路径。

从空环境复建（uv 已安装在 `~/.local/bin/uv`）：

```bash
~/.local/bin/uv venv --python 3.13.16 .venv
~/.local/bin/uv pip install --python .venv/bin/python --link-mode=copy \
  -r sync/环境/201_requirements.txt -e '.[data,dev]'
```

本机已配置的代理下载大 wheel 较慢，本次安装临时清除安装进程的 `http_proxy`、`https_proxy` 及其大写变量后，直连 PyPI 完成；未更改全局代理设置。运行实验无需重新下载安装。

### 无数据时的最小验证

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python sync/环境/gpu_smoke.py
```

本脚本使用随机初始化的 2L/H64 模型和合成输入，在每张可见卡上检查 FP32 前向与 CPU 对照、同模型 Full/Reuse 一致性、反向梯度，以及 Triton JIT；同时检查 Parquet/DuckDB 读写。仅验证计算环境，不更新或保存模型权重，不替代真实模型实验，也不验证 DDP。

2026-10-21 在 201 的两张 A800 上通过。CPU/GPU 最大打分误差约 `6.26e-7`，同模型 Full/Reuse 打分差为零，梯度有限；结果见 [`summary.json`](../../results/environment_201_2026_10/summary.json)。环境依赖检查也已通过。

仓库现有追加 K/V 和融合读取检查也已通过（16 passed）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m pytest tests/test_append_only_kv.py \
  tests/implementation_2026_10/test_reader.py -q
```

记录见 [`checks/summary.json`](../../results/environment_201_2026_10/checks/summary.json)。这验证了 A800 上的原生 attention 和当前融合读取核，不代表已复现真实模型质量。

### 两卡 DDP 最小实验

2026-10-21 在 201 的两张 A800 上通过 NCCL 通信及 HSTU DDP 更新检查。模型为随机初始化的 2L/H64，合成输入，总 batch 8、每卡 4、历史 32、候选 2；FP32，关闭 dropout/TF32，SGD（lr=0.01、momentum=0.9）更新 3 步。没有加载、训练或保存基础模型 checkpoint。

每一步均对照相同全局 batch 的单进程计算，比较 loss、梯度和更新后的参数；两卡副本参数逐位一致，DDP 与单进程的最大梯度/参数差约 `1.19e-7`。结果和命令见 [`ddp/summary.json`](../../results/environment_201_2026_10/ddp/summary.json)。这确认本机能执行小规模两卡训练，不是质量、吞吐或历史四卡训练入口等价性的证明。

复跑时选择一个新的输出路径，保留已有结果：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  sync/环境/ddp_smoke.py --output results/environment_201_2026_10/ddp_repeat/summary.json
```

## 需要留意的历史配置硬编码

- **卡数和 rank 数：** `scripts/recflow/run_expanded_daily.py` 写死可见设备 `0,1,2,3` 和 `torchrun --nproc-per-node=4`；Yambda 正式原始评价入口 `scripts/evaluate_yambda500m_foundation_raw.py` 也要求 `world == 4`；不少历史训练 execution 配置写 `world_size: 4`。两张卡不能直接照此启动。
- **GPU 型号、显存和互联：** 一些配置明确写 A40、48 GB 或 NVLink；已有显存峰值和吞吐数字依赖这些硬件。201 实测为两张 A800 80GB PCIe，拓扑显示 NV8 互联；不沿用 A40 性能结论。
- **每卡 batch 与总 batch：** 配置可能写 per-rank batch、global batch、尾批处理和梯度累积。改 world size 时要重点核对 global batch 和优化器更新语义，否则会改变实验本身。
- **设备编号：** `CUDA_VISIBLE_DEVICES` 会重排卡号；例如设为 `2,3` 后，进程内设备对应 `cuda:0, cuda:1`。`--gpu`/`LOCAL_RANK` 通常是可见设备编号，不是物理编号。
- **Triton 当前设备：** 本次 smoke 初版在同一进程从卡 0 切到卡 1 时遇到指针设备错误；显式用 `torch.cuda.device(device)` 包住该卡的计算后，两卡均通过。旧实验优先每进程只暴露分配的单卡；单进程跨卡入口需同时设置当前设备，不能只把张量移动过去。
- **显存上限：** `scripts/design_two/common.py` 的设备初始化把单进程显存上限设为可见 GPU 的 70%；在不同显存容量上，需确认它是否适合该任务。
- **绝对路径：** 一些旧 JSON 和少数旧脚本写有 `/home/gkl/work/evokv/...`。201 上确认数据、checkpoint 和输出目录；本 smoke 用命令行相对路径。新脚本通常从自身位置找仓库根目录，但旧入口不一定。
- **CPU 线程：** 有些命令固定 `OMP_NUM_THREADS`、`MKL_NUM_THREADS` 或配置 CPU worker 数；按 201 的 CPU 资源核对即可。

这些主要影响启动、资源预算和时间比较；遇到不匹配时优先改 201 本地副本，并保留原始配置。本页两卡 DDP smoke 已验证上述小模型的更新等价性；正式训练入口改成两卡时，仍需核对其采样、总 batch、梯度累积和优化器设置。

## 单卡环境基准：Yambda Medium

载入现有 Yambda Medium v2（6L_H192）模型，从 Medium 处理数据读一个用户的真实历史，在单张 GPU 上运行 FP32 HSTU KV 构建和一个候选的打分。它检查 CUDA、checkpoint、Parquet/DuckDB 和模型前向，不训练、不修改权重、不测论文质量或运行时间。

**201 已于 2026-10-21 通过真实模型 smoke。** 从 202 经 SSH 拉取下述选定文件（47 个，3,806,520,706 bytes，约 3.55 GiB），全部 SHA-256 与来源一致。使用 A800 物理卡 0、真实 Medium v2、UID 60 的 512 条历史，打分为 `2.489431142807007`，score/readout 均有限。结果和命令见 [`yambda/summary.json`](../../results/environment_201_2026_10/yambda/summary.json)，逐文件传输校验见 [`transfer/summary.json`](../../results/environment_201_2026_10/transfer/summary.json)。这不是反馈 AUC 或两机质量一致性的证明。

需要 `sync/202` 中 **D007**（Yambda 500M 处理数据）和 **Y001**（Medium v0–v5 模型；命令只读取 v2 checkpoint）。复制后保留仓库相对路径。在 201 仓库根目录运行，把 GPU 物理号 `0` 换成分配给自己的空闲卡：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python sync/环境/yambda_smoke.py \
  --dataset data/processed/yambda500m_unified_v1/scales/medium/dataset.json \
  --checkpoint results/unified_training_2026_09/medium/seed17/checkpoints/v2/checkpoint_100.pt \
  --device cuda:0
```

成功时会打印 `passed_yambda_single_gpu_smoke`，且 score/readout 都为有限值。该 smoke 只验证环境能加载现有模型和真实 Yambda 历史；不构成正式质量结果，也不验证两卡 A800 的 DDP 或两机时间对比。

若只为这个 smoke 搬文件，可以从 202 单独复制以下范围，无需整份 D007/Y001：

```text
data/processed/yambda500m_unified_v1/scales/medium/dataset.json
data/processed/yambda500m_unified_v1/scales/medium/users.parquet
data/processed/yambda500m_unified_v1/scales/medium/item_mapping.parquet
data/processed/yambda500m_unified_v1/shared/listens/  （保留全部分区及目录结构）
results/unified_training_2026_09/medium/seed17/checkpoints/v2/checkpoint_100.pt
```

这些是现有 smoke 所读的文件，已复制到 201 并更新 `sync/201`；其他版本 checkpoint、feedback 和冻结面板未随本次传输复制，需要真实反馈评价时再按实验补齐。201 的 `/data` 在本次复制后剩余约 319 GiB，后续复制前仍需按源清单核对容量。
