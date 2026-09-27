# HSTU 共享算子升级：2026-09-19

用户授权优化当前训练和评测，并使用已有六层模型验证。本轮实现共享的 Triton
pointwise attention 前向／反向；保留 PyTorch 模型、优化器、参数名称、checkpoint
架构字典和 K/V 布局。没有重训或覆盖已有模型，没有读取 theta3 或 final-role 数据。
这是执行后端的数值与性能验证，不是模型准入、推荐质量或适配方法结果。

## 已接入的范围

- `PointwiseAttention` 的完整序列、普通缓存追加、stale-KV 路径共享融合核，
  支持 legacy ELU+1、reference SiLU、ReLU、相对位置偏置及其梯度、因果边界和正窗口。
- 默认 `EVOKV_ATTENTION_BACKEND=auto`。CUDA FP32/BF16、head_dim 32/64/128、
  无激活中的 attention dropout、支持的张量布局与位置范围使用 Triton。
  CPU、其他 dtype/head_dim、自定义 mask 等继续使用原 PyTorch 算式。
- `EVOKV_ATTENTION_BACKEND=torch` 强制原算式；`triton` 请求融合路径，但不绕过
  上述兼容性限制。也可逐个修改 `PointwiseAttention.backend`，不进入 state_dict。
- 既有 `torch.compile` 区域保留其 Inductor 路径；首次编译与热态计时分开。
  本机旧默认 g++ 不支持当前 PyTorch 所需的 C++20；编译检查使用
  `CXX=/usr/bin/g++-13`，不修改系统编译器或环境依赖。
- Yambda 主训练器／release evaluator、RecFlow window_chain／parallel_evaluate
  的新输出记录后端策略、版本和执行源码 hash。历史 seals 保留原样。

融合核采用分块计算，不创建完整 `[B,H,L,L]` attention／bias 中间张量。
FP32 点积采用 TF32x3；BF16 保留各阶段舍入。位置偏置梯度以 FP32 原子累加，
与旧实现 BF16 批次归约略有差异，因此不承诺逐位复现或后续训练轨迹相同。
分块方法参考 [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)，
实际激活、缩放和 bias 梯度按本仓库 HSTU 方程实现，未替换为 softmax attention。

## 保留测量

[完整结果](summary.json)记录了源码起止 hash、checkpoint hash、设置、各参数梯度误差、
热态／首次调用耗时和峰值 allocated 显存；全部预设数值检查通过，测量期间源码未变化。
环境：NVIDIA A40、PyTorch 2.12.1+cu130、Triton 3.7.1，seed17。
合成输入为六层 H192／6 heads、batch8、长度1024，末行有效长度512；
每项预热3次、计时10次，使用同步 wall time。模型以 eval 模式关闭随机 dropout，
反向依然执行；训练耗时不包含数据装载、推荐头、优化器更新和通信。

| 六层 backbone | 精度 | Full+KV：torch → Triton | 前向+反向：torch → Triton | 反向测量峰值显存 MiB |
| --- | --- | --- | --- | --- |
| legacy | FP32 | 30.09 → 6.40 ms，4.70× | 67.99 → 22.30 ms，3.05× | 2917 → 438 |
| legacy | BF16 | 20.46 → 3.30 ms，6.20× | 48.90 → 11.04 ms，4.43× | 1867 → 351 |
| reference | FP32 | 47.47 → 7.11 ms，6.68× | 97.81 → 29.29 ms，3.34× | 3080 → 553 |
| reference | BF16 | 30.20 → 4.00 ms，7.55× | 68.37 → 14.18 ms，4.82× | 1984 → 430 |

已有 RecFlow A3 checkpoint 的8条真实 D19 development 请求，历史长度均1024：

- BF16 NLL 前向+反向（包含模型、生成头和 loss）：87.38 → 28.65 ms，**3.05×**；
  峰值显存3.718 → 2.454 GB，下降34.0%。不包含优化器和数据装载。
- 其中2条请求各1000候选的完整 FP32 `score_items`：394.97 → 387.34 ms，
  **1.02×**。该路径仍有大量逐请求／类别分支处理，本轮未解决；不能把 backbone
  的提速倍数写成整个候选评测的提速。
- FP32／BF16 NLL 绝对差均约9.54e-7，FP32候选分数最大差3.81e-6，
  所检查候选池 Top50 顺序一致。这里只验证固定样本数值，不增加质量指标或改候选协议。
- 同时严格加载本轮 Yambda Medium V0 checkpoint，用合成输入检查 Full、K/V、
  缓存追加与固定候选评分；Top50 顺序一致，未读取 Yambda 资格／最终用户数据。
- 六层合成 BF16 的最坏参数梯度相对 L2 差0.459%；实际 RecFlow FP32 最坏
  参数梯度相对 L2 差6.12e-6。非零误差完整保留。

初版 IEEE FP32 点积在 D64 反向时慢于 PyTorch，开发中改为 TF32x3 并重新核对误差。
较早探索测量的候选评分约1.07×，最终重复测量为1.02×；不宣称候选评测获得稳定大幅收益。
单次短基准不代替整轮训练时长估计。

## 正确性和兼容性

- 8项 CUDA／CPU 后端检查通过：输出、输入／参数／bias 梯度、旧 K/V、因果边界、
  窗口和显式自定义 mask 回退；原 checkpoint 严格加载通过。
- 18项缓存／状态转移／RecFlow 检查（含新增2项 append 回归）和18项基础训练、
  缓存谱系、三种 competitor 原语检查通过；未运行全量 suite。
- 六层 fullgraph+dynamic 编译 prefix 检查通过；首次调用47.23秒，K/V 最大误差
  约1.04e-6／1.07e-6。默认旧 g++ 的 C++20 失败属于环境，不吞掉编译失败。
- [两卡 FSDP canary](fsdp/canary.pass.json)：六层 BF16 参数（含 bias）、FP32
  梯度归约，一次 AdamW 更新。它验证训练接线与数值范围，不建立长期训练等价性。

检查还发现旧 `forward_one_with_cache_new_kv` 专用公式仅适用于 legacy/inclusive：
reference 缺少固定除数、exclusive 错误计入 self。本轮让这两种设置使用普通缓存公式，
新增2项回归；legacy/inclusive 原无复制快路保留。reference/exclusive 分支因此会临时
拼接缓存。普通 RecFlow decoder 原本调用正确的普通路径，不受此历史问题影响；
历史结果没有回写或重新解释。

## 使用与复查

原训练／评测命令无需改参数即可使用默认 auto。复现旧算子时在命令前设置
`EVOKV_ATTENTION_BACKEND=torch`。严格历史复现还需使用当时源码和环境。
比较 Exact、Reuse、competitor 的计算成本时使用同一后端与精度，重新测量各项；
不能将旧 eager 时间与新融合时间混作同一实验。

```bash
PYTHONPATH=src python scripts/benchmark_hstu_backends.py --device cuda:1 \
  --yambda-checkpoint results/unified_training_2026_09/medium/seed17/checkpoints/v0/checkpoint_100.pt \
  --recflow-checkpoint results/recflow/development/window_6l_full_seed17/A/A_epoch3/checkpoint.pt \
  --output /tmp/hstu_backend_recheck.json

CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=src torchrun --standalone --nproc-per-node=2 \
  scripts/check_hstu_backend_fsdp.py --output /tmp/hstu_fsdp_recheck.json
```

两条命令都是短开发检查，不启动完整训练；输出须选择未存在的路径。
现有 checkpoint／optimizer 格式和缓存布局不变，无须转换或重训。
本轮没有做十层／Max 性能资格，也没有重写 RecFlow 解码调度或自定义适配 reader。

后续[16层Max完整训练步诊断](../max_profile_2026_09_19/README.md)补充了四卡、
完整词表、FSDP和AdamW测量：默认Triton使step耗时下降19.48%，不能把上述
backbone局部倍数用作整轮训练收益。
