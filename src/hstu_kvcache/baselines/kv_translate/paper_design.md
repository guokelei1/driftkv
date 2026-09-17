# Cross-Model KV Cache Transfer：原论文核心 Design 笔记

整理日期：2026-09-16。中文概述，非逐字翻译；只保留闭式 mapper 主路径，不含背景、实验结果或本仓库的适配方案。

来源：Taekyung Heo 等，*Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse*，arXiv:2608.03893v1。[原论文](https://arxiv.org/html/2608.03893v1)，核心依据为 [§3.1–3.3](https://arxiv.org/html/2608.03893v1#S3)、Eq.(3)–(5)；源层探针定义见 §2.3、Eq.(2)。原文标注 CC BY 4.0。本仓库方案另见 [README](README.md)。

## 输入、输出与设计分解

给定 source 和 target 对同一 token 序列的缓存，用共享校准数据拟合一个有向 source→target mapper。推理时把 source K/V 转为 target 可直接读取的 K/V，省去 target 对共享前缀的完整 prefill。后续 token 仍由 target 正常计算。

令 source／target 层数为 `L_s/L_t`，KV heads 数为 `H_s/H_t`，每 head 宽度为 `d_s/d_t`，单层缓存宽度 `D_s=H_s d_s`、`D_t=H_t d_t`。映射分为三部分：**选源层、拟合仿射 ridge、分离 K 的 RoPE**。不同目标层可使用不同源层集合。

## 1. 为每个目标层选 top-k 源层（§3.2）

先用单源线性探针估计源层到目标层的可预测性。§2.3 的探针以对应 head 为单位，分别预测去 RoPE 的 K 和 V；对 heads 及 K/V 的 R² 取平均，得到每个 `(source layer,target layer)` 的分数。

每个目标层 l 选择分数最高的 k 个源层，形成集合 `S_l`。该目标层的所有目标 heads 和 K/V 共用这组源层；k 是按模型对选择的超参数。实际 mapper 使用 top-k 排序，不能把论文结构分析中的 greedy forward selection 当成其生产映射算法。

随后将所选源层的**全部源 heads**拼接。以 K 为例：

\[
X_K^l=\operatorname{concat}_{s\in S_l}\bar K_s
\in\mathbb R^{N\times kD_s},
\]

其中 N 为 token 行数，`bar K_s` 表示该层去掉 RoPE 后、展平所有 heads 的 K。`X_V^l` 同样由所选层的 V 构造。K 不与 V 交叉拼接。

## 2. K/V 独立、按目标 head 的 affine ridge（§3.1）

对目标层 l、目标 head h：

\[
\widehat{\bar K}_t^{l,h}=X_K^l W_K^{l,h}+b_K^{l,h},
\qquad
\widehat V_t^{l,h}=X_V^l W_V^{l,h}+b_V^{l,h}.
\]

`W` 为 `[kD_s,d_t]`，bias 为 `[d_t]`。K/V 参数独立，各目标 heads 的参数也独立。**Per-head 指目标输出的参数划分，不是只能读取同编号的源 head。** 单源探针与最终 mapper 的输入结构不同。

对成对校准行 `X,Y`，先中心化：

\[
X_c=X-\mu_X,\qquad Y_c=Y-\mu_Y.
\]

然后按 Eq.(4) 求解并恢复截距：

\[
W=(X_c^\top X_c+\lambda I)^{-1}X_c^\top Y_c,
\qquad b=\mu_Y-\mu_XW.
\]

原文取 `λ=0.01`；Gram 公式未按 token 数归一，bias 通过均值恢复。正则化用于处理多个相关源层导致的病态 Gram。拟合无需梯度训练，但需要 source／target 成对缓存和线性代数计算。

## 3. 在内容空间映射 K（§3.3）

RoPE 是随 token 位置变化的旋转。直接拟合已旋转 K，会把位置变换与内容映射耦合；论文将两者拆开：

1. 对每个源层、每个源 head 的 K，按该 token 位置撤销 source RoPE，再拼接成 `X_K`。
2. 校准时也撤销 target 教师 K 的 RoPE，作为回归目标 Y；V 直接作为目标。
3. 应用 mapper 后，对每个目标 head 的预测内容 K 施加 target RoPE，得到真正安装的缓存 K。

用行向量约定，单个旋转块可写成：

\[
K_s^{\mathrm{content}}(i)=K_s^{\mathrm{cache}}(i)R_s(i)^{-1},
\qquad
\widehat K_t^{\mathrm{cache}}(i)
=\bigl(X_K(i)W_K+b_K\bigr)R_t(i).
\]

旋转矩阵正交，因此逆等于转置。不同源 heads／层先各自去旋转，不能把跨层拼接向量误当成一个 head 处理。V 无 RoPE，不执行这一步。

## 4. 从校准到推理

```text
离线，每个有向模型对：
  同一序列 → source / target 成对缓存
  源和目标 K 去 RoPE
  单源探针 → 每个目标层的 top-k 源层
  拼接各源层全部 heads → K/V 分开拟合 centered ridge
  保存源层集合、W、bias 及所需位置配置

在线，给定 source 的前缀缓存：
  源 K 去 RoPE
  各目标层 gather → 仿射映射 K/V
  目标 K 加 RoPE → 构造 target 缓存
  target 使用转换后的前缀继续计算
```

这是对正文计算依赖的概括，不是作者代码或论文编号算法。

## 5. 低成本来自哪些计算结构

同一目标层的各输出 heads 使用相同 X，因此形成 `XᵀX` 的工作可共享；K 与 V 的 X 不同，Gram 分开。目标 heads 虽共享输入和 Gram，参数仍各自独立。[§3.1](https://arxiv.org/html/2608.03893v1#S3.SS1)

映射参数主项为：

\[
2L_t(kD_s)D_t,
\]

另有 `2L_tD_t` 个 bias。由这些矩阵尺寸可得，处理 N 个缓存 token 的矩阵乘主项为 `2NL_tkD_sD_t` MAC；这是公式推算，不是测量结果。

在线没有历史 attention 的 token 两两计算；固定模型和 k 时，映射工作随 N 线性增长。代价仍包括全部目标 K/V 写出、源层读取／拼接和临时张量；离线则包括成对缓存生成和 ridge 拟合。设计提供的是相对重做 prefill 的替代计算路径，不保证任意模型对、k 或上下文长度下都更便宜。
