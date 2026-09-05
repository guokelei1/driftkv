# Paired coupling depth 与 fixed-rank handoff preflight

日期：2026-09-03  
状态：**单 UID、非正式机制诊断；不冻结 Insight 2，不接纳 Design 1，不读取 confirmation**

## 1. 裁决先行

这轮得到一个值得继续检验、但还不能写成论文结论的结构信号：

> 相邻 release 的 equal-resolution subtraction 不是只在 layer 0 有用；它的主要 functional
> 收益在前三个 cache-formation layer 内形成。继续把 Parent approximate arm 传播到第五、六层的
> 平均边际收益很小，但该结论尚不在所有 edge 上稳定。

固定 `d=3` 相比 `d=1` 的五边 probability-gap recovery 均值从 `0.528` 跃升到 `0.831`；相比
full-depth `d=6` 只低 `3.54` 个百分点。其严格 migration-sufficient cost 是
`809,357,126 FLOPs/user = 16.9631% Exact-All`。因此，**early coupled formation** 是一个可证伪的
Insight 2 候选。

但更强的 **upper autonomous transport** 表述没有通过这轮的完整压力测试：

- `d=3` 在 `v1->v2` 仍比 `d=6` 低 `10.39` 个 probability-recovery 百分点；
- Parent 停止后把原 two-arm active-rank budget 合并给 Current、令 upper rank 从 4 增到 8，只把均值
  从 `0.831` 提到 `0.836`；它没有补回 full coupling 的 edge-specific gap；
- 先前 single-arm rank-8 control 的五边均值为 `0.937`，明显高于 handoff 的 `0.836`。当前 handoff
  因而没有越过 generic low-rank replay baseline。

所以当前最准确的裁决是：**保留“paired cancellation 有 early formation depth”这一科学假说；否定
“upper rank handoff 已经形成论文级 Design”以及“upper autonomous transport 已被证明”。**

## 2. 固定问题与协议

本轮只读：

- frozen discovery population 的第一个 UID `1930`；
- 五条 `v0->v1,...,v4->v5` edge；
- 每条 edge 的 held-out odd-32 candidate；
- Parent/Current Medium seed-17 checkpoints 和 cutover 前完整历史。

没有读取 label、anchor candidate、`[512,3000)` confirmation，也没有训练。旧 functional-boundary
contract 中的 research-plan hash 因探索文档持续追加而不再匹配；runner 单独重验了数据 artifact、
candidate panel 和六个 checkpoint hash，只有 live research-plan hash 被允许变化。

四个 coupling depth 在执行前固定为：

```text
d in {1,3,5,6}
Current: rank=4, oversample=4, power=1, seed=17, all six formation layers
Parent:  same numerical rule, but only prefix [0,d)
U0:      paired approximate layer-0 Delta[K,V], rank=8, oversample=4,
         power=0, seed=1017
```

对每层 `l`，migration core 为：

\[
E_l^K=
\begin{cases}
U_0^\top(\widehat K_l^C-\widehat K_l^P), & l<d,\\
U_0^\top(\widehat K_l^C-K_l^P), & l\ge d,
\end{cases}
\]

\[
K_l^{mig}=K_l^P+U_0E_l^K,
\]

V 同理。`U0` 在所有 depth 间逐元素一致；因此 depth intervention 没有偷偷改变 basis。Current arm、
candidate、rank、sketch 和 seed 也不随 `d` 变化。

这里的 `d` 是 **structural profile**，不是观察五边后选择的最佳 layer。whole-history range finder
仍只是 cutover-time compiler，不被称为 tokenwise causal rematerialization。

## 3. Depth intervention 结果

下表每格为 `probability-gap recovery / logit-gap recovery`。两者都以 Current Exact 为目标、Reuse 为
零恢复基线。

| profile | v0->v1 | v1->v2 | v2->v3 | v3->v4 | v4->v5 | probability mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `d=1` | .502 / .508 | .242 / .247 | .715 / .711 | .790 / .767 | .388 / .393 | .528 |
| `d=3` | .835 / .838 | .798 / .802 | .985 / .985 | .854 / .839 | .682 / .686 | .831 |
| `d=5` | .875 / .878 | .894 / .896 | .998 / .997 | .866 / .853 | .709 / .713 | .868 |
| `d=6` | .870 / .873 | .901 / .904 | .985 / .985 | .869 / .855 | .706 / .710 | .866 |

三个观察需要同时保留：

1. `d=1 -> d=3` 是最大的结构跃升：probability mean `+30.32` 个百分点。只在 dependency-free
   layer 0 做 matched subtraction 不够，至少还需要早期 contextual blocks。
2. `d=3 -> d=5` 只有 `+3.76` 点，`d=5 -> d=6` 反而为 `-0.22` 点。full-depth symmetric arm
   不是所有 edge 都需要。
3. `d=3` 与 `d=6` 的平均差只有 `-3.54` 点，但 `v1->v2` 单边差为 `-10.39` 点，不能用均值把
   edge heterogeneity 隐藏掉。按 probability recovery，`d=3` 是 3/5 edge 超过 `.80`，另有一条
   `.798`；它符合“值得进入 canary”的弱信号，不是 admission pass。

## 4. 严格静态成本

口径沿用现有 Medium ledger：`N=1024,H=192,L=6,heads=6`，multiply-add=`2 FLOPs`，
Exact-All 分母为 `4,771,282,944 FLOPs/user`。成本包含两份 model-specific input、固定 range finder、
三角 causal pair、activation arithmetic、gate/residual dense boundary、`U0` 和所有 signed-core build。

每条轨迹最后一个**被需要的 cache layer**只做 RMSNorm 和 K/V projection；它的 Q、attention、gate、
post-block state 与下一次 recompression 没有 consumer，因而不是 Design 所需计算。为了不把当前通用
semantic prototype 冒充 executor，下表同时列出它仍计算 terminal full block 的上界。

| d | Parent propagation | upper exact-Parent cores | migration-sufficient FLOPs | Exact-All | generic full-block prototype |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 102,085,206 | 31,923,200 | 654,783,130 | **13.7234%** | 17.1960% |
| 3 | 269,061,890 | 19,153,920 | 809,357,126 | **16.9631%** | 20.4357% |
| 5 | 436,038,574 | 6,384,640 | 963,931,122 | **20.2028%** | 23.6754% |
| 6 | 519,526,916 | 0 | 1,041,218,120 | **21.8226%** | 25.2952% |

这里修正了一个容易混淆的 rank 口径：`d<6` 的 upper Current arm 是 rank 4，所以每层
`Current-factor - exact-Parent` core 成本为 `6,384,640`，不能沿用 single-arm rank-8 control 的
`6,474,752`。`d=6` 没有这类 upper core，因此仍精确复现既有 `21.8226%` ledger。

所有 depth 的 persistent sidecar 相同：`26,624 scalars`，即 full Current KV 的约 `1.1285%`。

## 5. 唯一 fixed-rank handoff

为检验 `d=3` 以后性能差距是否只是 upper Current capacity 不足，本轮只增加一个、执行前固定的
handoff，不做 rank/depth grid：

```text
layers 0,1,2: Parent rank4 + Current rank4, matched subtraction
after layer 2: Parent stops; its active rank budget is handed to Current
layers 3,4,5: Current rank8 only; cores use Current8 - exact Parent
final layer:   K/V only
U0 and early signed cores: exactly unchanged from d=3
```

runner 对 `U0` 与前三层 K/V signed cores 做了逐元素相等断言。结果为：

| profile | v0->v1 | v1->v2 | v2->v3 | v3->v4 | v4->v5 | probability mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `d3: C4 upper` | .835 / .838 | .798 / .802 | .985 / .985 | .854 / .839 | .682 / .686 | .831 |
| `d3 handoff: C4->C8` | .833 / .836 | .800 / .805 | .973 / .973 | .857 / .842 | .718 / .722 | .836 |
| `d6: P4+C4` | .870 / .873 | .901 / .904 | .985 / .985 | .869 / .855 | .706 / .710 | .866 |

handoff 相对原 `d=3` 只有 `+0.55` 个 probability-recovery 点；五边变化为
`-0.19/+0.27/-1.21/+0.26/+3.62` 点。它相对 `d=6` 仍低 `2.99` 点，且 `v1->v2` 低
`10.12` 点。因此：

> Parent 停止后的质量缺口不是简单的 upper active rank 不足；把 rank 4+4 合并成 Current rank 8
> 不能替代 upper matched subtraction。

### 5.1 handoff cost 与 executor sensitivity

handoff 的原始、完整静态账本是：

| component | FLOPs/user |
| --- | ---: |
| Current rank4->rank8 arm | 669,327,938 |
| Parent rank4 prefix to `d=3` | 269,061,890 |
| paired layer-0 `U0` builder/core | 1,247,808 |
| paired cores at layers 1--2 | 366,592 |
| Current8-exactParent cores at layers 3--5 | 19,424,256 |
| **total** | **959,428,484 = 20.1084% Exact-All** |

所以 scientific handoff **单独仍然 FAIL 20% gate**，超出约 `5.17M FLOPs/user`。

仓库另已验证 initial factor 的 matrix-free 等价实现：每臂把
`raw dense input + initial compression = 101,441,366` 改写为 `18,033,494 FLOPs`，不改变 rank、
seed、range-finder semantics 或输出 factors。把该 executor component 与 handoff 组合的明确 sensitivity
是：

\[
959,428,484-2(101,441,366-18,033,494)
=792,612,740,
\]

即 **16.6122% Exact-All**。这只说明已有等价 kernel 能把构造落回甜点区；matrix-free randomized
range finding 是经典 operator rewrite，**不是 scientific handoff 的机制，也不是 Insight 2 创新点**。

## 6. 与 strongest generic control 的不利比较

先前同一 UID/五边的 single-arm rank-8 shared-layer0 splice 为：

```text
.861 / .917 / .985 / .947 / .975, probability mean=.937
```

它五条 edge 全部超过 `.80`，比 handoff 均值高 `10.09` 点。原冻结 full-block ledger 是
`22.8028%`；按本轮所有方法一致的 terminal-KV specialization，删除没有 consumer 的最后 block
output/recompression 后是 `934,810,304 FLOPs/user = 19.5924%`。后者是静态 matched sensitivity，
不是当前 dense prototype 的 wall-time 实现。

| method | probability mean | edges >=.80 | comparable terminal-KV cost | 研究身份 |
| --- | ---: | ---: | ---: | --- |
| `d3 P4+C4` | .831 | 3/5 (+ one at .798) | 16.9631% | early-coupling hypothesis |
| `d3 handoff C4->C8` | .836 | 4/5 | 20.1084%; 16.6122% with matrix-free input | current candidate **fails control** |
| `d6 P4+C4` | .866 | 4/5 | 21.8226%; 18.3264% with matrix-free input | full coupled diagnostic |
| single-arm `C8` | .937 | 5/5 | 19.5924% matched sensitivity | generic/xKV-adjacent control |

这张表是当前最重要的负面约束：即使 early coupling 的机制信号真实，当前表示/注入方式仍被 generic
single-arm rank-8 control 在质量上明显支配。不能因为 handoff 加上 matrix-free 后更便宜，就把
`10.09` 点平均 recovery 差距省略掉。

## 7. 可证伪解释与下一步

当前允许提出的窄假说是：

> Equal-resolution Parent/Current subtraction 需要穿过最初几个 attention--residual blocks，才能把
> raw version perturbation 转成 reader-relevant release difference；形成以后，多数 edge 的 upper cache
> 可以用 Current-only trajectory 相对 exact Parent 编译，但这个 formation depth 具有 edge/user
> heterogeneity，且不等价于把更多 low-rank capacity 分给 Current。

它做出以下可证伪预测：

1. 在 prospective 32-user canary 中，`d=1 -> d=3` 的 paired gain 应稳定存在，而不是 UID1930 特例；
2. stage diagnostic 中，`E_l^C-E_l^P` 对 reader response 的相消收益应集中在 early layers；如果 upper
   layers 仍系统需要 matched Parent response，拒绝“upper autonomous”部分；
3. `d=3` 与 `d=6` 必须在冻结 equivalence margin 下比较全部 user-edge，不能只比较 edge mean；
4. strongest single-arm rank-8 control 必须进入同一 runner、同一 cost semantics。若 early-coupled path
   不能在 matched compute/storage 下给出质量、persistence 或 lineage 上的独立优势，它仍应归类为
   generic low-rank compression 的复杂变体；
5. rolling append 若令 frozen `U0` 和 early cores 迅速失效，则 persistent migration object 不成立。

在正式 canary 前必须先冻结 equivalence margin、matched-control runner 和 rejection rule；不能根据本 UID
把 boundary 改成别的 layer，也不能继续扫描 rank。当前这轮没有授权读取 confirmation。

## 8. 实现与验证

- `scripts/insight_two/coupling_depth_replay.py`：Parent prefix、depth splice、唯一 rank handoff 与静态成本；
- `scripts/insight_two/run_coupling_depth_preflight.py`：固定 UID/五边非正式 runner；
- `tests/test_insight_two_coupling_depth_replay.py`：prefix 语义、full-rank exact limit、U0/early-core
  invariance、cost ledger；
- `scripts/insight_two/matrix_free_input_range.py`：独立、等价的 initial-factor executor component。

验证结果：coupling-depth 与 matrix-free 两组 focused tests 共 `19 passed`，相关文件 `ruff check` 通过。
