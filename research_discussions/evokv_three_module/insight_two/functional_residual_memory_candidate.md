# Insight 2 / Design 1 候选：Functional Residual Memory

日期：2026-09-02  
状态：**论文级机制候选；尚未通过 oracle coreset 与 executable constructor 裁决，不能写成冻结结论**

## 1. 为什么不再沿着 offset 或 mapping 继续做

Medium discovery 已经同时给出四条约束：

- 同请求的 S4 shared response 可以恢复 `95.34%`，说明历史侧误差经过 attention aggregation
  后确实形成了强功能边界；
- UID-disjoint 的 release basis 在 rank 8 时可恢复 `94.18%`，说明其 response range 很小；
- cutover/current correction 的方向 cosine 为 `0.9460`，但固定 offset 与 coverage scaling 分别只有
  `-34.2189` 和 `33.85%` recovery；
- 即使直接读取当前请求的 Current-Exact target，单个 oracle coefficient 也只有 `48.96%`，每层一个
  oracle coefficient 也只有 `65.04%`。

因此不能把 Insight 2 写成“存在一个稳定 AV 向量”，也不能把下一步降格成再训练一个
`query -> correction` 的 ridge/MLP。当前证据更支持：**紧凑的是跨版本 attention response 的值域，
而不是某个固定响应；真实 query、layer、time 与仍在 cache 中的 Parent support 共同决定瞬时坐标。**

## 2. 候选 Insight 2：跨版本误差是 attention response operator

对第 `l` 层 Current reader，同一个 query `q` 读取 Current 与 Parent 历史状态的差写成：

~~~text
Delta R_l(q, t)
  = sum_{i in active history(t)} [
      rho_l(q, k^C_li) v^C_li - rho_l(q, k^P_li) v^P_li
    ]
~~~

这里 `rho` 是模型原有的 query-key response kernel；HSTU 中是 activated qK，标准 softmax
Transformer 中则要同时考虑 numerator 与 normalization。这个对象有三个重要性质：

1. 它保留未来 query 的读取语义，而不是把所有 query 压成广播 offset；
2. 它是 Current 与 Parent 两个 attention measures 的 signed difference，而不是任意生成的新 token；
3. active history 随 append/eviction 改变，因此它天然要求可组合、可删除的状态，而不是时间标量。

当前待证伪的 Insight 2 候选是：

> Distributed cross-version K/V error remains non-local, but the Current reader sees it as a
> compact, query-conditioned attention-response operator whose active support evolves with the
> cache lineage.

“aggregation 后 tensor 变小”不是贡献；只有少量 query-readable state 能在 held-out candidate 和
rolling request 上因果近似这个 operator，候选 Insight 才成立。

## 3. Design 1 候选：Reuse control variate + sparse causal replay

Current reader 已经能从完整 Parent cache 精确得到 `R_P(q)`。因此 Design 不预测最终 correction，
而把 Reuse 当 control variate，只估计两版本 response 的小残差：

~~~text
R_hat_C(q)
  = R_P(q)
    + sum_{i in S} (1 / pi_i) [
        rho(q, k_tilde^C_i) v_tilde^C_i
        - rho(q, k^P_i) v^P_i
      ]
~~~

`S` 是覆盖完整历史的固定 chronological strata landmarks，`pi_i` 是冻结的 inclusion probability。
landmark 不是根据 token importance 选择的“关键 token”，而是对全历史 signed version residual 做
数值积分的固定节点。Parent 负项直接引用已有 cache；sidecar 只需持久化 approximate Current 正项、
权重、原位置和 lineage metadata。

### 3.1 Release-time constructor

1. 对完整 raw history 重算 Current layer-0 K/V。layer 0 的 projection 只依赖当前 token 输入，
   不需要展开 upper-layer dependency closure；
2. 按时间分层选择固定 landmarks，逐位置读取真实 raw event、Parent prefix 和已经生成的 earlier
   signed residual entries；
3. 用 Current block 对 landmark 做 causal replay，得到 `k_tilde^C/v_tilde^C`；
4. 可执行第二轮固定 causal refinement，使较早 landmark 产生的 defect 能继续传播；
5. 丢弃临时 layer-0 全量状态，只持久化各层 signed response entries。

这条 closure 是：

~~~text
raw landmark event
  -> exact Current layer-0 forcing
  -> complete Parent prefix as control
  -> earlier causal residual entries
  -> approximate Current landmark state
~~~

它不读取 Current Exact upper-layer K/V，不拟合 target K/V，不使用 label，也没有外部用户特征映射。

### 3.2 Request-time read

真实 serving query 在每层使用模型自身的 qK activation 读取 signed entries；其 response 加在完整
Parent-cache aggregate 之后、HSTU gate/output update 之前。不能把 entries 直接 append 到普通 cache，
否则会无意改变长度、位置和 normalization 语义。

### 3.3 Append 与 eviction transport

- 每个 entry 绑定 source stratum、原位置和 inclusion mass；Parent position 被逐出时，对应 mass
  按确定性 lineage 规则递减或删除；
- 新 event 在 corrected reader 上生成 Current K/V 并正常 append，不再把它当 Parent residual；
- refresh 只能是预注册的低频 causal replay，其摊销必须继续落在同一 `0%–20%` 预算内。

这不是 multi-version controller；全部状态仍只服务一个 `Parent -> Current` edge。

### 3.4 标准 softmax Transformer 接口

HSTU 的 pointwise unnormalized attention 允许 response residual 直接相加。softmax reader 必须分别
迁移 numerator 和 partition correction：

~~~text
N_C(q) ~= N_P(q) + Delta N_hat(q)
Z_C(q) ~= Z_P(q) + Delta Z_hat(q)
context_C(q) ~= (N_P + Delta N_hat) / (Z_P + Delta Z_hat)
~~~

因此泛化的是“signed functional residual state”，不是 HSTU 的 AV 名称或完全相同的注入算子。

## 4. 为什么它不是简单 mapping、prefix tuning 或普通 KV compression

- 没有 per-user `Mx`、ridge 或 MLP 去回归 correction；
- 没有自由 learned prompt，每个正负 entry 都有真实历史位置、版本差和 inclusion weight 语义；
- Parent response 被完整保留，sidecar 明确表达 state replacement residual，而不是额外上下文；
- 每次请求的系数由原生 Current query-key interaction 现场产生；
- append/eviction 改变的是 residual support，而不是由时间特征预测一个幅值；
- 普通 KV compression 压缩同一模型的完整 context；这里压缩的是两个 release 之间、由 Current reader
  消费的 signed compatibility defect。

与 inducing/memory-token 工作的关系只能作为 substrate 对照，不能把“使用少量 token”本身写成
创新。论文设计点必须完整包含：**version-residual decomposition、Reuse control variate、causal defect
propagation、native query read 和 lineage-aware transport。**

## 5. 两级证伪协议

### Level 1：Exact-state signed coreset oracle

先从 Current Exact/Parent cache 直接抽取固定 stratified landmarks，只测试表示是否成立：

- `R in {8,16,32,64,128}`，全部五边、同一冻结规则；
- held-out candidates 只能通过 native qK 读取 sidecar；不得拟合 score 或 response coefficient；
- `R=1024` 必须数值重建 Current Exact，作为 instrumentation；
- 比较同 storage 的 unsigned additive prefix、未加权 sampled replacement、Tail splice、fixed S4
  offset 与 full S4 oracle；
- 若 `R<=128` 在 canary 仍不能达到至少 `70%` 且五边正向，就停止这一 family，不开发 constructor。

这个阶段读取 Current Exact K/V，所以只能证明 response-operator coreset ceiling，不能进入成本 frontier。

### Level 2：Executable sparse causal replay

只有 Level 1 通过后，才冻结一份新 prospective contract：

- 对比 independent landmark step、一轮 causal replay、两轮 causal refinement；
- Current Exact 只在 construction 完成后作为 evaluator；
- 正式核算 input projection、每层 landmark-to-prefix attention、linear/gate、second sweep、state read/write
  与 request injection；
- 预算点在 `5%/10%/15%/20%` 内固定，目标 recovery 至少 `80%`、stretch `90%`；
- cutover 通过后必须再过 rolling transport；否则不能冻结为 persistent Design 1。

## 6. 当前论文口径

在 Level 1 和 Level 2 完成前，只能写：

- S4 是已观察到的最早 compact causal response boundary；
- 固定 offset、少量时间坐标、parameter probes 和 Tail-128 estimator 已被否决；
- signed attention-response memory 是由这些反例导出的新设计假设。

不能写“Insight 2 已经证明 response operator 可压缩”，也不能写“Design 1 就是 Functional Residual
Memory”。如果 native-query coreset 失败，当前证据只支持功能边界的存在，不支持该迁移对象。
