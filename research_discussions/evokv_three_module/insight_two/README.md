# Insight 2 / Design 1 探索入口

状态：**当前 KV-only 接口已完成阶段裁决；Migration Sketch 的 prospective mechanism 已形成，方法有效性与最终 admission 待验证。**

本目录只研究一次 `Parent -> Current` release 中，Transformer persistent
state 是否存在比 token/layer-local K/V 更合适的功能迁移边界。当前全部已测 KV-only
paired/defect/source/operator/suffix 路线均已退休；generic single-C8 只是数值硬对照，PRO、AV offset
和 reader correction 只是历史先验与 baseline。后继 Design 是在 state creation 时主动写入
Migration Sketch，不是复活任何已退休的 Parent-KV-only constructor。

当前 mechanism 的核心已在 Iteration 29 收紧为两条不可分割的约束：producer-time common-mode
certificate-error training，以及 migration-aware Full/append 共用的 sketch-conditioned clean state writer
$G_v$。旧 Iteration 28 的“两个 same-version decoder + ordinary append”与旧成本数字只保留为历史记录，
不再代表当前设计。

- [论文材料统一稿](../../../docs/insight2_design1_expert_brief.md)：单篇汇总 Motivation、Insight 1、
  Insight 2 结果、Migration Sketch 的状态/公式/成本/实验设计及论文表述；用于讨论，不替代下列原始
  证据入口，也不把待测 Migration Sketch recovery 写成既成结果。
- [探索计划](insight_two_exploration_plan.md)：问题、阶段协议、证据层级、准入门槛、迭代路线与停止条件。
- [探索日志](exploration_log.md)：按时间追加资产核对、每轮假设、执行、结果、裁决和下一步。
- [当前 KV-only 接口裁决](current_kv_only_interface_adjudication.md)：统一解释“finite-query functional
  compactness 不等于 generator closure”，汇总强对照、成本障碍、创新裁决与重新开启条件。
- [Functional Residual Memory 历史假设](functional_residual_memory_candidate.md)：保留 response-operator
  推导与两级证伪协议；其 sparse/operator 实现已被后续结果否决，不再是 active candidate。
- [Attention-address follow-up](address_aware_response_followup.md)：chronological coreset 失败后，冻结
  “地址空间而非时间空间”的单变量检验，以及最终 Design 必须超过普通 clustering/mapping 的贡献门。
- [Attention-cone response moments](attention_cone_response_moments.md)：由两类 K/V coreset 反例继续
  推出的新结构假设；检验推荐 query cone 是否把 HSTU 的跨版本响应精确收敛成用户级 `B/M` moments。
- [Paired functional-delta canary 裁决](paired_functional_delta_canary_adjudication.md)：正式否定 sparse
  causal-closure Design，并冻结“functional compactness 不等于 token-support sparsity”的下一轮边界。
- [History-mode replay preflight](history_mode_replay_preflight.md)：保留 support-dense/mode-compact 的结构
  线索；其中 single-arm replay 已因 prior-art 碰撞降级为 compression control。
- [History-quotient related-work boundary](history_quotient_related_work_boundary_preflight.md)：逐项审计
  xKV、DroidSpeak、MobiLoRA、ForkKV 等直接邻近工作，冻结哪些低秩/base-plus-delta claim 不能作为创新。
- [Mode-space 成本审计](mode_space_current_rematerialization_preflight.md)：保留双臂 dense-input 原始账本与
  novelty 边界；其 21.82% KV-only 成本已由后续 matrix-free 等价 executor 降到 18.33%，但机制仍未准入。
- [Paired release-differential 机制审查](paired_release_differential_mechanism_review.md)：用显式误差分解
  区分 representation failure 与 paired-error cancellation，并冻结 matched-compute、depth 和 persistence
  的五组否证实验；当前证据不足以宣称 paired 优于 single-arm。
- [Matched Parent-cache differential preflight](matched_parent_cache_differential_preflight.md)：约 14.15%
  的静态 Parent-cache control-variate shortcut 在单 UID 五边仅恢复 0.267，因而退休；正信号要求
  trajectory-matched Parent replay，而非事后低秩 cache approximation。
- [Matrix-free paired input preflight](matrix_free_paired_input_preflight.md)：保持 paired rank/oversampling/
  power 与双臂语义不变，以输入算子形式消除 dense temporal/in-projection 物化；完整 KV-only 账本
  降至 18.3264%，但仍只是待 formal mechanism gate 的 executor component。
- [Coupling-depth preflight](coupling_depth_preflight.md)：固定 `d={1,3,5,6}` 检验 matched release
  subtraction 的形成深度，并用唯一 `d3: Current 4->8` handoff 否证 upper-rank capacity 解释；当前只
  保留 early-coupling 假说，handoff 被 single-arm rank-8 control 支配，未接纳为 Design。
- [Paired S4 functional-boundary preflight](paired_functional_boundary_preflight.md)：paired functional
  compiler 在 19.66% 内达到 0.900 mean recovery，但被 single-arm r8 的 0.937 支配，且没有独立
  append/eviction closure；明确不接纳为创新 Design。
- [Parent-anchored delta-scan preflight](parent_anchored_delta_scan_preflight.md)：证明 joint K/V 加极小 RMS
  metadata 在信息上可恢复 Parent checkpoint；但历史 Q/gate 的 mandatory floor 在任何 delta attention
  前已达 25.32%，因此当前 KV-only interface 下退休。
- [Migration-ready source-tape preflight](migration_ready_source_tape_preflight.md)：允许 cache producer
  保存 exact Parent execution cut 后，finite-defect recurrence 在 full-rank/native limit 下语义成立；但稳定
  tape 需额外 19.5 MiB/user，且五层 native Current attention 单项已达 42.24%，因此不跑 UID、明确退休。
- [论文高度机制审计](paper_height_mechanism_audit.md)：exact K/V response interaction 分解与
  common-projection native-response control 的完整反例；冻结 single-arm r8 为所有新机制的硬对照。
- [Paired native-response preflight](paired_native_response_preflight.md)：让 paired r4/r4 两臂在完整原生
  activation/aggregation 后做 control-variate；18.28% 成本下达到 .901 mean，但仍被 single-r8 的 .937
  支配，按硬门退休。
- [Defect-first coordinates preflight](defect_first_native_response_preflight.md)：固定 Parent-base rank2 与
  release-defect rank4 的逐层坐标递推；18.46% 下仅恢复 .508 mean，且与 base/residual prior art 重叠，
  因而不调 rank、不开 formal canary、明确退休。
- [Source-certified reduced execution preflight](source_certified_reduction_preflight.md)：以 exact Parent
  causal response 分别检验 absolute-source residual 与 finite release-defect residual；两者虽满足
  full-rank exact limit 且成本仅 18.98%/19.47%，五边 mean recovery 只有 .644/.662，并被 paired-native
  与两个 single-r8 controls 支配。其 DEIM sampled-residual 骨架也与既有 hyper-reduction/control-variate
  工作重叠，故冻结为数值与 novelty 双重 NO-GO，不扩用户、不调 rank/pivot/lift。
- [Producer-state / reader-version commutator preflight](producer_reader_commutator_preflight.md)：四条 exact
  path 显示 adjacent release 的 state effect 在 raw score 与若干 S4 边界近似交换；但 centered decision
  effect 不稳定，且 reverse path 依赖 Exact Current K/V 与逐 candidate Parent reader，因此只保留正 oracle
  observation，不接纳为 Design。
- [Activation-boundary replay preflight](activation_boundary_replay_preflight.md)：Parent/Current activation
  topology 大体稳定，但 crossing-only response 不充分；same-region continuous deformation 主导，且
  Current graph discovery 已越过预算。
- [Probe-free affine-response falsifier](all_history_affine_response_preflight.md)：成本 18.55% 的全历史
  `B/M` invariant 在 full-Exact representation oracle 即灾难性失败，并与 linear-attention fast-weight
  state 同构；冻结 query-dependent activation geometry 不能删除。
- [Release tangent preflight](release_tangent_propagation_preflight.md)：说明 parameter delta 低秩为何不自动
  使 state/attention differential 低成本，以及现有 KV-only interface 的 28.48% 下界。
- [Whole-history operator transport](whole_history_operator_transport_preflight.md)：记录 aggregate mapping 的
  五边反例并明确退休，避免以后把 mapping 换名复活。
- [Response-operator prior-art collision](response_operator_prior_art_collision_preflight.md)：审计 moments、
  signed response、K/V interaction、functional transfer 与 persistent user state 的直接邻近工作，冻结
  不能再使用的创新表述。
- [Release algebra / invariant preflight](release_algebra_invariant_preflight.md)：逐项关闭 gauge、structured
  update、secant/commutator、native-query quotient 与 finite-state algebra 五类 exact shortcut。
- [Natural causal suffix self-probe](causal_suffix_self_probe_preflight.md)：证明错误 Parent lineage 也满足
  Current append consistency；suffix 只有 query coverage，没有 Current target-state information。
- [Release-circuit native recomputation](release_circuit_native_recompute_preflight.md)：证明 dense merge、gate
  与 residual 使 head salience 不能形成低成本跨层 causal module。
- [Causal state port prior-art audit](causal_state_port_prior_art_preflight.md)：说明 causal separator 不自动
  产生跨版本兼容；post-hoc ports 当前 NO-GO，未来 co-design 需先有 release homomorphism 与 delete law。

对应执行代码放在 `scripts/insight_two/`，原始结果放在
`results/yambda500m_medium_seed17/insight2_functional_boundary_v1/`。诊断性
Exact-state intervention 只用于定位边界，不能进入可执行成本 frontier。
