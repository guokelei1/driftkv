# CPU真实frontier批次计数

30,000用户；最大批次8；184,415种批联合形状。检查：{'all_individual_joint_shapes': 'exact_match_30k_retained_ledger'}。

零尾合批额外2.029738 TF；转换tile补齐额外12.399006 TF。均为显式逻辑工作，不是机器指令计数。

调度器只重排独立用户，不跨发布；每轮每用户一个最早待处理请求。此为饱和bulk调度形状，并非到达排队仿真。
