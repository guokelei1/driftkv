# CPU真实frontier批次计数

32用户；最大批次8；2,153种批联合形状。检查：{'actual_distinct_user_session_batch_shapes': 'exact_match'}。

零尾合批额外0.001145 TF；转换tile补齐额外0.011373 TF。均为显式逻辑工作，不是机器指令计数。

调度器只重排独立用户，不跨发布；每轮每用户一个最早待处理请求。此为饱和bulk调度形状，并非到达排队仿真。
