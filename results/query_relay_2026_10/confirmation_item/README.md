# Current-item 门控的新用户确认面板

只准备新成员、历史和请求，尚未评分。模型及方法设置冻结后构建。

从原始 Large 用户中排除整个 Medium30k、全部已记录的校准保留/fit/validation/evaluation/pilot角色，以及上一轮已经揭晓的1024名确认用户。剩余40,088人中按 D259 前驻留历史长度和过去72小时是否活跃进行比例分层，用独立 namespace `query_relay_item_strict_new_user_confirmation_2026_10_15` 的UID哈希确定1024人。选人不使用未来请求、标签或预测；后来没有推荐请求的22人仍保留。

固定使用已有 Medium V2→V3→V4→V5模型和物品映射，因此这是 Large 新用户迁移确认，不能称同一 Medium 人口的独立确认。历史的unknown物品保留在固定256个OOV桶，推荐目标限定known物品。16行为/16候选任务共48,381轮、774,096候选；原feedback请求11,394条另行保留，不能混用任务含义。

`panel/verification.json`记录所有角色零重叠、旧1024零重叠、时间和known/OOV核对、输入hash，以及原确认panel七个文件未改变。`prepare_panel.py`以旧builder为基础，inventory仅跳过自己的panel并显式登记旧1024；`verify_panel.py`独立核对成员、序列/请求时间、模型映射和完整原panel哈希。两者都是CPU准备，不加载模型或生成预测。
