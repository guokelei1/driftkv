# Repository agent notes

2026-10-30 02:01 用户明确要求“开始吧，用tmux”，已启动 RecFlow-Max Figure9d 正式全量评测：tmux `recflow-figure9d-formal`，协调PID1734332，输出 `experiments/figure9recflow/runs/formal_max`，真实当前边/子进程看live.json与各边progress_rankR.json。固定30000用户×7accepted边×29配置（Reuse/Rebuild、11Layer、5Tail、5Deviation、6Alignment），history1024/完整候选/持续独立缓存，主NDCG@10辅Recall@10；每边独立fit256最大前缀/8epoch，四A40 0–3、eager FP32 TF32off、chunk256、每卡2CPU预取。准备47CPU+实际GPU数值对照及7×64用户203点均通过；Graph因实测更慢不启用，正反结果全保留。中心预估27h，预算34–38h；无需等结束再答复用户。运行源码/配置/人口不改，不重复派发、不恢复旧训练/队列，无hash保护、论文不改，不训练推荐主模型。入口experiments/figure9recflow/STATUS.md；launch.json/authorization.json记录新授权，旧prepared_not_started历史计划不表示当前没启动。9c Large未混入；本次只9d，完成后无额外后继。

2026-10-29 22:34 本轮按用户授权时间数据耗尽终止并完成交付，goal=complete（不是10版科学目标成功）：真实accepted8模型7边，D29/30/31/32/33/34/36评测；主loss4.72306/3.11836/1.21867/1.89647/.68805/.41106/1.69807%，5/7主CI正、严格1–3仅3/7。两小效应有自适应full30k支持，不能称均匀稳定/独立确认。最后D36模型训练完整，D37pilot−.52948%与8192rep+.43064%均NDCGCI跨0，Recall反向pilot排0；V8未接纳、模型分数保留。全28探索及15非正/跨0记录齐、旧原9边/extraepoch4反例可追溯。raw/processed末日D37无D38，终止指未来日期窗口耗尽而非所有用户/配置穷尽。terminal_delivery_review/data_exhaustion_review/final_handoff_audit通过；actual CSV8/7、82引用核对；narrow completion_audit exit3/issues=[]仅结构正点，不证明科学目标。入口experiments/recflow_continuous_chain/final_report.txt与actual_chain_models.csv/actual_chain_edges.csv；无活动GPU/CPU实验或后继，不恢复旧队列，无hash保护/论文不改/本轮未清理数据模型。

2026-10-29 22:24 最后D36单轮全159106/2487步已完成(1797.49秒)；D37pilot4096完整22865请求，NDCG R.063132291/U.063466564、loss−.529480% CI[-.000950746,.000318382]跨0，Recall−1.50533% CI排0负；独立CSV/ledger/bootstrap通过，反例保留，未接纳V8。唯一运行为最后诊断 runs/m7_d36_lw48_replication_8192，actualPID1693858及四worker已核验；固定8192 offset4096零交/54141完整请求、同V7D35source/D36target，预计22:27–22:30，无自动后继/扩样/训练。accepted仍8模型7边；3/7严格1–3、5/7主CI正；末日D37最终审查尚待完成，不能提前称dataexhausted或科学目标达成。STATUS/current_edge_stability_review与actual_chain CSV为入口。无hash保护/论文不改/旧队列不恢复。

2026-10-29 21:53 最后D36训练仍actualPID1688137存活、step800/2487、约0.691s/step，预计22:16–22:20训练+pilot；actual_final_window_preflight四worker/完整selection/sampler/parent通过，数值源未改，无后继。新增最终交付准备：export另输出actual_chain_models.csv(8行)/actual_chain_edges.csv(7行)，原preferred占位CSV保留，边完整actual_training_path含未发布D34；实导出核对行数/指标/祖先无变，无新测试框架。独立current_edge_stability_review.json完整七边：7/7主点正、5/7主CI正；1/2/4/7较强，3弱pilot，5/6为自适应full小效应。严格1–3%仅3/7，均值1.96482%不能代表多数，最后若合格区间内也最多4/8；主值全<5但首边小群/时间切片可>5已保留。此分类不是新接受门或拒绝结果，也非最终目标完成；D37尚未评、未dataexhausted。STATUS有表和审查入口；无hash保护/论文不改/旧队列不恢复。

2026-10-29 21:45 两日发布V7已接纳，真实前缀8模型7边：D36复核8192完整55735请求/33693正/2819正UID/22078326候选，R.065800705144/U.064683360491、loss1.69807398%，CI[.000599935153,.001645318557]正；四NDCG slice1.74842/1.64585/1.86161/1.56662%且CI全正。pilot2.93321%同向，独立全CSV/UIDbootstrap最大差1.39e-17；Recall整体1.46875%CI正，odd/后12h CI跨0保留。decision/26项日志/manifest/export均齐，V7真实V6→未发布D34→D35，D36为第7边评测；主值不合并、不扩全量。当前7主值严格1–3%仅3/7，不能由均值推多数，旧弱/小效应边保留。唯一新训练m7_d36_listwise48于21:43:03启动、actualPID1688137已核验、step150/2487：acceptedV7/D35同时作训练父与publishedsource，D36全159106/一轮/λ4.8不变/AdamW/四GPU，完成仅D37固定4096全22865请求/9055149候选初测，预计22:16–22:20。conditional_last_window_preflight实际raw/processed目录均末日D37、无D38，空间116.47GiB，仍须完成实际最后窗口评测再审终止，当前非dataexhausted。新final_window_plan.json；旧observer59490已exit0勿poll，无自动复核/下一版。无hash保护/论文不改/旧队列不恢复，goalactive未完成。

2026-10-29 21:32 两日发布D36pilot清楚正向但尚未接纳：m6_d35_interval2_lw48 D35单轮154820/2420步、1726.88秒完成；真实V6/D33→未发布D34λ4.8→targetD35，缓存仍来自V6。4096全24069请求/13905正/1321正UID/9534084候选，NDCG R.065395691126/U.063477497704、loss2.93321072%，CI[.001081676201,.002731801095]正；四slice2.86910/3.00238/3.00043/2.88545%，四CI正。Recall2.18190%整体CI正，even/后12h CI跨0。independent_review全ledger/均值/bootstrap最大差1.39e-17，decision及25项日志全留；不能单因归因时间窗口，未接受V7。唯一新GPU runs/m6_d35_interval2_lw48_replication_8192，actualPID1685648核验存活，D36/offset4096固定8192/零交/55735全请求/4388活跃UID，同source/target/path/metric；预计21:38–21:41，完成再决定，无自动下一模型/全量。旧observer26310已exit0勿poll；真实前缀仍7/6、goal active，无hash保护/论文不改/旧队列不恢复。

2026-10-29 21:19 两日发布训练继续，实际PID1678688核验存活、step2000/2420，约0.685s/step；唯一tmux continuous-update-m6_d35_interval2_lw48，完成仅D36固定4096/24069请求初测，预计21:28–21:32，无自动复核/下一模型。CPU兼容准备已完成：aggregate_full_population支持真实祖先→未发布中间态→目标、逐中间态Max1024/data检查；19新测试/36相关检查通过，未聚合当前pair。validate_chain/export区分实际accepted前缀与preferred10/9占位，外部data_exhaustion_review只核对绑定/引用不自动证明科学终止；11项测试+独立复核通过，正式report/audit已刷新为ongoing、actual7/6、preferred=false、terminal缺失、goal=false，六条原主指标及24探索未改。STATUS已纠正旧“必缺3版”说法；数据仍未用尽。当前数值训练/评分源不变，无hash保护/论文不改/旧队列不恢复。训练结束后先核验真实target父与published cache_source两者，再独立完整pilot指标判断；末日D37终止还需实际最后窗口评测审查。

2026-10-29 20:56 新用户允许数据用尽时少于10版本，已转两日发布：4.8 D34训练1798.12秒完成、D35pilot23571请求NDCG R.0642523813/U.0640040969，loss+.386420% CI[-.000419668,.000898336]跨0，oddUID微负、四sliceCI跨0；native对2.4 +1.44899%CI正，gap增量CI跨0。独立全分数/均值/bootstrap零差，decision/24项日志与d34_weight_exploration_summary.json齐；4.8无8192复核，不能记作复核失败，作为未发布I34保留。唯一新训练m6_d35_interval2_lw48已20:55:21启动，actualPID1678688/tmux continuous-update-m6_d35_interval2_lw48：真实训练父I34 step002506，published cache_source仍acceptedV6/D33 step002450；D35全154820/2420步/一轮/同λ4.8/AdamW/seed17/四GPU，完成仅D36固定4096全24069请求初测。prepare_update/run_train_pair新增显式cache_source真实祖先支持，12项CPU测试与root实际dryrun通过；数值训练评分代码未改，无hash保护。source不能替换为直接parent或靠标签冒充；publication_interval_plan.json与STATUS入口，预计21:28–21:32。此路线若后两边成功为9模型8边，D37末日；不为凑10开warm前缀。原7模型6边仍是已接纳前缀，goalactive未完成。旧observer67350与CPUcompare53224已收尾，勿再poll/重启；无其他GPU任务/后继。

2026-10-29 20:29 用户最新目标修订：10发布版本仍优先，但多次不成功可增大训练/发布窗口；现有数据用尽允许以少于10版本结束，必须如实报告实际链长、失败与限制，不能称原十版本已实现。真实谱系、严格递增未来评测、同请求/候选/历史和正反保留不变；无哈希保护。当前m6_d34_listwise48继续（实际1800/2506、PID1668470），完成仅D35固定pilot；仍是本D34最后一档加权。若弱，优先把D34模型作为未发布中间态，真实续训D35后以acceptedV6/D33缓存对D35目标在D36测；再D36训/D37测最多9模型8边，不为凑数启动早期warm前缀。D37=2024-02-18最后可评日，不能训D37再假设有D38。run_train_pair默认即时父缓存，未来两日发布必须显式保持publishedsource=V6，不能只换标签。新objective_revision_2026_10_29.json与STATUS记录；此修订未启动额外GPU/后继，goalactive，数据尚未用尽。

2026-10-29 20:08 V7λ2.4复核已完成但原双正点规则未过，不接纳不扩全量：D35固定8192全53978请求，NDCG R.0649877971/U.0643729095，loss+.946159%、CI[.000147804,.001076951]正，Recall1.11451%CI正；evenUID/前12h NDCGCI跨0。pilot−.06607%CI跨0保留，不称已证实反向；合并描述性.65044%CI正不替代primary。独立ledger/全slice/UID bootstrap复算零差；同父1.2→2.4 native两群+1.537/+1.843%CI正，gap增量仅rep下界微正、pilot跨0，不能保证再增权有效。decision/manifest未接纳分支/23项日志/export齐，仍7模型6边。唯一新训练m6_d34_listwise48于20:07:57启动，actualPID1668470/tmux continuous-update-m6_d34_listwise48：同acceptedV6/fullD34 160322/2506步/一轮/AdamW/seed17/LR，仅λ2.4→4.8及输出不同；实际命令preflight通过。这是本D34同父窗口最后一档增权，弱则转发布间隔/目标诊断，不再翻倍；完成仅同4096 D35pilot后判断，无自动复核/V8。预计20:42–20:46。4GPU授权，磁盘实测144GiB free（本轮未清理），无哈希保护/论文不改/旧队列不恢复；STATUS与experiment_log为入口，goal active未完成。

2026-10-29 19:17 V7λ1.2复核不支持稳定正边，已不接纳且不扩全量：D35固定8192全53978请求/31951正/2840正UID/21387419候选，NDCG R.0638118440/U.0636640060，loss+.231678%、CI[−.000308572,.000610248]跨0；前12h−.76685%、后12h+1.02247%（仅后者NDCG CI正）。与负pilot−.40613%不一致；描述合并12288/77549请求loss+.0449479%、CI[−.000346173,.000404205]，Recall−.07159%CI跨0。独立ledger/所有slice均值与bootstrap复算零差，两附件与decision.json齐，manifest未接纳分支/21项日志/export保留全部反向，仍7模型6边、目标未完成。唯一新训练m6_d34_listwise24已19:15:51启动，actualPID1657614/tmux continuous-update-m6_d34_listwise24：同acceptedV6 step002450、同D34全160322/2506步/单轮/完整AdamW/seed17/LR，仅λ1.2→2.4及输出路径不同；sibling_comparison_plan.json已实际命令逐项确认。完成仅相同4096 D35pilot，再CPU compare_training_recipe对已保留1.2 sibling；不自动继续增权/复核/V8。动机支持native质量多于cachegap，不能假设有效，预计19:49–19:55。当前STATUS/experiment_log为入口，旧候选checkpoint/全部分数保留；无哈希保护、论文不改、旧队列不恢复。

2026-10-29 19:07 V7λ1.2单轮已完成但初测反向，未接纳：真实父acceptedV6 step002450，D34全160322/2506步，含保存1784.89秒，endpoint training_runs/m6_d34_listwise12/training/step_002506.pt。D35 pilot4096全23571请求/13531正/1337正UID/9342557候选，NDCG Rebuild .0623760095/Reuse .0626293379，loss−.406131%、CI[−.000889797,.000382029]跨0，Recall−.524786%CI跨0；四NDCG slice三负，全部CI跨0。实际ledger/全候选审计及独立均值/UID bootstrap复算零差，independent_review.json与decision.json保存全部反向。唯一新GPU为runs/m6_d34_lw12_replication_8192，同actualpair/D35/固定offset4096/8192零交/53978完整请求，actualPID1654720（后续查active_process真实存活），预计19:13–19:16；是诊断复核，不是正向确认，无自动扩全量/V8/新λ训练。链仍7已接纳模型6边，goal active未完成；原6主结果仅2/6严格1–3%，均值约2%不能替代多数。剩D35/D36训练及D36/D37全量候选数据齐，空间足够原计划三版；D37是终点。STATUS/experiment_log为入口，无哈希保护/论文不改/不恢复旧队列。

2026-10-29 18:32 V6λ1.2据终局原30k接纳为小效应第6边：remaining17712完整118776请求loss.641408% CI[.000072176,.000711882]正；三组CPU全量198384请求/118349正/10319正UID/78475586候选，NDCG R.0605952112/U.0601909385，loss.667169% CI[.000161247,.000643710]正，Recall1.00628%CI正。独立重建全ledger/全局split/均值/CI零差，full/independent_review.json及decision.json；原8192primary.41106%CI跨0不替换，pilot1.42幅度更大、rep even近零负点、full even/后12h NDCGCI跨0及自适应开发/效应<1%全留。manifest/export实际7模型6边，剩3模型3边未证，goal仍active。唯一新训练m6_d34_listwise12，18:30:06启动、实际PID见active_process/tmux continuous-update-m6_d34_listwise12；真实父acceptedV6 step002450，新D34全160322/2506步/单轮/不变1.2/AdamW/seed17/四GPU，完成仅D35固定4096全23571请求初测，无后继，预计19:03–19:10。未加2.4，不恢复旧队列，无hash保护/论文不改。STATUS/experiment_log及chain_edges.csv入口。

2026-10-29 18:12 V6λ1.2主8192复核完成但未接纳：D34 55423请求/33304正/2805正UID，NDCG R.06014658/U.05989934，loss.41106%、CI[-.000205511,.000706424]跨0；evenUID−.03688%近零，其余三点正/全部CI跨0，Recall.82442%CI跨0。sameparent.6 nativeNDCG+2.20803%CI正，但NDCGgap增量两群CI均跨0；全保留。独立CSV重算整体/4slice零差；两群描述合并.70662% CI[.000053208,.000800844]正，不替代8192主复核。唯一当前GPU m5_d33_lw12_remaining_17712，同pair/D34/固定offset12288/118776请求补齐原30k总198384；这是最后一次扩样，不换窗口/seed/bootstrap，属于看过结果后开发补全。预计18:24–18:27；完成后仅现有CPU aggregate_full_population.py汇总到m5_d33_lw12_full_30000，再判断，不自动V7。population_completion_plan.json保存精确CPU命令。链仍6模型5边、goal active；无hash保护/论文不改/旧队列不恢复。

2026-10-29 17:59 V6λ1.2单轮及pilot完成：sameV5/fullD33 156752/2450步，连保存1805.94秒；D34固定4096全24185请求/14032正/1347正UID/9578244候选，NDCG R.05914367/U.05830383，loss1.42000% CI[.000213908,.001481045]正，Recall1.72384%CI正；四NDCG点正，evenUID/前12hCI跨0。实际同父.6对照nativeNDCG+3.7674% CI正，gap增量CI下界−1.477e−6略跨0，全部保留，不先称改善已证。唯一新GPU为m5_d33_lw12_replication_8192，同pair/D34/8192offset4096/55423完整请求，实际PID见active_process，预计18:05–18:08，无后继，不接纳V6直到复核。链仍6模型5边。新增只读fallback_warm_prefix_evidence证实早期warm4096→resume64431→M0真谱系及中间49152来源，但没有早期warm→M0/D28cache证据，未安排GPU/工具更改；仅条件后路。无hash保护/论文不改/goal active。

2026-10-29 17:32 当前唯一V6 λ1.2训练实际PID1628065仍活跃（最新步数见STATUS），sameV5/fullD33/2450steps/oneepoch/四GPU；近期.67s/step，训练+pilot预计17:55–18:05。独立CPU实际plan/selection逐字段一致、差仅weight/output；training_curve_comparison.json共同350步排序loss降/BCE升，仅优化诊断非质量。完成pilot后用compare_training_recipe.py对同父V6.6，不与originalM6比。修正纯报告validator/export过度声明：structural_and_point_requirements_proven与稳定性分开，goal_requirements_proven不再被该窄工具置真；CLI2缺项、CLI3结构/正点齐但待单独逐边审核，不加新显著性门槛/哈希保护，不改活动数值源。当前实际审计6模型5边不变、goal active未完成。

2026-10-29 17:24 V6.6复核反向且不明确，不接纳、不扩全量：m5_d33_lw06_replication_8192完整D34 55423请求/33304正/2805正UID，NDCG R.05884721/U.05902531，loss−.30263% CI[-.000596246,.000255873]；两个UID半组及后12h负，Recall后12h负CI排0。pilot+.26146%与复核不一致，全部留档/manifest未接纳。唯一新训练m5_d33_listwise12于17:22:53启动，实际PID1628065/tmux continuous-update-m5_d33_listwise12：同acceptedV5父/sameD33全156752/2450步/单轮/AdamW/seed17/LR，仅λ.6→1.2；sibling_comparison_plan实际命令核对通过。动机为此前权重调整改善native+gap，属自适应开发，不保证成功；完成只同4096 D34pilot，再CPU同父native/gap比较，无自动后继/扫权重。预计17:55–18:00，链仍6模型5边、goal active。永久无哈希保护已独立复核当前调用链；同日重训+LR重启四epoch2无新增正边，不继续机械加epoch。论文不改，旧队列不恢复。

2026-10-29 17:21 当前V6.6已完成D33单轮156752请求/2450步/1672.77秒，但未接纳：D34 pilot4096全24185请求，NDCG Rebuild .05699639/Reuse .05684737，loss.26146%、差CI[-.000443982,.000730469]跨0；四NDCG切片点正但全部CI跨0，Recall偶UID/后12h点负。唯一当前GPU为m5_d33_lw06_replication_8192，17:14:25启动/实际PID1625197，同父子同D34/固定零交8192/全55423请求，预计17:21–17:23完成，无后继。不自动扩全量或训V7，无λ1.2训练；链仍6模型5边，goal active。用户再次明确永久不要任何哈希保护；现有同日重训+LR重启四epoch2无新增正边，不恢复此路线，也不泛化为所有epoch无效。STATUS/experiment_log入口，论文不改。

2026-10-29 16:43 连续链V5已据完整总体接纳为小效应正边，goal仍active：原4096/8192 NDCGloss .65937/.68805%各CI跨0仍保留；唯一remaining17712同D33全116258请求loss.75400% CI[.000161414,.000727983]正。三源CPU汇总及独立核对零差：原30000/194452请求/116269正/10219正UID/76881994候选，Rebuild .0579009182/Reuse .0574816962，loss.724033%、gapCI[.000216112,.000632534]正；Recall .638066%CI正。全量四slice点正，后12h NDCGCI仍跨0、pilot后半天反向及效应<1%全留；扩样为看过结果后的开发证据，非独立确认/多seed。full证据在runs/m4_d32_lw06_full_30000；manifest原8192primary不换，另记full_population_sensitivity_directory及stability_review。实际validator6模型5边，余4模型4边未证；第三边弱点仍保留。当前唯一训练training_runs/m5_d33_listwise06，tmux continuous-update-m5_d33_listwise06，16:41:52启动，从真实V5 step002494继承完整AdamW，新D33全156752/2450步/单轮/λ.6不变/四GPU，完成仅D34固定4096全24185请求初测，无后继，预计17:13–17:17。originalM6不同父不可单因素比较；无λ1.2训练、无哈希保护、论文不改、旧队列不恢复。STATUS/experiment_log为入口。

2026-10-29 16:27 V5.6唯一8192复核已完成，仍未接纳：D33 53647请求/32237正/2805正UID，NDCG Rebuild .05845220/Reuse .05805002，loss+.68805%，差CI[-.0000003604,.000807393]略跨0；四固定slice点均正但区间均跨0，Recall+.51595%CI跨0。初测+.65937%与复核接近；两群合并描述性+.67936% CI[.00006113,.00072789]，不能替代8192primary或称独立确认。当前唯一新GPU为m4_d32_lw06_remaining_17712，固定offset12288、同模型同D33、全部116258请求，补齐原30000用户总194452请求；这是本候选最后一次扩样，不再换人口/窗口/seed/bootstrap。选全量因两群方向接近且更省时间/磁盘，追加发生在看过前两群后，需公开开发选择局限。无λ1.2训练/无后继。expected16:38–16:42，实际PID见active_process；existing_pairs只写纯CPU三源全量ledger/指标汇总，不改活动脚本。链仍5已接纳模型4边、goal active未完成，无哈希保护，论文不改。STATUS/experiment_log及各decision.json为入口。

2026-10-29 16:16 连续链V5候选单轮D32已完成，尚未接纳：m4_d32_listwise06从已接纳V4.6完整继承AdamW，159607请求/2494步/1717.79秒。D33固定4096全24547请求、9717197候选核对过，NDCG Rebuild .05669693/Reuse .05632309、loss+.65937%，差CI[-.00019269,.00095397]跨0；两个UID半组点正但CI跨0，前12h+1.7741%CI正、后12h−.21959%CI跨0，Recall整体+1.08337%CI跨0及时间切片反向全保留。当前唯一GPU任务m4_d32_lw06_replication_8192，实际PID1609466，16:14:47启动，同模型/同D33/固定零交8192/全53647请求，预计16:21–16:24；完整复核前不接纳、不训练后继，不与不同父originalM5作单因素比较。现链仍5已接纳模型4边，goal active未完成，第三边弱点保留。只读fallback_alignment_resume_evidence_2026_10_29.json证明原alignment E1→E2真实连续resume，可作条件后路但未评分/未安排，不能拼接原M1；当前路线继续。无哈希保护、论文不改、旧队列不恢复，STATUS/experiment_log为入口。

2026-10-29 15:42 连续链第四边0.6已接纳，goal仍active未完成：同父newV3/fullD31单轮仅listwise .3→.6，完整164443/2570步1740.48秒完成。D32 pilot4096 loss1.83038% CI正；固定零交8192复核55873请求/33382正/2818正UID，NDCG Rebuild .05597780/Reuse .05491619，loss1.89647%、差CI[.000620264,.001531737]，四固定NDCGslice CI均正。Recall1.07866%CI略跨0保留；pilot oddUID NDCGCI跨0保留。同父.3 sibling配对比较两群均native质量与gap增量区间正，rep native+6.40276%；并非与旧originalM4同父。合并12288/81342描述性loss1.87505% CI正，非新确认/不替换8192primary。弱.3候选全部留档，manifest/validator与CSV现5模型4正边、剩5模型5边未证；第三边原弱点仍保留，不自动依赖点正validator宣称全部稳定。当前唯一训练m4_d32_listwise06，tmux continuous-update-m4_d32_listwise06，实际PID1601118，15:41:45启动；父为accepted newV4 .6 step002570，D32全159607/2494步/一轮/不变.6/完整AdamW/四GPU，完成仅4096 D33初测，无自动后继，预计16:13–16:17。原originalM5父不同不能作单因素对照。新fallback_feasibility_2026_10_29.json核实官方早期all_stage为抽样，D24–37才完整ranking候选，不能直接拿D18/20/22延长同口径评测；未重建foundation/未请求额外数据。原foundation实测warm5.78h、累计alignment9.88h（包含warm，不能相加）。无哈希保护、论文不改，旧队列不恢复。STATUS/experiment_log及chain_models.csv/chain_edges.csv为入口。

2026-10-29 14:59 连续链第四边0.3候选未接纳，goal仍active：m3_d31_listwise03从已接纳newV3训练D31全164443/2570步，单轮1742秒完成。D32 pilot4096/25469请求NDCG loss+.09290% CI跨0，8192零交复核55873请求/33382正/2818正UID loss+.62231%、CI[-.00007154,.00072368]，四切片全部CI跨0；合并12288/81342请求loss+.45206% CI跨0、Recall反向，全保留。当前唯一训练m3_d31_listwise06，tmux continuous-update-m3_d31_listwise06，实际PID1592366，14:57启动、已100/2570步；从同一newV3 step002573重新训练同D31一轮，仅listwise .3→.6（除输出路径，其他命令一致），不是从失败V4加epoch。完整AdamW/同LR/seed17/四GPU；训练完只同4096 D32pilot，无自动复核或后继。sibling_comparison_plan.json已保存，完成后用compare_training_recipe.py与已完成.3同父分支比较native质量和gap；旧originalM4父不同不可作单因素对照。不保证.6有效，不自动继续增权；预估15:29–15:33完成训练+初测。manifest/validator仍4已接纳模型3正边，其余6边未证；点正validator不是最终稳定性证明。新export_chain.py导出10模型/9边CSV与chain_report.json，待定留空；首边全量Recall CI纯CPU补齐，第二/三/第四边合并描述性结果均保留，不替代8192primary或称新确认。无哈希保护、论文不改、旧队列不恢复；STATUS/experiment_log为入口。

2026-10-29 14:15 连续链第三条边带弱点接纳，goal仍active未完成：新V3为newV2真实子版，D30全164631/2573步单轮λ.3。D31 pilot4096/25712请求NDCGloss仅+.05355%且CI跨0、切片混合；一次预定不重叠8192诊断复核56546请求/34070正/2877正UID，Rebuild .05361607/Reuse .05296267，loss1.21867%、CI[.00023826,.00107654]；四slice点正，但前12hCI跨0，Recall+.99454%CI略跨0，全部保留。合并12288描述性敏感性loss.85518%、CI[.00011954,.00082397]，非新确认/不替换复核主指标，不能只报更高点。实际manifest/validator现在4模型3相邻边，余6模型6边未证。λ.6仅讨论、从未训练。当前唯一训练m3_d31_listwise03，tmux continuous-update-m3_d31_listwise03，父newV3 step002573，D31全164443/2570步/1epoch/λ.3/完整AdamW/四GPU；完成只4096UID D32初测，无后继，预计14:47–14:52。本轮CPU compare_training_recipe已兼容continuous summary.plan，旧两项实际结果内存重算完全一致、不同父仍拒绝；不用于新V3对旧M3单因素因果比较。无哈希保护、论文不改。STATUS/experiment_log为权威入口。

2026-10-29 13:29 连续链第二条边已接纳：新M2 λ.3的8192不重叠复核D30完成55813请求/33414正/2827正UID，NDCG Rebuild .05291519/Reuse .05126510，loss3.1184%，UID95差[.00107253,.00219231]；四预定NDCG切片均CI正，Recall loss2.9891%（前12hCI跨0保留）。同请求原配方native提升约15.87%，全部候选/指标audit及CPU recipe比较过；pilot4.1680%单列。同模型/日期用户复核非独立训练复现。chain_manifest已登记真实V0/M0→V1/M1→V2/newM2，validator实际3模型2边，其余7模型/7边未证，goal仍active。当前唯一训练m2_d30_listwise03，tmux continuous-update-m2_d30_listwise03，实际PID1574313，父为新M2 step002526（不能接旧M3），D30全164631条/2573步/1epoch/λ.3/完整AdamW，四GPU；完成只4096UID D31初测，无后继。预计本次训练+初测14:00–14:05。新prepare_update.py默认只准备、dry-run纯CPU、launch单版，实际读取请求/日期/模型结构，不加哈希保护。STATUS/experiment_log为入口，论文不改。

2026-10-29 13:23 连续十链新M2单轮已完成2526步/161619请求：λ.1→.3唯一优化变化，完整AdamW/真实父M1/D29保留；4096初测D30完整25367请求，NDCG Rebuild .05199769/Reuse .04983043，loss4.1680%，UID95差[.00127346,.00315096]，四UID/time切片均正；Recall loss4.7490%。同请求原配方native提高16.89%、gap从.4528%升至4.168%，recipe_comparison.json已CPU独立完成，候选/指标audit过。当前唯一GPU任务为m1_m2_lw03_replication_8192，实际scoring1570927，55813完整D30请求，不重叠8192UID，四卡、无后继训练。复核前不接纳V2；真实已接纳仅M0→M1，整条链未完成。结果后若复现才能登记V2并从它训练D30下一单版，禁止接回旧M3。无哈希保护，论文不改；STATUS/experiment_log已更新。

2026-10-29 13:06 连续十链goal继续active，未完成：唯一新M2训练PID1562863已1550/2526步、0.658s/step，四GPU约99%，真实进程与输出路径已核对；同一tmux训练完只D30/4096初测+audit，无后继模型。新增CPU证据：training_curve_comparison.json共同100–1200步所有点listwise下降/BCE上升，1200步分别−1.21%/+1.19%，仅训练优化证据非NDCG。original_nine_fixed_cohort_diagnostic.json/csv全9边×固定4096/8192×全天/12h共54行已完成：原M1→M2两群+.4528/+.5836%，CI均跨0且Recall负，仍弱；5/6混合、7/8两群负。旧六配对不能替代新目标。新chain_manifest.json实际仅V0/M0,V1/M1；validate_chain.py已实际运行completion_audit.json给2真实模型/1边、其余8模型8边未证，exit2预期，无哈希门。训练预计13:20–13:25连初测结束；届时先运行compare_training_recipe.py对原m2_d30同请求配对，再按结果决定是否唯一8192不重叠复核，不接回旧M3。入口continuous_chain/STATUS.json。

2026-10-29 连续十版本新目标首试反向：warm_m0_pilot_4096完整26303请求/14309正请求，Reuse高于Rebuild5.8162%，四固定切片负，原生/候选/指标audit通过；不纳入链、不扩大。现在唯一训练training_runs/m1_d29_listwise03已tmux recflow-chain-m1-d29-lw03启动，原M1为真实父，D29全161619请求1epoch/2526步，BCE+0.3listwise；只改变原M2的lossweight0.1→0.3，seed17/完整AdamW/1e-4日程/批64×微16/4GPU同原，log50。run_train_pair.py结束该训练后仅4096UID整日D30评分+独立分析，无后继模型。当前候选真实链只确定原M0→M1，V2在训、其后7版本未训，不能接回原M3；9评测日期D29–37。新STATUS.json为权威实时入口，目标active未完成；旧六配对历史成果保留，无哈希保护，论文不改。

2026-10-29 新用户目标覆盖上条六配对完成范围：现在必须一条真实V0→…→V9十发布模型链、9相邻边均稳定ReuseNDCG10低于Rebuild，评测日期严格递增且目标训练后，保留所有反向。原六组只是历史成果，不是新目标验收；不能拼接不同训练分支。新工作目录experiments/recflow_continuous_chain，STATUS.json与existing_checkpoint_inventory.json入口。首个唯一小实验warm_m0_pilot_4096已后台启动（tmuxrecflow-chain-warm-m0），Wfinal→M0/D28，4096固定UID整日请求/完整候选/1024/四GPU；无后继队列。W梯度D23但词表/人口D27，最早因果评测D28，不能假称D23已发布。候选若成立：W→M0→M1→M3→M4→5个待训连续后继，日期D28/29/31–37；只是未验证候选。不要复原旧已停队列，不改论文，继续无哈希保护。新goal active，未完成。

2026-10-29 11:44 本轮6–7配对目标已完成：原4组+新增M1→M3与M2→M4共6个不同有序配对，停止扩展，无GPU/后继。新两组先4096初测再8192不重叠用户复核，主指标只报复核：2.6608%/2.8778%，56546/55873完整请求；NDCG与Recall均正，UID95和四预定切片同向。M3→M5初测−.5508%跨0与四epoch2反例全留。六组相对损失均值2.4595%、范围.9299–4.7231%，是选择的机制案例，非整体成功率/独立训练重复；原30k未来整日、完整候选、1024 matched-prefix快照，非持久化serving。5次新评测/分析合计24.70分钟、188147请求，无新训练。最终语义/路径/指标/记录核验通过，4卡0MiB，原10模型在；额外epoch四checkpoint按授权释放51.60GiB，配置/分数/日志全部保留。入口experiments/recflow_release_design/release_interval/FINAL_RESULTS.json、FINAL_RESULTS.txt与model_groups.csv；两个STATUS均completed。无哈希保护，不改论文，不恢复任何旧队列。

2026-10-29 11:32 当前新目标已5组：M1→M3在4096初测+2.0777%、不重叠8192复核+2.6608%（56546请求/34070正请求/2877正反馈UID），UID95区间正，两个UID半组和两个12小时均正，native/candidate/metric audit全部过。新组以复核单独指标计数，不称独立训练/日期确认；registry=model_groups.json/csv。M3→M5初测−.5508%跨0保留不扩大。下一唯一实验M2→M4 pilot4096，tmux interval-m2_m4_pilot_4096，无后继队列；四A40、1024、完整候选/整日请求，无新训练/无哈希保护、不改论文。release_interval/STATUS.json与experiment_log.json为入口。达到6–7后停止搜索；goal尚未完成。

2026-10-29 新目标6–7独立配对 active：用户明确要求一步一个小实验，四GPU、无哈希保护，不重复同日+LR重启。首个小实验 release_interval/runs/m3_m5_pilot_4096 于11:13启动，tmux recflow-interval-one，实际scoring PID1509447；仅此实验，无后继队列。原M3→M4→M5实际路径，M4视作未发布中间件；D33整日，原30k中固定RNG4096UID（2163活跃/24547请求），所有候选完整，1024上限、FP32、matched-prefix快照；原64 Rebuild分数先核对，再四卡新评分/独立CPU分析。新 interval_protocol/evaluate/one/analyze 与旧链独立，不修改旧结果，不恢复旧队列。原四正组已整理release_interval/model_groups.json/csv。按用户许可已清理仅epoch2四个失败补训.pt，释放51.60GiB，配置/日志/评测/sidecar全留，原十模型保留；cleanup.json在epoch2 run。完成首个小实验后才决定扩大或下一个。目标尚未完成。

2026-10-29 10:44 当前 RecFlow full_population + release_design 的数据准备/加载、训练/续训、评测/对比与编排入口已移除哈希拦截；旧字段仅记录，不做准入。真实父模型按记录路径检查，实际时间窗口、样本、数值/指标检查仍在，29 CPU tests通过，旧四分支metadata兼容读取通过。epoch2_attempt_1 的18阶段全部exit0；原论文变化错误已归档 historical_failures/2026_10_29_paper_hash_check.json，completion/progress/inventory/STATUS正常completed，未重训、未改模型或任何指标。无新GPU任务。当前配方为同日追加一轮并重启LR：四分支均未新增正向边，M3由+1.445%变−1.728%，M5/6/7为−.903/−.865/−.276%；M3/5/7原生约−3%。不把该结果外推为所有多epoch训练无效。

2026-10-29 用户明确要求：今后不要任何哈希保护。不得因论文、源码、配置、数据、checkpoint 或历史记录的哈希变化而拒绝启动、暂停、终止或判定实验失败；不得以改名、默认开关或其他文件指纹方式恢复这类保护。本条覆盖下方所有旧哈希冻结/准入指令。训练正确性继续通过实际模型加载、样本/标签/时间窗口、真实父模型路径、数值和指标检查判断。旧实验中的哈希与失败记录只作历史记录，不作为以后运行的前置条件；算法本身的 embedding hashing、固定采样不受此条影响。当前 epoch2_attempt_1 四分支训练/完整评测/逐支核验均已完成，M3/M5/M6 反向区间排0、M7跨0，无新增正向边；不要重跑旧队列或自动追加 epoch。

2026-10-29 06:35 独立补训 `recflow_release_design/runs/epoch2_attempt_1` actual64三路与原分数逐位一致(maxdiff0)，CPU25项过；M5_e2训练PID1419712已100/2494步，0.657s/step，四卡约100%利用。每卡全部1,154,235,993模型元素和2,308,470,883优化器元素恢复逐位一致，原159607请求第二遍全覆盖已核验。协调1419284依次M5/M6/M7/M3 e2，逐支完整评测+audit+原单轮配对；尚无新quality结果。首支预计07:30–07:45，全批10:30–11:30（Asia/Shanghai，实测预测）；handoff.json与STATUS.json为入口，原模型/raw/处理数据均保留，未清理。勿改冻结新code/spec/原源码或重启队列。

2026-10-29 06:31 用户新授权独立发布边补训（暂不做连续链），已启动 `experiments/recflow_release_design/runs/epoch2_attempt_1`，tmux `recflow-epoch2-branches`，协调 PID1419284；实际阶段/子进程看 active_process.json。第一批固定 M5_e2、M6_e2、M7_e2、M3_e2（正向对照），各从原对应一轮 endpoint 及完整 AdamW 状态续训同一天一轮，总两轮；seed18等价原sampler的epoch1，日程重新warmup/cosine 1e-4→5e-6。训练直接父为原Mx，Reuse发布父为原M(x-1)，两者分别绑定且验证真实祖先，不伪造元数据、不拼连续链。四A40 0/1/2/3，CPU25项过；启动时处于actual64原分数parity门，随后全元素恢复核对、完整续训/次日全量三路/独立候选指标audit/与原单轮配对分析。新code及原源码/配置/人口/plan冻结，勿修改或重复派发。全部正反/失败保留，无按gap调参或自动加轮数。本轮初始余186.95GiB，每支checkpoint12.90GiB+eval约1.86GiB，四支预留72GiB后仍约115GiB；无删除raw/处理数据/旧模型。raw45GiB当前不直接参与训练但保留重建源，处理后stage_v2/history经symlink共享不可当废物删。原full_population十模型九边已05:37完成：明确正边1/3/4/9，反向7，其余跨0；全边相对损失均值0.9724%，正四边均值2.3046%，最大4.7231%。论文不改；所有旧完成/暂停队列不恢复。

2026-10-28 10:04 新用户明确授权 **full population 30000 / 十版本** 已后台启动，与下方旧已完成链独立。入口 `experiments/recflow_full_population/STATUS.md`；tmux `recflow-full30k`，协调 PID1119724，run `experiments/recflow_full_population/runs/attempt_1`，真实子进程看 active_process.json。四张 A40 0/1/2/3；真 Max16×320/10heads、1.154B、history<=1024，空/短历史均纳入；只用D27前信息选固定30000(20000 ranking+10000 early)，独立val2000/cache384+128。M0 warm D1–23全4123554一次，alignment D24–27全647394最多两次；M1–9续训D28–36，各自同训练UID下一整日D29–37评测，M0原生D28。185输入SHA/22列表因果审计、18CPU及实际四卡Max混合长度/保存状态逐位恢复/25步续训/同版三路score差0均过；跨独立训练逐位轨迹测试失败已保留，断点前已有差异，原atomic backward不承诺确定性，未放宽保存状态相等门。启动时尚无学习结论：先固定未训control和4096步heldout BCE门，再完整基模native NDCG胜random/pop门，才进九边；不因cache gap正负或新旧native高低截断。15h foundation硬预算、全程预计22–29.5h，模型/正反/失败全留。源码/config/人口/plan冻结，勿改数值或重复启动；论文/旧队列不动。用户要求启动后报告并等待其下一步指示，无需持续等训练结束。

2026-10-28 新RecFlow请求排序训练目标已完成，无本轮GPU进程/后继：真Max16层/320宽/10×32、1.154B、1024历史，M0+9实际日更新共10模型；训练阶段约4.27h，含试跑/原评测/追加准备和评测连续预算7.50h。原19窗全审计，开发3正、未见UID确认2正(边1/5)，原D37负跨0全保留。新temporal_cohort/attempt_1完整9窗/370105请求、2048固定训练分组居民(排除512校准UID)，边1/2/3/5/7/8名义95%正、4/6反向、9跨0；边2/4边界弱，非多重检验整体显著。追加组不是独立未见用户确认，真实训练UID覆盖2031/2048详见audit，不改人口。全部28窗候选/标签/指标独立核对，actual64三路分数新旧完全一致，原60+新10 CPU通过。final_training_handoff.json goal_satisfied=true；RESULTS.md为最终报告，model_chain_inventory.json为模型路径/父版SHA。1005528已自然退出、四GPU0MiB；数值源/人口/计划/模型/全部正反冻结，论文及旧队列不改，不恢复旧任务。仅matched-prefix快照，持续serving/FIFO及论文校准方法为后续任务，不自动启动。

2026-10-28 temporal_cohort追加评测已完成并独立核对前三边(D28–30)：固定2048训练分组居民，正反馈UID1589/1613/1592，Rebuild−Reuse=.00547306/.00061714/.00083979，名义UID95%下界=.00401917/.00002982/.00028977；相对损失13.89/1.51/2.04%，第二条很弱，不夸大或称多重检验整体显著。全部native高于两基线，三窗全候选核对通过。actual64新旧三路分数完全相同。原确认仍仅两明确边，不能将追加组叫独立未见用户确认。实际foundation UID覆盖2031/2048，前五窗所有活跃UID均已被父版监督，后几窗个别1人未见，名单不变，training_uid_coverage_audit.json留档。协调1005528继续全部九窗，当前D31；源/plan冻结，不能提前截断。用户目标质量已有候选但完整九窗及最终交付未齐，goal active；四卡、论文不改。

2026-10-28 原Max链已最终核对完成：10模型/19窗，completion_audit=artifacts_verified，约6.00h含小模型/训练/评测；开发3明确、未见用户确认2明确(首/第五边)，其余和D37负差跨0全保留，原goal质量条件尚未达三条。原951359/947795/956029均自然完成，勿重启。用户授权扩人口下新建temporal_cohort/attempt_1，协调1005528、先actual64 parity再九边；2048固定D26训练居民，排除原384+128校准UID，UID哈希选择、不按未来活跃/标签/score，D28–36共370105完整请求、全部1024。单独已有用户未来时序留出，不能称未见用户独立确认；不新训练、不改指标/原结果。新10 CPU tests通过，源与plan SHA d3b68638...冻结，预算预测2.12h/余3.72h。四卡授权，论文不改，旧队列不恢复；实时PID/进度看新run，STATUS为入口，goal active。

2026-10-28 十个Max/九开发边全部完成；确认已完成D28–32五窗，2明确(D28/D32)、3跨0。主边D28 Rebuild .03846631/Reuse .03222126，差+.00624504，95%[.00400995,.00854586]、相对损失16.24%；D32 .04295585/.04127803，差+.00167781，95%[.00082374,.00253846]、相对损失3.91%。全部确认native均高于两基线；D29/D30/D31跨0照实保留。开发3明确/6跨0亦全留。CPU verifier956029已核对九开发+五确认14窗全部候选/指标；最终CPU60项再次核对通过，日志/源码绑定已保存reports/completion_cpu_checks.log/.json。完整链951359/外层947795仍live、继续剩余确认，当前role3D33；实时窗口/PID看active_process.json。全部四卡授权，数值/编排源冻结、论文不改，不重复训练/开旧队列。最终10模型/19窗口审计及其余确认未齐，goal active；STATUS为入口。

2026-10-28 M3已完成、第三条开发边小幅正差：D30 14091请求/9638正/540UID，Rebuild .04109721、Reuse .04005029，差+.00104691，95%[.00009226,.00205583]，相对损失2.55%、下界近0；native也优于父版/两基线。现三开发边2正1不明确，独立确认未开。完整链951359现M4训练D30 PID962048，外层947795；CPU verifier956029已全量核对D28/D29/D30原始候选/指标，仍live。冻结源码不改，不开重复队列，继续全部九边与确认；目标active。

2026-10-28 M2已完成、第二条cache差异未成立：D29 13745请求/9443正/546UID，Rebuild .04112983、Reuse .04097466，差+.00015517、95%[−.00074279,+.00109714]跨0；native高于父版与两基线，全部保留，不调配方。完整链951359现M3训练D29 PID956845；外层947795继续。新增只读CPU `verify_version_chain.py` 实活PID956029，`runs/version_chain_verification/attempt_1`已核对role2D28/D29全原始候选和指标；结束后复核十模型SHA/真实父版/全部19窗口并汇总。60 CPU tests通过（54+6验收范围），无新的GPU测试。新CPU核对脚本/audit_scored_requests.py及原活动core/driver均冻结勿改，不开重复任务。确认分数未开，整个goal未完成；STATUS为入口。

2026-10-28 首条Max真实cache损失已在开发集成立，完整链实际启动：有界 `max_budgeted/attempt_1` completed/quality_passed=true，旧923772自然结束；D28完整日14188请求/9332正/530UID，Rebuild .03847252、Reuse .03426543、差+.00420709（相对损失10.94%），95%[.00192203,.00650752]。first293UID区间跨0，128first点差负且区间跨0，保留。M1一轮99931请求/1562步20.90min、eval219.90s；Max总110.13min+小模型=131.93min。CPU独立审计M0/M1全部5784928/5619016候选ID/标签相符，指标重算误差<=2.23e-16；first_actual_update_chain_audit passed。队列947795已派发完整链协调951359，`runs/version_chain/attempt_1` 当前M2训练D28 PID951478；M0 SHA a9db4eff.../M1 d67432fc...核验复用，不重训。formal_recipe_freeze.json已生成，确认分数未开；十模型/九开发边完成后才评确认。后续仍要全部九边/确认，当前不能标goal完成。不要改冻结core、run_after_pilot.py、version_chain.py或run_max_pilot.py，不重启重复任务；全部四卡授权、旧队列/论文不改。

2026-10-28 RecFlow完整链后继已审阅并排队：M0预发布和下一整日native显著高于两基线、数据量决定早于任何Max cache分数，故不用等待cache符号选配方。`runs/version_chain_queue/attempt_1` PID947795实活CPU等待，绑定原协调器923772 PID/start ticks；仅待其自然结束且completion通过后调用 `version_chain --execute`，复用核验通过M0/M1/M2并做齐D27–35九次更新及全部开发/确认窗口。当前GPU仍只有Max updateD27 PID944314；勿重复派发。新队列/chain编排与全部活动数值源码已冻结，源变化会拒绝派发。54 CPU tests+ruff通过；最新准入审计reports/chain_dispatch_readiness_2026_10_28.json。十小时保守总预算，确认尚未打开、freeze文件尚未生成；真实Max复用质量仍未验。入口experiments/recflow_request_ranking/STATUS.md，目标active，论文/旧队列不改。

2026-10-27 最新RecFlow目标：用户允许十模型中数条有Reuse/Rebuild差距，不要求每条；约10小时预算。与旧队列独立、四张A40可用、论文不改。当前 `experiments/recflow_request_ranking/runs/max_budgeted/attempt_1`：warmup262144请求实测1503s=25.05min；alignment131072×2、4096步3169s=52.82min，均完成exit0。epoch1/2验证NDCG .0309085/.0461707（4096请求/2798正/583UID），E2−E1+.0152622，95%[.0104312,.0201098]，选E2 SHA a9db4eff...；BCE .60798→.70456反向保留，不称收敛。M0下一整日D27 NDCG .0344997，流行度.0115388、random.0151013，差区间均正（14595请求/10027正/557UID）。协调器923772现真实update D27，PID944314，随后完整D28 Rebuild/Reuse；真实进程看active_process.json，不重复启动。Max cache损失尚未验证。`native_learning_analysis.json` passed；`foundation_data_decision.json` 在任何Max cache结果前暂定当前数据量，为九次更新留预算，不宣称样本最优。新 `code/version_chain.py` 仅CPU prepared，`runs/version_chain/attempt_1/preparation.json`；当前有界Max结束再审阅启动，按原生质量/预算准入，固定九次更新/全部正反，前两条cache符号不选择配方或截断全链。确认在配方冻结且十模型/九开发边完成后打开，当前未开。全套50 CPU tests通过（35+15），37份活动Max冻结源未变；九天843144训练请求及全开发/确认评测含25%余量预计5.59h，whole-chain保守10h门。不要改活动数值源码/config，不恢复旧队列；目标active。入口STATUS.md及reports/readiness_2026_10_27.json。

2026-10-27 新 RecFlow 请求排序准备/训练仍在进行，**与旧第三章队列独立**：用户已授权 g202 四张 A40，并将数据量选择交给本轮。入口 `experiments/recflow_request_ranking/STATUS.md`。原始14天 all_stage/37天曝光已审核转换；使用 `data/processed/recflow_request_ranking_v1/stage_v2`，旧失败保留。正式目标为 Max16层/320宽/10×32、约11.54亿参数、历史1024。小模型两条1024历史边的NDCG复用损失未成立，全部正反保留，不以128历史结果替换。实际Max受控训练在 `runs/max_budgeted/attempt_1`，协调器923772、当前alignment930421；warmup262144请求已完成1503秒，对齐131072请求×最多两轮，按预发布原生validation选checkpoint，再检查至多两条真实日更新。总pilot三小时（含此前约21.8分钟），不自动派发九次正式后继。运行中的训练/模型/读取/数据代码与configs/max_budgeted.json均冻结，勿重复启动或修改数值源码。35项CPU检查及真实四卡resume、完整Max参数/AdamW恢复与FP32读取通过；质量门和正式1+9链未完成，目标仍active。论文不改，不恢复任何旧暂停/完成队列。

2026-10-27 用户确认评测已结束并授权目录重构。公共实现已迁至 src/hstu_kvcache（methods/baselines、methods/kv_map、methods/cachetide、evaluation、training、common）；scripts 只保留 evokv.py 薄入口。第三章探索位于 experiments/ch3_motivation/code，RecFlow 编排在 experiments/foundation_recflow/code，历史探索在 experiments/archive。入口见 README.md、docs/code_layout.md，逐文件映射见 docs/code_migration.json。旧结果/模型/人口/冻结计划及其源码哈希不改；新源码不得续写冻结旧 run，不恢复旧队列。原提交 91fff42 可用于追溯。不要再把公共实现放进日期命名的 scripts 包；探索留在 experiments，稳定实现再提取到 src。


2026-10-24 A+B本轮已完成、无GPU/后继：candidate_strength_direction/attempt_1 固定8929居民/edges2、3；A共同写/read-only九λ新算280块717489分数、4032/4073反馈UID。主候选45/38覆盖913/798UID，near0 19/13、strong1.5–2 20/25，bootstrap stable29/45、20/38，split一致14/36、12/29；同context支持仅47/31UID，标签比例/共同分数偏移明显，不能称候选身份作用或gate必要已证明。B全部100head、两口径、r1–32完整，每head7577143/7718811行；Δr6中位98.38–98.81%，native亦98.51–98.78%，uncentered Δr6超过两native仅K/V8/8及41/34head，不支持Δ普遍更紧凑。原绝对cache门失败经真FP64/逐head谱敏感性证实为数值差，新门和directSVD均过，旧失败保留；一次外部显存高峰OOM留档后原配置重跑成功。completion_audit=passed、GPU1 only、论文SHA未变；主报告experiments/docs/ch3_candidate_strength_direction_2026_10_24.md。原7000→8929扩展发生在任何新loss/谱前，人口和plan SHA未改。不恢复旧队列。

2026-10-24 用户新授权执行3.3 A+B，两个都完成，仅candidate_strength_direction/attempt_1：A固定linear256_rows/read-only λ0..2九点，共同Reuse/current写轨迹，不报按item oracle选λ后整体AUC；B全部layer/head32d ΔK/ΔV与native/rebuilt参照谱，FP64 Gram/directSVD数值门，uncentered+centered，不训练/不测rank质量。edges2/3先定，原7000主item支持edge3仅494UID，**未看新loss/谱前**并入原sens20261024得到8929，支持UID913/798；全部3.2 eligible75807、main2k包含、reserved零交。finalpop SHA b5fbca0709...；root plan.json已冻结。GPU1 only、论文不改，正反结果保留，旧停止队列不恢复。


2026-10-24 simple_config_sweep/attempt_1本轮完成（无本轮GPU/后继）：width32/64/128×steps256/1024六组×target3/4全12点，旧W64/256复用；另W64/256 native-pair输入控制2点。共12个新quality/384块498288分数，949/932真实反馈UID，GPU1 only、论文不改。W128/1024 Direct−Chain两终点+.4306/+.5241pp逐点UID区间排0，反例W32/1024 target3−.7662pp排0，均开发探索非独立确认；target4 nativegap负，只比两path不讲恢复。nativepair KV MSE增10.09/11%但AUC协议差混合且CI跨0，不能讲overwrite必损。候选新增逐反馈loss分布/CDF全7000纯CPU：正/负/近零并存，主要随label和共享logit下移；同context正label805/813UID内差异小，不能称强candidate特异性。CPU/旧W64权重逐位/actual8/全块/端点/论文audit通过。主报告experiments/docs/ch3_simple_config_sweep_2026_10_24.md，两图及全部正反CSV在新run。


2026-10-24 用户新授权继续第二组简单配置（覆盖前条已完成范围，仅新simple_config_sweep/attempt_1）：固定原2048人口/target3、4/cached-prefix-gap，width32/64/128×steps256/1024，旧W64/256质量复用，5新配置；只Direct/Chain+原生参照，无新增模块，GPU1 only、论文不改、正反保留。先缓存实际fit行复核旧配方一致性，再actual8及>=512真实反馈UID质量。候选新steer为同一校准逐反馈Δlogloss分布，不给单候选算AUC、不以单次标签噪声宣称稳定候选规律。计划在新run/plan.json。


2026-10-24 新收缩实验已完成，无GPU/后继队列：lineage_candidate/attempt_1候选只复核旧7000固定Linear256按见过/未见过分组（commonUID1700/1629），不加候选输入；原生直接vs链式仅四路，在固定2048居民949/932反馈UID上完成原实时gap+唯一cached-prefix-gap两视图，4run/128块166096方法分数。Direct−Chain原两边−.001668/+.007239，cached-gap−.001464/+.008869，四个AUC区间均跨0；V4 native gap负，不能讲高恢复或多次必然退化，亦不证明持久化必要。原anchor/short已拟合但不评分，candidate ridge从未GPU跑；GPU1 only，论文SHA未变。主报告experiments/docs/ch3_lineage_candidate_2026_10_24.md，completion_audit.json。


2026-10-24 最新用户收缩覆盖前条新实验范围：候选只分析**同一固定校准对不同候选的AUC/恢复反应**，不新增候选描述符、ridge或洗牌模型；candidate_information_probe.py为未运行草稿，不启动。状态只评同一目标版本的原生KV直接映射与链式多次映射，加Reuse/Rebuild参照；不评anchor/original+previous。final3/4拟合已自然完成，旧冗余anchor权重保留但不评分。仅GPU1，固定2048居民与>=512真实反馈UID、论文不改；plan.json保留原计划，scope_revision.json记录此次收缩。入口experiments/docs/ch3_lineage_candidate_2026_10_24.md。

2026-10-24 05:16 新用户明确授权小规模GPU检验两条motivation，覆盖前条完成状态，仅限新lineage_candidate/attempt_1：①固定base写轨迹下额外候选描述符ridge/盲/洗牌真实AUC；②普通同预算MLP比较chain/direct/original+same_chainprev，源V1最终3/4，target4补同目标2/3跳。2048固定hash人口，e2/3/4反馈921/949/932、reserved零交；GPU1 only，先一组小配置，再据诊断调整保留全部结果。state为固定cache真实候选/gap机制case非serving，anchor额外两份KV字节如实报；论文不改，不新foundation，不恢复旧取消队列。plan.json与experiments/docs/ch3_lineage_candidate_2026_10_24.md为入口。


2026-10-24 00:42 两轮7000有界补测均完成，无GPU任务/后继：candidate_robustness与candidate_strength各2边、220块504272方法分数，共440块1008544分数；每边3168/3190真实反馈UID。6静态map+6档Linear256全局alpha均保留，主3属性同UID>=512、原4000/新增3000切片齐。alpha1→2整体AUC两边+.013626/+.026029且UID95%排0；alpha4→8两边共同退化；候选反向取舍不跨边/新人群复现，不能称gate必要/有效。Rebuild非真实AUC上界，e3 alpha2超其+.017265且区间排0，为开发结果非独立确认。完整三方法旧新分数逐位一致，GPU1 only；高强度FP32差已FP64及原scoretol论证，旧失败保留，新core未改，CPU最终2pass。两completion_audit=passed、图/CSV/全部反例已保存；主报告 experiments/docs/ch3_candidate_robustness_2026_10_24.md。后续应纳入调强全局映射为强基线，候选动机仍待更直接控制。本轮完成，不启动旧阶梯/nativecross/E6，不改论文、不重训foundation。


2026-10-24 00:25 candidate_strength完整7000评分已启动：PID467869/467870，两边同物理GPU1，alpha .25/.5/1/2/4/8，base为已完成linear256_rows。原actual8在近零cache元素超1e−5；全8对照score≤7.15e−7原门槛全过，sameinput FP64≤1.78e−14；仅新runner本进程cache_atol3e−5，旧core/math/scoretol不改，失败源/日志保留。fresh diagnostic8_validated、alpha1/Reuse/Rebuild旧新对照均passed，CPU后继已耐久等待两边完整summary。不要另开重复任务或改冻结源。文档ch3_candidate_robustness_2026_10_24.md，论文不改。


2026-10-24 00:06 新7000鲁棒性全完成：两边3168/3190反馈UID、220块504272分数；六种静态map/三属性同UID分析与原4000/新增3000切片齐。history e2线性map收益差区间排0，e3变弱且新增3000连同native-gap反向，热门/重复无稳定跨边规律。不能称统一map候选失灵。用户已授权更多配置；追加限定candidate_strength/attempt_1，同7000/两边、已完成linear256_rows权重alpha .25/.5/1/2/4/8，真实AUC对强度与相邻步取舍；全部alpha保留、不按结果选人/阈值、GPU1 only、论文不改。原鲁棒性完成、旧阶梯/nativecross不恢复。

## Active work — 2026-10-19 (read first)

2026-10-23 23:16 用户新授权补配置实验，candidate_robustness/attempt_1 active：原结论仅部分支持candidate motivation，保持原cost证据，不恢复旧阶梯/KV/原生cross。固定mismatch5000+sensA2000=7000，confirmation和全部fit/val/dev零交；前4000重叠，属敏感性非独立确认。原top20%热门不改，history common1700/1629、popular common801/811，secondary出现1次/重复>=2 common825/728均>=512。静态map配置原linear100row+新100user-balanced/256row/256user-balanced(可带diag参照)，F1/F2保留较强静态基线；同head64dim，val16选lambda，无test标签拟合。GPU1 only，新合batch无oracle/KV误差额外开销，源码/实际8数值检查后完整两边新评分。论文不改，不重训foundation；新plan.json及人口/属性已冻结，旧结果保留。

2026-10-23 21:42 本轮已完成并收敛：固定4000两边完整反馈(cap1024+128)4run/252块703920GPU分数、candidate-class AUC/恢复与CPU FLOPs均完成；completion_audit.json=passed，论文SHA未变。主候选linear，预发布history seen/unseen同UID990/913恢复点27.71/41.50%、57.68/71.55%；AUC增益异质性仅后一边区间排零，F2可改变比例关系，不能称所有静态map失灵或gate有效。热门20%规则未调，sharedUID465/451不足512仅诊断。固定Large1024row，F一次5.99GF；F+fullpair h在Q16=107.39GF=5.93×一次Rebuild（解析新增forward，排共同nativequery/fit），图/CSV已保存。当前任务完成，无GPU队列；nativecross/Q64从未启动，不要启动；E6/旧阶梯仍不恢复。主简报experiments/docs/ch3_candidate_motivation_2026_10_23.md；完整探索/反例另文保存。

2026-10-23 21:23 最新用户收敛覆盖前条nativecross：K/V不拆，**不要启动native_cross_quality.py或Q64补测**；它们仅代码/CPU ready，从未GPU运行，不是结果。当前重点为(1)原完整4000真实scores按预发布热门/长尾、本人cutoverhistory见过/未见过分组，简单Reuse/候选无关map/Rebuild AUC与恢复率，配对同UID并检查分母；(2)固定实际Large和历史快照，只变候选数，比较一次map与逐候选逐行修正的已有FLOPs。两项仅CPU，无新模型训练/GPU工作；原已完成第一批结果及所有反例保留。用户要最简单、直观、与Design有关的1–2条，不再堆rank/能量新指标，不force候选gate必要性或阶梯。见experiments/docs/ch3_real_quality_angles_2026_10_23.md。

2026-10-23 21:09 真实反馈第一批已完成，新增原生K/V端点诊断继续：real_quality_angles/attempt_1 四run(cap1024与128各两边)全exit0，252块703920原始GPU分数行、四份500UID分析及两份candidate-drift分析完成。cap1024 Reuse/F1/F2/Linear/Rebuild AUC边2=.671848/.676521/.678804/.676149/.681184、边3=.704639/.715270/.725577/.725307/.734606；不能把F1当静态映射上限。FIFO后gap淡出、不支持越追加越坏；候选embedding漂移未找到稳定异质性。原论文SHA不变。小cap成本原分母后端错已用独立cost_reference_corrected附件修正，不改原分数，主1024成本无变化。现在采纳用户外部建议前两项，准备固定4000两边native old/current K/V四格、同cutover复用64候选读，不修改已冻结旧源、不做row-rank。GPU1 only；旧阶梯仍不恢复。详见experiments/docs/ch3_real_quality_angles_2026_10_23.md及initial_batch_completion_audit.json。

2026-10-23 20:36 新真实反馈多角度探索进行中：用户要求不限单一映射精度例子，围绕Design寻找多个insight。real_quality_angles/attempt_1固定main2000+sensA2000=4000，fit/val/dev/confirmation零交叉，两边1838/1834反馈UID；只用GPU1，论文不改。先actual8核对合批native路径，再完整反馈AUC/logloss、强度/K/V/简单平移线性/时间与预发布分组，后续cap128/512检验FIFO后的上下文影响。数值检查首尝试allclose失败，诊断中，不能当质量结论。人口/覆盖已冻结，原实验和旧暂停队列不恢复。见experiments/docs/ch3_real_quality_angles_2026_10_23.md。

2026-10-23 20:08 拓展机制分析完成：insight_extensions/attempt_1新增6个main2000运行（query-gap、候选panel、校准源2x2各两边）均exit0，GPU1 only，当前本轮无GPU任务；相关35项CPU检查通过，权重/输入/源/角色与论文SHA核查通过。两个候选正向角度：broad item描述器解释25.07/10.49%中心化read残差，但仅占总read0.01905/0.004927%；near teacher次序不一致0.673→2.658%、0.775→3.403%。973/1027固定哈希组方向均保留（同权重敏感性，不是独立确认）。固定query gap后绝对中心误差增长消失，teacher尺度归一化也下降；不能讲历史越长越坏。源匹配无一致优势，near/logit可退化；全部反例保留。候选read输出rank1约99%仅oracle几何，不能推导每行code秩。完整文档含动机链、费用与图：experiments/docs/ch3_insight_mechanisms_2026_10_23.md；新root completion_audit.json及candidate_context_angles.png/pdf。原论文和上一轮原结果未改，不恢复旧阶梯。

Latest new-goal completion2026-10-23: bounded insight_mechanisms/attempt_1 study COMPLETE: main_edge2/main_edge3 and version_main_edge2/main_edge3 all2000 users/32blocks each exit0, paired500UID analyses,128-block/source/role/paper audit pass,CPU18 tests pass. GPU1 only; no GPU job remains. Version-normalized old-state residual grows18.51→36.31% and29.49→104.82%, paired CIs positive; refresh reduces new-target residual32.51/62.30% but mixes extra capacity/fitting. Within-release total F error DECREASES; oracle new-row replacement reduces residual5.62/12.82% atappend128. Candidate gate necessity NOT established (centered energy0.10–0.19%). Main conclusion: mapping helps, compatibility is target-version-sensitive; no monotonic worsening or final-Design validation claimed. All results/PNG/PDF, costs and limits in experiments/docs/ch3_insight_mechanisms_2026_10_23.md; completion_audit/combined_observations under run root. Paper SHA unchanged. Do not resume old ladders or expand releases by default; this bounded new goal is complete.

Latest USER NEW GOAL 2026-10-23 supersedes pause for new work only: 3.3 now mechanism case studies, not f/fg/fgh ladder or forced recovery targets. Two successful release-edge cases suffice; retain all explored failures and limit claims. GPU1 only, paper unchanged. New insight_mechanisms/attempt_1 fits fresh F-only full_item W64/N100/256steps/lr1e-4 on edges2/3, main2000/val16 isolated; tmux ch3-kv-insight-fit-gpu1. New kv_evolution/candidate_residual helpers +CPU11 tests pass. Read experiments/docs/ch3_insight_mechanisms_2026_10_23.md and STATUS top. Old paused ladder queues MUST NOT resume. Goal active for new study.

Latest 2026-10-23 14:30 USER PAUSE OVERRIDES ALL QUEUES: Stop3.3 and think through observed failure, strength of hypotheses, and implications for Design. Coordinator223543/scorer229923 terminated, both absent; user_pause.json preserves13/15 points. Edge5 F2000 completed−109.1727%, FG/H incomplete. Do not resume any queue or launch experiments until explicit user resumption. Keep adverse data. User now permits stopping expansion after1–2 clearly bad edges; no requirement to finish every edge before diagnosis. Read-only analysis requested, paper unchanged, goal paused.

Latest 2026-10-23 14:14: Cheap5% round now11/15 complete. Edge4 F/FG recoveries−178.9111/−233.4789%, rawAUC .69585928/.68925193 versusReuse .71752272 andRebuild .72963122. Preserve adverse results; H first256 committed, actual229923 live under223543. All5 F selected final128 steps, validation still improving, NOT saturation. Edge3 F/FG/H38.6662/41.1123/46.8025%, edge2 56.8622/66.8224/78.0712%, edge1 21.9592/34.1505/34.1505%. Finish H+edge5/all15 before adjusting. Then diagnose calibration time coverage, inadequate optimization, row-map capacity and causal state semantics using prerelease validation; no new branch yet. GPU1 only, goal incomplete.

Latest 2026-10-23 13:05: Simple5% round attempt2 now3/15 complete, edge1 F/FG/H recovery21.9592/34.1505/34.1505% on2000 users. H selected step0; all10590 FG/H scores and event fields bit-identical, saved edge_1_zero_h_consistency.json. Actual scoring PID229923 live under223543, continues remaining4edges on GPU1. Keep fixed recipe, finish all15 before tuning/expanding; goal incomplete.

Latest 2026-10-23 12:35: Old fixed512 center_std/N256/W128/cap2048 run finished15/15 +analysis, PID214313 exit0. F/FG/H recovery54.919/49.660/65.243%, Hgain15.583pp positiveedges2/3/4, H AUC CI[-.001141,.004908] crosses0; Gnegative, twoHzero. Fcost1331%Rebuild makes it expensive reference only, not selected cheap method. Added comparison_v3.csv9configs/135points with131-artifact +execution-binding verification; priorv2 unchanged. USER-priority simple_trend_round/attempt_2 remains fixed5%/100fit/2000test/W64/128steps seed17, live scoring229923 under223543. FirstedgeF2000complete recovery21.96%; do not tune/stop after first point, run all15 first. No new configs/precision/preparation; GPU1 only. See STATUS; goal incomplete.

Latest 2026-10-23 12:27: simple_trend_round/attempt_2 completed ALL5 fits/15weights, zero errors; all5 actual budgets pass. cost_preflight_points.csv contains15 real FLOP rows, no AUC. One actual edge2 trained-H8UID check_projected passed in30s. Main2000 actual scoring now live PID229923 under coordinator223543, original15point quality_plan, GPU1 only. Rootprogress still says fittingcompleted5 because it is not updated per phase; inspect live child/quality.log, do not restart from stale progress. Old512 PID21431313/15complete. No new configs or precision branches: finish this one full cheap round first, then interpret. Old expensive formal/E6/N256 queues remain superseded; goal incomplete.

Latest USER STEERING 2026-10-23 12:25: prioritize ONE complete cheap trend round, not broad preparation/precision. Old formal2k/E6 and N256W1282048 coordinators/workers intentionally superseded, committed data preserved; DO NOT resume them. New simple_trend_round/attempt_2 PID223543 is active on GPU1: fit100/val16/main2000 (5% gradient ratio, teacher+val5.8%), layerwise W64depth1,Q16, each128fixed steps,valevery32,pat0,seed17,center_std/balanced/cutover/fb75. Attempt1 cap256 measured Hcalibration100.64% above budget, preserved then reduced all stages equally. First2edges complete, Ffullcost27.36%, all cumulative fits15.8/19.5/57.1%<Rebuild. All5 cost checks -> one actual8UID numeric -> complete15main2k points -> full CSV/cost/UID analysis, then interpret. Original fixed main2k has all5 positive baseline gaps .012-.042, inputs/pop SHA match; no need more population search. Keep existing batched512 PID214313 to finish15points(11complete). Old OS-stopped parent153714 now retired safely; no holds left. 3seed/independent confirmation remain later requirements, goal incomplete. Read STATUS top; only GPU1, no paper edits.

Latest 2026-10-23 11:49: Verified batched512 handover occurred after all3 edge1 points committed; new query_centering_batched_development_queue/attempt_1 PID214313 now4/15complete +FGedge2first256. Old153897 intentionally interrupted/Z; preserve original point/block SHAs and hardware, no rerun/discard. Newdriver writes SINGLE/development_exit only after all15 points and analysis; oldparent153714 remains OS-stopped to suppress obsolete formal. Migration even-main151411 genuinely exit0; main7141 still live, formal2k213176/E6156641 waiting all15 pairs. Edge1 full512 Rebuild−Reuse=-.00472431, F/FG raw gains positive despite negative recovery; FG/Hscores bit-exact, no Hgain. Fixed N128 edge1 seed17/23 actual Houtput0 vsseed29 nonzero825600 parameters and slightly lower teacherloss than extendedFG; saved formal_seed_training_diagnostic/attempt_1 (13/15jobs complete), not quality or deterministic-bug proof. OnlyGPU1; see STATUS top; goal incomplete.

Latest 2026-10-23 batched append input optimization: actual8UID FG+nonzeroH parity passed bit-exact scores/finalKV/causal ledgers; CPU9tests pass. Profile found34.94/37.97s in6231 scalar torch.tensor CUDA transfers. New batched_append_inputs.py + run_quality_batched.py bind helper/readiness and separately count transient input bytes; original numerical sources/FLOPs unchanged. Formal2k queue now formal_quality_batched_queue/attempt_1 PID213176, GPU1 only, waits all formal fit exit markers. Old CPU coordinator153714 is intentionally OS-stopped to suppress duplicate obsolete formal dispatch; its development child153897 continues. Do NOT CONT old parent; new coordinator retires it after development_exit exists. New quality outputs formal_quality_batched/attempt_1; old512 outputs preserved. Layerwise edge1 pairedFG exact and common18cells done, but full-item Hteacherloss still lower; do not claim superiority from positive h increment alone. See STATUS top; goal incomplete.

Latest 2026-10-23 11:09: Single-GPU1 only remains in force; GPU0 now has another user lty's PID175134, do not touch. New actual audit95weights(main50/curve45), complete9main/45curve; quality incomplete. Added isolated layerwise_query_centering + run_probe_layerwise_centered (old frozen sources unchanged), correct per-head kv_output_scale synchronization, CPU5tests and actual Large8fit/2val raw gradient/state/read/score oracle passed (maxscore2.384e-7), no quality from diagnostic. Durable layerwise_centered_queue/attempt_1 PID178606/178804 compares standard/balanced h at N256/W128/2048 on all5 edges then90 common validation cells; no automatic formal promotion or population dispatch. Original formal main/migration/512quality/E6 wait continue; see STATUS top.

Latest 2026-10-23 single-GPU allocation: user accepts GPU1 only; physical GPU0 is now idle (0MiB verified). All new experiments MUST use physical GPU1. Resume from STATUS top and experiments/ch3_motivation/runs/gpu1_only/attempt_1/resource_change.json. Old waiting dual-GPU coordinators14207/22635/55899 intentionally superseded; old GPU0 main interrupted with stages preserved, GPU0 curve genuinely complete. Migrated even-main PID151411, quality queue153714; original GPU1 main7141/curve7148 continue. E6 attempt2 coordinator is single GPU1 (Gram actual both-protocol all8/all-rank comparisons now passed, maxabs2.384e-7, readiness bound); no population E6 yet. Do not resume old dual-GPU queues or rewrite old exits/provenance. Goals incomplete.

Latest 2026-10-23 10:23: E6 selected_release8 AND selected_timestamp8 both passed actual formal FG/H gates; coordinator now waits all15pairs. Current formal audit55weights(main29/curve26),31jobs complete, population quality incomplete. Query-centering selected-development queue PID55899 waits all90pre-release cells, then actualH GPU checks and fixed512 full ladder. New projected-SVD Gram candidate is DIAGNOSTIC ONLY: primitive CPU/GPU evidence exists, actual same8 queue PID75992 waits reference exits, comparator PID78641 fixed1e-5 all-rank scores. Do not enable it or edit frozen runtime sources without actual comparison. Exact SVD spotchecks already occur only initially, not every snapshot; formal count2 remains. Read STATUS top; all goals remain active/incomplete.

Latest 2026-10-23 09:58: Read experiments/STATUS.md top. New query-centering edge2 pair fits,18common cells and actual trained-H GPU oracle passed (wrapper Path failure retained; independent root hash verification explicit). Common Hloss improves~9.55%; no population/five-edge claim. Matched RMS/center_std fits now extend edges1/3 on GPU0,4/5 on GPU1, PID43177/43180; do not duplicate. Formal fitting first edge2seed17 ladder completed, remaining jobs live. E6 formal_factorized_queue/attempt_1 now runs ACTUAL selected FG/H release8/timestamp8 diagnostics (PIDs40691/40687); stale missing-weight progress is not terminal, inspect ps/logs. E5/E6 formal quality still incomplete; preserve all signs and goalactive.

Latest 2026-10-23 09:40: E3/E4 accepted complete; focus E5/E6. Read experiments/STATUS.md top. Equivalent h first-affine GPU gradient/state/read oracle passed. formal_factorized_queue/attempt_1 now runs four durable fitting processes (15 main ladders including controls plus45 F curves) on GPUs0/1; query_centering_pair_queue/attempt_1 separately runs matched pre-release edge2 RMS/center_std fits. Sources frozen; do not duplicate or alias diagnostic weights. All formal quality/E6 still incomplete, expected stable trend not established; goal remains active.

Latest small-width limitation — 2026-10-23: actual W32 complete512 F costs
on edges1/2 are86.76/86.01% Rebuild, not the16.68% preparation-only floor.
A weight-only full-state audit (small_width_calibration/attempt_1/
nested_null_audit.json) proves H identically zero for finite inputs on W32
edges1/2 and W64 edges1/3/5, with exact predecessor FG state. Thus these
small-budget recipes cannot meet4/5 positive H increments; do not promote
them as the formal mechanism configuration or pick a non-best checkpoint.
Retain their real cost/quality exploration; focus main trend work on original
PhaseB/active scenario and justified calibration/conditioning diagnostics.
No population quality metric was computed by this parameter audit.

Latest small-width quality launch — 2026-10-23: small_width_calibration and
small_width_common/attempt_1 both GPUs completed0, all30 weights/90 cells.
W32 H common loss1.684616 versus W64 1.676031; predeclared1% minimum band
selectsW32 within this separate cost sweep, not global475 or superiority.
small_width_projected_readiness/attempt_1 completed0, both actual64 and
artificial32 pressure nonzero execution checks passed; pressure stays forbidden
for quality/E6. Original kernels/weights unchanged.
Selected actual15-point original512 quality is running in
small_width_quality_queue/attempt_1, workers4051666/4051669, sessions
ch3-small-width-quality-0/1. Exact same-goal native anchors are reused only
after UID/input/runtime/backend/model/key/shard SHA checks; new probe states
and complete costs are recomputed. FirstF512 committed, not a full ladder.
Five-edge CPU aggregate successor is small_width_aggregate_queue/attempt_1.
Keep all active new scripts frozen. Formal2k/three-seed/independent/curve/E6
remain incomplete and final475 is still separate. Read latest STATUS/live state.

Latest small-cost exploration — 2026-10-23: five full original512/seed17
configurations are summarized in development_comparison_aggregate/attempt_1/
comparison.csv; h increments+2.36/+2.23/+1.97/+3.51/−0.25pp, all UID intervals
cross0 and positive point edges2/3/5. They share users/weights, not independent
repeats. Frozen/init three-group aggregate queue completed08:03:04.
Actual128 f preparation alone is65.11% Rebuild; analytic width32 preparation
floor16.68% excludes calibration/setup and is not new quality evidence.
Six actual Large width32/64 resident-fitter numerical/timing cases passed,
peak allocated13.36–17.33GiB, no quality weights/AUC. Fresh two-width fits now
run in small_width_calibration/attempt_1 (4020944/4020949), followed by
small_width_common/attempt_1 (4024255/4024258) for90 pre-release validation
cells. Both keep original numerical source files unchanged. New fit configs:
full_item32/64,fit64/val16/Q16/fb75/cutover,lr1e-4,cap256,pat128,val32,
seed17. This is a separate reserved64/cutover cost recipe, not merely a width
change from fixed256/gap_mix50 and not a final475 replacement.
Before population quality, require small_width_projected_readiness/attempt_1
actual exit0: actual64 plus artificial32 nonzero-H stress on8 fitUIDs.
The pressure .pt is diagnostic_only and quality_use_permitted=false; never use
it for quality/E6. Original trained weights untouched. Keep all new active
run.py/run_sweep.py sources frozen. Main/confirmation/three-seed/curve/E6
remain incomplete; continue from STATUS and live processes.

Latest active-scenario development — 2026-10-23: feedback50 completed all5
edges/15 points; h mean recovery increment only+1.9673pp, AUC UID interval
crosses0. Complete costs/CSV and independent checks are in
runs/e5/development_comparison_aggregate/attempt_1/feedback50_mix50.
A separate pre-first-release active cohort now freezes512 development plus
2000/2000 supplemental held-out users, excluding original roles/sensitivities
and E6 reservations, before future requests are loaded. SameUIDs all5edges;
no future-activity intersection or model-outcome selection. Original roles
remain unchanged. CPU five-edge packs/calibration-history preservation passed.
Active queue is active_pre_first_release_quality_queue/attempt_2 ONLY,
coordinators3998826/3998830; sessions ch3-active-pre-v1-quality-0-v2/1-v2.
Attempt1 was cancelled for a plan UID-hash metadata correction before method
quality plans; numerical sources unchanged, bounded native progress retained.
Do not restart attempt1. Keep new prepare_active_development.py and
qualify_active_development.py frozen while running. It evaluates actual512
fresh three anchors then original75mix seed17 probes, all signs retained;
no new fitting or formal-config substitution. Formal E5/E6 remain incomplete.

Latest E5 scheduling efficiency — 2026-10-23: the selected development
waiting queue is now selected_development_queue/attempt_3, coordinator3972164,
session ch3-selected-development-3. Attempt2 was cancelled while waiting with
no GPU child; do not restart it. New frozen prepare_selected_development_reuse.py
indexes exact checkpoint SHA only, then invokes the original full planner audit;
actual62 completed candidates/514 artifact hashes passed readiness. It never
selects recipes by score. New inference admits with measured32GiB free memory,
not whole-GPU idleness. Actual final475 selection remains pending; no formal
training was launched. Initialization edges1–4 and candidate50 edges2/4 have
mixed effects, not an established h trend. See STATUS for actual current state.

Latest Chapter3 development milestone — 2026-10-23 06:50: both original
feedback75 time-law quality runs completed all5 edges/15 points and paired UID
analyses. Five-edge h recovery increments are only+2.36/+2.23pp, with3/5
positive points and mean AUC-difference intervals crossing zero; expected E5
trend is not established. Full raw/corrected development512 costs and source
bindings are in temporal_comparison512_aggregate/attempt_1. Fixed calibration
costs are large; do not relabel512 costs as2000/30k costs or call f cheap.
Both feedback50 quality workers and new standard/balanced h initialization
quality workers are actually running beside PhaseB/supplementary work. The
initialization comparison audits and reuses only exact architecture/state/
calibration-ledger-equivalent completed points and verified same-goal native
anchors; all new H trajectories/costs are measured. Its frozen entry is
scripts/ch3_motivation/evaluate_probe_initialization_pair.py and queue is
initialization_pair_quality_queue/attempt_1. Keep all active sources unchanged.
Formal E5 three seeds/main/confirmation/fit-size curves and E6 remain incomplete.
Resume from STATUS for actual PIDs and current inputs, not old queue attempts.

Latest E5 supplementary protocol — 2026-10-23 05:40: joint fits and
fixed512 time-law quality remain running. A separate new-term-only protocol
reuses the exact trained f, freezes f during FG and f/g during FGH, retaining
its own driver/training_protocol identity and conservative backward cost.
Its five-edge fitting queue is frozen_predecessor_queue/attempt_1; common
validation and fixed512 quality follow in frozen_predecessor_quality_queue/
attempt_3, admitted with measured32GiB memory headroom. Common validation
completed for edges1/2 and actual512 quality is running on both GPUs. No
final supplementary five-edge result exists yet; development edges1/2 are
complete, with mixed/negative AUC effects retained. The failed/cancelled
quality attempts1/2 must not be restarted. These are development supplements, not replacement formal results.
Do not edit frozen numerical/audit/planner sources of active queues. Resume
from STATUS for current PIDs and actual submitted blocks; formal E5/E6 remain
incomplete and no final protocol has been selected.

Latest section3.3 exploration clarification — 2026-10-23: the user explicitly
permits concurrent experiments on a GPU when measured memory headroom is
adequate; complete GPU idleness is not a prerequisite for quality/FLOPs work.
Use both GPUs actively, keep code efficient, and explore scientifically justified
configuration, test-protocol and implementation changes to expose the f→fg→fgh
mechanism. Preserve causal data, matched comparisons, information available to
each component, and retained adverse outcomes. Record actual concurrency and
memory; shared execution wall time is progress, not controlled timing evidence.
Do not modify frozen numerical sources in an already running job; introduce
new exploratory implementations separately and verify them before costly runs.

Latest Chapter 3 priority — 2026-10-23 04:06: the user accepts the current
section 3.2 results as complete and explicitly shifts the remaining effort to
section 3.3 (E5/E6). Preserve E3/E4 results, stop expanding their experiments,
and use both GPUs for a justified E5 configuration and its E6 low-rank study.
The not-yet-started natural-active E3 queue was cancelled; its input cohort is
retained but is no longer a mandatory comparison or dependency. E5 may use a
different configuration from E3. Keep f/g/h forms and objectives, inspect
implementation and efficiency, explore candidate/time sampling, fitting size,
budget/capacity/initialization, then validate the chosen configuration across
five edges, three fitting seeds and the independent population. No selected
E5/E6 result is yet complete. Resume from the newest STATUS entry; stopped
waiting queue attempts must not be restarted.

Latest Chapter 3 configuration clarification — 2026-10-23: Motivation
sections 3.2 and 3.3 may use different scientifically justified configurations,
including different prerelease-defined populations, history settings and
calibration settings on the same model. Do not require one winning recipe for
the entire chapter. Explore configurations for the mechanism each figure is
intended to expose. Keep methods within a comparison on matched inputs and
fresh same-population Reuse/Rebuild anchors; record conditions and all explored
outcomes. E5's five-edge common recipe is a within-figure comparison, not a
cross-figure constraint. E6 must identify the actual frozen h and paired FG
whose interaction it analyzes. Cross-figure quantitative superiority claims
need matched-condition evidence; distinct scenarios support conditional
mechanism findings. Continue both GPUs and existing useful exploration; do not
rewrite running frozen configurations or historical results.

Latest Chapter 3 authorization — 2026-10-22 evening: the active goal is to
recompute Yambda-Large E3–E6 on partial populations using g201 physical GPUs
0/1. This supersedes the Chapter 3 plan's old E3/endpoint reuse instructions.
Recheck every imported numerical path against the latest protocol; old results
are historical comparisons, not authoritative new measurements. First verify
numerics and efficient execution. Fewer than 512 evaluation users may ONLY
produce numerical/performance diagnostics, never paper quality results.
Development is 512, E3/E5 main and independent confirmation each 2000, E4 at
least 5000, E6 at least 2000 feedback users per edge. Preserve existing role
separation. If trends differ, examine reasonable configuration/rule changes,
multiple fixed population draws, training randomness and deterministic effects;
retain contrasting outcomes and choose configurations on validation. New work
lives in scripts/ch3_motivation and experiments/ch3_motivation. No paper edits
or foundation retraining. Missing assets may be scp'd from g202. Both Large
V0–V5 models and new five-edge CPU input packs are already present on g201.
Latest population clarification: qualify the intended update/mismatch scenario
using freshly measured OldFull / Current Reuse / Current Rebuild on multiple
fixed 2000-user panels and causally defined pre-release strata. Retain every
candidate panel. Outcome-conditioned scenario selection must be labelled and
checked on an independent panel; do not prune users by a compared method's
quality or force all users/edges to have favorable outcomes.

Latest RecFlow-Max completion — 2026-10-22 afternoon: Large and Max V0–V5
foundation training both finished naturally; Max completed 95,082 optimizer
steps and all twelve permanent epoch checkpoints. The user's Max Recall@100
request is now complete on physical GPUs 2/3 only; both inference processes
exited 0. GPU 0/1 jobs were untouched. Max uses its own 3.5M-video catalog and
new deterministic candidates, not Large panels/predictions. Actual prepared
data shares Large's frozen 3,210 final-role users and 94,809 D20–24 requests;
do not invent a different test population or alter roles after observing results.
The protocol remains observed positives plus 1,000 unique uniform distractors,
including all-negative/OOV zeros. Recall@100 gains are +10.56473/−2.60571/
−4.66152/+7.06526/−2.28181%; only 2/5 positive, so this does NOT satisfy the
user's three-of-five positive-release rule. Equal-release mean is +1.61619%,
paired UID95% [0.37400,2.93966]%; current/random ratios 1.924–2.180.
Both positive edges and the V1→V2/V2→V3 regressions exclude zero in pointwise
UID intervals; V4→V5 crosses zero. All seven metrics remain in the analysis;
Recall@50 and MRR have three positive point estimates but their mean intervals
include zero. Do not switch the headline after inspecting Max outcomes.
See experiments/foundation_recflow/runs/chain_2gpu/attempt_1/max/
sampled_retrieval_full/{paired_uid_analysis/summary.json,metrics_by_release.csv,
completion_audit.json}. All 1,484 rank blocks, eight frozen data hashes, ten
training sources and six final checkpoint marker identities/sizes passed.
Inference took 20m43s with peak reserved memory 7.15GB. No further job queued.

Latest Yambda-Large completion — 2026-10-22: all five independent adjacent
baseline edges V0→V1 through V4→V5 are complete (24 points/edge, 75,807 eligible
quality/cost users per edge). The GPU 1 four-edge queue finished at 08:07
Shanghai, taking 12h27m14s; no Yambda-Large job remains running. All 95 committed
score blocks, input/config hashes and formula/row-count consistency passed.
Full 120-point report, CSV, TeX and derived summary are in
experiments/01_quality_cost/runs/yambda/large/baselines_all_edges/seed_17/attempt_1/.
Original edge summaries/weights/scores were not rewritten. V4→V5 remains
E14_partial; its wide recovery intervals do not establish superiority over
Rebuild. Layer2 recovery is 84.88/85.48/55.92/47.28/60.46% across the five
edges at about 21.04–21.05% cost. All 15 Query-only point estimates are below
Reuse; History256 falls to 4.57% and −40.42% on the last two edges. Preserve
these adverse outcomes and diagnose independently rather than tuning on test
outcomes. No new experiment or CacheTide quality run was launched.
Latest RecFlow metric clarification — 2026-10-22: the user wants common,
task-natural metrics (AUC/NDCG/Recall/MRR), roughly 1–3% relative update gains
with at least three of five releases showing a clear phenomenon, plus scores
clearly above a matched random baseline. RecFlow should emphasize a
generative/retrieval task rather than only like/dislike discrimination; this
new task preference supersedes the older RecFlow feedback-only planning
direction, without rewriting existing results. Current Large/Max checkpoints
are still effective_view BCE models with a scalar candidate head. Their
sampled positive-item retrieval results are not autoregressive or full-catalog
generation evidence. No change to the running training recipe or new
foundation training is authorized by this metric search.
The user explicitly instructed: stop ONLY this new metric-search inference on
GPUs 0/1; DO NOT stop the existing foundation training. Metric inference was
migrated to physical GPUs 2/3, retaining 177 committed blocks and original
hardware/hash provenance. Foundation tmux session recflow-foundation and
training PIDs remain running on 0/1. Read experiments/STATUS.md and
experiments/foundation_recflow/runs/chain_2gpu/attempt_1/large/
sampled_retrieval_full/plan.json. Full inference completed in tmux sessions
recflow-metric-full-2/3 (socket recflow202), covering all 94,809 eligible D20–24
requests, includes all-negative/OOV zero scores, and retains NDCG/Recall
20/50/100 and MRR, including regressions. Inspect gpu2/gpu3 outputs, not the
stopped gpu0/gpu1 attempts. NDCG@50 and Recall@100 were selected as focus
after a 1024-request/day pilot; full-population verification includes that
pilot and is exploratory, not independent confirmation. The user's three-of-
five rule does not permit averaging only favorable releases or describing
positive point estimates as statistically confirmed gains. Full-population
results corrected the pilot: NDCG@50 has only one positive edge and mean
-1.56585%; MRR also regresses on average. Recall@100 has five positive point
estimates (three above1%), mean +0.94667%, paired UID95% [0.32784,1.58641]%,
and current/random ratios 2.107–2.272. Only V3→V4's per-edge Recall interval
excludes zero; do not call three or five releases statistically significant.
Recommend Recall@100 for current candidate coverage, retain adverse NDCG@50
as ordering evidence. See sampled_retrieval_full/report.md and
paired_uid_analysis/summary.json. Both inference jobs exited0; original
training continues unchanged on0/1. This does not launch further jobs.

Latest execution authorization — 2026-10-21 evening: after the completed
Yambda-Large V0→V1 report, the user requested all remaining independent adjacent
edges V1→V2, V2→V3, V3→V4 and V4→V5 on physical GPU 1. The four-edge sequential
queue started at 19:39 Shanghai on g201. Resume from experiments/STATUS.md and
experiments/01_quality_cost/runs/yambda/large/baselines_v1_v5_gpu1_201/attempt_1/.
Keep the same 24 natural points, all 75,807 eligible quality/cost users and
4096-user checkpoints. Each edge fits its own six learned points and profiles
its own layer intervals; reuse UID roles only, not correction weights or state.
V4→V5 [287,301) includes partial source day 300 and must remain E14_partial,
not complete 14-day evidence. Do not rerun V0→V1 or launch CacheTide quality.
The queue freezes source/config hashes; do not edit its execution Python files
while queued edges remain pending. Runtime estimate: 3–4h/edge, 12–16h total,
progress only, not controlled timing evidence. This supersedes GPU 0 and the
first-edge-only queue restriction in the historical paragraph below.

Latest RecFlow execution clarification — 2026-10-21: prepare a single sequential
job on machine 202 using TWO GPUs (NVLink pair 0/1): finish Large V0-V5, then
Max V0-V5. This supersedes the earlier four-GPU and stop-after-V1 scope.
Every version uses two complete epochs, preserving all twelve permanent full
model/AdamW/RNG/progress checkpoints per scale, fresh -> v01 -> v02 -> v11 ->
v12 -> ... -> v51 -> v52. V0 fits D1-18; V1-V5 fit only D19 through D23,
one new day per version. Inherit optimizer state within each scale; Max starts
fresh independently. Hourly rolling recovery retains two generations and
permanent epoch checkpoints are also resumable. Preserve frozen roles and
paired multi-metric future-day evaluation on all eligible final-role targets.
The user has NOW explicitly authorized starting training and requires a durable
launch that survives network/client disconnection. The formal sequence is
running in detached tmux socket recflow202, session recflow-foundation, on 202
GPUs 0/1. Entry: scripts/recflow/run_foundation_chain.py; configuration:
experiments/foundation_recflow/configs/chain_large_max_2gpu_202.json now has
training_authorized=true. Run: experiments/foundation_recflow/runs/chain_2gpu/
attempt_1; launch.json records commands/PIDs/logs. Do not change frozen numeric
source, design/run configuration or data during execution. Resume only after
the original process has exited; never start a duplicate sequence. Source data
is already on 202; earlier copying to 201 did not start training there.
The user additionally requires explicit GPU-memory risk assessment and long-run
reliability. Large remains global96/micro48 per GPU. Max now uses global80,
micro20 per GPU with two accumulated microbatches, replacing micro40 to leave
headroom. Cover all 15 bounded history/candidate shapes, repeated mixed replays,
full checkpoint saves and fresh-process recovery before reporting readiness.
Do not equate these bounded probes with an already completed long training run.

Latest RecFlow planning clarification — 2026-10-21: the user designates the old
4096-user 6L/H192 generative chain RecFlow-Medium and requests a design only for
Large/Max matching Yambda parameter counts and structure, with different video
table sizes. See experiments/docs/recflow_model_design_2026_10_21.md and its
config/statistics. Proposed Large/Max have 2.2M/3.5M independent videos,
10/16 layers, width320 and 65,536 OOV hash rows each: 730,211,201/1,149,285,121
parameters. Both use the existing development population and preserve
admission/final roles; original Medium NDCG is not feedback AUC. This design
does not authorize new training, GPU probes or a launch queue.

Latest user clarification — 2026-10-21: experiment 01 compares independent
adjacent-version transitions. All methods start each edge from the same old
native state; correction/student state is not carried between edges. Normal
causal reads/appends within an edge remain required. Experiment 02 alone tests
our method's continuous multi-release state; do not require baselines to support
continuous releases or reuse experiment 01 edges as a continuous trajectory.
On machine 201 the user initially allocated physical GPU 1, then changed
subsequent experiments to physical GPU 0. The user authorized fetching needed
data and existing V0–V5 models from g202 using the sync workflow. Current work
prioritizes baselines only, first Yambda-Large V0→V1, roughly 3–5 natural
operating points per adjustable method, with method-specific counts/settings.
Latest clarification: choose natural controls, then report actual FLOPs; do not
invert costs to force exact 15/50/80% points. Layer uses 2/4/6/8 layers; Tail
uses 10/20/40/80% tokens; Deviation/Query use 10/20/40/60% selected positions.
The original roughly 15–80% cost range is guidance, not an exact grid.
Our method's quality runs and
other version edges wait until this first comparison has been reported.
The user subsequently required full eligible test-population quality, rather
than the initially proposed 512-user pilot. Preserve fit/validation and earlier
calibration/development reservations; do not exclude ordinary historical test
users based on their outcomes. The current Large V0→V1 test pool is 75,807.
Before a population run, optimize execution and report a measured-input-based
30k-user runtime estimate. This is done: the 24-point plan is 70–100 minutes for
30k resident users and about 3–3.5 hours for all 75,807 eligible users.
The full run completed on GPU 0 in 3h06m on 2026-10-21: 75,807 eligible users,
34,190 feedback users, 345,630 feedback events, all 24 points and UID intervals.
Resume from experiments/STATUS.md; this task has no active GPU job.
Selected Layer intervals all start at zero and require no persistent old hidden
boundary cache. The earlier generic 45% claim was corrected in report metadata;
all measured metrics/FLOPs/scores/weights are unchanged. Layer2 is 84.88% recovery
at 21.04% added FLOPs / once-Rebuild, not the earlier partial-preview ~95–98%.
Full deployment costs still include events
omitted solely because they cannot affect an independent edge's scored quality.
Latest reliability instruction: save results promptly and support bounded-loss
resume. The runner commits each 4096-user score block (normally about 8–10 minutes)
and reuses completed fitting; resume validates frozen settings, inputs, code,
weights and block hashes. Never discard an entire long run after interruption.

Latest organization — 2026-10-21: formal experiment plans, configurations and
new results are centralized under `experiments/`. Resume from
`experiments/README.md` and `experiments/STATUS.md`; readiness and scheduling
are in `experiments/docs/experiment_readiness_2026_10_21.md`. Quality/FLOPs
runs may share GPUs; formal timing requires controlled resource conditions.
Legacy system-study logs and results retain their historical paths. This
directory organization does not launch or queue experiments.

Latest evaluation-scope clarification: the user explicitly excludes end-to-end
serving experiments, including the briefly proposed limited arrival replay.
Validate the design using real-model feedback quality, continuous cache/code
evolution, complete FLOPs, total auxiliary memory, and controlled measurements
of preparation/read/append/handover paths. Keep calibration, shared preparation,
all per-event work and transfers accounted for; measured path time is not online
throughput, p99 or complete-system wall time. Optional scale-time estimates must
state their assumptions and remain distinct from measurements. This supersedes
older service-platform and full-cycle-time requirements below. The active task
is evaluation planning, not launching GPU work. See paper/new_eval.md and
docs/system_study_2026_10/evaluation_rationale.md for the claim/evidence audit.

Latest user clarification during evaluation figure review: final evaluation covers
Yambda and RecFlow, each at Large and Max scale (four conditions). Medium is
development-only and is not a final-result configuration. This supersedes the
earlier Medium/30k final-evaluation priorities below; existing evidence remains
unchanged. See paper/new_eval.md and eval_figure/README.md for the latest plan
and explicitly synthetic figure previews. RecFlow Large/Max model/task/population
settings still need to be established; its existing six-layer NDCG evidence must
not be relabeled as feedback-AUC results. This figure task launches no training.

Later evaluation-planning clarification: foundation models are fixed backbones;
do not retrain their chains for multiple seeds. Proposed repeats concern only
the paper's correction-training modules and key learned baselines. The current
new_eval.md separates training seeds, fitting-user samples and timing repeats,
with single-run exploration and shared main trajectories to limit costs. This
is an evaluation plan, not authorization to launch a new training queue.

The user authorized organizing and beginning the parallel system study. Resume
from `docs/system_study_2026_10/README.md` and its `STATUS.md`; these point to
the actual commands, outputs and pending work. Do not rely on chat memory.
The current reference is `A_itemgate6_rms_roll4` (query_relay_2026_10), with
shared12/private6 persistent codes, shared3/private3 online reads, item gates,
corrected behavior writes and composable anchor-code lifecycle. The rank8 and
support-flag candidate below is historical, not the active default.
Quality exploration, strong baselines, service replay and cross-condition
preparation may proceed in parallel. 80% recovery is a historical reference,
not a pass/fail requirement; retain the compact state and approximately20%
complete added-cost design target and report the quality/cost frontier.
New small development implementations and probes are authorized. Record
concrete settings before expensive runs; do not restart completed experiments
or launch old configurations merely because they are retained. No foundation
model retraining is part of the initial work. Current workload roles, task
separation and complete cost rules are in the active plan. Later dated user
instructions take precedence over the archived settings below.
The user's latest clarification on 2026-10-19 restores the original real
like/dislike feedback task as primary, matching Motivation. Next-item results
remain separate historical/exploratory evidence; do not reuse their quality,
request counts or cost/time projections as feedback results. Existing next-item
run configurations are not the default continuation queue. This supersedes the
earlier same-day decision to prioritize next-item. Small-panel timing, including
4096 users, is diagnostic rather than a final deployment-scale verdict. First
derive fixed/shared versus per-user costs and the scale break-even; then test
whether implementation converts that saving into complete preparation/service
time. Common foundation training must be distinguished from correction fitting.
Sample multiple model chains, version edges and pre-release history strata;
unadapted H10/RecFlow cases are not evidence of correction-quality transfer.
The user clarified that every foundation-model group requires its own correction
calibration and fitting users; cross-model correction transfer is not a goal.
Experiments must support the paper's central quality/cost claim or determine its
final settings. Prioritize the actual 30k Medium lifecycle time/cost accounting
and fitting-population choice; broader inventories and numerical probes are
supporting evidence, not reasons to expand the experiment matrix on their own.

## Research workflow — applies to code and experiments

EvoKV is a paper research repository. Optimize for fast, credible idea
validation and simple, readable implementations. Production readiness,
exhaustive edge-case handling and test coverage are not project goals.

- Start with the smallest implementation and experiment that can answer the
  current research question. Reuse existing code and inspect the signal before
  building a general framework or scaling up.
- Implement the inputs and execution modes the experiment actually uses.
  Add boundary checks only for a concrete risk to numerical correctness,
  evidence, data causality or expensive execution. Do not build speculative
  compatibility layers, generic validators, fallback stacks or exhaustive
  configuration support.
- Defer serialization, concurrent lifecycle handling, fault recovery and
  deployment machinery until the experiment or paper claim needs them.
  A synchronous in-memory prototype is a valid first implementation.
- Keep exploratory configuration and notes lightweight. Reuse an existing
  experiment description; do not create a new contract, audit document or
  approval step for every small code/configuration change. Small development
  probes within an authorized task can proceed directly. Long training and
  formal population runs retain the authorization requirements below.
- Preserve scientific validity: causal data, honest controls and metrics,
  held-out separation, reproducible essential settings and retained evidence.
  These checks serve the research question; they do not require industrial
  hardening or an exhaustive engineering checklist before exploration.

## Archived direction and evidence — 2026-10-05

The user fixed the three-part framework on 2026-10-05. Select clear, technically
justified internal choices; do not change the three problems or add modules just
to improve a development endpoint. A fixed-model mechanism case is useful; a
majority-of-cases or transfer claim needs evidence for its stated scope.

- D1 keeps one native K/V cache and rank8 codes. Head-local native K/V and the
  complete Current item embedding feed a small encoder. Shared Key/Value
  projections adjust scores and the aggregated read in one history scan.
  Native writes remain unchanged; no corrected full K/V copy is retained.
- D2 fits shared parameters before release: local read alignment followed by
  joint fitting through the frozen complete query. Current-teacher outputs
  supply targets; no future request labels or online teacher enter serving.
- D3 binds releases, initializes on the first read, follows appends/evictions,
  and retires codes after reconstruction or all-layer causal clearance.
  Existing per-layer/row inherited-support flags zero codes of cleared rows.
  Current-produced upper-layer rows may still depend on inherited lower state.
  Clearance means independence from inherited state, not equality to Fresh.

The current development candidate is
`fullitem8_local_kv32_joint16_support_lazy`: 84.80659% recovery at 18.43754713%
complete cost, with release recovery 92.86093/69.32950/100.33430%.
Recovery UID95% is [51.635,102.364]%. The paired mean AUC gain over the unmasked
76.13645%/18.42950495% control is 0.00173493, UID95%
[-0.00099248,0.00481246]; V5 regresses. This is not independent confirmation or
proof of stable 80% recovery. Masking reuses original weights and flags, adds
0.016216TF and 337,840,368 transfer bytes, and adds no persistent tensor/refit.

The balanced mean/relative-score objective completed at 72.3082%/18.5421% and
is not selected. Key-auxiliary and Value-capacity branches also failed to improve.
Historical key outcomes remain in final summary/analysis files and
docs/design/plan.md; redundant per-run reports have been retired.
Better teacher MSE does not imply better AUC; the V5 mechanism case found a
large common-offset contribution and concentrated gains, not all-layer synergy.
Do not rerun completed controls, probes, candidates or the old stopped learned
strength run. No model training or population experiment is queued.

## Scientific and resource boundaries

- Quality: fixed4096 users, 51,731 requests, V2→V3→V4→V5 continuous development.
  Equal-release mean AUC differences define recovery; do not average recovery
  percentages or pick favorable subsets/releases. Freeze before confirmation.
- Complete costs: separately fixed30k real event ledger, including shared
  preparation and every service operation. Sum actual FLOPs before taking the
  ratio. Quality/calibration/deployment populations are separate quantities.
- Full/Rebuild: once per user/release at the first request, then normal native
  incremental service. Per-request Fresh is a historical diagnostic. Report
  eager all-resident rebuilding separately, including idle users.
- Keep the20% complete added-cost ceiling and compact state boundary. The
  historical80% target is a reference, not sufficient acceptance. Codes/flags
  occupy12.5%/0.0651% of native K/V. If source-KV reconstruction is required for
  calibration, the selected point costs21.9382%; retain that sensitivity.
- Query work must also be reported against native query/attention itself.
  Isolated kernels/graphs are not loaded throughput or p99. No loaded serving
  advantage or negligible-query claim is established.
- Current fit settings: local32/joint16,256fit/16validation users,16queries,
  rank8,seed17. Retained real histories and native trajectories are causal.
  Calibration roles and previously reserved validation/fit users remain
  excluded from later evaluation; do not tune on old confirmation outcomes.
- Paper Design explains problem–mechanism connections. Concrete numerical
  settings and outcomes belong in Implementation/Evaluation. Paper source and
  current diagrams describe the selected development candidate, with limits.
  Motivation and its measured points remain unchanged. Do not compile the paper
  without a user request.

## Repository cleanup and retained assets

On2026-10-05 the user explicitly authorized deleting branches proven not to
work, completed experiment code/intermediates while keeping important results,
and assets known to be unnecessary. This supersedes earlier blanket retention
instructions for every historical source snapshot, weight and per-user score.

Retain the24 selected foundation models, current original/processed datasets,
latest method's original calibration and local checkpoints, frozen calibration
and quality panels, matched merged controls, and primary Motivation evidence.
Keep historical key summaries, settings, costs, confidence intervals and negative
results at original paths. Their recorded hashes/paths describe past execution;
files outside the retained asset list may now be absent. Do not promise complete
historical reruns or manufacture replacement raw evidence.

On2026-10-07 the user requested a narrower result inventory. Result JSONs were
reduced from12,036 to411, including199 summary/analysis/report files instead
of1006. Retain final branch aggregates, unique adverse outcomes and essential
current dependencies; do not restore redundant shard, per-user, epoch, canary,
pilot, status or source-manifest records merely for historical completeness.
Keep frozen user-role membership even when its experiment has ended. The
remaining JSON contents, weights and scores were not rewritten. Current panel,
calibration, matched-control, cost and90-point Motivation reads passed.

Later on2026-10-07 the user also authorized removing obsolete raw score/request
Parquet files and exit/status text. Result Parquet files dropped from5868 to98;
5770 Parquet and296 .exit/.txt files were deleted. The98 retained files are90
Motivation Q/H merged scores,7 current calibration/quality/control inputs and
one unique historical Max20k-user membership list. Full/Reuse now retains all15
final edge summaries instead of raw score/request shards. Historical raw paths
are provenance only; do not promise per-request reanalysis or resume from them.
Current datasets, weights, histories, JSON conclusions and retained scores were
not rewritten; current reads passed. This stage left785 result files; the later
source/doc cleanup below reduced that count to680.

Retired experiment-specific code/tests, failed candidate weights, old history
packs/raw outputs, per-user checkpoints, logs, repeated source copies and old
previews have been removed. Shared older modules remain only where a retained
entry actually imports them. Capture/signature helpers were extracted without
changing their numerical logic; live source hashes can differ from original
execution snapshots after cleanup. Original results and frozen weights were not
rewritten.

The2026-10-07 source/doc pass removed30 retired code files,3 launchers,5
branch-specific test files,22 obsolete/duplicate documents,85 execution-source
copies and107 Python caches. The repository now contains1342 regular files
excluding .git, of which680 are results. Legacy calibration CLIs were trimmed
to the shared functions still used by current dependencies; retained function
ASTs are unchanged. Lifecycle tests now exercise the current full-item KV
adapter. CPU checks passed176 tests with7 CUDA skips; all411 result JSON hashes,
current inputs, weights and retained scores remain unchanged. Historical source
hashes describe past execution; they do not imply exact source copies remain.

Read README.md, scripts/README.md and results/README.md for current navigation.
Details and retained history are in docs/design/plan.md and
 docs/design/exploration_history.md. Cleanup does not authorize new training,
change sample roles, or make completed experiments a launch queue.
