# 六层 Design 探索

`freeze_expanded_split.py`冻结按初始历史分层的6000开发/6000独立确认用户；只读人口与
历史拟合UID元数据。原split不变，新split后续用于拟合/开发隔离。
`select_release.py`在发布前独立校准用户上拟合各层修正强度，再自动选择稳定或source×time
翻译；另一校准UID折检验实际修正forward，计入两候选拟合及选择成本。
`run.py --quality-only --split-path <split> --dev-offset <offset>`复用连续五段执行器，
按互斥UID块评价冻结译者；每个用户的C1–C5持续状态保留，合并原请求重算全体AUC。
`evaluate_cohort.py --reference <run> --gpus <ids> --estimate-seconds <seconds>`调度同一
冻结开发集合的完整UID块并合并；大于30分钟须由tmux启动并传`--detached`。
`--no-op-targets 2`用于用户指定的四边优化范围：M2仍发布模型并保留真实原生写入和摘要，
但不拟合/加载Translator或安装读取修正；M1/M3/M4/M5继续适配。旧M2结果保留，
原生No-op阶段在完整轨迹中单列，不能跳过M2模型来构造M1→M3。
`compare_cohorts.py --control <run> --adaptive <run>`只读取两组已完成的相同用户结果，
核对请求和两基线，报告每个M及历史长度分层，不对分块AUC取平均。
`run.py --split-path <split> --stratified-calibration`从剩余校准池按初始历史覆盖人口，
原开发panel仅允许原有UID顺序；短历史缺少合法早期快照时保留切换点场景。
`profile_release_quality.py`在独立校准用户的发布前14天真实反馈上提取Parent/Current
Full AUC、概率偏向和更新幅度；不读取发布后标签，不改变模型准入，不代表persistent-Reuse质量。
`probe_release_feedback.py`复用这些发布前请求，在Parent-Exact源状态上比较冻结翻译设计；
`diagnose_query_dispersion.py`在校准状态分解候选差异与条件均值误差，均不是连续质量结论。

当前完整研究原型与结果解释见 [计划](../../docs/design/plan.md)及
[迭代记录](../../docs/design/iterations.md)。冻结 Medium 实际使用 legacy ELU+1，
不是 SiLU/max_seq_len；执行器检查真实配置，不改变 backbone。

对外记号统一为方法“方案N”、模型M0–M5、实际缓存服务阶段C0–C5。五次更新是
M0→M1→…→M5，C_i可能混有多个模型的K/V。既有命令、run_id和文件路径中的v编号
保持不变以便复现；旧正文小写v为方法、大写V为模型，后续报告表使用M。

在仓库根目录运行（Python 环境需已有 torch、numpy、pandas、pyarrow、duckdb、PyYAML）：

```bash
PYTHONPATH=src:scripts CUDA_VISIBLE_DEVICES=0 python -u scripts/design/run.py \
  --run-id example_v0 --fit-users 32 --dev-users 8 --trajectory-users 2 \
  --steps 200 --targets 2 --tail-days 1
```

`--score-weight 1` 是校准用户输出蒸馏对照；`--context local` 是每segment独立翻译和
局部刷新变体。其余初值、对应已跑命令和正负结果记录在各运行configuration及迭代文档。
这些变体均未通过方法有效性验收，不能作为正式方法结论。
`--representation functional --summary-weight 0 --score-weight 1` 是当前v2功能view探索，
以1024事件上限×1slot、发布封段的writer提供实际source均值和producer统计，学习每层常量基
响应变化。它保留普通cache、paired read和真实生命周期，并未被验证有效。
`--representation ridge --summary-weight 0 --steps 3 --rank 32` 是后续共享闭式译者候选；
此模式的steps表示以实际修正query重新拟合的pass数，按producer输入统计预测per-old-event响应率。
它同样尚未被验证有效；参数与结果以各run configuration为准。
`--adjacent-calibration` 为后续目标补入独立Parent-Exact源缓存校准场景，原有实际连续场景保留。
`--score-steps N` 在岭回归后固定source投影，用校准用户的闭环logit误差微调共享decoder；
保留0.1权重的响应约束，最后折叠回同一映射，不增加推理网络。实测无稳定收益，当前不采用。
`--write-mode reuse` 是写入token诊断触发的v3对照：真实事件使用原模型读取路径，CC候选
保留paired correction；writer仍记录实际K/V，旧段淘汰标dirty，下一次CC读取前刷新。
它避免把CC训练的常量响应用于不同的真实write token；默认`shared`保留v0–v2复现行为。
`--history-scope all --lifetime-users 32`是v4：纳入实际Current后代摘要，拟合整个保留历史
的响应变化，并在固定前32名calibration用户上补充发布前真实长回放和Current-Exact零控制。
实际写入数也是输入；任何写入使view失效，下一次CC前刷新。仍只支持ridge与native写入，
新增teacher/source/回放费用单列。该变体已完成开发评价，仍未达标；没有确认或有效性结论。
`--temporal`是v5：沿用全历史统计，将view扩为常量项与原模型32维sin/cos时间基系数，
系数由实际producer占比条件化。读取按真实query delta加权，校准增加同一可见前缀上的
四组因果时间查询；时间变化本身不触发重翻译。该变体尚在开发检验。
`--layer-clearance`是v6结构对照：全历史native写入下，第j层（从0计）在(j+1)×1024
次实际写入后关闭常量和时间修正；6144次写入后跳过当前view刷新，writer继续维护，
下一发布重置年龄。四条真实缓存已验证依赖消退上界；该读取策略尚在开发验证。
可先与`--evaluation-from`合用，检查同一组冻结权重；不是以标签或旧producer数调度。
`--temporal-source-rank 8`是v7用户条件化时间项：从calibration source的标准化统计
提取8个白化PCA坐标，与producer占比共同生成同一32维时间基的系数；PCA不使用目标
响应或标签。共享ridge仍为rank32，source/view维护依赖不变，额外投影与校准成本照计。
`--source-confidence`是方案10小变体：按本次发布拟合源输入标准化范数的95分位，
在超出该范围时以`min(1,sqrt(q95/norm²))`同时减弱常量与时间项；不按模型编号分支，
不读评价标签，不随查询时间改变。范数阈值存在该译者权重中，额外刷新费用照计。
`probe_source_confidence.py`在独立校准的真实连续/adjacent/lifetime场景检验它，另保留
同一新权重关闭缩减的对照；小样本teacher距离不代替固定6000人的实际质量。
该探针的`--candidate-reference <run>`也可比较同64名拟合用户的其他冻结译者。
`--observed-queries`让两组冻结译者共享发布前真实CC item/gap探针；不能与默认探针
跨任务比较误差绝对值。`fit_observed_queries.py --run-id <new> --users 64 --targets 5`
是方案13：保持16候选×4时间组的预算，从拟合用户发布前14天真实请求取item和gap
四分位，不加载label列；无请求用户共享拟合组的真实请求池，query时间裁到各场景
可见prefix的合法范围。`calibration_query_selection`包含这一步IO与构造，进入必要准备。
`fit_adaptive_ridge.py --run-id <new> --users 64 --targets 5`是方案11：按完整UID删除
的解析响应残差，在预先固定的五个正则倍率中选择，再执行原rank32压缩。每次发布、
每轮修正query分别选择，数据/教师/真实谱系/保存全复用run.py；没有新增推理参数或
逐模型编号分支。它是固定本轮query、源标准化及响应尺度下的未压缩ridge代理，
不等于最终低秩闭环的独立质量；额外求解时间计入拟合。公式用
`PYTHONPATH=src:scripts pytest -q tests/test_design_ridge.py`对照显式按用户删组重拟合。
`--second-moments`是方案12的摘要信息对照：writer随实际追加/淘汰维护K²/V²平方和，
Translator除原均值外读取按producer聚合的中心对角方差。实际K/V、事件归属、时间
修正与发布顺序不变；新增存储、增删和拟合成本照计，不把统计量增加称为质量提升。
`diagnose_query_dispersion.py --independent --users 128`仅看既定独立校准的查询表示与
history-head几何；`probe_inactive_response.py`保留屏蔽零响应head修正的负对照，
该临时reader未安装到方法主执行器。
`probe_native_response.py --users 64`在原拟合用户上估计逐head的native响应残差斜率；
`--users 128 --coefficients <fit_run>`在独立校准用户上检查真实六层forward。包装仅用于
这个诊断，保留原self、summary/time修正与依赖消退mask；关闭包装必须逐元素匹配原reader。
固定译者上后加36个系数的五段误差均略升，未安装到人口方法。
`fit_source_kernel.py --run-id <new> --users 64 --targets 5` / `--source-kernel`是方案14：
用本次发布拟合source状态之间的Gaussian相似度替换线性source映射，距离尺度来自
校准状态距离中位数；时间块仍为按维度归一的producer×Fourier线性项。三轮实际query、
rank32、native写入、依赖消退及source维护不变，校准support和kernel刷新费用照计。
`tests/test_design_kernel.py`对照显式自由截距方程及发布后时间分解；该方案6000人未改善主要短板。
`diagnose_time_view.py`与`diagnose_query_view.py`只拟合逐场景教师view，分别检查广时间基
及实际query仿射表示；不把它们当作共享方法或真实质量。`fit_passes.py`保留三轮与六轮
共享拟合的收敛控制。`fit_query_view.py --run-id <new> --users 64 --targets 5`是方案15：
从原摘要预测逐层逐head的`b + Q A`，实际query坐标标准化在发布时折回raw-Q系数。
校准按层推进，已拟合的下层固定，下一层使用实际修正query；每层共享rank32源映射，
每场景64个合法item/time配对。原生追加与真实摘要增删不变，query矩阵及更大的共享
映射计入成本；当前只实现eager读取。`tests/test_design_query_view.py`检查坐标折叠、
count权重、标量/批量发布和逐层消退。教师表示信号不等于共享拟合有效。
`--calibration-batch 16`按真实cache长度分组，共享native写入状态和特征；每pass仍重算
修正后的query。标量与批量canary最大logit差小于1e−6。ridge不计算从未用于其loss的
teacher logits；发布前短尾回放复用已验证native band算子。这些不改变监督或用户划分。
后续实现把lifetime source/teacher长尾也按真实长度批量回放，保留各自writer与全部成本；
实际六层canary检查其K/V和事件归属。整数metadata留在CPU，K/V sums仍为FP32；
dirty CC刷新复用writer批量特征，所有层的时间基响应合并计算，数学与权重格式不变。
最新准备路径把原teacher/Parent/初始历史按实际长度批量构建，同UID同时间的不可变teacher
在一次校准中复用；校准谱系按原时间边界批量回放。空事件尾段保持状态不变。
`--compile-reads`在冻结v5权重的评价中编译相同native/paired CC计算，已核对512人14164请求，
方法五段AUC不变，最大logit差4.0e−6；五段服务均未新增Dynamo图。完整成本仍未合格。
需要本机已有的C++20编译器；当前命令使用`CXX=/usr/bin/x86_64-linux-gnu-g++-13`
及`CC=/usr/bin/x86_64-linux-gnu-gcc-13`。图输出先复制再保留为视图，各目标单列
编译/热身时间与数据独立性检查；原生和方法都编译，TF32仍关闭。暂不据单算子探针
声称完整成本达标。
真实候选固定分入1/2/4/8/16桶，大组按16分块；各候选独立，padding重复第一候选并
丢弃其输出，所有三分支同样处理。候选输入拥有连续存储，避免NumPy/slice的stride及
base形状触发额外编译。`compiled_service_v*.json`记录每段服务中新Dynamo图的数量。

执行器先保存配置、UID划分hash、模型及admission引用和源码快照，取数后更新资源估计，
通过真实六层代数/梯度/追加canary后短校准，再运行独立用户的单边比较与持续轨迹。
单边参考独立初始化，连续分支只在Exact分支发布时重建；learned/Reuse保留自己实际状态。
同timestamp全部请求先评分，再写入真实事件。校准只使用每个目标发布前的历史。
`--evaluation-from <completed_run_id>` 可扩大同一组冻结Translator的开发评价：加载原运行的
表示和写入模式，跳过拟合，记录原配置与权重hash，教师/拟合成本仍引用原运行，不能记作零。
例如传入`--dev-users 128 --trajectory-users 128 --targets 2 --tail-days 14`评价两段完整真实轨迹；
先依事件数估时，超过30分钟仍使用带资源记录和退出码的tmux。
`--replay-chunk 128` 仅用于native写入：在两次真实CC请求之间按严格1024窗口mask批量回放，
每个query均排除已淘汰及未来事件，全部分支共用已有native block。所有同timestamp请求
仍先评分再纳入该组写入；跨请求边界前先flush真实历史。此为回放吞吐优化，不代表在线
单请求延迟加速。默认逐事件路径保留，128人1144条真实请求已逐项核对两种路径。
`--release-batch 16`将同一writer统计/低秩译者的发布计算和view安装批量执行；Exact分支
也按实际prefix长度用相同batch重建，默认1保留原路径。普通K/V和producer不被修改，
安装后的paired差及时间系数允许融合保存，下一次source变化仍触发普通刷新；
常量路径已在56条真实请求逐项核对，方法/Reuse预测相同，Exact最大差2.38e−7。

`data.py` 复用当前模型、history、item mapping及request primitives；新划分只使用历史计数
和既有selector次序，排除原3000 Insight cohort并先预留2048独立确认用户。这里的runner仅
使用calibration与development，尚不提供正式确认入口。执行器支持五个目标：V1–V4引用原
Full-only诊断准入seal；V5单独校验其E14_partial Full-only adjudication及hash，保留
`serving_admission=false`，不把不完整尾段变成完整E14或serving admission。V5反馈读取原
extension manifest以保留实际day-300尾段。五条边都是本次Design的诊断序列；五目标canary
已通过，v3/v4/v5的512人五边轨迹已完成；v5继续在同一用户集合验证完整成本。

输出在 `results/design/<run_id>/`；目录拒绝覆盖。`configuration.json`、`summary.json`、
`resource_estimate.json`、`canary.pass.json` 是紧凑证据。局部raw、译者权重、源码tar和patch
留本地；每条完成的边保留cutover/logit/loss/真实反馈/连续切换记录，失败不删除已有结果。
timing是同步实际Python/GPU路径的墙钟，包含该路径CPU发射开销，不等同于理论FLOPs。
教师构建、校准、谱系生成、初次backfill、发布转换、持续读写分项报告。

只汇总已有证据可运行 `python scripts/design/report.py <run_id> ...`，输出
`results/design/analysis/` 的对照CSV与说明；不加载模型。保留所有列出的正负变体。
`PYTHONPATH=src python scripts/design/quality_report.py <run_id>`汇总实际反馈，报告全部请求的
绝对指标、按旧状态量分层、固定seed17的1000次配对UID bootstrap及原校准成本引用。
该区间只描述一个backbone seed内的用户抽样不确定性；AUC恢复仅在正gap>0.0001时显示。
`diagnose_lifetime.py --run-id <new_id> --reference <completed_run_id>`复用已冻结译者，
仅用于v2/v3旧段修正，在前8名开发用户的V1真实生命周期上分解旧状态及current后代误差；精确K/V替换和共享
响应oracle仅为诊断，不进入持久状态或可执行方法。
`population_history_cost.py`只扫描固定30000人口的UID/时间计数，生成真实cutover历史长度
分布；包括预留用户的计数，但不读取其模型输出或质量。`benchmark_release.py`在32名
固定开发用户的真实混合producer状态上比较实际view安装和相同batch的Exact-All kernel：
`benchmark_release.py --run-id <new_id> --reference <completed_run_id>`。另按固定30000人口
的实际历史长度，以32事件桶和batch16做显式计算外推，保留全局校准/教师/回放成本。
`--compile-exact`给Exact使用相同编译kernel，先核对全部测量形状的K/V，启动费用单列。
使用V5形状计时近似同架构各版本；初次摘要backfill另外实测并作全窗口人口外推；
人口I/O、实际人口执行和持续服务开销尚未测量，不可直接作20%资格结论。
`profile_calibration.py --run-id <new_id>`分解固定V1校准的响应生成、ridge/SVD和准备费用；
`--compile-prefixes`与`--compile-response`只作相同公式的编译诊断，默认完整校准仍用eager。
`diagnose_clearance.py --run-id <new_id>`在已开放calibration中按历史计数选前四个长历史，
比较同前缀Parent/Current初始化后6144个真实native写入的逐层K/V；不读取评价标签或确认。

用户已授权该六层Design范围的持续探索。预计超过30分钟的作业仍先有前瞻配置、资源估计和
通过的canary，使用detached tmux并保留退出码；`--detached`只记录执行方式，不会自动创建
tmux，也不代替这些要求。短作业直接监控。禁止因此重跑历史封存诊断或开启未授权backbone。

必要验证：`PYTHONPATH=src:scripts pytest -q tests/test_adaptation.py`；真实六层梯度/追加canary在
运行入口，不需要另跑整个模型测试套件。
统计汇总改动使用`tests/test_design_quality.py`。计数式bootstrap预先固定概率并列组，
与原显式重复UID的1000次结果在五个已保留目标上区间完全一致；不更改抽样单位或种子。
