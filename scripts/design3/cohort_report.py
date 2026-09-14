"""Render the fixed-ready-work probe; no model execution."""
import json, statistics
from pathlib import Path

root=Path(__file__).resolve().parents[2]
out=root/'results/design3/cohort_01'
s=json.loads((out/'summary.json').read_text());rows=s['rows']
aggregate=[]
for batch in (1,4,8):
    for mix in ('homogeneous','mixed'):
        values={v:statistics.median(r['median_ms'] for r in rows if r['batch']==batch and r['mix']==mix and r['variant']==v) for v in ('individual','path_group','operator_group')}
        aggregate.append(dict(batch=batch,mix=mix,**values,reduction_vs_path=1-values['operator_group']/values['path_group']))
errors=dict(logit_absolute=max(r['score_error'] for r in rows),geometry_absolute=max(r['geometry_error'] for r in rows),geometry_relative=max(r['geometry_relative_error'] for r in rows),calibrated_absolute=max(abs(d['actual']-d['reference']) for r in rows for d in r['decisions']),decision_comparisons=sum(len(r['decisions']) for r in rows))
result=dict(aggregate=aggregate,errors=errors,setup_seconds=sum(r['setup_seconds'] for r in rows),max_plan_workspace_bytes=max(r['workspace_bytes'] for r in rows))
(out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
text=['# Design 3：状态独立、算子共享','',
'当前初稿：从完整用户路径分组，改为共同模型读取合并、修正和筛查按行选择。它接收已经确定要执行的任务，不预测请求到达，不改变Design 2动作。', '',
'## 问题与机制', '',
'Design 1/2让同一目标模型下出现native、修正、修正加筛查三条路径。状态差异必要，但按完整路径分批会重复发起共同模型主干。本轮在层内先共享投影和各自历史读取，再对修正行计算delta并scatter，所有行进入下一层。筛查只消费筛查行的实际native响应，原225维三角求解、长度、mask、校准和阈值不变。没有跨用户注意力，也不对native行执行被丢弃的修正。', '',
'## 固定工作包验证', '',
'冻结六层M1/M3/M4/M5，每目标8个已保存真实快照，取每个面板首个实际候选。快照不是32个独立UID；执行模式按预定循环0/1/2分配，不声称为自然策略混合或真实请求同时到达。完整历史1024，任务数1/4/8。没有新教师、拟合、标签决策或确认集读取。', '',
'三种实现均使用同一份FP32参数、原三角检测和CUDA Graph；逐任务对照也在单图内执行所有任务，收益不是少几次Python图调用。每次输入复制、device gather、scatter和实际读取全部计时；固定分组与图构建另计。输入GPU驻留；不含摘要/view生成、CPU动态分组、目标构建、真实写入和排队。', '',
'同质包全部筛查；混合4任务为native/修正/筛查/native，8任务为三类循环。每目标5组×3次同步计时，表为四目标中位数。所有配置保留，没有按结果筛选目标或规模。', '',
'|任务数|组合|逐任务ms|完整路径分组ms|算子共享ms|相对完整路径减少|','|---:|---|---:|---:|---:|---:|']
for r in aggregate:text.append(f"|{r['batch']}|{r['mix']}|{r['individual']:.3f}|{r['path_group']:.3f}|{r['operator_group']:.3f}|{100*r['reduction_vs_path']:.2f}%|")
text += ['',
'混合路径时，共同模型层批次数从18降为6（六层×三路径→六层×一组），不是把算术总量降到1/3。4/8任务相对完整路径分组分别减少约36.3%/25.1%的读工作包时间。同质包两者计算组织相同，没有额外收益；单任务也无收益。批次更大不保证增量收益更大。', '',
'## 正确性、费用与证据边界', '',
f"独立原逐状态reader为数值参考。logit最大绝对差{errors['logit_absolute']:.3g}；几何最大绝对差{errors['geometry_absolute']:.6g}，最大相对差{errors['geometry_relative']:.3g}；校准后最大差{errors['calibrated_absolute']:.3g}。{errors['decision_comparisons']}个重复配置内筛查比较全部保持冻结阈值判定，不作为独立样本数。所有原输入逐值未修改。浮点批量运算不承诺逐位一致或任意阈值邻域保证。",
f"72个目标/规模/组合/实现配置的图准备合计{result['setup_seconds']:.3f}秒，单计划最大CUDA allocated增量{result['max_plan_workspace_bytes']/2**20:.2f}MiB；这是工作缓冲/图相关增量，不是完整进程峰值。发布模型与因子准备为共享既有依赖，没有计入稳态表。没有减少原检测矩阵FLOPs，新增gather/scatter和输出存储；不据计时改写D1/D2算法费用或人口外推。", '',
'当前实现覆盖共同读取主干、选择性修正、选择性证据与结果归属。尚未接入完整连续状态回放，旧binding的72UID逐位一致结果不能代证本版本；不宣称新质量收益、人口吞吐或到达延迟。目标构建/候选安装仍归D2，本轮没有把第二缓存或构建工作伪装成已测共享路径。', '',
'## 初稿判断与后续', '',
'路线通过初步可行性：在明确的路径混合条件下，比完整路径批处理有额外收益，同时同质与单任务对照符合机制预期。论文中心改为“状态需要隔离，但共同模型计算无需随完整路径一起分裂”。下一步才扩展真实就绪组接入、不同历史长度，以及请求边界状态动作的连续验证；不需要回到阈值/AUC搜索。', '',
'复现：`PYTHONPATH=src:scripts python scripts/design3/cohort_probe.py`；`python scripts/design3/cohort_report.py`。探针约22秒（当前A40），运行日志和原始计时保留。初次数值探针保留为initial_numerical.json；正式报告使用补齐阈值核对后的summary.json。源代码与输入SHA在summary中。此前initial_01、binding_01所有正负结果保留。']
(out/'report.md').write_text('\n'.join(text)+'\n')
print(json.dumps(result,indent=2))
