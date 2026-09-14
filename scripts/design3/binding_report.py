"""Final state-binding evidence and its negative controls; no model execution."""
import json,hashlib
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];out=ROOT/'results/design3/binding_01';old=ROOT/'results/design3/initial_01/reference72'
data=json.loads((out/'device_controls.json').read_text());d=pd.DataFrame(data['rows']);med=d.groupby('variant').median(numeric_only=True).median_ms.to_dict()
baseline=pd.read_parquet(old/'quality_raw.parquet')
verified=[]
for name in ('final72','timed_prior72','timed_binding72'):
    p=out/name
    pd.testing.assert_frame_equal(baseline,pd.read_parquet(p/'quality_raw.parquet'),check_exact=True)
    for f in ('witness_reads.parquet','candidate_lifetimes.parquet','decisions.parquet','costs.parquet'):
        pd.testing.assert_frame_equal(pd.read_parquet(old/f),pd.read_parquet(p/f),rtol=1e-8,atol=1e-8)
    assert json.loads((old/'state_counts.json').read_text())==json.loads((p/'state_counts.json').read_text())
    verified.append(name)
stats=pd.DataFrame(json.loads((out/'timed_binding72/execution.json').read_text())['stats']).fillna(0).sum(numeric_only=True).to_dict()
prior=pd.DataFrame(json.loads((out/'timed_prior72/execution.json').read_text())['stats']).fillna(0).sum(numeric_only=True).to_dict()
total=lambda s:s.get('timed_read_seconds',0)+s.get('timed_screen_seconds',0)
prior_time,new_time=total(prior),total(stats)
q=stats['bound_screen_queries'];states=stats['binding_evidence_updates']
# Leading detector matrix arithmetic, with FMA=2. Detailed scalar normalizations
# are common or separately observable; this is not a replacement D2 total ledger.
matrix=dict(original_solve=36*225**2*q,prepared_response=36*2*192**2*q,
    source_preparation=36*2*(33**2+192*33)*states,inverse_and_residual_check=4*3*36*225**3)
summary=dict(status='complete',users=72,rows=len(baseline),verified=verified,scores_bitwise=True,actions_and_state_records_match=True,
    median_ms=med,max_geometry_error=float(d.geometry_error.max()),runtime=stats,prior_runtime=prior,
    timed_region_seconds=dict(prior=prior_time,binding=new_time,speedup=prior_time/new_time),
    kv_copy_reduction=1-stats['binding_kv_bytes']/stats['binding_naive_kv_bytes'],
    geometry_matrix_flops=matrix,preparation=data['preparation'],
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('scripts/design3','src/hstu_kvcache/design3') for p in (ROOT/folder).glob('*.py')})
(out/'summary.json').write_text(json.dumps(summary,indent=2))
text=f'''# Design 3第二版：按依赖有效期维护执行绑定

日期：2026-09-13。有限开发验证完成；当前论文主方案从“图执行＋批量求解＋调度”收紧为**发布准备固定变换，状态变化时更新依赖结果，请求消费有效绑定**。不再把合作式调度作为中心模块。

## 核心问题确实存在在哪里

原图版本在每次请求中仍转换相同的FP64读取参数、缩放相同状态的view、根据固定Cholesky因子执行逐query三角求解，并为各模式/形状复制输入。原方法已经只做一次Cholesky，不能把它描述成每query重新分解矩阵；普通读取也已经共享一次KV扫描。

本次72UID真实D2服务路径有{int(stats['bound_calls'])}次常见形状绑定调用。按原程序张量操作数计，每次重新转换会产生合计{int(stats['avoided_release_cast_elements'])}个FP32参数元素的重复转换输出；新版本只按四个目标准备{int(stats['release_parameter_elements'])}个。这里是操作/张量字节账本，不是硬件HBM流量测量。参数一次转换本身已有类似D1实现，不能单独包装成创新；匹配对照保留它。

大多数状态确实在请求之间变化：{int(stats['binding_kv_updates'])}次KV更新、{int(stats['binding_kv_hits'])}次命中；源项{int(stats['binding_evidence_updates'])}次更新、{int(stats.get('binding_evidence_hits',0))}次命中。原全量绑定需{stats['binding_naive_kv_bytes']/1e9:.3f}GB，实际复制{stats['binding_kv_bytes']/1e9:.3f}GB，仅减少{summary['kv_copy_reduction']:.2%}。因此本负载不支持“多数KV搬运都被消除”，主要重复工作位于固定算子与参数的请求侧执行，完整状态复用是有限的附加收益。

## 解法：一个绑定，三种依赖

- 发布依赖：保持原FP32读取参数；在FP64准备选定225维检测的逆作用，将L逆分为U/V/W。36个层/head独立因子全保留，不降秩、不重拟合。
- 状态依赖：生成`bc=b*count`、`ac=A*count`、active count，并保留`h=Vx,d=||Ux||²`。依赖未变化则继续用；变化才执行对应准备图。KV、view和源项分开判定。
- 请求依赖：实际query/native响应进入已有修正；检测只算`d+||Wo+h||²`。一个驻留绑定供常见形状与模式图共同消费，其他形状仍直接运行原公式。近筛查阈值用原求解复核，在线不需要标签或Fresh教师。

这是维护执行表示，不是再次修改用户模型状态。原始KV仍由Design1/2管理；缓存写入、淘汰、接管、版本变化都会经真实输入变化进入绑定失效判断。实现用保留引用的张量identity与mutation version保守判断，不根据数值碰巧接近复用结果。

## 执行结果与匹配对照

31个已保存的实际1/2候选快照，四目标，七组×四次同步计时。结果为完整六层读取区域，不只计几何；dirty每次强制执行完整KV/状态绑定，reused为状态不变的敏感性，不冒充人口均值。

|实现|评分＋筛查中位数ms|
|---|---:|
|上一版批量求解图|{med['prior_graph']:.3f}|
|只把参数转换移到发布|{med['hoist_cast_only']:.3f}|
|准备逆作用，但每次仍准备所有状态项|{med['release_only']:.3f}|
|新绑定：每次状态都变化|{med['bound_dirty']:.3f}|
|新绑定：状态保持不变|{med['bound_reused']:.3f}|

普通修正读取（不含筛查）为{med['ordinary_prior']:.3f}→{med['ordinary_dirty']:.3f}ms；状态不变时{med['ordinary_reused']:.3f}ms。所有评分逐值一致，检测最大数值差{summary['max_geometry_error']:.3g}。

发布准备贡献了主要收益，单纯转换参数不足以解释它。逐依赖维护比“每次全部准备”的实现，在全dirty条件下有小额管理开销；不能说每个机制在每个条件下都更快。实测低命中率也不支持把unchanged结果作为主数值。

进一步在同一GPU顺序运行同一72UID轨迹，基线和新版本都把独立Design1对照排除在D2绑定之外。D2适配/筛查执行区间包括首次因子准备、图准备、绑定、实际读取及几何，合计{prior_time:.3f}→{new_time:.3f}s（{prior_time/new_time:.2f}倍）。不含摘要源编码、原始事件回放、目标构建及其他评估分支，不能称整个生命周期加速。全部{len(baseline)}评分与reference逐值一致，观察/安装、额度、状态记录一致。

## 成本、准备与真正的取舍

GPU更适合这里的矩阵乘法，但密集逆作用并不减少所有FLOPs。该轨迹中被转换路径的检测矩阵主项为原三角求解{matrix['original_solve']/1e9:.3f}GF，新响应作用{matrix['prepared_response']/1e9:.3f}GF，加源准备{matrix['source_preparation']/1e9:.3f}GF；四目标逆作用准备及L乘逆残差核验约{matrix['inverse_and_residual_check']/1e9:.3f}GF。标量归一化、范数和控制仍需收费，以上只给矩阵主项，不能替换完整D2账本。

每目标逆作用块和FP32读参数约13.65MB，原H及因子仍供准备/复核。实际逆作用准备{stats['inverse_prepare_seconds']:.4f}s，读取图准备{stats['graph_setup_seconds']:.4f}s，状态准备图首次准备{stats.get('binding_preparation_graph_setup_seconds',0):.4f}s。原因子加载/准备另记{stats['factor_prepare_seconds']:.4f}s。图/模型/原始cache、临时workspace不能从这些数中消失；本轮没有人口峰值显存主张。

冻结原D2逻辑费用与观察额度是为了保持同一策略。新增加的执行算术另外报告，不拿速度倍率改写D1/D2旧FLOPs比例，也不把准备成本均摊为零。这里的收益是更少的重复表示准备和更高效的等价算子执行，代价包含更多的密集几何算术与共享存储。

## 保留与停止

前一版图执行、批量求解、源/响应分块和合作式构建证据仍在initial_01。本轮早期让独立Design1对照与D2共用一个驻留绑定，会人为挤掉有效状态；最终运行隔离评估控制，旧canary/replay72仅作实现检查，不使用其命中率作为服务负载结果。有限实验已足以支持当前纸面中心和初版实现；不承诺全负载AUC新收益、大量KV免复制、最终吞吐或调度保证。

复现：`binding_benchmark.py`、`replay.py --execution binding/graph_service --timed-execution`、`binding_report.py`。仅使用此前开放的72UID和真实输入，不改C/H/分组/阈值，不运行训练，不读取确认集。原始结果、参数、源hash和日志均保留。
'''
(out/'report.md').write_text(text)
print(text[:1500])
