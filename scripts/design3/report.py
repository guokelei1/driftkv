"""Summarize retained execution evidence without running a model."""
import json,hashlib,ast
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'results/design3/initial_01'
timing=json.loads((out/'timing_controls.json').read_text());all_timing=pd.DataFrame(timing['rows'])
d=all_timing[all_timing.q<=2]
median=d.groupby('variant').median(numeric_only=True)['median_ms'].to_dict()
pivot=d.pivot(index='sample',columns='variant',values='median_ms')
reference=pd.read_parquet(out/'reference72/quality_raw.parquet');candidate=pd.read_parquet(out/'batched72/quality_raw.parquet')
pd.testing.assert_frame_equal(reference,candidate,check_exact=True)
for f in ('witness_reads.parquet','candidate_lifetimes.parquet','decisions.parquet','costs.parquet'):
    pd.testing.assert_frame_equal(pd.read_parquet(out/'reference72'/f),pd.read_parquet(out/'batched72'/f),rtol=1e-8,atol=1e-8)
counts=json.loads((out/'reference72/state_counts.json').read_text());other=json.loads((out/'batched72/state_counts.json').read_text())
assert counts==other
runtime=json.loads((out/'batched72/execution.json').read_text());stats=pd.DataFrame(runtime['stats']).fillna(0)
sched=json.loads((out/'scheduling_distinct.json').read_text());s=pd.DataFrame(sched['records'])
s['ordinary_first_ms']=s.ordinary_completion_ms.map(lambda v:v[0]);s['ordinary_last_ms']=s.ordinary_completion_ms.map(lambda v:v[-1])
schedule=s.groupby('policy')[['total_ms','target_completion_ms','ordinary_first_ms','ordinary_last_ms']].median().to_dict('index')
prep=pd.DataFrame(timing['preparation']);prep=prep[prep['shape'].apply(lambda s:isinstance(s,str) and ast.literal_eval(s)[1][-1]<=2)]
graph_setup=prep[prep.variant=='batch_graph'].graph_setup_seconds.sum()
shape_counts=d[d.variant=='batch_graph'].groupby(['target','n','q']).size().reset_index(name='snapshots').to_dict('records')
sources=[p for folder in ('src/hstu_kvcache/design3','scripts/design3') for p in (ROOT/folder).glob('*.py')]
summary=dict(status='complete',policy='batched225 common-shape graphs; direct eager for other shapes; original thresholds and allowance',
    users=72,rows=len(candidate),timed_common_snapshots=int(d['sample'].nunique()),exploratory_snapshots=32,scores_bitwise_equal=True,state_records_equal=True,observation_and_costs_match=True,
    targets=[1,3,4,5],timing_median_ms=median,
    paired_speedup={name:dict(median=float((pivot[name]/pivot.batch_graph).median()),minimum=float((pivot[name]/pivot.batch_graph).min()),maximum=float((pivot[name]/pivot.batch_graph).max())) for name in ('clean_eager','clean_graph','batch_eager','block_graph')},
    maximum_geometry_difference=float(d.geometry_error.max()),snapshot_shapes=shape_counts,
    timing_graph_setup_seconds=float(graph_setup),graph_additional_allocated_bytes=int(prep[prep.variant=='batch_graph'].allocated_bytes.sum()),
    runtime=stats.sum(numeric_only=True).drop('target').to_dict(),scheduling=schedule,
    replay_seconds={name:json.loads((out/name/'summary.json').read_text())['seconds'] for name in ('reference72','graph72_fixed','batched72')},
    invalid_attempt=dict(name='graph72',reason='dynamic-shape four-slot replacement encountered illegal CUDA memory access during M3; incomplete and excluded; root cause not established'),
    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(out/'summary.json').write_text(json.dumps(summary,indent=2))
calls=stats.screen_calls.sum()+stats.read_calls.sum();hit=1-stats.shape_eager_calls.sum()/calls
text=f'''# Design 3初版：按状态和读取组织执行

本轮完成实现、小型执行测量和72UID真实连续回放；没有重新训练、拟合检测器或改变Design 2参数。论文采用的主版本为**发布共享检测布局＋读取后联合执行＋固定常见形状图槽**。目标构建另有保持真实KV的合作式执行原型。

## 结论与实际范围

- 对32个真实筛查快照（M1/M3/M4/M5各8个；不是32个独立用户）核对计算。其中31个属于最终图路径的1/2候选形状，完整六层评分＋筛查主实现中位数{median['batch_graph']:.3f}ms，简洁eager {median['clean_eager']:.3f}ms，同样使用CUDA Graph但保持原逐层求解的对照{median['clean_graph']:.3f}ms。另1个8候选快照的探索计时保留在原数据中，最终回放该形状走直接计算。
- 计时包含每次整KV、view、源编码和请求输入的槽位复制；不含源摘要编码、事件写入、模型加载、初次因子准备及目标重建。它是完整读取区域收益，不是整生命周期加速。
- 72UID、{len(candidate)}评分行逐值一致；状态记录、观察、安装和原额度账本保持。72UID由8个已开放动作canary加自然和富集队列各前32名组成，主要验证执行正确性，不是新增质量确认。
- 合作式构建与原构建的KV逐值一致；一个已准入目标构建与四个其他UID就绪读取的固定工作包中，普通首个完成由{schedule['serial']['ordinary_first_ms']:.3f}降至{schedule['cooperative']['ordinary_first_ms']:.3f}ms，最后一个由{schedule['serial']['ordinary_last_ms']:.3f}降至{schedule['cooperative']['ordinary_last_ms']:.3f}ms。目标完成由{schedule['serial']['target_completion_ms']:.3f}变为{schedule['cooperative']['target_completion_ms']:.3f}ms，总工作包由{schedule['serial']['total_ms']:.3f}变为{schedule['cooperative']['total_ms']:.3f}ms。

## 三个机制与证据

1. **在发布/状态边界准备，在读取时消费。** 因子按目标准备；view继续由原dirty规则生成，目标/形状图槽跨状态复用。共4个槽/目标（普通/筛查×1/2候选，长度1024），其他形状直接执行同一公式。本轮保守地每次复制状态，不用相同状态命中省流量来抬高结果。
2. **读取修正与证据共用输入，并批量组织225维求解。** 保留各层/head独立H及原尺度，将36个因子作为一个批量求解，不保存完整诊断trace。通用图对照同样去掉trace并获得图执行，所以新增收益不能全归因于删除日志或CUDA Graph。
3. **目标工作按合法边界让出执行机会。** 原64事件tile、层投影和层结束作为合作边界，不改变目标动作或计算依赖。触发请求仍等候目标计算，其他用户可就绪读取穿插。只评价固定就绪工作包，尚未构造到达负载或生产p99。

## 保留的机制对照

|六层评分＋筛查实现|31个常见形状快照中位数ms|
|---|---:|
|简洁逐层eager|{median['clean_eager']:.3f}|
|逐层CUDA Graph|{median['clean_graph']:.3f}|
|合并225维求解eager|{median['batch_eager']:.3f}|
|合并225维求解CUDA Graph（主版本）|{median['batch_graph']:.3f}|
|源/响应分块eager|{median['block_eager']:.3f}|
|源/响应分块CUDA Graph|{median['block_graph']:.3f}|
|native读取eager（不做修正或筛查）|{median['native_eager']:.3f}|
|native读取CUDA Graph（同等通用优化）|{median['native_graph']:.3f}|

状态分块虽然公式正确，但完整读取中未稳定优于直接合并求解，故不是初稿必需机制。同状态q=1/2/4/16的分块复用局部测量保留在timing_controls.json，复制query仅测几何执行，不能当新增质量数据。所有实际评分差为0；最大检测数值差{summary['maximum_geometry_difference']:.3g}。

## 准备、驻留和回放账本

- 独立区域测量中，主实现五个常见目标/形状槽的warmup＋capture共{graph_setup:.4f}s，额外allocated bytes总计{summary['graph_additional_allocated_bytes']}；这是跨槽累加，不是同一时刻的GPU进程峰值。原H准备完整保留在timing_controls.json，新增布局没有教师计算。
- 72UID运行实际{int(stats.captures.sum())}次capture，共{stats.graph_setup_seconds.sum():.4f}s；{int(calls)}次适配/筛查调用中，{hit:.2%}进入图路径，{int(stats.shape_eager_calls.sum())}次走直接路径。第一次使用支付capture；其后共享，不为每个UID缓存一份图。
- reference72/batched72全四臂实验器墙钟为{summary['replay_seconds']['reference72']:.2f}/{summary['replay_seconds']['batched72']:.2f}s，包含模型加载、原Exact对照、所有native回放、诊断与IO。单次运行且存在CPU并发，不作为端到端部署加速证明，也不能用读区域倍率外推整个生命周期。
- 冻结Design 2的算法额度收费，用于保持相同观察与状态轨迹；批量225求解和图执行不降低其主阶检测算术。D3首先降低执行开销，不把时间倍率换算成FLOPs降低。

## 失败与修订

最初任意实际形状都进入4槽替换池，8UID通过后，72UID在M3出现CUDA非法内存访问，未完成。该尝试保留日志及配置，不给性能/质量结论；具体根因尚未确定。初稿缩为固定常见形状池，无动态图槽替换，其他形状走原等价计算。修订后分块和主批量版本都完成同一72UID验证。不是忽略不支持用户：全部用户和请求都继续服务并计入比较。

## 初稿可写与下一步边界

当前可写：状态/形状共享执行计划，读证据的批量组织减少读取区域时间，合作式目标构建改善一个固定工作包内其他读取的完成时间，且真实连续轨迹保持。还不能写：生产吞吐/p99保证、全生命周期相对Exact加速、大人口内存可扩展性、消除准备教师成本或所有形状均有相同收益。

## 复现入口

`capture.py`取得原输入；`benchmark.py`测同等通用图对照与状态分块；`replay.py --execution reference/graph`复核同一72UID；`scheduling.py`验证固定就绪工作；`report.py`仅聚合和核对。入口均在scripts/design3。环境为{timing['device']}、PyTorch {timing['torch']}，每进程4CPU线程，TF32关闭。每项实验低于两分钟，没有长训练。原日志、输入、数值和运行配置保留；评价未新增Fresh全请求教师。
'''
(out/'report.md').write_text(text)
print(text[:1800])
