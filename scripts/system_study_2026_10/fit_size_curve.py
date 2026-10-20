#!/usr/bin/env python3
"""Join the two predeclared V4 probes without rerunning completed methods."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from read_correction_2026_09.cost import CostModel
from system_study_2026_10.fit_size_evaluate import analyze

BASE=ROOT/'results/system_study_2026_10/calibration/fit_size_v4'
ORIGINAL=ROOT/'results/query_relay_2026_10/probes/A_itemgate6_rms_roll4'
COUNTS=(16,32,64,128,256)


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    first=BASE/'attempt1/evaluation159_attempt2'
    second=BASE/'extended_attempt1/evaluation159'
    left=pq.read_table(first/'scores.parquet').to_pylist();right=pq.read_table(second/'scores.parquet').to_pylist()
    match={(r['protocol'],r['request_id']):r for r in right}
    assert len(match)==len(left)==len(right)
    joined=[]
    for row in left:
        other=match[row['protocol'],row['request_id']]
        for key in ('uid','version','label','item_idx','query_timestamp','fit64','reuse','rebuild'):
            assert row[key]==other[key],key
        joined.append(dict(row,fit128=other['fit128'],fit256=other['fit256']))
    config=json.loads((second/'configuration.json').read_text())['config']
    config=dict(config,fit_sizes=[16,32,128,256])
    settings=dict(config=config,scope='Fixed-predecessor V4; first128 original evaluation order plus separate registered32 panel; no new confirmation',
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (first/'scores.parquet',second/'scores.parquet')})
    (args.output/'configuration.json').write_text(json.dumps(settings,indent=2)+'\n')
    pq.write_table(pa.Table.from_pylist(joined),args.output/'scores.parquet',compression='zstd')
    quality=analyze(args.output,config)
    cm={kind:CostModel(192,6,6,kind) for kind in ('torch','triton')}
    old_source=json.loads((ORIGINAL/'source_v4/summary.json').read_text())
    source_by_uid={r['uid']:r for r in old_source['records']}
    training=[]
    for count in COUNTS:
        path=(ORIGINAL if count==64 else BASE/('attempt1' if count<64 else 'extended_attempt1')/f'fit{count}')/'calibration/v4/calibration.json'
        row=json.loads(path.read_text());joint=row['diagnostics']['joint'];selected=next(r for r in joint['history'] if r['epoch']==joint['selected_epoch'])
        charged_teacher=row['cost']['components']['teacher_cache_capture_flops']
        needed_teacher=charged_teacher
        if count<64:
            lengths=[source_by_uid[u]['history_length'] for u in row['fit_uids']+row['validation_uids']]
            needed_teacher=sum(cm['triton' if length==1024 else 'torch'].full_cache(length) for length in lengths)
        training.append(dict(fit_users=count,validation_users=16,elapsed_seconds=row['diagnostics']['elapsed_seconds'],
            calibration_flops_as_recorded=row['cost']['calibration_flops'],teacher_flops_as_recorded=charged_teacher,
            modeled_required_teacher_flops=needed_teacher,
            modeled_calibration_flops_required_rows=row['cost']['calibration_flops']-charged_teacher+needed_teacher,
            teacher_repricing_scope='For16/32 only: analytic per-UID required teacher geometry; actual retained80-row capture not rerun or erased.128/256 source packs already exact fit+16 rows.',
            validation_teacher_mse=selected['validation_teacher_logit_mse'],selected_epoch=joint['selected_epoch'],
            checkpoint_sha256=row['weights_sha256'],source_pack_sha256=row['source_pack_sha256'],metadata_path=str(path)))
    (args.output/'training.json').write_text(json.dumps(training,indent=2)+'\n')
    # Paired UID bootstrap, using grouped score ranks to preserve ties exactly.
    fields=[f'fit{n}' for n in COUNTS]+['reuse','rebuild'];intervals={}
    users=config['trend_evaluation_uids'];lookup={uid:i for i,uid in enumerate(users)}
    for protocol in config['protocols']:
        rows=[r for r in joined if r['uid'] in lookup and r['protocol']==protocol]
        uid=np.array([lookup[r['uid']] for r in rows]);y=np.array([r['label'] for r in rows]);pos,neg=y==1,y==0
        index={key:np.unique([r[key] for r in rows],return_inverse=True)[1] for key in fields}
        contrasts={f'{a}_minus_{b}':[] for a,b in zip(COUNTS[1:],COUNTS[:-1])}
        rng=np.random.default_rng(20261019)
        for _ in range(1000):
            w=np.bincount(rng.integers(0,len(users),len(users)),minlength=len(users))[uid];den=w[pos].sum()*w[neg].sum()
            if den==0:continue
            auc={}
            for key,rank in index.items():
                n=int(rank.max())+1;wp=np.bincount(rank[pos],weights=w[pos],minlength=n);wn=np.bincount(rank[neg],weights=w[neg],minlength=n)
                auc[key]=float(np.sum(wp*(np.cumsum(wn)-.5*wn))/den)
            for a,b in zip(COUNTS[1:],COUNTS[:-1]):contrasts[f'{a}_minus_{b}'].append(auc[f'fit{a}']-auc[f'fit{b}'])
        intervals[protocol]={name:dict(uid95=np.quantile(values,[.025,.975]).tolist(),valid_replicates=len(values)) for name,values in contrasts.items()}
    (args.output/'uncertainty.json').write_text(json.dumps(dict(seed=20261019,repetitions=1000,method='paired UID percentile95 for adjacent fit-size AUC differences; fixed first128, conditional on one fitted seed and prefix sample; pointwise, not simultaneous multiple-comparison intervals',protocols=intervals),indent=2)+'\n')
    lines=['# 固定 V3 → V4 的拟合人数曲线','',
        '只扩展原fit256角色，使用嵌套16/32/64/128/256人、同16人验证集、同local8/joint4/batch8/seed17/rollout4。前版V3与起点不变，原16/32/64不重训。主评价固定原4096 evaluation文件顺序前128人；另保留原登记32人的独立描述，两者159人并集只重放新128/256分支。','',
        '|V4拟合人数|校准流程秒数|验证 teacher MSE|主128 next-item AUC|主128 feedback AUC|记录校准TFLOPs|','|---|---:|---:|---:|---:|---:|']
    q=quality['panels']['fixed_first128']
    for row in training:
        n=row['fit_users'];lines.append(f"|{n}|{row['elapsed_seconds']:.2f}|{row['validation_teacher_mse']:.5f}|{q['behavior_stride16_candidates16']['auc'][f'fit{n}']:.5f}|{q['original_feedback']['auc'][f'fit{n}']:.5f}|{row['calibration_flops_as_recorded']/1e12:.3f}|")
    lines+=['','|相邻人数AUC增益|next-item UID95|feedback UID95|','|---|---|---|']
    for a,b in zip(COUNTS[1:],COUNTS[:-1]):
        ranges=[intervals[p][f'{a}_minus_{b}']['uid95'] for p in config['protocols']]
        lines.append(f'|{a}−{b}|[{ranges[0][0]:.5f}, {ranges[0][1]:.5f}]|[{ranges[1][0]:.5f}, {ranges[1][1]:.5f}]|')
    lines+=['','在这条固定配方的 Medium V4 开发曲线上，64人兼有最高的主128 next-item AUC和较低的校准成本，暂时保留原64人设置。128/256人继续降低验证 teacher MSE，却降低该面板的 next-item AUC；因此不能用拟合误差或样本数单独决定设置。feedback的相邻差区间均跨零，不能据此确认人数带来的质量差异。该结论不是样本已饱和、退化原因或其他模型所需人数的证明。','',
        '人数增多也增加优化步数，所以曲线衡量原配方的成本/质量，并未隔离样本多样性。每个时间点是一次实际流程，64人时间来自历史记录，跨次负载并未受控，不能据此主张严格加速倍数。区间只反映固定权重下的UID抽样，不包含训练种子或拟合子集变动，也未作多重比较校正。全部分支继承256人训练的起点与原64人V3；这里的人数仅指V4这次增量拟合。单边、开发面板以及feedback稀疏标签均限制外推，不能据此规定工业人数必须固定或必须按人口比例增长。','',
        'source只新增192名原fit角色，原80行直接复用；为128/256分别保存144/272行并精确累加对应teacher与源状态操作。新增执行时间包括两份产物保存，历史80人捕获另列一次。16/32原记录沿用全80人teacher费，training.json另提供按真实必要32/48行重价的理论FLOPs，未把该重价伪装为重新执行或实际时间。','',
        '评分身份及原64/Reuse/Rebuild列在两次评估间完全一致。全部五点及两面板见quality.json；代码、角色、source和权重绑定沿configuration.json及原训练metadata可查。恢复率分母较小时仅看百分比会夸大差异，主要比较直接AUC和成对UID区间。未根据结果改epoch或扩充新的校准角色。']
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(training=training,primary=q,intervals=intervals),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
