#!/usr/bin/env python3
"""CPU-only candidate embedding-drift strata, fixed from prerelease fit items.

Prepare attributes/coverage first; analyze committed real scores afterward.
No model score or evaluation label chooses a threshold, method, or alpha.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
import numpy as np
import pyarrow.parquet as pq
import torch

RUN=ROOT/'experiments/ch3_motivation/runs/real_quality_angles/attempt_1'
METHODS=('reuse','kv_shift','kv_linear','f025','f05','f1','f15','f2','f_key','f_value','rebuild')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def counts(rows):
    users={int(r['uid']) for r in rows}
    positives=sum(int(r['label']) for r in rows)
    return dict(feedback_users=len(users),events=len(rows),positive_events=positives,
        negative_events=len(rows)-positives,
        mechanism_use_permitted=len(users)>=512 and 0<positives<len(rows))


def drift_values(states,ids):
    index=torch.as_tensor(ids,dtype=torch.long)
    old,new=[s['model']['item_emb.weight'].index_select(0,index).double() for s in states]
    return ((new-old).norm(dim=1)/(old.norm(dim=1)+1e-12)).numpy()


def prepare(root,edge):
    output=root/'item_drift'/f'edge_{edge}'
    if (output/'preparation.json').exists():
        raise ValueError('Prepared attributes already exist; retain them and use analyze or a new run root')
    population_path=root/'population.json'
    population=json.loads(population_path.read_text())
    users=set(population['users'])
    candidate=ROOT/f'experiments/ch3_motivation/runs/insight_extensions/attempt_1/candidate_edge{edge}'
    config=json.loads((candidate/'configuration.json').read_text())
    if len(config['roles']['fit'])!=100 or users & set(config['roles']['fit']+config['roles']['validation']):
        raise ValueError('Requires isolated original fit100/validation candidate collection')
    pool_ids=[];fit_uids=[];sources={str((candidate/'configuration.json').resolve()):sha(candidate/'configuration.json')}
    frozen=digest(config)
    for path in sorted((candidate/'fit').glob('*.npz')):
        manifest=path.with_suffix('.json');record=json.loads(manifest.read_text())
        if record['configuration_sha256']!=frozen or record['npz_sha256']!=sha(path) or record['role']!='fit':
            raise ValueError('Prerelease fit candidate block differs from its frozen manifest')
        with np.load(path,allow_pickle=False) as block:
            if block['uid'].tolist()!=record['uids']:raise ValueError('Fit candidate UID binding changed')
            pool_ids.extend(block['pool_ids'].reshape(-1).tolist());fit_uids.extend(block['uid'].tolist())
        sources[str(path.resolve())]=record['npz_sha256'];sources[str(manifest.resolve())]=sha(manifest)
    if fit_uids!=config['roles']['fit']:raise ValueError('Incomplete fit100 candidate pool')
    fit_ids=np.unique(np.asarray(pool_ids,dtype=np.int64))
    paths=[ROOT/f'results/unified_training_2026_09/large/seed17/checkpoints/v{v}/checkpoint_100.pt'
           for v in (edge-1,edge)]
    models={str(p.relative_to(ROOT)):sha(p) for p in paths}
    if models!=config['models']:raise ValueError('Models differ from prerelease candidate collection')
    states=[torch.load(p,map_location='cpu',mmap=True,weights_only=False) for p in paths]
    if states[0]['model']['item_emb.weight'].shape!=states[1]['model']['item_emb.weight'].shape:
        raise ValueError('Item embedding table geometry changed')
    fit_drift=drift_values(states,fit_ids)
    threshold=float(np.median(fit_drift))
    definition=dict(format='candidate_item_drift_threshold_v1',edge=edge,models=models,
        population_sha256=sha(population_path),fit_users=config['roles']['fit'],
        validation_users=config['roles']['validation'],fit_unique_items=len(fit_ids),
        sources=sources,source_sha256=sha(__file__),
        formula='norm(e_current(item)-e_source(item),2)/(norm(e_source(item),2)+1e-12), CPU float64',
        threshold=threshold,threshold_rule='Median over unique item IDs in the prerelease fit100 full64-item pools; no teacher near-panel selection.',
        groups=dict(low_drift='drift <= threshold',high_drift='drift > threshold'),
        fit_zero_drift_items=int((fit_drift==0).sum()),
        fit_drift_quantiles=dict(zip(['min','q25','q50','q75','q95','max'],
            np.quantile(fit_drift,[0,.25,.5,.75,.95,1]).tolist(),strict=True)),
        evaluation_labels_or_scores_used_to_select_threshold=False)
    # Persist the full rule before parsing evaluation request labels for coverage.
    write(output/'threshold_definition.json',definition)
    directory=ROOT/f'experiments/ch3_motivation/inputs/edge_{edge}'
    binding=json.loads((directory/'inputs.json').read_text())
    requests_path=directory/binding['requests']['path']
    if sha(requests_path)!=binding['requests']['sha256']:raise ValueError('Frozen observed requests changed')
    rows=[r for r in pq.read_table(requests_path).to_pylist() if int(r['uid']) in users]
    observed_ids=np.unique([r['item_idx'] for r in rows])
    observed_drift=drift_values(states,observed_ids)
    attributes=dict(zip(observed_ids.tolist(),observed_drift.tolist(),strict=True))
    attribute_path=output/'item_attributes.npz'
    np.savez_compressed(attribute_path,item_idx=observed_ids,drift=observed_drift,
                        high_drift=(observed_drift>threshold).astype(np.uint8))
    coverage={name:counts([r for r in rows if predicate(attributes[r['item_idx']])])
        for name,predicate in [('low_drift',lambda d:d<=threshold),('high_drift',lambda d:d>threshold)]}
    shared={r['uid'] for r in rows if attributes[r['item_idx']]<=threshold}&{
        r['uid'] for r in rows if attributes[r['item_idx']]>threshold}
    result=dict(definition,threshold_definition_sha256=sha(output/'threshold_definition.json'),
        observed_unique_items=len(observed_ids),observed_zero_drift_items=int((observed_drift==0).sum()),
        requests_path=str(requests_path.resolve()),requests_sha256=sha(requests_path),
        attributes_path=str(attribute_path.resolve()),attributes_sha256=sha(attribute_path),
        coverage=coverage,users_with_both_drift_groups=len(shared),
        two_group_quality_analysis_permitted=all(c['mechanism_use_permitted'] for c in coverage.values()),
        scope='Item attribute strata within a fixed4000 resident population; a UID may occur in both groups.',
        limitations=['Candidate drift correlates with item popularity/training exposure and request composition; this is conditional heterogeneity, not a causal gate ablation.',
            'Attribute values use release-available checkpoints; only requested IDs are extracted to avoid scanning the whole catalogue.',
            'Both groups need512 feedback UIDs; source/current model changes include other parameters.',
            'Different F alpha performance is descriptive; no evaluation label selects a deployed alpha or gate.'])
    write(output/'preparation.json',result)
    lines=[f'# 候选embedding漂移角度：edge{edge}（CPU预备）','',
        '阈值仅由隔离fit100的完整64候选池中唯一item确定，先固定阈值再解析测试反馈标签作覆盖统计。',
        f'归一化漂移阈值：`{threshold:.12g}`；fit唯一item {len(fit_ids)}，精确零漂移{int((fit_drift==0).sum())}。','',
        '| 属性组 | 反馈UID | 事件 | 正/负事件 | ≥512 UID |','|---|---:|---:|---:|---|']
    for name,c in coverage.items():
        lines.append(f"| {name} | {c['feedback_users']} | {c['events']} | {c['positive_events']}/{c['negative_events']} | {'是' if c['mechanism_use_permitted'] else '否：仅诊断'} |")
    lines+=['',f'同一UID可能出现在两组；两组皆出现的UID为{len(shared)}。分组未修改任何角色。', '',
        '后续只在原始真实请求上比较Reuse/F及各alpha的AUC、log loss和同UID配对区间。'+
        '若漂移组没有一致异质性，也保留该否定结果；分组效应不证明特定门控形式或可学习性。','']
    (output/'coverage.md').write_text('\n'.join(lines))
    return result


def analyze(root,edge):
    from experiments.ch3_motivation.code.analyze_real_quality import aligned_rows, summarize, interval
    output=root/'item_drift'/f'edge_{edge}'
    prep=json.loads((output/'preparation.json').read_text())
    if not prep['two_group_quality_analysis_permitted']:
        raise ValueError('Both item-drift groups lack the predeclared >=512 UID support')
    attributes_path=Path(prep['attributes_path'])
    if sha(attributes_path)!=prep['attributes_sha256'] or sha(root/'population.json')!=prep['population_sha256']:
        raise ValueError('Frozen candidate attributes or resident population changed')
    definition=output/'threshold_definition.json'
    if sha(definition)!=prep['threshold_definition_sha256']:raise ValueError('Frozen threshold definition changed')
    folder=root/'quality_cap1024'/f'edge_{edge}'
    config=json.loads((folder/'configuration.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    population=json.loads((root/'population.json').read_text())
    if (summary['status']!='complete' or config['diagnostic'] or not summary['quality_use_permitted']
            or config['models']!=prep['models'] or config['population_sha256']!=prep['population_sha256']
            or config['users']!=population['users']):
        raise ValueError('Requires matching complete fresh4000 run, checkpoints and population')
    rows=[];frozen=digest(config)
    for artifact in summary['artifacts']:
        path=folder/artifact['path'];manifest=folder/artifact['manifest']
        if sha(path)!=artifact['sha256'] or sha(manifest)!=artifact['manifest_sha256']:
            raise ValueError('Committed real score block changed')
        record=json.loads(manifest.read_text())
        if record['configuration_sha256']!=frozen or record['score_sha256']!=artifact['sha256']:
            raise ValueError('Score block configuration differs')
        rows.extend(r for r in pq.read_table(path).to_pylist() if r['method'] in METHODS)
    groups=aligned_rows(rows)
    with np.load(attributes_path,allow_pickle=False) as table:
        high=dict(zip(table['item_idx'].tolist(),table['high_drift'].astype(bool).tolist(),strict=True))
    points={};draws={}
    for name,select_high in [('low_drift',False),('high_drift',True)]:
        selected={m:[r for r in values if high[r['item_idx']]==select_high] for m,values in groups.items()}
        point,samples=summarize(selected,population['users'])
        if counts(selected['reuse'])!=prep['coverage'][name]:raise ValueError('Real scored coverage differs from frozen attribute preparation')
        points[name]=dict(feedback_users=point['feedback_users'],events=point['events'],
            positive_events=point['positive_events'],negative_events=point['negative_events'],
            mechanism_use_permitted=point['mechanism_use_permitted'],
            methods={m:{k:metrics[k] for k in ('AUC','log_loss','AUC_minus_reuse','log_loss_minus_reuse')}
                     for m,metrics in point['methods'].items()})
        draws[name]=samples
    contrasts={}
    for method in points['low_drift']['methods']:
        contrasts[method]={}
        for metric in ('AUC','log_loss'):
            key=metric+'_minus_reuse'
            lo=points['low_drift']['methods'][method][key]['value']
            hi=points['high_drift']['methods'][method][key]['value']
            lo_draw=draws['low_drift'][method][metric]-draws['low_drift']['reuse'][metric]
            hi_draw=draws['high_drift'][method][metric]-draws['high_drift']['reuse'][metric]
            contrasts[method][metric+'_effect_high_minus_low']=interval(hi-lo,hi_draw-lo_draw)
    result=dict(edge=edge,groups=points,effect_heterogeneity=contrasts,
        preparation_sha256=sha(output/'preparation.json'),attributes_sha256=sha(attributes_path),
        threshold_definition_sha256=sha(definition),score_summary_sha256=sha(folder/'summary.json'),
        configuration_sha256=sha(folder/'configuration.json'),analysis_source_sha256=sha(__file__),
        metric_dependency_sha256=sha(ROOT/'experiments/ch3_motivation/code/analyze_real_quality.py'),
        scope='Actual-label prediction within candidate-attribute strata. Threshold fixed from prerelease fit100; paired UID bootstrap shares4000 weights across groups.',
        limitations=prep['limitations'])
    write(output/'quality_summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','analyze'))
    parser.add_argument('--root',type=Path,default=RUN)
    parser.add_argument('--edge',type=int,choices=(2,3),action='append')
    args=parser.parse_args()
    torch.set_num_threads(4)
    for edge in args.edge or (2,3):
        value=(prepare if args.stage=='prepare' else analyze)(args.root,edge)
        print(json.dumps(dict(edge=edge,stage=args.stage,status='complete',
            coverage=value.get('coverage',{}),threshold=value.get('threshold'))),flush=True)
