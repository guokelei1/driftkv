#!/usr/bin/env python3
"""Expand fixed source roles to7000; preserve causal candidate group rules."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'experiments/ch3_motivation/inputs'
PRIOR=ROOT/'experiments/ch3_motivation/runs/real_quality_angles/attempt_1'
DAY=86400


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def role_sha(uids):return hashlib.sha256(json.dumps(uids,separators=(',',':')).encode()).hexdigest()


def freeze(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('frozen file changed; use another attempt')
    else:path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def count(rows):
    users={r['uid'] for r in rows};pos={r['uid'] for r in rows if r['label']==1};neg={r['uid'] for r in rows if r['label']==0}
    return dict(feedback_users=len(users),events=len(rows),positive_events=sum(r['label']==1 for r in rows),
        negative_events=sum(r['label']==0 for r in rows),positive_users=len(pos),negative_users=len(neg),
        both_label_users=len(pos&neg),mechanism_support_permitted=len(users)>=512 and bool(pos) and bool(neg))


def run(output):
    output=Path(output);source_path=SOURCE/'population.json';source=json.loads(source_path.read_text())
    fit_path=ROOT/'experiments/ch3_motivation/inputs_feedback_pool/fit256/population.json';fit=json.loads(fit_path.read_text())
    roles=dict(mismatch=source['mismatch'],sensA=source['population_sensitivity']['20261023'])
    users=roles['mismatch']+roles['sensA'];selected=set(users)
    reserved={name:source[name] for name in ['fit','validation','development','confirmation']}
    if len(users)!=len(selected) or len(users)!=7000:raise ValueError('expected fixed7000 source-role union')
    if any(selected&set(values) for values in reserved.values()):raise ValueError('reserved role overlap')
    if selected&set(fit['fit'][:256]+fit['validation']):raise ValueError('fit256/val16 overlap')
    prior_population=PRIOR/'population.json';prior=json.loads(prior_population.read_text())
    if not set(prior['users'])<=selected:raise ValueError('not the intended expansion of prior4000')
    attr_root=PRIOR/'candidate_groups/attempt_2'
    original_def=json.loads((attr_root/'definitions.json').read_text())
    if original_def['popularity_rank_fraction']!=.2:raise ValueError('original fixed20% rule changed')
    population=dict(format='candidate_robustness_population_v1',users=users,roles=roles,
        user_source_role={str(u):name for name,values in roles.items() for u in values},
        original_main_users=source['main'],counts=dict(mismatch=5000,sensA=2000,total=7000),
        source_population=dict(path=str(source_path),sha256=sha(source_path)),
        prior4000_population=dict(path=str(prior_population),sha256=sha(prior_population)),
        added_residents=3000,users_sha256=role_sha(users),
        role_bindings=dict(mismatch=dict(source_key='mismatch',sha256=role_sha(roles['mismatch'])),
            sensA=dict(source_key='population_sensitivity/20261023',sha256=role_sha(roles['sensA']))),
        calibration_population=dict(path=str(fit_path),sha256=sha(fit_path)),
        role_separation=dict(full_original_fit_overlap=0,first256_fit_overlap=0,validation16_overlap=0,development_overlap=0,confirmation_overlap=0),
        selection='All original mismatch5000 plus all original sensA2000, before any new score is read; no method-outcome admission or user pruning.',
        scope='Coverage/robustness expansion including prior4000, not independent confirmation; confirmation remains untouched.',
        original_roles_modified=False,input_extraction_required=False,score_values_read=False,
        source_sha256=sha(__file__),edges={})
    definitions=dict(format='candidate_robustness_attributes_v1',primary_attributes=['history_familiarity','popularity'],
        secondary_attributes=['seen_repetition'],history_cap=1024,popularity_rank_fraction=.2,
        groups=dict(original_def['groups'],history_once='candidate appears exactly1time in retained cutover history',
            history_repeated='candidate appears at least2times in retained cutover history; both repetition groups condition on history_seen'),
        original_attribute_definition=dict(path=str(attr_root/'definitions.json'),sha256=sha(attr_root/'definitions.json')),
        repeated_split_is_natural_integer_boundary=True,threshold_search_performed=False,score_values_read=False,edges={})
    occurrences={}
    for edge in [2,3]:
        directory=SOURCE/f'edge_{edge}';binding_path=directory/'inputs.json';binding=json.loads(binding_path.read_text())
        if binding['population_sha256']!=sha(source_path):raise ValueError('original roles changed')
        cfg=binding['settings'];cut=int(cfg['release_day'])*DAY
        inputs={name:dict(path=str(directory/binding[name]['path']),sha256=binding[name]['sha256']) for name in ['history','requests']}
        inputs['binding']=dict(path=str(binding_path),sha256=sha(binding_path))
        for ref in inputs.values():
            if sha(ref['path'])!=ref['sha256']:raise ValueError('native pack changed')
        metadata={};counter={}
        with np.load(inputs['history']['path'],allow_pickle=False) as history:
            offsets,timestamps,items=history['offsets'],history['timestamps'],history['item_ids']
            for index,uid in enumerate(history['uids']):
                uid=int(uid)
                if uid not in selected:continue
                a,b=map(int,offsets[index:index+2]);times=timestamps[a:b]
                end=int(np.searchsorted(times,cut,side='left'));start=max(0,end-1024);pre=times[start:end]
                if not len(pre):raise ValueError('empty cutover prefix')
                counter[uid]=Counter(map(int,items[a+start:a+end]))
                metadata[str(uid)]=dict(history_count=len(pre),prerelease14activity=int(len(pre)-np.searchsorted(pre,cut-14*DAY,side='left')),
                    pre_gap_seconds=int(cut-pre[-1]),last_prerelease_timestamp=int(pre[-1]),
                    original_role=population['user_source_role'][str(uid)],original_main_member=uid in set(source['main']))
        if set(map(int,metadata))!=selected:raise ValueError('native pack omitted frozen residents')
        population['edges'][str(edge)]=dict(settings=cfg,inputs=inputs,user_metadata=metadata,
            feature_scope='Retained cutover native prefix capped1024, not full lifetime history; strict timestamp<release')
        prior_manifest_path=attr_root/f'edge_{edge}/manifest.json';prior_manifest=json.loads(prior_manifest_path.read_text())
        frequency=prior_manifest['frequency']
        if sha(frequency['path'])!=frequency['sha256']:raise ValueError('fixed global item frequencies changed')
        definitions['edges'][str(edge)]=dict(global_frequency=frequency,
            original_manifest=dict(path=str(prior_manifest_path),sha256=sha(prior_manifest_path)),native_inputs=inputs)
        occurrences[edge]=counter
    # The expanded role union and unchanged rules are persisted before requests.
    freeze(output/'population.json',population)
    definitions['population_sha256']=sha(output/'population.json');freeze(output/'definitions.json',definitions)
    coverage=dict(format='candidate_robustness_coverage_v1',population_sha256=sha(output/'population.json'),
        definition_sha256=sha(output/'definitions.json'),score_values_read=False,edges={})
    for edge in [2,3]:
        entry=population['edges'][str(edge)];directory=output/f'edge_{edge}'
        edge_def=dict(definitions,edge=edge);edge_def.pop('edges');edge_def['source']=definitions['edges'][str(edge)]
        freeze(directory/'definition.json',edge_def)
        freq=pq.read_table(definitions['edges'][str(edge)]['global_frequency']['path']).to_pydict()
        frequency={int(item):(int(n),int(rank),bool(popular)) for item,n,rank,popular in zip(
            freq['raw_item_id'],freq['global_pre14_item_events'],freq['global_pre14_item_rank'],freq['popular'])}
        requests=[r for r in pq.read_table(entry['inputs']['requests']['path']).to_pylist() if int(r['uid']) in selected]
        keys=[(r['uid'],r['query_timestamp'],r['item_idx']) for r in requests]
        if len(keys)!=len(set(keys)):raise ValueError('duplicate/conflicting observation key')
        groups={name:[] for name in definitions['groups']};mapping=[]
        for request in requests:
            uid=int(request['uid']);repeats=occurrences[edge][uid][int(request['item_idx'])]
            n,rank,popular=frequency.get(int(request['raw_item_id']),(0,0,False))
            mapping.append(dict(uid=uid,query_timestamp=int(request['query_timestamp']),item_idx=int(request['item_idx']),
                raw_item_id=int(request['raw_item_id']),history_seen=repeats>0,history_item_occurrences=repeats,
                popular=popular,global_pre14_item_events=n,global_pre14_item_rank=rank))
            groups['history_seen' if repeats else 'history_unseen'].append(request)
            groups['popular' if popular else 'long_tail'].append(request)
            if repeats:groups['history_once' if repeats==1 else 'history_repeated'].append(request)
        pairs={}
        for attribute,a,b in [('history_familiarity','history_seen','history_unseen'),('popularity','popular','long_tail'),('seen_repetition','history_once','history_repeated')]:
            common={r['uid'] for r in groups[a]}&{r['uid'] for r in groups[b]}
            pairs[attribute]=dict(common_feedback_users=len(common),uids=sorted(common),mechanism_support_permitted=len(common)>=512,
                groups={name:count([r for r in groups[name] if r['uid'] in common]) for name in [a,b]})
        schema=pa.schema([('uid',pa.uint64()),('query_timestamp',pa.int64()),('item_idx',pa.int64()),('raw_item_id',pa.uint64()),
            ('history_seen',pa.bool_()),('history_item_occurrences',pa.int64()),('popular',pa.bool_()),
            ('global_pre14_item_events',pa.int64()),('global_pre14_item_rank',pa.int64())])
        mapping_path=directory/'eventgroups.parquet';pq.write_table(pa.Table.from_pylist(mapping,schema=schema),mapping_path)
        result=dict(all=count(requests),groups={name:count(v) for name,v in groups.items()},paired_classes=pairs,resident_users=7000,
            event_mapping=dict(path=str(mapping_path.resolve()),sha256=sha(mapping_path),rows=len(mapping)))
        coverage['edges'][str(edge)]=result
        freeze(directory/'manifest.json',dict(definition=dict(path=str((directory/'definition.json').resolve()),sha256=sha(directory/'definition.json')),
            event_mapping=result['event_mapping'],global_frequency=definitions['edges'][str(edge)]['global_frequency'],
            population_sha256=sha(output/'population.json'),coverage=result))
        print(json.dumps(dict(edge=edge,groups=result['groups'],paired={k:v['common_feedback_users'] for k,v in pairs.items()})),flush=True)
    freeze(output/'coverage.json',coverage)
    lines=['# Fixed7000 candidate robustness population','',population['selection'],population['scope'],'',
        'Source roles: mismatch5000 (including original main2000) + complete sensA2000. Zero overlap with original fit, first256 fitting users, val16, development and confirmation. Existing packs suffice. No scores read, no thresholds changed.','',
        '| Edge | Class | Feedback UIDs | Events | Positive/negative |','|---|---|---:|---:|---:|']
    for edge,result in coverage['edges'].items():
        for name,v in result['groups'].items():lines.append(f"| {edge} | {name} | {v['feedback_users']} | {v['events']} | {v['positive_events']}/{v['negative_events']} |")
        for name,pair in result['paired_classes'].items():lines.append(f"\nEdge{edge} same UID both classes, {name}: {pair['common_feedback_users']}.\n")
    lines+=['','Primary attributes remain cutover history seen/unseen and original global top20% popularity. Secondary repetition is conditional on seen: exactly1 vs at least2 occurrences, a natural integer boundary. All three are recorded; no outcome-specific users or candidate pruning.','',
        'Event mapping joins by(uid,query_timestamp,item_idx), with history_item_occurrences added to the prior two-boolean schema. Both primary attributes partition all feedback; repetition only partitions seen feedback. Use paired7000 UID bootstrap weights, and show sameUID class intersections to control user composition. Original4000 and7000 overlap, so their results are correlated.']
    (output/'coverage.md').write_text('\n'.join(lines)+'\n')
    return coverage


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
