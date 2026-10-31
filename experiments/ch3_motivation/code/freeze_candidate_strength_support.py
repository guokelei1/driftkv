#!/usr/bin/env python3
"""Outcome-independent item support for a common-trajectory read-strength probe."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
OUTPUT=ROOT/'experiments/ch3_motivation/runs/candidate_strength_direction/attempt_1'
SOURCE=ROOT/'experiments/ch3_motivation/runs/candidate_robustness/attempt_1/population.json'
ORIGINAL=ROOT/'experiments/ch3_motivation/inputs/population.json'
PREFIX='candidate_strength_direction/attempt_1/context_split_v1|'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError(f'frozen file changed: {path}')
    else:path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def context_half(uid,timestamp):
    return hashlib.sha256((PREFIX+str(int(uid))+'|'+str(int(timestamp))).encode()).digest()[0]&1


def run():
    population=json.loads(SOURCE.read_text());original=json.loads(ORIGINAL.read_text())
    role_path=ROOT/original['role_source'];roles=json.loads(role_path.read_text())['roles']
    if sha(role_path)!=original['role_source_sha256']:raise ValueError('3.2 eligible-role source changed')
    users=population['users'];selected=set(users)
    if len(users)!=7000 or len(selected)!=7000:raise ValueError('requires frozen original7000')
    destination=OUTPUT/'population.json';OUTPUT.mkdir(parents=True,exist_ok=True)
    if destination.exists() and destination.read_bytes()!=SOURCE.read_bytes():raise ValueError('population changed')
    if not destination.exists():destination.write_bytes(SOURCE.read_bytes())
    plan=dict(format='candidate_read_strength_support_plan_v1',population=dict(path=str(destination.resolve()),sha256=sha(destination),
        exact_copy_source=str(SOURCE.resolve()),source_sha256=sha(SOURCE)),
        edges=[2,3],primary_support=dict(min_feedback_events=20,min_distinct_uid=10),
        stricter_label_support=dict(min_label0_events=5,min_label1_events=5,additional_to_primary=True),
        context_split=dict(hash_rule="SHA256(UTF8('"+PREFIX+"'+decimalUID+'|'+decimalTimestamp)), first-byte low bit",
            same_context_candidates_kept_together=True,min_events_each_half=10,min_uid_each_half=5,
            contexts_disjoint_but_uids_may_overlap=True),
        lambdas=[0,.25,.5,.75,1,1.25,1.5,1.75,2],
        intervention='One common Reuse/current-native append trajectory. Materialize linear256 cutover map once; only surviving inherited rows are interpolated read-only. New native rows never receive a lambda correction; no independent per-lambda writes.',
        selection_metric='Per-item event-mean observed BCE/logloss for each lambda; this is a descriptive oracle loss curve, not a learned or deployed item policy.',
        flat_rule='Mean-loss max−min<=1e−6nat => flat/unidentifiable; no representative lambda and no near1 assignment.',
        tie_rule='For nonflat curves retain every lambda whose mean loss is within1e−6nat of minimum; report tie set/span/groups. Representative is closest to1, then smaller lambda. Broad ties remain ambiguous.',
        lambda_groups=dict(near0=[0,.25],middle=[.5],near1=[.75,1,1.25],high=[1.5,1.75,2],flat='separate',ambiguous='tie-set spans multiple groups, reported separately'),
        strictly_above1='Report lambda*>1 separately; overlaps near1 at1.25 and high, so is not another mutually exclusive group.',
        bootstrap=dict(replicates=200,seed=17,unit='all7000 residentUID clusters',zero_item_weight_draws='invalid/unidentified, retain counts',
            outputs='Full flat/group/lambda frequencies and tie ambiguity; no posthoc stability cutoff'),
        forbidden=['Score/loss/lambda-based item eligibility','Outcome-based UID replacement','Overall AUC/recovery/gain after per-item oracle lambda selection','Calling flat/tie-broken1 a preference for1'],
        source_sha256=sha(__file__),score_values_read=False)
    freeze(OUTPUT/'support_plan.json',plan)
    trace=dict(original_eligible_role_source=dict(path=str(role_path.resolve()),sha256=sha(role_path),eligible_users=len(roles['quality'])),
        original_population=dict(path=str(ORIGINAL.resolve()),sha256=sha(ORIGINAL)),
        population_users=7000,not_in_original_3_2_eligible_pool=sorted(selected-set(roles['quality'])),
        original_main2000_users=len(original['main']),original_main2000_missing_from7000=sorted(set(original['main'])-selected),
        original_confirmation2000_overlap=len(selected&set(original['confirmation'])),original_development512_overlap=len(selected&set(original['development'])))
    coverage=dict(format='candidate_read_strength_support_coverage_v1',population_sha256=sha(destination),support_plan_sha256=sha(OUTPUT/'support_plan.json'),
        score_values_read=False,traceability=trace,edges={})
    for edge in (2,3):
        entry=population['edges'][str(edge)];requests=Path(entry['inputs']['requests']['path'])
        if sha(requests)!=entry['inputs']['requests']['sha256']:raise ValueError('real feedback source changed')
        rows=[r for r in pq.read_table(requests,columns=['uid','query_timestamp','item_idx','raw_item_id','label']).to_pylist() if int(r['uid']) in selected]
        stats=defaultdict(lambda:dict(rows=[],uids=set(),contexts=set(),positive_uids=set(),negative_uids=set(),halves=[[],[]]))
        for row in rows:
            u,t,item,y=map(int,[row['uid'],row['query_timestamp'],row['item_idx'],row['label']])
            if y not in (0,1):raise ValueError('nonbinary feedback')
            s=stats[item];s['rows'].append(row);s['uids'].add(u);s['contexts'].add((u,t));s['positive_uids' if y else 'negative_uids'].add(u)
            s['halves'][context_half(u,t)].append(row)
        table=[]
        for item,s in sorted(stats.items()):
            raw={int(r['raw_item_id']) for r in s['rows']}
            if len(raw)!=1:raise ValueError('item catalog identity changed')
            pos=sum(r['label']==1 for r in s['rows']);neg=len(s['rows'])-pos
            primary=len(s['rows'])>=20 and len(s['uids'])>=10
            h=[dict(events=len(v),users=len({int(r['uid']) for r in v}),positive=sum(r['label']==1 for r in v),negative=sum(r['label']==0 for r in v)) for v in s['halves']]
            table.append(dict(item_idx=item,raw_item_id=next(iter(raw)),events=len(s['rows']),feedback_users=len(s['uids']),contexts=len(s['contexts']),
                positive_events=pos,negative_events=neg,positive_users=len(s['positive_uids']),negative_users=len(s['negative_uids']),
                primary=primary,both_labels_ge5=primary and pos>=5 and neg>=5,
                split_supported=primary and all(v['events']>=10 and v['users']>=5 for v in h),
                half0_events=h[0]['events'],half0_users=h[0]['users'],half0_positive=h[0]['positive'],half0_negative=h[0]['negative'],
                half1_events=h[1]['events'],half1_users=h[1]['users'],half1_positive=h[1]['positive'],half1_negative=h[1]['negative']))
        path=OUTPUT/f'support_items_edge{edge}.parquet';pq.write_table(pa.Table.from_pylist(table),path)
        def covered(items):
            chosen={r['item_idx'] for r in items};observed=[r for r in rows if int(r['item_idx']) in chosen]
            return dict(items=len(items),feedback_users=len({r['uid'] for r in observed}),events=len(observed),positive_events=sum(r['label']==1 for r in observed),
                negative_events=sum(r['label']==0 for r in observed),only_positive_items=sum(r['negative_events']==0 for r in items),only_negative_items=sum(r['positive_events']==0 for r in items))
        primary=[r for r in table if r['primary']]
        coverage['edges'][str(edge)]=dict(all=covered(table),primary=covered(primary),stricter_both_labels_ge5=covered([r for r in table if r['both_labels_ge5']]),
            context_split_supported=covered([r for r in table if r['split_supported']]),primary_item_indices=[r['item_idx'] for r in primary],
            input_requests=dict(path=str(requests.resolve()),sha256=sha(requests)),input_history=entry['inputs']['history'],
            item_support_artifact=dict(path=str(path.resolve()),sha256=sha(path),rows=len(table)))
    common=set(coverage['edges']['2']['primary_item_indices'])&set(coverage['edges']['3']['primary_item_indices'])
    coverage['common_primary_items_across_edges']=sorted(common)
    freeze(OUTPUT/'support_coverage.json',coverage)
    lines=['# Fixed item support before lambda outcomes','',
        'Population is an exact copy of the original frozen7000. Main support is>=20feedback events and>=10UID per item, separately per release; no score/loss/lambda outcome was read.','',
        '| Edge | Scope | Items | FeedbackUID | Events | Positive/negative | Positive-only items |','|---|---|---:|---:|---:|---:|---:|']
    for edge,d in coverage['edges'].items():
        for scope in ['all','primary','stricter_both_labels_ge5','context_split_supported']:
            s=d[scope];lines.append(f"| {edge} | {scope} | {s['items']} | {s['feedback_users']} | {s['events']} | {s['positive_events']}/{s['negative_events']} | {s['only_positive_items']} |")
    lines+=['',f'Common primary items across both releases: {len(common)}.',
        f"All7000 are in the original3.2 eligible pool of{len(roles['quality'])}; outside count{len(trace['not_in_original_3_2_eligible_pool'])}. All original main2000 are contained; missing count{len(trace['original_main2000_missing_from7000'])}. Confirmation/development overlaps remain0.",
        '', 'Primary item support does not require both labels; positive-only/imbalanced items must be marked, since observed loss preferences depend on label composition. Context halves are disjoint contexts but may share UID; this is a stability exploration, not independent confirmation.',
        '', 'Flat curves and broad near-minimum ties are separate from identifiable preferences. All oracle item-loss preferences remain descriptive; no overall AUC/recovery after selecting per-item lambda will be reported.']
    (OUTPUT/'support_coverage.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='complete',coverage={e:{k:v for k,v in d.items() if k in ['all','primary','stricter_both_labels_ge5','context_split_supported']} for e,d in coverage['edges'].items()},common_primary_items=len(common),traceability=trace)),flush=True)


if __name__=='__main__':run()
