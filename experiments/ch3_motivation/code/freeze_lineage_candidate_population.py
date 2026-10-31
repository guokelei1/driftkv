#!/usr/bin/env python3
"""Freeze a hash-ordered subset of existing development roles, without scores."""
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
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'experiments/ch3_motivation/inputs'
OUTPUT=ROOT/'experiments/ch3_motivation/runs/lineage_candidate/attempt_1'
BASE=ROOT/'experiments/ch3_motivation/runs/candidate_robustness/attempt_1/population.json'
CALIB=ROOT/'experiments/ch3_motivation/inputs_feedback_pool/fit256/population.json'
PREFIX='lineage_candidate/attempt_1/v1|'
DAY=86400


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def role_sha(users):return hashlib.sha256(json.dumps(users,separators=(',',':')).encode()).hexdigest()


def freeze(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError(f'frozen content differs: {path}')
    else:path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def counts(rows):
    users={int(r['uid']) for r in rows}
    positives=[r for r in rows if r['label']==1];negatives=[r for r in rows if r['label']==0]
    return dict(feedback_users=len(users),events=len(rows),positive_events=len(positives),negative_events=len(negatives),
        both_label_users=len({int(r['uid']) for r in positives}&{int(r['uid']) for r in negatives}),
        mechanism_support_permitted=len(users)>=512 and bool(positives) and bool(negatives))


def run():
    base=json.loads(BASE.read_text());calib=json.loads(CALIB.read_text())
    original=json.loads((SOURCE/'population.json').read_text())
    ordered=sorted(base['users'],key=lambda u:(hashlib.sha256((PREFIX+str(u)).encode()).digest(),u))
    if len(ordered)!=7000 or len(set(ordered))!=7000:raise ValueError('expected source7000')
    selection=dict(hash_rule="SHA256(UTF8('lineage_candidate/attempt_1/v1|' + decimalUID)), digest ascending, UID tie break",
        source_population=dict(path=str(BASE.resolve()),sha256=sha(BASE)),candidate_sizes=[2048,3072],
        candidates={str(n):ordered[:n] for n in [2048,3072]},
        expansion_rule='Use2048 if edges2/3/4 each contain>=512 feedbackUID; otherwise use3072 globally. No further selection or user pruning.',
        role_selection_uses_score_values=False,role_selection_uses_label_values=False,source_sha256=sha(__file__))
    freeze(OUTPUT/'population_selection.json',selection)
    # Presence-only request reads decide only the previously declared fallback.
    bindings={};present={};inputs={}
    for edge in [2,3,4]:
        directory=SOURCE/f'edge_{edge}';binding_path=directory/'inputs.json';b=json.loads(binding_path.read_text())
        refs={k:dict(path=str((directory/b[k]['path']).resolve()),sha256=b[k]['sha256']) for k in ['history','requests']}
        refs['binding']=dict(path=str(binding_path.resolve()),sha256=sha(binding_path))
        for ref in refs.values():
            if sha(ref['path'])!=ref['sha256']:raise ValueError('source input changed')
        inputs[edge]=refs;bindings[edge]=b
        present[edge]=set(map(int,pq.read_table(refs['requests']['path'],columns=['uid'])['uid'].to_pylist()))
    size=next((n for n in [2048,3072] if all(len(set(ordered[:n])&present[e])>=512 for e in present)),None)
    if size is None:raise ValueError('both predeclared populations have insufficient feedback support')
    users=ordered[:size];selected=set(users)
    separation={name:len(selected&set(values)) for name,values in {
        'original_fit512':original['fit'],'fit100':calib['fit'][:100],'fit128':calib['fit'][:128],
        'fit256':calib['fit'][:256],'validation16':calib['validation'],
        'development512':original['development'],'confirmation2000':original['confirmation']}.items()}
    if any(separation.values()):raise ValueError('reserved-role overlap')
    population=dict(format='lineage_candidate_population_v1',users=users,users_sha256=role_sha(users),resident_users=size,
        source_population=selection['source_population'],selection_sha256=sha(OUTPUT/'population_selection.json'),
        source_roles={k:[u for u in users if u in set(v)] for k,v in base['roles'].items()},
        user_source_role={str(u):base['user_source_role'][str(u)] for u in users},
        fit_uids=calib['fit'][:100],fit_exploration128_uids=calib['fit'][:128],validation_uids=calib['validation'],
        calibration_population=dict(path=str(CALIB.resolve()),sha256=sha(CALIB)),role_separation=separation,
        evaluation_role='Hash-selected development subset of prior7000; not independent confirmation',
        original_roles_modified=False,score_values_read=False,label_values_used_for_selection=False,
        input_extraction_required=False,edges={})
    for edge in [2,3,4]:
        b=bindings[edge];cut=int(b['settings']['release_day'])*DAY;metadata={}
        with np.load(inputs[edge]['history']['path'],allow_pickle=False) as history:
            offsets=history['offsets'];timestamps=history['timestamps']
            for index,uid0 in enumerate(history['uids']):
                uid=int(uid0)
                if uid not in selected:continue
                a,z=map(int,offsets[index:index+2]);times=timestamps[a:z]
                end=int(np.searchsorted(times,cut,side='left'));start=max(0,end-1024);pre=times[start:end]
                if not len(pre):raise ValueError('empty cutover history')
                metadata[str(uid)]=dict(history_count=len(pre),last_prerelease_timestamp=int(pre[-1]),
                    pre_gap_seconds=int(cut-pre[-1]),prerelease14activity=int(len(pre)-np.searchsorted(pre,cut-14*DAY,side='left')))
        if set(map(int,metadata))!=selected:raise ValueError('history pack does not contain every selectedUID')
        population['edges'][str(edge)]=dict(settings=b['settings'],inputs=inputs[edge],user_metadata=metadata,
            feedback_uid_presence=len(selected&present[edge]),cutover_timestamp=cut,
            history_protocol='Last1024 real events strictly before target release; same frozen snapshot across paths and migration stages')
    freeze(OUTPUT/'population.json',population)
    # Labels are first inspected after users and all role memberships are frozen.
    coverage=dict(format='lineage_candidate_coverage_v1',population_sha256=sha(OUTPUT/'population.json'),
        score_values_read=False,label_values_used_for_selection=False,edges={})
    for edge in [2,3,4]:
        rows=[r for r in pq.read_table(inputs[edge]['requests']['path']).to_pylist() if int(r['uid']) in selected]
        cut=population['edges'][str(edge)]['cutover_timestamp'];first={}
        keys=[(int(r['uid']),int(r['query_timestamp']),int(r['item_idx'])) for r in rows]
        if len(keys)!=len(set(keys)):raise ValueError('duplicate real request key')
        if not all(cut<=int(r['query_timestamp'])<int(bindings[edge]['settings']['end_day'])*DAY for r in rows):raise ValueError('request outside release window')
        for r in rows:first[int(r['uid'])]=min(first.get(int(r['uid']),int(r['query_timestamp'])),int(r['query_timestamp']))
        firstrows=[r for r in rows if int(r['query_timestamp'])==first[int(r['uid'])]]
        coverage['edges'][str(edge)]=dict(all_window=counts(rows),first_feedback_timestamp=counts(firstrows),
            real_request_rows_sha256=hashlib.sha256(json.dumps(sorted(keys),separators=(',',':')).encode()).hexdigest(),
            query_protocol='Full real feedback window; real candidates and query_gap from last event in ALL real history with timestamp<query. Cache and teacher stay at target-cutover snapshot; no native append. Controlled counterfactual, not serving replay.',
            input_binding_sha256=inputs[edge]['binding']['sha256'])
    freeze(OUTPUT/'coverage.json',coverage)
    lines=['# Frozen lineage/candidate population','',selection['hash_rule'],selection['expansion_rule'],'',
        f'Selected {size} residents from existing7000. No scores or label values used for selection; no new extraction. Original roles and confirmation are unchanged. All reserved-role overlaps are zero.','',
        '| Target | Full-window feedbackUID | Events | Positive/negative | First-timestamp UID | First-timestamp events |','|---|---:|---:|---:|---:|---:|']
    for e,d in coverage['edges'].items():
        a=d['all_window'];f=d['first_feedback_timestamp'];lines.append(f"| {e} | {a['feedback_users']} | {a['events']} | {a['positive_events']}/{a['negative_events']} | {f['feedback_users']} | {f['events']} |")
    lines+=['','First state recipe: original fit100/val16, sourceV1, finaltargets3/4; fixed target-cutover history throughout V1→V2→V3(/V4). Full-window real candidate/time inputs; query gap derives from all pre-query real history, but every method and teacher intentionally retain the same cutover cache. This isolates representation migration under a counterfactual context and is not native serving replay.','',
        'The feedbackUID threshold is only an eligibility floor. Results must retain adverse effects and all explored configurations. Different final targets cannot alone establish a same-target monotonic effect of chain length.']
    (OUTPUT/'coverage.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(population=str((OUTPUT/'population.json').resolve()),sha256=sha(OUTPUT/'population.json'),resident_users=size,coverage=coverage['edges'])),flush=True)


if __name__=='__main__':run()
