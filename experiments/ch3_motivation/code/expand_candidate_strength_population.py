#!/usr/bin/env python3
"""Quantity-only union expansion before any new lambda scores are observed."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import hashlib,json,shutil
from collections import defaultdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from experiments.ch3_motivation.code.freeze_candidate_strength_support import context_half

RUN=ROOT/'experiments/ch3_motivation/runs/candidate_strength_direction/attempt_1'
BASE=ROOT/'experiments/ch3_motivation/runs/candidate_robustness/attempt_1/population.json'
ORIGINAL=ROOT/'experiments/ch3_motivation/inputs/population.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def role_sha(users):return hashlib.sha256(json.dumps(users,separators=(',',':')).encode()).hexdigest()


def run():
    if (RUN/'population_expansion.json').exists():raise ValueError('Expansion already frozen; do not change after new scores')
    archive=RUN/'preflight_7000';archive.mkdir(exist_ok=True)
    oldpaths=['population.json','support_plan.json','support_coverage.json','support_coverage.md']
    bindings=[]
    for name in oldpaths:
        old=RUN/name;new=archive/name
        if new.exists() and new.read_bytes()!=old.read_bytes():raise ValueError('preflight archive differs')
        if not new.exists():shutil.copyfile(old,new)
        bindings.append(dict(original_path=str(old.resolve()),archived_path=str(new.resolve()),sha256=sha(new)))
    original=json.loads(ORIGINAL.read_text());population=json.loads(BASE.read_text())
    prior=population['users'];prior_set=set(prior);sensitivity=original['population_sensitivity']['20261024']
    added=[u for u in sensitivity if u not in prior_set];users=prior+added;selected=set(users)
    if len(users)!=8929 or len(selected)!=8929 or len(added)!=1929:raise ValueError('unexpected original sensitivity union')
    reserved={name:len(selected&set(original[name])) for name in ['fit','validation','development','confirmation']}
    role_source=ROOT/original['role_source'];eligible=json.loads(role_source.read_text())['roles']['quality']
    if any(reserved.values()) or selected-set(eligible) or set(original['main'])-selected:raise ValueError('source population role mismatch')
    population.update(format='candidate_strength_population_v2',users=users,users_sha256=role_sha(users),
        counts=dict(mismatch=5000,sensA=2000,sensB_new_unique=1929,total=8929),added_residents=4929,
        scope='Original7000 union complete pre-existing sensitivity20261024; duplicateUID removed. Expanded on support counts only before new lambda losses, not independent confirmation.',
        selection='Full source-role union; all original7000 retained, all sensitivity20261024 added unless already present; no extra user sampling or pruning.',
        source_sha256=sha(__file__),source_population=dict(path=str(ORIGINAL.resolve()),sha256=sha(ORIGINAL)),
        original7000_population=dict(path=str(BASE.resolve()),sha256=sha(BASE)),label_values_used_for_expansion=False,
        original_roles_modified=False,score_values_read=False,input_extraction_required=False)
    population['roles']['sensB_unique_additions']=added
    population['role_bindings']['sensB_unique_additions']=dict(source_key='population_sensitivity/20261024',source_users=2000,
        overlap_with_prior7000=71,unique_additions=1929,sha256=role_sha(added))
    population['user_source_role'].update({str(u):'sensB_unique_additions' for u in added})
    packs={}
    for edge in (2,3):
        entry=population['edges'][str(edge)];cut=entry['settings']['release_day']*86400;metadata={}
        with np.load(entry['inputs']['history']['path'],allow_pickle=False) as h:
            offsets,times=h['offsets'],h['timestamps'];available=set(map(int,h['uids']))
            if selected-available:raise ValueError('existing pack lacks selected users')
            for i,u0 in enumerate(h['uids']):
                u=int(u0)
                if u not in selected:continue
                a,z=map(int,offsets[i:i+2]);t=times[a:z];end=int(np.searchsorted(t,cut,'left'));start=max(0,end-1024);pre=t[start:end]
                if not len(pre):raise ValueError('empty cutover prefix')
                metadata[str(u)]=dict(history_count=len(pre),prerelease14activity=int(len(pre)-np.searchsorted(pre,cut-14*86400,'left')),
                    pre_gap_seconds=int(cut-pre[-1]),last_prerelease_timestamp=int(pre[-1]),original_role=population['user_source_role'][str(u)],original_main_member=u in set(original['main']))
        entry['user_metadata']=metadata
        packs[str(edge)]=dict(pack_users=len(available),selected_present=len(selected&available),missing=0,input_binding=entry['inputs']['binding'],history=entry['inputs']['history'])
    save(RUN/'population.json',population);population_sha=sha(RUN/'population.json')
    expansion=dict(format='candidate_strength_population_expansion_v1',reason='Original fixed item20events/10UID coverage on edge3 was494UID<512; expand before reading any new lambda outcomes, retaining the same support thresholds.',
        prior7000=dict(path=str(BASE.resolve()),sha256=sha(BASE)),preflight_archive=bindings,
        original_population=dict(path=str(ORIGINAL.resolve()),sha256=sha(ORIGINAL)),source_sensitivity_key='population_sensitivity/20261024',
        source_sensitivity_users=2000,overlap_with_prior7000=71,new_unique_users=1929,final_users=8929,
        final_population=dict(path=str((RUN/'population.json').resolve()),sha256=population_sha),reserved_role_overlaps=reserved,
        eligible_pool=dict(path=str(role_source.resolve()),sha256=sha(role_source),users=len(eligible),outside_count=0),
        original_main2000_missing=0,canonical_pack_coverage=packs,new_scores_or_losses_read=False,original_roles_modified=False,source_sha256=sha(__file__))
    save(RUN/'population_expansion.json',expansion)
    plan=json.loads((archive/'support_plan.json').read_text());plan.update(format='candidate_read_strength_support_plan_v2',
        population=dict(path=str((RUN/'population.json').resolve()),sha256=population_sha,original7000_source=str(BASE.resolve()),original7000_sha256=sha(BASE)),
        expansion_manifest=dict(path=str((RUN/'population_expansion.json').resolve()),sha256=sha(RUN/'population_expansion.json')))
    plan['bootstrap']['unit']='all8929 fixed residentUID clusters'
    save(RUN/'support_plan.json',plan);save(RUN/'support_plan_v2.json',plan)
    coverage=dict(format='candidate_read_strength_support_coverage_v2',population_sha256=population_sha,support_plan_sha256=sha(RUN/'support_plan.json'),score_values_read=False,edges={})
    for edge in (2,3):
        entry=population['edges'][str(edge)];request_path=Path(entry['inputs']['requests']['path'])
        if sha(request_path)!=entry['inputs']['requests']['sha256']:raise ValueError('request pack changed')
        rows=[r for r in pq.read_table(request_path,columns=['uid','query_timestamp','item_idx','raw_item_id','label']).to_pylist() if r['uid'] in selected]
        items=defaultdict(list)
        for r in rows:items[r['item_idx']].append(r)
        table=[]
        for item,rs in sorted(items.items()):
            us={r['uid'] for r in rs};pos=sum(r['label']==1 for r in rs);neg=len(rs)-pos;primary=len(rs)>=20 and len(us)>=10
            halves=[[r for r in rs if context_half(r['uid'],r['query_timestamp'])==h] for h in (0,1)]
            h=[dict(events=len(v),users=len({r['uid'] for r in v}),positive=sum(r['label']==1 for r in v),negative=sum(r['label']==0 for r in v)) for v in halves]
            table.append(dict(item_idx=item,raw_item_id=int(rs[0]['raw_item_id']),events=len(rs),feedback_users=len(us),contexts=len({(r['uid'],r['query_timestamp']) for r in rs}),
                positive_events=pos,negative_events=neg,positive_users=len({r['uid'] for r in rs if r['label']==1}),negative_users=len({r['uid'] for r in rs if r['label']==0}),
                primary=primary,both_labels_ge5=primary and pos>=5 and neg>=5,split_supported=primary and all(v['events']>=10 and v['users']>=5 for v in h),
                **{f'half{i}_{k}':v for i,one in enumerate(h) for k,v in one.items()}))
        path=RUN/f'support_items_edge{edge}_v2.parquet';pq.write_table(pa.Table.from_pylist(table),path)
        def covered(chosen):
            ids={r['item_idx'] for r in chosen};observed=[r for r in rows if r['item_idx'] in ids]
            return dict(items=len(chosen),feedback_users=len({r['uid'] for r in observed}),events=len(observed),positive_events=sum(r['label']==1 for r in observed),negative_events=sum(r['label']==0 for r in observed),
                only_positive_items=sum(r['negative_events']==0 for r in chosen),only_negative_items=sum(r['positive_events']==0 for r in chosen))
        coverage['edges'][str(edge)]=dict(all=covered(table),primary=covered([r for r in table if r['primary']]),
            stricter_both_labels_ge5=covered([r for r in table if r['both_labels_ge5']]),context_split_supported=covered([r for r in table if r['split_supported']]),
            primary_item_indices=[r['item_idx'] for r in table if r['primary']],item_support_artifact=dict(path=str(path.resolve()),sha256=sha(path),rows=len(table)),input_requests=entry['inputs']['requests'])
    coverage['common_primary_items_across_edges']=sorted(set(coverage['edges']['2']['primary_item_indices'])&set(coverage['edges']['3']['primary_item_indices']))
    if any(e['primary']['feedback_users']<512 for e in coverage['edges'].values()):raise ValueError('expanded support still below512')
    save(RUN/'support_coverage.json',coverage);save(RUN/'support_coverage_v2.json',coverage)
    lines=['# Expanded fixed support, before new lambda results','',expansion['reason'],'',
        'Original7000 + all pre-existing sensitivity20261024 =8929 uniqueUID (71overlap,1929new). No extra users sampled. Full original7000 preflight files are preserved in preflight_7000; original item-support parquet files are also retained unchanged.','',
        '| Edge | Scope | Items | FeedbackUID | Events | Positive/negative | Positive-only items |','|---|---|---:|---:|---:|---:|---:|']
    for edge,d in coverage['edges'].items():
        for scope in ['all','primary','stricter_both_labels_ge5','context_split_supported']:
            s=d[scope];lines.append(f"| {edge} | {scope} | {s['items']} | {s['feedback_users']} | {s['events']} | {s['positive_events']}/{s['negative_events']} | {s['only_positive_items']} |")
    lines+=['',f"Final population SHA256: {population_sha}.",'All8929 belong to the original3.2 eligible75807; main2000 fully included. Fit/val/dev/confirmation overlaps0; canonical packs contain every selectedUID; no new extraction.',
        'The original support thresholds, lambda grid, flat/tie rules, context hashes and bootstrap seed are unchanged. Population expansion was based solely on inadequate UID coverage before new lambda losses. Final cohort is now frozen for bothA andB.']
    (RUN/'support_coverage.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='complete',population_sha256=population_sha,residents=len(users),coverage={e:{k:v for k,v in d.items() if k in ['all','primary','stricter_both_labels_ge5','context_split_supported']} for e,d in coverage['edges'].items()})),flush=True)


if __name__=='__main__':run()
