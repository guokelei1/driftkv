#!/usr/bin/env python3
"""Disjoint development slices inside fixed7000; candidate coverage only."""

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
import pyarrow.parquet as pq

ATTRIBUTES={'history_familiarity':('history_seen','history_unseen'),
    'popularity':('popular','long_tail'),'seen_repetition':('history_once','history_repeated')}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('fixed slice artifact changed')
    else:path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def support(rows):
    return dict(feedback_users=len({r['uid'] for r in rows}),events=len(rows))


def run(root):
    root=Path(root);population_path=root/'population.json';population=json.loads(population_path.read_text())
    old_ref=population['prior4000_population'];old_path=Path(old_ref['path'])
    if sha(old_path)!=old_ref['sha256']:raise ValueError('old4000 source changed')
    old=json.loads(old_path.read_text());old_uids=old['users'];old_set=set(old_uids)
    added=[u for u in population['users'] if u not in old_set]
    if len(old_set)!=4000 or len(added)!=3000 or old_set&set(added) or old_set|set(added)!=set(population['users']):
        raise ValueError('fixed disjoint4000+3000 partition failed')
    slices=dict(format='candidate_development_population_slices_v1',population_sha256=sha(population_path),
        prior4000_population=old_ref,slices=dict(original4000=dict(users=old_uids,resident_users=4000),
            added3000=dict(users=added,resident_users=3000)),overlap_users=0,union_users=7000,
        selection='Original frozen4000 versus all additional3000 inherited from original mismatch role; no scores or candidate outcomes select membership.',
        scope='Disjoint development subsets; added3000 is not pristine confirmation. Full7000 overlaps original4000.',
        score_values_read=False,numeric_sources_modified=False,source_sha256=sha(__file__))
    freeze(root/'population_slices.json',slices)
    coverage=dict(format='candidate_development_slice_coverage_v1',slice_sha256=sha(root/'population_slices.json'),
        population_sha256=sha(population_path),score_values_read=False,edges={})
    lines=['# Fixed7000: original4000 versus added3000','',slices['selection'],slices['scope'],'',
        '| Edge | Slice | Attribute | First class UID | Second class UID | Common UID | Common>=512 |',
        '|---|---|---|---:|---:|---:|---|']
    for edge in [2,3]:
        manifest=json.loads((root/f'edge_{edge}/manifest.json').read_text());mapping=manifest['event_mapping']
        if sha(mapping['path'])!=mapping['sha256']:raise ValueError('fixed event mapping changed')
        data=pq.read_table(mapping['path']).to_pylist();edge_result=dict(event_mapping=mapping,slices={})
        for name,slice_entry in slices['slices'].items():
            members=set(slice_entry['users']);rows=[r for r in data if r['uid'] in members]
            groups={category:[] for pair in ATTRIBUTES.values() for category in pair}
            for r in rows:
                groups['history_seen' if r['history_seen'] else 'history_unseen'].append(r)
                groups['popular' if r['popular'] else 'long_tail'].append(r)
                if r['history_item_occurrences']:
                    groups['history_once' if r['history_item_occurrences']==1 else 'history_repeated'].append(r)
            result=dict(all=support(rows),groups={k:support(v) for k,v in groups.items()},attributes={})
            for attribute,(first,second) in ATTRIBUTES.items():
                common=sorted({r['uid'] for r in groups[first]}&{r['uid'] for r in groups[second]})
                result['attributes'][attribute]=dict(first_group=first,second_group=second,
                    common_feedback_users=len(common),common_uids=common,common_uid_support_permitted=len(common)>=512,
                    common_class_support={k:support([r for r in groups[k] if r['uid'] in set(common)]) for k in [first,second]})
                lines.append(f"| {edge} | {name} | {attribute} | {len({r['uid'] for r in groups[first]})} | {len({r['uid'] for r in groups[second]})} | {len(common)} | {'yes' if len(common)>=512 else 'diagnostic only'} |")
            edge_result['slices'][name]=result
        coverage['edges'][str(edge)]=edge_result
    freeze(root/'population_slice_coverage.json',coverage)
    lines+=['','This CPU coverage does not read labels or scores. UID support alone does not establish a defined AUC. The one follow-up quality question is whether the history-class direction persists within added3000; calculate both classes on their same-UID intersection. No new thresholds, figures, metrics or population search.']
    (root/'population_slices.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({edge:{name:{a:v['common_feedback_users'] for a,v in s['attributes'].items()}
        for name,s in value['slices'].items()} for edge,value in coverage['edges'].items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
