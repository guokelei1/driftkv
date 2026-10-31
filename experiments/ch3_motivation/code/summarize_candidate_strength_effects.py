#!/usr/bin/env python3
"""Fixed-endpoint effect ranges and descriptive stable/split intersections."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
DEFAULT=ROOT/'experiments/ch3_motivation/runs/candidate_strength_direction/attempt_1'
GROUPS=('near_zero','intermediate','near_one','above_one_strong','flat','ambiguous')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(root,output):
    effects=[];intersections=[];sources={};summaries={}
    for edge in (2,3):
        folder=root/f'A/full/edge_{edge}';analysis=folder/'item_analysis';summary_path=analysis/'summary.json'
        d=json.loads(summary_path.read_text());score_summary=json.loads((folder/'summary.json').read_text())
        curves={}
        with (analysis/'loss_curves.csv').open() as f:
            for r in csv.DictReader(f):curves.setdefault((r['scope'],int(r['item_idx'])),{})[float(r['lambda_value'])]=float(r['mean_log_loss'])
        events=[]
        for a in score_summary['artifacts']:
            p=folder/a['path']
            if sha(p)!=a['sha256']:raise ValueError('frozen score artifact changed')
            events.extend(pq.read_table(p,columns=['uid','item_idx','inherited_rows'],filters=[('method','=','lambda0')]).to_pylist())
        edge_summaries={}
        for scope,s in d['scopes'].items():
            items=[r for r in d['items'] if r['scope']==scope]
            selected=[r for r in items if r['bootstrap']['stable_descriptive'] is True and r['context_split']['halves_within_one_grid_step'] is True]
            selected_events=events if scope=='all_events' else [r for r in events if r['inherited_rows']>0]
            for g in ('all_intersection',*GROUPS):
                ids={r['item_idx'] for r in selected if g=='all_intersection' or r['preference']['group']==g}
                subset=[r for r in selected_events if r['item_idx'] in ids]
                intersections.append(dict(edge=edge,scope=scope,group=g,items=len(ids),feedback_users=len({r['uid'] for r in subset}),
                    events=len(subset),all_supported_items=s['items'],aggregate_feedback_support_at_least512=len({r['uid'] for r in subset})>=512))
            values={name:[] for name in ('L1_minus_L0','L2_minus_L0','curve_max_minus_min')}
            for item in items:
                c=curves[scope,item['item_idx']]
                if len(c)!=9:raise ValueError('complete frozen nine-point curve required')
                span=max(c.values())-min(c.values())
                if not np.isclose(span,item['preference']['loss_range'],atol=1e-14,rtol=0):raise ValueError('saved curve range differs')
                values['L1_minus_L0'].append(c[1.]-c[0.]);values['L2_minus_L0'].append(c[2.]-c[0.]);values['curve_max_minus_min'].append(span)
            for name,one in values.items():
                q=np.quantile(one,[0,.1,.5,.9,1])
                effects.append(dict(edge=edge,scope=scope,contrast=name,items=len(one),minimum=float(q[0]),p10=float(q[1]),median=float(q[2]),p90=float(q[3]),maximum=float(q[4])))
            edge_summaries[scope]=dict(all_items=s['items'],bootstrap_stable=s['bootstrap_rule_satisfied_items'],
                split_supported=s['split_supported_items'],split_within_step=s['split_within_one_step_items'],
                stable_and_split_agree_items=len(selected),selected_item_indices=[r['item_idx'] for r in selected])
        summaries[str(edge)]=edge_summaries
        sources[str(edge)]=dict(analysis_summary_sha256=sha(summary_path),loss_curves_sha256=sha(analysis/'loss_curves.csv'),score_summary_sha256=sha(folder/'summary.json'))
        # Presentation-only correction: the stable denominator is every supported item.
        report=analysis/'report.md';lines=report.read_text().splitlines();corrected=[]
        for line in lines:
            if 'Bootstrap rule satisfied/identified' in line:line=line.replace('Bootstrap rule satisfied/identified','Bootstrap stable / all items')
            for scope,s in d['scopes'].items():
                if line.startswith('| '+scope+' |'):
                    cells=line.split('|');cells[5]=f" {s['bootstrap_rule_satisfied_items']}/{s['items']} ";line='|'.join(cells)
            corrected.append(line)
        note='Presentation note: stable fractions above use ALL supported items as denominator. Stable-and-identified counts are separately preserved in summary.json; no numerical analysis or selection rule has changed.'
        if note not in corrected:corrected+=['',note]
        report.write_text('\n'.join(corrected)+'\n')
    output.mkdir(parents=True,exist_ok=True);write_csv(output/'fixed_contrast_effect_sizes.csv',effects);write_csv(output/'stable_split_intersection.csv',intersections)
    result=dict(format='candidate_strength_fixed_contrast_readout_v1',effect_units='nats per feedback event, first averaged within each supported item',
        direction='L1−L0 and L2−L0 compare fixed strengths: positive means higher loss at the higher lambda. Distributions use EVERY supported item; no oracle-selected overall benefit is formed.',
        scopes=summaries,effect_ranges=effects,stable_split_descriptive_counts=intersections,sources=sources,source_sha256=sha(__file__),
        limitations=['Stable∩split subsets are descriptive counts/coverage, never a newly selected primary quality population. UID can occur in more than one bin, so UID counts across bins are not additive.',
            'Boundary minima at0/2 describe this bounded grid; they do not establish an unrestricted optimal strength. Flat/tied and unstable items remain in the original complete tables.',
            'Fixed-endpoint effect ranges are descriptive item-level distributions, not an oracle-selected overall gain, deployed policy evaluation or significance claim.'])
    (output/'effects_and_stability.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# A: effect magnitudes and stability intersections','',result['direction'],'',
        '| Edge | Scope | Stable / all items | Split agreement / supported | Stable ∩ split agree / all items |','|---|---|---:|---:|---:|']
    for edge,scopes in summaries.items():
        for scope,s in scopes.items():lines.append(f"| {edge} | {scope} | {s['bootstrap_stable']}/{s['all_items']} | {s['split_within_step']}/{s['split_supported']} | {s['stable_and_split_agree_items']}/{s['all_items']} |")
    lines+=['','| Edge | Scope | Fixed contrast | Min | P10 | Median | P90 | Max |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in effects:lines.append('| '+' | '.join([str(r['edge']),r['scope'],r['contrast'],*[f"{r[k]:+.6f}" for k in ('minimum','p10','median','p90','maximum')]])+' |')
    lines+=['','| Edge | Scope | Stable ∩ split bin | Items | FeedbackUID | Events |','|---|---|---|---:|---:|---:|']
    for r in intersections:lines.append(f"| {r['edge']} | {r['scope']} | {r['group']} | {r['items']} | {r['feedback_users']} | {r['events']} |")
    lines+=['',*['- '+x for x in result['limitations']]]
    (output/'effects_and_stability.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='complete',scopes=summaries,effect_ranges=effects)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--output',type=Path)
    a=p.parse_args();run(a.root,a.output or a.root/'A/figures')
