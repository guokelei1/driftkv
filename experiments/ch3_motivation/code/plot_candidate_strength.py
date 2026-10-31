#!/usr/bin/env python3
"""Static count plots of all predeclared item-strength groups and stability."""
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[3]
DEFAULT=ROOT/'experiments/ch3_motivation/runs/candidate_strength_direction/attempt_1'
GROUPS=('near_zero','intermediate','near_one','above_one_strong','flat','ambiguous')
LABELS=('Near 0\n0/.25','Middle\n.5','Near 1\n.75–1.25','High\n1.5–2','Flat','Ties across\ngroups')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root,output):
    sources={};data={}
    for edge in (2,3):
        p=root/f'A/full/edge_{edge}/item_analysis/summary.json'
        d=json.loads(p.read_text())
        if d['edge']!=edge or not d['zero_inherited_scores_bitexact']:raise ValueError('analysis edge or flat check differs')
        if d['provenance']['root_plan_sha256']!=sha(root/'plan.json') or d['provenance']['population_sha256']!=sha(root/'population.json'):
            raise ValueError('figure sources do not match the frozen population/plan')
        data[edge]=d;sources[str(edge)]=dict(path=str(p.resolve()),sha256=sha(p))
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    rows=[];scope_names=('all_events','active_inherited_only')
    for row,scope in enumerate(scope_names):
        for col,edge in enumerate((2,3)):
            d=data[edge];s=d['scopes'][scope];items=[r for r in d['items'] if r['scope']==scope]
            if len(items)!=s['items'] or s['identified_group_counts']['invalid']:
                raise ValueError('supported item accounting differs or invalid point curve')
            total=np.asarray([sum(i['preference']['group']==g for i in items) for g in GROUPS])
            stable=np.asarray([sum(i['preference']['group']==g and i['preference']['identified_preference']
                and i['bootstrap']['stable_descriptive'] is True for i in items) for g in GROUPS])
            if int(total.sum())!=len(items):raise ValueError('group count conservation failed')
            ax=axes[row,col];x=np.arange(len(GROUPS))
            identified=np.arange(4);ax.bar(x[identified],stable[identified],width=.72,color='#2f63a6')
            ax.bar(x[identified],total[identified]-stable[identified],bottom=stable[identified],width=.72,color='#bbd1e7')
            ax.bar(x[4],total[4],width=.72,color='#aaaaaa');ax.bar(x[5],total[5],width=.72,color='#d59a52')
            for i,n in enumerate(total):ax.text(i,n+.25,str(n),ha='center',va='bottom',fontsize=10)
            ax.set_xticks(x,LABELS);ax.set_ylim(0,max(3,int(total.max())+3));ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
            name='All-event primary' if scope=='all_events' else 'Active inherited rows only'
            warning='' if s['aggregate_feedback_support_at_least512'] else ' · diagnostic (<512 UID)'
            ax.set_title(f"Edge {edge} · {name}{warning}\n{s['items']} items, {s['feedback_users']} feedback UID\n"
                f"Bootstrap stable {s['bootstrap_rule_satisfied_items']}/{s['items']} · "
                f"Context split agrees {s['split_within_one_step_items']}/{s['split_comparable_nonflat_items']}",fontsize=10)
            ax.set_ylabel('Supported item count')
            if row==0:ax.tick_params(labelbottom=True)
            for g,n,b in zip(GROUPS,total,stable,strict=True):
                rows.append(dict(edge=edge,scope=scope,group=g,item_count=int(n),item_fraction=float(n/len(items)) if items else None,
                    bootstrap_stable_identified_count=int(b),scope_items=len(items),scope_feedback_users=s['feedback_users'],scope_events=s['events']))
    fig.legend(handles=[Patch(color='#2f63a6',label='Bootstrap stable, identified group'),Patch(color='#bbd1e7',label='Other identified group'),
        Patch(color='#aaaaaa',label='Flat / no preference'),Patch(color='#d59a52',label='Near-minimum tie spans groups')],loc='upper center',ncol=2,bbox_to_anchor=(.5,1.02),frameon=False)
    fig.text(.5,.012,'Same frozen calibration; shared native writes; read-only strength on surviving inherited rows.\n'
        'Observed item loss preferences; label/context confounded. Flat/tie tolerance: 1e−6 nat.\n'
        'Stability is descriptive; split denominator = comparable supported halves. No oracle-selected policy AUC.',ha='center',fontsize=9,color='#444444')
    fig.tight_layout(rect=(0,.085,1,.92))
    fig.savefig(output/'candidate_strength_groups.png',dpi=220,bbox_inches='tight');fig.savefig(output/'candidate_strength_groups.pdf',bbox_inches='tight');plt.close(fig)
    with (output/'group_counts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    support_rows=[]
    for edge,d in data.items():
        for scope,s in d['scopes'].items():
            items=[r for r in d['items'] if r['scope']==scope]
            support_rows.append(dict(edge=edge,scope=scope,items=s['items'],feedback_users=s['feedback_users'],events=s['events'],
                positive_events=sum(i['positive_events'] for i in items),negative_events=sum(i['negative_events'] for i in items),
                positive_only_items=s['positive_only_items'],both_labels_ge5_items=s['both_labels_ge5_items'],
                flat_items=s['identified_group_counts']['flat'],ambiguous_items=s['identified_group_counts']['ambiguous'],
                bootstrap_rule_satisfied=s['bootstrap_rule_satisfied_items'],bootstrap_rule_satisfied_and_identified=s['bootstrap_rule_satisfied_and_identified_items'],
                context_split_supported=s['split_supported_items'],context_split_comparable_nonflat=s['split_comparable_nonflat_items'],context_split_within_step=s['split_within_one_step_items'],
                representative_strict_gt1=s['representative_strictly_above_one_items'],all_near_minima_strict_gt1=s['all_near_minima_strictly_above_one_items'],
                aggregate_feedback_support_at_least512=s['aggregate_feedback_support_at_least512']))
    with (output/'support_stability.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(support_rows[0]));writer.writeheader();writer.writerows(support_rows)
    lines=['# A: candidate strength support and stability','',
        '| Edge | Scope | Items / feedbackUID | Events positive/negative | Flat / ambiguous | Bootstrap stable / all items | Split supported / comparable / within .25 |','|---|---|---:|---:|---:|---:|---:|']
    for r in support_rows:lines.append(f"| {r['edge']} | {r['scope']} | {r['items']} / {r['feedback_users']} | {r['positive_events']}/{r['negative_events']} | {r['flat_items']}/{r['ambiguous_items']} | {r['bootstrap_rule_satisfied']}/{r['items']} | {r['context_split_supported']}/{r['context_split_comparable_nonflat']}/{r['context_split_within_step']} |")
    lines+=['','Main item support is20events/10UID; active-only sensitivity repeats the same rule on inherited_rows>0. Bootstrap flag requires>=190 valid joint-feedbackUID draws and P(|lambda_boot−lambda_point|<=.25)>=.70. Flat curves have no preferred lambda; broad cross-group ties stay ambiguous.',
        'The split columns distinguish sufficient input support, two nonflat comparable halves, and agreement within one grid step. Unsupported/flat halves are not called stable. Strict>1 is a separate overlapping description, including1.25 in the near1 bin.',
        'The stable counts in this table also have identified groups in all four scopes. Full group-identification counts remain in support_stability.csv; the denominator above is always every supported item, not only stable/identified items.',
        'Many minima lie at the tested0/2 boundaries; these are preferences within the bounded grid, not estimates of an unrestricted optimal strength. Fixed contrast effect ranges and stable∩split descriptions are recorded separately in effects_and_stability.md.',
        'These are observed-label per-item oracle loss preferences, not deployed item choices or proof of a candidate gate. Label imbalance, low item-level UID counts and frozen-weight uncertainty remain material limits. Every item and both releases are retained.']
    (output/'support_stability.md').write_text('\n'.join(lines)+'\n')
    artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='figure_manifest.json'}
    (output/'figure_manifest.json').write_text(json.dumps(dict(sources=sources,plot_source_sha256=sha(__file__),population_sha256=sha(root/'population.json'),
        root_plan_sha256=sha(root/'plan.json'),matplotlib=matplotlib.__version__,artifacts=artifacts),indent=2)+'\n')
    print(json.dumps(dict(status='complete',output=str(output.resolve()),rows=support_rows)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--output',type=Path)
    a=p.parse_args();run(a.root,a.output or a.root/'A/figures')
