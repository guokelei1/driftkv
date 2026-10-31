#!/usr/bin/env python3
"""Render the already-computed simple candidate AUC tables (no inference)."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import json
from pathlib import Path
ATTRIBUTES={'popularity':('popular','long_tail'),'history_familiarity':('history_seen','history_unseen')}

def plot_outputs(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#2166ac','#d95f02'];methods=['kv_linear','f1','f2'];labels=['Linear (primary)','F1','F2 (sensitivity)']
    fig,axes=plt.subplots(1,2,figsize=(11,3.7))
    for axis,(attribute,categories) in zip(axes,ATTRIBUTES.items()):
        for offset,(category,color) in enumerate(zip(categories,colors)):
            point=result['groups']['all4000/'+category]
            for i,method in enumerate(methods):
                value=point['methods'][method]['AUC_gain_over_reuse'];x=i+(offset-.5)*.15
                center=100*value['value'];lo,hi=[100*v for v in value['ci95']]
                axis.errorbar(x,center,yerr=[[center-lo],[hi-center]],fmt='o',color=color,capsize=3,
                    label=category.replace('_',' ')+f' (UID={point["feedback_users"]})' if i==0 else None,
                    mfc=color if point['mechanism_use_permitted'] else 'white')
            gap=point['rebuild_minus_reuse_AUC']['value']
            axis.axhline(100*gap,color=color,ls=':',alpha=.55)
        axis.axhline(0,color='black',lw=.8);axis.set_xticks(range(3),labels);axis.set_ylabel('AUC gain over Reuse (percentage points)')
        axis.set_title(attribute.replace('_',' '));axis.legend(fontsize=8)
    fig.suptitle(f'Edge{result["edge"]}: candidate classes; dotted lines = Rebuild gains; hollow = UID<512')
    fig.tight_layout();fig.savefig(output/'candidate_class_auc_gain.png',dpi=180);fig.savefig(output/'candidate_class_auc_gain.pdf');plt.close(fig)
    comparisons=list(result['effect_differences'].items())
    fig,axes=plt.subplots(1,len(comparisons),figsize=(4.1*len(comparisons),3.4),squeeze=False)
    for axis,(name,comparison) in zip(axes[0],comparisons):
        for i,method in enumerate(methods):
            value=comparison['methods'][method];center=100*value['value'];lo,hi=[100*v for v in value['ci95']]
            axis.errorbar(center,i,xerr=[[center-lo],[hi-center]],fmt='o',color='#2166ac',capsize=3,
                mfc='#2166ac' if comparison['mechanism_use_permitted'] else 'white')
        axis.axvline(0,color='gray',ls='--');axis.set_yticks(range(3),labels);axis.invert_yaxis()
        scope=('Same '+str(result['scopes'][comparison['scope']]['resident_users'])+' users'
               if comparison['scope'].startswith('common_') else 'Fixed4000 cohort')
        axis.set_title(scope+'\n'+comparison['first_group'].replace('_',' ')+' − '+comparison['second_group'].replace('_',' '),fontsize=9)
        axis.set_xlabel('Difference in AUC gains (pp)')
    fig.suptitle(f'Edge{result["edge"]}: paired UID class-effect differences; hollow = UID<512');fig.tight_layout()
    fig.savefig(output/'candidate_class_effect_difference.png',dpi=180);fig.savefig(output/'candidate_class_effect_difference.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    plot_outputs(json.loads(args.input.read_text()),args.output)
