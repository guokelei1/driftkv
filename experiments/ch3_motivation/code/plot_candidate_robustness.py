#!/usr/bin/env python3
"""Plot ordinary remaining AUC gaps from completed same-UID class analyses."""
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

ROOT=Path(__file__).resolve().parents[3]
DEFAULT=ROOT/'experiments/ch3_motivation/runs/candidate_robustness/attempt_1'
METHODS=(('reuse','Reuse (native gap)'),('f1','F1'),('f2','F2 (×2)'),
    ('linear100_rows','Linear 100 · rows'),('linear100_users','Linear 100 · users'),
    ('linear256_rows','Linear 256 · rows'),('linear256_users','Linear 256 · users'))
ATTRIBUTES=(('history_familiarity','History familiarity',('history_seen','history_unseen'),('Seen','Unseen')),
            ('popularity','Global pre-release popularity',('popular','long_tail'),('Popular','Long tail')))
COLORS=('#2863A5','#CB772C')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();data={};sources=[]
    for edge in (2,3):
        folder=args.root/f'quality/edge_{edge}'
        source=folder/'analysis/summary.json';result=json.loads(source.read_text())
        summary=json.loads((folder/'summary.json').read_text())
        archive=folder/'analysis/analysis_source.py'
        if (summary['status']!='complete' or not summary['quality_use_permitted']
                or result['edge']!=edge or Path(result['provenance']['source']).resolve()!=folder.resolve()
                or sha(archive)!=result['provenance']['analysis_sha256']):
            raise ValueError('Completed, archived fresh7000 analysis required')
        if set(result['method_names'])!={m for m,_ in METHODS}|{'rebuild'}:
            raise ValueError('Refuse to omit an unexpected static method or missing F1/F2')
        data[edge]=result
        for path in (source,folder/'analysis/metrics.csv',folder/'analysis/paired_class_differences.csv',archive):
            sources.append(dict(path=str(path.resolve()),sha256=sha(path)))
    if len({v['provenance']['population_sha256'] for v in data.values()})!=1:
        raise ValueError('Two releases must use the same frozen7000 population')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12.6,8.2),sharex=True)
    supports=[]
    for row,edge in enumerate((2,3)):
        for col,(attribute,title,categories,labels) in enumerate(ATTRIBUTES):
            ax=axes[row,col];scope='common_'+attribute
            common=data[edge]['scopes'][scope]['resident_users']
            if common<512:raise ValueError(f'{edge}/{scope}: fewer than512 common UIDs')
            ax.axvline(0,color='.5',lw=.8,ls='--')
            counts=[]
            for ci,category in enumerate(categories):
                group=data[edge]['groups'][scope+'/'+category]
                if not group['mechanism_use_permitted'] or group['feedback_users']!=common:
                    raise ValueError('Each plotted class must contain the same>=512 feedback UIDs')
                counts.append(group['events'])
                supports.append(dict(edge=edge,scope=scope,category=category,
                    feedback_users=common,events=group['events']))
                for i,(method,_) in enumerate(METHODS):
                    point=group['methods'][method]['teacher_minus_map_AUC']
                    if point['value'] is None or point['ci95'] is None:
                        raise ValueError('Undefined paired AUC interval')
                    value=point['value']*100;low,high=[v*100 for v in point['ci95']]
                    y=i+(-.115 if ci==0 else .115)
                    ax.hlines(y,low,high,color=COLORS[ci],lw=1.25)
                    ax.vlines([low,high],y-.055,y+.055,color=COLORS[ci],lw=.9)
                    ax.scatter(value,y,s=24,color=COLORS[ci],edgecolors='white',linewidth=.4,zorder=3)
            ax.set_yticks(range(len(METHODS)),[label for _,label in METHODS])
            ax.set_ylim(len(METHODS)-.5,-.5);ax.grid(axis='x',alpha=.15)
            ax.ticklabel_format(axis='x',style='plain',useOffset=False)
            ax.set_title(f'V{edge-1} → V{edge} · {title} · same {common:,} users',pad=31)
            handles=[Line2D([0],[0],marker='o',linestyle='none',color=COLORS[i],
                label=f'{label} ({counts[i]:,} events)') for i,label in enumerate(labels)]
            ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(-.015,1.015),
                      ncol=2,frameon=False,fontsize=8,borderaxespad=0)
            ax.set_xlabel('Rebuild − method AUC (percentage points; 0 = Rebuild)')
    fig.suptitle('Remaining AUC gaps across candidate classes',fontsize=14,y=.99)
    fig.text(.5,.948,'Frozen 7,000-resident development population · same-UID class controls · all static maps retained',
             ha='center',fontsize=9,color='.3')
    fig.text(.5,.018,'Points and 95% paired UID-bootstrap intervals; fixed model and mapping weights. F2 is the original F correction scaled by 2.\n'
        'Linear maps differ in fit users (100/256) and row/user weighting. Recovery ratios and all absolute AUCs remain in the source tables.\n'
        'The Reuse row shows each class’s native gap. Different remaining gaps can reflect baseline difficulty; this plot does not establish gate necessity.',
        ha='center',fontsize=8,color='.35',linespacing=1.45)
    fig.subplots_adjust(left=.15,right=.985,top=.845,bottom=.145,hspace=.72,wspace=.39)
    output=args.output or args.root/'figures';output.mkdir(parents=True,exist_ok=True);artifacts=[]
    for extension in ('png','pdf'):
        path=output/f'static_candidate_robustness.{extension}'
        fig.savefig(path,dpi=220,bbox_inches='tight');artifacts.append(dict(path=str(path.resolve()),sha256=sha(path)))
    plt.close(fig)
    manifest=dict(format='candidate_robustness_plot_v1',sources=sources,artifacts=artifacts,supports=supports,
        plotting_source_sha256=sha(__file__),matplotlib_version=matplotlib.__version__,
        primary_attributes=['history_familiarity','popularity'],secondary_repetition_plotted=False,
        plotted_static_methods=[m for m,_ in METHODS if m!='reuse'],native_reference='reuse',
        unit='100 × ordinary Rebuild-minus-method AUC; no clipped recovery ratios',
        scope='Same UID class intersection per attribute and release; exploratory, not independent confirmation.')
    (output/'static_candidate_robustness_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',artifacts=artifacts,supports=supports)))


if __name__=='__main__':main()
