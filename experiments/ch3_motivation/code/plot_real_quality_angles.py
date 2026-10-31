#!/usr/bin/env python3
"""Plot completed real-feedback analyses without reading or fitting raw scores.

Example:
  python plot_real_quality_angles.py --root RUN_ROOT --output RUN_ROOT/figures

The two figures use actual-label AUC/log loss and paired UID intervals already
stored in analysis/summary.json. Oracle methods and <512-UID slices are omitted.
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

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT/'experiments/ch3_motivation/runs/real_quality_angles/attempt_1'
EDGES = (2,3)
EDGE_LABELS = {2:'V1 → V2',3:'V2 → V3'}
COLORS = {2:'#2863A5',3:'#CB772C'}
METHODS = (('reuse','Reuse'),('kv_shift','KV shift'),('kv_linear','Head-linear KV'),
           ('f1','F mapping'),('f2','F mapping ×2'),('f_key','F: Key only'),('f_value','F: Value only'),
           ('rebuild','Rebuild'))
METRICS = (('AUC_minus_reuse',100.,'AUC change vs Reuse (percentage points; ↑ better)'),
           ('log_loss_minus_reuse',1000.,'Log-loss change vs Reuse (×10⁻³; ↓ better)'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_analysis(root,cap,edge):
    path = root/f'quality_cap{cap}'/f'edge_{edge}'/'analysis/summary.json'
    data = json.loads(path.read_text())
    expected_source = path.parent.parent.resolve()
    if (data.get('edge') != edge
            or not data.get('scope','').startswith('Fresh all-request observed-feedback replay;')
            or Path(data.get('provenance',{}).get('source','')).resolve() != expected_source):
        raise ValueError(f'{path}: expected completed fresh all-request analysis for this run')
    all_users = data.get('groups',{}).get('all',{})
    if all_users.get('resident_users') != 4000 or not eligible(all_users):
        raise ValueError(f'{path}: expected frozen4000 residents and >=512 observed-feedback UIDs')
    return data,dict(path=str(path.resolve()),sha256=sha(path),cap=cap,edge=edge,
        feedback_users=all_users['feedback_users'],events=all_users['events'],
        population_sha256=data['provenance']['population_sha256'])


def eligible(group):
    return group.get('feedback_users',0)>=512 and group.get('mechanism_use_permitted',False)


def metric_interval(group,method,key,scale):
    item = group['methods'][method][key]
    if item['value'] is None or item.get('ci95') is None:
        raise ValueError(f'{method}/{key}: undefined point or UID interval')
    low,high = item['ci95']
    if low is None or high is None or low>high:
        raise ValueError(f'{method}/{key}: invalid UID interval')
    values = (item['value']*scale,low*scale,high*scale)
    if any(not (-float('inf')<v<float('inf')) for v in values):
        raise ValueError(f'{method}/{key}: nonfinite value')
    return values


def style(plt):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.titlesize':10,'axes.labelsize':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
        'savefig.facecolor':'white','figure.facecolor':'white'})


def horizontal_interval(ax,y,values,color):
    point,low,high = values
    ax.hlines(y,low,high,color=color,lw=1.4,zorder=2)
    ax.vlines([low,high],y-.075,y+.075,color=color,lw=1.)
    ax.scatter(point,y,s=27,color=color,edgecolors='white',linewidths=.45,zorder=3)


def vertical_interval(ax,x,values,color):
    point,low,high = values
    ax.vlines(x,low,high,color=color,lw=1.25,zorder=2)
    ax.hlines([low,high],x-.035,x+.035,color=color,lw=1.)
    ax.scatter(x,point,s=27,color=color,edgecolors='white',linewidths=.45,zorder=3)


def save(fig,output,name,plt):
    paths=[]
    for suffix in ('png','pdf'):
        path=output/f'{name}.{suffix}'
        fig.savefig(path,dpi=220,bbox_inches='tight')
        paths.append(dict(path=str(path.resolve()),sha256=sha(path)))
    plt.close(fig)
    return paths


def structures(data,output,plt):
    fig,axes=plt.subplots(2,2,figsize=(11.8,7.5),sharex='col')
    for row,edge in enumerate(EDGES):
        group=data[1024,edge]['groups']['all']
        for col,(key,scale,label) in enumerate(METRICS):
            ax=axes[row,col]
            ax.axvline(0,color='.55',lw=.8,ls='--',zorder=1)
            for i,(method,_) in enumerate(METHODS):
                horizontal_interval(ax,i,metric_interval(group,method,key,scale),COLORS[edge])
            ax.set_yticks(range(len(METHODS)),[label for _,label in METHODS])
            ax.set_ylim(len(METHODS)-.5,-.5)
            ax.grid(axis='x',alpha=.15)
            ax.ticklabel_format(axis='x',style='plain',useOffset=False)
            ax.set_xlabel(label)
            ax.set_title(f"{EDGE_LABELS[edge]}  ·  {group['feedback_users']:,} feedback users / {group['events']:,} events",
                         color=COLORS[edge],pad=9)
        anchor=group['methods']['reuse']
        axes[row,0].text(0,-.23,f"Reuse AUC: {anchor['AUC']['value']:.5f}",
            transform=axes[row,0].transAxes,fontsize=8,color='.35')
        axes[row,1].text(0,-.23,f"Reuse log loss: {anchor['log_loss']['value']:.5f}",
            transform=axes[row,1].transAxes,fontsize=8,color='.35')
    fig.suptitle('Simple cache corrections on real observed feedback',fontsize=14,y=.985)
    fig.text(.5,.943,'History cap 1,024 · frozen main + sensA 4,000 residents per edge · matched requests',
             ha='center',fontsize=9,color='.3')
    fig.text(.5,.018,'Points and 95% paired UID-bootstrap intervals. Changes initialize K/V once; subsequent native writes remain independent.\n'
             'Shift/linear and F use different calibration objectives. Research-only oracle paths are omitted.',
             ha='center',fontsize=8,color='.35',linespacing=1.5)
    fig.subplots_adjust(left=.12,right=.985,top=.885,bottom=.145,hspace=.63,wspace=.32)
    return save(fig,output,'real_feedback_structures',plt)


def lifecycle(data,output,plt):
    panels=(('FIFO lifetime · cap128',128,
        (('paired_fifo_before','Old rows present'),('paired_fifo_after','All old rows evicted'))),
        ('Elapsed time · cap1024',1024,
        (('days_0_3','Days 0–3'),('days_3_7','Days 3–7'),('days_7_14','Days 7–14'))),
        ('Pre-release conditions · cap1024',1024,
        (('pre_gap_lt1d','Gap <1d'),('pre_gap_ge1d','Gap ≥1d'),
         ('activity_lt256','Activity <256'),('activity_ge256','Activity ≥256'))))
    fig,axes=plt.subplots(2,3,figsize=(14.,6.5),sharex='col')
    omissions=[]
    for col,(title,cap,slices) in enumerate(panels):
        supports=[]
        for group_name,label in slices:
            counts=[]
            for edge in EDGES:
                group=data[cap,edge]['groups'].get(group_name,{})
                counts.append(str(group.get('feedback_users',0)))
                if not eligible(group):
                    omissions.append(dict(cap=cap,edge=edge,group=group_name,
                        feedback_users=group.get('feedback_users',0),reason='Below512 UID support or mechanism flag false'))
            supports.append(label+'\nn='+ '/'.join(counts))
        for row,(metric,scale,_) in enumerate(METRICS):
            ax=axes[row,col]
            ax.axhline(0,color='.55',ls='--',lw=.8,zorder=1)
            plotted=0
            for i,(group_name,_) in enumerate(slices):
                for edge in EDGES:
                    group=data[cap,edge]['groups'].get(group_name,{})
                    if not eligible(group):continue
                    offset=-.10 if edge==2 else .10
                    vertical_interval(ax,i+offset,metric_interval(group,'f1',metric,scale),COLORS[edge])
                    plotted+=1
            if not plotted:
                raise ValueError(f'No supported F1 effects to plot in {title}')
            ax.set_xticks(range(len(slices)),supports,fontsize=8)
            ax.set_xlim(-.5,len(slices)-.5)
            ax.grid(axis='y',alpha=.15)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            if row==0:ax.set_title(title,pad=10)
            if col==0:
                ax.set_ylabel('F − Reuse AUC (pp; ↑ better)' if row==0
                              else 'F − Reuse log loss (×10⁻³; ↓ better)')
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker='o',linestyle='none',color=COLORS[e],
             label=EDGE_LABELS[e]) for e in EDGES]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.948),ncol=2,frameon=False)
    fig.suptitle('F mapping across cache lifetimes and request conditions',fontsize=14,y=.992)
    fig.text(.5,.023,'95% paired UID-bootstrap intervals; n gives feedback users for V1→V2 / V2→V3. Groups below 512 UIDs are omitted.\n'
        'FIFO compares the same users observed before/after physical old-row eviction; causal influence can remain in new rows.\n'
        'All time slices preserve continuous replay; their requests and label composition can differ. Pre-release groups are fixed before release.',
        ha='center',fontsize=8,color='.35',linespacing=1.45)
    fig.subplots_adjust(left=.075,right=.988,top=.865,bottom=.21,hspace=.48,wspace=.30)
    return save(fig,output,'real_feedback_lifecycle',plt),omissions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_RUN,help='Run root containing quality_cap1024 and quality_cap128')
    parser.add_argument('--output',type=Path,help='Defaults to ROOT/figures')
    parser.add_argument('--kind',choices=('all','structures','lifecycle'),default='all')
    args=parser.parse_args()
    needed=[(1024,e) for e in EDGES]
    if args.kind in ('all','lifecycle'):needed += [(128,e) for e in EDGES]
    data={};sources=[]
    try:
        for cap,edge in needed:
            data[cap,edge],source=read_analysis(args.root,cap,edge);sources.append(source)
        if len({s['population_sha256'] for s in sources}) != 1:
            raise ValueError('Figures require the same frozen population across caps/edges')
    except (FileNotFoundError,ValueError) as error:
        parser.exit(2,f'Plots require completed analyses: {error}\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style(plt)
    output=args.output or args.root/'figures'
    output.mkdir(parents=True,exist_ok=True)
    figures=[];omissions=[]
    if args.kind in ('all','structures'):figures += structures(data,output,plt)
    if args.kind in ('all','lifecycle'):
        created,omissions=lifecycle(data,output,plt);figures+=created
    manifest=dict(format='real_feedback_angle_plots_v1',kind=args.kind,
        matplotlib_version=matplotlib.__version__,
        plotting_source_sha256=sha(__file__),sources=sources,artifacts=figures,
        omitted_groups=omissions,minimum_feedback_users=512,
        plotted_oracle_methods=False,plotted_teacher_mse_as_quality=False,
        interval_scope='Paired UID-bootstrap intervals read directly from each archived analysis; no new fitting, bootstrap or metric selection.',
        scales=dict(AUC_difference='100 × absolute AUC change, percentage points',
                    log_loss_difference='1000 × absolute log-loss change'),
        population_scope='Frozen main+sensA4000 exploration; not an independent confirmation.')
    (output/f'plot_manifest_{args.kind}.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',artifacts=figures,omitted_groups=omissions)))


if __name__=='__main__':
    main()
