#!/usr/bin/env python3
"""All-head native/delta spectrum figures from completed full-population runs."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
DEFAULT=ROOT/'experiments/ch3_motivation/runs/candidate_strength_direction/attempt_1'
MODES=('uncentered','centered')
QUANTITIES=(('source','Source native','#2863A5'),('rebuilt','Current native','#C57725'),('delta','Version delta','#8855A5'))
FIELDS=('K','V')
RANKS=(4,6,8)
THRESHOLDS=(.8,.9,.95)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def distribution(values):
    values=np.asarray(values,dtype=float);keep=values[np.isfinite(values)]
    return dict(valid_heads=len(keep),invalid_heads=len(values)-len(keep),minimum=float(keep.min()) if len(keep) else None,
        q10=float(np.quantile(keep,.1)) if len(keep) else None,median=float(np.median(keep)) if len(keep) else None,
        q90=float(np.quantile(keep,.9)) if len(keep) else None,maximum=float(keep.max()) if len(keep) else None)


def load_full(root):
    data={};sources=[]
    for edge in (2,3):
        folder=root/f'B/full/edge_{edge}';sp=folder/'summary.json';summary=json.loads(sp.read_text())
        config=json.loads((folder/'configuration.json').read_text());row_manifest=json.loads((folder/'row_manifest.json').read_text())
        if (summary['status']!='complete' or summary['diagnostic'] or config['diagnostic']
                or summary['configuration']!=config or config['edge']!=edge
                or (config['layers'],config['heads'],config['head_width'])!=(10,10,32)):
            raise ValueError('Completed full-population Large100head spectrum runs required')
        users=config['population_users']
        if (summary['resident_users']!=len(users) or len(users)<512 or len(set(users))!=len(users)
                or set(map(str,users))!=set(row_manifest['users'])
                or row_manifest['configuration_sha256']!=digest(config)
                or summary['pooled_rows_per_head']!=sum(v['history_rows'] for v in row_manifest['users'].values())
                or sha(Path(config['population']))!=config['population_sha256']):
            raise ValueError('Full resident/row/population coverage mismatch')
        table={}
        with (folder/'per_head.csv').open() as handle:
            for row in csv.DictReader(handle):
                key=(row['centering'],row['quantity'],int(row['layer']),int(row['head']))
                if key in table or int(row['edge'])!=edge or int(row['pooled_rows'])!=summary['pooled_rows_per_head']:
                    raise ValueError('Duplicate or mismatched spectrum row')
                valid=row['nonzero_energy']=='True'
                curve=np.array([float(row[f'energy_r{r}']) if row[f'energy_r{r}'] else np.nan for r in range(1,33)])
                if valid and (not np.isfinite(curve).all() or np.any(np.diff(curve)<-1e-12)
                        or np.any(curve<-1e-12) or np.any(curve>1+1e-12) or abs(curve[-1]-1)>1e-9):
                    raise ValueError('Invalid cumulative squared-singular-value energy')
                if not valid and np.isfinite(curve).any():raise ValueError('Zero-energy head should have undefined curve')
                table[key]=dict(curve=curve,valid=valid,total_energy=float(row['total_energy']),
                    ranks={str(t):int(row['rank_'+str(t)]) if row['rank_'+str(t)] else None for t in THRESHOLDS})
        expected={(mode,q+'_'+field,layer,head) for mode in MODES for q,_,_ in QUANTITIES
            for field in FIELDS for layer in range(1,11) for head in range(1,11)}
        if set(table)!=expected:raise ValueError('Every mode/quantity/layer/head must be present exactly once')
        data[edge]=dict(summary=summary,configuration=config,table=table)
        for path in (sp,folder/'configuration.json',folder/'row_manifest.json',folder/'per_head.csv',folder/'rank_curves.csv'):
            sources.append(dict(path=str(path.resolve()),sha256=sha(path)))
    if data[2]['configuration']['population_sha256']!=data[3]['configuration']['population_sha256']:
        raise ValueError('Both releases must use the same fixed resident population')
    return data,sources


def curves(data,edge,mode,field,quantity):
    return np.array([data[edge]['table'][mode,quantity+'_'+field,l,h]['curve'] for l in range(1,11) for h in range(1,11)])


def analyze(data):
    summaries={};rows=[]
    for mode in MODES:
        summaries[mode]={}
        for edge in (2,3):
            summaries[mode][str(edge)]={}
            for field in FIELDS:
                entry=dict(quantities={},paired_delta_vs_natives={},paired_required_rank={})
                matrices={q:curves(data,edge,mode,field,q) for q,_,_ in QUANTITIES}
                for quantity,_,_ in QUANTITIES:
                    matrix=matrices[quantity];rank_distributions={}
                    for threshold in THRESHOLDS:
                        values=np.array([data[edge]['table'][mode,quantity+'_'+field,l,h]['ranks'][str(threshold)]
                            for l in range(1,11) for h in range(1,11)],dtype=float)
                        rank_distributions[str(threshold)]=dict(distribution(values),
                            histogram={str(rank):int((values==rank).sum()) for rank in range(1,33)})
                    entry['quantities'][quantity]=dict(energy={str(rank):distribution(matrix[:,rank-1]) for rank in RANKS},
                        required_rank=rank_distributions)
                for rank in RANKS:
                    a,b,d=[matrices[q][:,rank-1] for q in ('source','rebuilt','delta')]
                    valid=np.isfinite(a)&np.isfinite(b)&np.isfinite(d);eps=1e-12
                    greater=valid&(d>a+eps)&(d>b+eps);lower=valid&(d<a-eps)&(d<b-eps)
                    tied=valid&(np.abs(d-a)<=eps)&(np.abs(d-b)<=eps)
                    entry['paired_delta_vs_natives'][str(rank)]=dict(valid_heads=int(valid.sum()),
                        delta_higher_than_both=int(greater.sum()),delta_lower_than_both=int(lower.sum()),
                        tied_with_both=int(tied.sum()),mixed_or_tied_with_one=int((valid&~greater&~lower&~tied).sum()),
                        rule='Strict paired energy comparison at the same rank, tolerance1e-12; higher energy means more concentration')
                for threshold in THRESHOLDS:
                    values={q:np.array([data[edge]['table'][mode,q+'_'+field,l,h]['ranks'][str(threshold)]
                        for l in range(1,11) for h in range(1,11)],dtype=float) for q,_,_ in QUANTITIES}
                    a,b,d=[values[q] for q in ('source','rebuilt','delta')];valid=np.isfinite(a)&np.isfinite(b)&np.isfinite(d)
                    smaller=valid&(d<a)&(d<b);larger=valid&(d>a)&(d>b);tied=valid&(d==a)&(d==b)
                    entry['paired_required_rank'][str(threshold)]=dict(valid_heads=int(valid.sum()),
                        delta_smaller_than_both=int(smaller.sum()),delta_larger_than_both=int(larger.sum()),
                        tied_with_both=int(tied.sum()),mixed_or_tied_with_one=int((valid&~smaller&~larger&~tied).sum()),
                        rule='Required rank for the same retained energy; smaller rank means more concentration')
                for pair in range(100):
                    row=dict(edge=edge,centering=mode,field=field,layer=pair//10+1,head=pair%10+1)
                    for quantity,_,_ in QUANTITIES:
                        item=data[edge]['table'][mode,quantity+'_'+field,pair//10+1,pair%10+1]
                        row[quantity+'_nonzero_energy']=item['valid']
                        for rank in RANKS:row[f'{quantity}_energy_r{rank}']=float(item['curve'][rank-1]) if item['valid'] else None
                        for threshold in THRESHOLDS:row[f'{quantity}_rank{int(threshold*100)}']=item['ranks'][str(threshold)]
                    rows.append(row)
                summaries[mode][str(edge)][field]=entry
    return summaries,rows


def figures(data,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    output.mkdir(parents=True,exist_ok=True);artifacts=[]
    def record(path):artifacts.append(dict(path=str(path.resolve()),sha256=sha(path)))
    for mode in MODES:
        fig,axes=plt.subplots(2,2,figsize=(13.8,8.2),sharex=True,sharey=True)
        for row,edge in enumerate((2,3)):
            residents=data[edge]['summary']['resident_users'];pooled=data[edge]['summary']['pooled_rows_per_head']
            for col,field in enumerate(FIELDS):
                ax=axes[row,col]
                for q,label,color in QUANTITIES:
                    matrix=curves(data,edge,mode,field,q);valid=int(np.isfinite(matrix[:,5]).sum())
                    ax.plot(np.arange(100),matrix[:,5],color=color,marker='.',markersize=3,lw=.75,
                        label=label+(f' ({valid}/100 nonzero)' if valid!=100 else ''))
                for boundary in np.arange(9.5,99,10):ax.axvline(boundary,color='.86',lw=.6)
                ax.set_xticks(np.arange(4.5,100,10),[str(l) for l in range(1,11)]);ax.set_xlim(-1,100)
                ax.set_ylim(0,1.015);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.15)
                ax.set_title(f'V{edge-1} → V{edge} · {field} · {residents:,} residents · {pooled:,} rows/head')
                ax.set_xlabel('Layer (all 10 heads per layer, in fixed head order)');ax.set_ylabel('Squared singular-value energy in first 6 directions')
                ax.legend(frameon=False,fontsize=8,loc='lower right')
        fig.suptitle(f'All 100 layer/head pairs · rank-6 energy · {mode}',fontsize=14,y=.985)
        fig.text(.5,.018,'All heads and all retained resident rows are included. K and V remain separate. Higher energy at fixed rank means greater concentration.\n'
            'These are oracle pooled-row geometries; they do not establish a learned code dimension or preservation of task quality.',ha='center',fontsize=8,color='.35')
        fig.subplots_adjust(top=.90,bottom=.12,left=.07,right=.985,hspace=.32,wspace=.16)
        for extension in ('png','pdf'):
            path=output/f'all_heads_r6_{mode}.{extension}';fig.savefig(path,dpi=220,bbox_inches='tight');record(path)
        plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(13.8,8.2),sharex=True,sharey=True)
        for row,edge in enumerate((2,3)):
            for col,field in enumerate(FIELDS):
                ax=axes[row,col];delta=curves(data,edge,mode,field,'delta')[:,5]
                ax.axhline(0,color='.4',lw=.9)
                for quantity,label,color in QUANTITIES[:2]:
                    difference=100*(delta-curves(data,edge,mode,field,quantity)[:,5])
                    ax.plot(np.arange(100),difference,color=color,marker='.',markersize=3,lw=.75,
                        label='Delta − '+label.lower())
                for boundary in np.arange(9.5,99,10):ax.axvline(boundary,color='.86',lw=.6)
                ax.set_xticks(np.arange(4.5,100,10),[str(l) for l in range(1,11)]);ax.set_xlim(-1,100)
                ax.grid(axis='y',alpha=.15);ax.set_title(f'V{edge-1} → V{edge} · {field} · all 100 layer/head pairs')
                ax.set_xlabel('Layer (all 10 heads per layer, in fixed head order)')
                ax.set_ylabel('Delta − native rank-6 energy (percentage points)')
                ax.legend(frameon=False,fontsize=8,loc='lower right')
        fig.suptitle(f'Paired rank-6 concentration differences · {mode}',fontsize=14,y=.985)
        fig.text(.5,.018,'Every head is paired with its own native references. Positive: delta has more energy in six directions; negative: the native reference has more.\n'
            'All 100 head positions are retained, including opposite signs; paired higher/lower/tie counts are recorded in analysis_summary.json.',ha='center',fontsize=8,color='.35')
        fig.subplots_adjust(top=.90,bottom=.12,left=.075,right=.985,hspace=.32,wspace=.17)
        for extension in ('png','pdf'):
            path=output/f'paired_r6_differences_{mode}.{extension}';fig.savefig(path,dpi=220,bbox_inches='tight');record(path)
        plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11.5,8.0),sharex=True,sharey=True)
        for row,edge in enumerate((2,3)):
            for col,field in enumerate(FIELDS):
                ax=axes[row,col]
                for q,label,color in QUANTITIES:
                    matrix=curves(data,edge,mode,field,q);keep=np.isfinite(matrix).all(1)
                    if not keep.any():continue
                    low,median,high=np.quantile(matrix[keep],[.1,.5,.9],axis=0)
                    ax.fill_between(np.arange(1,33),low,high,color=color,alpha=.12,lw=0)
                    ax.plot(np.arange(1,33),median,color=color,lw=1.5,label=label)
                ax.set_title(f'V{edge-1} → V{edge} · {field} · all 100 layer/head pairs')
                ax.set_xticks([1,4,6,8,16,24,32]);ax.set_xlim(1,32);ax.set_ylim(0,1.015)
                ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.15)
                ax.set_xlabel('Number of directions');ax.set_ylabel('Cumulative squared singular-value energy')
                ax.legend(frameon=False,fontsize=8,loc='lower right')
        fig.suptitle(f'All-head median spectra · {mode}',fontsize=14,y=.985)
        fig.text(.5,.018,'Lines: median across nonzero-energy heads. Bands: head q10–q90, descriptive heterogeneity rather than confidence intervals.\n'
            'The accompanying all-head plots, CSV and layer PDFs retain every individual head; no spectral outcome selects a head or user.',ha='center',fontsize=8,color='.35')
        fig.subplots_adjust(top=.90,bottom=.12,left=.075,right=.985,hspace=.32,wspace=.15)
        for extension in ('png','pdf'):
            path=output/f'median_rank_curves_{mode}.{extension}';fig.savefig(path,dpi=220,bbox_inches='tight');record(path)
        plt.close(fig)
        path=output/f'all_layer_head_curves_{mode}.pdf'
        with PdfPages(path) as pdf:
            for edge in (2,3):
                for field in FIELDS:
                    for layer in range(1,11):
                        fig,axes=plt.subplots(2,5,figsize=(14.2,6.4),sharex=True,sharey=True)
                        for head,ax in enumerate(axes.flat,start=1):
                            for q,label,color in QUANTITIES:
                                item=data[edge]['table'][mode,q+'_'+field,layer,head]
                                if item['valid']:ax.plot(np.arange(1,33),item['curve'],color=color,lw=1.1,label=label)
                                else:ax.text(.04,.12+.08*[v[0] for v in QUANTITIES].index(q),label+': zero energy',transform=ax.transAxes,fontsize=7,color=color)
                            ax.set_title(f'Head {head}');ax.set_xlim(1,32);ax.set_ylim(0,1.015)
                            ax.set_xticks([1,6,16,32]);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.15)
                            if head>5:ax.set_xlabel('Directions')
                            if head in (1,6):ax.set_ylabel('Energy')
                        handles,labels=axes[0,0].get_legend_handles_labels()
                        fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.015))
                        fig.suptitle(f'V{edge-1} → V{edge} · {field} · layer {layer} · all 10 heads · {mode}',fontsize=13)
                        fig.subplots_adjust(top=.85,bottom=.17,left=.055,right=.99,hspace=.35,wspace=.20)
                        pdf.savefig(fig);plt.close(fig)
        record(path)
    return artifacts,matplotlib.__version__


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT);args=p.parse_args()
    data,sources=load_full(args.root);summary,head_rows=analyze(data)
    output=args.root/'B';output.mkdir(exist_ok=True)
    per_head=output/'analysis_per_head.csv'
    with per_head.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(head_rows[0]));writer.writeheader();writer.writerows(head_rows)
    artifacts,version=figures(data,output/'figures');artifacts.append(dict(path=str(per_head.resolve()),sha256=sha(per_head)))
    result=dict(format='kv_delta_spectrum_analysis_v1',population_sha256=data[2]['configuration']['population_sha256'],
        resident_users=data[2]['summary']['resident_users'],all_heads=100,summary=summary,sources=sources,
        plotting_source_sha256=sha(__file__),matplotlib_version=version,artifacts=artifacts,
        limitations=['Every layer/head is retained; no head or resident is selected using spectral outcomes.',
            'Input matrices pool all retained rows, so active/longer histories contribute more rows; these are not equal-UID covariance spectra.',
            'Uncentered is primary. Centered removes one pooled-row mean per layer/head, not each user mean.',
            'Head q10/median/q90 are descriptive heterogeneity, not confidence intervals or independent repetitions.',
            'A zero-energy head has undefined normalized energy and required rank; it is listed explicitly and is not silently assigned rank1.',
            'Concentration of native version delta cannot alone prove a deployable projection, correction-code dimension, head skipping, or task-quality preservation.'])
    lines=['# Native/delta spectra across every head','','Full-population pooled-row geometry; all 100 layer/head pairs are shown before aggregation.','',
        '| Centering | Release | Field | Quantity | Energy r4 q10/med/q90 | r6 | r8 | Rank80 q10/med/q90 | Rank90 | Rank95 |',
        '|---|---|---|---|---|---|---|---|---|---|']
    def fmt(d,scale=1):return 'NA' if d['median'] is None else '/'.join(f'{d[k]*scale:.2f}' for k in ('q10','median','q90'))
    for mode in MODES:
        for edge in (2,3):
            for field in FIELDS:
                entry=summary[mode][str(edge)][field]
                for q,_,_ in QUANTITIES:
                    value=entry['quantities'][q]
                    cells=[mode,f'V{edge-1}→V{edge}',field,q,*[fmt(value['energy'][str(r)],100)+'%' for r in RANKS],
                        *[fmt(value['required_rank'][str(t)]) for t in THRESHOLDS]]
                    lines.append('| '+' | '.join(cells)+' |')
    lines+=['','## Paired head counts','','All comparisons use the same head and rank. Counts are descriptive; no favorable direction is assumed.','',
        '| Centering | Release | Field | Rank | Valid | Delta higher than both | Lower than both | Tied with both | Mixed/one tied |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        for edge in (2,3):
            for field in FIELDS:
                for rank,value in summary[mode][str(edge)][field]['paired_delta_vs_natives'].items():
                    lines.append('| '+' | '.join(map(str,[mode,f'V{edge-1}→V{edge}',field,rank,value['valid_heads'],
                        value['delta_higher_than_both'],value['delta_lower_than_both'],value['tied_with_both'],value['mixed_or_tied_with_one']]))+' |')
    lines+=['','## Paired required-rank counts','','Smaller required rank at the same retained energy means greater concentration.','',
        '| Centering | Release | Field | Retained energy | Valid | Delta rank smaller than both | Larger than both | Tied with both | Mixed/one tied |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        for edge in (2,3):
            for field in FIELDS:
                for threshold,value in summary[mode][str(edge)][field]['paired_required_rank'].items():
                    lines.append('| '+' | '.join(map(str,[mode,f'V{edge-1}→V{edge}',field,threshold,value['valid_heads'],
                        value['delta_smaller_than_both'],value['delta_larger_than_both'],value['tied_with_both'],value['mixed_or_tied_with_one']]))+' |')
    lines+=['','Complete rank80/90/95 histograms (ranks1–32) and all min/max values are in analysis_summary.json. Per-head values are in analysis_per_head.csv.','',
        *['- '+text for text in result['limitations']]]
    (output/'analysis_summary.md').write_text('\n'.join(lines)+'\n')
    (output/'analysis_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',artifacts=artifacts,resident_users=result['resident_users'],all_heads=100)))


if __name__=='__main__':main()
