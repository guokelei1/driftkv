"""Basic method paper artifacts; read frozen evidence only, no model execution."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results/design/analysis'
PDF=ROOT/'figures/pic/pdf'
PREVIEW=ROOT/'figures/pic/jpg'
TABLE=ROOT/'figures/tables'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})


def save(fig,name):
    PDF.mkdir(parents=True,exist_ok=True); PREVIEW.mkdir(parents=True,exist_ok=True)
    fig.savefig(PDF/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(PREVIEW/f'{name}.png',dpi=170,bbox_inches='tight')
    plt.close(fig)


def mechanism():
    records=json.loads((DATA/'base_method_final_01/representation.json').read_text())
    rows=[r for r in records if r['group']=='diagnostic_uid' and r['scope']=='all_states' and r['layer']=='all_layers']
    rows=sorted(rows,key=lambda r:r['target'])
    fig,ax=plt.subplots(figsize=(6.4,3.4))
    x=np.arange(4); width=.24
    for j,(name,label,color) in enumerate([('reuse','Uncorrected response','#afb7bf'),('constant','Constant','#d8843c'),('affine','Query affine','#277eac')]):
        vals=np.array([100 if name=='reuse' else 100*r[f'{name}_aggregate_ratio'] for r in rows])
        ax.bar(x+(j-1)*width,vals,width,color=color,label=label)
        if name!='reuse':
            for xx,v in zip(x+(j-1)*width,vals,strict=True):
                ax.text(xx,v*1.35,f'{v:.4f}' if name=='affine' else f'{v:.2f}',ha='center',fontsize=8)
    ax.set(yscale='log',ylim=(.001,230),xticks=x,xticklabels=[f"M{r['target']}" for r in rows],ylabel='Held-out residual / response energy (%)')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.15),ncol=3,frameon=False,fontsize=8)
    ax.grid(axis='y',which='major',alpha=.2); ax.set_axisbelow(True)
    fig.text(.5,-.025,'128 reused diagnostic UIDs; fit64 / held64; common affine-lower-context queries',ha='center',fontsize=8)
    save(fig,'design1_mechanism')


def flow():
    fig,ax=plt.subplots(figsize=(11.5,3.5)); ax.set(xlim=(0,12),ylim=(0,3.7)); ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05,rounding_size=0.08',facecolor=color,edgecolor='#536371',lw=1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,linespacing=1.4)
    def arrow(a,b,color='#526878',style='-'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.4,color=color,linestyle=style))
    box(.2,2.8,3.2,.7,'Calibration users only\nActual / Current teacher KV','#f9ebd7')
    box(4.3,2.8,3.1,.7,'Release translation\nShared projection, W and T','#f9ebd7')
    arrow((3.45,3.15),(4.23,3.15))
    box(.2,1.3,2.4,.7,'Actual ordinary KV\nReal producer lineage','#e2eef4')
    box(.2,.1,2.4,.65,'Native maintenance\nAppend / evict','#e5f0df')
    arrow((1.4,.8),(1.4,1.23))
    box(3.25,.1,2.6,.65,'Cache summary\nK/V sums, count, age','#e5f0df')
    arrow((2.65,.425),(3.18,.425))
    box(6.6,.1,2.6,.65,'User functional view\nPer-layer b and A','#e5f0df')
    arrow((5.9,.425),(6.53,.425))
    ax.text(6.2,.97,'publish / dirty refresh',ha='center',fontsize=8)
    arrow((7.25,2.75),(7.8,2.07),style='--')
    ax.text(7.65,2.49,'T',fontsize=9)
    ax.plot([7.45,9.55,9.55],[3.18,3.18,.425],ls='--',lw=1.2,color='#526878')
    arrow((9.55,.425),(9.26,.425),style='--')
    ax.text(8.5,3.32,'W → view',fontsize=9,ha='center')
    box(3.25,1.3,2.6,.7,'Actual query q\nHistory response r','#e2eef4')
    arrow((2.65,1.65),(3.18,1.65))
    box(6.6,1.3,2.6,.7,'Read correction: r + m c\nc = T r + b + A q','#e2eef4')
    arrow((5.9,1.65),(6.53,1.65)); arrow((7.9,.8),(7.9,1.23))
    box(9.85,1.3,1.9,.7,'Self + layer output\nNext query / score','#e2eef4')
    arrow((9.25,1.65),(9.78,1.65))
    ax.text(.25,2.43,'SHARED PREPARATION',fontsize=9,weight='bold',color='#7d592b')
    ax.text(.25,2.24,'POPULATION EXECUTION — no teacher access',fontsize=9,weight='bold',color='#2c586e')
    save(fig,'design1_flow')


def scale():
    data=json.loads((DATA/'native_flops_01/ledger.json').read_text())['native']
    u=np.logspace(3,6.2,250); exact=u*data['e']/1e12; method=(data['K']+u*data['a'])/1e12
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.25))
    ax=axes[0]; ax.loglog(u,exact,label='Exact rebuild: Ue',color='#697681',lw=2)
    ax.loglog(u,method,label='Adapter: K + Ua',color='#277eac',lw=2)
    ax.axhline(data['K']/1e12,color='#d8843c',linestyle=':',label='Shared preparation K')
    for r in data['scales'][1:]:
        ax.scatter(r['users'],r['method_TF'],color='#277eac',s=23,zorder=4)
    ax.set(xlabel='Mature cached users (U)',ylabel='Incremental computation (TFLOPs)'); ax.legend(frameon=False,fontsize=8)
    ax=axes[1]; ax.semilogx(u,method/exact*100,color='#277eac',lw=2)
    ax.axhline(100,color='#697681',ls='--',lw=1,label='Lifecycle break-even')
    for r in data['scales'][1:]:
        v=r['life_ratio']*100; ax.scatter(r['users'],v,color='#277eac',s=24,zorder=4)
        ax.annotate(f"{r['users']//1000:,}k: {v:.2f}%",(r['users'],v),xytext=(0,11),textcoords='offset points',ha='center',fontsize=8)
    ax.set(xlabel='Mature cached users (U)',ylabel='Adapter / Exact increment (%)',ylim=(0,115),xlim=(4000,1.6e6)); ax.legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2,which='major')
    fig.tight_layout(w_pad=2)
    fig.text(.5,-.07,'Four updates; N=1024; fixed 256-user calibration; 21.72 candidate scores/user across four windows\nQuality measured on 4,091 users; larger populations are FLOPs extrapolations',ha='center',fontsize=8)
    save(fig,'design1_scale')


def quality_table():
    TABLE.mkdir(parents=True,exist_ok=True)
    q=json.loads((DATA/'native_base_quality4091_01_report/summary.json').read_text())['quality']
    values={(r['target'],r['method']):r['metrics']['ROC_AUC'] for r in q}
    methods=('reuse','exact','native','summary_input','no_source'); targets=(1,3,4,5)
    boot=json.loads((DATA/'native_flops_01/quality_bootstrap.json').read_text())['differences']
    lines=[r'\begin{table*}[t]',r'\centering\small',r'\caption{Basic-domain quality: all 4,091 development users with $n_{M0}\ge1024$, four independent adjacent migrations, and 88,860 real feedback requests. AUC is pooled within each edge; summary rows average edge-wise differences, not mixed-version scores. Intervals are paired-UID descriptive 95\% intervals at one backbone seed.}',r'\label{tab:design1-quality}',r'\begin{tabular}{lrrrrr}',r'\toprule',r'Target / contrast & Reuse & Current Exact & Native & Summary input & No source \\',r'\midrule']
    for t in targets:
        lines.append(f'M{t} & '+' & '.join(f'{values[t,m]:.7f}' for m in methods)+r' \\')
    lines += [r'\midrule']
    for right,label in [('reuse',r'Mean $\Delta$AUC vs. Reuse (pp)'),('exact',r'Mean $\Delta$AUC vs. Exact (pp)')]:
        lines.append(label+' & '+' & '.join(f'{100*np.mean([values[t,m]-values[t,right] for t in targets]):+.4f}' for m in methods)+r' \\')
        ints=['---','---']
        for m in methods[2:]:
            r=next(x for x in boot if x['target']=='equal_edge' and x['left']==m and x['right']==right)
            ints.append('['+', '.join(f'{100*v:+.4f}' for v in r['interval'])+']')
        lines.append(r'95\% interval (pp) & '+' & '.join(ints)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (TABLE/'design1_quality.tex').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    mechanism(); flow(); scale(); quality_table()
