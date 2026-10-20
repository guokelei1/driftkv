"""SIMULATED DATA: jointly show five-release cost and recovery for four groups.

One percent axis, shallow cost bars and a recovery line at each release.
The two percentages have different explicit denominators; their vertical
separation is not an additional metric. Delta labels give absolute AUC gaps.
The paper caption and file metadata disclose the synthetic values.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from style import COLORS,finish,load_data
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


def main():
    d=load_data(__file__)['lifecycle_preview']
    versions=d['release_labels'];xs=np.arange(len(versions))
    plt.rcParams.update({'font.size':7,'axes.titlesize':7.3,'axes.labelsize':7.1,
                         'xtick.labelsize':6.7,'ytick.labelsize':6.7})
    fig,axes=plt.subplots(2,2,figsize=(3.35,3.05),sharey=True)
    fig.subplots_adjust(left=.14,right=.99,top=.87,bottom=.13,wspace=.22,hspace=.63)
    barcolor='#B7CDDD';edge='#7996AE';line=COLORS['CacheTide']
    for i,(ax,series) in enumerate(zip(axes.flat,d['series'])):
        auc=series['auc'];gap_auc=np.array(auc['rebuild'])-np.array(auc['reuse'])
        gap=100*gap_auc
        recovery=100*(np.array(auc['ct'])-np.array(auc['reuse']))/gap_auc
        cost=np.array(series['added_flops_percent'])
        ax.bar(xs,cost,width=.55,color=barcolor,edgecolor=edge,lw=.45,hatch='///',zorder=2)
        ax.plot(xs,recovery,color=line,marker='o',ms=2.7,lw=1.2,zorder=4)
        # Annotate absolute quality gaps, not a third time series or lower table.
        for x,r,g,c in zip(xs,recovery,gap,cost):
            ax.annotate(f'Δ{g:.1f}',(x,r),xytext=(0,5),textcoords='offset points',
                        ha='center',va='bottom',fontsize=6.1,color='#65707A')
            ax.annotate(f'{c:.0f}',(x,c),xytext=(0,2),textcoords='offset points',
                        ha='center',va='bottom',fontsize=6.1,color='#4A5D6D')
        ax.set_title(f'({chr(97+i)}) {series["configuration"]}',loc='left',pad=4)
        ax.set(xlim=(-.55,len(xs)-.45),ylim=(0,112),xticks=xs,
               xticklabels=versions,yticks=[0,25,50,75,100])
        ax.tick_params(length=2,pad=2);ax.grid(axis='y',lw=.4)
        if i>=2:ax.set_xlabel('Release',labelpad=2)
        if i%2==0:ax.set_ylabel('Recovery / cost (%)',labelpad=2)
    handles=[Line2D([],[],color=line,marker='o',ms=3,lw=1.2,label='Recovery'),
             Patch(facecolor=barcolor,edgecolor=edge,hatch='///',lw=.45,label='Added cost'),
             Line2D([],[],color='none',ls='',label='Δ AUC gap (pp)')]
    legend=fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.54,1.005),
                      fontsize=6.6,handlelength=1.3,columnspacing=.8,handletextpad=.4)
    legend.get_texts()[2].set_color('#65707A')
    finish(fig,__file__,'6-3','Recovery and cost through five releases',layout=False,paper_column=True)


if __name__=='__main__':main()
