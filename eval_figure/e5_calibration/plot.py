"""SIMULATED DATA: discrete calibration quality/cost tradeoffs.

Borrow the paired scatter-panel structure of gallery oa-geo-esd-17-843-2026-f08;
do not borrow its regression lines or statistical interpretation. Current JSON
is illustrative, with three fitting sizes per independently calibrated model.
Simulation is disclosed by the manuscript caption and export metadata.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from style import DATASET_COLORS,finish,load_data
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    d=load_data(__file__)
    plt.rcParams.update({'font.size':7,'axes.titlesize':7.3,'axes.labelsize':7.3,
                         'xtick.labelsize':7,'ytick.labelsize':7})
    fig,axes=plt.subplots(1,2,figsize=(3.35,1.85),sharex=True,sharey=True)
    fig.subplots_adjust(left=.15,right=.985,top=.70,bottom=.24,wspace=.25)
    marks=['^','o','s']
    for ax,dataset,letter in zip(axes,['Yambda','RecFlow'],'ab'):
        color=DATASET_COLORS[dataset]
        for scale in ['Large','Max']:
            name=f'{dataset}-{scale}'
            for j,(x,y) in enumerate(zip(d['incremental_fit_seconds'][name],d['recovery_percent'][name])):
                ax.scatter(x,y,marker=marks[j],s=22,facecolors=color if scale=='Large' else 'white',
                           edgecolors=color,linewidths=.9,zorder=3)
        ax.set_title(f'({letter}) {dataset}',loc='left',pad=4)
        ax.set(xlim=(0,2050),ylim=(45,95),xticks=[0,1000,2000],xticklabels=['0','1','2'],yticks=[50,70,90])
        ax.grid(axis='y',lw=.4);ax.tick_params(length=2,pad=2)
        ax.set_xlabel('Fit time (ks)',labelpad=2)
    axes[0].set_ylabel('Recovery (%)',labelpad=2)
    handles=[Line2D([],[],ls='',marker=m,color='#505961',ms=3.5,label=f'{n} users') for m,n in zip(marks,d['fit_users'])]
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.56,1.015),
               fontsize=6.6,columnspacing=.8,handletextpad=.4,handlelength=1,labelspacing=.4)
    scales=[Line2D([],[],ls='',marker='o',markerfacecolor=c,markeredgecolor='#505961',ms=3.5,label=lab)
            for c,lab in [('#505961','Large'),('white','Max')]]
    fig.legend(handles=scales,ncol=2,loc='upper center',bbox_to_anchor=(.56,.90),
               fontsize=6.6,columnspacing=1.4,handletextpad=.4,handlelength=1)
    finish(fig,__file__,'6-5','Calibration tradeoff',paper_column=True,layout=False)


if __name__=='__main__':main()
