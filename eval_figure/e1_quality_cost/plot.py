"""SIMULATED DATA: twelve release frontiers on a double-column canvas.

Rows are datasets. Within each row, Large and Max each occupy three release
columns. All values remain explicitly synthetic; do not tune real baselines
to conform to these illustrative shapes. Paper captions identify simulation.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from style import COLORS, MARKERS, finish, load_data
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


def main():
    d=load_data(__file__)
    plt.rcParams.update({'font.size':7, 'axes.titlesize':7.2,'axes.labelsize':7.3,
        'xtick.labelsize':6.5,'ytick.labelsize':6.3,'legend.fontsize':7})
    fig=plt.figure(figsize=(7.0,2.95))
    # A slightly wider gap separates the two model scales within each dataset.
    outer=fig.add_gridspec(2,2,left=.075,right=.994,top=.80,bottom=.17,
                          wspace=.16,hspace=.55)
    linestyles={'CacheTide':'-', 'Static map':'--',
                'Partial rebuild':'-.', 'Budget rebuild':':'}
    for cfg in d['configurations']:
        row=0 if cfg['dataset']=='Yambda' else 1
        col=0 if cfg['scale']=='Large' else 1
        grid=outer[row,col].subgridspec(1,3,wspace=.17)
        panels=cfg['release_panels']
        lo=min(p['reuse_auc'] for p in panels)
        hi=max(p['rebuild_auc'] for p in panels)
        span=hi-lo
        shared=None
        for i,p in enumerate(panels):
            ax=fig.add_subplot(grid[0,i],sharey=shared)
            if shared is None:shared=ax
            for method in ['CacheTide','Static map','Partial rebuild','Budget rebuild']:
                x,y=zip(*p['curves'][method])
                ax.plot(x,y,color=COLORS[method],marker=MARKERS[method],
                        linestyle=linestyles[method],lw=1.0,ms=2.4,mew=.35)
            # Open circle versus filled square remains distinct in grayscale.
            ax.plot(0,p['reuse_auc'],marker='o',ls='',mfc='white',mec='#555D65',mew=.8,ms=3.5,zorder=5)
            ax.plot(100,p['rebuild_auc'],marker='s',ls='',mfc=COLORS['Rebuild'],mec=COLORS['Rebuild'],ms=3.1,zorder=5)
            ax.axhline(p['reuse_auc'],color=COLORS['Reuse'],ls=':',lw=.4)
            ax.axhline(p['rebuild_auc'],color=COLORS['Rebuild'],ls=':',lw=.4)
            x,y=p['curves']['CacheTide'][d['meta']['default_point_index']]
            ax.scatter(x,y,s=27,facecolors='none',edgecolors=COLORS['CacheTide'],lw=.6,zorder=6)
            ax.set_title(p['release'],pad=2)
            ax.set(xlim=(-5,105),ylim=(lo-.08*span,hi+.08*span),xticks=[0,50,100])
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.tick_params(length=2,pad=1.3)
            if i:ax.tick_params(axis='y',labelleft=False)
            ax.grid(axis='y',lw=.35)
            if not col and not i:ax.set_ylabel(f"{cfg['dataset']} AUC",labelpad=3)
    fig.text(.285,.88,'Large',ha='center',va='center',fontsize=8)
    fig.text(.77,.88,'Max',ha='center',va='center',fontsize=8)
    fig.text(.53,.025,'Complete added FLOPs / Rebuild budget (%)',ha='center',fontsize=7.5)
    handles=[Line2D([],[],color=COLORS[m],marker=MARKERS[m],ls=linestyles[m],lw=1.1,ms=3,label=m)
             for m in linestyles]
    handles.extend([
        Line2D([],[],color='#555D65',marker='o',mfc='white',mec='#555D65',ls='',ms=3.5,label='Reuse'),
        Line2D([],[],color=COLORS['Rebuild'],marker='s',ls='',ms=3.3,label='Rebuild')])
    fig.legend(handles=handles,ncol=6,loc='upper center',bbox_to_anchor=(.54,1.005),
               handlelength=1.6,handletextpad=.4,columnspacing=1.3,fontsize=7)
    finish(fig,__file__,'6-1','Twelve per-release quality-cost frontiers',layout=False)


if __name__=='__main__':main()
