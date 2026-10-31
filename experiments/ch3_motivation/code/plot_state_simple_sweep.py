#!/usr/bin/env python3
"""All prespecified width/budget contrasts with pointwise paired UID intervals."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'experiments/ch3_motivation/runs/simple_config_sweep/attempt_1'
rows=json.loads((RUN/'aggregate/summary.json').read_text())['grid']
if len(rows)!=12:raise ValueError('All six recipes and two endpoints required')
font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
plt.rcParams.update({'font.family':['Droid Sans Fallback','DejaVu Sans'],'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(10.7,5.1),sharey=True)
for ax,t in zip(axes,[3,4]):
 for steps,color,offset in [(256,'#2863A5',-.055),(1024,'#C67A29',.055)]:
  selected=[next(r for r in rows if r['final_target']==t and r['steps_per_stage']==steps and r['width']==w) for w in [32,64,128]]
  xs=[x+offset for x in [0,1,2]];ys=[100*r['direct_minus_chain_AUC'] for r in selected]
  low=[100*r['direct_minus_chain_AUC_ci_low'] for r in selected];hi=[100*r['direct_minus_chain_AUC_ci_high'] for r in selected]
  ax.plot(xs,ys,marker='o',color=color,lw=1.7,label=f'{steps} 步／每次映射')
  ax.vlines(xs,low,hi,color=color,lw=1.2)
  for x,a,b in zip(xs,low,hi):ax.hlines([a,b],x-.035,x+.035,color=color,lw=1.2)
 ax.axhline(0,color='#555555',ls='--',lw=1);ax.grid(axis='y',alpha=.23)
 ax.set_xticks([0,1,2],['32','64','128']);ax.set_xlabel('映射宽度');ax.set_title(f'V1 → V{t}（{949 if t==3 else 932} 名反馈用户）')
axes[0].set_ylabel('AUC 差：直接映射 − 多次映射（百分点）');axes[0].legend(frameon=False,loc='upper left')
fig.suptitle('同一批用户：容量与训练预算改变映射路径的差异',fontsize=14,y=.98)
fig.text(.5,.055,'正值：直接映射更好；负值：多次映射更好。误差条：500 次配对 UID 的逐点 95% 区间。',ha='center',fontsize=9)
fig.text(.5,.015,'全部六组均展示；每次映射同预算，链式累计预算更多。开发探索，未作独立确认。',ha='center',fontsize=9,color='#555555')
fig.subplots_adjust(top=.80,bottom=.19,left=.08,right=.98,wspace=.13)
fig.savefig(RUN/'aggregate/state_simple_sweep.png',dpi=180);fig.savefig(RUN/'aggregate/state_simple_sweep.pdf')
print(str(RUN/'aggregate/state_simple_sweep.png'))
