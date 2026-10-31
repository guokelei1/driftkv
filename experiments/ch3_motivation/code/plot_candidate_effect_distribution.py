#!/usr/bin/env python3
"""Untrimmed empirical distributions for one frozen calibration; CPU only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'experiments/ch3_motivation/runs/simple_config_sweep/attempt_1/candidate_distribution'
font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
plt.rcParams.update({'font.family':['Droid Sans Fallback','DejaVu Sans'],'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
data={e:np.load(RUN/f'events_edge{e}.npz') for e in (2,3)}
lo=min(d['delta_log_loss'].min() for d in data.values());hi=max(d['delta_log_loss'].max() for d in data.values())
fig,axes=plt.subplots(1,2,figsize=(12.2,4.8),sharex=True,sharey=True)
for ax,e in zip(axes,(2,3)):
 d=data[e];x=d['delta_log_loss'];m=d['positive_multi_candidate_context'].astype(bool)
 ax.axvspan(lo,0,color='#fae2df',alpha=.45);ax.axvspan(0,hi,color='#e1f0e8',alpha=.45)
 for values,label,color,style in [(x,'全部真实反馈','#2863A5','-'),(x[m],'同一用户、同一时间的正反馈候选','#C67A29','--')]:
  ordered=np.sort(values);ax.step(ordered,100*np.arange(1,len(ordered)+1)/len(ordered),where='post',label=label,color=color,lw=1.9,ls=style)
 ax.axvline(0,color='#333333',lw=1);ax.set_xlim(lo,hi);ax.set_ylim(0,102);ax.grid(axis='y',alpha=.22)
 n=len(np.unique(d['uid']));nc=len(np.unique(d['uid'][m]));ax.set_title(f'V{e-1} → V{e}：{n:,} 名反馈用户\n同一时间正反馈控制：{nc:,} 人')
 fractions=[100*np.mean(x<-.001),100*np.mean(np.abs(x)<=.001),100*np.mean(x>.001)]
 ax.text(.98,.28,'受损 / 近零 / 改善\n'+' / '.join(f'{v:.1f}%' for v in fractions),transform=ax.transAxes,ha='right',va='top',bbox={'facecolor':'white','alpha':.9,'edgecolor':'#dddddd'},fontsize=10)
 ax.set_xlabel('预测损失改善（Reuse loss − Map loss，nat）')
axes[0].set_ylabel('累计反馈比例（%）');axes[0].legend(loc='upper left',bbox_to_anchor=(0,-.22),frameon=False,fontsize=9)
fig.suptitle('每条发布边固定校准：不同反馈候选的损益分布',fontsize=15,y=.99)
fig.text(.995,.055,'正值：损失降低；负值：损失增加。近零定义为绝对变化 ≤ 0.001 nat。',ha='right',fontsize=9,color='#555555')
fig.text(.995,.015,'每条真实反馈一个点，完整范围未裁尾；不是单个候选的 AUC，也不代表稳定候选因果效应。',ha='right',fontsize=9,color='#555555')
fig.subplots_adjust(top=.80,bottom=.24,wspace=.12,left=.065,right=.98)
fig.savefig(RUN/'candidate_effect_distribution.png',dpi=180);fig.savefig(RUN/'candidate_effect_distribution.pdf')
print(str(RUN/'candidate_effect_distribution.png'))
