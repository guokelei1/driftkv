"""SIMULATED DATA: table preview; main.tex uses native LaTeX tables."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from style import finish,load_data
import matplotlib.pyplot as plt
import numpy as np

def main():
    d=load_data(__file__);fig,ax=plt.subplots(figsize=(3.35,1.9));ax.axis('off')
    fig.subplots_adjust(left=.03,right=.98,bottom=.02,top=.98)
    rows=[]
    for name,q in d['release_quality'].items():
        base=np.mean(q['reuse']); st=100*(np.mean(q['constant_gate'])-base);dy=100*(np.mean(q['ct'])-base)
        rows.append([name,f'{st:.2f}',f'{dy:.2f}'])
    t=ax.table(cellText=rows,colLabels=['Configuration','Const. gain','Dynamic gain'],bbox=[0,.52,1,.48],colWidths=[.46,.27,.27],cellLoc='center')
    t.auto_set_font_size(False);t.set_fontsize(6.7)
    rows=[[s['short_label'],f"{s['auc_gain_pp']:.2f}",str(s['added_read_flops_pct']),f"{s['code_bytes_pct_native_kv']:.2f}"] for s in d['structures']]
    t2=ax.table(cellText=rows,colLabels=['Store/read','Gain (pp)','Read +%','Code %'],bbox=[0,.02,1,.36],cellLoc='center')
    t2.auto_set_font_size(False);t2.set_fontsize(6.7)
    for table in [t,t2]:
        for (r,c),cell in table.get_celld().items():
            cell.set_linewidth(.35);cell.set_edgecolor('#d0d6dc')
            if r==0:cell.set_facecolor('#e8eef3')
    finish(fig,__file__,'6-4','Reading controls (table)',note='AUC gains over Reuse; lower block: Yambda-L.',paper_column=True,layout=False)
if __name__=='__main__':main()
