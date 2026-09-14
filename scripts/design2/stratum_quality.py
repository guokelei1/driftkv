"""Real global AUC and explicitly secondary within-UID AUC, fixed strata."""
import json
import numpy as np,pandas as pd
from design.data import ROOT
from design2.report_sparse import load,RUN
from design2.report_scale_followup import boot,TARGETS
from design2.audit_benchmark import save
from hstu_kvcache.evaluation.binary_metrics import binary_metrics

STRATA=['all','short','old_lineage','mixed','rare_activity','challenge','routine']

def quality(frame,meta,uids,strata=STRATA):
    f=frame.merge(meta,on=['uid','target'],validate='many_to_one');f['all']=True;out=[]
    for name in strata:
        part=f[f[name]];intervals=boot(part,uids) if len(part) else []
        for t,g in part.groupby('target'):
            metrics={m:binary_metrics(g.label,g[m]) for m in ['reuse','design1','design2','exact']}
            a=metrics['design1']['ROC_AUC'];b=metrics['design2']['ROC_AUC'];pairs=[]
            for uid,uu in g.groupby('uid'):
                if uu.label.nunique()!=2:continue
                pairs.append([binary_metrics(uu.label,uu[m])['ROC_AUC'] for m in ['design1','design2']])
            pair=np.array(pairs);support=dict(uids=int(g.uid.nunique()),rows=len(g),positive=int(g.label.sum()),negative=int((1-g.label).sum()))
            within=dict(users=len(pair),scope='Secondary equal-UID mean AUC among UID/target windows containing both classes; coverage reported, not a replacement for global AUC')
            if len(pair):
                delta=pair[:,1]-pair[:,0];rng=np.random.default_rng(17)
                vals=np.array([delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(1000)])
                within.update(design1=float(pair[:,0].mean()),design2=float(pair[:,1].mean()),difference=float(delta.mean()),interval=np.quantile(vals,[.025,.975]).tolist())
            out.append(dict(stratum=name,target=int(t),**support,metrics=metrics,auc_difference=None if a is None or b is None else b-a,
                auc_interval=next((x['interval'] for x in intervals if x['target']==t),None),
                supported=support['uids']>=30 and min(support['positive'],support['negative'])>=20,within_uid_auc=within))
    return out

def previous():
    f,_,_=load(RUN/'excursion_cap1024_2048');meta=pd.read_parquet(ROOT/'results/design2/analysis/scale_followup_01/members.parquet');meta=meta[meta.cap==1024]
    ids=json.loads((ROOT/'configs/design2/scale_followup_01_uids.json').read_text());rows=quality(f,meta,ids['original']+ids['extension'])
    out=ROOT/'results/design2/scan_30k_01';save(out/'previous_stratum_quality.json',dict(rows=rows,scope='Existing2048 development; all group rows retained'))
    text=['# 既有2048用户：特殊场景真实AUC','','|组/目标|UID/评分行|AUC差pp|UID95%区间pp|支持足够|','|---|---:|---:|---|---|']
    for r in rows:
        diff='NA' if r['auc_difference'] is None else f"{100*r['auc_difference']:+.4f}"
        ci=None if r['auc_interval'] is None else [round(x*100,4) for x in r['auc_interval']]
        text.append(f"|{r['stratum']}/M{r['target']}|{r['uids']}/{r['rows']}|{diff}|{ci}|{r['supported']}|")
    (out/'previous_stratum_quality.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':previous()
