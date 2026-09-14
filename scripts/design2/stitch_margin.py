"""Exact unchanged-trajectory reuse; all affected UID lifetimes really replayed."""
import json
import pandas as pd,numpy as np
from design.data import ROOT
from design2.report_sparse import load,KEY
from design2.audit_benchmark import save

p=ROOT/'results/design2/scan_30k_01';f,c,d=load(p/'frozen');rr=p/'margin_replay'
assert json.loads((rr/'summary.json').read_text())['status']=='complete'
uids=set(d[d.action=='rebuild'].uid.unique());assert uids==set(json.loads((p/'margin_uids.json').read_text())['original'])
out=p/'margin_gate/assembled';out.mkdir(parents=True,exist_ok=True)
for name,base in [('quality_raw',f),('costs',c),('decisions',d)]:
    new=pd.read_parquet(rr/f'{name}.parquet');assert set(new.uid)==uids
    if name=='quality_raw':
        a=base[base.uid.isin(uids)].sort_values(KEY);b=new.sort_values(KEY)
        np.testing.assert_array_equal(a[KEY],b[KEY])
        # Replaying a smaller UID batch changes FP32 kernels' rounding.
        # Check its magnitude and require exactly unchanged control AUC.
        from hstu_kvcache.evaluation.binary_metrics import binary_metrics
        delta=float(np.max(np.abs(a[['reuse','design1','exact']].to_numpy()-b[['reuse','design1','exact']].to_numpy())))
        np.testing.assert_allclose(a[['reuse','design1','exact']],b[['reuse','design1','exact']],rtol=0,atol=2e-6)
        for t in a.target.unique():
            for m in ['reuse','design1','exact']:
                aa=a[a.target==t];bb=b[b.target==t]
                assert binary_metrics(aa.label,aa[m])['ROC_AUC']==binary_metrics(bb.label,bb[m])['ROC_AUC']
    whole=pd.concat([base[~base.uid.isin(uids)],new],ignore_index=True)
    assert len(whole)==len(base);whole.to_parquet(out/f'{name}.parquet',index=False)
save(out.parent/'summary.json',dict(status='complete',jobs=[dict(name='assembled',offset=0,users=4082)],replayed_uids=len(uids),
    scope='Only gate changes, called solely on existing commit path. No old-commit UID omitted; every other lifetime identical by induction. All4082 retained in analysis.',
    control_max_abs_delta=delta,control_auc_identical=True,
    sources=['frozen/summary.json','margin_replay/summary.json','margin_uids.json']))
