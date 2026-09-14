"""Focused causal/control comparisons on the opened eight-user canaries."""
import json
import pandas as pd,numpy as np
from design.data import ROOT
from design2.audit_benchmark import save

p=ROOT/'results/design2/bounded_01';old=ROOT/'results/design2/scan_30k_01/canary_witness_actions';keys=['uid','target','request_id'];report=[]
for mode in ['post','bounded']:
    run=p/f'canary_{mode}';assert json.loads((run/'summary.json').read_text())['status']=='complete'
    f=pd.read_parquet(run/'quality_raw.parquet');o=pd.read_parquet(run/'witness_reads.parquet');l=pd.read_parquet(run/'candidate_lifetimes.parquet')
    f['candidate']=f.groupby(['uid','target','timestamp']).cumcount();m=f.merge(o,on=['uid','target','timestamp','candidate','item_idx'],validate='one_to_one')
    assert len(m)==len(o) and (m.design2==m.service).all();assert set(o.target)<={1,3,4,5}
    assert not l.duplicated(['uid','target']).any() and (l.spent<=l.allowance).all()
    future=o.merge(l[['uid','target','ended']],on=['uid','target']);assert (future.timestamp<=future.ended).all()
    report.append(dict(mode=mode,post_response_rows=len(m),lifetimes=len(l),closures=l.reason.value_counts().to_dict(),budget_ok=True))
a=pd.read_parquet(p/'canary_post/quality_raw.parquet').sort_values(keys);b=pd.read_parquet(old/'quality_raw.parquet').sort_values(keys)
np.testing.assert_array_equal(a[keys],b[keys]);np.testing.assert_array_equal(a[['reuse','design1','exact']],b[['reuse','design1','exact']])
d=pd.read_parquet(p/'canary_post/decisions.parquet');od=pd.read_parquet(old/'decisions.parquet');cols=['uid','target','timestamp','reason']
x=d[d.action=='rebuild'][cols].sort_values(['uid','target']);y=od[od.action=='rebuild'][cols].sort_values(['uid','target']);np.testing.assert_array_equal(x,y)
m=a.merge(b[keys+['design2']],on=keys,suffixes=('_post','_old'));changed=m[m.design2_post!=m.design2_old]
assert len(changed.merge(x[['uid','target','timestamp']],on=['uid','target','timestamp']))==len(changed)
save(p/'canary_verification.json',dict(status='passed',checks=report,post_actions_identical=True,changed_response_rows=len(changed),changed_only_on_commit_requests=True))
print('Canary passed:',report)

if (p/'canary_candidate/summary.json').exists():
    run=p/'canary_bounded';f=pd.read_parquet(run/'quality_raw.parquet');d=pd.read_parquet(run/'decisions.parquet');o=pd.read_parquet(run/'witness_reads.parquet')
    f['candidate']=f.groupby(['uid','target','timestamp']).cumcount();a=d[d.action=='rebuild'][['uid','target','timestamp']]
    o=o.merge(a,on=['uid','target','timestamp']);f=f.merge(o[['uid','target','timestamp','candidate','item_idx','witness']],on=['uid','target','timestamp','candidate','item_idx'],how='left');f['design2']=f.witness.fillna(f.design2)
    g=pd.read_parquet(p/'canary_candidate/quality_raw.parquet');np.testing.assert_array_equal(f.sort_values(keys)[keys+['design1','design2','reuse','exact']],g.sort_values(keys)[keys+['design1','design2','reuse','exact']])
    dd=pd.read_parquet(p/'canary_candidate/decisions.parquet');cols=['uid','target','action','timestamp'];pd.testing.assert_frame_equal(d[cols],dd[cols])
    pd.testing.assert_frame_equal(pd.read_parquet(run/'costs.parquet'),pd.read_parquet(p/'canary_candidate/costs.parquet'))
    save(p/'candidate_canary_verification.json',dict(status='passed',scores_actions_costs_identical=True,rows=len(g),users=8,scope='Direct executable candidate-return runner vs paired response-selection control'))
    print('Direct candidate-return execution matches paired scores, actions and costs.')
