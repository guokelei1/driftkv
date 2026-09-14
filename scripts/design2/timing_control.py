"""Paired response-rule control on a genuinely replayed state trajectory.

Requests/events are fixed offline inputs and never depend on returned scores.
Only return selection differs; construction/admission/install/state transitions
stay exactly those executed. First validate the mapping against old full replay.
"""
import json
import numpy as np,pandas as pd
from design.data import ROOT
from design2.report_sparse import load,KEY
from design2.audit_benchmark import save

p=ROOT/'results/design2/bounded_01';old=ROOT/'results/design2/scan_30k_01'
def materialize(policy):
    f,c,d=load(p/policy);jobs=json.loads((p/policy/'summary.json').read_text())['jobs']
    obs=pd.concat([pd.read_parquet(p/policy/j['name']/'witness_reads.parquet') for j in sorted(jobs,key=lambda x:x['offset'])],ignore_index=True)
    f['candidate']=f.groupby(['uid','target','timestamp']).cumcount()
    action=d[d.action=='rebuild'][['uid','target','timestamp']]
    chosen=obs.merge(action,on=['uid','target','timestamp'],validate='many_to_one')
    f=f.merge(chosen[['uid','target','timestamp','candidate','item_idx','witness']],on=['uid','target','timestamp','candidate','item_idx'],how='left',validate='one_to_one')
    assert f.witness.notna().sum()==len(chosen)
    # A fixed branch selection between the two scores already computed online.
    f.loc[f.witness.notna(),'design2']=f.loc[f.witness.notna(),'witness']
    return f.drop(columns=['candidate','witness']),c,d

def main():
    check,_,_=materialize('post');base=load(old/'witness')[0]
    a=check.sort_values(KEY);b=base.sort_values(KEY);np.testing.assert_array_equal(a[KEY],b[KEY]);np.testing.assert_array_equal(a.design2,b.design2)
    f,c,d=materialize('bounded');out=p/'bounded_immediate/paired';out.mkdir(parents=True,exist_ok=False)
    for name,frame in [('quality_raw',f),('costs',c),('decisions',d)]:frame.to_parquet(out/f'{name}.parquet',index=False)
    save(out.parent/'summary.json',dict(status='complete',jobs=[dict(name='paired',offset=0,users=4082)],
        scope='Paired return-selection control on bounded actual replay. Exactly same states/actions/events; chooses already computed target score iff installation executes. No labels used, no weighted score mixing or state splicing.',
        validation='Identical transformation of post-unlimited reproduces every old immediate-witness response bitwise.',
        source='bounded/summary.json',source_script_sha256=__import__('hashlib').sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()))
    print('Paired timing mapping exactly reproduces old full replay; bounded immediate control written.')
if __name__=='__main__':main()
