"""Label-free census and fixed expanded cohorts from the Medium30k inventory."""
import hashlib,json,time
from pathlib import Path
import duckdb,numpy as np,pandas as pd,pyarrow.parquet as pq
from design.data import ROOT,DATASET,DAY,REQUEST_ROOT,V5_REQUEST_ROOT
from design2.audit_benchmark import save

OUT=ROOT/'results/design2/scan_30k_01'

def main():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'configs/design2/scan_30k_01.json').read_text())
    users=pq.read_table(DATASET.parent/'users.parquet',columns=['uid','selector_rank','n_theta0']).to_pandas()
    assert len(users)==30000 and users.uid.is_unique
    sealed=set();past=set()
    for name in ['evokv_design_medium_v0','evokv_design_medium_6000_v1']:
        s=json.loads((ROOT/f'data/manifests/{name}/split.json').read_text())
        for k in ['confirmation','reserved_legacy_confirmation']:sealed.update(s[k])
        past.update(s.get('historical_fitted_uids',[]))
    d=json.loads((ROOT/'configs/design2/detection_01.json').read_text())
    for x in d['groups'].values():past.update(x)
    past.update(d['excluded_design1_fitting_uids'])
    old=json.loads((ROOT/'configs/design2/scale_followup_01_uids.json').read_text());past.update(old['original']+old['extension'])
    users['sealed']=users.uid.isin(sealed);users['previously_used']=users.uid.isin(past)
    users.to_parquet(OUT/'inventory.parquet',index=False)
    scan_users=users[~users.sealed][['uid']]
    data=json.loads(DATASET.read_text());glob=str((DATASET.parent/data['shared_listens_glob']).resolve())
    con=duckdb.connect();con.execute('SET threads=16');con.register('selected',scan_users)
    table=con.execute('SELECT uid,list(timestamp ORDER BY timestamp) AS times FROM read_parquet(?) JOIN selected USING(uid) WHERE timestamp<? GROUP BY uid',[glob,301*DAY]).fetch_df();con.close()
    cuts=np.array([231,245,259,273,287])*DAY;rows=[]
    for row in table.itertuples(index=False):
        ts=np.asarray(row.times,dtype=np.int64)
        for t in [1,3,4,5]:
            cut=int(cuts[t-1]);end=int(np.searchsorted(ts,cut));kept=ts[max(0,end-1024):end]
            if not len(kept):continue
            producer=np.searchsorted(cuts,kept,side='right')
            rows.append(dict(uid=int(row.uid),target=t,retained_count=len(kept),prior_writes=int(end-np.searchsorted(ts,cut-14*DAY)),
                idle_days=float((cut-kept[-1])/DAY),old_fraction=float(np.mean(producer<=t-2)),producers=len(np.unique(producer)),
                events_in_window=int(np.searchsorted(ts,cuts[t] if t<5 else 301*DAY)-end)))
    f=pd.DataFrame(rows);f['rare_activity']=False
    for b in json.loads((ROOT/'results/design2/analysis/benchmark_01/calibration_bounds.json').read_text()):
        f.loc[(f.target==b['target'])&((f[b['feature']]<b['lower'])|(f[b['feature']]>b['upper'])),'rare_activity']=True
    f['short']=f.retained_count<=128;f['old_lineage']=(f.target>=3)&(f.old_fraction>=.25);f['mixed']=f.producers>=3
    strata=['short','old_lineage','mixed','rare_activity'];f['challenge']=f[strata].any(axis=1);f['routine']=~f.challenge
    counts=[]
    for t in [1,3,4,5]:
        root=V5_REQUEST_ROOT if t==5 else REQUEST_ROOT
        q=pq.read_table(root/'requests_quality.parquet',columns=['uid','query_timestamp'],filters=[('uid','in',scan_users.uid.tolist()),('time_block','=','matrix_horizon'),('target_known','=',True),('query_timestamp','>=',int(cuts[t-1])),('query_timestamp','<',int(cuts[t] if t<5 else 301*DAY))]).to_pandas()
        counts.append(q.groupby('uid').size().rename('scoring_rows').reset_index().assign(target=t))
    f=f.merge(pd.concat(counts),on=['uid','target'],how='left');f.scoring_rows=f.scoring_rows.fillna(0).astype(int)
    f=f.merge(users[['uid','previously_used']],on='uid',validate='many_to_one');f.to_parquet(OUT/'members.parquet',index=False)
    fresh=set(users.loc[~users.sealed & ~users.previously_used,'uid']) & set(f[f.target==1].uid)
    order=lambda name,vals:sorted(map(int,vals),key=lambda u:hashlib.sha256(f'd2-scan30k:17:{name}:{u}'.encode()).digest())
    natural=order('natural',fresh)[:2048];assert len(natural)==2048
    taken=set(natural);groups={};allocation=[]
    for name in strata:
        eligible=set(f.loc[f[name]&(f.scoring_rows>0),'uid']) & fresh-taken
        chosen=order(name,eligible)[:512];groups[name]=chosen;taken.update(chosen)
        allocation.append(dict(stratum=name,eligible=len(eligible),selected=len(chosen)))
    chosen=natural+sum(groups.values(),[]);assert len(chosen)==len(set(chosen)) and not set(chosen)&sealed
    save(OUT/'uids.json',dict(original=natural,extension=sum(groups.values(),[]),natural=natural,enriched=groups,protocol=cfg))
    summary=dict(inventory_users=len(users),sealed_metadata_only=int(users.sealed.sum()),timestamp_scanned_users=int(f.uid.nunique()),fresh_available=len(fresh),
        natural=len(natural),enriched=len(chosen)-len(natural),allocation=allocation,
        census=[dict(stratum=s,target=int(t),users=len(g),active_users=int((g.scoring_rows>0).sum()),scoring_rows=int(g.scoring_rows.sum())) for s in strata+['challenge','routine'] for t,g in f[f[s]].groupby('target')],
        seconds=time.monotonic()-start,labels_read=False,model_outputs_read=False,confirmation_outputs_read=False)
    save(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
