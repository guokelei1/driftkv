"""UID-cross-calibrated cheap evidence, with fixed ten-percent screening."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from design2.scale_calibration import fit,predict,describe,digest
from design2.report_detection import weights,quantile
from design2.audit_benchmark import save

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/design2/evidence_01'
FROZEN=ROOT/'configs/design2/evidence_01_fitted.json'
KEY=['uid','target','state_ordinal']

def frame(role):
    original='calibration_oof' if role=='residual_calibration' else 'development_predictions'
    old=pd.read_parquet(ROOT/f'results/design2/analysis/scale_calibration_01/{original}.parquet')
    new=pd.concat([pd.read_parquet(p) for p in sorted((OUT/'chunks').glob(f'{role}_*/states_m*.parquet'))])
    assert not new.duplicated(KEY).any()
    f=old.merge(new[KEY+['source_bound','observed_bound','full_score','error']],on=KEY,validate='one_to_one')
    assert len(f)==len(old)
    np.testing.assert_allclose(f.detection,f.full_score,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(f.max_abs_error,f.error,rtol=1e-10,atol=1e-10)
    return f

def metrics(f,score,threshold):
    d=describe(f,score,{'0.9':threshold},[.1,.5,1.])
    selected=f[score]>threshold;w=weights(f)
    d['screened_states']=int(selected.sum());d['screened_uids']=int(f.loc[selected,'uid'].nunique())
    d['screened_uid_weighted_fraction']=float(w[selected].sum()/w.sum())
    d['severe_capture']={str(t):dict(total_uids=int(f.loc[f.max_abs_error>=t,'uid'].nunique()),
        screened_uids=int(f.loc[selected & (f.max_abs_error>=t),'uid'].nunique()),
        weighted_mass_fraction=float(w[selected & (f.max_abs_error>=t)].sum()/w[f.max_abs_error>=t].sum())) for t in [.1,.5,1.]}
    return d

def main(stage):
    out=OUT/'analysis';out.mkdir(exist_ok=True)
    f=frame('residual_calibration' if stage=='calibrate' else 'development')
    summaries={};frozen={}
    if stage=='calibrate':
        assert not FROZEN.exists()
        for name in ['source_bound','observed_bound','full_score']:
            source=f.assign(detection=f[name]);parts=[]
            for k in range(5):
                parts.append(predict(source[source.fold==k],fit(source[source.fold!=k])))
            oof=pd.concat(parts).sort_index()
            tau=quantile(oof.calibrated.to_numpy(),weights(oof),.9)
            f[name+'_cal']=oof.calibrated
            frozen[name]=dict(params=fit(source),threshold=tau)
            summaries[name]=metrics(oof,'calibrated',tau)
        tau=quantile(oof.background.to_numpy(),weights(oof),.9)
        frozen['background']=dict(threshold=tau,params=frozen['full_score']['params'])
        f['background_new']=oof.background
        summaries['background']=metrics(f,'background_new',tau)
        save(FROZEN,dict(candidates=frozen,config_sha256=digest(ROOT/'configs/design2/evidence_01.json'),
            source_sha256=digest(Path(__file__)),selection='both cheap candidates reported at fixed 10% screening; no development outcomes used',users=512))
    else:
        frozen=json.loads(FROZEN.read_text())['candidates']
        for name,p in frozen.items():
            predicted=predict(f.assign(detection=f['full_score' if name=='background' else name]),p['params'])
            col='background' if name=='background' else 'calibrated'
            f[name+'_cal']=predicted[col]
            summaries[name]=metrics(predicted,col,p['threshold'])
    save(out/f'{stage}.json',summaries)
    f.to_parquet(out/f'{stage}.parquet',index=False)
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibrate','evaluate']);main(p.parse_args().stage)
