"""State evidence gates costly verification; every accepted renewal persists."""
import argparse,json
from design2 import run_scale_followup as r
from design2.audit_benchmark import save
from hstu_kvcache.design2.evidence import prepare as prepare_evidence,source_score

OLD_DECIDE=r.decide
MODE=None
PARAMS=None
PACKS={}

def decide(model,state,adapter,packs,factors,fitted,history,uid,t,stamp,ledger,charges,policy):
    n=state.cache.seq_len
    g=next(g['group'] for g in fitted['groups'] if g['target']==t and g['lower']<=n<=g['upper'])
    spec=PARAMS['background' if MODE=='background' else 'source_bound'];coef=spec['params'][g]
    charges['screen_control']+=128
    u=None
    if MODE=='background':estimate=coef['background']
    elif coef['a']==0:estimate=coef['b']
    else:
        if t not in PACKS:PACKS[t]=prepare_evidence(factors,include_observed=False)
        p=r.base.prepare(adapter,state,ledger,charges,'service_view')
        u=float(source_score(p['latent'],p['counts'].double(),p['active'],PACKS[t])[0])
        charges['source_evidence']+=36*(33*33+3*33+3)+12
        estimate=coef['b']+coef['a']*u
    screened=estimate>spec['threshold']
    record=dict(uid=uid,target=t,timestamp=stamp,count=n,group=g,source_u=u,screen_estimate=estimate,
                screened=screened,screen_threshold=spec['threshold'])
    if not screened:
        return dict(**record,u=None,estimate=None,action='continue',reason='screen_continue',fallback=False)
    if MODE=='direct':
        r.rebuild_state(model,state,history,uid,stamp,charges)
        return dict(**record,u=None,estimate=None,action='rebuild',reason='screen_direct',fallback=False)
    result=OLD_DECIDE(model,state,adapter,packs,factors,fitted,history,uid,t,stamp,ledger,charges,'risk')
    result.update(record);return result

def main(cli):
    global MODE,PARAMS
    MODE=cli.mode
    PARAMS=json.loads((r.ROOT/'configs/design2/evidence_01_fitted.json').read_text())['candidates']
    r.decide=decide;cli.policy='risk';r.main(cli)
    out=r.ROOT/'results/design2'/cli.run_id
    cfg=json.loads((out/'configuration.json').read_text())
    cfg.update(sparse_mode=MODE,sparse_protocol=json.loads((r.ROOT/'configs/design2/sparse_01.json').read_text()),
        sparse_sources={str(p.relative_to(r.ROOT)):r.base.sha(p) for p in [r.ROOT/'scripts/design2/run_sparse.py',r.ROOT/'src/hstu_kvcache/design2/evidence.py',r.ROOT/'configs/design2/evidence_01_fitted.json']})
    save(out/'configuration.json',cfg)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['source','direct','background'],required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--users',type=int,default=8);p.add_argument('--offset',type=int,default=0);p.add_argument('--cap',type=int,default=1024)
    p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');main(p.parse_args())
