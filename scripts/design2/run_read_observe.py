"""Reuse the real C response to screen; reuse target construction to renew."""
import argparse,json,math
import torch
from design2 import run_sparse_observe as o
from design2.audit_benchmark import save
from design.report_native_flops import read_cost
from hstu_kvcache.design2.evidence import prepare,observed_score
from hstu_kvcache.design2.geometry import score_layer
from hstu_kvcache.design2.conditional import arithmetic

r=o.r;BANDS={};CAPTURE={};ORIGINAL=o.OLD_CORRECTED
FULL=False

def captured(model,state,adapter,panel,ledger,charges):
    z=CAPTURE.pop(id(state),None)
    return ORIGINAL(model,state,adapter,panel,ledger,charges) if z is None else z

def decide(model,state,adapter,packs,factors,fitted,history,uid,t,stamp,ledger,charges,policy):
    n=state.cache.seq_len
    g=next(g['group'] for g in fitted['groups'] if g['target']==t and g['lower']<=n<=g['upper'])
    spec=o.s.PARAMS['full_score' if FULL else 'observed_bound'];coef=spec['params'][g]
    rows=[v for v in o.REQUESTS[uid] if int(v['query_timestamp'])==stamp];q=len(rows);assert q
    p=r.base.prepare(adapter,state,ledger,charges,'service_view')
    panel=(r.original.tensor([[int(v['item_idx']) for v in rows]],state.cache.k.device),
        r.original.tensor([stamp-int(state.writer.events[-1][3])],state.cache.k.device,floating=True))
    z,trace,obs,_=adapter.read(model,state.cache,panel,p,[0])
    charges['service_correction']+=read_cost(q,1)
    if FULL:
        u=float(torch.stack([score_layer(p['latent'],trace.queries[l],obs[l],p['counts'],p['active'][:,l],adapter.parameters[l],factors[l])[0].max() for l in range(6)]).max())
        charges['full_read_evidence']+=arithmetic(q)['exact_panel_flops']
    else:
        if t not in BANDS:BANDS[t]=prepare(factors)
        u=float(observed_score(p['latent'],obs,adapter.parameters,p['counts'].double(),p['active'],BANDS[t]).max())
        charges['observed_evidence']+=q*(36*(225**2+3*225+3)+6*192*2+12)
    assert math.isfinite(u)
    charges['screen_control']+=128
    estimate=coef['b']+coef['a']*u;screened=estimate>spec['threshold']
    if decide.canary:
        reference=r.base.read_input(model,state.cache,panel,p['b'],p['a'],adapter.parameters,p['counts'],p['active'],capture=False)[0]
        torch.testing.assert_close(z,reference,rtol=0,atol=0)
    if screened:
        CAPTURE[id(state)]=z
        r.rebuild_state(model,state,history,uid,stamp,charges)
        assert id(state) not in CAPTURE
    else:o.MEMO[id(state)]=(z,panel[0],panel[1],state.revision)
    return dict(uid=uid,target=t,timestamp=stamp,count=n,group=g,source_u=None,observed_u=u,
        screen_estimate=estimate,screen_threshold=spec['threshold'],screened=screened,u=u if FULL else None,estimate=None,
        action='rebuild' if screened else 'continue',reason='observed_screen' if screened else 'screen_continue',fallback=False)

def main(cli):
    global FULL
    FULL=cli.mode=='fullreadobserve'
    o.OLD_DECIDE=decide;o.OLD_CORRECTED=captured;decide.canary=cli.canary
    o.main(cli);assert not CAPTURE
    out=r.ROOT/'results/design2'/cli.run_id
    cfg=json.loads((out/'configuration.json').read_text());cfg.update(read_observation_protocol=json.loads((r.ROOT/'configs/design2/read_observe_01.json').read_text()),read_observation_sha256=r.base.sha(__file__))
    save(out/'configuration.json',cfg)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',default='readobserve');p.add_argument('--run-id',required=True);p.add_argument('--users',type=int,default=8)
    p.add_argument('--offset',type=int,default=0);p.add_argument('--cap',type=int,default=1024);p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');main(p.parse_args())
