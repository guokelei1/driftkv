"""Paid state rebinding versus unchanged-state reuse on retained real inputs."""
import json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from design.data import ROOT,frozen_model
from design2.common import FrozenC
from design3.benchmark import elapsed
from hstu_kvcache.design3.execution import Evidence,read,Slot
from hstu_kvcache.design3.binding import PreparedEvidence,release_parameters,read_bound,DeviceBinding as Binding,BoundPlan
from hstu_kvcache.design2.evidence import prepare


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/'results/design3/binding_01';out.mkdir(exist_ok=True,parents=True);rows=[];preparation=[]
    for target in (1,3,4,5):
        model=frozen_model(target,'cuda:0');adapter=FrozenC(target,'cuda:0')
        factors=torch.load(ROOT/f'results/design2/geometry_01/cholesky_m{target}.pt',map_location='cuda:0',weights_only=True)
        packs=prepare(factors);base=Evidence(packs,adapter.parameters)
        torch.cuda.synchronize();start=time.perf_counter()
        ev=PreparedEvidence(packs,adapter.parameters);params=release_parameters(adapter.parameters);torch.cuda.synchronize()
        preparation.append(dict(target=target,inverse_and_parameter_seconds=time.perf_counter()-start,inverse_residual=ev.inverse_residual,
            inverse_and_check_leading_flops=3*36*225**3,prepared_bytes=sum(t.numel()*t.element_size() for t in (ev.xx,ev.ox,ev.oo))+sum(t.numel()*t.element_size() for p in params for t in p.values())))
        def old(cache,panel,p):
            z,obs=read(model,cache,panel,p,adapter.parameters)
            return z,base.batched(p['latent'],obs,p['counts'],p['active'])
        def hoisted(cache,panel,p):
            z,obs=read(model,cache,panel,p,params)
            return z,base.batched(p['latent'],obs,p['counts'],p['active'])
        def release_only(cache,panel,p):
            pp=dict(p)
            pp.update(bc=p['b']*p['counts'][:,None,None,None],ac=p['a']*p['counts'][:,None,None,None,None],mass=p['active']*p['counts'][:,None])
            pp['h'],pp['d']=ev.state(p['latent'])
            z,obs=read_bound(model,cache,panel,pp,params)
            return z,ev.evaluate(obs,pp)
        def bound(cache,panel,p):
            z,obs=read_bound(model,cache,panel,p,params)
            return z,ev.evaluate(obs,p)
        plans={}
        for path in sorted((ROOT/'results/design3/initial_01/inputs').glob(f'm{target}_*.pt')):
            raw=torch.load(path,map_location='cuda:0',weights_only=True);panel=raw['panel'];p=raw['prepared']
            if panel[0].numel()>2:continue
            cache=SimpleNamespace(k=raw['k'],v=raw['v']);key=tuple(panel[0].shape)
            if key not in plans:
                start=time.perf_counter();binding=Binding(cache,p);binding.load(cache,p,ev)
                slot=BoundPlan(bound,binding,panel);prior=Slot(old,cache,panel,p)
                cast_control=Slot(hoisted,cache,panel,p);release_control=Slot(release_only,cache,panel,p)
                def ordinary_old(cache,panel,p):return read(model,cache,panel,p,adapter.parameters)[0]
                def ordinary_bound(cache,panel,p):return read_bound(model,cache,panel,p,params)[0]
                ordinary_prior=Slot(ordinary_old,cache,panel,p);ordinary_plan=BoundPlan(ordinary_bound,binding,panel)
                torch.cuda.synchronize()
                preparation.append(dict(target=target,shape=str(key),paired_setup_seconds=time.perf_counter()-start))
                plans[key]=(binding,slot,prior,cast_control,release_control,ordinary_prior,ordinary_plan)
            binding,slot,prior,cast_control,release_control,ordinary_prior,ordinary_plan=plans[key]
            expected=old(cache,panel,p)
            def execute(force):binding.load(cache,p,ev,force);return slot.run(panel)
            for name,fn in [('prior_graph',lambda:prior.run(cache,panel,p,True)),('hoist_cast_only',lambda:cast_control.run(cache,panel,p,True)),('release_only',lambda:release_control.run(cache,panel,p,True)),('bound_dirty',lambda:execute(True)),('bound_reused',lambda:execute(False))]:
                actual=fn();torch.cuda.synchronize()
                ze=float((actual[0]-expected[0]).abs().max());ue=float((actual[1]-expected[1]).abs().max())
                torch.testing.assert_close(actual[0],expected[0],atol=0,rtol=0)
                torch.testing.assert_close(actual[1],expected[1],atol=1e-7,rtol=1e-10)
                rows.append(dict(target=target,sample=path.stem,q=panel[0].numel(),variant=name,score_error=ze,geometry_error=ue,**elapsed(fn)))
            def ordinary(force):binding.load(cache,p,None,force);return ordinary_plan.run(panel)
            for name,fn in [('ordinary_prior',lambda:ordinary_prior.run(cache,panel,p,True)),('ordinary_dirty',lambda:ordinary(True)),('ordinary_reused',lambda:ordinary(False))]:
                actual=fn();torch.testing.assert_close(actual,expected[0],rtol=0,atol=0)
                rows.append(dict(target=target,sample=path.stem,q=panel[0].numel(),variant=name,score_error=0.,geometry_error=0.,**elapsed(fn)))
            print(json.dumps(dict(target=target,sample=path.stem)),flush=True)
        del plans,slot,prior,binding,model,adapter,base,ev;torch.cuda.empty_cache()
    (out/'device_controls.json').write_text(json.dumps(dict(rows=rows,preparation=preparation,scope='prior batched graph vs release-prepared inverse action and state binding; dirty pays full KV/view binding every request; reused is unchanged-state sensitivity, not workload average'),indent=2))

if __name__=='__main__':main()
