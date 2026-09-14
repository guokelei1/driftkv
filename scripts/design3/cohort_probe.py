"""Fixed ready-work feasibility, no arrival prediction or new model teachers."""
import json, time, hashlib
from types import SimpleNamespace
import torch
from design.data import ROOT, frozen_model
from design2.common import FrozenC
from design3.benchmark import elapsed
from hstu_kvcache.design3.execution import Evidence, read
from hstu_kvcache.design3.binding import release_parameters
from hstu_kvcache.design3.cohort import execute
from hstu_kvcache.design2.evidence import prepare
from hstu_kvcache.adaptation.reader import score


class Plan:
    def __init__(self, fn, raw):
        self.raw = {k:v.clone() for k,v in raw.items()}
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): fn(self.raw)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph): self.result = fn(self.raw)
    def run(self, raw):
        for k,v in raw.items(): self.raw[k].copy_(v)
        self.graph.replay()
        return self.result


@torch.no_grad()
def main():
    torch.set_num_threads(4); torch.cuda.set_device(0); torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/'results/design3/cohort_01'; out.mkdir(parents=True,exist_ok=True)
    rows=[]; inputs=[];started=time.perf_counter()
    spec=json.loads((ROOT/'configs/design2/evidence_01_fitted.json').read_text())['candidates']['observed_bound']
    for target in (1,3,4,5):
        model=frozen_model(target,'cuda:0'); adapter=FrozenC(target,'cuda:0')
        params=release_parameters(adapter.parameters)
        evidence=Evidence(prepare(torch.load(ROOT/f'results/design2/geometry_01/cholesky_m{target}.pt',map_location='cuda:0',weights_only=True)),adapter.parameters)
        paths=sorted((ROOT/'results/design3/initial_01/inputs').glob(f'm{target}_*.pt'))[:8]
        samples=[torch.load(p,map_location='cuda:0',weights_only=True) for p in paths]
        inputs.extend(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths)
        for batch in (1,4,8):
            if len(samples)<batch: continue
            subset=samples[:batch]
            raw={k:torch.cat([s['prepared'][k] for s in subset],0) for k in subset[0]['prepared']}
            raw.update(k=torch.cat([s['k'] for s in subset],1),v=torch.cat([s['v'] for s in subset],1),
                       items=torch.cat([s['panel'][0][:,:1] for s in subset],0),dt=torch.cat([s['panel'][1] for s in subset],0))
            for mix in ('homogeneous','mixed'):
                modes=[2 if mix=='homogeneous' else i%3 for i in range(batch)]
                if batch==1: modes=[2]
                groups={}
                for variant in ('individual','path_group','operator_group'):
                    groups[variant]=([[i] for i in range(batch)] if variant=='individual' else
                        [[i for i,m in enumerate(modes) if m==mode] for mode in sorted(set(modes))] if variant=='path_group' else [list(range(batch))])
                def make_fn(group):
                    specs=[]
                    for ids in group:
                        idx=torch.tensor(ids,device='cuda:0'); c=torch.tensor([j for j,i in enumerate(ids) if modes[i]>0],device='cuda:0',dtype=torch.long)
                        s=torch.tensor([j for j,i in enumerate(ids) if modes[i]==2],device='cuda:0',dtype=torch.long)
                        specs.append((idx,c,s))
                    def fn(r):
                        z=r['counts'].new_empty(batch,1,dtype=torch.float32); u=r['latent'].new_zeros(batch,1)
                        for idx,c,s in specs:
                            p={k:r[k][idx] for k in ('latent','counts','active','b','a')}
                            zz,uu=execute(model,SimpleNamespace(k=r['k'][:,idx],v=r['v'][:,idx]),(r['items'][idx],r['dt'][idx]),p,params,evidence,c,s)
                            z.index_copy_(0,idx,zz)
                            if s.numel(): u.index_copy_(0,idx[s],uu)
                        return z,u
                    return fn
                # Independent original single-state reference, not the new grouping path.
                rz=[];ru=[]
                for i,mode in enumerate(modes):
                    p={k:raw[k][i:i+1] for k in ('latent','counts','active','b','a')}
                    cache=SimpleNamespace(k=raw['k'][:,i:i+1],v=raw['v'][:,i:i+1]); panel=(raw['items'][i:i+1],raw['dt'][i:i+1])
                    if mode:
                        z,obs=read(model,cache,panel,p,params)
                        u=evidence.batched(p['latent'],obs,p['counts'],p['active']) if mode==2 else p['latent'].new_zeros(1,1)
                    else: z=score(model,cache,*panel)[0];u=p['latent'].new_zeros(1,1)
                    rz.append(z);ru.append(u)
                reference=(torch.cat(rz),torch.cat(ru)); saved={k:v.clone() for k,v in raw.items()}
                for variant,group in groups.items():
                    fn=make_fn(group); torch.cuda.synchronize(); before=torch.cuda.memory_allocated(); start=time.perf_counter(); plan=Plan(fn,raw);torch.cuda.synchronize()
                    setup=time.perf_counter()-start;memory=torch.cuda.memory_allocated()-before
                    actual=plan.run(raw)
                    ze=float((actual[0]-reference[0]).abs().max());ue=float((actual[1]-reference[1]).abs().max())
                    relative=float(((actual[1]-reference[1]).abs()/reference[1].abs().clamp_min(1e-12)).max())
                    decisions=[]
                    for i,mode in enumerate(modes):
                        if mode!=2: continue
                        n=int(raw['counts'][i]);g=next(g for g in spec['params'] if g.startswith(f'm{target}_') and int(g.split('n')[1].split('-')[0])<=n<=int(g.split('-')[-1]))
                        coef=spec['params'][g]
                        aa=coef['b']+coef['a']*float(actual[1][i]);rr=coef['b']+coef['a']*float(reference[1][i])
                        assert (aa>spec['threshold'])==(rr>spec['threshold'])
                        decisions.append(dict(row=i,reference=rr,actual=aa,threshold=spec['threshold']))
                    torch.testing.assert_close(actual[0],reference[0],atol=2e-5,rtol=2e-5)
                    torch.testing.assert_close(actual[1],reference[1],atol=1e-6,rtol=2e-5)
                    timing=elapsed(lambda:plan.run(raw),repeats=5,inner=3)
                    rows.append(dict(target=target,batch=batch,mix=mix,modes=modes,variant=variant,shared_layer_batches=6*len(group),setup_seconds=setup,workspace_bytes=memory,score_error=ze,geometry_error=ue,geometry_relative_error=relative,decisions=decisions,**timing))
                    del plan
                for k in raw: assert torch.equal(raw[k],saved[k])
                print(json.dumps(dict(target=target,batch=batch,mix=mix)),flush=True)
        del model,adapter,evidence;torch.cuda.empty_cache()
    sources=['scripts/design3/cohort_probe.py','src/hstu_kvcache/design3/cohort.py']
    (out/'summary.json').write_text(json.dumps(dict(status='complete',elapsed_seconds=time.perf_counter()-started,rows=rows,inputs=inputs,sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},torch=torch.__version__,gpu=torch.cuda.get_device_name(0),scope='fixed ready-work snapshots; assigned execution modes; one actual candidate per snapshot; no arrival or lifecycle performance claim'),indent=2))

if __name__=='__main__': main()
