"""Real six-layer read execution: identical policy inputs, paid slot input updates."""
import argparse,json,time,statistics
from pathlib import Path
from types import SimpleNamespace
import torch
from design.data import frozen_model,ROOT
from design2.common import FrozenC
from design.diagnose_native_input import read_input
from hstu_kvcache.design2.evidence import prepare,observed_score
from hstu_kvcache.design3.execution import Evidence,read,Slot
from hstu_kvcache.adaptation.reader import score


def elapsed(fn,repeats=7,inner=4):
    for _ in range(3):fn()
    torch.cuda.synchronize();points=[]
    for _ in range(repeats):
        begin=time.perf_counter()
        for _ in range(inner):fn()
        torch.cuda.synchronize();points.append((time.perf_counter()-begin)*1000/inner)
    return dict(median_ms=statistics.median(points),min_ms=min(points),max_ms=max(points),samples_ms=points)


@torch.no_grad()
def main(args):
    torch.set_num_threads(4);torch.cuda.set_device(args.device);torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/'results/design3/initial_01';rows=[];preps=[];component=[]
    for target in (1,3,4,5):
        model=frozen_model(target,args.device);adapter=FrozenC(target,args.device)
        start=time.perf_counter()
        factors=torch.load(ROOT/f'results/design2/geometry_01/cholesky_m{target}.pt',map_location=args.device,weights_only=True)
        packs=prepare(factors);evidence=Evidence(packs,adapter.parameters);torch.cuda.synchronize()
        preps.append(dict(target=target,factor_load_prepare_seconds=time.perf_counter()-start))
        def clean(cache,panel,p):
            z,obs=read(model,cache,panel,p,adapter.parameters)
            return z,observed_score(p['latent'],obs,adapter.parameters,p['counts'].double(),p['active'],packs)
        def blocked(cache,panel,p):
            state=evidence.state(p['latent'])
            z,obs=read(model,cache,panel,p,adapter.parameters)
            return z,evidence.evaluate(obs,p['counts'],p['active'],state)
        def batched(cache,panel,p):
            z,obs=read(model,cache,panel,p,adapter.parameters)
            return z,evidence.batched(p['latent'],obs,p['counts'],p['active'])
        def native(cache,panel,p):return score(model,cache,*panel)[0]
        slots={}
        for path in sorted((out/'inputs').glob(f'm{target}_*.pt')):
            raw=torch.load(path,map_location=args.device,weights_only=True)
            cache=SimpleNamespace(k=raw['k'],v=raw['v']);panel=raw['panel'];p=raw['prepared'];q=panel[0].numel();n=cache.k.shape[-2]
            ref=read_input(model,cache,panel,p['b'],p['a'],adapter.parameters,p['counts'],p['active'])
            ref_u=observed_score(p['latent'],ref[2],adapter.parameters,p['counts'].double(),p['active'],packs)
            key=(tuple(cache.k.shape),tuple(panel[0].shape),tuple(panel[1].shape))
            if key not in slots:
                # Cap lifetime to current target. These are shape slots, not per-UID graphs.
                entry={}
                for name,fn in [('clean_graph',clean),('block_graph',blocked),('batch_graph',batched),('native_graph',native)]:
                    torch.cuda.synchronize();before=torch.cuda.memory_allocated();start=time.perf_counter()
                    entry[name]=Slot(fn,cache,panel,p);torch.cuda.synchronize()
                    preps.append(dict(target=target,shape=str(key),variant=name,graph_setup_seconds=time.perf_counter()-start,allocated_bytes=torch.cuda.memory_allocated()-before))
                slots[key]=entry
            entry=slots[key]
            variants=dict(clean_eager=lambda:clean(cache,panel,p),block_eager=lambda:blocked(cache,panel,p),batch_eager=lambda:batched(cache,panel,p),native_eager=lambda:native(cache,panel,p))
            variants.update({name:(lambda slot=slot:slot.run(cache,panel,p,True)) for name,slot in entry.items()})
            for name,fn in variants.items():
                actual=fn();torch.cuda.synchronize()
                if name.startswith('native'):torch.testing.assert_close(actual,native(cache,panel,p),atol=2e-6,rtol=2e-6);ze=ue=0.
                else:
                    ze=float((actual[0]-ref[0]).abs().max());ue=float((actual[1]-ref_u).abs().max())
                    torch.testing.assert_close(actual[0],ref[0],atol=2e-6,rtol=2e-6)
                    torch.testing.assert_close(actual[1],ref_u,atol=1e-8,rtol=1e-8)
                record=dict(sample=path.stem,target=target,n=n,q=q,variant=name,score_error=ze,geometry_error=ue,**elapsed(fn))
                rows.append(record)
            # Same-state geometry only: query replication probes arithmetic, not new quality examples.
            if path.stem.endswith('_00'):
                for nq in (1,2,4,16):
                    obs=[o[:,:1].expand(-1,nq,-1).contiguous() for o in ref[2]]
                    state=evidence.state(p['latent'])
                    fs=dict(original=lambda:observed_score(p['latent'],obs,adapter.parameters,p['counts'].double(),p['active'],packs),
                        split_dirty=lambda:evidence.evaluate(obs,p['counts'],p['active'],evidence.state(p['latent'])),
                        split_reused=lambda:evidence.evaluate(obs,p['counts'],p['active'],state))
                    for name,fn in fs.items():component.append(dict(target=target,q=nq,variant=name,**elapsed(fn)))
            print(json.dumps(dict(target=target,sample=path.stem,rows=len(rows))),flush=True)
        del slots,entry,model,adapter,factors,packs,evidence;torch.cuda.empty_cache()
    (out/args.output).write_text(json.dumps(dict(scope='GPU resident actual snapshots; six-layer scoring plus screening, dirty input copies included; no state-summary encoding/event replay in this timed region',device=torch.cuda.get_device_name(),torch=torch.__version__,rows=rows,preparation=preps,geometry=component),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--output',default='timing_controls.json');main(p.parse_args())
