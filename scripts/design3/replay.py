"""Execute bounded Design2 unchanged, with a small target/shape graph-slot pool."""
import argparse,json,time
from pathlib import Path
from collections import Counter
import torch
from design2 import run_bounded_candidate as c
from design2.common import FrozenC
from hstu_kvcache.design2.evidence import prepare,observed_score as reference_evidence
from hstu_kvcache.design3.execution import Evidence,read,Slot


class Observations(list):
    pass


class Runtime:
    def __init__(self,adapter,model,mode):
        self.adapter=adapter;self.model=model;self.mode=mode;self.slots={};self.stats=Counter();self.packs=None;self.evidence=None
    def factors(self):
        if self.packs is None:
            start=time.perf_counter()
            original=c.b.e.OLD_LOAD
            factors=original(c.b.r.ROOT/f'results/design2/geometry_01/cholesky_m{self.adapter.target}.pt',map_location=next(self.model.parameters()).device,weights_only=True)
            self.packs=prepare(factors);self.evidence=Evidence(self.packs,self.adapter.parameters);torch.cuda.synchronize()
            self.stats['factor_prepare_seconds']+=time.perf_counter()-start
    def execute(self,cache,panel,p,screen):
        if screen:self.factors()
        def fn(cache,panel,p):
            z,obs=read(self.model,cache,panel,p,self.adapter.parameters)
            if screen:
                u=(self.evidence.evaluate(obs,p['counts'],p['active'],self.evidence.state(p['latent']))
                   if self.mode=='block_eager' else self.evidence.batched(p['latent'],obs,p['counts'],p['active']))
            else:u=z.new_zeros(1)
            return z,u
        self.stats['screen_calls' if screen else 'read_calls']+=1
        if self.mode=='block_eager':return fn(cache,panel,p)
        # Initial bounded graph vocabulary: common full-window 1/2-candidate reads.
        # Rare shapes run the identical eager computation, with no per-UID graph cache.
        if cache.k.shape[-2]!=1024 or panel[0].numel() not in (1,2):
            self.stats['shape_eager_calls']+=1
            return fn(cache,panel,p)
        key=(screen,tuple(cache.k.shape),tuple(panel[0].shape),tuple(panel[1].shape))
        if key not in self.slots:
            if len(self.slots)>=4:
                self.slots.pop(next(iter(self.slots)));self.stats['evictions']+=1
            torch.cuda.synchronize();start=time.perf_counter()
            self.slots[key]=Slot(fn,cache,panel,p);torch.cuda.synchronize()
            self.stats['graph_setup_seconds']+=time.perf_counter()-start;self.stats['captures']+=1
        slot=self.slots[key]
        # Conservative initial implementation: always update KV and state, even on hits.
        z,u=slot.run(cache,panel,p,True)
        # Preserve original decision for numerical near-threshold cases. Not a new gate.
        if screen:
            spec=c.b.e.PARAMS['observed_bound'];n=int(p['counts'][0])
            group=next(g for g in spec['params'] if g.startswith(f'm{self.adapter.target}_') and int(g.split('n')[1].split('-')[0])<=n<=int(g.split('-')[-1]))
            coef=spec['params'][group];estimate=coef['b']+coef['a']*float(u.max())
            if abs(estimate-spec['threshold'])<=1e-8:
                self.stats['numerical_fallbacks']+=1
                z,obs=read(self.model,cache,panel,p,self.adapter.parameters)
                u=reference_evidence(p['latent'],obs,self.adapter.parameters,p['counts'].double(),p['active'],self.packs)
        return z,u


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--users',type=int,default=8);p.add_argument('--offset',type=int,default=0);p.add_argument('--cap',type=int,default=1024);p.add_argument('--device',default='cuda:0');p.add_argument('--canary',action='store_true');p.add_argument('--uids-file',default='results/design2/scan_30k_01/canary_existing_uids.json');p.add_argument('--execution',choices=['reference','block_eager','graph','graph_service','binding'],default='graph');p.add_argument('--timed-execution',action='store_true');a=p.parse_args();a.variant='bounded'
    if a.execution=='binding':
        from design3.binding_runtime import BindingRuntime
        Runtime=BindingRuntime
    runtimes=[]
    def runtime(adapter,model):
        if not hasattr(adapter,'execution'):
            adapter.execution=Runtime(adapter,model,a.execution);runtimes.append(adapter.execution)
            # Release old graph storage/model references at a model transition.
            for previous in runtimes[:-1]:previous.slots.clear();previous.model=None
        return adapter.execution
    if a.execution!='reference':
        if a.execution in ('binding','graph_service'):
            # The other policy arms are evaluation controls, not consumers of the
            # same serving binding. Keep their reads outside the D2 resident arena.
            prior_hook=c.b.e.hook
            def mark_service(*args):
                args[1]._d3_service=True
                return prior_hook(*args)
            c.b.e.hook=mark_service
        original_corrected=c.b.r.base.corrected_score
        def execute(rt,cache,panel,prepared,screen):
            if not a.timed_execution:return rt.execute(cache,panel,prepared,screen)
            torch.cuda.synchronize();begin=time.perf_counter()
            result=rt.execute(cache,panel,prepared,screen);torch.cuda.synchronize()
            rt.stats['timed_screen_seconds' if screen else 'timed_read_seconds']+=time.perf_counter()-begin
            return result
        def frozen_read(adapter,model,cache,panel,prepared,indices):
            z,u=execute(runtime(adapter,model),cache,panel,prepared,True)
            obs=Observations();obs.geometry=u
            return z,None,obs,None
        FrozenC.read=frozen_read
        c.b.e.observed_score=lambda x,obs,*args:obs.geometry
        def corrected(model,state,adapter,panel,ledger,charges):
            if a.execution in ('binding','graph_service') and not getattr(state,'_d3_service',False):
                return original_corrected(model,state,adapter,panel,ledger,charges)
            prepared=c.b.r.base.prepare(adapter,state,ledger,charges,'service_view')
            charges['service_correction']+=c.b.r.base.read_cost(panel[0].numel(),1)
            return execute(runtime(adapter,model),state.cache,panel,prepared,False)[0]
        c.b.r.base.corrected_score=corrected
    c.b.observation=c.capture;c.b.replay.replay_user=c.replay_with_candidate;c.b.main(a)
    out=c.b.r.ROOT/'results/design2'/a.run_id
    c.b.save(out/'execution.json',dict(mode=a.execution,stats=[dict(target=r.adapter.target,**r.stats) for r in runtimes],notes='4 shape slots per target; all input copies paid; frozen D2 algorithm charge ledger, execution arithmetic reported separately; replay includes all four policy branches and diagnostic IO'))
