"""Dependency-bound execution: release transforms, state bindings, request work.

The inverse action is prepared once in FP64, not approximated or refitted.
Only one resident binding is retained; shape/mode plans share its buffers.
"""
from collections import Counter
from types import SimpleNamespace
import time
import torch
from hstu_kvcache.adaptation.reader import score


def size(t):return t.numel()*t.element_size()


class PreparedEvidence:
    def __init__(self,packs,parameters):
        chol=torch.stack([p['observed'] for p in packs])
        identity=torch.eye(225,device=chol.device,dtype=chol.dtype).expand(6,6,-1,-1)
        inverse=torch.linalg.solve_triangular(chol,identity,upper=False)
        self.inverse_residual=float((chol@inverse-identity).abs().max())
        assert self.inverse_residual<1e-8
        self.xx=inverse[...,:33,:33].contiguous()
        self.ox=inverse[...,33:,:33].contiguous()
        self.oo=inverse[...,33:,33:].contiguous()
        self.center=torch.stack([p['read_center'] for p in parameters])
        self.scale=torch.stack([p['read_scale'] for p in parameters])
    def state(self,x):
        x=x[0,:,None]
        return self.ox@x,(self.xx@x).square().sum(-2)
    def evaluate(self,observed,p):
        o=(torch.stack(observed)[:,0].double()-self.center[:,None])/self.scale[:,None]
        v=self.oo@o.transpose(-1,-2)[:,None]+p['h']
        u=(v.square().sum(-2)+p['d']).sqrt()
        return (u*p['counts'][0].double()*p['active'][0,:,None,None]).amax((0,1))[None]


def release_parameters(parameters):
    return [{k:v.float() for k,v in p.items() if k in ('read_center','read_scale','read_weights')} for p in parameters]


def read_bound(model,cache,panel,p,parameters):
    observations=[]
    def transform(layer,query,native):
        observed=native.transpose(1,2).flatten(2)/p['counts'][:,None,None]
        param=parameters[layer]
        normalized=(observed-param['read_center'])/param['read_scale']
        conditional=torch.einsum('bqa,had->bhqd',normalized,param['read_weights'])
        delta=p['bc'][:,layer,:,None]+query@p['ac'][:,layer]
        delta=delta+conditional*p['mass'][:,layer,None,None,None]
        observations.append(observed)
        return native+delta
    return score(model,cache,*panel,history_override=transform)[0],observations


class Binding:
    def __init__(self,cache,p):
        self.cache=SimpleNamespace(k=torch.empty_like(cache.k),v=torch.empty_like(cache.v))
        self.p={k:torch.empty_like(p[k]) for k in ('latent','counts','active')}
        self.p.update(bc=torch.empty_like(p['b']),ac=torch.empty_like(p['a']),mass=torch.empty_like(p['active']*p['counts'][:,None]),
                      h=p['latent'].new_empty(6,6,192,1),d=p['latent'].new_empty(6,6,1))
        self.refs={};self.stats=Counter()
    def changed(self,key,tensors,force):
        old=self.refs.get(key)
        match=old is not None and all(a is b and version==b._version for (a,version),b in zip(old,tensors))
        if match and not force:self.stats[key+'_hits']+=1;return False
        self.refs[key]=[(t,t._version) for t in tensors];self.stats[key+'_updates']+=1;return True
    def load(self,cache,p,evidence=None,force=False):
        self.stats['calls']+=1
        self.stats['naive_kv_bytes']+=size(cache.k)+size(cache.v)
        if self.changed('kv',(cache.k,cache.v),force):
            self.cache.k.copy_(cache.k);self.cache.v.copy_(cache.v)
            self.stats['kv_bytes']+=size(cache.k)+size(cache.v)
        if self.changed('view',(p['b'],p['a'],p['counts'],p['active']),force):
            torch.mul(p['b'],p['counts'][:,None,None,None],out=self.p['bc'])
            torch.mul(p['a'],p['counts'][:,None,None,None,None],out=self.p['ac'])
            torch.mul(p['active'],p['counts'][:,None],out=self.p['mass'])
            self.p['counts'].copy_(p['counts']);self.p['active'].copy_(p['active'])
            self.stats['view_elements_prepared']+=p['b'].numel()+p['a'].numel()+p['active'].numel()
        if evidence is not None and self.changed('evidence',(p['latent'],),force):
            h,d=evidence.state(p['latent']);self.p['h'].copy_(h);self.p['d'].copy_(d)
            self.p['latent'].copy_(p['latent'])


class BoundPlan:
    def __init__(self,fn,binding,panel):
        self.binding=binding;self.panel=tuple(v.clone() for v in panel);self.fn=fn
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):fn(binding.cache,self.panel,binding.p)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.output=fn(binding.cache,self.panel,binding.p)
    def run(self,panel):
        for a,b in zip(self.panel,panel):a.copy_(b)
        self.graph.replay();return self.output


class DeviceBinding(Binding):
    """The same invalidation rule, with small reusable preparation graphs."""
    def __init__(self,cache,p):
        super().__init__(cache,p)
        self.raw_b=torch.empty_like(p['b']);self.raw_a=torch.empty_like(p['a'])
        self.preparations={}
    def prepare(self,name,fn):
        if name not in self.preparations:
            torch.cuda.synchronize();begin=time.perf_counter()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):fn()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):fn()
            self.preparations[name]=graph
            torch.cuda.synchronize()
            self.stats['preparation_graph_setup_seconds']+=time.perf_counter()-begin
            self.stats['preparation_graph_captures']+=1
        self.preparations[name].replay()
    def load(self,cache,p,evidence=None,force=False):
        self.stats['calls']+=1
        self.stats['naive_kv_bytes']+=size(cache.k)+size(cache.v)
        if self.changed('kv',(cache.k,cache.v),force):
            self.cache.k.copy_(cache.k);self.cache.v.copy_(cache.v)
            self.stats['kv_bytes']+=size(cache.k)+size(cache.v)
        if self.changed('view',(p['b'],p['a'],p['counts'],p['active']),force):
            self.raw_b.copy_(p['b']);self.raw_a.copy_(p['a'])
            self.p['counts'].copy_(p['counts']);self.p['active'].copy_(p['active'])
            def view():
                torch.mul(self.raw_b,self.p['counts'][:,None,None,None],out=self.p['bc'])
                torch.mul(self.raw_a,self.p['counts'][:,None,None,None,None],out=self.p['ac'])
                torch.mul(self.p['active'],self.p['counts'][:,None],out=self.p['mass'])
            self.prepare('view',view)
            self.stats['view_elements_prepared']+=p['b'].numel()+p['a'].numel()+p['active'].numel()
        if evidence is not None and self.changed('evidence',(p['latent'],),force):
            self.p['latent'].copy_(p['latent'])
            def projection():
                h,d=evidence.state(self.p['latent']);self.p['h'].copy_(h);self.p['d'].copy_(d)
            self.prepare('evidence',projection)
