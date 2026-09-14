"""One resident dependency binding shared by the four admitted read plans."""
import time
import torch
from design3.replay import Runtime
from design2 import run_bounded_candidate as c
from hstu_kvcache.design3.binding import DeviceBinding as Binding,BoundPlan,PreparedEvidence,release_parameters,read_bound
from hstu_kvcache.design3.execution import read
from hstu_kvcache.design2.evidence import observed_score


class BindingRuntime(Runtime):
    def __init__(self,adapter,model,mode):
        super().__init__(adapter,model,mode)
        self.params=release_parameters(adapter.parameters);self.binding=None;self.transform=None
        self.stats['release_parameter_elements']=sum(t.numel() for p in self.params for t in p.values())
    def execute(self,cache,panel,p,screen):
        if cache.k.shape[-2]!=1024 or panel[0].numel() not in (1,2):
            return super().execute(cache,panel,p,screen)
        if screen and self.transform is None:
            self.factors();torch.cuda.synchronize();start=time.perf_counter()
            self.transform=PreparedEvidence(self.packs,self.adapter.parameters);torch.cuda.synchronize()
            self.stats['inverse_prepare_seconds']=time.perf_counter()-start
            self.stats['inverse_residual']=self.transform.inverse_residual
        self.stats['screen_calls' if screen else 'read_calls']+=1
        self.stats['bound_calls']+=1
        if screen:self.stats['bound_screen_queries']+=panel[0].numel()
        self.stats['avoided_release_cast_elements']+=self.stats['release_parameter_elements']
        if self.binding is None:self.binding=Binding(cache,p)
        self.binding.load(cache,p,self.transform if screen else None)
        def fn(cache,panel,p):
            z,obs=read_bound(self.model,cache,panel,p,self.params)
            return z,self.transform.evaluate(obs,p) if screen else z.new_zeros(1)
        key=(screen,tuple(panel[0].shape),tuple(panel[1].shape))
        if key not in self.slots:
            assert len(self.slots)<4
            torch.cuda.synchronize();start=time.perf_counter()
            self.slots[key]=BoundPlan(fn,self.binding,panel);torch.cuda.synchronize()
            self.stats['graph_setup_seconds']+=time.perf_counter()-start;self.stats['captures']+=1
        z,u=self.slots[key].run(panel)
        if screen:
            spec=c.b.e.PARAMS['observed_bound'];n=int(p['counts'][0])
            group=next(g for g in spec['params'] if g.startswith(f'm{self.adapter.target}_') and int(g.split('n')[1].split('-')[0])<=n<=int(g.split('-')[-1]))
            coef=spec['params'][group]
            if abs(coef['b']+coef['a']*float(u.max())-spec['threshold'])<=1e-8:
                z,obs=read(self.model,cache,panel,p,self.adapter.parameters)
                u=observed_score(p['latent'],obs,self.adapter.parameters,p['counts'].double(),p['active'],self.packs)
                self.stats['numerical_fallbacks']+=1
        for key,value in self.binding.stats.items():self.stats['binding_'+key]=value
        return z,u
