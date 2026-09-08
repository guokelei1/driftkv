"""Small synchronous serving adapter for the three frozen native-design paths."""

import torch
from design.diagnose_decoder_closure import SMALL
from design.diagnose_native_input import producer_mean_cache, read_input
from design.diagnose_summary_objective import install_layer
from design.run import timed

from hstu_kvcache.adaptation.translator import QueryTranslator

METHODS = ("native", "summary_input", "no_source")


class NativeRelease:
    def __init__(self,target,device,full,ablations):
        self.target = target
        self.mapper = QueryTranslator(target=target).to(device)
        self.projection = torch.load(SMALL / f"source_projection_mean_m{target}.pt",map_location=device,weights_only=True)
        self.parameters = {}
        for method in METHODS:
            path = full / f"translator_C_m{target}.pt" if method=="native" else ablations / f"translator_{method}_m{target}.pt"
            params = torch.load(path,map_location=device,weights_only=True)
            # The reader already casts these values to FP32; retain that identical representation once.
            for p in params:
                for key in ("read_weights","read_center","read_scale"):
                    p[key] = p[key].float()
            self.parameters[method] = params

    def prepare(self,state,ledger,prefix):
        source = timed(state.pack_source,ledger,prefix+"_source_pack")
        def encode():
            base = self.mapper.features(source)[None]
            p = self.projection
            x = (base.double()-p["center"])/p["scale"]@p["projection"]
            return torch.cat((torch.ones_like(x[:,:1]),x),-1),self.mapper.active_layers(base)
        latent,active = timed(encode,ledger,prefix+"_source_encode")
        counts = source.count.sum().reshape(1)
        views = {}
        for method in METHODS:
            def install(method=method):
                x = torch.ones_like(latent[:,:1]) if method=="no_source" else latent
                values = [install_layer(x,p["weights"],p["query_center"],p["query_scale"],active[:,layer])
                          for layer,p in enumerate(self.parameters[method])]
                return torch.stack([v[0] for v in values],1),torch.stack([v[1] for v in values],1)
            views[method] = timed(install,ledger,prefix+"_"+method+"_install")
        means,masses = timed(lambda:producer_mean_cache([source],self.target),ledger,prefix+"_summary_response_build")
        return dict(views=views,counts=counts,active=active,means=means,masses=masses)

    def score(self,model,state,candidates,delta,prepared,method):
        b,a = prepared["views"][method]
        return read_input(model,state.cache,(candidates,delta),b,a,self.parameters[method],prepared["counts"],prepared["active"],
                          prepared["means"] if method=="summary_input" else None,prepared["masses"],capture=False)[0]
