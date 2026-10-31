"""Stronger row-information variants of the Chapter 3 probe ladder.

Each variant retains the same f -> f+g -> f+g+h nesting, frozen-backbone
training, scalar-head scoring and causal append interfaces as the head-local
reference. This file does not change that reference or its checkpoints.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import nn

from experiments.ch3_motivation.code.probes import ProbeConfig, ProbeLadder
from hstu_kvcache.models import HSTUKVCache


class FullItemProbeLadder(ProbeLadder):
    """Every head sees the complete CURRENT item embedding, not one slice."""
    variant_name="full_item"

    @classmethod
    def from_model(cls,model,*,stage="f",**kwargs):
        cfg=replace(ProbeConfig.from_model(model,**kwargs),item_dim=model.cfg.hidden_size)
        return cls(cfg,stage)

    def _items(self,model,item_ids,item_embeddings=None):
        if not self.cfg.include_item:
            return None
        value=(model.lookup_item_embeddings(item_ids) if item_embeddings is None else item_embeddings).detach()
        if value.shape[-1]!=self.cfg.item_dim:
            raise ValueError("full-item probe requires the complete configured item embedding")
        return value[:,None].expand(-1,self.cfg.num_heads,-1,-1)

    def expanded(self,stage):
        order={"f":0,"fg":1,"fgh":2}
        if stage not in order or order[stage]<order[self.stage]:
            raise ValueError("nested expansion cannot remove a probe term")
        result=type(self)(copy.deepcopy(self.cfg),stage).to(self.row_mean)
        result.load_state_dict(self.state_dict(),strict=False)
        return result


@dataclass
class LayerwiseKVProbeConfig(ProbeConfig):
    """All heads' K/V and the full item embedding form one row input."""
    @property
    def row_dim(self):
        return 2*self.num_heads*self.head_dim+(self.item_dim if self.include_item else 0)


class DenseRowMLP(nn.Module):
    """One nonlinear row map per layer, outputting every head's K/V at once."""
    def __init__(self,input_width,output_width,hidden_width,depth):
        super().__init__()
        widths=[input_width]+[hidden_width]*depth+[output_width]
        modules=[]
        for index,(left,right) in enumerate(zip(widths[:-1],widths[1:],strict=True)):
            linear=nn.Linear(left,right)
            nn.init.normal_(linear.weight,std=left**-.5)
            nn.init.zeros_(linear.bias)
            modules.append(linear)
            if index+1!=len(widths)-1:
                modules.append(nn.SiLU())
        self.network=nn.Sequential(*modules)
        nn.init.zeros_(self.network[-1].weight)

    def forward(self,rows):
        return self.network(rows)


class LayerwiseKVProbeLadder(FullItemProbeLadder):
    """Strong candidate-independent f with cross-head row information.

    f maps the concatenated K/V of ALL heads and the full item embedding to
    all heads' K/V jointly. h receives the SAME enlarged row information in
    every head, so adding g/h does not reduce or change what f sees.
    """
    variant_name="layerwise_kv"

    def __init__(self,cfg:LayerwiseKVProbeConfig,stage="f"):
        if not isinstance(cfg,LayerwiseKVProbeConfig):
            raise TypeError("layerwise KV probe requires LayerwiseKVProbeConfig")
        super().__init__(cfg,stage)
        width=cfg.num_heads*cfg.head_dim
        self.f=nn.ModuleList(DenseRowMLP(cfg.row_dim,2*width,cfg.hidden_width,cfg.depth)
                             for _ in range(cfg.num_layers))
        self.register_buffer("kv_output_scale",torch.ones(cfg.num_layers,cfg.num_heads,2*cfg.head_dim))

    @classmethod
    def from_model(cls,model,*,stage="f",**kwargs):
        source=ProbeConfig.from_model(model,**kwargs)
        cfg=LayerwiseKVProbeConfig(**{**asdict(source),"item_dim":model.cfg.hidden_size})
        return cls(cfg,stage)

    def _raw_rows(self,layer,cache,items):
        rows=torch.cat((cache.k[layer],cache.v[layer]),dim=-1)
        rows=rows[:,None].expand(-1,self.cfg.num_heads,-1,-1)
        return torch.cat((rows,items),dim=-1) if items is not None else rows

    @torch.no_grad()
    def fit_input_statistics(self,model,samples):
        super().fit_input_statistics(model,samples)
        width=self.cfg.num_heads*self.cfg.head_dim
        for head in range(self.cfg.num_heads):
            start,end=head*self.cfg.head_dim,(head+1)*self.cfg.head_dim
            self.kv_output_scale[:,head,:self.cfg.head_dim].copy_(self.row_scale[:,0,start:end])
            self.kv_output_scale[:,head,self.cfg.head_dim:].copy_(self.row_scale[:,0,width+start:width+end])

    def _output_scale(self,layer,ndim):
        if not self.cfg.normalize:
            return 1.
        return self.kv_output_scale[layer].reshape(1,self.cfg.num_heads,*([1]*(ndim-3)),-1)

    def prepare_cache(self,model,parent_cache,item_ids,*,item_embeddings=None,read_features=None):
        features=(self.prepare_read_features(model,parent_cache,item_ids,item_embeddings=item_embeddings)
                  if read_features is None else read_features)
        keys,values=[],[]
        width=self.cfg.num_heads*self.cfg.head_dim
        for layer in range(self.cfg.num_layers):
            # The raw feature and its fixed normalization are identical across
            # heads: f consumes this shared row exactly once.
            delta=self.f[layer](features[layer][:,0])
            if self.cfg.normalize:
                delta=delta*self.row_scale[layer,0,:2*width]
            dk,dv=delta.chunk(2,-1)
            keys.append(parent_cache.k[layer]+dk)
            values.append(parent_cache.v[layer]+dv)
        return HSTUKVCache(torch.stack(keys),torch.stack(values),parent_cache.seq_len)

    def forward_flops(self,history_length,queries,*,batch=1):
        result=super().forward_flops(history_length,queries,batch=batch)
        c=self.cfg
        width=c.num_heads*c.head_dim
        widths=[c.row_dim]+[c.hidden_width]*c.depth+[2*width]
        mlp=sum(2*i*o+o for i,o in zip(widths[:-1],widths[1:],strict=True))+2*c.depth*c.hidden_width
        # prepare_read_features really normalizes the duplicated per-head
        # tensors, even though f then uses one copy; charge that work.
        normalize=2*c.row_dim*c.num_heads if c.normalize else 0
        f=int(batch*history_length*c.num_layers*(normalize+mlp+(2*width if c.normalize else 0)+2*width))
        result["f_prepare_flops"]=f
        result["total_correction_forward_flops"]=f+result["read_extra_flops"]
        return result


VARIANTS={"head":ProbeLadder,"full_item":FullItemProbeLadder,"layerwise_kv":LayerwiseKVProbeLadder}


def build_variant(name,model,*,stage="f",**kwargs):
    if name not in VARIANTS:
        raise ValueError(f"unknown Chapter 3 probe variant: {name}")
    if name=="head":
        return ProbeLadder(ProbeConfig.from_model(model,**kwargs),stage)
    return VARIANTS[name].from_model(model,stage=stage,**kwargs)


def save_variant(path,probe,metadata):
    """Persist the variant identity, architecture, fixed statistics and fit."""
    name=getattr(probe,"variant_name","head")
    if name not in VARIANTS or type(probe) is not VARIANTS[name]:
        raise ValueError("unregistered probe variant cannot be serialized as a reference probe")
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".writing")
    torch.save(dict(format="ch3_probe_variant_v1",variant=name,config=asdict(probe.cfg),stage=probe.stage,
        state_dict={name:value.detach().cpu() for name,value in probe.state_dict().items()},
        metadata=metadata),temporary)
    temporary.replace(path)


def load_variant(path,device="cpu"):
    value=torch.load(path,map_location="cpu",weights_only=False)
    if "variant" not in value and "format" not in value:
        # Existing run_probes.py checkpoints have exactly this head-local
        # representation. New complete-item/all-head inputs must carry their
        # variant tag; do not infer those architectures from tensor shapes.
        required={"config","stage","state_dict","metadata"}
        if not required<=value.keys():
            raise ValueError("expected an existing head probe checkpoint or explicit variant tag")
        config=ProbeConfig(**value["config"])
        if config.item_dim!=config.head_dim:
            raise ValueError("untagged checkpoints are head-local only; full-item variants require an explicit tag")
        probe=ProbeLadder(config,value["stage"]).to(device)
        probe.load_state_dict(value["state_dict"])
        return probe,value["metadata"]
    if value.get("format")!="ch3_probe_variant_v1" or value.get("variant") not in VARIANTS:
        raise ValueError("expected a registered Chapter 3 variant checkpoint")
    name=value["variant"]
    config_type=LayerwiseKVProbeConfig if name=="layerwise_kv" else ProbeConfig
    probe=VARIANTS[name](config_type(**value["config"]),value["stage"]).to(device)
    probe.load_state_dict(value["state_dict"])
    return probe,value["metadata"]
