"""Narrow Value codes whose encoder sees the complete Current item embedding.

Raw item coordinates do not have attention-head semantics. Each head retains
its own native K/V but sees all Current item coordinates before compressing to
the same narrow code. Storage, attention and decoding are unchanged.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .lightweight import LightweightValueAdapter,LightweightValueLayer


class FullItemValueLayer(LightweightValueLayer):
    def __init__(self,heads,head_dim,rank):
        nn.Module.__init__(self)
        self.heads,self.head_dim,self.rank=heads,head_dim,rank
        self.input_width=(2+heads)*head_dim
        self.register_buffer("input_center",torch.zeros(heads,self.input_width))
        self.register_buffer("input_scale",torch.ones(heads,self.input_width))
        self.input_weight=nn.Parameter(torch.empty(heads,self.input_width,rank))
        self.input_bias=nn.Parameter(torch.zeros(heads,rank))
        self.decoder=nn.Parameter(torch.zeros(heads,rank,head_dim))
        nn.init.uniform_(self.input_weight,-1/math.sqrt(self.input_width),1/math.sqrt(self.input_width))

    def encode(self,k,v,item_features):
        shape=(*k.shape[:-1],self.heads,self.head_dim)
        full_item=item_features.unsqueeze(-2).expand(*k.shape[:-1],self.heads,self.heads*self.head_dim)
        features=torch.cat((k.reshape(shape),v.reshape(shape),full_item),dim=-1)
        normalized=(features.to(self.input_weight.dtype)-self.input_center)/self.input_scale
        code=F.silu(normalized.transpose(1,2)@self.input_weight[None]+self.input_bias[None,:,None])
        return code.transpose(1,2).to(k.dtype)


class LightweightFullItemAdapter(LightweightValueAdapter):
    kind="design_one_lightweight_full_item_value_v1"

    def __init__(self,num_layers,heads,head_dim,rank=8,response_hidden_width=0,max_length=1024):
        nn.Module.__init__(self)
        if response_hidden_width:
            raise ValueError("the full-item candidate has no Response branch")
        self.num_layers,self.heads,self.head_dim=num_layers,heads,head_dim
        self.rank,self.response_hidden_width,self.max_length=rank,0,max_length
        self.layers=nn.ModuleList([FullItemValueLayer(heads,head_dim,rank) for _ in range(num_layers)])
        self.response=None

    def estimate_flops(self,*,batch=1,tokens=1,candidates=1):
        result=super().estimate_flops(batch=batch,tokens=tokens,candidates=candidates)
        width=(self.heads+2)*self.head_dim
        result["token_transform"]=int(self.num_layers*batch*tokens*self.heads*
                                      (2*width+2*width*self.rank+2*self.rank))
        return result
