"""Offline coordinates for end-to-end fitting of the existing Value code path."""
import torch
from torch import nn
from torch.nn import functional as F

from hstu_kvcache.methods.cachetide.compact.lightweight import LightweightValueState
from hstu_kvcache.methods.cachetide.compact.lightweight_fullitem import LightweightFullItemAdapter
from .decoder_refinement import DecoderRefinement


class JointValueRefinement(nn.Module):
    """Tune code encoders and decoders; publication folds all coordinates.

    Native KV, Current item inputs, and input statistics remain fixed. Codes
    must be recomputed with gradients on every training forward. They cannot
    be reused across optimizer steps because their encoder is being fitted.
    """
    def __init__(self, adapter, fallback_scales):
        super().__init__()
        if not isinstance(adapter, LightweightFullItemAdapter):
            raise TypeError("joint fitting currently uses the full-item Value adapter")
        weights=torch.stack([x.input_weight.detach().clone() for x in adapter.layers])
        biases=torch.stack([x.input_bias.detach().clone() for x in adapter.layers])
        weight_scale=weights.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
        bias_rms=biases.square().mean(-1,keepdim=True).sqrt()
        bias_scale=torch.where(bias_rms>0,bias_rms,weight_scale.squeeze(-1))
        self.register_buffer("weight_origin",weights)
        self.register_buffer("bias_origin",biases)
        self.register_buffer("weight_scale",weight_scale)
        self.register_buffer("bias_scale",bias_scale)
        self.weight_delta=nn.Parameter(torch.zeros_like(weights))
        self.bias_delta=nn.Parameter(torch.zeros_like(biases))
        self.decoder=DecoderRefinement(adapter,fallback_scales)

    def encoders(self):
        return (self.weight_origin+self.weight_scale*self.weight_delta,
                self.bias_origin+self.bias_scale*self.bias_delta)

    def encode_cache(self,adapter,cache,item_features):
        weights,biases=self.encoders()
        codes=[]
        for index,layer in enumerate(adapter.layers):
            k,v=cache.k[index],cache.v[index]
            shape=(*k.shape[:-1],layer.heads,layer.head_dim)
            item=item_features[:,:,None,:].expand(*k.shape[:-1],layer.heads,
                                                  layer.heads*layer.head_dim)
            features=torch.cat((k.reshape(shape),v.reshape(shape),item),dim=-1)
            normalized=(features.to(weights.dtype)-layer.input_center)/layer.input_scale
            code=F.silu(normalized.transpose(1,2)@weights[index][None]
                        +biases[index][None,:,None])
            codes.append(code.transpose(1,2).to(k.dtype))
        return LightweightValueState(torch.stack(codes),cache.seq_len)

    def make_history_read(self,adapter,state):
        return self.decoder.make_history_read(adapter,state)

    def identity_penalty(self):
        parameters=tuple(self.parameters())
        return sum(p.square().sum() for p in parameters)/sum(p.numel() for p in parameters)

    def coordinate_rms(self):
        return self.identity_penalty().sqrt()

    @torch.no_grad()
    def fold_into(self,adapter):
        weights,biases=self.encoders()
        for layer,weight,bias in zip(adapter.layers,weights,biases,strict=True):
            layer.input_weight.copy_(weight)
            layer.input_bias.copy_(bias)
        self.decoder.fold_into(adapter)
