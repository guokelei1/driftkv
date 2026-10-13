"""Offline-only decoder coordinates; serving receives ordinary folded weights."""
import torch
from torch import nn


class DecoderRefinement(nn.Module):
    """D = D0 + s*a, with release-fixed, per-head scales and identity a=0.

    Scale uses the frozen decoder's RMS. Exactly zero heads use their original
    fitting-only output scale (floored at 1e-6); validation/evaluation statistics
    do not define the coordinates. Only ``delta`` is trainable. No extra module
    remains after folding the selected delta into an ordinary adapter.
    """
    def __init__(self, adapter, fallback_scales):
        super().__init__()
        origin=torch.stack([layer.decoder.detach().clone() for layer in adapter.layers])
        rms=origin.square().mean((-2,-1),keepdim=True).sqrt()
        fallback=torch.as_tensor(fallback_scales,device=origin.device,dtype=origin.dtype)
        if fallback.shape != origin.shape[:2] or not torch.isfinite(fallback).all():
            raise ValueError("fallback must be the finite fitting-only output scale for every head")
        scale=torch.where(rms>0,rms,fallback.clamp_min(1e-6)[...,None,None])
        self.register_buffer("origin",origin)
        self.register_buffer("scale",scale)
        self.delta=nn.Parameter(torch.zeros_like(origin))

    def decoders(self):
        return self.origin+self.scale*self.delta

    def make_history_read(self,adapter,state):
        decoders=self.decoders()

        def read(index,attention,q,k,v):
            if state.seq_len != k.shape[1]:
                raise ValueError("decoder refinement code rows must align with native cache")
            return adapter.layers[index].read(attention,q,k,v,state.codes[index],decoder=decoders[index])

        return read

    def identity_penalty(self):
        return self.delta.square().mean()

    @torch.no_grad()
    def fold_into(self,adapter):
        for layer,decoder in zip(adapter.layers,self.decoders(),strict=True):
            layer.decoder.copy_(decoder)
