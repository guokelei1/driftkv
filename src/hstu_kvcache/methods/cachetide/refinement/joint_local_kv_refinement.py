"""Joint coordinates preserve locally fitted, possibly nonzero Key decoders."""
import torch
from torch import nn

from .joint_kv_refinement import JointKVRefinement
from .joint_value_refinement import JointValueRefinement


class JointLocalKVRefinement(JointKVRefinement):
    def __init__(self,adapter,fallback_scales,key_fallback_scales):
        JointValueRefinement.__init__(self,adapter,fallback_scales)
        origin=torch.stack([layer.key_decoder.detach().clone() for layer in adapter.layers])
        rms=origin.square().mean((-2,-1),keepdim=True).sqrt()
        fallback=torch.as_tensor(key_fallback_scales,device=origin.device,dtype=origin.dtype)
        if fallback.shape!=origin.shape[:2] or not torch.isfinite(fallback).all():
            raise ValueError("Key fallback must contain only fixed fit-user scales")
        scale=torch.where(rms>0,rms,fallback.clamp_min(1e-6)[...,None,None])
        self.register_buffer("key_origin",origin)
        self.register_buffer("key_scale",scale)
        self.key_delta=nn.Parameter(torch.zeros_like(origin))
