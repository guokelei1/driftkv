"""PhaseB h initialization only: identical dense MLP, zero output and losses."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from hstu_kvcache.common.artifacts import sha256

MODES=("standard","balanced")


def initialization_spec(query_dim,row_dim,mode):
    if mode not in MODES or min(query_dim,row_dim)<1:
        raise ValueError("h initialization requires standard/balanced and positive actual widths")
    total=query_dim+row_dim
    return dict(mode=mode,query_dim=query_dim,row_dim=row_dim,
        query_gain=math.sqrt(total/(2*query_dim)) if mode=="balanced" else 1.,
        row_gain=math.sqrt(total/(2*row_dim)) if mode=="balanced" else 1.,
        expected_query_energy_fraction=.5 if mode=="balanced" else query_dim/total,
        formula="Wq*=sqrt((D+F)/(2D)); Wr*=sqrt((D+F)/(2F)); first bias unchanged; final output weights/bias remain zero" if mode=="balanced" else "original concatenated-input fan-in initialization",
        helper_sha256=sha256(Path(__file__)),scope="new h first affine weights only; same MLP/form/width/depth/objective; no rank restriction")


@torch.no_grad()
def apply_h_initialization(probe,mode):
    """Call once just after expanding FG to FGH, before any h optimizer step."""
    if probe.stage!="fgh" or probe.h is None:
        raise ValueError("apply initialization only to a newly expanded h stage")
    spec=initialization_spec(probe.cfg.head_dim,probe.cfg.row_dim,mode)
    for network in probe.h:
        if (network.weights[0].shape[1]!=spec["query_dim"]+spec["row_dim"] or
                torch.count_nonzero(network.weights[-1]) or torch.count_nonzero(network.biases[-1])):
            raise ValueError("h must have the declared concat input and an untouched zero output")
    for network in probe.h:
        if mode=="balanced":
            network.weights[0][:,:spec["query_dim"]].mul_(spec["query_gain"])
            network.weights[0][:,spec["query_dim"]:].mul_(spec["row_gain"])
    return dict(spec,output_zero=True,applied_layers=len(probe.h),
        weight_rescale_flops=sum(network.weights[0].numel() for network in probe.h) if mode=="balanced" else 0)
