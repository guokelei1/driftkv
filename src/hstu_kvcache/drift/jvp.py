from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.func import functional_call, jvp

from ..models import HSTU, HSTUKVCache


def _flatten_kv(kv: HSTUKVCache) -> torch.Tensor:
    return torch.cat([kv.k.reshape(-1), kv.v.reshape(-1)])


def make_kv_func(model: HSTU, batch: dict, device: torch.device):
    """Return (params_dict, buffers_dict, forward_fn) for torch.func.

    forward_fn(params) -> flattened KV tensor, so jvp gives J_params . dtheta
    as a flattened KV-shaped tensor (the first-order drift estimate).
    """

    def kv_flat_fn(params: dict[str, torch.Tensor]) -> torch.Tensor:
        item_ids = batch["item_ids"].to(device)
        behaviors = batch["behaviors"].to(device)
        time_deltas = batch["time_deltas"].to(device)
        _, kv = functional_call(
            model,
            (params, {}),
            (item_ids, behaviors, time_deltas),
            dict(return_kv=True, return_hidden=False),
        )
        return _flatten_kv(kv)

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    return params, buffers, kv_flat_fn


@dataclass
class JVPEstimate:
    """Result of the naive per-user JVP drift estimate.

    drift_vec: flattened J_F(theta, x_u) . dtheta  (first-order linearisation).
    Compare its norm to the ground-truth ||F(theta+dtheta) - F(theta)|| to
    validate the linearisation, and compare its COST to a full recompute to
    validate Insight 4 (JVP should be ~2x a forward).
    """

    drift_vec: torch.Tensor  # [kv_numel]
    kv_numel: int
    forward_time_ms: float
    jvp_time_ms: float


def _cuda_time(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    return None, None


def _time_fn(fn, device: torch.device, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            fn()
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))
        return sum(times) / len(times)
    times = []
    for _ in range(repeats):
        s = time.perf_counter()
        fn()
        times.append((time.perf_counter() - s) * 1000.0)
    return sum(times) / len(times)


def naive_per_user_jvp(
    model: HSTU,
    batch: dict,
    dtheta: dict[str, torch.Tensor],
    device: torch.device,
    warmup: int = 2,
    repeats: int = 5,
) -> JVPEstimate:
    """Estimate drift as ||J_F . dtheta|| via a single forward-mode JVP.

    This is the Insight-4 baseline: one JVP costs ~2x a forward, so it is
    *more* expensive than recomputing the KV (1x forward). The whole research
    programme is about beating this cost via cross-user sharing / offline
    Fisher spectra.
    """
    was_training = model.training
    model.eval()
    try:
        params, _, kv_flat_fn = make_kv_func(model, batch, device)

        fwd_ms = _time_fn(lambda: torch.no_grad()(kv_flat_fn)(params), device, warmup, repeats)

        def _jvp():
            return jvp(kv_flat_fn, (params,), (dtheta,))

        jvp_ms = _time_fn(_jvp, device, warmup, repeats)
        _, tangent = _jvp()

        return JVPEstimate(
            drift_vec=tangent.detach(),
            kv_numel=tangent.numel(),
            forward_time_ms=fwd_ms,
            jvp_time_ms=jvp_ms,
        )
    finally:
        if was_training:
            model.train()


def dtheta_as_dict(model: nn.Module, dtheta_vec: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reshape a flattened dtheta vector into the per-parameter dict torch.func expects."""
    out: dict[str, torch.Tensor] = {}
    idx = 0
    for name, p in model.named_parameters():
        n = p.numel()
        out[name] = dtheta_vec[idx : idx + n].view_as(p).to(p.device)
        idx += n
    return out


def ground_truth_drift(
    model: HSTU,
    batch: dict,
    dtheta_vec: torch.Tensor,
    device: torch.device,
) -> tuple[HSTUKVCache, HSTUKVCache, dict]:
    """Compute the true drift F(theta+dtheta) - F(theta) by recomputing both.

    This is the oracle (upper-bound cost) used to score any cheap estimator.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            params = {k: v.detach().clone() for k, v in model.named_parameters()}
            _, kv0 = model(
                batch["item_ids"].to(device),
                batch["behaviors"].to(device),
                batch["time_deltas"].to(device),
                return_kv=True,
                return_hidden=False,
            )
            kv0 = kv0.detach()
            new_params = {k: v + dtheta_as_dict(model, dtheta_vec)[k] for k, v in params.items()}
            _, kv1 = functional_call(
                model,
                (new_params, {}),
                (
                    batch["item_ids"].to(device),
                    batch["behaviors"].to(device),
                    batch["time_deltas"].to(device),
                ),
                dict(return_kv=True, return_hidden=False),
            )
            kv1 = kv1.detach()
        metrics = kv0.drift_norm(kv1)
        metrics["drift_l2"] = (kv0.k - kv1.k).float().norm().item() + (kv0.v - kv1.v).float().norm().item()
        return kv0, kv1, metrics
    finally:
        if was_training:
            model.train()
