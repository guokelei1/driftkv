"""Independent head-local residual maps for RecFlow Figure 9 preparations.

The backbone supplies detached prerelease cache pairs. Neither future ranking
labels nor query outputs enter fitting. Six fixed configurations are trained
independently on prefixes of the same prerelease UID order; all eight epochs
run. The channel order is K/V, layer, head. No fingerprint/permission gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import math
import time

import torch
import torch.nn.functional as F

from hstu_kvcache.models.kv_cache import HSTUKVCache
from .cost import fitting_flops


DEFAULT_SETTINGS = dict(epochs=8, batch_rows=8192, learning_rate=.001,
                        weight_decay=.0001, beta1=.9, beta2=.999,
                        epsilon=1e-8, normalization_floor=1e-5)
MAX_POLICIES = [dict(tag=f"kv_map_u{u}_w{w}", method="kv_map", fit_users=u, width=w)
                for u, w in [(32, 40), (64, 64), (96, 96), (128, 128), (192, 160), (256, 184)]]


@dataclass(frozen=True)
class MapWeights:
    layers: int
    heads: int
    dim: int
    tensors: tuple

    @property
    def width(self):
        return self.tensors[2].shape[-1]

    def to(self, device):
        return MapWeights(self.layers, self.heads, self.dim,
                          tuple(t.detach().to(device) for t in self.tensors))

    def cpu(self):
        return self.to("cpu")


def normalized_forward(x, w1, b1, w2, b2):
    return torch.bmm(F.silu(torch.bmm(x, w1) + b1), w2) + b2


@torch.no_grad()
def normalize_inplace(x, floor):
    mean = x.mean(1, keepdim=True)
    x.sub_(mean)
    scale = x.square().mean(1, keepdim=True).sqrt().clamp_min(floor)
    x.div_(scale)
    return mean, scale


def fit_one(x, y, width, settings=None, seed=17):
    """Mutates owned source/delta workspaces, never the input caches/backbone."""
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    if x.ndim != 3 or x.shape != y.shape or x.shape[1] < 1:
        raise ValueError("Nonempty paired [channel, observation, head_dim] tensors required")
    if x.requires_grad or y.requires_grad:
        raise ValueError("Cache teacher inputs must be detached")
    if min(width, settings["epochs"], settings["batch_rows"], settings["normalization_floor"]) <= 0:
        raise ValueError("Positive width, epochs, minibatch size and normalization floor required")
    channels, observations, dim = x.shape
    with torch.no_grad():
        mx, sx = normalize_inplace(x, settings["normalization_floor"])
        my, sy = normalize_inplace(y, settings["normalization_floor"])
        generator = torch.Generator(device=x.device).manual_seed(seed)
        w1 = torch.nn.Parameter(torch.randn(channels, dim, width, device=x.device,
                                            dtype=x.dtype, generator=generator) * math.sqrt(2 / dim))
        b1 = torch.nn.Parameter(torch.zeros(channels, 1, width, device=x.device, dtype=x.dtype))
        w2 = torch.nn.Parameter(torch.zeros(channels, width, dim, device=x.device, dtype=x.dtype))
        b2 = torch.nn.Parameter(torch.zeros(channels, 1, dim, device=x.device, dtype=x.dtype))
        probe_x, probe_y = x[:, :min(2048, observations)], y[:, :min(2048, observations)]
        initial = float(F.mse_loss(normalized_forward(probe_x, w1, b1, w2, b2), probe_y))
    params = (w1, b1, w2, b2)
    optimizer = torch.optim.AdamW(params, lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"], betas=(settings["beta1"], settings["beta2"]),
        eps=settings["epsilon"], foreach=False, fused=False)
    steps, batch_sizes = 0, []
    with torch.enable_grad():
        for epoch in range(settings["epochs"]):
            order = torch.randperm(observations, device=x.device, generator=generator)
            for start in range(0, observations, settings["batch_rows"]):
                index = order[start:start + settings["batch_rows"]]
                bx, by = x.index_select(1, index), y.index_select(1, index)
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(normalized_forward(bx, *params), by)
                loss.backward()
                optimizer.step()
                steps += 1
                if epoch == 0:
                    batch_sizes.append(len(index))
    with torch.no_grad():
        final = float(F.mse_loss(normalized_forward(probe_x, *params), probe_y))
    if not math.isfinite(final):
        raise RuntimeError("Nonfinite prerelease teacher fit")
    weights = tuple(t.detach() for t in (mx, sx, w1, b1, w2, b2, my, sy))
    return weights, dict(steps=steps, parameters=sum(t.numel() for t in params),
        observations=observations, epoch_batch_sizes=batch_sizes,
        normalized_teacher_probe_mse_before=initial, normalized_teacher_probe_mse_after=final,
        diagnostic_scope="Fixed first <=2048 prerelease fit rows; no validation selection or future labels")


def _cache_channels(cache, heads):
    layers, batch, n, hidden = cache.k.shape
    if cache.v.shape != cache.k.shape or cache.seq_len != n or hidden % heads:
        raise ValueError("Cache shape is inconsistent with declared head geometry")
    dim = hidden // heads
    return (torch.stack((cache.k, cache.v)).reshape(2, layers, batch, n, heads, dim)
            .permute(0, 1, 4, 2, 3, 5).reshape(2 * layers * heads, batch * n, dim))


def fit_cache_pairs(cache_pairs, policies=None, *, device="cpu", num_heads=10,
                    settings=None, seed=17, teacher_flops=None, progress=None):
    """Capture each user's teacher once on CPU, then fit all configurations.

    ``cache_pairs`` yields (old,new), one unpadded user each, in fixed fit UID
    order. ``teacher_flops`` is a cumulative per-user list, a policy-tag dict,
    or a list parallel to policies; it counts old+current cache capture for
    each standalone point. Returned maps are portable CPU weights; move each
    once via weights.to(device) before population inference.
    """
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    policies = [p for p in (MAX_POLICIES if policies is None else policies)
                if p.get("method", "kv_map") in ("kv_map", "alignment")]
    if not policies:
        return dict(maps={}, costs={}, profiles={}, capture={"users": 0, "observations": 0})
    max_users = max(p["fit_users"] for p in policies)
    source_x, source_y, counts = [], [], []
    geometry = None
    observations = 0
    started = time.perf_counter()
    for pair in islice(cache_pairs, max_users):
        old, new = pair
        if old.k.shape != new.k.shape or old.k.shape[1] != 1:
            raise ValueError("Capture one matched unpadded user per pair")
        current_geometry = (old.k.shape[0], num_heads, old.k.shape[-1] // num_heads)
        if geometry is not None and current_geometry != geometry:
            raise ValueError("All pairs must have the same model geometry")
        geometry = current_geometry
        with torch.no_grad():
            x = _cache_channels(old, num_heads).detach().float().cpu()
            # Subtract on the capture device before CPU offload, as Figure9b.
            y = (_cache_channels(new, num_heads) - _cache_channels(old, num_heads)).detach().float().cpu()
        source_x.append(x)
        source_y.append(y)
        observations += x.shape[1]
        counts.append(observations)
    if len(counts) < max_users:
        raise ValueError(f"Need {max_users} fit users, received {len(counts)}")
    capture_seconds = time.perf_counter() - started
    maps, costs, profiles = {}, {}, {}
    layers, heads, dim = geometry
    for index, p in enumerate(policies):
        begin = time.perf_counter()
        users, width, tag = p["fit_users"], p["width"], p["tag"]
        if users < 1 or counts[users - 1] == 0:
            raise ValueError("Each fit policy needs nonempty prerelease observations")
        # torch.cat owns these workspaces even for CPU execution; normalization
        # never modifies source pairs retained for the other configurations.
        x = torch.cat(source_x[:users], 1).to(device)
        y = torch.cat(source_y[:users], 1).to(device)
        weights, profile = fit_one(x, y, width, settings, seed + 1000 * users + width)
        if teacher_flops is None:
            teacher = 0
        elif isinstance(teacher_flops, dict):
            teacher = teacher_flops[tag]
        elif len(teacher_flops) == max_users:
            teacher = teacher_flops[users - 1]
        elif len(teacher_flops) == len(policies):
            teacher = teacher_flops[index]
        else:
            raise ValueError("Teacher FLOPs must cover user prefixes or policy points")
        audit = fitting_flops(counts[users - 1], width, settings["epochs"],
                              settings["batch_rows"], teacher, 2 * layers * heads, dim)
        if audit["steps"] != profile["steps"] or audit["parameters"] != profile["parameters"]:
            raise RuntimeError("Fit execution differs from arithmetic ledger")
        maps[tag] = MapWeights(layers, heads, dim, weights).cpu()
        costs[tag] = audit["total_flops"]
        profiles[tag] = dict(**profile, flop_breakdown=audit, seconds=time.perf_counter() - begin,
                             fit_users=users, width=width, teacher_cost_supplied=teacher_flops is not None)
        if progress is not None:
            progress(tag, profiles[tag])
        del x, y, weights
    return dict(maps=maps, costs=costs, profiles=profiles,
                capture=dict(users=max_users, observations=observations, cumulative_rows=counts,
                             seconds=capture_seconds, total_seconds=time.perf_counter() - started,
                             layers=layers, heads=heads, head_dim=dim))


@torch.no_grad()
def apply_map(cache, weights):
    if cache.k.shape[0] != weights.layers or cache.k.shape[-1] != weights.heads * weights.dim:
        raise ValueError("Map geometry differs from cache")
    mx, sx, w1, b1, w2, b2, my, sy = weights.tensors
    x = _cache_channels(cache, weights.heads)
    mapped = x + (normalized_forward((x - mx) / sx, w1, b1, w2, b2) * sy + my)
    n, batch = cache.seq_len, cache.k.shape[1]
    shaped = (mapped.reshape(2, weights.layers, weights.heads, batch, n, weights.dim)
              .permute(0, 1, 3, 4, 2, 5).reshape(2, weights.layers, batch, n, weights.heads * weights.dim))
    return HSTUKVCache(shaped[0], shaped[1], n)


@torch.no_grad()
def apply_maps(cache, weights_sequence):
    """One independent map per batch slot; equal widths share a bmm call."""
    if len(weights_sequence) != cache.k.shape[1]:
        raise ValueError("Provide one map for each method-batch slot")
    if not weights_sequence:
        return cache.clone()
    keys, values = cache.k.clone(), cache.v.clone()
    groups = {}
    for index, weights in enumerate(weights_sequence):
        groups.setdefault((weights.layers, weights.heads, weights.dim, weights.width), []).append(index)
    for (layers, heads, dim, _), indices in groups.items():
        if (layers, heads * dim) != (cache.k.shape[0], cache.k.shape[-1]):
            raise ValueError("Map geometry differs from cache")
        selected = torch.as_tensor(indices, device=cache.k.device)
        part = HSTUKVCache(cache.k.index_select(1, selected), cache.v.index_select(1, selected), cache.seq_len)
        # Channel-major -> method-major, so all channel weights stay independent.
        x = _cache_channels(part, heads).reshape(2 * layers * heads, len(indices), cache.seq_len, dim)
        x = x.permute(1, 0, 2, 3).reshape(-1, cache.seq_len, dim)
        mx, sx, w1, b1, w2, b2, my, sy = tuple(torch.cat([weights_sequence[i].tensors[j]
            for i in indices], 0) for j in range(8))
        y = x + (normalized_forward((x - mx) / sx, w1, b1, w2, b2) * sy + my)
        shaped = (y.reshape(len(indices), 2, layers, heads, cache.seq_len, dim)
                  .permute(1, 2, 0, 4, 3, 5).reshape(2, layers, len(indices), cache.seq_len, heads * dim))
        keys.index_copy_(1, selected, shaped[0])
        values.index_copy_(1, selected, shaped[1])
    return HSTUKVCache(keys, values, cache.seq_len)
