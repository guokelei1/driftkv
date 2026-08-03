from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist

from ..models import HSTU
from .low_rank import LowRankCacheAdapter, LowRankLayerAdapter


@torch.no_grad()
def capture_embedded_normed_states(
    model: HSTU,
    item_vectors: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if (
        item_vectors.ndim != 3
        or behaviors.shape != item_vectors.shape[:2]
        or time_deltas.shape != behaviors.shape
        or lengths.shape != (item_vectors.shape[0],)
        or item_vectors.shape[-1] != model.cfg.hidden_size
    ):
        raise ValueError("XP residual-fit embedded inputs differ")
    was_training = model.training
    model.eval()
    try:
        lengths = lengths.to(item_vectors.device)
        valid = (
            torch.arange(item_vectors.shape[1], device=item_vectors.device)[
                None, :
            ]
            < lengths[:, None]
        )
        x = model.combine_input_features(
            item_vectors,
            behaviors,
            time_deltas,
        )
        x = x * valid.unsqueeze(-1)
        values = []
        for block in model.blocks:
            values.append(block.norm(x).detach())
            x = block(x, return_kv=False)
            x = x * valid.unsqueeze(-1)
        return tuple(values)
    finally:
        if was_training:
            model.train()


def _all_reduce(value: torch.Tensor, process_group) -> torch.Tensor:
    if dist.is_initialized() and dist.get_world_size(process_group) > 1:
        dist.all_reduce(value, group=process_group)
    return value


@torch.no_grad()
def fit_distributed_low_rank_layer_adapter(
    features: torch.Tensor,
    residuals: torch.Tensor,
    *,
    rank: int,
    ridge: float,
    process_group=None,
) -> tuple[LowRankLayerAdapter, int]:
    if (
        features.ndim != 2
        or residuals.ndim != 2
        or features.shape[0] != residuals.shape[0]
        or features.shape[0] < 1
        or not 0 <= rank <= min(features.shape[1], residuals.shape[1])
        or ridge < 0
        or features.device != residuals.device
    ):
        raise ValueError("XP distributed residual-fit matrices differ")
    x = features.float()
    y = residuals.float()
    count = torch.tensor(
        [x.shape[0]],
        device=x.device,
        dtype=torch.float64,
    )
    sum_x = x.double().sum(dim=0)
    sum_y = y.double().sum(dim=0)
    _all_reduce(count, process_group)
    _all_reduce(sum_x, process_group)
    _all_reduce(sum_y, process_group)
    global_count = int(count.item())
    if global_count < 2:
        raise ValueError("XP residual fit needs at least two global tokens")
    feature_mean = (sum_x / count).float()
    residual_mean = (sum_y / count).float()
    centered_x = x - feature_mean
    centered_y = y - residual_mean
    gram = centered_x.T @ centered_x
    cross = centered_x.T @ centered_y
    _all_reduce(gram, process_group)
    _all_reduce(cross, process_group)
    gram = gram / global_count
    cross = cross / global_count
    scale = gram.diagonal().mean().clamp_min(
        torch.finfo(gram.dtype).eps
    )
    identity = torch.eye(
        gram.shape[0],
        device=gram.device,
        dtype=gram.dtype,
    )
    correction = torch.linalg.solve(
        gram + ridge * scale * identity,
        cross,
    )
    if rank:
        left, singular, right = torch.linalg.svd(
            correction,
            full_matrices=False,
        )
        left = left[:, :rank] * singular[:rank]
        right = right[:rank]
    else:
        left = torch.empty(
            correction.shape[0],
            0,
            device=correction.device,
        )
        right = torch.empty(
            0,
            correction.shape[1],
            device=correction.device,
        )
    return (
        LowRankLayerAdapter(
            feature_mean=feature_mean,
            residual_mean=residual_mean,
            left=left,
            right=right,
        ),
        global_count,
    )


def _sample_pair(
    features: torch.Tensor,
    residuals: torch.Tensor,
    maximum_tokens: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.shape[0] <= maximum_tokens:
        return features, residuals
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected = torch.randperm(
        features.shape[0],
        generator=generator,
    )[:maximum_tokens].sort().values
    return features[selected], residuals[selected]


@torch.no_grad()
def fit_distributed_low_rank_cache_adapter(
    feature_layers: Sequence[torch.Tensor],
    residual_layers: Sequence[torch.Tensor],
    *,
    rank: int,
    ridge: float,
    maximum_tokens_per_rank: int,
    seed: int,
    device: torch.device,
    process_group=None,
) -> tuple[LowRankCacheAdapter, dict[str, object]]:
    if (
        not feature_layers
        or len(feature_layers) != len(residual_layers)
        or maximum_tokens_per_rank < 2
        or seed < 0
    ):
        raise ValueError("XP residual-fit layer family differs")
    layers = []
    local_tokens = []
    global_tokens = []
    for layer, (features, residuals) in enumerate(
        zip(feature_layers, residual_layers, strict=True)
    ):
        sampled_features, sampled_residuals = _sample_pair(
            features,
            residuals,
            maximum_tokens_per_rank,
            seed + layer * 104729,
        )
        local_tokens.append(sampled_features.shape[0])
        fitted, count = fit_distributed_low_rank_layer_adapter(
            sampled_features.to(device),
            sampled_residuals.to(device),
            rank=rank,
            ridge=ridge,
            process_group=process_group,
        )
        layers.append(fitted)
        global_tokens.append(count)
    adapter = LowRankCacheAdapter(
        layers=tuple(layers),
        ridge=ridge,
    )
    return adapter, {
        "rank": rank,
        "ridge": ridge,
        "maximum_tokens_per_rank": maximum_tokens_per_rank,
        "local_sampled_tokens_per_layer": local_tokens,
        "global_sampled_tokens_per_layer": global_tokens,
        "layers": len(layers),
        "adapter_parameter_numel": adapter.numel,
        "labels_used": False,
        "recommendation_metrics_used": False,
    }
