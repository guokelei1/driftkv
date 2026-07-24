from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models import HSTU, HSTUKVCache
from .layerwise import LayerwiseCacheState


@dataclass(frozen=True)
class LowRankLayerAdapter:
    feature_mean: torch.Tensor
    residual_mean: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor

    @property
    def rank(self) -> int:
        return self.left.shape[1]

    @property
    def numel(self) -> int:
        return sum(
            value.numel()
            for value in (
                self.feature_mean,
                self.residual_mean,
                self.left,
                self.right,
            )
        )

    def truncate(self, rank: int) -> LowRankLayerAdapter:
        if not 0 <= rank <= self.rank:
            raise ValueError(f"rank must be in [0, {self.rank}]")
        return LowRankLayerAdapter(
            feature_mean=self.feature_mean,
            residual_mean=self.residual_mean,
            left=self.left[:, :rank],
            right=self.right[:rank],
        )


@dataclass(frozen=True)
class LowRankCacheAdapter:
    layers: tuple[LowRankLayerAdapter, ...]
    ridge: float

    @property
    def rank(self) -> int:
        ranks = {layer.rank for layer in self.layers}
        if len(ranks) != 1:
            raise ValueError("all layer adapters must have the same rank")
        return next(iter(ranks))

    @property
    def numel(self) -> int:
        return sum(layer.numel for layer in self.layers)

    def truncate(self, rank: int) -> LowRankCacheAdapter:
        return LowRankCacheAdapter(
            layers=tuple(layer.truncate(rank) for layer in self.layers),
            ridge=self.ridge,
        )


@dataclass(frozen=True)
class CompiledCacheAdapter:
    weights: torch.Tensor
    biases: torch.Tensor
    source_rank: int
    ridge: float

    @property
    def numel(self) -> int:
        return self.weights.numel() + self.biases.numel()

    @property
    def nbytes(self) -> int:
        return (
            self.weights.numel() * self.weights.element_size()
            + self.biases.numel() * self.biases.element_size()
        )


def _fused_current_projections(
    model: HSTU,
    state: LayerwiseCacheState,
) -> torch.Tensor:
    normed = torch.stack(state.normed_states)
    weights = torch.stack(
        [
            torch.cat(
                (block.attn.k_proj.weight, block.attn.v_proj.weight),
                dim=0,
            )
            for block in model.blocks
        ]
    )
    projected = torch.bmm(
        normed.flatten(1, 2),
        weights.transpose(1, 2),
    ).unflatten(1, normed.shape[1:3])
    if any(
        block.attn.k_proj.bias is not None or block.attn.v_proj.bias is not None
        for block in model.blocks
    ):
        biases = []
        for block in model.blocks:
            k_bias = block.attn.k_proj.bias
            v_bias = block.attn.v_proj.bias
            if k_bias is None:
                k_bias = torch.zeros(
                    block.attn.k_proj.out_features,
                    device=projected.device,
                    dtype=projected.dtype,
                )
            if v_bias is None:
                v_bias = torch.zeros(
                    block.attn.v_proj.out_features,
                    device=projected.device,
                    dtype=projected.dtype,
                )
            biases.append(torch.cat((k_bias, v_bias)))
        projected = projected + torch.stack(biases)[:, None, None, :]
    positions = torch.arange(state.kv.seq_len, device=state.lengths.device)
    valid = positions.unsqueeze(0) < state.lengths.unsqueeze(1)
    return projected * valid.unsqueeze(0).unsqueeze(-1)


@torch.no_grad()
def migrate_fused_projection_cache(
    model: HSTU,
    state: LayerwiseCacheState,
) -> HSTUKVCache:
    num_layers = len(model.blocks)
    if len(state.normed_states) != num_layers:
        raise ValueError("state and model depth differ")
    was_training = model.training
    model.eval()
    try:
        projected = _fused_current_projections(model, state)
        width = projected.shape[-1] // 2
        return HSTUKVCache(
            k=projected[..., :width],
            v=projected[..., width:],
            seq_len=state.kv.seq_len,
        )
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def compile_projection_cache_adapter(
    model: HSTU,
) -> CompiledCacheAdapter:
    weights = []
    biases = []
    for block in model.blocks:
        weight = torch.cat(
            (block.attn.k_proj.weight, block.attn.v_proj.weight),
            dim=0,
        ).transpose(0, 1)
        bias = torch.zeros(
            weight.shape[1],
            device=weight.device,
            dtype=torch.float32,
        )
        if block.attn.k_proj.bias is not None:
            bias[: block.attn.k_proj.out_features] = block.attn.k_proj.bias
        if block.attn.v_proj.bias is not None:
            bias[block.attn.k_proj.out_features :] = block.attn.v_proj.bias
        weights.append(weight.float())
        biases.append(bias)
    return CompiledCacheAdapter(
        weights=torch.stack(weights),
        biases=torch.stack(biases),
        source_rank=0,
        ridge=0.0,
    )


@torch.no_grad()
def compile_low_rank_cache_adapter(
    model: HSTU,
    adapter: LowRankCacheAdapter,
) -> CompiledCacheAdapter:
    if len(model.blocks) != len(adapter.layers):
        raise ValueError("adapter and model depth differ")
    base = compile_projection_cache_adapter(model)
    weights = []
    biases = []
    for index, layer in enumerate(adapter.layers):
        correction_weight = layer.left @ layer.right
        weights.append(base.weights[index] + correction_weight)
        biases.append(
            base.biases[index]
            + layer.residual_mean
            - layer.feature_mean @ correction_weight
        )
    return CompiledCacheAdapter(
        weights=torch.stack(weights),
        biases=torch.stack(biases),
        source_rank=adapter.rank,
        ridge=adapter.ridge,
    )


@torch.no_grad()
def migrate_compiled_low_rank_cache(
    state: LayerwiseCacheState,
    adapter: CompiledCacheAdapter,
) -> HSTUKVCache:
    if len(state.normed_states) != adapter.weights.shape[0]:
        raise ValueError("state and compiled adapter depth differ")
    normed = torch.stack(state.normed_states)
    projected = torch.bmm(
        normed.float().flatten(1, 2),
        adapter.weights,
    ).unflatten(1, normed.shape[1:3])
    projected = projected + adapter.biases[:, None, None, :]
    positions = torch.arange(state.kv.seq_len, device=state.lengths.device)
    valid = positions.unsqueeze(0) < state.lengths.unsqueeze(1)
    projected = projected * valid.unsqueeze(0).unsqueeze(-1)
    projected = projected.to(normed.dtype)
    width = projected.shape[-1] // 2
    return HSTUKVCache(
        k=projected[..., :width],
        v=projected[..., width:],
        seq_len=state.kv.seq_len,
    )


@torch.no_grad()
def fit_low_rank_layer_adapter(
    features: torch.Tensor,
    residuals: torch.Tensor,
    rank: int,
    ridge: float,
) -> LowRankLayerAdapter:
    if features.ndim != 2 or residuals.ndim != 2:
        raise ValueError("features and residuals must be matrices")
    if features.shape[0] != residuals.shape[0] or features.shape[0] == 0:
        raise ValueError("features and residuals must have the same nonzero row count")
    max_rank = min(features.shape[1], residuals.shape[1])
    if not 0 <= rank <= max_rank:
        raise ValueError(f"rank must be in [0, {max_rank}]")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")

    x = features.float()
    y = residuals.float()
    feature_mean = x.mean(dim=0)
    residual_mean = y.mean(dim=0)
    x = x - feature_mean
    y = y - residual_mean
    gram = x.transpose(0, 1) @ x / x.shape[0]
    cross = x.transpose(0, 1) @ y / x.shape[0]
    scale = gram.diagonal().mean().clamp_min(torch.finfo(gram.dtype).eps)
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    weights = torch.linalg.solve(gram + ridge * scale * identity, cross)
    u, singular, vh = torch.linalg.svd(weights, full_matrices=False)
    left = u[:, :rank] * singular[:rank]
    right = vh[:rank]
    return LowRankLayerAdapter(
        feature_mean=feature_mean,
        residual_mean=residual_mean,
        left=left,
        right=right,
    )


@torch.no_grad()
def fit_low_rank_cache_adapter(
    feature_layers: list[torch.Tensor],
    residual_layers: list[torch.Tensor],
    rank: int,
    ridge: float = 1e-3,
) -> LowRankCacheAdapter:
    if len(feature_layers) != len(residual_layers) or not feature_layers:
        raise ValueError("feature and residual layer lists must have equal nonzero length")
    return LowRankCacheAdapter(
        layers=tuple(
            fit_low_rank_layer_adapter(features, residuals, rank, ridge)
            for features, residuals in zip(feature_layers, residual_layers, strict=True)
        ),
        ridge=ridge,
    )


@torch.no_grad()
def migrate_low_rank_cache(
    model: HSTU,
    state: LayerwiseCacheState,
    adapter: LowRankCacheAdapter,
) -> HSTUKVCache:
    num_layers = len(model.blocks)
    if len(state.normed_states) != num_layers or len(adapter.layers) != num_layers:
        raise ValueError("state, adapter, and model depth differ")
    was_training = model.training
    model.eval()
    try:
        normed = torch.stack(state.normed_states)
        feature_mean = torch.stack(
            [layer.feature_mean for layer in adapter.layers]
        )
        left = torch.stack([layer.left for layer in adapter.layers])
        right = torch.stack([layer.right for layer in adapter.layers])
        residual_mean = torch.stack(
            [layer.residual_mean for layer in adapter.layers]
        )
        centered = normed.float() - feature_mean[:, None, None, :]
        correction = torch.bmm(
            centered.flatten(1, 2),
            left,
        )
        correction = torch.bmm(correction, right)
        correction = correction.unflatten(1, normed.shape[1:3])
        correction = correction + residual_mean[:, None, None, :]
        positions = torch.arange(state.kv.seq_len, device=state.lengths.device)
        valid = positions.unsqueeze(0) < state.lengths.unsqueeze(1)
        correction = correction * valid.unsqueeze(0).unsqueeze(-1)
        migrated = _fused_current_projections(model, state)
        migrated = migrated + correction.to(migrated.dtype)
        width = migrated.shape[-1] // 2
        return HSTUKVCache(
            k=migrated[..., :width],
            v=migrated[..., width:],
            seq_len=state.kv.seq_len,
        )
    finally:
        if was_training:
            model.train()
