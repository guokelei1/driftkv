"""Defect-first finite-release replay for one Parent-to-Current migration.

The approximation budget in this module is assigned to two different
scientific objects at every Transformer layer:

``B_l``
    a small approximation of the Parent/base trajectory; and
``D_l``
    a larger approximation of the *release defect* from Parent to Current.

The Current input is never compressed as one absolute state.  It is evaluated
from the exact factor sum ``B_l + D_l``.  After the native Parent and Current
blocks have run, the next defect is formed before compression::

    B_{l+1} = C_b(F_l^P(B_l))
    D_{l+1} = C_d(F_l^C(B_l + D_l) - F_l^P(B_l))

For the sole preflight configuration, ``rank(B)=2`` and ``rank(D)=4``.  The
Current block therefore sees rank six while the two simultaneously active
block inputs use a total of eight modes, matching the ordinary rank-4/rank-4
paired control.  This is a mechanism probe for release-defect coordinate
closure, not a claim that low-rank factorization or base-plus-delta storage is
new.

No function accepts Current-Exact upper-layer state, candidates, or labels.
Current Exact is used only by the runner's evaluation path.  The final block
forms K/V only because its post-block state has no migration consumer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from hstu_kvcache.models import HSTUKVCache
from insight_two.matrix_free_input_range import (
    MatrixFreeHSTUInput,
    matrix_free_input_cost,
    matrix_free_randomized_token_factors,
)
from insight_two.mode_space_replay import (
    FactorizedCacheLayer,
    FactorizedReplay,
    SharedModeSplice,
    TokenModeFactors,
    _compress_token_modes,
    _factorized_legacy_attention_impl,
    factorized_rmsnorm,
    randomized_token_basis,
)
from insight_two.paired_native_response import medium_paired_native_response_cost


@dataclass(frozen=True)
class MatrixFreeReleaseDefectInput:
    """Matrix-free operator for ``X_current - X_parent``.

    ``base_features`` is exposed only as a device anchor required by the
    generic matrix-free range finder.  Its values are never interpreted as the
    release defect; every operator application explicitly subtracts the two
    release-specific input operators.
    """

    parent: MatrixFreeHSTUInput
    current: MatrixFreeHSTUInput

    def __post_init__(self) -> None:
        if self.parent.base_features.shape != self.current.base_features.shape:
            raise ValueError("Parent and Current input operator shapes differ")
        if self.parent.time_features.shape != self.current.time_features.shape:
            raise ValueError("Parent and Current temporal feature shapes differ")
        if self.parent.base_features.device != self.current.base_features.device:
            raise ValueError("Parent and Current input operators use different devices")

    @property
    def base_features(self) -> torch.Tensor:
        """Return an allocation/device anchor, not a materialized defect."""

        return self.current.base_features

    @property
    def history_length(self) -> int:
        return self.parent.history_length

    @property
    def hidden_size(self) -> int:
        return self.parent.hidden_size

    @torch.inference_mode()
    def right_multiply(self, right: torch.Tensor) -> torch.Tensor:
        return self.current.right_multiply(right) - self.parent.right_multiply(right)

    @torch.inference_mode()
    def transpose_multiply(self, left: torch.Tensor) -> torch.Tensor:
        return self.current.transpose_multiply(left) - self.parent.transpose_multiply(left)


@dataclass(frozen=True)
class DefectFirstReplay:
    """Parent base, constructed Current, and carried release-defect factors."""

    parent: FactorizedReplay
    current: FactorizedReplay
    defect_input_factors: tuple[TokenModeFactors, ...]
    post_block_defects: tuple[torch.Tensor, ...]
    base_rank: int
    defect_rank: int

    def __post_init__(self) -> None:
        layers = len(self.parent.layers)
        if layers < 1 or len(self.current.layers) != layers:
            raise ValueError("Parent and Current replay layer counts differ")
        if len(self.defect_input_factors) != layers:
            raise ValueError("one defect input factor is required per layer")
        if len(self.post_block_defects) != layers - 1:
            raise ValueError("only nonterminal blocks may expose a post-block defect")
        if self.base_rank < 1 or self.defect_rank < 1:
            raise ValueError("base and defect ranks must be positive")
        for parent_layer, current_layer, defect in zip(
            self.parent.layers,
            self.current.layers,
            self.defect_input_factors,
            strict=True,
        ):
            if parent_layer.rank != self.base_rank:
                raise ValueError("Parent layer rank differs from the base budget")
            if defect.rank != self.defect_rank:
                raise ValueError("defect factor rank differs from the defect budget")
            if current_layer.rank != self.base_rank + self.defect_rank:
                raise ValueError("Current layer does not carry base-plus-defect factors")


def add_token_factors(
    base: TokenModeFactors,
    defect: TokenModeFactors,
) -> TokenModeFactors:
    """Represent ``base.materialize() + defect.materialize()`` exactly."""

    if base.left.shape[:2] != defect.left.shape[:2]:
        raise ValueError("base and defect token dimensions differ")
    if base.right.shape[0] != defect.right.shape[0]:
        raise ValueError("base and defect batches differ")
    if base.right.shape[2] != defect.right.shape[2]:
        raise ValueError("base and defect hidden widths differ")
    if base.left.device != defect.left.device or base.left.dtype != defect.left.dtype:
        raise ValueError("base and defect factors must share device and dtype")
    return TokenModeFactors(
        left=torch.cat((base.left, defect.left), dim=2),
        right=torch.cat((base.right, defect.right), dim=1),
    )


def _validate_models(parent_model, current_model) -> None:
    if parent_model.training or current_model.training:
        raise ValueError("defect-first replay requires both models in eval mode")
    if not parent_model.blocks or len(parent_model.blocks) != len(current_model.blocks):
        raise ValueError("release models must have the same positive layer count")
    if parent_model.cfg.block_variant != "legacy" or current_model.cfg.block_variant != "legacy":
        raise ValueError("defect-first replay currently covers legacy blocks")
    if parent_model.cfg.hidden_size != current_model.cfg.hidden_size:
        raise ValueError("release models use different hidden widths")


def _validate_initial_factors(
    parent_model,
    parent_initial: TokenModeFactors,
    defect_initial: TokenModeFactors,
    *,
    base_rank: int,
    defect_rank: int,
) -> tuple[int, int]:
    if parent_initial.rank != base_rank or defect_initial.rank != defect_rank:
        raise ValueError("initial factor ranks differ from the frozen budgets")
    if parent_initial.left.shape[:2] != defect_initial.left.shape[:2]:
        raise ValueError("initial base and defect token dimensions differ")
    if parent_initial.right.shape[2] != parent_model.cfg.hidden_size:
        raise ValueError("initial Parent factors use the wrong hidden width")
    if defect_initial.right.shape[2] != parent_model.cfg.hidden_size:
        raise ValueError("initial defect factors use the wrong hidden width")
    return int(parent_initial.left.shape[0]), int(parent_initial.left.shape[1])


def _factorized_block_update(
    block,
    input_factors: TokenModeFactors,
) -> tuple[torch.Tensor, FactorizedCacheLayer]:
    """Return only the dense nonlinear update and factorized K/V."""

    normalized = factorized_rmsnorm(block.norm, input_factors)
    attention_output, key_factors, value_factors = _factorized_legacy_attention_impl(
        block, normalized
    )
    if block.gating == "silu_gate":
        gate_core = normalized.right @ block.gate_proj.weight.transpose(0, 1)
        update = attention_output * F.silu(normalized.left @ gate_core)
    elif block.gating == "glu":
        gate_core = normalized.right @ block.gate_proj.weight.transpose(0, 1)
        update = attention_output * torch.sigmoid(normalized.left @ gate_core)
    elif block.gating == "none":
        update = attention_output
    else:
        raise ValueError("defect-first replay does not cover the FFN variant")
    return update, FactorizedCacheLayer(
        left=normalized.left,
        key_core=key_factors.right,
        value_core=value_factors.right,
    )


def _factorized_kv_only(
    block,
    input_factors: TokenModeFactors,
) -> FactorizedCacheLayer:
    """Form terminal K/V without Q, attention, gate, or post-block state."""

    normalized = factorized_rmsnorm(block.norm, input_factors)
    attention = block.attn
    if attention.k_proj.bias is not None or attention.v_proj.bias is not None:
        raise ValueError("factorized terminal K/V requires bias-free projections")
    return FactorizedCacheLayer(
        left=normalized.left,
        key_core=normalized.right @ attention.k_proj.weight.transpose(0, 1),
        value_core=normalized.right @ attention.v_proj.weight.transpose(0, 1),
    )


@torch.inference_mode()
def matrix_free_defect_first_initial_factors(
    parent_operator: MatrixFreeHSTUInput,
    current_operator: MatrixFreeHSTUInput,
    *,
    base_rank: int = 2,
    defect_rank: int = 4,
    sketch_oversample: int = 4,
    sketch_power_iterations: int = 1,
    sketch_seed: int = 17,
) -> tuple[TokenModeFactors, TokenModeFactors]:
    """Construct the initial Parent and release-defect factors without dense X."""

    parent_factors = matrix_free_randomized_token_factors(
        parent_operator,
        rank=base_rank,
        oversample=sketch_oversample,
        power_iterations=sketch_power_iterations,
        seed=sketch_seed,
    )
    defect_factors = matrix_free_randomized_token_factors(
        MatrixFreeReleaseDefectInput(parent_operator, current_operator),
        rank=defect_rank,
        oversample=sketch_oversample,
        power_iterations=sketch_power_iterations,
        seed=sketch_seed,
    )
    return parent_factors, defect_factors


@torch.inference_mode()
def defect_first_release_replay(
    parent_model,
    current_model,
    parent_embedded_history: torch.Tensor,
    current_embedded_history: torch.Tensor,
    *,
    base_rank: int = 2,
    defect_rank: int = 4,
    compression: Literal["exact_svd", "fixed_range_finder"] = "fixed_range_finder",
    sketch_oversample: int = 4,
    sketch_power_iterations: int = 1,
    sketch_seed: int = 17,
) -> DefectFirstReplay:
    """Dense semantic entry point used for exact-limit and recurrence tests."""

    _validate_models(parent_model, current_model)
    if parent_embedded_history.shape != current_embedded_history.shape:
        raise ValueError("Parent and Current embedded histories differ in shape")
    if parent_embedded_history.ndim != 3:
        raise ValueError("embedded histories must have shape [B,N,H]")
    initial_parent = _compress_token_modes(
        parent_embedded_history,
        rank=base_rank,
        compression=compression,
        oversample=sketch_oversample,
        power_iterations=sketch_power_iterations,
        seed=sketch_seed,
    )
    initial_defect = _compress_token_modes(
        current_embedded_history - parent_embedded_history,
        rank=defect_rank,
        compression=compression,
        oversample=sketch_oversample,
        power_iterations=sketch_power_iterations,
        seed=sketch_seed,
    )
    return defect_first_replay_from_initial_factors(
        parent_model,
        current_model,
        initial_parent,
        initial_defect,
        base_rank=base_rank,
        defect_rank=defect_rank,
        compression=compression,
        sketch_oversample=sketch_oversample,
        sketch_power_iterations=sketch_power_iterations,
        sketch_seed=sketch_seed,
    )


@torch.inference_mode()
def defect_first_replay_from_initial_factors(
    parent_model,
    current_model,
    parent_initial: TokenModeFactors,
    defect_initial: TokenModeFactors,
    *,
    base_rank: int = 2,
    defect_rank: int = 4,
    compression: Literal["exact_svd", "fixed_range_finder"] = "fixed_range_finder",
    sketch_oversample: int = 4,
    sketch_power_iterations: int = 1,
    sketch_seed: int = 17,
) -> DefectFirstReplay:
    """Advance the Parent base and finite-release defect in separate coordinates."""

    _validate_models(parent_model, current_model)
    _, length = _validate_initial_factors(
        parent_model,
        parent_initial,
        defect_initial,
        base_rank=base_rank,
        defect_rank=defect_rank,
    )

    base_factors = parent_initial
    defect_factors = defect_initial
    parent_inputs: list[TokenModeFactors] = []
    current_inputs: list[TokenModeFactors] = []
    defect_inputs: list[TokenModeFactors] = []
    parent_layers: list[FactorizedCacheLayer] = []
    current_layers: list[FactorizedCacheLayer] = []
    parent_materialized: list[tuple[torch.Tensor, torch.Tensor]] = []
    current_materialized: list[tuple[torch.Tensor, torch.Tensor]] = []
    post_block_defects: list[torch.Tensor] = []

    for layer, (parent_block, current_block) in enumerate(
        zip(parent_model.blocks, current_model.blocks, strict=True)
    ):
        parent_input = base_factors
        defect_input = defect_factors
        current_factors = add_token_factors(parent_input, defect_input)
        terminal = layer + 1 == len(parent_model.blocks)
        if terminal:
            parent_cache = _factorized_kv_only(parent_block, parent_input)
            current_cache = _factorized_kv_only(current_block, current_factors)
        else:
            parent_update, parent_cache = _factorized_block_update(
                parent_block, parent_input
            )
            current_update, current_cache = _factorized_block_update(
                current_block, current_factors
            )
            # Algebraically, Current residual minus Parent residual is exactly
            # the carried defect.  Constructing the next states this way makes
            # it impossible for Parent-dominant modes to consume defect rank.
            parent_state = parent_input.materialize() + parent_update
            defect_state = (
                defect_input.materialize() + current_update - parent_update
            )
            post_block_defects.append(defect_state)
            base_factors = _compress_token_modes(
                parent_state,
                rank=base_rank,
                compression=compression,
                oversample=sketch_oversample,
                power_iterations=sketch_power_iterations,
                seed=sketch_seed + layer + 1,
            )
            defect_factors = _compress_token_modes(
                defect_state,
                rank=defect_rank,
                compression=compression,
                oversample=sketch_oversample,
                power_iterations=sketch_power_iterations,
                seed=sketch_seed + layer + 1,
            )

        parent_inputs.append(parent_input)
        current_inputs.append(current_factors)
        defect_inputs.append(defect_input)
        parent_layers.append(parent_cache)
        current_layers.append(current_cache)
        parent_materialized.append(parent_cache.materialize())
        current_materialized.append(current_cache.materialize())
    parent_replay = FactorizedReplay(
        cache=HSTUKVCache.from_layer_list(parent_materialized, seq_len=length),
        layers=tuple(parent_layers),
        block_input_factors=tuple(parent_inputs),
    )
    current_replay = FactorizedReplay(
        cache=HSTUKVCache.from_layer_list(current_materialized, seq_len=length),
        layers=tuple(current_layers),
        block_input_factors=tuple(current_inputs),
    )
    return DefectFirstReplay(
        parent=parent_replay,
        current=current_replay,
        defect_input_factors=tuple(defect_inputs),
        post_block_defects=tuple(post_block_defects),
        base_rank=base_rank,
        defect_rank=defect_rank,
    )


@torch.inference_mode()
def absolute_replay_from_initial_factors(
    model,
    initial_factors: TokenModeFactors,
    *,
    rank: int,
    compression: Literal["exact_svd", "fixed_range_finder"] = "fixed_range_finder",
    sketch_oversample: int = 4,
    sketch_power_iterations: int = 1,
    sketch_seed: int = 17,
) -> FactorizedReplay:
    """Terminal-K/V absolute-state replay used only as a matched control."""

    if model.training or model.cfg.block_variant != "legacy" or not model.blocks:
        raise ValueError("absolute control requires a nonempty eval-mode legacy model")
    if initial_factors.rank != rank:
        raise ValueError("initial factors differ from the absolute-state rank")
    factors = initial_factors
    inputs: list[TokenModeFactors] = []
    layers: list[FactorizedCacheLayer] = []
    materialized: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer, block in enumerate(model.blocks):
        inputs.append(factors)
        if layer + 1 == len(model.blocks):
            cache = _factorized_kv_only(block, factors)
        else:
            update, cache = _factorized_block_update(block, factors)
            state = factors.materialize() + update
            factors = _compress_token_modes(
                state,
                rank=rank,
                compression=compression,
                oversample=sketch_oversample,
                power_iterations=sketch_power_iterations,
                seed=sketch_seed + layer + 1,
            )
        layers.append(cache)
        materialized.append(cache.materialize())
    return FactorizedReplay(
        cache=HSTUKVCache.from_layer_list(materialized, seq_len=initial_factors.left.shape[1]),
        layers=tuple(layers),
        block_input_factors=tuple(inputs),
    )


def factorized_cache_scalars(replay: FactorizedReplay) -> int:
    """Count a factorized K/V cache with one left factor shared by K and V."""

    return sum(
        layer.left.numel() + layer.key_core.numel() + layer.value_core.numel()
        for layer in replay.layers
    )


def native_response_sidecar_scalars(
    parent: FactorizedReplay,
    current: FactorizedReplay,
) -> int:
    """Count both approximate trajectories required by response differencing."""

    if len(parent.layers) != len(current.layers):
        raise ValueError("response sidecar trajectories have different depths")
    return factorized_cache_scalars(parent) + factorized_cache_scalars(current)


def _factorized_prefix_heads(
    block,
    query: torch.Tensor,
    layer: FactorizedCacheLayer,
) -> torch.Tensor:
    """Native activated prefix response before the shared output projection."""

    attention = block.attn
    batch, length, rank = layer.left.shape
    heads, head_dim = attention.num_heads, attention.head_dim
    key_core = layer.key_core.view(batch, rank, heads, head_dim).permute(0, 2, 1, 3)
    value_core = layer.value_core.view(batch, rank, heads, head_dim).permute(0, 2, 1, 3)
    mode_logits = query @ key_core.transpose(-2, -1)
    logits = mode_logits @ layer.left.transpose(1, 2)[:, None]
    logits = logits * attention.scale
    query_position = torch.tensor([length], device=query.device)
    key_positions = torch.arange(length, device=query.device)
    bias = attention._relative_position_bias(query_position, key_positions, logits.dtype)
    if bias is not None:
        logits = logits + bias
    weights = attention.attn_dropout(attention._activate(logits))
    mode_weights = weights @ layer.left[:, None]
    return mode_weights @ value_core


def _dense_prefix_heads(
    block,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    attention = block.attn
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("dense Parent K/V must have matching [B,N,H] shapes")
    batch, length, _ = key.shape
    heads, head_dim = attention.num_heads, attention.head_dim
    parent_key = key.view(batch, length, heads, head_dim).transpose(1, 2)
    parent_value = value.view(batch, length, heads, head_dim).transpose(1, 2)
    logits = (query @ parent_key.transpose(-2, -1)) * attention.scale
    query_position = torch.tensor([length], device=query.device)
    key_positions = torch.arange(length, device=query.device)
    bias = attention._relative_position_bias(query_position, key_positions, logits.dtype)
    if bias is not None:
        logits = logits + bias
    weights = attention.attn_dropout(attention._activate(logits))
    return weights @ parent_value


def _native_response_difference_attention_one(
    block,
    normalized_query: torch.Tensor,
    exact_parent_key: torch.Tensor,
    exact_parent_value: torch.Tensor,
    approximate_parent: FactorizedCacheLayer,
    approximate_current: FactorizedCacheLayer,
) -> torch.Tensor:
    """Exact-Parent response plus a native approximate release response defect."""

    attention = block.attn
    if block.block_variant != "legacy" or normalized_query.shape[1] != 1:
        raise ValueError("native response differencing requires one-token legacy input")
    query, key_new, value_new = attention._project(normalized_query)
    exact_heads = _dense_prefix_heads(
        block, query, exact_parent_key, exact_parent_value
    )
    parent_heads = _factorized_prefix_heads(block, query, approximate_parent)
    current_heads = _factorized_prefix_heads(block, query, approximate_current)
    corrected_prefix = exact_heads + current_heads - parent_heads

    length = exact_parent_key.shape[1]
    self_weight = (query * key_new).sum(dim=-1, keepdim=True) * attention.scale
    query_position = torch.tensor([length], device=query.device)
    self_bias = attention._relative_position_bias(
        query_position, query_position, self_weight.dtype
    )
    if self_bias is not None:
        self_weight = self_weight + self_bias
    self_weight = attention.attn_dropout(attention._activate(self_weight))
    return attention._finish(corrected_prefix + self_weight * value_new)


@torch.inference_mode()
def forward_one_with_native_response_defect(
    model,
    exact_parent: HSTUKVCache,
    approximate_parent: FactorizedReplay,
    approximate_current: FactorizedReplay,
    embedded_query: torch.Tensor,
) -> torch.Tensor:
    """Read the release defect at the earliest native functional boundary.

    For every layer this evaluates three response terms under the Current
    reader's original query, activation, KxV interaction, gate and residual::

        R(exact Parent) + R(approx Current) - R(approx Parent).

    The two approximate terms are not linearized or replaced by moments.  At
    full token rank they cancel the exact Parent response and recover the
    Current-Exact reader by induction over query layers.
    """

    if embedded_query.ndim != 3 or embedded_query.shape[1:] != (
        1,
        model.cfg.hidden_size,
    ):
        raise ValueError("embedded query must have shape [B,1,H]")
    if exact_parent.k.shape[1] != embedded_query.shape[0]:
        raise ValueError("query and Parent cache batches differ")
    if len(approximate_parent.layers) != len(model.blocks):
        raise ValueError("approximate Parent depth differs from reader")
    if len(approximate_current.layers) != len(model.blocks):
        raise ValueError("approximate Current depth differs from reader")

    x = embedded_query
    for layer, block in enumerate(model.blocks):
        residual = x
        normalized = block.norm(x)
        attention_output = _native_response_difference_attention_one(
            block,
            normalized,
            exact_parent.k[layer],
            exact_parent.v[layer],
            approximate_parent.layers[layer],
            approximate_current.layers[layer],
        )
        if block.gating == "silu_gate":
            update = attention_output * F.silu(block.gate_proj(normalized))
        elif block.gating == "glu":
            update = attention_output * torch.sigmoid(block.gate_proj(normalized))
        elif block.gating == "none":
            update = attention_output
        else:
            raise ValueError("native response differencing does not cover the FFN variant")
        x = residual + update
    return model.final_norm(x)


@torch.inference_mode()
def forward_one_with_factorized_cache(
    model,
    approximate_current: FactorizedReplay,
    embedded_query: torch.Tensor,
) -> torch.Tensor:
    """Generic compressed-Current reader used as the single-rank-8 control."""

    if embedded_query.ndim != 3 or embedded_query.shape[1:] != (
        1,
        model.cfg.hidden_size,
    ):
        raise ValueError("embedded query must have shape [B,1,H]")
    if len(approximate_current.layers) != len(model.blocks):
        raise ValueError("approximate Current depth differs from reader")
    x = embedded_query
    for block, cache_layer in zip(
        model.blocks, approximate_current.layers, strict=True
    ):
        residual = x
        normalized = block.norm(x)
        attention = block.attn
        query, key_new, value_new = attention._project(normalized)
        prefix = _factorized_prefix_heads(block, query, cache_layer)
        length = cache_layer.left.shape[1]
        position = torch.tensor([length], device=query.device)
        self_weight = (query * key_new).sum(dim=-1, keepdim=True) * attention.scale
        self_bias = attention._relative_position_bias(position, position, self_weight.dtype)
        if self_bias is not None:
            self_weight = self_weight + self_bias
        self_weight = attention.attn_dropout(attention._activate(self_weight))
        attention_output = attention._finish(prefix + self_weight * value_new)
        if block.gating == "silu_gate":
            update = attention_output * F.silu(block.gate_proj(normalized))
        elif block.gating == "glu":
            update = attention_output * torch.sigmoid(block.gate_proj(normalized))
        elif block.gating == "none":
            update = attention_output
        else:
            raise ValueError("factorized control does not cover the FFN variant")
        x = residual + update
    return model.final_norm(x)


@torch.inference_mode()
def approximate_release_defect_basis(
    parent: FactorizedReplay,
    current: FactorizedReplay,
    *,
    rank: int = 8,
    oversample: int = 4,
    power_iterations: int = 0,
    seed: int = 1017,
) -> torch.Tensor:
    """Build the persistent history basis from approximate layer-0 defect only."""

    if not parent.layers or len(parent.layers) != len(current.layers):
        raise ValueError("approximate release trajectories have different depths")
    parent_key, parent_value = parent.layers[0].materialize()
    current_key, current_value = current.layers[0].materialize()
    return randomized_token_basis(
        torch.cat((current_key - parent_key, current_value - parent_value), dim=-1),
        rank=rank,
        oversample=oversample,
        power_iterations=power_iterations,
        seed=seed,
    )


@torch.inference_mode()
def splice_approximate_release_defect(
    exact_parent: HSTUKVCache,
    parent: FactorizedReplay,
    current: FactorizedReplay,
    basis: torch.Tensor,
) -> SharedModeSplice:
    """Compile an approximate finite-release defect on immutable exact Parent K/V."""

    if exact_parent.k.ndim != 4 or exact_parent.k.shape != exact_parent.v.shape:
        raise ValueError("exact Parent cache must be [L,B,N,H]")
    if parent.cache.k.shape != exact_parent.k.shape or current.cache.k.shape != exact_parent.k.shape:
        raise ValueError("approximate and exact cache shapes differ")
    layers, batch, length, _ = (int(value) for value in exact_parent.k.shape)
    if len(parent.layers) != layers or len(current.layers) != layers:
        raise ValueError("approximate replay depth differs from exact Parent")
    if basis.ndim != 3 or basis.shape[:2] != (batch, length):
        raise ValueError("basis must have shape [B,N,r]")
    basis = basis.to(device=exact_parent.k.device, dtype=exact_parent.k.dtype)
    gram = basis.transpose(1, 2) @ basis
    identity = torch.eye(basis.shape[2], device=basis.device, dtype=basis.dtype).expand_as(gram)
    if not torch.allclose(gram, identity, atol=2e-4, rtol=2e-4):
        raise ValueError("basis columns must be orthonormal")

    transpose = basis.transpose(1, 2)
    migrated: list[tuple[torch.Tensor, torch.Tensor]] = []
    delta_keys: list[torch.Tensor] = []
    delta_values: list[torch.Tensor] = []
    for layer, (parent_layer, current_layer) in enumerate(
        zip(parent.layers, current.layers, strict=True)
    ):
        parent_alignment = transpose @ parent_layer.left
        current_alignment = transpose @ current_layer.left
        delta_key = (
            current_alignment @ current_layer.key_core
            - parent_alignment @ parent_layer.key_core
        )
        delta_value = (
            current_alignment @ current_layer.value_core
            - parent_alignment @ parent_layer.value_core
        )
        migrated.append(
            (
                exact_parent.k[layer] + basis @ delta_key,
                exact_parent.v[layer] + basis @ delta_value,
            )
        )
        delta_keys.append(delta_key.detach())
        delta_values.append(delta_value.detach())
    return SharedModeSplice(
        cache=HSTUKVCache.from_layer_list(migrated, seq_len=length),
        basis=basis.detach(),
        delta_k_cores=tuple(delta_keys),
        delta_v_cores=tuple(delta_values),
    )


def _factorized_rms_flops(*, n: int, h: int, rank: int) -> int:
    return 2 * (h + n) * rank * rank + 3 * n * rank + 2 * n + rank * h


def _factorized_block_body_flops(
    *, n: int, h: int, heads: int, causal_pairs: int, rank: int
) -> int:
    head_dim = h // heads
    return (
        _factorized_rms_flops(n=n, h=h, rank=rank)
        + 8 * rank * h * h
        + 2 * heads * rank * rank * head_dim
        + 2 * heads * n * rank * rank
        + 4 * heads * causal_pairs * rank
        + 2 * heads * causal_pairs
        + 2 * rank * h * h
        + 2 * n * heads * rank * h
        + 2 * n * rank * h
        + 3 * n * h
        + n * h
    )


def _terminal_kv_flops(*, n: int, h: int, rank: int) -> int:
    return _factorized_rms_flops(n=n, h=h, rank=rank) + 4 * rank * h * h


def _dense_range_flops(
    *, n: int, h: int, rank: int, oversample: int = 4, power_iterations: int = 1
) -> int:
    sketch = min(rank + oversample, n, h)
    qr = math.ceil(2 * n * sketch * sketch - (2 * sketch**3) / 3)
    truncate = (
        2 * h * sketch * sketch
        + 9 * sketch**3
        + 2 * n * sketch * rank
        + 2 * rank * sketch * h
    )
    return 4 * (1 + power_iterations) * n * h * sketch + (1 + power_iterations) * qr + truncate


def _absolute_trajectory_flops(*, n: int, h: int, layers: int, heads: int, rank: int) -> int:
    initial = matrix_free_input_cost(
        history_length=n,
        hidden_size=h,
        temporal_num_freqs=16,
        rank=rank,
        oversample=4,
        power_iterations=1,
    ).flops
    pairs = n * (n + 1) // 2
    transition = (
        _factorized_block_body_flops(
            n=n, h=h, heads=heads, causal_pairs=pairs, rank=rank
        )
        + 2 * n * rank * h
        + n * h
        + _dense_range_flops(n=n, h=h, rank=rank)
    )
    return initial + (layers - 1) * transition + _terminal_kv_flops(n=n, h=h, rank=rank)


def _factor_difference_basis_flops(
    *, n: int, h: int, factor_rank_sum: int, basis_rank: int
) -> int:
    sketch = basis_rank + 4
    width = 2 * h
    factor_apply = 2 * width * factor_rank_sum * sketch + 2 * n * factor_rank_sum * sketch
    right_apply = factor_apply + n * sketch
    transpose_apply = factor_apply + width * sketch
    qr = math.ceil(2 * n * sketch * sketch - (2 * sketch**3) / 3)
    truncate = (
        2 * width * sketch * sketch
        + 9 * sketch**3
        + 2 * n * sketch * basis_rank
        + 2 * basis_rank * sketch * width
    )
    return right_apply + transpose_apply + qr + truncate


def _factor_difference_core_flops(
    *, n: int, h: int, basis_rank: int, factor_ranks: tuple[int, int]
) -> int:
    total = 0
    for rank in factor_ranks:
        total += 2 * n * basis_rank * rank + 4 * basis_rank * rank * h
    return total + 2 * basis_rank * h


def _single_factor_minus_dense_basis_flops(
    *, n: int, h: int, factor_rank: int, basis_rank: int
) -> int:
    sketch = basis_rank + 4
    width = 2 * h
    factor_apply = 2 * width * factor_rank * sketch + 2 * n * factor_rank * sketch
    dense_apply = 2 * n * width * sketch
    right_apply = factor_apply + dense_apply + n * sketch
    transpose_apply = factor_apply + dense_apply + width * sketch
    qr = math.ceil(2 * n * sketch * sketch - (2 * sketch**3) / 3)
    truncate = (
        2 * width * sketch * sketch
        + 9 * sketch**3
        + 2 * n * sketch * basis_rank
        + 2 * basis_rank * sketch * width
    )
    return right_apply + transpose_apply + qr + truncate


def _single_factor_minus_dense_core_flops(
    *, n: int, h: int, factor_rank: int, basis_rank: int
) -> int:
    return (
        2 * n * basis_rank * factor_rank
        + 4 * basis_rank * factor_rank * h
        + 4 * n * basis_rank * h
        + 2 * basis_rank * h
    )


def medium_defect_first_costs() -> dict[str, dict[str, int | float | bool | str]]:
    """Return a strict, matched Medium ledger for the sole preflight point.

    Unlike the older fused lower-bound ledgers, this audit charges the dense
    residual/defect materializations performed by the semantic executor before
    every upper-layer range finder.  Thus comparisons cannot make the primary
    path look cheaper by silently granting it a fused kernel unavailable to
    the controls.
    """

    n, h, layers, heads = 1024, 192, 6, 6
    exact_all = 4_771_282_944
    base_rank, defect_rank = 2, 4
    current_rank = base_rank + defect_rank
    basis_rank = 8
    pairs = n * (n + 1) // 2

    parent_initial = matrix_free_input_cost(
        history_length=n,
        hidden_size=h,
        temporal_num_freqs=16,
        rank=base_rank,
        oversample=4,
        power_iterations=1,
    )
    one_defect_arm = matrix_free_input_cost(
        history_length=n,
        hidden_size=h,
        temporal_num_freqs=16,
        rank=defect_rank,
        oversample=4,
        power_iterations=1,
    )
    defect_sketch = defect_rank + 4
    defect_initial_operator = (
        2
        * (
            one_defect_arm.right_operator_applications
            + one_defect_arm.transpose_operator_applications
        )
        + 2 * n * defect_sketch
        + 2 * h * defect_sketch
        + one_defect_arm.thin_qr
        + one_defect_arm.small_gram_eigh_rotation
    )
    # Parent lookup/time features are already prepared for its base factor.
    current_input_preparation = (
        one_defect_arm.base_lookup_additions
        + one_defect_arm.temporal_phase_multiplications
    )
    primary_initial = parent_initial.flops + current_input_preparation + defect_initial_operator

    base_state = 2 * n * base_rank * h + n * h
    defect_state = 2 * n * defect_rank * h + 2 * n * h
    primary_transition = (
        _factorized_block_body_flops(
            n=n, h=h, heads=heads, causal_pairs=pairs, rank=base_rank
        )
        + _factorized_block_body_flops(
            n=n, h=h, heads=heads, causal_pairs=pairs, rank=current_rank
        )
        + base_state
        + defect_state
        + _dense_range_flops(n=n, h=h, rank=base_rank)
        + _dense_range_flops(n=n, h=h, rank=defect_rank)
    )
    primary_trajectory = (
        primary_initial
        + (layers - 1) * primary_transition
        + _terminal_kv_flops(n=n, h=h, rank=base_rank)
        + _terminal_kv_flops(n=n, h=h, rank=current_rank)
    )
    primary_total = primary_trajectory

    def control_total(parent_rank: int, current_absolute_rank: int) -> int:
        return (
            _absolute_trajectory_flops(
                n=n, h=h, layers=layers, heads=heads, rank=parent_rank
            )
            + _absolute_trajectory_flops(
                n=n, h=h, layers=layers, heads=heads, rank=current_absolute_rank
            )
        )

    asymmetric_total = control_total(2, 6)
    paired_authority = medium_paired_native_response_cost()
    paired_total = paired_authority.total_constructor_flops
    # Authoritative value from the dedicated paired-native route eliminator;
    # it is shared by its single-r8 cache and shared-U0 controls.
    single_total = 853_836_992
    # Every primary/matched-control sidecar stores factorized K/V whose total
    # active cache rank is eight: one Nxr left and two rxH cores per layer.
    sidecar = layers * (n + 2 * h) * basis_rank

    def factor_prefix_response(rank: int) -> int:
        # Q@core, mode-logits@left, native activation, weights@left,
        # mode-response@Vcore.  Query/self/output/gate/residual work is common
        # to Reuse and therefore reported outside this incremental count.
        return 4 * heads * rank * (h // heads + n)

    two_factor_response = layers * (
        factor_prefix_response(base_rank)
        + factor_prefix_response(current_rank)
        + 2 * h
    )
    single_factor_response = layers * factor_prefix_response(8)
    two_factor_activations = 2 * layers * heads * n
    single_factor_activations = layers * heads * n

    common: dict[str, int | float | bool | str] = {
        "exact_all_flops_per_user": exact_all,
        "persistent_sidecar_scalars_fp32": sidecar,
        "persistent_sidecar_bytes_fp32": 4 * sidecar,
        "cost_semantics": (
            "matrix-free initial factors; factor-aware triangular legacy attention; "
            "explicit dense residual/defect materialization; fixed power-1 upper "
            "range finders; terminal K/V only; no probe/moment/U0 compiler"
        ),
    }

    def row(total: int, identity: str, **extra: int | float | bool | str) -> dict[str, int | float | bool | str]:
        return {
            **common,
            "identity": identity,
            "total_flops_per_user": total,
            "over_exact_all": total / exact_all,
            "within_twenty_percent": total / exact_all <= 0.20,
            **extra,
        }

    return {
        "defect_first_b2_d4": row(
            primary_total,
            "Parent base rank2 plus separately compressed release defect rank4; Current effective rank6",
            initial_factor_flops=primary_initial,
            per_nonterminal_transition_flops=primary_transition,
            trajectory_flops=primary_trajectory,
            per_request_extra_two_factor_response_flops=two_factor_response,
            native_activation_evaluations_per_request=two_factor_activations,
            reader_semantics=(
                "R_current(exact Parent) + R_current(approx Current) - "
                "R_current(approx Parent), before Current gate/residual"
            ),
        ),
        "ordinary_asymmetric_p2_c6": row(
            asymmetric_total,
            "independent absolute Parent rank2 and absolute Current rank6",
            per_request_extra_two_factor_response_flops=two_factor_response,
            native_activation_evaluations_per_request=two_factor_activations,
        ),
        "paired_absolute_p4_c4": row(
            paired_total,
            "independent absolute Parent rank4 and absolute Current rank4",
            per_request_extra_two_factor_response_flops=two_factor_response,
            native_activation_evaluations_per_request=two_factor_activations,
            cost_authority=(
                "insight_two.paired_native_response.medium_paired_native_response_cost"
            ),
            cost_semantics=(
                "authoritative paired final-KV matrix-free ledger minus superseded "
                "U0 builder and signed-core compiler"
            ),
        ),
        "single_absolute_c8": row(
            single_total,
            "generic absolute Current rank8 factorized cache reader",
            per_request_factor_prefix_flops=single_factor_response,
            native_activation_evaluations_per_request=single_factor_activations,
            cost_authority=(
                "scripts/insight_two/run_paired_native_response_preflight.py"
            ),
            cost_semantics="authoritative dedicated single-r8 matched-control ledger",
        ),
    }
