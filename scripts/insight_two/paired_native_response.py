"""Trajectory-matched native-response control variate for one release.

The constructor keeps the two factorized trajectories produced by
``paired_release_replay``.  A Current-version query reads the persistent
Parent cache and the two reduced trajectories through the *same native
attention operator*::

    R_mig(q) = R_C(q; K_parent, V_parent)
             + R_C(q; Khat_current, Vhat_current)
             - R_C(q; Khat_parent, Vhat_parent).

The subtraction therefore happens after query--key activation and value
aggregation.  Unlike ``common_projection_response``, the two arms retain
their own data-dependent token factors; unlike the S4 moment compiler, this
path uses no probe, activation-region approximation, or fitted map.

This is a fixed route-elimination experiment, not an admitted Design.  Its
scientific gate is deliberately strict: it must beat the matched-compute
single-Current rank-8 control.  At full token rank the paired recurrence and
this native reader recover the exact Current reader (up to decomposition
roundoff).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hstu_kvcache.models import HSTUKVCache
from insight_two.common_projection_response import factorized_prefix_heads
from insight_two.cone_response_memory import (
    _block_update,
    _native_prefix_heads,
    _native_self_heads,
)
from insight_two.mode_space_replay import (
    FactorizedCacheLayer,
    PairedReleaseReplay,
)

MEDIUM_EXACT_ALL_FLOPS = 4_771_282_944
MEDIUM_PAIRED_KV_ONLY_MATRIX_FREE_FLOPS = 874_402_376


def _detached_layer(layer: FactorizedCacheLayer) -> FactorizedCacheLayer:
    return FactorizedCacheLayer(
        left=layer.left.detach(),
        key_core=layer.key_core.detach(),
        value_core=layer.value_core.detach(),
    )


@dataclass(frozen=True)
class PairedNativeResponseMemory:
    """Both reduced release trajectories retained as a persistent sidecar."""

    current: tuple[FactorizedCacheLayer, ...]
    parent: tuple[FactorizedCacheLayer, ...]
    source_length: int
    source_kv_scalars: int

    def __post_init__(self) -> None:
        if not self.current or len(self.current) != len(self.parent):
            raise ValueError("paired memory requires equal nonempty layer counts")
        if self.source_length < 1:
            raise ValueError("source_length must be positive")
        if self.source_kv_scalars < 1:
            raise ValueError("source_kv_scalars must be positive")
        for current, parent in zip(self.current, self.parent, strict=True):
            if current.left.shape[:2] != parent.left.shape[:2]:
                raise ValueError("paired factor histories differ in shape")
            if current.left.shape[1] != self.source_length:
                raise ValueError("factor history length differs from source_length")
            if current.rank != parent.rank:
                raise ValueError("paired factor ranks differ")
            if current.key_core.shape[2] != parent.key_core.shape[2]:
                raise ValueError("paired factor widths differ")

    @property
    def stored_scalars(self) -> int:
        return sum(
            layer.left.numel() + layer.key_core.numel() + layer.value_core.numel()
            for arm in (self.current, self.parent)
            for layer in arm
        )

    @property
    def stored_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for arm in (self.current, self.parent)
            for layer in arm
            for tensor in (layer.left, layer.key_core, layer.value_core)
        )

    @property
    def storage_ratio_to_parent_kv(self) -> float:
        return self.stored_scalars / self.source_kv_scalars

    @property
    def factor_reads_per_query(self) -> int:
        """Logical sidecar reads: Current and Parent factors at every layer."""

        return 2 * len(self.current)


@dataclass(frozen=True)
class PairedNativeResponseIntervention:
    scores: torch.Tensor
    readout: torch.Tensor
    layer_signed_heads: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class PairedNativeResponseCost:
    """Audited Medium constructor, storage, and per-request arithmetic."""

    paired_kv_ledger_flops: int
    superseded_shared_basis_flops: int
    superseded_signed_core_flops: int
    exact_all_flops: int
    sidecar_scalars: int
    layers: int
    heads: int
    head_dim: int
    history_length: int
    rank_per_arm: int

    @property
    def total_constructor_flops(self) -> int:
        return (
            self.paired_kv_ledger_flops
            - self.superseded_shared_basis_flops
            - self.superseded_signed_core_flops
        )

    @property
    def constructor_fraction(self) -> float:
        return self.total_constructor_flops / self.exact_all_flops

    @property
    def within_twenty_percent(self) -> bool:
        return self.constructor_fraction <= 0.20

    @property
    def sidecar_fp32_bytes(self) -> int:
        return 4 * self.sidecar_scalars

    @property
    def factor_reads_per_layer_per_query(self) -> int:
        return 2

    @property
    def factor_reads_per_query(self) -> int:
        return self.layers * self.factor_reads_per_layer_per_query

    @property
    def factorized_read_flops_per_arm_per_layer_per_query(self) -> int:
        # q Ck^T, (...) L^T, weights L, (...) Cv.  Activation evaluations
        # and memory traffic are reported separately rather than hidden here.
        return 4 * self.heads * self.rank_per_arm * (
            self.head_dim + self.history_length
        )

    @property
    def incremental_reader_flops_per_query(self) -> int:
        factor_reads = (
            self.layers
            * self.factor_reads_per_layer_per_query
            * self.factorized_read_flops_per_arm_per_layer_per_query
        )
        signed_additions = self.layers * 2 * self.heads * self.head_dim
        return factor_reads + signed_additions

    @property
    def native_activation_evaluations_per_query(self) -> int:
        return (
            self.layers
            * self.factor_reads_per_layer_per_query
            * self.heads
            * self.history_length
        )

    @property
    def logical_factor_sidecar_scalar_reads_per_query(self) -> int:
        # One ideal logical read of every persistent factor.  A concrete
        # kernel may reread L and must report measured bytes separately.
        return self.sidecar_scalars


def medium_paired_native_response_cost() -> PairedNativeResponseCost:
    """Return the fixed r4/r4 matrix-free Medium cost ledger.

    The starting point is the already audited paired final-K/V constructor.
    Native response persistence keeps the two factor arms that constructor
    already produces, and removes only the shared-U0 builder and the signed
    K/V-core compilation that this representation does not use.
    """

    context = 1024
    layers = 6
    heads = 6
    head_dim = 32
    rank = 4
    inner = heads * head_dim
    per_arm_per_layer = context * rank + 2 * rank * inner
    return PairedNativeResponseCost(
        paired_kv_ledger_flops=MEDIUM_PAIRED_KV_ONLY_MATRIX_FREE_FLOPS,
        superseded_shared_basis_flops=1_247_808,
        superseded_signed_core_flops=916_480,
        exact_all_flops=MEDIUM_EXACT_ALL_FLOPS,
        sidecar_scalars=2 * layers * per_arm_per_layer,
        layers=layers,
        heads=heads,
        head_dim=head_dim,
        history_length=context,
        rank_per_arm=rank,
    )


@torch.inference_mode()
def build_paired_native_response_memory(
    replay: PairedReleaseReplay,
    *,
    source_kv_scalars: int,
) -> PairedNativeResponseMemory:
    """Freeze the two legal factorized trajectories as reader-side state."""

    if replay.parent.cache.seq_len != replay.current.cache.seq_len:
        raise ValueError("paired replay arm lengths differ")
    return PairedNativeResponseMemory(
        current=tuple(_detached_layer(layer) for layer in replay.current.layers),
        parent=tuple(_detached_layer(layer) for layer in replay.parent.layers),
        source_length=replay.parent.cache.seq_len,
        source_kv_scalars=source_kv_scalars,
    )


def _validate_exact_parent(
    model,
    exact_parent: HSTUKVCache,
    memory: PairedNativeResponseMemory,
) -> None:
    if model.training:
        raise ValueError("paired native response requires model.eval()")
    if exact_parent.k.ndim != 4 or exact_parent.k.shape != exact_parent.v.shape:
        raise ValueError("exact Parent must contain matching [L,B,N,H] K/V")
    if exact_parent.k.shape[0] != len(model.blocks):
        raise ValueError("exact Parent and model layer counts differ")
    if exact_parent.k.shape[1] != 1:
        raise ValueError("paired native response supports one user")
    if exact_parent.seq_len != memory.source_length:
        raise ValueError("exact Parent and paired memory lengths differ")
    if exact_parent.k.shape[2] != memory.source_length:
        raise ValueError("exact Parent tensor length differs")
    if len(memory.current) != len(model.blocks):
        raise ValueError("paired memory and model layer counts differ")
    for block in model.blocks:
        if block.attn.position_bias is not None:
            raise ValueError("paired factor reader does not support position bias")


def _validate_current_suffix(
    model,
    suffix: HSTUKVCache | None,
) -> None:
    if suffix is None:
        return
    if suffix.k.ndim != 4 or suffix.k.shape != suffix.v.shape:
        raise ValueError("Current suffix must contain matching [L,B,N,H] K/V")
    if suffix.k.shape[0] != len(model.blocks) or suffix.k.shape[1] != 1:
        raise ValueError("Current suffix layout differs from model")
    if suffix.k.shape[2] != suffix.seq_len:
        raise ValueError("Current suffix tensor length differs")


@torch.inference_mode()
def intervene_paired_native_response(
    model,
    exact_parent: HSTUKVCache,
    memory: PairedNativeResponseMemory,
    candidate_ids: torch.Tensor,
    query_time_deltas: torch.Tensor,
    *,
    current_suffix: HSTUKVCache | None = None,
    query_type_ids: torch.Tensor | None = None,
    query_action_ids: torch.Tensor | None = None,
    candidate_item_vectors: torch.Tensor | None = None,
) -> PairedNativeResponseIntervention:
    """Read exact Parent plus the native paired trajectory defect.

    ``current_suffix`` is optional exact Current K/V appended after cutover.
    Legacy HSTU attention is an unnormalised token sum, so its response can be
    added as a disjoint segment without changing the frozen paired sidecar.
    """

    _validate_exact_parent(model, exact_parent, memory)
    _validate_current_suffix(model, current_suffix)
    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != 1:
        raise ValueError("candidate_ids must have shape [1,C]")
    if candidate_ids.shape[1] < 1:
        raise ValueError("at least one candidate is required")

    candidates = candidate_ids.shape[1]
    x = model.embed_query_tokens(
        candidate_ids,
        query_time_deltas,
        query_type_ids=query_type_ids,
        query_action_ids=query_action_ids,
        item_vectors=candidate_item_vectors,
    ).reshape(candidates, 1, model.cfg.hidden_size)
    corrections: list[torch.Tensor] = []
    for layer_index, (block, current_layer, parent_layer) in enumerate(
        zip(model.blocks, memory.current, memory.parent, strict=True)
    ):
        residual = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        exact_parent_heads = _native_prefix_heads(
            block.attn,
            q,
            exact_parent.k[layer_index],
            exact_parent.v[layer_index],
        )
        current_heads = factorized_prefix_heads(block.attn, q, current_layer)
        parent_heads = factorized_prefix_heads(block.attn, q, parent_layer)
        signed_heads = current_heads - parent_heads
        prefix_heads = exact_parent_heads + signed_heads
        if current_suffix is not None and current_suffix.seq_len > 0:
            prefix_heads = prefix_heads + _native_prefix_heads(
                block.attn,
                q,
                current_suffix.k[layer_index],
                current_suffix.v[layer_index],
            )
        self_heads = _native_self_heads(block.attn, q, k_new, v_new)
        corrections.append(
            signed_heads.reshape(
                1,
                candidates,
                block.attn.num_heads,
                block.attn.head_dim,
            )
        )
        x = residual + _block_update(block, x_norm, prefix_heads + self_heads)

    readout = model.final_norm(x).reshape(1, candidates, model.cfg.hidden_size)
    return PairedNativeResponseIntervention(
        scores=model.cc_score_head(readout).squeeze(-1),
        readout=readout,
        layer_signed_heads=tuple(corrections),
    )


@torch.inference_mode()
def select_paired_native_response_rows(
    memory: PairedNativeResponseMemory,
    positions: torch.Tensor,
) -> PairedNativeResponseMemory:
    """Evict source rows without recompressing either factorized arm.

    Row selection commutes exactly with ``K=L@C`` and ``V=L@C``.  Positions
    must be strictly increasing so the retained cache remains chronological.
    The caller applies the same selection to the exact Parent K/V.
    """

    positions = positions.to(device=memory.current[0].left.device, dtype=torch.long)
    if positions.ndim != 1 or positions.numel() < 1:
        raise ValueError("positions must be a nonempty vector")
    if bool((positions < 0).any()) or bool((positions >= memory.source_length).any()):
        raise ValueError("selected position lies outside the source history")
    if positions.numel() > 1 and not bool((positions[1:] > positions[:-1]).all()):
        raise ValueError("positions must be strictly increasing")

    def selected(layer: FactorizedCacheLayer) -> FactorizedCacheLayer:
        return FactorizedCacheLayer(
            left=layer.left.index_select(1, positions),
            key_core=layer.key_core,
            value_core=layer.value_core,
        )

    current = tuple(selected(layer) for layer in memory.current)
    parent = tuple(selected(layer) for layer in memory.parent)
    new_length = int(positions.numel())
    inner = int(current[0].key_core.shape[2])
    return PairedNativeResponseMemory(
        current=current,
        parent=parent,
        source_length=new_length,
        source_kv_scalars=2 * len(current) * new_length * inner,
    )
