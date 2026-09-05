"""Oracle first-order response moments over attention-address cells.

The address partition is selected once from normalized, concatenated layer-0
Current and Parent keys.  Every layer and head then summarizes the exact
Current-minus-Parent response measure inside each shared cell around a common
raw-key center ``c``::

    S = sum_i (v_current_i - v_parent_i)

    M = sum_i [(k_current_i - c) outer v_current_i
               - (k_parent_i - c) outer v_parent_i]

For a real query, the native activated response residual is approximated by
the first-order expression::

    phi(scale * q.c) S
      + phi'(scale * q.c) * scale * (q contracted with M)

This is an oracle mathematical diagnostic, not an executable migration
constructor.  It reads complete Current and Parent K/V.  Relative-position
bias is deliberately rejected because a cell's per-position biases cannot be
represented by the stated key moments without additional moments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hstu_kvcache.models import HSTUKVCache
from insight_two.address_response_memory import select_address_landmarks
from insight_two.signed_response_memory import (
    _block_update,
    _native_prefix_heads,
    _native_self_heads,
    _validate_cache,
)


@dataclass(frozen=True)
class OracleAddressMomentMemory:
    """Per-cell signed response moments.

    ``centers`` and ``signed_zeroth`` have shape ``[L,B,H,R,D]``.
    ``signed_first`` has shape ``[L,B,H,R,D,D]`` with key dimension before
    value dimension.  Address assignments are shared across all layers/heads.
    """

    centers: torch.Tensor
    signed_zeroth: torch.Tensor
    signed_first: torch.Tensor
    selected_positions: torch.Tensor
    assignments: torch.Tensor
    cluster_masses: torch.Tensor
    source_length: int

    @property
    def sample_count(self) -> int:
        return int(self.selected_positions.numel())

    @property
    def stored_scalars_per_user(self) -> int:
        return int(
            self.centers[:, 0].numel()
            + self.signed_zeroth[:, 0].numel()
            + self.signed_first[:, 0].numel()
        )


@dataclass(frozen=True)
class OracleAddressMomentIntervention:
    """Outputs from coherent layer-by-layer moment intervention."""

    scores: torch.Tensor
    readout: torch.Tensor
    layer_residual_heads: tuple[torch.Tensor, ...]


def native_activation_and_derivative(
    activation: str,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return HSTU native pointwise activation and its first derivative."""

    if activation == "elu_plus1":
        activated = F.elu(logits) + 1.0
        derivative = torch.where(logits >= 0, torch.ones_like(logits), torch.exp(logits))
        return activated, derivative
    if activation == "relu":
        return F.relu(logits), (logits > 0).to(dtype=logits.dtype)
    if activation == "silu":
        sigmoid = torch.sigmoid(logits)
        derivative = sigmoid * (1.0 + logits * (1.0 - sigmoid))
        return F.silu(logits), derivative
    raise ValueError(f"unsupported attention activation: {activation}")


@torch.inference_mode()
def build_oracle_address_moment_memory(
    exact_cache: HSTUKVCache,
    reuse_cache: HSTUKVCache,
    *,
    sample_count: int,
    num_heads: int,
) -> OracleAddressMomentMemory:
    """Build exact signed zeroth/first moments inside layer-0 address cells."""

    selection = select_address_landmarks(
        exact_cache, reuse_cache, sample_count=sample_count
    )
    if num_heads < 1:
        raise ValueError("num_heads must be positive")
    layers, batch, length, width = exact_cache.k.shape
    if width % num_heads:
        raise ValueError("cache KV width must be divisible by num_heads")
    head_dim = width // num_heads

    def _heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(layers, batch, length, num_heads, head_dim).permute(
            0, 1, 3, 2, 4
        )

    exact_k = _heads(exact_cache.k)
    exact_v = _heads(exact_cache.v)
    reuse_k = _heads(reuse_cache.k)
    reuse_v = _heads(reuse_cache.v)
    centers: list[torch.Tensor] = []
    zeroth: list[torch.Tensor] = []
    first: list[torch.Tensor] = []
    for cell in range(sample_count):
        mask = selection.assignments == cell
        mass = int(mask.sum().item())
        if mass < 1:
            raise RuntimeError("address cell cannot be empty")
        current_k_cell = exact_k[:, :, :, mask, :]
        parent_k_cell = reuse_k[:, :, :, mask, :]
        current_v_cell = exact_v[:, :, :, mask, :]
        parent_v_cell = reuse_v[:, :, :, mask, :]
        center = (current_k_cell.sum(dim=3) + parent_k_cell.sum(dim=3)) / (
            2.0 * mass
        )
        signed_s = (current_v_cell - parent_v_cell).sum(dim=3)
        current_delta = current_k_cell - center.unsqueeze(3)
        parent_delta = parent_k_cell - center.unsqueeze(3)
        signed_m = (
            current_delta.unsqueeze(-1) * current_v_cell.unsqueeze(-2)
            - parent_delta.unsqueeze(-1) * parent_v_cell.unsqueeze(-2)
        ).sum(dim=3)
        centers.append(center)
        zeroth.append(signed_s)
        first.append(signed_m)

    return OracleAddressMomentMemory(
        centers=torch.stack(centers, dim=3).detach(),
        signed_zeroth=torch.stack(zeroth, dim=3).detach(),
        signed_first=torch.stack(first, dim=3).detach(),
        selected_positions=selection.selected_positions.detach(),
        assignments=selection.assignments.detach(),
        cluster_masses=selection.cluster_masses.detach(),
        source_length=selection.source_length,
    )


def _validate_reader(
    attention,
    q: torch.Tensor,
    memory: OracleAddressMomentMemory,
    layer: int,
    candidate_count: int,
) -> int:
    if attention.training:
        raise ValueError("address-moment reads require eval mode")
    if attention.position_bias is not None:
        raise ValueError("address moments do not represent relative-position bias")
    if q.ndim != 4 or q.shape[2] != 1:
        raise ValueError("q must have shape [B*C,H,1,D]")
    if candidate_count < 1 or q.shape[0] % candidate_count:
        raise ValueError("candidate_count does not divide the query batch")
    batch = q.shape[0] // candidate_count
    if memory.centers.shape != memory.signed_zeroth.shape:
        raise ValueError("moment centers and zeroth moments differ in shape")
    if memory.centers.ndim != 5:
        raise ValueError("centers must have shape [L,B,H,R,D]")
    layers, memory_batch, heads, cells, head_dim = memory.centers.shape
    if memory.signed_first.shape != (
        layers,
        memory_batch,
        heads,
        cells,
        head_dim,
        head_dim,
    ):
        raise ValueError("first moments must have shape [L,B,H,R,D,D]")
    if not 0 <= layer < layers:
        raise ValueError("moment layer is out of range")
    if memory_batch != batch:
        raise ValueError("moment memory and query batches differ")
    if q.shape[1:] != (heads, 1, head_dim):
        raise ValueError("query head layout differs from moment memory")
    if cells != memory.sample_count:
        raise ValueError("moment cell axis differs from selected positions")
    return batch


def read_address_moment_residual(
    attention,
    q: torch.Tensor,
    memory: OracleAddressMomentMemory,
    *,
    layer: int,
    candidate_count: int,
) -> torch.Tensor:
    """Read one layer's first-order signed response moments with real queries."""

    batch = _validate_reader(attention, q, memory, layer, candidate_count)
    centers = memory.centers[layer].to(device=q.device, dtype=q.dtype)
    zeroth = memory.signed_zeroth[layer].to(device=q.device, dtype=q.dtype)
    first = memory.signed_first[layer].to(device=q.device, dtype=q.dtype)
    centers = centers.repeat_interleave(candidate_count, dim=0)
    zeroth = zeroth.repeat_interleave(candidate_count, dim=0)
    first = first.repeat_interleave(candidate_count, dim=0)
    query = q.squeeze(2)
    center_logits = (
        torch.einsum("bhd,bhrd->bhr", query, centers) * attention.scale
    )
    activated, derivative = native_activation_and_derivative(
        attention.activation, center_logits
    )
    contracted_first = torch.einsum("bhd,bhrde->bhre", query, first)
    response_by_cell = activated.unsqueeze(-1) * zeroth + (
        derivative * attention.scale
    ).unsqueeze(-1) * contracted_first
    response = response_by_cell.sum(dim=2).unsqueeze(2)
    if attention.block_variant == "hstu_reference":
        response = response / attention.cfg.max_seq_len
    # Dropout is necessarily an identity in eval mode.  Applying it after cell
    # aggregation would otherwise not match native per-token dropout semantics.
    assert batch * candidate_count == response.shape[0]
    return response


@torch.inference_mode()
def intervene_oracle_address_moment_memory(
    model,
    reuse_cache: HSTUKVCache,
    memory: OracleAddressMomentMemory,
    candidate_ids: torch.Tensor,
    query_time_deltas: torch.Tensor,
    *,
    query_type_ids: torch.Tensor | None = None,
    query_action_ids: torch.Tensor | None = None,
    candidate_item_vectors: torch.Tensor | None = None,
) -> OracleAddressMomentIntervention:
    """Run a coherent Current reader with a moment residual at every layer."""

    _validate_cache(reuse_cache, "reuse_cache")
    if model.training:
        raise ValueError("address-moment intervention requires model.eval()")
    if candidate_ids.ndim != 2 or candidate_ids.shape[1] < 1:
        raise ValueError("candidate_ids must have shape [B,C] with C >= 1")
    batch, candidates = candidate_ids.shape
    if reuse_cache.k.shape[1] != batch:
        raise ValueError("Reuse cache and candidate batches differ")
    if reuse_cache.seq_len != memory.source_length:
        raise ValueError("Reuse cache length differs from moment-memory source")
    if reuse_cache.k.shape[0] != memory.centers.shape[0]:
        raise ValueError("Reuse cache and moment memory layer counts differ")
    if len(model.blocks) != memory.centers.shape[0]:
        raise ValueError("model and moment memory layer counts differ")

    x = model.embed_query_tokens(
        candidate_ids,
        query_time_deltas,
        query_type_ids=query_type_ids,
        query_action_ids=query_action_ids,
        item_vectors=candidate_item_vectors,
    ).reshape(batch * candidates, 1, model.cfg.hidden_size)
    layer_residuals: list[torch.Tensor] = []
    for layer, block in enumerate(model.blocks):
        residual_x = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        reuse_heads = _native_prefix_heads(
            block.attn, q, reuse_cache, layer, candidates
        )
        moment_residual = read_address_moment_residual(
            block.attn,
            q,
            memory,
            layer=layer,
            candidate_count=candidates,
        )
        layer_residuals.append(
            moment_residual.reshape(
                batch,
                candidates,
                block.attn.num_heads,
                block.attn.head_dim,
            )
        )
        self_heads = _native_self_heads(
            block.attn, q, k_new, v_new, memory.source_length
        )
        x = residual_x + _block_update(
            block, x_norm, reuse_heads + moment_residual + self_heads
        )

    readout = model.final_norm(x).reshape(batch, candidates, model.cfg.hidden_size)
    return OracleAddressMomentIntervention(
        scores=model.cc_score_head(readout).squeeze(-1),
        readout=readout,
        layer_residual_heads=tuple(layer_residuals),
    )
