"""Exact finite-release K/V response-coupling diagnostics.

This module is deliberately an oracle diagnostic, not a migration action.  At
one fixed Current-reader query it decomposes the prefix attention response as

``R(Kc,Vc)-R(Kp,Vp) = Dk + Dv + Dkv``

where ``Dk`` changes only the attention address, ``Dv`` changes only the value
content, and ``Dkv`` is their finite-release interaction.  The identity holds
for the model's native attention activation; no Taylor expansion, mapper, SVD
of K/V, or fitted target is involved.

The coherent interventions are useful falsifiers.  They ask whether routing
only, content only, or the additive first-order pair can reproduce the exact
joint endpoint as the Current query evolves through all Transformer layers.
They read Current Exact K/V and therefore must never enter an executable
frontier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from insight.reader_compatibility_correction import _self_heads

from hstu_kvcache.models import HSTUKVCache
from insight_two.cone_response_memory import _block_update

CouplingMode = Literal[
    "reuse",
    "current",
    "key_only",
    "value_only",
    "additive_no_interaction",
    "interaction_only",
]


@dataclass(frozen=True)
class KVResponseComponents:
    """Native prefix responses and their exact finite-release components."""

    parent: torch.Tensor
    current: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    interaction: torch.Tensor

    @property
    def joint(self) -> torch.Tensor:
        return self.current - self.parent

    @property
    def additive_no_interaction(self) -> torch.Tensor:
        return self.parent + self.key + self.value

    @property
    def interaction_only(self) -> torch.Tensor:
        return self.parent + self.interaction


@dataclass(frozen=True)
class CouplingIntervention:
    """One coherent reader output plus common-path layer diagnostics."""

    scores: torch.Tensor
    readout: torch.Tensor
    layer_metrics: tuple[dict[str, float | int], ...]


def _validate_cache_pair(
    current_cache: HSTUKVCache,
    parent_cache: HSTUKVCache,
) -> None:
    if current_cache.k.ndim != 4 or current_cache.k.shape != current_cache.v.shape:
        raise ValueError("Current cache must contain matching [L,B,N,W] K/V")
    if parent_cache.k.shape != current_cache.k.shape:
        raise ValueError("Parent and Current cache K shapes differ")
    if parent_cache.v.shape != current_cache.v.shape:
        raise ValueError("Parent and Current cache V shapes differ")
    if parent_cache.seq_len != current_cache.seq_len:
        raise ValueError("Parent and Current cache lengths differ")


def _cache_heads(attention, values: torch.Tensor, candidates: int) -> torch.Tensor:
    if values.ndim != 3 or values.shape[-1] != attention.inner:
        raise ValueError("cache layer must have shape [B,N,attention.inner]")
    if candidates < 1:
        raise ValueError("candidate count must be positive")
    batch, length, _ = values.shape
    repeated = values.repeat_interleave(candidates, dim=0)
    return repeated.view(
        batch * candidates,
        length,
        attention.num_heads,
        attention.head_dim,
    ).transpose(1, 2)


@torch.inference_mode()
def mixed_prefix_heads(
    attention,
    q: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    candidates: int,
) -> torch.Tensor:
    """Read arbitrary cache K/V pairing with the native attention kernel."""

    if key.shape != value.shape:
        raise ValueError("mixed prefix K/V shapes differ")
    keys = _cache_heads(attention, key, candidates)
    values = _cache_heads(attention, value, candidates)
    if q.shape[0] != keys.shape[0]:
        raise ValueError("query batch differs from repeated prefix batch")
    weights = attention._activate(
        torch.matmul(q, keys.transpose(-2, -1)) * attention.scale
    )
    if attention.block_variant == "hstu_reference":
        weights = weights / attention.cfg.max_seq_len
    weights = attention.attn_dropout(weights)
    return torch.matmul(weights, values)


@torch.inference_mode()
def decompose_prefix_response(
    attention,
    q: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    parent_k: torch.Tensor,
    parent_v: torch.Tensor,
    *,
    candidates: int,
) -> KVResponseComponents:
    """Return the exact K-only, V-only and KxV interaction decomposition."""

    pp = mixed_prefix_heads(
        attention, q, parent_k, parent_v, candidates=candidates
    )
    cp = mixed_prefix_heads(
        attention, q, current_k, parent_v, candidates=candidates
    )
    pc = mixed_prefix_heads(
        attention, q, parent_k, current_v, candidates=candidates
    )
    cc = mixed_prefix_heads(
        attention, q, current_k, current_v, candidates=candidates
    )
    return KVResponseComponents(
        parent=pp,
        current=cc,
        key=cp - pp,
        value=pc - pp,
        interaction=cc - cp - pc + pp,
    )


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.float().reshape(-1)
    right_flat = right.float().reshape(-1)
    denominator = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(
        right_flat
    )
    if float(denominator) <= 1e-20:
        return 0.0
    return float(torch.dot(left_flat, right_flat) / denominator)


def _norm(values: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(values.float()))


def _rank_at_energy(matrix: torch.Tensor, threshold: float) -> int:
    if matrix.ndim != 2:
        raise ValueError("rank diagnostic requires a matrix")
    singular = torch.linalg.svdvals(matrix.float())
    energy = singular.square()
    if float(energy.sum()) <= 1e-20:
        return 0
    cumulative = torch.cumsum(energy, dim=0) / energy.sum()
    return int(torch.searchsorted(cumulative, threshold).item() + 1)


def _candidate_matrix(values: torch.Tensor, candidates: int) -> torch.Tensor:
    if values.shape[0] % candidates:
        raise ValueError("response batch is not divisible by candidates")
    batch = values.shape[0] // candidates
    if batch != 1:
        raise ValueError("single-user preflight expected")
    return values.reshape(candidates, -1)


def coupling_metrics(
    block,
    x_norm: torch.Tensor,
    components: KVResponseComponents,
    *,
    candidates: int,
    layer: int,
) -> dict[str, float | int]:
    """Measure cancellation/interaction before and after gate/output transform."""

    joint = components.joint
    terms = (components.key, components.value, components.interaction)
    term_norm_sum = sum(_norm(term) for term in terms)
    joint_norm = _norm(joint)

    # Use finite differences around the Parent response so this remains valid
    # even if a future block adapter adds an affine bias.
    update_parent = _block_update(block, x_norm, components.parent)
    update_current = _block_update(block, x_norm, components.current)
    update_cp = _block_update(
        block, x_norm, components.parent + components.key
    )
    update_pc = _block_update(
        block, x_norm, components.parent + components.value
    )
    update_key = update_cp - update_parent
    update_value = update_pc - update_parent
    update_joint = update_current - update_parent
    update_interaction = update_joint - update_key - update_value
    update_term_sum = (
        _norm(update_key) + _norm(update_value) + _norm(update_interaction)
    )

    candidate_joint = _candidate_matrix(joint, candidates)
    centered_joint = candidate_joint - candidate_joint.mean(dim=0, keepdim=True)
    candidate_key = _candidate_matrix(components.key, candidates)
    candidate_value = _candidate_matrix(components.value, candidates)
    decomposition_error = torch.linalg.vector_norm(
        (
            joint
            - components.key
            - components.value
            - components.interaction
        ).float()
    ) / torch.linalg.vector_norm(joint.float()).clamp_min(1e-20)
    return {
        "layer": layer,
        "response_joint_norm": joint_norm,
        "response_key_norm": _norm(components.key),
        "response_value_norm": _norm(components.value),
        "response_interaction_norm": _norm(components.interaction),
        "response_joint_over_component_norm_sum": joint_norm
        / max(term_norm_sum, 1e-20),
        "response_key_value_cosine": _cosine(components.key, components.value),
        "response_key_interaction_cosine": _cosine(
            components.key, components.interaction
        ),
        "response_value_interaction_cosine": _cosine(
            components.value, components.interaction
        ),
        "response_interaction_over_joint": _norm(components.interaction)
        / max(joint_norm, 1e-20),
        "response_joint_rank90": _rank_at_energy(candidate_joint, 0.90),
        "response_joint_centered_rank90": _rank_at_energy(centered_joint, 0.90),
        "response_key_rank90": _rank_at_energy(candidate_key, 0.90),
        "response_value_rank90": _rank_at_energy(candidate_value, 0.90),
        "gated_joint_norm": _norm(update_joint),
        "gated_key_norm": _norm(update_key),
        "gated_value_norm": _norm(update_value),
        "gated_interaction_norm": _norm(update_interaction),
        "gated_joint_over_component_norm_sum": _norm(update_joint)
        / max(update_term_sum, 1e-20),
        "gated_key_value_cosine": _cosine(update_key, update_value),
        "finite_decomposition_max_abs_error": float(
            torch.max(
                torch.abs(
                    joint
                    - components.key
                    - components.value
                    - components.interaction
                )
            )
        ),
        "finite_decomposition_relative_l2_error": float(decomposition_error),
    }


def _selected_prefix(
    components: KVResponseComponents,
    mode: CouplingMode,
) -> torch.Tensor:
    if mode == "reuse":
        return components.parent
    if mode == "current":
        return components.current
    if mode == "key_only":
        return components.parent + components.key
    if mode == "value_only":
        return components.parent + components.value
    if mode == "additive_no_interaction":
        return components.additive_no_interaction
    if mode == "interaction_only":
        return components.interaction_only
    raise ValueError(f"unsupported coupling mode: {mode}")


@torch.inference_mode()
def intervene_kv_response_coupling(
    model,
    current_cache: HSTUKVCache,
    parent_cache: HSTUKVCache,
    candidate_ids: torch.Tensor,
    query_time_deltas: torch.Tensor,
    *,
    mode: CouplingMode,
) -> CouplingIntervention:
    """Run one coherent Current reader under a fixed K/V coupling ablation."""

    _validate_cache_pair(current_cache, parent_cache)
    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != 1:
        raise ValueError("single-user candidate_ids must have shape [1,C]")
    candidates = candidate_ids.shape[1]
    x = model.embed_query_tokens(candidate_ids, query_time_deltas).reshape(
        candidates, 1, model.cfg.hidden_size
    )
    metrics: list[dict[str, float | int]] = []
    for layer, block in enumerate(model.blocks):
        residual = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        components = decompose_prefix_response(
            block.attn,
            q,
            current_cache.k[layer],
            current_cache.v[layer],
            parent_cache.k[layer],
            parent_cache.v[layer],
            candidates=candidates,
        )
        metrics.append(
            coupling_metrics(
                block,
                x_norm,
                components,
                candidates=candidates,
                layer=layer,
            )
        )
        prefix = _selected_prefix(components, mode)
        self_response = _self_heads(block, q, k_new, v_new)
        x = residual + _block_update(block, x_norm, prefix + self_response)
    readout = model.final_norm(x).reshape(1, candidates, model.cfg.hidden_size)
    return CouplingIntervention(
        scores=model.cc_score_head(readout).squeeze(-1),
        readout=readout,
        layer_metrics=tuple(metrics),
    )
