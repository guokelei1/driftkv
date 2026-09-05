"""History-mode replay diagnostics for a single Parent-to-Current release.

The scientific object in this module is a *dense token-axis mode*, not a
selected token and not a hidden-channel projection.  A reduced Current replay
keeps every history position in a small user-specific token subspace.  The
resulting Current coefficients can then replace only the same subspace of an
existing Parent cache::

    K_migrated = K_parent + U (U^T K_current_reduced - U^T K_parent)
    V_migrated = V_parent + U (U^T V_current_reduced - U^T V_parent)

The orthogonal complement of the Parent cache is therefore preserved.  This
is deliberately different from storing a globally compressed Current cache.
It is also not yet an admitted Design: the dense implementation below is a
semantic prototype.  A factorized executor and an honest cost audit are
required before a formal migration action can be claimed.

No function accepts Current-Exact upper-layer K/V or request candidates.
Truncated SVD is used only as a deterministic prototype implementation of the
token-mode restriction; it is not itself a novelty claim.  The single-Current
path is an xKV-adjacent compression control.  ``paired_release_replay`` is the
stronger structural diagnostic: Parent and Current are advanced at the same
fixed numerical resolution, and the finite release defect is carried between
layers explicitly.  Its current audited cost is above the admitted 20% action
budget, so this module must not be cited as an accepted Design 1 executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from hstu_kvcache.models import HSTUKVCache


@dataclass(frozen=True)
class TokenModeFactors:
    """A batched matrix ``left @ right`` in token-axis factor form."""

    left: torch.Tensor
    right: torch.Tensor

    def __post_init__(self) -> None:
        if self.left.ndim != 3 or self.right.ndim != 3:
            raise ValueError("token-mode factors must be rank-3 tensors")
        if self.left.shape[0] != self.right.shape[0]:
            raise ValueError("token-mode factor batches differ")
        if self.left.shape[2] != self.right.shape[1]:
            raise ValueError("token-mode inner ranks differ")
        if self.left.device != self.right.device:
            raise ValueError("token-mode factors must share a device")
        if self.left.dtype != self.right.dtype:
            raise ValueError("token-mode factors must share a dtype")

    @property
    def rank(self) -> int:
        return int(self.left.shape[2])

    def materialize(self) -> torch.Tensor:
        return self.left @ self.right


@dataclass(frozen=True)
class ReducedReplay:
    """Dense semantic result plus the per-layer cache token bases."""

    cache: HSTUKVCache
    cache_bases: tuple[torch.Tensor, ...]
    block_input_factors: tuple[TokenModeFactors, ...]


@dataclass(frozen=True)
class ModeSplice:
    """Parent cache plus a signed low-rank replacement sidecar."""

    cache: HSTUKVCache
    delta_k_cores: tuple[torch.Tensor, ...]
    delta_v_cores: tuple[torch.Tensor, ...]
    bases: tuple[torch.Tensor, ...]

    @property
    def sidecar_scalars(self) -> int:
        return sum(
            basis.numel() + delta_k.numel() + delta_v.numel()
            for basis, delta_k, delta_v in zip(
                self.bases,
                self.delta_k_cores,
                self.delta_v_cores,
                strict=True,
            )
        )


@dataclass(frozen=True)
class FactorizedCacheLayer:
    """One layer's K/V sharing the normalized-state token factor."""

    left: torch.Tensor
    key_core: torch.Tensor
    value_core: torch.Tensor

    def __post_init__(self) -> None:
        TokenModeFactors(self.left, self.key_core)
        TokenModeFactors(self.left, self.value_core)
        if self.key_core.shape != self.value_core.shape:
            raise ValueError("factorized K/V cores differ in shape")

    @property
    def rank(self) -> int:
        return int(self.left.shape[2])

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.left @ self.key_core, self.left @ self.value_core


@dataclass(frozen=True)
class FactorizedReplay:
    """Bounded-work replay semantics and materialized diagnostic cache."""

    cache: HSTUKVCache
    layers: tuple[FactorizedCacheLayer, ...]
    block_input_factors: tuple[TokenModeFactors, ...]


@dataclass(frozen=True)
class PairedReleaseReplay:
    """Equal-resolution Parent/Current trajectories and their layer defects.

    ``post_block_defects[l]`` is exactly the difference between the two
    approximate post-block states used to seed layer ``l + 1``.  Keeping this
    field makes the finite-release recurrence observable in tests; it is
    transient diagnostic state, not a proposed persistent sidecar.
    """

    parent: FactorizedReplay
    current: FactorizedReplay
    post_block_defects: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if len(self.parent.layers) != len(self.current.layers):
            raise ValueError("paired replay layer counts differ")
        if len(self.post_block_defects) != len(self.parent.layers):
            raise ValueError("one post-block defect is required per layer")
        for parent_layer, current_layer in zip(
            self.parent.layers, self.current.layers, strict=True
        ):
            if parent_layer.rank != current_layer.rank:
                raise ValueError("paired replay arms must use equal rank")


@dataclass(frozen=True)
class SharedModeSplice:
    """One history basis shared by signed K/V cores at every layer."""

    cache: HSTUKVCache
    basis: torch.Tensor
    delta_k_cores: tuple[torch.Tensor, ...]
    delta_v_cores: tuple[torch.Tensor, ...]

    @property
    def sidecar_scalars(self) -> int:
        return self.basis.numel() + sum(
            key.numel() + value.numel()
            for key, value in zip(
                self.delta_k_cores, self.delta_v_cores, strict=True
            )
        )


def _validate_matrix(matrix: torch.Tensor) -> tuple[int, int, int]:
    if matrix.ndim != 3 or not matrix.is_floating_point():
        raise ValueError("matrix must be floating point with shape [B,N,D]")
    batch, rows, columns = matrix.shape
    if min(batch, rows, columns) < 1:
        raise ValueError("matrix dimensions must be positive")
    return int(batch), int(rows), int(columns)


@torch.inference_mode()
def truncated_token_factors(
    matrix: torch.Tensor,
    *,
    rank: int,
) -> TokenModeFactors:
    """Return the deterministic best rank-``rank`` token-axis prototype.

    The singular values are folded into ``left``.  Production execution must
    replace the full SVD with a preregistered bounded-work range finder or an
    algebraically carried basis and count that work explicitly.
    """

    _, rows, columns = _validate_matrix(matrix)
    if not 1 <= rank <= min(rows, columns):
        raise ValueError("rank must be in [1,min(N,D)]")
    left, singular, right = torch.linalg.svd(matrix.float(), full_matrices=False)
    coefficients = left[:, :, :rank] * singular[:, None, :rank]
    return TokenModeFactors(
        left=coefficients.to(dtype=matrix.dtype),
        right=right[:, :rank].to(dtype=matrix.dtype),
    )


@torch.inference_mode()
def orthonormal_token_basis(
    matrix: torch.Tensor,
    *,
    rank: int,
) -> torch.Tensor:
    """Return the leading orthonormal left singular vectors of ``matrix``."""

    _, rows, columns = _validate_matrix(matrix)
    if not 1 <= rank <= min(rows, columns):
        raise ValueError("rank must be in [1,min(N,D)]")
    left, _, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
    return left[:, :, :rank].to(dtype=matrix.dtype)


def _fixed_gaussian(
    rows: int,
    columns: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate a device-independent, label-free Gaussian test matrix."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(rows, columns, generator=generator, dtype=torch.float32).to(
        device=device, dtype=dtype
    )


@torch.inference_mode()
def randomized_token_factors(
    matrix: torch.Tensor,
    *,
    rank: int,
    oversample: int = 4,
    power_iterations: int = 1,
    seed: int = 17,
) -> TokenModeFactors:
    """Bounded-work deterministic range-finder compression.

    The Gaussian matrix and iteration count are fixed independently of users,
    releases and quality outcomes.  This is the executable numerical kernel
    used to test mode replay; it remains an implementation component rather
    than the research contribution.
    """

    batch, rows, columns = _validate_matrix(matrix)
    if not 1 <= rank <= min(rows, columns):
        raise ValueError("rank must be in [1,min(N,D)]")
    if oversample < 0 or power_iterations < 0:
        raise ValueError("oversample and power_iterations must be nonnegative")
    sketch_rank = min(rank + oversample, rows, columns)
    omega = _fixed_gaussian(
        columns,
        sketch_rank,
        seed=seed,
        device=matrix.device,
        dtype=torch.float32,
    )
    source = matrix.float()
    sample = source @ omega
    for _ in range(power_iterations):
        q, _ = torch.linalg.qr(sample, mode="reduced")
        sample = source @ (source.transpose(1, 2) @ q)
    q, _ = torch.linalg.qr(sample, mode="reduced")
    core = q.transpose(1, 2) @ source
    small_left, singular, right = torch.linalg.svd(core, full_matrices=False)
    left = (q @ small_left[:, :, :rank]) * singular[:, None, :rank]
    return TokenModeFactors(
        left=left.to(dtype=matrix.dtype),
        right=right[:, :rank].to(dtype=matrix.dtype),
    )


@torch.inference_mode()
def randomized_token_basis(
    matrix: torch.Tensor,
    *,
    rank: int,
    oversample: int = 4,
    power_iterations: int = 1,
    seed: int = 17,
) -> torch.Tensor:
    """Return the orthonormal token span of the bounded-work factors."""

    factors = randomized_token_factors(
        matrix,
        rank=rank,
        oversample=oversample,
        power_iterations=power_iterations,
        seed=seed,
    )
    basis, _ = torch.linalg.qr(factors.left.float(), mode="reduced")
    return basis.to(dtype=matrix.dtype)


def project_onto_token_basis(
    matrix: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Orthogonally project ``matrix`` onto a batched token-axis basis."""

    batch, rows, _ = _validate_matrix(matrix)
    if basis.ndim != 3 or basis.shape[:2] != (batch, rows):
        raise ValueError("basis must have shape [B,N,r]")
    if basis.device != matrix.device or basis.dtype != matrix.dtype:
        raise ValueError("basis and matrix must share device and dtype")
    gram = basis.transpose(1, 2) @ basis
    identity = torch.eye(
        basis.shape[2], device=basis.device, dtype=basis.dtype
    ).expand_as(gram)
    if not torch.allclose(gram, identity, atol=2e-4, rtol=2e-4):
        raise ValueError("basis columns must be orthonormal")
    return basis @ (basis.transpose(1, 2) @ matrix)


def joint_cache_basis(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    rank: int,
) -> torch.Tensor:
    """Find one token basis shared by a layer's reduced K and V."""

    if key.shape != value.shape:
        raise ValueError("key and value shapes differ")
    return orthonormal_token_basis(torch.cat((key, value), dim=-1), rank=rank)


def _validate_legacy_replay_model(model) -> None:
    if model.training:
        raise ValueError("reduced replay requires model.eval()")
    if not model.blocks:
        raise ValueError("model must contain at least one block")
    if model.cfg.block_variant != "legacy":
        raise ValueError("the current semantic prototype covers legacy blocks")
    if model.cfg.input_dropout != 0 and model.training:
        raise ValueError("dropout must be disabled during migration")


@torch.inference_mode()
def reduced_current_replay(
    model,
    embedded_history: torch.Tensor,
    *,
    rank: int,
    lengths: torch.Tensor | None = None,
    initial_basis: torch.Tensor | None = None,
    compression: Literal["exact_svd", "fixed_range_finder"] = "exact_svd",
    sketch_oversample: int = 4,
    sketch_power_iterations: int = 1,
    sketch_seed: int = 17,
) -> ReducedReplay:
    """Run Current blocks on a rank-restricted dense history trajectory.

    This routine intentionally calls the native dense block after each rank
    restriction.  It defines the approximation unambiguously and is suitable
    for small quality diagnostics.  It does *not* claim the dense PyTorch path
    has reduced runtime or FLOPs.
    """

    _validate_legacy_replay_model(model)
    batch, length, hidden = _validate_matrix(embedded_history)
    if hidden != model.cfg.hidden_size:
        raise ValueError("embedded history width differs from model")
    if not 1 <= rank <= min(length, hidden):
        raise ValueError("rank must be in [1,min(history,hidden)]")
    valid = None
    if lengths is not None:
        if lengths.shape != (batch,):
            raise ValueError("lengths must have shape [B]")
        lengths = lengths.to(device=embedded_history.device, dtype=torch.long)
        valid = (
            torch.arange(length, device=embedded_history.device)[None]
            < lengths[:, None]
        )
    x = embedded_history
    if valid is not None:
        x = x * valid.unsqueeze(-1)
    cache_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    cache_bases: list[torch.Tensor] = []
    input_factors: list[TokenModeFactors] = []
    def compress(values: torch.Tensor, layer: int) -> TokenModeFactors:
        if compression == "exact_svd":
            return truncated_token_factors(values, rank=rank)
        if compression == "fixed_range_finder":
            return randomized_token_factors(
                values,
                rank=rank,
                oversample=sketch_oversample,
                power_iterations=sketch_power_iterations,
                seed=sketch_seed + layer,
            )
        raise ValueError("compression must be exact_svd or fixed_range_finder")

    for layer, block in enumerate(model.blocks):
        if layer == 0 and initial_basis is not None:
            x = project_onto_token_basis(x, initial_basis)
            factors = compress(x, layer)
        else:
            factors = compress(x, layer)
            x = factors.materialize()
        if valid is not None:
            x = x * valid.unsqueeze(-1)
        input_factors.append(factors)
        x, (key, value) = block(x, return_kv=True)
        if valid is not None:
            x = x * valid.unsqueeze(-1)
            key = key * valid.unsqueeze(-1)
            value = value * valid.unsqueeze(-1)
        # RMSNorm row scaling and the Q/K/V linear maps preserve the input
        # token rank.  Recovering the orthonormal basis here is only a dense
        # prototype convenience; a factorized executor carries it directly.
        cache_bases.append(joint_cache_basis(key, value, rank=rank))
        cache_layers.append((key, value))
    return ReducedReplay(
        cache=HSTUKVCache.from_layer_list(cache_layers, seq_len=length),
        cache_bases=tuple(cache_bases),
        block_input_factors=tuple(input_factors),
    )


def _validate_cache_pair(
    parent: HSTUKVCache,
    reduced_current: HSTUKVCache,
) -> tuple[int, int, int, int]:
    if parent.k.ndim != 4 or parent.k.shape != parent.v.shape:
        raise ValueError("Parent cache must contain matching [L,B,N,D] K/V")
    if reduced_current.k.shape != parent.k.shape or reduced_current.v.shape != parent.v.shape:
        raise ValueError("Parent and reduced-Current cache shapes differ")
    if parent.seq_len != reduced_current.seq_len or parent.seq_len != parent.k.shape[2]:
        raise ValueError("cache seq_len differs")
    return tuple(int(value) for value in parent.k.shape)


@torch.inference_mode()
def splice_current_modes_into_parent(
    parent: HSTUKVCache,
    reduced_current: HSTUKVCache,
    bases: tuple[torch.Tensor, ...],
) -> ModeSplice:
    """Replace Parent coefficients only inside reduced Current token modes."""

    layers, batch, length, _ = _validate_cache_pair(parent, reduced_current)
    if len(bases) != layers:
        raise ValueError("one token basis is required per cache layer")
    migrated_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    delta_k_cores: list[torch.Tensor] = []
    delta_v_cores: list[torch.Tensor] = []
    detached_bases: list[torch.Tensor] = []
    for layer, basis in enumerate(bases):
        if basis.ndim != 3 or basis.shape[:2] != (batch, length):
            raise ValueError("cache basis must have shape [B,N,r]")
        parent_k = parent.k[layer]
        parent_v = parent.v[layer]
        current_k = reduced_current.k[layer]
        current_v = reduced_current.v[layer]
        basis = basis.to(device=parent_k.device, dtype=parent_k.dtype)
        delta_k = basis.transpose(1, 2) @ (current_k - parent_k)
        delta_v = basis.transpose(1, 2) @ (current_v - parent_v)
        migrated_layers.append(
            (
                parent_k + basis @ delta_k,
                parent_v + basis @ delta_v,
            )
        )
        detached_bases.append(basis.detach())
        delta_k_cores.append(delta_k.detach())
        delta_v_cores.append(delta_v.detach())
    return ModeSplice(
        cache=HSTUKVCache.from_layer_list(migrated_layers, seq_len=parent.seq_len),
        delta_k_cores=tuple(delta_k_cores),
        delta_v_cores=tuple(delta_v_cores),
        bases=tuple(detached_bases),
    )


@torch.inference_mode()
def dependency_free_layer0_defect_basis(
    parent_model,
    current_model,
    parent_embedded_history: torch.Tensor,
    current_embedded_history: torch.Tensor,
    *,
    rank: int,
) -> torch.Tensor:
    """Build a legal token basis from the exact layer-0 projection defect.

    Layer-0 K/V depends only on each raw event embedding and therefore has no
    contextual dependency closure.  Upper-layer Current state is never read.
    """

    _validate_legacy_replay_model(parent_model)
    _validate_legacy_replay_model(current_model)
    if parent_embedded_history.shape != current_embedded_history.shape:
        raise ValueError("Parent and Current embedded histories differ in shape")
    parent_norm = parent_model.blocks[0].norm(parent_embedded_history)
    current_norm = current_model.blocks[0].norm(current_embedded_history)
    parent_k, parent_v = parent_model.blocks[0].attn.project_kv(parent_norm)
    current_k, current_v = current_model.blocks[0].attn.project_kv(current_norm)
    defect = torch.cat((current_k - parent_k, current_v - parent_v), dim=-1)
    return orthonormal_token_basis(defect, rank=rank)


def factorized_legacy_attention(
    block,
    normalized_factors: TokenModeFactors,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate native legacy attention from token factors.

    The nonlinear causal attention matrix is still evaluated exactly.  Only
    its Q/K contractions and its product with V use the token rank.  This
    function is a numerical identity for ``normalized_factors.materialize()``
    and exposes the algebra a future bounded-work executor must implement.
    """

    if block.block_variant != "legacy":
        raise ValueError("factorized attention currently covers legacy blocks")
    output, key_factors, value_factors = _factorized_legacy_attention_impl(
        block, normalized_factors
    )
    return output, key_factors.materialize(), value_factors.materialize()


def _factorized_legacy_attention_impl(
    block,
    normalized_factors: TokenModeFactors,
) -> tuple[torch.Tensor, TokenModeFactors, TokenModeFactors]:
    """Internal attention implementation that leaves K/V factorized."""

    if block.block_variant != "legacy":
        raise ValueError("factorized attention currently covers legacy blocks")
    attention = block.attn
    if attention.training:
        raise ValueError("factorized attention requires evaluation mode")
    left = normalized_factors.left
    core = normalized_factors.right
    batch, length, rank = left.shape
    heads = attention.num_heads
    head_dim = attention.head_dim

    def projected(weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        dense = core @ weight.transpose(0, 1)
        if bias is not None:
            # A bias adds the all-ones token mode and is not rank preserving.
            # Medium uses bias=False; reject instead of hiding rank expansion.
            raise ValueError("factorized legacy projection requires bias=False")
        return dense.view(batch, rank, heads, head_dim).permute(0, 2, 1, 3)

    q_core = projected(attention.q_proj.weight, attention.q_proj.bias)
    k_core = projected(attention.k_proj.weight, attention.k_proj.bias)
    v_core = projected(attention.v_proj.weight, attention.v_proj.bias)
    # Q_h K_h^T = U (Cq_h Ck_h^T) U^T.  The pointwise activation may make
    # the N-by-N matrix full rank, but A_h V_h = (A_h U) Cv_h remains rank r.
    head_outputs = []
    keep = attention._build_keep_mask(
        length, None, left.device, left.dtype
    )
    positions = torch.arange(length, device=left.device)
    for head in range(heads):
        middle = q_core[:, head] @ k_core[:, head].transpose(1, 2)
        logits = (left @ middle) @ left.transpose(1, 2)
        logits = logits * attention.scale
        bias = attention._relative_position_bias(positions, positions, logits.dtype)
        if bias is not None:
            logits = logits + bias[:, head : head + 1].squeeze(1)
        # Native masks have a singleton head axis and broadcast across heads.
        weights = attention._activate(logits) * keep[:, 0]
        weights = attention.attn_dropout(weights)
        head_outputs.append((weights @ left) @ v_core[:, head])
    heads_dense = torch.stack(head_outputs, dim=1)
    output = attention._finish(heads_dense)
    key_core = k_core.permute(0, 2, 1, 3).reshape(batch, rank, -1)
    value_core = v_core.permute(0, 2, 1, 3).reshape(batch, rank, -1)
    return (
        output,
        TokenModeFactors(left=left, right=key_core),
        TokenModeFactors(left=left, right=value_core),
    )


def factorized_rmsnorm(norm, factors: TokenModeFactors) -> TokenModeFactors:
    """Apply repository RMSNorm exactly while retaining token factors."""

    if not hasattr(norm, "weight") or not hasattr(norm, "eps"):
        raise ValueError("factorized normalization requires RMSNorm")
    left = factors.left.float()
    right = factors.right.float()
    hidden = right.shape[2]
    metric = (right @ right.transpose(1, 2)) / hidden
    variance = torch.einsum("bnr,brs,bns->bn", left, metric, left)
    normalized_left = left * torch.rsqrt(variance + float(norm.eps)).unsqueeze(-1)
    normalized_right = right * norm.weight.float()[None, None, :]
    return TokenModeFactors(
        left=normalized_left.to(dtype=factors.left.dtype),
        right=normalized_right.to(dtype=factors.right.dtype),
    )


def _compress_token_modes(
    values: torch.Tensor,
    *,
    rank: int,
    compression: Literal["exact_svd", "fixed_range_finder"],
    oversample: int,
    power_iterations: int,
    seed: int,
) -> TokenModeFactors:
    if compression == "exact_svd":
        return truncated_token_factors(values, rank=rank)
    if compression == "fixed_range_finder":
        return randomized_token_factors(
            values,
            rank=rank,
            oversample=oversample,
            power_iterations=power_iterations,
            seed=seed,
        )
    raise ValueError("compression must be exact_svd or fixed_range_finder")


def _factorized_legacy_block_step(
    block,
    input_factors: TokenModeFactors,
) -> tuple[torch.Tensor, FactorizedCacheLayer]:
    """Advance one legacy block from a fixed token-factor input."""

    normalized = factorized_rmsnorm(block.norm, input_factors)
    attention_output, key_factors, value_factors = (
        _factorized_legacy_attention_impl(block, normalized)
    )
    if block.gating == "silu_gate":
        gate_core = normalized.right @ block.gate_proj.weight.transpose(0, 1)
        gate = F.silu(normalized.left @ gate_core)
        update = attention_output * gate
    elif block.gating == "glu":
        gate_core = normalized.right @ block.gate_proj.weight.transpose(0, 1)
        gate = torch.sigmoid(normalized.left @ gate_core)
        update = attention_output * gate
    elif block.gating == "none":
        update = attention_output
    else:
        raise ValueError("factorized replay does not cover the FFN variant")
    next_state = input_factors.materialize() + update
    return next_state, FactorizedCacheLayer(
        left=normalized.left,
        key_core=key_factors.right,
        value_core=value_factors.right,
    )


@torch.inference_mode()
def factorized_reduced_current_replay(
    model,
    embedded_history: torch.Tensor,
    *,
    rank: int,
    compression: Literal["exact_svd", "fixed_range_finder"] = "fixed_range_finder",
    sketch_oversample: int = 4,
    sketch_power_iterations: int = 1,
    sketch_seed: int = 17,
) -> FactorizedReplay:
    """Execute the reduced Current trajectory without dense Q/K/V projections.

    Attention logits retain the model's native nonlinear activation and causal
    mask.  Dense ``N x H`` state is materialized only at the gate/residual
    rank-expansion boundary before the next fixed compression.
    """

    _validate_legacy_replay_model(model)
    _, length, hidden = _validate_matrix(embedded_history)
    if hidden != model.cfg.hidden_size:
        raise ValueError("embedded history width differs from model")
    if not 1 <= rank <= min(length, hidden):
        raise ValueError("rank must be in [1,min(history,hidden)]")
    dense_state = embedded_history
    cache_layers: list[FactorizedCacheLayer] = []
    input_layers: list[TokenModeFactors] = []
    materialized: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer, block in enumerate(model.blocks):
        input_factors = _compress_token_modes(
            dense_state,
            rank=rank,
            compression=compression,
            oversample=sketch_oversample,
            power_iterations=sketch_power_iterations,
            seed=sketch_seed + layer,
        )
        input_layers.append(input_factors)
        dense_state, cache_layer = _factorized_legacy_block_step(
            block, input_factors
        )
        key, value = cache_layer.materialize()
        cache_layers.append(cache_layer)
        materialized.append((key, value))
    return FactorizedReplay(
        cache=HSTUKVCache.from_layer_list(materialized, seq_len=length),
        layers=tuple(cache_layers),
        block_input_factors=tuple(input_layers),
    )


@torch.inference_mode()
def paired_release_replay(
    parent_model,
    current_model,
    parent_embedded_history: torch.Tensor,
    current_embedded_history: torch.Tensor,
    *,
    rank: int,
    compression: Literal["exact_svd", "fixed_range_finder"] = "fixed_range_finder",
    sketch_oversample: int = 4,
    sketch_power_iterations: int = 1,
    sketch_seed: int = 17,
) -> PairedReleaseReplay:
    """Propagate a finite Parent-to-Current defect at equal resolution.

    The recurrence is written in base-plus-defect form rather than hidden as
    two unrelated replays::

        Xc_l = Xp_l + D_l
        Xp_{l+1} = Fp_l(Compress(Xp_l, r))
        D_{l+1} = Fc_l(Compress(Xc_l, r)) - Xp_{l+1}

    Both arms use the same rank, sketch width, power count and layer seed, but
    obtain their own data-dependent token bases.  No Current-Exact upper-layer
    state is an input.  At full token rank the recurrence reduces to the two
    native release trajectories exactly (up to numerical decomposition error).
    """

    _validate_legacy_replay_model(parent_model)
    _validate_legacy_replay_model(current_model)
    if len(parent_model.blocks) != len(current_model.blocks):
        raise ValueError("Parent and Current layer counts differ")
    if parent_embedded_history.shape != current_embedded_history.shape:
        raise ValueError("Parent and Current embedded histories differ in shape")
    _, length, hidden = _validate_matrix(parent_embedded_history)
    if hidden != parent_model.cfg.hidden_size or hidden != current_model.cfg.hidden_size:
        raise ValueError("embedded history width differs from a release model")
    if not 1 <= rank <= min(length, hidden):
        raise ValueError("rank must be in [1,min(history,hidden)]")

    parent_state = parent_embedded_history
    defect_state = current_embedded_history - parent_embedded_history
    parent_inputs: list[TokenModeFactors] = []
    current_inputs: list[TokenModeFactors] = []
    parent_layers: list[FactorizedCacheLayer] = []
    current_layers: list[FactorizedCacheLayer] = []
    parent_materialized: list[tuple[torch.Tensor, torch.Tensor]] = []
    current_materialized: list[tuple[torch.Tensor, torch.Tensor]] = []
    post_block_defects: list[torch.Tensor] = []

    for layer, (parent_block, current_block) in enumerate(
        zip(parent_model.blocks, current_model.blocks, strict=True)
    ):
        current_state = parent_state + defect_state
        shared_arguments = {
            "rank": rank,
            "compression": compression,
            "oversample": sketch_oversample,
            "power_iterations": sketch_power_iterations,
            "seed": sketch_seed + layer,
        }
        parent_factors = _compress_token_modes(parent_state, **shared_arguments)
        current_factors = _compress_token_modes(current_state, **shared_arguments)
        parent_next, parent_cache_layer = _factorized_legacy_block_step(
            parent_block, parent_factors
        )
        current_next, current_cache_layer = _factorized_legacy_block_step(
            current_block, current_factors
        )
        defect_state = current_next - parent_next
        parent_state = parent_next

        parent_inputs.append(parent_factors)
        current_inputs.append(current_factors)
        parent_layers.append(parent_cache_layer)
        current_layers.append(current_cache_layer)
        parent_materialized.append(parent_cache_layer.materialize())
        current_materialized.append(current_cache_layer.materialize())
        post_block_defects.append(defect_state)

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
    return PairedReleaseReplay(
        parent=parent_replay,
        current=current_replay,
        post_block_defects=tuple(post_block_defects),
    )


@torch.inference_mode()
def approximate_layer0_defect_basis(
    parent: HSTUKVCache,
    replay: FactorizedReplay,
    *,
    rank: int,
    oversample: int = 4,
    power_iterations: int = 1,
    seed: int = 1017,
) -> torch.Tensor:
    """Form the shared release basis from legal approximate layer-0 drift."""

    layers, batch, length, _ = _validate_cache_pair(parent, replay.cache)
    if layers != len(replay.layers):
        raise ValueError("factorized replay layer count differs")
    current_k, current_v = replay.layers[0].materialize()
    defect = torch.cat(
        (current_k - parent.k[0], current_v - parent.v[0]), dim=-1
    )
    if defect.shape[:2] != (batch, length):
        raise RuntimeError("layer-0 defect layout differs")
    return randomized_token_basis(
        defect,
        rank=rank,
        oversample=oversample,
        power_iterations=power_iterations,
        seed=seed,
    )


@torch.inference_mode()
def approximate_paired_layer0_defect_basis(
    replay: PairedReleaseReplay,
    *,
    rank: int,
    oversample: int = 4,
    power_iterations: int = 0,
    seed: int = 1017,
) -> torch.Tensor:
    """Build ``U0`` only from the two approximate layer-0 release arms."""

    if not replay.parent.layers:
        raise ValueError("paired replay has no cache layers")
    parent_key, parent_value = replay.parent.layers[0].materialize()
    current_key, current_value = replay.current.layers[0].materialize()
    defect = torch.cat(
        (current_key - parent_key, current_value - parent_value), dim=-1
    )
    return randomized_token_basis(
        defect,
        rank=rank,
        oversample=oversample,
        power_iterations=power_iterations,
        seed=seed,
    )


@torch.inference_mode()
def splice_shared_modes_from_factorized_replay(
    parent: HSTUKVCache,
    replay: FactorizedReplay,
    basis: torch.Tensor,
) -> SharedModeSplice:
    """Compile Parent-plus-defect cores without materializing upper Current K/V."""

    layers, batch, length, _ = _validate_cache_pair(parent, replay.cache)
    if len(replay.layers) != layers:
        raise ValueError("factorized replay layer count differs")
    if basis.ndim != 3 or basis.shape[:2] != (batch, length):
        raise ValueError("shared basis must have shape [B,N,r]")
    basis = basis.to(device=parent.k.device, dtype=parent.k.dtype)
    gram = basis.transpose(1, 2) @ basis
    identity = torch.eye(
        basis.shape[2], device=basis.device, dtype=basis.dtype
    ).expand_as(gram)
    if not torch.allclose(gram, identity, atol=2e-4, rtol=2e-4):
        raise ValueError("shared basis columns must be orthonormal")
    migrated: list[tuple[torch.Tensor, torch.Tensor]] = []
    delta_keys: list[torch.Tensor] = []
    delta_values: list[torch.Tensor] = []
    basis_transpose = basis.transpose(1, 2)
    for layer, factors in enumerate(replay.layers):
        alignment = basis_transpose @ factors.left
        current_key_core = alignment @ factors.key_core
        current_value_core = alignment @ factors.value_core
        delta_key = current_key_core - basis_transpose @ parent.k[layer]
        delta_value = current_value_core - basis_transpose @ parent.v[layer]
        migrated.append(
            (
                parent.k[layer] + basis @ delta_key,
                parent.v[layer] + basis @ delta_value,
            )
        )
        delta_keys.append(delta_key.detach())
        delta_values.append(delta_value.detach())
    return SharedModeSplice(
        cache=HSTUKVCache.from_layer_list(migrated, seq_len=parent.seq_len),
        basis=basis.detach(),
        delta_k_cores=tuple(delta_keys),
        delta_v_cores=tuple(delta_values),
    )


@torch.inference_mode()
def splice_shared_modes_from_paired_replay(
    exact_parent: HSTUKVCache,
    replay: PairedReleaseReplay,
    basis: torch.Tensor,
) -> SharedModeSplice:
    """Add the paired approximate release defect to exact Parent state.

    Unlike the single-arm control, subtraction is performed between matched
    approximate Parent and Current trajectories.  Exact Parent participates
    only as the immutable serving control variate to which the resulting
    signed cores are added.
    """

    layers, batch, length, _ = _validate_cache_pair(
        exact_parent, replay.current.cache
    )
    if replay.parent.cache.k.shape != exact_parent.k.shape:
        raise ValueError("approximate Parent and exact Parent cache shapes differ")
    if len(replay.parent.layers) != layers or len(replay.current.layers) != layers:
        raise ValueError("paired replay layer count differs")
    if basis.ndim != 3 or basis.shape[:2] != (batch, length):
        raise ValueError("shared basis must have shape [B,N,r]")
    basis = basis.to(device=exact_parent.k.device, dtype=exact_parent.k.dtype)
    gram = basis.transpose(1, 2) @ basis
    identity = torch.eye(
        basis.shape[2], device=basis.device, dtype=basis.dtype
    ).expand_as(gram)
    if not torch.allclose(gram, identity, atol=2e-4, rtol=2e-4):
        raise ValueError("shared basis columns must be orthonormal")

    migrated: list[tuple[torch.Tensor, torch.Tensor]] = []
    delta_keys: list[torch.Tensor] = []
    delta_values: list[torch.Tensor] = []
    basis_transpose = basis.transpose(1, 2)
    for layer, (parent_factors, current_factors) in enumerate(
        zip(replay.parent.layers, replay.current.layers, strict=True)
    ):
        parent_alignment = basis_transpose @ parent_factors.left
        current_alignment = basis_transpose @ current_factors.left
        delta_key = (
            current_alignment @ current_factors.key_core
            - parent_alignment @ parent_factors.key_core
        )
        delta_value = (
            current_alignment @ current_factors.value_core
            - parent_alignment @ parent_factors.value_core
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
        cache=HSTUKVCache.from_layer_list(migrated, seq_len=exact_parent.seq_len),
        basis=basis.detach(),
        delta_k_cores=tuple(delta_keys),
        delta_v_cores=tuple(delta_values),
    )


def _legacy_attention_one_with_shared_splice(
    block,
    normalized_query: torch.Tensor,
    parent_key: torch.Tensor,
    parent_value: torch.Tensor,
    basis: torch.Tensor,
    delta_key_core: torch.Tensor,
    delta_value_core: torch.Tensor,
) -> torch.Tensor:
    """Read ``Parent + U DeltaCore`` without expanding the migrated cache."""

    attention = block.attn
    if block.block_variant != "legacy" or normalized_query.shape[1] != 1:
        raise ValueError("shared-splice reader requires one-token legacy input")
    batch, length, width = parent_key.shape
    if parent_value.shape != parent_key.shape:
        raise ValueError("Parent K/V shapes differ")
    if basis.shape[:2] != (batch, length):
        raise ValueError("shared basis shape differs")
    if delta_key_core.shape != (batch, basis.shape[2], width):
        raise ValueError("delta key core shape differs")
    if delta_value_core.shape != delta_key_core.shape:
        raise ValueError("delta value core shape differs")
    query, key_new, value_new = attention._project(normalized_query)
    heads, head_dim = attention.num_heads, attention.head_dim
    parent_k = parent_key.view(batch, length, heads, head_dim).transpose(1, 2)
    parent_v = parent_value.view(batch, length, heads, head_dim).transpose(1, 2)
    key_core = delta_key_core.view(
        batch, basis.shape[2], heads, head_dim
    ).permute(0, 2, 1, 3)
    value_core = delta_value_core.view(
        batch, basis.shape[2], heads, head_dim
    ).permute(0, 2, 1, 3)
    logits = torch.matmul(query, parent_k.transpose(-2, -1))
    mode_logits = torch.matmul(query, key_core.transpose(-2, -1))
    logits = logits + torch.matmul(mode_logits, basis.transpose(1, 2)[:, None])
    logits = logits * attention.scale
    query_position = torch.tensor([length], device=query.device)
    key_positions = torch.arange(length, device=query.device)
    position_bias = attention._relative_position_bias(
        query_position, key_positions, logits.dtype
    )
    if position_bias is not None:
        logits = logits + position_bias
    weights = attention.attn_dropout(attention._activate(logits))
    prefix = torch.matmul(weights, parent_v)
    mode_weights = torch.matmul(weights, basis[:, None])
    prefix = prefix + torch.matmul(mode_weights, value_core)

    self_weight = (query * key_new).sum(dim=-1, keepdim=True) * attention.scale
    self_bias = attention._relative_position_bias(
        query_position, query_position, self_weight.dtype
    )
    if self_bias is not None:
        self_weight = self_weight + self_bias
    self_weight = attention.attn_dropout(attention._activate(self_weight))
    return attention._finish(prefix + self_weight * value_new)


@torch.inference_mode()
def forward_one_with_shared_mode_splice(
    model,
    parent: HSTUKVCache,
    splice: SharedModeSplice,
    embedded_query: torch.Tensor,
) -> torch.Tensor:
    """Execute one Current query against the factorized migration sidecar."""

    if embedded_query.ndim != 3 or embedded_query.shape[1:] != (
        1,
        model.cfg.hidden_size,
    ):
        raise ValueError("embedded query must have shape [B,1,H]")
    if parent.k.shape[1] != embedded_query.shape[0]:
        raise ValueError("query and Parent cache batches differ")
    if len(splice.delta_k_cores) != len(model.blocks):
        raise ValueError("splice layer count differs from model")
    x = embedded_query
    for layer, block in enumerate(model.blocks):
        residual = x
        normalized = block.norm(x)
        attention_output = _legacy_attention_one_with_shared_splice(
            block,
            normalized,
            parent.k[layer],
            parent.v[layer],
            splice.basis,
            splice.delta_k_cores[layer],
            splice.delta_v_cores[layer],
        )
        if block.gating == "silu_gate":
            update = attention_output * F.silu(block.gate_proj(normalized))
        elif block.gating == "glu":
            update = attention_output * torch.sigmoid(block.gate_proj(normalized))
        elif block.gating == "none":
            update = attention_output
        else:
            raise ValueError("shared-splice reader does not cover the FFN variant")
        x = residual + update
    return model.final_norm(x)
