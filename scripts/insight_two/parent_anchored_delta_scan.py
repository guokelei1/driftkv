"""Preflight algebra for a Parent-anchored finite-release delta scan.

This module deliberately separates two facts that are easy to conflate:

* joint Parent K/V plus one RMS denominator per token is an information-
  sufficient checkpoint for the legacy HSTU pre-block residual state; and
* decoding that checkpoint, then evaluating the missing historical query and
  gate coordinates, is not compute-sufficient under EvoKV's 20% budget.

The routines below are exact, non-learned interface checks and a static FLOP
lower bound.  They do not implement a migration action and do not consume
Current-Exact state, candidates, or labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


def _validate_square_weight(weight: torch.Tensor, name: str) -> int:
    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError(f"{name} must be a square rank-2 tensor")
    if not weight.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    return int(weight.shape[0])


@torch.inference_mode()
def joint_kv_decoder(
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
) -> torch.Tensor:
    """Return the minimum-norm decoder from ``[K,V]`` to RMSNorm output.

    PyTorch linear layers use ``K = N @ W_K.T`` and ``V = N @ W_V.T``.
    Therefore ``[K,V] = N @ B`` for ``B=[W_K.T,W_V.T]`` and the decoder is
    ``pinv(B)``.  It is model-global and can be compiled once per release.
    Double precision is used to keep this interface oracle separate from a
    production precision choice.
    """

    hidden = _validate_square_weight(key_weight, "key_weight")
    if _validate_square_weight(value_weight, "value_weight") != hidden:
        raise ValueError("key/value projection widths differ")
    if key_weight.device != value_weight.device:
        raise ValueError("key/value weights must share a device")
    basis = torch.cat(
        (key_weight.double().transpose(0, 1), value_weight.double().transpose(0, 1)),
        dim=1,
    )
    if int(torch.linalg.matrix_rank(basis)) != hidden:
        raise ValueError("joint K/V projection is not injective")
    return torch.linalg.pinv(basis)


@torch.inference_mode()
def recover_parent_rms_output(
    key: torch.Tensor,
    value: torch.Tensor,
    decoder: torch.Tensor,
) -> torch.Tensor:
    """Decode the Parent RMSNorm output from cache-layout K/V."""

    if key.ndim != 3 or key.shape != value.shape:
        raise ValueError("key/value must share [B,N,H] layout")
    if decoder.shape != (2 * key.shape[-1], key.shape[-1]):
        raise ValueError("decoder shape differs from K/V width")
    return torch.cat((key.double(), value.double()), dim=-1) @ decoder


@torch.inference_mode()
def recover_parent_preblock_hidden(
    key: torch.Tensor,
    value: torch.Tensor,
    decoder: torch.Tensor,
    rms_denominator: torch.Tensor,
    norm_weight: torch.Tensor,
) -> torch.Tensor:
    """Recover the exact pre-block residual checkpoint in oracle precision.

    ``rms_denominator`` is ``sqrt(mean(x**2)+eps)`` with shape ``[B,N,1]``.
    It is the only user-specific metadata beyond the existing Parent K/V.
    """

    normalized = recover_parent_rms_output(key, value, decoder)
    if rms_denominator.shape != normalized.shape[:-1] + (1,):
        raise ValueError("rms_denominator must have shape [B,N,1]")
    if norm_weight.shape != (normalized.shape[-1],):
        raise ValueError("norm_weight width differs from K/V")
    if bool((norm_weight == 0).any()):
        raise ValueError("zero RMSNorm weight prevents inversion")
    return (
        normalized
        / norm_weight.to(device=normalized.device, dtype=normalized.dtype)
        * rms_denominator.to(device=normalized.device, dtype=normalized.dtype)
    )


@dataclass(frozen=True)
class ParentAnchoredDeltaScanFloor:
    """Optimistic lower bound before any actual delta attention arithmetic."""

    exact_all_flops: int
    one_dense_history_transform_flops: int
    rms_metadata_scalars: int
    parent_kv_scalars: int
    k_only_checkpoint_decode_flops: int
    stable_joint_checkpoint_decode_flops: int
    historical_query_and_gate_flops: int
    k_only_subtotal_flops: int
    stable_joint_subtotal_flops: int
    k_only_subtotal_over_exact: float
    stable_joint_subtotal_over_exact: float
    omitted_work: str

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def medium_parent_anchored_delta_scan_floor() -> ParentAnchoredDeltaScanFloor:
    """Return the strict Medium lower bound under the repository FLOP rules.

    Six normalized checkpoints are required to form all six cache layers.
    Only the first five layers need a historical attention query and HSTU
    gate, because layer six can terminate after K/V projection.  A generic
    exact query or gate coordinate is one ``N x H`` by ``H x H`` transform.

    The bound intentionally charges neither input/current defects nor RMSNorm,
    attention moments, output projection, nonlinearities, compression,
    sidecar construction, or I/O.  Exceeding 20% here is therefore decisive
    for this KV-plus-scalar source interface, not a pessimistic kernel model.
    """

    context = 1024
    hidden = 192
    cache_layers = 6
    active_blocks = cache_layers - 1
    exact_all = 4_771_282_944
    dense = 2 * context * hidden * hidden

    # A single square K inverse is an algebraic but ill-conditioned best case.
    k_only_decode = cache_layers * dense
    # Stable joint decoding multiplies N x 2H by a 2H x H right inverse.
    joint_decode = 2 * cache_layers * dense
    # Q (attention score coordinates) and G (elementwise residual gate) are
    # independent H-wide coordinates absent from a K/V-only cache.
    query_and_gate = 2 * active_blocks * dense
    k_subtotal = k_only_decode + query_and_gate
    joint_subtotal = joint_decode + query_and_gate
    return ParentAnchoredDeltaScanFloor(
        exact_all_flops=exact_all,
        one_dense_history_transform_flops=dense,
        rms_metadata_scalars=cache_layers * context,
        parent_kv_scalars=2 * cache_layers * context * hidden,
        k_only_checkpoint_decode_flops=k_only_decode,
        stable_joint_checkpoint_decode_flops=joint_decode,
        historical_query_and_gate_flops=query_and_gate,
        k_only_subtotal_flops=k_subtotal,
        stable_joint_subtotal_flops=joint_subtotal,
        k_only_subtotal_over_exact=k_subtotal / exact_all,
        stable_joint_subtotal_over_exact=joint_subtotal / exact_all,
        omitted_work=(
            "Current input/state defect, RMSNorm, delta K/V, ELU+1 region handling, "
            "causal prefix scans, response/output projection, SiLU/Hadamard, "
            "compression, sidecar construction, and all I/O"
        ),
    )

