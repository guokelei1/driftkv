"""Matrix-free initial token range finding for legacy HSTU histories.

This module is an execution component for the paired finite-release replay.
It does not change that replay's scientific object and is not a standalone
novelty claim.  Its only purpose is to avoid materializing either the dense
temporal projection or the post-``in_proj`` ``N x H`` input before the first
fixed range finder.

For an evaluation-mode legacy HSTU input,

``X = (E_item + E_behavior + Phi W_time^T) W_in^T``,

the range finder needs only applications of ``X`` and ``X^T``.  The routines
below evaluate those applications from embedding lookups, the small temporal
feature matrix ``Phi``, and model weights.  Four operator applications are
still required by the frozen ``power_iterations=1`` protocol; none is treated
as free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from insight_two.mode_space_replay import TokenModeFactors


@dataclass(frozen=True)
class MatrixFreeHSTUInput:
    """The linear operator for one release's embedded history input.

    ``base_features`` is the unavoidable item-plus-behavior lookup result and
    ``time_features`` has width ``2F``.  In particular, this object never
    stores a temporal projection or an input-projection result of shape
    ``[B,N,H]``.
    """

    base_features: torch.Tensor
    time_features: torch.Tensor
    temporal_weight: torch.Tensor
    input_weight: torch.Tensor

    def __post_init__(self) -> None:
        if self.base_features.ndim != 3 or not self.base_features.is_floating_point():
            raise ValueError("base_features must be floating point [B,N,H]")
        if self.time_features.ndim != 3 or not self.time_features.is_floating_point():
            raise ValueError("time_features must be floating point [B,N,2F]")
        if self.base_features.shape[:2] != self.time_features.shape[:2]:
            raise ValueError("base and time feature batch/token dimensions differ")
        hidden = self.base_features.shape[2]
        time_width = self.time_features.shape[2]
        if self.temporal_weight.shape != (hidden, time_width):
            raise ValueError("temporal weight must have shape [H,2F]")
        if self.input_weight.shape != (hidden, hidden):
            raise ValueError("input weight must have shape [H,H]")
        tensors = (
            self.base_features,
            self.time_features,
            self.temporal_weight,
            self.input_weight,
        )
        if any(value.device != tensors[0].device for value in tensors[1:]):
            raise ValueError("operator tensors must share a device")
        if any(value.dtype != torch.float32 for value in tensors):
            raise ValueError("the audited matrix-free prototype requires float32")

    @property
    def batch_size(self) -> int:
        return int(self.base_features.shape[0])

    @property
    def history_length(self) -> int:
        return int(self.base_features.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.base_features.shape[2])

    def _validate_right(self, right: torch.Tensor) -> None:
        if right.ndim == 2:
            valid = right.shape[0] == self.hidden_size
        elif right.ndim == 3:
            valid = right.shape[:2] == (self.batch_size, self.hidden_size)
        else:
            valid = False
        if not valid:
            raise ValueError("right operand must have shape [H,S] or [B,H,S]")
        if right.device != self.base_features.device or right.dtype != torch.float32:
            raise ValueError("right operand must share the operator device and float32 dtype")

    @torch.inference_mode()
    def right_multiply(self, right: torch.Tensor) -> torch.Tensor:
        """Return ``X @ right`` without forming ``X`` or temporal ``N x H``."""

        self._validate_right(right)
        # X R = A (W_in^T R) + Phi (W_time^T W_in^T R).
        projected = torch.matmul(self.input_weight.transpose(0, 1), right)
        base_result = torch.matmul(self.base_features, projected)
        time_core = torch.matmul(self.temporal_weight.transpose(0, 1), projected)
        return base_result + torch.matmul(self.time_features, time_core)

    @torch.inference_mode()
    def transpose_multiply(self, left: torch.Tensor) -> torch.Tensor:
        """Return ``X^T @ left`` without forming ``X`` or temporal ``N x H``."""

        if left.ndim != 3 or left.shape[:2] != (
            self.batch_size,
            self.history_length,
        ):
            raise ValueError("left operand must have shape [B,N,S]")
        if left.device != self.base_features.device or left.dtype != torch.float32:
            raise ValueError("left operand must share the operator device and float32 dtype")
        # X^T Q = W_in (A^T Q + W_time Phi^T Q).
        base_core = self.base_features.transpose(1, 2) @ left
        time_core = self.time_features.transpose(1, 2) @ left
        combined = base_core + torch.matmul(self.temporal_weight, time_core)
        return torch.matmul(self.input_weight, combined)


@torch.inference_mode()
def hstu_input_operator(
    model,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
) -> MatrixFreeHSTUInput:
    """Create one release-specific input operator from raw history fields.

    The Medium checkpoints are evaluated in float32.  Rejecting other model
    dtypes is deliberate: a half-precision dense formation would introduce
    intermediate roundings that a fused operator cannot claim to reproduce.
    """

    if model.training:
        raise ValueError("matrix-free input range finding requires model.eval()")
    if item_ids.ndim != 2 or behaviors.shape != item_ids.shape:
        raise ValueError("item_ids and behaviors must have shape [B,N]")
    if time_deltas.shape != item_ids.shape or not time_deltas.is_floating_point():
        raise ValueError("time_deltas must be floating point with shape [B,N]")
    if item_ids.device != behaviors.device or item_ids.device != time_deltas.device:
        raise ValueError("raw history fields must share a device")
    parameters = (
        model.item_emb.weight,
        model.behavior_emb.embed.weight,
        model.temporal_enc.proj.weight,
        model.in_proj.weight,
    )
    if any(value.dtype != torch.float32 for value in parameters):
        raise ValueError("the audited matrix-free prototype requires float32 model weights")
    if any(value.device != item_ids.device for value in parameters):
        raise ValueError("raw history and model weights must share a device")

    item_vectors = model.lookup_item_embeddings(item_ids)
    behavior_vectors = model.behavior_emb(behaviors)
    base_features = item_vectors + behavior_vectors

    num_freqs = int(model.temporal_enc.num_freqs)
    frequencies = torch.exp(
        -math.log(float(model.temporal_enc.max_period))
        * torch.arange(num_freqs, device=time_deltas.device, dtype=torch.float32)
        / num_freqs
    )
    phases = time_deltas.float().unsqueeze(-1) * frequencies
    time_features = torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)
    return MatrixFreeHSTUInput(
        base_features=base_features.float(),
        time_features=time_features,
        temporal_weight=model.temporal_enc.proj.weight,
        input_weight=model.in_proj.weight,
    )


def _fixed_gaussian(
    rows: int,
    columns: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Match ``mode_space_replay.randomized_token_factors`` exactly."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(
        rows,
        columns,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device)


@torch.inference_mode()
def matrix_free_randomized_token_factors(
    operator: MatrixFreeHSTUInput,
    *,
    rank: int,
    oversample: int = 4,
    power_iterations: int = 1,
    seed: int = 17,
    truncation: Literal["svd", "gram_eigh"] = "svd",
) -> TokenModeFactors:
    """Run the fixed token range finder using only ``X``/``X^T`` products.

    ``svd`` matches the existing dense semantic prototype's small-core
    decomposition.  ``gram_eigh`` is its algebraically equivalent executable
    form and corresponds to the explicit small-matrix cost in the ledger.
    Factor signs/rotations are not promised to be bitwise identical.
    """

    rows = operator.history_length
    columns = operator.hidden_size
    if not 1 <= rank <= min(rows, columns):
        raise ValueError("rank must be in [1,min(N,H)]")
    if oversample < 0 or power_iterations < 0:
        raise ValueError("oversample and power_iterations must be nonnegative")
    sketch_rank = min(rank + oversample, rows, columns)
    omega = _fixed_gaussian(
        columns,
        sketch_rank,
        seed=seed,
        device=operator.base_features.device,
    )
    sample = operator.right_multiply(omega)
    for _ in range(power_iterations):
        q, _ = torch.linalg.qr(sample, mode="reduced")
        sample = operator.right_multiply(operator.transpose_multiply(q))
    q, _ = torch.linalg.qr(sample, mode="reduced")
    core = operator.transpose_multiply(q).transpose(1, 2)
    if truncation == "svd":
        small_left, singular, right = torch.linalg.svd(core, full_matrices=False)
        left = (q @ small_left[:, :, :rank]) * singular[:, None, :rank]
        return TokenModeFactors(left=left, right=right[:, :rank])
    if truncation == "gram_eigh":
        _, rotation = torch.linalg.eigh(core @ core.transpose(1, 2))
        rotation = rotation[:, :, -rank:].flip(-1)
        return TokenModeFactors(
            left=q @ rotation,
            right=rotation.transpose(1, 2) @ core,
        )
    raise ValueError("truncation must be svd or gram_eigh")


@dataclass(frozen=True)
class MatrixFreeInputCost:
    """Transparent FLOP/non-FLOP ledger for one release arm."""

    base_lookup_additions: int
    temporal_phase_multiplications: int
    right_operator_applications: int
    transpose_operator_applications: int
    thin_qr: int
    small_gram_eigh_rotation: int
    gaussian_draws: int
    sin_cos_evaluations: int
    frequency_exponentials: int
    embedding_lookup_scalars: int
    raw_history_scalars: int

    @property
    def flops(self) -> int:
        return (
            self.base_lookup_additions
            + self.temporal_phase_multiplications
            + self.right_operator_applications
            + self.transpose_operator_applications
            + self.thin_qr
            + self.small_gram_eigh_rotation
        )


def matrix_free_input_cost(
    *,
    history_length: int,
    hidden_size: int,
    temporal_num_freqs: int,
    rank: int,
    oversample: int,
    power_iterations: int,
) -> MatrixFreeInputCost:
    """Count the initial matrix-free range finder under multiply-add=2.

    The count includes every operator application.  For the frozen
    ``power_iterations=1`` point there are two ``X@R`` and two ``X^T@Q``
    applications, two thin QRs, and one small Gram/eigendecomposition/rotation.
    Transcendental evaluations, RNG draws, and lookup traffic are returned
    separately rather than silently converted into matmul FLOPs.
    """

    n = history_length
    h = hidden_size
    f = temporal_num_freqs
    if min(n, h, f, rank) < 1:
        raise ValueError("dimensions and rank must be positive")
    if oversample < 0 or power_iterations < 0:
        raise ValueError("oversample and power_iterations must be nonnegative")
    s = min(rank + oversample, n, h)
    if rank > s:
        raise ValueError("rank exceeds the effective sketch width")
    time_width = 2 * f

    # X R: W_in^T R, A @ (...), W_time^T (...), Phi @ (...), final add.
    one_right = (
        2 * h * h * s
        + 2 * n * h * s
        + 2 * time_width * h * s
        + 2 * n * time_width * s
        + n * s
    )
    # X^T Q: A^T Q, Phi^T Q, W_time (...), add, W_in (...).
    one_transpose = (
        2 * n * h * s
        + 2 * n * time_width * s
        + 2 * h * time_width * s
        + h * s
        + 2 * h * h * s
    )
    right_count = 1 + power_iterations
    transpose_count = 1 + power_iterations
    qr_count = 1 + power_iterations
    one_qr = math.ceil(2 * n * s * s - (2 * s * s * s) / 3)
    truncate = (
        2 * h * s * s
        + 9 * s * s * s
        + 2 * n * s * rank
        + 2 * rank * s * h
    )
    return MatrixFreeInputCost(
        base_lookup_additions=n * h,
        temporal_phase_multiplications=n * f,
        right_operator_applications=right_count * one_right,
        transpose_operator_applications=transpose_count * one_transpose,
        thin_qr=qr_count * one_qr,
        small_gram_eigh_rotation=truncate,
        gaussian_draws=h * s,
        sin_cos_evaluations=2 * n * f,
        frequency_exponentials=f,
        embedding_lookup_scalars=2 * n * h,
        raw_history_scalars=3 * n,
    )


@dataclass(frozen=True)
class PairedKVOnlyCost:
    """Replacement ledger for the audited paired KV-only construction."""

    exact_all_flops: int
    old_paired_kv_only_flops: int
    old_two_raw_and_initial_flops: int
    new_two_matrix_free_initial_flops: int

    @property
    def total_flops(self) -> int:
        return (
            self.old_paired_kv_only_flops
            - self.old_two_raw_and_initial_flops
            + self.new_two_matrix_free_initial_flops
        )

    @property
    def exact_all_fraction(self) -> float:
        return self.total_flops / self.exact_all_flops


def medium_paired_kv_only_cost() -> PairedKVOnlyCost:
    """Recompute the frozen Medium paired-D KV-only numerator."""

    per_arm = matrix_free_input_cost(
        history_length=1024,
        hidden_size=192,
        temporal_num_freqs=16,
        rank=4,
        oversample=4,
        power_iterations=1,
    )
    return PairedKVOnlyCost(
        exact_all_flops=4_771_282_944,
        old_paired_kv_only_flops=1_041_218_120,
        old_two_raw_and_initial_flops=2 * (88_489_984 + 12_951_382),
        new_two_matrix_free_initial_flops=2 * per_arm.flops,
    )
