"""Source-certified residual closure for one Parent-to-Current release.

This module tests a constructor that is deliberately stronger than subtracting
two independently reduced trajectories at the serving reader.  The exact
Parent cache is a known solution of the Parent Transformer.  At every
non-terminal block we use it to evaluate a small set of *exact source block
responses*.  Those responses certify the residual of the reduced Parent
execution, and an interpolatory lift closes the same residual inside both the
Parent and Current reduced trajectories before the next layer::

    e_l[I_l] = U_l^P(X_l^P)[I_l] - U_l^P(Xhat_l^P)[I_l]
    E_l       = Q_l (Q_l[I_l])^{-1} e_l[I_l]
    Xhat^P_{l+1} = Xhat^P_l + Uhat^P_l + E_l
    Xhat^C_{l+1} = Xhat^C_l + Uhat^C_l + E_l.

``Q_l`` is the orthonormal token trial space carried by the reduced Parent
block and ``I_l`` is its deterministic DEIM interpolation set.  Thus the
source residual is zero at the selected test functionals after closure.  The
Current target cache, labels, candidates, and future events are not inputs.

The scientific hypothesis is not that DEIM or low-rank factors are new.  It
is that a materialized source Transformer trajectory can act as an internal
nonlinear residual certificate for a finite-release reduced execution.  The
paired native-response path is the deletion control: it has the same two
rank-4 arms and reader, but never consults exact Parent responses while it is
forming upper-layer state.

This is a fixed one-user preflight implementation, not an admitted Design.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch

from hstu_kvcache.models import HSTUKVCache
from insight_two.defect_first_replay import (
    _factorized_block_update,
    _factorized_kv_only,
)
from insight_two.mode_space_replay import (
    FactorizedCacheLayer,
    FactorizedReplay,
    PairedReleaseReplay,
    TokenModeFactors,
    _compress_token_modes,
    factorized_rmsnorm,
)
from insight_two.parent_anchored_delta_scan import joint_kv_decoder

MEDIUM_EXACT_ALL_FLOPS = 4_771_282_944
MEDIUM_PAIRED_NATIVE_FLOPS = 872_238_088


@dataclass(frozen=True)
class SourceResidualCertificate:
    """One layer's deterministic source residual interpolation record."""

    positions: torch.Tensor
    trial_basis: torch.Tensor
    sampled_residual: torch.Tensor
    lifted_residual: torch.Tensor
    interpolation_max_abs_error: float
    interpolation_condition: float


@dataclass(frozen=True)
class SourceResidualClosedReplay:
    """Paired reduced trajectories after internal exact-source closure."""

    paired: PairedReleaseReplay
    certificates: tuple[SourceResidualCertificate, ...]

    def __post_init__(self) -> None:
        layers = len(self.paired.parent.layers)
        if len(self.paired.current.layers) != layers:
            raise ValueError("source-closed replay arm depths differ")
        if len(self.certificates) != max(layers - 1, 0):
            raise ValueError("one certificate is required per non-terminal layer")


@dataclass(frozen=True)
class SourceResidualClosureCost:
    """Strict Medium per-user arithmetic ledger for the fixed r4 route."""

    base_paired_native_flops: int
    active_layers: int
    history_length: int
    hidden_size: int
    heads: int
    trial_rank: int
    basis_qr_flops: int
    deim_selection_and_solve_flops: int
    exact_source_decode_flops: int
    exact_source_query_gate_output_flops: int
    exact_source_attention_flops: int
    source_activation_pointwise_flops: int
    residual_lift_flops: int
    residual_state_add_flops: int
    exact_all_flops: int

    @property
    def certificate_flops(self) -> int:
        return (
            self.basis_qr_flops
            + self.deim_selection_and_solve_flops
            + self.exact_source_decode_flops
            + self.exact_source_query_gate_output_flops
            + self.exact_source_attention_flops
            + self.source_activation_pointwise_flops
            + self.residual_lift_flops
            + self.residual_state_add_flops
        )

    @property
    def total_constructor_flops(self) -> int:
        return self.base_paired_native_flops + self.certificate_flops

    @property
    def constructor_fraction(self) -> float:
        return self.total_constructor_flops / self.exact_all_flops

    @property
    def within_twenty_percent(self) -> bool:
        return self.constructor_fraction <= 0.20

    def to_dict(self) -> dict[str, int | float | bool]:
        payload = asdict(self)
        payload.update(
            {
                "certificate_flops": self.certificate_flops,
                "total_constructor_flops": self.total_constructor_flops,
                "constructor_fraction": self.constructor_fraction,
                "within_twenty_percent": self.within_twenty_percent,
            }
        )
        return payload


@dataclass(frozen=True)
class SourceDefectClosureCost:
    """Strict cost for certifying only the finite release-defect equation."""

    base_paired_native_flops: int
    trial_and_interpolation_flops: int
    exact_parent_normalized_decode_flops: int
    exact_parent_row_update_flops: int
    anchored_current_row_projection_flops: int
    logical_current_exact_parent_read_flops: int
    logical_current_factor_delta_read_flops: int
    normalized_defect_row_flops: int
    two_update_pointwise_flops: int
    defect_lift_and_add_flops: int
    exact_all_flops: int

    @property
    def certificate_flops(self) -> int:
        return (
            self.trial_and_interpolation_flops
            + self.exact_parent_normalized_decode_flops
            + self.exact_parent_row_update_flops
            + self.anchored_current_row_projection_flops
            + self.logical_current_exact_parent_read_flops
            + self.logical_current_factor_delta_read_flops
            + self.normalized_defect_row_flops
            + self.two_update_pointwise_flops
            + self.defect_lift_and_add_flops
        )

    @property
    def total_constructor_flops(self) -> int:
        return self.base_paired_native_flops + self.certificate_flops

    @property
    def constructor_fraction(self) -> float:
        return self.total_constructor_flops / self.exact_all_flops

    @property
    def within_twenty_percent(self) -> bool:
        return self.constructor_fraction <= 0.20

    def to_dict(self) -> dict[str, int | float | bool]:
        payload = asdict(self)
        payload.update(
            {
                "certificate_flops": self.certificate_flops,
                "total_constructor_flops": self.total_constructor_flops,
                "constructor_fraction": self.constructor_fraction,
                "within_twenty_percent": self.within_twenty_percent,
            }
        )
        return payload


def medium_source_residual_closure_cost() -> SourceResidualClosureCost:
    """Return a conservative r4 certificate ledger for Medium.

    The exact source attention term assumes every selected causal query reads
    all ``N`` Parent rows.  Decoder construction is model-global and excluded;
    applying the joint-K/V decoder per user is fully charged.  The existing
    paired-native ledger remains the base, including matrix-free inputs,
    terminal K/V specialization, and both reduced trajectories.
    """

    n = 1024
    h = 192
    heads = 6
    rank = 4
    active = 5

    # Orthonormalise the carried trial factor once per active layer.
    qr_per_layer = 2 * n * rank * rank - (2 * rank**3) // 3
    # Conservative DEIM bookkeeping plus an r x r solve with H right-hand
    # sides.  This intentionally overcounts the tiny greedy pivot loop.
    deim_per_layer = 2 * n * rank * rank + 2 * rank**3 + 2 * rank * rank * h
    # [K,V] (2H) times the 2H x H joint decoder at r source rows.
    decode_per_layer = 4 * rank * h * h
    # Parent Q, gate, and output projections at the r certified rows.
    q_gate_output_per_layer = 6 * rank * h * h
    # QK and weighted-V for r causal rows, each granted the full N prefix.
    attention_per_layer = 4 * rank * n * h
    # ELU activation, SiLU gate, Hadamard product, and sampled subtraction.
    pointwise_per_layer = rank * n * heads + 4 * rank * h
    # Q @ coefficient plus the interpolation residual solve contribution not
    # already charged above.
    lift_per_layer = 2 * n * rank * h
    # Add one common closure tensor to both reduced next states.
    state_add_per_layer = 2 * n * h
    return SourceResidualClosureCost(
        base_paired_native_flops=MEDIUM_PAIRED_NATIVE_FLOPS,
        active_layers=active,
        history_length=n,
        hidden_size=h,
        heads=heads,
        trial_rank=rank,
        basis_qr_flops=active * qr_per_layer,
        deim_selection_and_solve_flops=active * deim_per_layer,
        exact_source_decode_flops=active * decode_per_layer,
        exact_source_query_gate_output_flops=active * q_gate_output_per_layer,
        exact_source_attention_flops=active * attention_per_layer,
        source_activation_pointwise_flops=active * pointwise_per_layer,
        residual_lift_flops=active * lift_per_layer,
        residual_state_add_flops=active * state_add_per_layer,
        exact_all_flops=MEDIUM_EXACT_ALL_FLOPS,
    )


def medium_source_defect_closure_cost() -> SourceDefectClosureCost:
    """Return the conservative fixed-r4 finite-defect certificate ledger.

    Relative to :func:`medium_source_residual_closure_cost`, this evaluates a
    second Current-version block response at the certified rows.  Its logical
    prefix is read exactly as ``Z_parent + (Zhat_current-Zhat_parent)``: the
    exact Parent QK/weighted-V terms and both factorized delta terms are all
    charged.  The certificate still fits below 20% without counting on causal
    rows being early.
    """

    n = 1024
    h = 192
    heads = 6
    rank = 4
    tests = rank
    active = 5
    qr = 2 * n * rank * rank - (2 * rank**3) // 3
    deim = 2 * n * rank * rank + 2 * rank**3 + 2 * rank * rank * h
    decode = 4 * tests * h * h
    # q, gate, out for the exact Parent update.
    parent_projections = 6 * tests * h * h
    parent_attention = 4 * tests * n * h
    # q, gate, out for the anchored Current normalized row.
    current_projections = 6 * tests * h * h
    # Exact Parent component of logical Current K/V: QK and weighted V.
    logical_parent_read = 4 * tests * n * h
    # For each of approximate Current and Parent: q Ck^T, result L^T,
    # weights L, and result Cv.  This preserves the activation on the summed
    # logical logits rather than subtracting two independent responses.
    one_factor_delta_read = 4 * tests * rank * h + 4 * tests * heads * rank * n
    logical_factor_delta = 2 * one_factor_delta_read
    normalized_rows = 2 * (2 * tests * rank * h)
    pointwise = 2 * (tests * n * heads + 4 * tests * h)
    lift_and_add = 2 * n * rank * h + n * h
    return SourceDefectClosureCost(
        base_paired_native_flops=MEDIUM_PAIRED_NATIVE_FLOPS,
        trial_and_interpolation_flops=active * (qr + deim),
        exact_parent_normalized_decode_flops=active * decode,
        exact_parent_row_update_flops=active
        * (parent_projections + parent_attention),
        anchored_current_row_projection_flops=active * current_projections,
        logical_current_exact_parent_read_flops=active * logical_parent_read,
        logical_current_factor_delta_read_flops=active * logical_factor_delta,
        normalized_defect_row_flops=active * normalized_rows,
        two_update_pointwise_flops=active * pointwise,
        defect_lift_and_add_flops=active * lift_and_add,
        exact_all_flops=MEDIUM_EXACT_ALL_FLOPS,
    )


def _validate_exact_parent(
    parent_model,
    current_model,
    exact_parent: HSTUKVCache,
) -> tuple[int, int, int]:
    if parent_model.training or current_model.training:
        raise ValueError("source residual closure requires eval-mode models")
    if parent_model.cfg.block_variant != "legacy":
        raise ValueError("source residual closure currently covers legacy blocks")
    if current_model.cfg.block_variant != "legacy":
        raise ValueError("source residual closure currently covers legacy blocks")
    if len(parent_model.blocks) != len(current_model.blocks):
        raise ValueError("release model depths differ")
    if exact_parent.k.ndim != 4 or exact_parent.k.shape != exact_parent.v.shape:
        raise ValueError("exact Parent must contain matching [L,B,N,H] K/V")
    layers, batch, length, width = (int(value) for value in exact_parent.k.shape)
    if layers != len(parent_model.blocks) or batch != 1:
        raise ValueError("exact Parent lineage differs from source model")
    if exact_parent.seq_len != length:
        raise ValueError("exact Parent seq_len differs")
    if width != parent_model.blocks[0].attn.inner:
        raise ValueError("exact Parent cache width differs")
    if parent_model.cfg.hidden_size != current_model.cfg.hidden_size:
        raise ValueError("release hidden widths differ")
    return layers, length, parent_model.cfg.hidden_size


def _validate_initial_factors(
    parent_initial: TokenModeFactors,
    current_initial: TokenModeFactors,
    *,
    rank: int,
    length: int,
    hidden: int,
) -> None:
    if parent_initial.rank != rank or current_initial.rank != rank:
        raise ValueError("initial factor ranks differ from frozen rank")
    expected = (1, length)
    if parent_initial.left.shape[:2] != expected:
        raise ValueError("Parent initial token layout differs")
    if current_initial.left.shape[:2] != expected:
        raise ValueError("Current initial token layout differs")
    if parent_initial.right.shape[2] != hidden:
        raise ValueError("Parent initial hidden width differs")
    if current_initial.right.shape[2] != hidden:
        raise ValueError("Current initial hidden width differs")


@torch.inference_mode()
def deim_interpolation_rows(trial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an orthonormal trial basis and deterministic DEIM row set.

    ``trial`` has shape ``[1,N,r]``.  The routine is model/data deterministic
    and observes neither Current Exact state nor recommendation candidates.
    """

    if trial.ndim != 3 or trial.shape[0] != 1:
        raise ValueError("trial must have shape [1,N,r]")
    if trial.shape[2] < 1 or trial.shape[1] < trial.shape[2]:
        raise ValueError("trial rank must be in [1,N]")
    basis, _ = torch.linalg.qr(trial.float(), mode="reduced")
    basis = basis[:, :, : trial.shape[2]].to(dtype=trial.dtype)
    matrix = basis[0]
    pivots: list[int] = [int(torch.argmax(torch.abs(matrix[:, 0])).item())]
    for column in range(1, matrix.shape[1]):
        selected = matrix[pivots, :column]
        rhs = matrix[pivots, column]
        coefficients = torch.linalg.solve(selected, rhs)
        residual = matrix[:, column] - matrix[:, :column] @ coefficients
        order = torch.argsort(torch.abs(residual), descending=True, stable=True)
        pivot = next(int(value) for value in order.tolist() if int(value) not in pivots)
        pivots.append(pivot)
    return basis, torch.tensor(pivots, device=trial.device, dtype=torch.long)


@torch.inference_mode()
def interpolate_sampled_residual(
    basis: torch.Tensor,
    positions: torch.Tensor,
    sampled_residual: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Lift H-wide residual samples while reproducing every sample exactly."""

    if basis.ndim != 3 or basis.shape[0] != 1:
        raise ValueError("basis must have shape [1,N,r]")
    rank = basis.shape[2]
    if positions.shape != (rank,):
        raise ValueError("one interpolation position is required per basis column")
    if sampled_residual.shape[:2] != (rank, 1):
        raise ValueError("sampled residual must have shape [r,1,H]")
    square = basis[0].index_select(0, positions)
    coefficients = torch.linalg.solve(square.float(), sampled_residual[:, 0].float())
    lifted = (basis.float() @ coefficients).to(dtype=sampled_residual.dtype)
    condition = float(torch.linalg.cond(square.float()))
    return lifted, condition


@torch.inference_mode()
def exact_parent_normalized_rows(
    block,
    exact_parent_key: torch.Tensor,
    exact_parent_value: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Decode exact Parent RMS-normalized rows from joint persistent K/V."""

    attention = block.attn
    if exact_parent_key.ndim != 3 or exact_parent_key.shape != exact_parent_value.shape:
        raise ValueError("exact Parent layer must contain matching [1,N,H] K/V")
    if exact_parent_key.shape[0] != 1:
        raise ValueError("exact source row evaluation supports one user")
    length = exact_parent_key.shape[1]
    positions = positions.to(device=exact_parent_key.device, dtype=torch.long)
    if positions.ndim != 1 or positions.numel() < 1:
        raise ValueError("positions must be a nonempty vector")
    if bool((positions < 0).any()) or bool((positions >= length).any()):
        raise ValueError("source response position lies outside the history")

    decoder = joint_kv_decoder(
        attention.k_proj.weight,
        attention.v_proj.weight,
    )
    selected_k = exact_parent_key[0].index_select(0, positions)
    selected_v = exact_parent_value[0].index_select(0, positions)
    return (
        torch.cat((selected_k.double(), selected_v.double()), dim=-1) @ decoder
    ).to(dtype=exact_parent_key.dtype)[:, None, :]


def _causal_dense_heads(
    attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Read selected causal rows from one dense K/V layer."""

    length = key.shape[1]
    heads = attention.num_heads
    head_dim = attention.head_dim
    keys = key.view(1, length, heads, head_dim).transpose(1, 2)
    values = value.view(1, length, heads, head_dim).transpose(1, 2)
    keys = keys.expand(positions.numel(), -1, -1, -1)
    values = values.expand(positions.numel(), -1, -1, -1)
    logits = (query @ keys.transpose(-2, -1)) * attention.scale
    key_positions = torch.arange(length, device=positions.device)
    keep = key_positions[None, :] <= positions[:, None]
    weights = attention._activate(logits) * keep[:, None, None, :]
    return attention.attn_dropout(weights) @ values


def _update_from_normalized_and_heads(
    block,
    normalized: torch.Tensor,
    heads: torch.Tensor,
) -> torch.Tensor:
    attention_output = block.attn._finish(heads)
    if block.gating == "silu_gate":
        gate = torch.nn.functional.silu(block.gate_proj(normalized))
        return attention_output * gate
    if block.gating == "glu":
        gate = torch.sigmoid(block.gate_proj(normalized))
        return attention_output * gate
    if block.gating == "none":
        return attention_output
    raise ValueError("source residual closure does not cover the FFN variant")


@torch.inference_mode()
def exact_parent_block_update_rows(
    block,
    exact_parent_key: torch.Tensor,
    exact_parent_value: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Evaluate exact Parent block updates at selected causal source rows.

    Joint K/V injectively recovers the RMS-normalized Parent state at the
    selected rows.  Each query then reads the exact Parent prefix through the
    source model's native activation, output projection, and gate.  The dense
    Parent hidden trajectory is never reconstructed.
    """

    attention = block.attn
    if block.block_variant != "legacy" or attention.position_bias is not None:
        raise ValueError("exact source row evaluation requires legacy/no-bias attention")
    positions = positions.to(device=exact_parent_key.device, dtype=torch.long)
    normalized = exact_parent_normalized_rows(
        block, exact_parent_key, exact_parent_value, positions
    )
    query, _, _ = attention._project(normalized)
    heads = _causal_dense_heads(
        attention, query, exact_parent_key, exact_parent_value, positions
    )
    return _update_from_normalized_and_heads(block, normalized, heads)


def _factorized_logits(
    attention,
    query: torch.Tensor,
    layer: FactorizedCacheLayer,
) -> torch.Tensor:
    batch, _, rank = layer.left.shape
    if batch != 1:
        raise ValueError("source defect closure supports one factorized user")
    key_core = layer.key_core.view(
        1, rank, attention.num_heads, attention.head_dim
    ).permute(0, 2, 1, 3)
    return (query @ key_core.transpose(-2, -1)) @ layer.left.transpose(
        1, 2
    ).unsqueeze(1)


def _factorized_weighted_values(
    attention,
    weights: torch.Tensor,
    layer: FactorizedCacheLayer,
) -> torch.Tensor:
    batch, _, rank = layer.left.shape
    value_core = layer.value_core.view(
        batch, rank, attention.num_heads, attention.head_dim
    ).permute(0, 2, 1, 3)
    mode_weights = weights @ layer.left.unsqueeze(1)
    return mode_weights @ value_core


@torch.inference_mode()
def logical_current_block_update_rows(
    current_block,
    anchored_current_normalized: torch.Tensor,
    exact_parent_key: torch.Tensor,
    exact_parent_value: torch.Tensor,
    approximate_parent: FactorizedCacheLayer,
    approximate_current: FactorizedCacheLayer,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Read ``Parent exact + approximate Current - approximate Parent`` K/V.

    The three K tensors are added *before* native activation and the three V
    tensors are read by the resulting common weights.  This preserves the
    finite K-by-V interaction of the logical Current endpoint and is not the
    paired-native response subtraction used by the serving deletion control.
    """

    attention = current_block.attn
    if current_block.block_variant != "legacy" or attention.position_bias is not None:
        raise ValueError("logical Current row evaluation requires legacy/no-bias attention")
    if approximate_parent.rank != approximate_current.rank:
        raise ValueError("logical Current factor ranks differ")
    positions = positions.to(device=exact_parent_key.device, dtype=torch.long)
    query, _, _ = attention._project(anchored_current_normalized)
    length = exact_parent_key.shape[1]
    heads = attention.num_heads
    head_dim = attention.head_dim
    exact_keys = exact_parent_key.view(1, length, heads, head_dim).transpose(1, 2)
    exact_keys = exact_keys.expand(positions.numel(), -1, -1, -1)
    logits = (
        query @ exact_keys.transpose(-2, -1)
        + _factorized_logits(attention, query, approximate_current)
        - _factorized_logits(attention, query, approximate_parent)
    ) * attention.scale
    key_positions = torch.arange(length, device=positions.device)
    keep = key_positions[None, :] <= positions[:, None]
    weights = attention.attn_dropout(
        attention._activate(logits) * keep[:, None, None, :]
    )
    exact_values = exact_parent_value.view(
        1, length, heads, head_dim
    ).transpose(1, 2)
    exact_values = exact_values.expand(positions.numel(), -1, -1, -1)
    response = (
        weights @ exact_values
        + _factorized_weighted_values(attention, weights, approximate_current)
        - _factorized_weighted_values(attention, weights, approximate_parent)
    )
    return _update_from_normalized_and_heads(
        current_block, anchored_current_normalized, response
    )


@torch.inference_mode()
def source_residual_closed_replay_from_initial_factors(
    parent_model,
    current_model,
    exact_parent: HSTUKVCache,
    parent_initial: TokenModeFactors,
    current_initial: TokenModeFactors,
    *,
    rank: int = 4,
    compression: Literal["exact_svd", "fixed_range_finder"] = "fixed_range_finder",
    sketch_oversample: int = 4,
    sketch_power_iterations: int = 1,
    sketch_seed: int = 17,
) -> SourceResidualClosedReplay:
    """Advance matched reduced trajectories with exact-source residual closure."""

    layers, length, hidden = _validate_exact_parent(
        parent_model, current_model, exact_parent
    )
    _validate_initial_factors(
        parent_initial,
        current_initial,
        rank=rank,
        length=length,
        hidden=hidden,
    )
    parent_factors = parent_initial
    current_factors = current_initial
    parent_inputs: list[TokenModeFactors] = []
    current_inputs: list[TokenModeFactors] = []
    parent_layers: list[FactorizedCacheLayer] = []
    current_layers: list[FactorizedCacheLayer] = []
    parent_materialized: list[tuple[torch.Tensor, torch.Tensor]] = []
    current_materialized: list[tuple[torch.Tensor, torch.Tensor]] = []
    defects: list[torch.Tensor] = []
    certificates: list[SourceResidualCertificate] = []

    for layer, (parent_block, current_block) in enumerate(
        zip(parent_model.blocks, current_model.blocks, strict=True)
    ):
        parent_inputs.append(parent_factors)
        current_inputs.append(current_factors)
        terminal = layer + 1 == layers
        if terminal:
            parent_cache_layer = _factorized_kv_only(parent_block, parent_factors)
            current_cache_layer = _factorized_kv_only(current_block, current_factors)
        else:
            parent_update, parent_cache_layer = _factorized_block_update(
                parent_block, parent_factors
            )
            current_update, current_cache_layer = _factorized_block_update(
                current_block, current_factors
            )
            trial_basis, positions = deim_interpolation_rows(
                parent_cache_layer.left
            )
            exact_samples = exact_parent_block_update_rows(
                parent_block,
                exact_parent.k[layer],
                exact_parent.v[layer],
                positions,
            )
            approximate_samples = parent_update[0].index_select(
                0, positions
            )[:, None, :]
            sampled_residual = exact_samples - approximate_samples
            lifted, condition = interpolate_sampled_residual(
                trial_basis, positions, sampled_residual
            )
            reproduced = lifted[0].index_select(0, positions)
            interpolation_error = float(
                torch.max(torch.abs(reproduced - sampled_residual[:, 0]))
            )
            certificates.append(
                SourceResidualCertificate(
                    positions=positions.detach(),
                    trial_basis=trial_basis.detach(),
                    sampled_residual=sampled_residual.detach(),
                    lifted_residual=lifted.detach(),
                    interpolation_max_abs_error=interpolation_error,
                    interpolation_condition=condition,
                )
            )

            parent_state = (
                parent_factors.materialize() + parent_update + lifted
            )
            current_state = (
                current_factors.materialize() + current_update + lifted
            )
            defects.append(current_state - parent_state)
            shared = {
                "rank": rank,
                "compression": compression,
                "oversample": sketch_oversample,
                "power_iterations": sketch_power_iterations,
                "seed": sketch_seed + layer + 1,
            }
            parent_factors = _compress_token_modes(parent_state, **shared)
            current_factors = _compress_token_modes(current_state, **shared)

        parent_layers.append(parent_cache_layer)
        current_layers.append(current_cache_layer)
        parent_materialized.append(parent_cache_layer.materialize())
        current_materialized.append(current_cache_layer.materialize())

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
    return SourceResidualClosedReplay(
        paired=PairedReleaseReplay(
            parent=parent_replay,
            current=current_replay,
            # The terminal post-block defect has no downstream consumer; use
            # the exact cache-space release defect as an observable terminal
            # record while preserving PairedReleaseReplay's one-per-layer API.
            post_block_defects=tuple(
                defects
                + [current_replay.cache.k[-1] - parent_replay.cache.k[-1]]
            ),
        ),
        certificates=tuple(certificates),
    )


@torch.inference_mode()
def source_defect_closed_replay_from_initial_factors(
    parent_model,
    current_model,
    exact_parent: HSTUKVCache,
    parent_initial: TokenModeFactors,
    current_initial: TokenModeFactors,
    *,
    rank: int = 4,
    compression: Literal["exact_svd", "fixed_range_finder"] = "fixed_range_finder",
    sketch_oversample: int = 4,
    sketch_power_iterations: int = 1,
    sketch_seed: int = 17,
) -> SourceResidualClosedReplay:
    """Close only the finite release-defect equation at source test rows.

    Unlike :func:`source_residual_closed_replay_from_initial_factors`, this
    routine never inserts an absolute Parent approximation residual into the
    trajectories.  It anchors a Current normalized row at the exact Parent
    normalized state plus the paired normalized-state defect, reads a logical
    Current K/V endpoint, and compares its Current-minus-Parent update with the
    ordinary paired update difference.  Only that version-essential residual
    is lifted, and only the Current arm receives it.
    """

    layers, length, hidden = _validate_exact_parent(
        parent_model, current_model, exact_parent
    )
    _validate_initial_factors(
        parent_initial,
        current_initial,
        rank=rank,
        length=length,
        hidden=hidden,
    )
    parent_factors = parent_initial
    current_factors = current_initial
    parent_inputs: list[TokenModeFactors] = []
    current_inputs: list[TokenModeFactors] = []
    parent_layers: list[FactorizedCacheLayer] = []
    current_layers: list[FactorizedCacheLayer] = []
    parent_materialized: list[tuple[torch.Tensor, torch.Tensor]] = []
    current_materialized: list[tuple[torch.Tensor, torch.Tensor]] = []
    defects: list[torch.Tensor] = []
    certificates: list[SourceResidualCertificate] = []

    for layer, (parent_block, current_block) in enumerate(
        zip(parent_model.blocks, current_model.blocks, strict=True)
    ):
        parent_inputs.append(parent_factors)
        current_inputs.append(current_factors)
        terminal = layer + 1 == layers
        if terminal:
            parent_cache_layer = _factorized_kv_only(parent_block, parent_factors)
            current_cache_layer = _factorized_kv_only(current_block, current_factors)
        else:
            parent_update, parent_cache_layer = _factorized_block_update(
                parent_block, parent_factors
            )
            current_update, current_cache_layer = _factorized_block_update(
                current_block, current_factors
            )
            trial_basis, positions = deim_interpolation_rows(
                parent_cache_layer.left
            )
            exact_parent_normalized = exact_parent_normalized_rows(
                parent_block,
                exact_parent.k[layer],
                exact_parent.v[layer],
                positions,
            )
            approximate_parent_normalized = factorized_rmsnorm(
                parent_block.norm, parent_factors
            ).materialize()[0].index_select(0, positions)[:, None, :]
            approximate_current_normalized = factorized_rmsnorm(
                current_block.norm, current_factors
            ).materialize()[0].index_select(0, positions)[:, None, :]
            anchored_current_normalized = (
                exact_parent_normalized
                + approximate_current_normalized
                - approximate_parent_normalized
            )
            exact_parent_update = exact_parent_block_update_rows(
                parent_block,
                exact_parent.k[layer],
                exact_parent.v[layer],
                positions,
            )
            anchored_current_update = logical_current_block_update_rows(
                current_block,
                anchored_current_normalized,
                exact_parent.k[layer],
                exact_parent.v[layer],
                parent_cache_layer,
                current_cache_layer,
                positions,
            )
            approximate_parent_samples = parent_update[0].index_select(
                0, positions
            )[:, None, :]
            approximate_current_samples = current_update[0].index_select(
                0, positions
            )[:, None, :]
            anchored_defect_update = anchored_current_update - exact_parent_update
            approximate_defect_update = (
                approximate_current_samples - approximate_parent_samples
            )
            sampled_residual = anchored_defect_update - approximate_defect_update
            lifted, condition = interpolate_sampled_residual(
                trial_basis, positions, sampled_residual
            )
            reproduced = lifted[0].index_select(0, positions)
            interpolation_error = float(
                torch.max(torch.abs(reproduced - sampled_residual[:, 0]))
            )
            certificates.append(
                SourceResidualCertificate(
                    positions=positions.detach(),
                    trial_basis=trial_basis.detach(),
                    sampled_residual=sampled_residual.detach(),
                    lifted_residual=lifted.detach(),
                    interpolation_max_abs_error=interpolation_error,
                    interpolation_condition=condition,
                )
            )

            parent_state = parent_factors.materialize() + parent_update
            current_state = (
                current_factors.materialize() + current_update + lifted
            )
            defects.append(current_state - parent_state)
            shared = {
                "rank": rank,
                "compression": compression,
                "oversample": sketch_oversample,
                "power_iterations": sketch_power_iterations,
                "seed": sketch_seed + layer + 1,
            }
            parent_factors = _compress_token_modes(parent_state, **shared)
            current_factors = _compress_token_modes(current_state, **shared)

        parent_layers.append(parent_cache_layer)
        current_layers.append(current_cache_layer)
        parent_materialized.append(parent_cache_layer.materialize())
        current_materialized.append(current_cache_layer.materialize())

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
    return SourceResidualClosedReplay(
        paired=PairedReleaseReplay(
            parent=parent_replay,
            current=current_replay,
            post_block_defects=tuple(
                defects
                + [current_replay.cache.k[-1] - parent_replay.cache.k[-1]]
            ),
        ),
        certificates=tuple(certificates),
    )
