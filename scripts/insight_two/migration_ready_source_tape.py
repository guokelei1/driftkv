"""Static audit for an exact-Parent-anchored migration source tape.

The object audited here is deliberately not a ``KV_P -> KV_C`` mapper.  A
Parent cache producer may retain a cut through the original Transformer
execution graph, so a later release can advance only the finite defect

    D[l + 1] = D[l] + update_C(X_P[l] + D[l]) - update_P(X_P[l]).

The algebraic recurrence has an exact full-rank/native-attention limit.  This
module also records why that limit is not an executable Medium design under
EvoKV's 20% cap: the five non-terminal native attention responses alone are
already more than 40% of Exact-All.  It performs no user-data experiment and
does not admit a migration action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@torch.inference_mode()
def finite_linear_defect(
    parent_source: torch.Tensor,
    source_defect: torch.Tensor,
    parent_weight: torch.Tensor,
    current_weight: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a finite linear-layer defect without a Current target pair.

    PyTorch linear convention is used.  The identity includes the mixed
    ``source_defect @ (current_weight-parent_weight).T`` term through the
    multiplication by ``current_weight``; it is not a first-order JVP.
    """

    if parent_source.shape != source_defect.shape or parent_source.ndim < 2:
        raise ValueError("source and defect must have the same rank>=2 shape")
    hidden = int(parent_source.shape[-1])
    if parent_weight.shape != (hidden, hidden):
        raise ValueError("parent weight must be square at the source width")
    if current_weight.shape != parent_weight.shape:
        raise ValueError("release weights differ in shape")
    return source_defect @ current_weight.transpose(0, 1) + parent_source @ (
        current_weight - parent_weight
    ).transpose(0, 1)


@torch.inference_mode()
def finite_residual_defect(
    incoming_defect: torch.Tensor,
    parent_update: torch.Tensor,
    current_update: torch.Tensor,
) -> torch.Tensor:
    """Advance the exact finite defect across one residual boundary."""

    if incoming_defect.shape != parent_update.shape:
        raise ValueError("incoming defect and Parent update shapes differ")
    if current_update.shape != parent_update.shape:
        raise ValueError("release update shapes differ")
    return incoming_defect + current_update - parent_update


@torch.inference_mode()
def recover_post_attention_output(
    parent_update: torch.Tensor,
    parent_gate: torch.Tensor,
) -> torch.Tensor:
    """Recover ``O_P`` from ``U_P=O_P*silu(G_P)`` when the gate is nonzero.

    This is the reason a 21-field tape is only an algebraic generic-case cut.
    A fixed-layout, numerically stable tape stores ``O_P`` explicitly and has
    26 fields.  Silently clamping a zero or near-zero gate would change the
    finite-release computation and is therefore rejected here.
    """

    if parent_update.shape != parent_gate.shape:
        raise ValueError("Parent update and gate shapes differ")
    activated = torch.nn.functional.silu(parent_gate)
    if bool((activated == 0).any()):
        raise ValueError("zero Parent gate makes post-attention output unidentifiable")
    return parent_update / activated


@dataclass(frozen=True)
class MigrationReadyTapeAudit:
    """Medium source-state, I/O, and native-attention cost verdict."""

    context: int
    hidden: int
    layers: int
    active_layers: int
    exact_all_flops: int
    parent_kv_scalars: int
    parent_kv_bytes_fp32: int
    algebraic_tape_fields: int
    algebraic_tape_scalars: int
    algebraic_tape_bytes_fp32: int
    stable_tape_fields: int
    stable_tape_scalars: int
    stable_tape_bytes_fp32: int
    stable_tape_over_parent_kv: float
    stable_total_source_read_bytes_fp32: int
    stable_extra_storage_30000_gib: float
    stable_total_source_30000_gib: float
    causal_pairs_per_layer: int
    native_current_attention_floor_flops: int
    native_current_attention_floor_over_exact: float
    single_current_rank8_control_flops: int
    single_current_rank8_control_over_exact: float
    single_current_rank8_control_recovery: float
    within_twenty_percent_before_other_work: bool
    verdict: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


def medium_migration_ready_tape_audit() -> MigrationReadyTapeAudit:
    """Return the strict Medium no-go audit.

    Tape fields are counted in full ``N x H`` coordinates:

    * generic algebraic cut: ``X_0``; five Parent residual updates; and five
      each of historical Q, gate preactivation, and pre-output attention;
    * stable fixed-layout cut: the above plus five post-output attention
      tensors, avoiding division through a zero/near-zero SiLU gate.

    Existing Parent K/V remain necessary.  The FLOP floor grants every
    projection, normalization, defect compression, output projection, gate,
    sidecar build, and I/O for free.  It charges only one native Current QK
    contraction and one native weighted-V contraction in each of the five
    non-terminal blocks.  Consequently exceeding 20% is decisive.
    """

    context = 1024
    hidden = 192
    layers = 6
    active_layers = layers - 1
    exact_all = 4_771_282_944
    fp32 = 4
    parent_kv_scalars = 2 * layers * context * hidden

    algebraic_fields = 1 + active_layers + 3 * active_layers
    stable_fields = algebraic_fields + active_layers
    algebraic_scalars = algebraic_fields * context * hidden
    stable_scalars = stable_fields * context * hidden

    pairs = context * (context + 1) // 2
    # One QK and one attention-weighted V, multiply-add=2.
    native_attention = active_layers * 4 * pairs * hidden
    control = 853_836_992
    return MigrationReadyTapeAudit(
        context=context,
        hidden=hidden,
        layers=layers,
        active_layers=active_layers,
        exact_all_flops=exact_all,
        parent_kv_scalars=parent_kv_scalars,
        parent_kv_bytes_fp32=fp32 * parent_kv_scalars,
        algebraic_tape_fields=algebraic_fields,
        algebraic_tape_scalars=algebraic_scalars,
        algebraic_tape_bytes_fp32=fp32 * algebraic_scalars,
        stable_tape_fields=stable_fields,
        stable_tape_scalars=stable_scalars,
        stable_tape_bytes_fp32=fp32 * stable_scalars,
        stable_tape_over_parent_kv=stable_scalars / parent_kv_scalars,
        stable_total_source_read_bytes_fp32=fp32 * (parent_kv_scalars + stable_scalars),
        stable_extra_storage_30000_gib=fp32 * stable_scalars * 30_000 / 2**30,
        stable_total_source_30000_gib=(
            fp32 * (parent_kv_scalars + stable_scalars) * 30_000 / 2**30
        ),
        causal_pairs_per_layer=pairs,
        native_current_attention_floor_flops=native_attention,
        native_current_attention_floor_over_exact=native_attention / exact_all,
        single_current_rank8_control_flops=control,
        single_current_rank8_control_over_exact=control / exact_all,
        single_current_rank8_control_recovery=0.937,
        within_twenty_percent_before_other_work=native_attention / exact_all <= 0.20,
        verdict=(
            "NO_GO: exact/native finite-release attention alone exceeds 20%; "
            "a sub-20 path requires an architecture-specific affine-region "
            "approximation and is not the audited exact invariant"
        ),
    )

