from __future__ import annotations

import pytest
import torch
from insight_two.migration_ready_source_tape import (
    finite_linear_defect,
    finite_residual_defect,
    medium_migration_ready_tape_audit,
    recover_post_attention_output,
)


def test_finite_linear_defect_includes_all_finite_terms() -> None:
    generator = torch.Generator().manual_seed(17)
    parent = torch.randn(2, 7, 8, generator=generator, dtype=torch.float64)
    defect = torch.randn(2, 7, 8, generator=generator, dtype=torch.float64)
    parent_weight = torch.randn(8, 8, generator=generator, dtype=torch.float64)
    current_weight = torch.randn(8, 8, generator=generator, dtype=torch.float64)
    expected = (parent + defect) @ current_weight.T - parent @ parent_weight.T
    observed = finite_linear_defect(
        parent, defect, parent_weight, current_weight
    )
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)


def test_residual_defect_is_an_exact_finite_difference() -> None:
    generator = torch.Generator().manual_seed(31)
    parent = torch.randn(3, 5, generator=generator)
    defect = torch.randn(3, 5, generator=generator)
    parent_update = torch.randn(3, 5, generator=generator)
    current_update = torch.randn(3, 5, generator=generator)
    parent_next = parent + parent_update
    current_next = parent + defect + current_update
    torch.testing.assert_close(
        finite_residual_defect(defect, parent_update, current_update),
        current_next - parent_next,
    )


def test_update_and_nonzero_gate_recover_post_attention_output() -> None:
    generator = torch.Generator().manual_seed(47)
    output = torch.randn(4, 9, generator=generator, dtype=torch.float64)
    gate = torch.randn(4, 9, generator=generator, dtype=torch.float64) + 0.25
    update = output * torch.nn.functional.silu(gate)
    torch.testing.assert_close(
        recover_post_attention_output(update, gate),
        output,
        rtol=1e-12,
        atol=1e-12,
    )


def test_zero_gate_requires_explicit_post_attention_coordinate() -> None:
    with pytest.raises(ValueError, match="unidentifiable"):
        recover_post_attention_output(torch.zeros(1, 2), torch.zeros(1, 2))


def test_medium_stable_tape_and_native_attention_are_decisive_no_go() -> None:
    audit = medium_migration_ready_tape_audit()
    assert audit.parent_kv_scalars == 2_359_296
    assert audit.parent_kv_bytes_fp32 == 9 * 2**20
    assert audit.algebraic_tape_fields == 21
    assert audit.algebraic_tape_scalars == 4_128_768
    assert audit.algebraic_tape_bytes_fp32 == 15.75 * 2**20
    assert audit.stable_tape_fields == 26
    assert audit.stable_tape_scalars == 5_111_808
    assert audit.stable_tape_bytes_fp32 == 19.5 * 2**20
    assert audit.stable_tape_over_parent_kv == pytest.approx(13 / 6)
    assert audit.stable_total_source_read_bytes_fp32 == 28.5 * 2**20
    assert audit.native_current_attention_floor_flops == 2_015_232_000
    assert audit.native_current_attention_floor_over_exact == pytest.approx(
        0.4223669029174221
    )
    assert audit.single_current_rank8_control_over_exact == pytest.approx(
        0.17895333435920416
    )
    assert not audit.within_twenty_percent_before_other_work
    assert audit.verdict.startswith("NO_GO")
