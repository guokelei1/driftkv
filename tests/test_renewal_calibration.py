import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import torch

from hstu_kvcache.migration import (
    JaggedMigratedKVBatch,
    fit_renewal_calibrated_direct_oldkv_program,
)
from hstu_kvcache.models import HSTU, HSTUConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
SCRIPT = SCRIPT_DIR / "run_cohortkv_stage4_10_renewal_calibrated_smoke.py"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "run_cohortkv_stage4_10_renewal_calibrated_smoke",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def jagged(
    joined: torch.Tensor,
    version: int,
) -> JaggedMigratedKVBatch:
    width = joined.shape[-1] // 2
    return JaggedMigratedKVBatch(
        record_ids=(3, 7),
        migration_anchor_version=f"theta{version}",
        served_kv_target=f"theta{version}",
        k=joined[..., :width].contiguous(),
        v=joined[..., width:].contiguous(),
        lengths=torch.tensor([6, 6]),
        offsets=torch.tensor([0, 6, 12]),
    )


def test_direct_kv_residual_ridge_fits_actual_pairs() -> None:
    generator = torch.Generator().manual_seed(7)
    old = torch.randn(2, 12, 8, generator=generator)
    target_weight = (
        torch.eye(8).expand(2, -1, -1)
        + 0.02 * torch.randn(2, 8, 8, generator=generator)
    )
    target_bias = 0.01 * torch.randn(2, 8, generator=generator)
    fresh = torch.bmm(old, target_weight) + target_bias[:, None]

    program, metrics = fit_renewal_calibrated_direct_oldkv_program(
        jagged(old, 0),
        jagged(fresh, 1),
        source_version="theta0",
        target_version="theta1",
        mode="direct_kv_residual_ridge",
        ridge=1e-6,
        max_fit_tokens=12,
    )
    predicted = (
        torch.bmm(old, program.weights.float())
        + program.biases.float()[:, None]
    )

    assert torch.allclose(predicted, fresh, atol=2e-3, rtol=2e-3)
    assert program.weights.dtype == torch.float16
    assert metrics.paired_records == 2
    assert metrics.paired_tokens == 12
    assert metrics.sampled_tokens == 12
    assert not metrics.labels_used
    assert not metrics.semantic_gate_used


def test_inverse_norm_ridge_compiles_to_direct_old_kv() -> None:
    cfg = HSTUConfig(
        num_items=32,
        num_behaviors=4,
        hidden_size=8,
        num_layers=2,
        num_heads=1,
        head_dim=8,
        max_seq_len=16,
        input_dropout=0.0,
    )
    source = HSTU(cfg)
    target = HSTU(cfg)
    generator = torch.Generator().manual_seed(11)
    norm = torch.randn(2, 12, 8, generator=generator)

    def project(model: HSTU) -> torch.Tensor:
        return torch.stack(
            [
                torch.cat(
                    (
                        block.attn.k_proj(norm[layer]),
                        block.attn.v_proj(norm[layer]),
                    ),
                    dim=-1,
                )
                for layer, block in enumerate(model.blocks)
            ]
        )

    old = project(source)
    fresh = project(target)
    program, metrics = fit_renewal_calibrated_direct_oldkv_program(
        jagged(old, 0),
        jagged(fresh, 1),
        source_version="theta0",
        target_version="theta1",
        mode="inverse_norm_ridge",
        ridge=1e-6,
        max_fit_tokens=12,
        source_model=source,
        target_model=target,
    )
    predicted = (
        torch.bmm(old, program.weights.float())
        + program.biases.float()[:, None]
    )

    assert torch.allclose(predicted, fresh, atol=2e-3, rtol=2e-3)
    assert metrics.source_width == cfg.hidden_size
    assert metrics.target_width == 2 * cfg.num_heads * cfg.head_dim


def test_calibration_rejects_misaligned_pairs() -> None:
    values = torch.randn(2, 12, 8)
    fresh = jagged(values, 1)
    mismatched = JaggedMigratedKVBatch(
        record_ids=(3, 8),
        migration_anchor_version="theta0",
        served_kv_target="theta0",
        k=fresh.k.clone(),
        v=fresh.v.clone(),
        lengths=fresh.lengths.clone(),
        offsets=fresh.offsets.clone(),
    )

    with pytest.raises(ValueError, match="pairs differ"):
        fit_renewal_calibrated_direct_oldkv_program(
            mismatched,
            fresh,
            source_version="theta0",
            target_version="theta1",
            mode="direct_kv_residual_ridge",
        )


def test_program_build_is_counted_once_per_edge() -> None:
    groups = [
        {
            "scheduled_exact_retained_ms": {
                "samples_ms": [2.0],
                "median_ms": 2.0,
            }
        },
        {
            "scheduled_exact_retained_ms": {
                "samples_ms": [3.0],
                "median_ms": 3.0,
            }
        },
    ]
    result = RUNNER._aggregate_components(
        groups,
        {
            "program_fit_and_compile_ms": {
                "samples_ms": [7.0],
                "median_ms": 7.0,
            }
        },
        1,
    )

    assert result["samples_ms"] == [12.0]
    assert result["sum_of_component_medians_ms"] == 12.0
    assert (
        result["components"]["program_fit_and_compile_ms"]["median_ms"]
        == 7.0
    )


def test_runner_uses_scheduled_pairs_without_old_serialized_program() -> None:
    prepare_source = inspect.getsource(RUNNER._prepare_calibration)
    edge_source = inspect.getsource(RUNNER._run_edge)
    main_source = inspect.getsource(RUNNER.main)

    assert "selection.scheduled_exact_ids" in prepare_source
    assert "_crop_actual_retained" in prepare_source
    assert "fit_renewal_calibrated_direct_oldkv_program" in prepare_source
    assert "_load_program" not in prepare_source
    assert "_load_program" not in edge_source
    assert "semantic_gate" not in prepare_source
    assert '"serialized_program_loaded": False' in main_source
    assert '"extra_exact_fit_records": 0' in main_source
