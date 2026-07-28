import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hstu_kvcache.migration import JaggedMigratedKVBatch
from hstu_kvcache.models import HSTU, HSTUConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
SCRIPT = SCRIPT_DIR / "run_cohortkv_stage4_9_rollout_boundary.py"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "run_cohortkv_stage4_9_rollout_boundary",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def frozen_args(**overrides):
    values = {
        "candidate": "token_debt_total10",
        "device": None,
        "prepared_data": MODULE.PREPARED_PATH,
        "training_result": MODULE.TRAINING_PATH,
        "checkpoint_dir": MODULE.CHECKPOINT_DIR,
        "compiler_result": MODULE.COMPILER_OUTPUT,
        "runtime_dir": MODULE.RUNTIME_DIR,
        "baseline": MODULE.stage48.BASELINE_PATH,
        "output_dir": MODULE.OUTPUT_DIR,
        "seed": 0,
        "batch_size": MODULE.BATCH_SIZE,
        "smoke_test": True,
        "runtime_smoke_test": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def cost_summary():
    return MODULE.rollout_cost_summary(
        {
            "retained_crop_ms": 1.0,
            "retained_transform_ms": 2.0,
            "scheduled_exact_retained_ms": 3.0,
            "missing_exact_retained_ms": 5.0,
            "retained_materialization_ms": 4.0,
        },
        20.0,
        {
            "mixed_target_delta_append_ms": 100.0,
            "mixed_target_prefix_assembly_ms": 5.0,
            "mixed_latest_append_ms": 10.0,
            "mixed_final_split_ms": 1.0,
            "exact_target_delta_append_ms": 110.0,
            "exact_target_prefix_assembly_ms": 6.0,
            "exact_latest_append_ms": 11.0,
            "mixed_natural_target_prefix_build_ms": 12.0,
            "exact_natural_target_prefix_build_ms": 13.0,
        },
        150.0,
    )


def test_cost_summary_excludes_target_append_from_primary_ratio() -> None:
    result = cost_summary()

    assert result["timed_retained_repair"]["mixed_u_ms"] == 15.0
    assert result["timed_retained_repair"]["paired_exact_e_ms"] == 20.0
    assert result["timed_retained_repair"]["primary_u_over_e"] == 0.75
    assert result["target_append_excluded_from_u_and_e"]
    assert result["diagnostic_final_state_ready"]["mixed_ms"] == 143.0
    assert not result["diagnostic_final_state_ready"]["is_migration_speedup"]


def test_minimal_runner_requires_exactly_one_smoke_mode() -> None:
    with pytest.raises(ValueError, match="exactly one smoke mode"):
        MODULE.validate_args(
            frozen_args(smoke_test=False, runtime_smoke_test=False)
        )
    with pytest.raises(ValueError, match="exactly one smoke mode"):
        MODULE.validate_args(
            frozen_args(smoke_test=True, runtime_smoke_test=True)
        )
    MODULE.validate_args(frozen_args())


def test_runtime_smoke_requires_explicit_available_cuda(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="requires --device"):
        MODULE.validate_args(
            frozen_args(
                smoke_test=False,
                runtime_smoke_test=True,
                device=None,
            )
        )
    with pytest.raises(ValueError, match="explicit CUDA index"):
        MODULE.validate_args(
            frozen_args(
                smoke_test=False,
                runtime_smoke_test=True,
                device="cuda:1",
            )
        )


def test_static_smoke_is_non_scientific_and_does_not_reuse_denominator() -> None:
    result = MODULE.smoke_payload(frozen_args())

    assert result["status"] == "smoke_passed"
    assert not result["scientific_result"]
    assert not result["formal_result_written"]
    assert result["baseline"]["used_for_provenance_only"]
    assert not result["baseline"]["old_exact_denominator_reused"]
    assert result["checks"]["formal_execution_disabled"]


def test_latest_only_path_builds_from_empty_prefix(monkeypatch) -> None:
    model = HSTU(
        HSTUConfig(
            num_items=16,
            num_behaviors=4,
            hidden_size=8,
            num_layers=2,
            num_heads=1,
            head_dim=8,
            max_seq_len=4,
            input_dropout=0.0,
        )
    )
    model.eval()
    history = SimpleNamespace(
        item_ids=[3],
        behaviors=[2],
        time_deltas=[0.0],
    )
    window = SimpleNamespace(
        records={7: SimpleNamespace(history=history)},
    )
    monkeypatch.setattr(
        MODULE,
        "timed_cuda",
        lambda function, device: (function(), 0.0),
    )

    cache, hidden, elapsed = MODULE._append_fresh_latest(
        model,
        (4,),
        {4: {"user_id": 7}},
        window,
        1,
        model.cfg,
        torch.device("cpu"),
        torch.float16,
    )

    assert cache.record_ids == (4,)
    assert cache.lengths.tolist() == [1]
    assert cache.k.dtype == torch.float16
    assert hidden.shape == (1, model.cfg.hidden_size)
    assert elapsed == 0.0
    one_hidden, one_cache = model(
        torch.tensor([[3]], dtype=torch.long),
        torch.tensor([[2]], dtype=torch.long),
        torch.tensor([[0.0]], dtype=torch.float32),
        return_kv=True,
        lengths=torch.tensor([1]),
    )
    assert one_cache is not None
    assert torch.allclose(
        cache.k.float(),
        one_cache.k[:, 0].float(),
        atol=1e-3,
        rtol=1e-3,
    )
    assert torch.allclose(
        cache.v.float(),
        one_cache.v[:, 0].float(),
        atol=1e-3,
        rtol=1e-3,
    )
    assert torch.allclose(
        hidden,
        model.last_hidden(one_hidden, torch.tensor([1])),
        atol=1e-6,
        rtol=1e-6,
    )


def test_final_merge_restores_record_and_hidden_order(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE.stage48,
        "timed_cuda",
        lambda function, device: (function(), 0.0),
    )

    def source(record_id, value):
        lengths = torch.tensor([1], dtype=torch.long)
        return JaggedMigratedKVBatch(
            record_ids=(record_id,),
            migration_anchor_version="theta1",
            served_kv_target="theta1",
            k=torch.full((2, 1, 3), value, dtype=torch.float16),
            v=torch.full((2, 1, 3), -value, dtype=torch.float16),
            lengths=lengths,
            offsets=torch.tensor([0, 1], dtype=torch.long),
        )

    second = source(2, 2.0)
    first = source(1, 1.0)
    merged, hidden, elapsed = MODULE._merge_final_outputs(
        (1, 2),
        (
            (second, torch.full((1, 4), 20.0)),
            (first, torch.full((1, 4), 10.0)),
        ),
        1,
        torch.device("cpu"),
    )

    assert merged.record_ids == (1, 2)
    assert torch.equal(merged.k[:, 0], first.k[:, 0])
    assert torch.equal(merged.k[:, 1], second.k[:, 0])
    assert torch.equal(hidden[0], torch.full((4,), 10.0))
    assert torch.equal(hidden[1], torch.full((4,), 20.0))
    assert elapsed == 0.0
