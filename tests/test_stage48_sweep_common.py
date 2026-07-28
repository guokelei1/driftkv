import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hstu_kvcache.migration import JaggedMigratedKVBatch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
SCRIPT = SCRIPT_DIR / "cohortkv_stage4_8_sweep_common.py"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "cohortkv_stage4_8_sweep_common",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def frozen_args(**overrides):
    values = {
        "baseline": MODULE.BASELINE_PATH,
        "prepared_data": (
            "data/processed/"
            "kuairand_long_context_4plus12_exploration_v1.npz"
        ),
        "training_result": (
            "results/motivation_scale/"
            "long_context_4plus12_training_exploration_seed0.json"
        ),
        "checkpoint_dir": (
            "checkpoints/kuairand_long_context_4plus12_exploration/seed0"
        ),
        "compiler_result": (
            "results/system/cohortkv_single_config_full_chain_v1/"
            "stage4_7_organic_adjacent_compiler_seed0.json"
        ),
        "runtime_dir": (
            "checkpoints/kuairand_long_context_4plus12_exploration/seed0/"
            "single_config_v1/stage4_7_organic_runtime"
        ),
        "seed": 0,
        "batch_size": 4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def cache(
    record_ids: tuple[int, ...],
    values: tuple[float, ...],
    anchor: str,
) -> JaggedMigratedKVBatch:
    lengths = torch.ones(len(record_ids), dtype=torch.long)
    offsets = torch.arange(len(record_ids) + 1, dtype=torch.long)
    tensor = torch.tensor(values, dtype=torch.float32).view(1, -1, 1)
    return JaggedMigratedKVBatch(
        record_ids=record_ids,
        migration_anchor_version=anchor,
        served_kv_target="theta1",
        k=tensor,
        v=tensor + 100,
        lengths=lengths,
        offsets=offsets,
    )


def test_stage48_grids_have_four_predeclared_points() -> None:
    assert set(MODULE.SCHEME_GRIDS) == {
        "staggered_renewal",
        "token_debt",
        "aoi_maxweight",
        "model_time_renewal",
    }
    for scheme in MODULE.SCHEME_GRIDS:
        specs = MODULE.variant_specs(scheme)
        assert len(specs) == 4
        assert tuple(value.grid_index for value in specs) == (0, 1, 2, 3)
        assert len({value.label for value in specs}) == 4


def test_target_assembly_preserves_requested_record_order() -> None:
    first = cache((1, 3), (11, 13), "theta0")
    second = cache((0, 2), (10, 12), "theta1")
    result = MODULE._assemble_target_sources(
        (0, 1, 2, 3),
        (first, second),
        1,
    )

    assert result.record_ids == (0, 1, 2, 3)
    assert result.migration_anchor_version == "theta1"
    assert result.served_kv_target == "theta1"
    assert result.k.flatten().tolist() == [10, 11, 12, 13]
    assert result.v.flatten().tolist() == [110, 111, 112, 113]


def test_single_target_source_is_relabelled_without_copy() -> None:
    source = cache((0, 1), (10, 11), "theta0")
    result, elapsed = MODULE._target_prefix(
        (0, 1),
        (source,),
        1,
        torch.device("cpu"),
    )

    assert elapsed == 0
    assert result.k.data_ptr() == source.k.data_ptr()
    assert result.v.data_ptr() == source.v.data_ptr()
    assert result.migration_anchor_version == "theta1"


def test_task_summary_uses_record_weighting_and_external_exact() -> None:
    first = MODULE._mixed_task_summary(
        [
            {"catalog_auc": 0.6, "ndcg@100": 0.2, "hit@100": 0.4},
            {"catalog_auc": 0.8, "ndcg@100": 0.4, "hit@100": 0.6},
        ],
        {
            "records": 2,
            "catalog_auc": 0.5,
            "ndcg_at_100": 0.25,
            "hit_at_100": 0.5,
        },
    )
    second = MODULE._mixed_task_summary(
        [{"catalog_auc": 0.9, "ndcg@100": 0.8, "hit@100": 1.0}],
        {
            "records": 1,
            "catalog_auc": 0.75,
            "ndcg_at_100": 0.5,
            "hit_at_100": 0.5,
        },
    )
    weighted = MODULE._record_weighted_task(
        [{"task_metrics": first}, {"task_metrics": second}]
    )

    assert weighted["records"] == 3
    assert weighted["mixed"]["catalog_auc"] == 2.3 / 3
    assert weighted["all_exact_external"]["catalog_auc"] == 1.75 / 3
    assert weighted["mixed_over_exact"]["hit_at_100"] == 2.0 / 1.5


def test_smoke_scheduler_covers_every_record_for_every_point() -> None:
    records = MODULE._smoke_records()
    for scheme in MODULE.SCHEME_GRIDS:
        for spec in MODULE.variant_specs(scheme):
            selection = MODULE._select_actions(
                records,
                spec,
                1,
                1.0,
                None,
            )
            selected = (
                set(selection.scheduled_exact_ids)
                | set(selection.natural_exact_ids)
                | set(selection.migrate_ids)
            )
            assert selected == {value.record_id for value in records}
            assert set(selection.natural_exact_ids) == {0}


def test_complete_result_is_bound_to_device_inputs_and_implementation(
    tmp_path: Path,
) -> None:
    args = frozen_args()
    baseline = MODULE.load_exact_baseline(args.baseline)
    spec = MODULE.variant_specs("token_debt")[0]
    result = {
        "protocol": MODULE.PROTOCOL,
        "status": "complete",
        "scheme": spec.scheme,
        "variant": spec.to_dict(),
        "repository_commit": MODULE._repository_commit(),
        "implementation": MODULE.implementation_snapshot(spec.scheme),
        "configuration": {
            "dataset": baseline["configuration"]["dataset"],
            "split": baseline["configuration"]["split"],
            "seed": baseline["configuration"]["training_seed"],
            "batch_size": baseline["configuration"]["batch_size"],
            "device": "cuda:0",
            "device_name": baseline["configuration"]["device_class"],
            "records": baseline["configuration"]["records"],
            "model": baseline["configuration"]["model"],
        },
        "exact_baseline": {
            "sha256": MODULE.EXPECTED_BASELINE_SHA256,
            "protocol": MODULE.BASELINE_PROTOCOL,
            "source_result": baseline["source_artifacts"]["stage4_7_chain"],
        },
        "inputs": MODULE._expected_result_inputs(baseline),
        "checks": {"all_passed": True},
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))

    assert MODULE.result_complete(path, spec, args, "cuda:0")
    assert not MODULE.result_complete(path, spec, args, "cuda:1")

    result["implementation"]["organic_schedulers"]["sha256"] = "0" * 64
    path.write_text(json.dumps(result))
    assert not MODULE.result_complete(path, spec, args, "cuda:0")


def test_exact_baseline_rejects_content_drift(tmp_path: Path) -> None:
    baseline = json.loads((ROOT / MODULE.BASELINE_PATH).read_text())
    baseline["endpoint_exact_task"][0]["catalog_auc"] = 0.0
    path = tmp_path / "altered_baseline.json"
    path.write_text(json.dumps(baseline))

    with pytest.raises(ValueError, match="SHA256"):
        MODULE.load_exact_baseline(path)


def test_launcher_rejects_implicit_or_aliased_cuda_devices(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    args = SimpleNamespace(
        scheme="token_debt",
        smoke_test=False,
        runtime_smoke_test=False,
        seed=0,
        batch_size=4,
        grid_index=None,
        device=None,
        devices=("cuda", "cuda:0", "cuda:1", "cuda:2"),
    )

    with pytest.raises(ValueError, match="explicit CUDA indices"):
        MODULE.validate_args(args)
