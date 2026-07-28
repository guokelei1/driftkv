import ast
import importlib.util
import inspect
import json
import os
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hstu_kvcache.migration import JaggedMigratedKVBatch
from hstu_kvcache.migration.stage45_oldkv import DirectOldKVProgram

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
SCRIPT = SCRIPT_DIR / "run_cohortkv_stage5_full_cow.py"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "run_cohortkv_stage5_full_cow",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def frozen_args(**overrides):
    values = {
        "devices": None,
        "candidate": "token_debt_total10",
        "seed": 0,
        "batch_size": MODULE.BATCH_SIZE,
        "smoke_test": True,
        "result_schema": MODULE.RESULT_SCHEMA,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_static_cli_and_formal_cuda_contract(monkeypatch) -> None:
    assert MODULE.validate_args(frozen_args()) == ()
    with pytest.raises(ValueError, match="does not accept --devices"):
        MODULE.validate_args(frozen_args(devices=("cuda:0", "cuda:1")))
    with pytest.raises(ValueError, match="requires --candidate"):
        MODULE.validate_args(frozen_args(candidate=None))

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    formal = frozen_args(
        smoke_test=False,
        devices=("cuda:0", "cuda:1"),
    )
    assert MODULE.validate_args(formal) == (
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    )
    with pytest.raises(ValueError, match="distinct"):
        MODULE.validate_args(
            frozen_args(
                smoke_test=False,
                devices=("cuda:0", "cuda:0"),
            )
        )


def test_formal_candidate_requires_explicit_or_frozen_selection() -> None:
    summary = {}
    with pytest.raises(ValueError, match="pass --candidate"):
        MODULE.resolve_formal_candidate(summary, None)
    assert (
        MODULE.resolve_formal_candidate(summary, "token_debt_total10")
        == "token_debt_total10"
    )
    selected = {"selected_candidate": "staggered_renewal_h12"}
    assert (
        MODULE.resolve_formal_candidate(selected, None)
        == "staggered_renewal_h12"
    )
    with pytest.raises(ValueError, match="differs"):
        MODULE.resolve_formal_candidate(
            selected,
            "token_debt_total10",
        )


def test_formal_confirmation_gate_rejects_non_scientific(tmp_path) -> None:
    result_path = tmp_path / "candidate.json"
    result = {
        "protocol": MODULE.stage49_formal.PROTOCOL,
        "status": "complete",
        "scientific_result": True,
        "candidate_name": "token_debt_total10",
        "checks": {"all_passed": True},
        "implementation": MODULE.stage49_formal.implementation_snapshot(),
        "steps": [
            {"source_version": 0, "target_version": 1},
            *[
                {"source_version": value, "target_version": value + 1}
                for value in range(1, 11)
            ],
        ],
        "measurement_boundary": {
            "recursive_state": "previous_actual_post_append_mixed_cache",
            "old_exact_denominator_reused": False,
        },
    }
    result_path.write_text(json.dumps(result))
    summary_path = tmp_path / "summary.json"
    summary = {
        "protocol": MODULE.stage49_formal.PROTOCOL,
        "status": "complete",
        "scientific_result": False,
        "checks": {"all_passed": True},
        "results": [
            {
                "candidate_name": "token_debt_total10",
                "path": str(result_path),
                "sha256": MODULE.sha256(result_path),
            }
        ],
    }
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="formal summary is invalid"):
        MODULE.load_formal_confirmation(
            summary_path,
            "token_debt_total10",
        )

    summary["scientific_result"] = True
    summary_path.write_text(json.dumps(summary))
    candidate, loaded_summary, loaded_result = (
        MODULE.load_formal_confirmation(
            summary_path,
            "token_debt_total10",
        )
    )
    assert candidate == "token_debt_total10"
    assert loaded_summary["scientific_result"] is True
    assert loaded_result["scientific_result"] is True

    result["implementation"]["formal_runner"]["sha256"] = "0" * 64
    result_path.write_text(json.dumps(result))
    summary["results"][0]["sha256"] = MODULE.sha256(result_path)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="candidate confirmation is invalid"):
        MODULE.load_formal_confirmation(
            summary_path,
            "token_debt_total10",
        )


def test_two_gpu_placement_is_deterministic_and_balanced() -> None:
    groups = (
        ({"record_id": 0}, {"record_id": 1}),
        ({"record_id": 2}, {"record_id": 3}),
        ({"record_id": 4},),
    )
    old = {0: 10, 1: 10, 2: 6, 3: 6, 4: 2}
    final = {0: 10, 1: 10, 2: 6, 3: 6, 4: 2}

    assert MODULE.place_groups_two_gpu(groups, old, final) == (0, 1, 1)


def test_requests_cover_migrate_scheduled_and_natural_exact() -> None:
    plans = {
        0: SimpleNamespace(
            final_tokens=8,
            retained_tokens=5,
            target_prefix_tokens=7,
            timed_retained_rebuild=False,
        ),
        1: SimpleNamespace(
            final_tokens=7,
            retained_tokens=4,
            target_prefix_tokens=6,
            timed_retained_rebuild=False,
        ),
        2: SimpleNamespace(
            final_tokens=6,
            retained_tokens=0,
            target_prefix_tokens=5,
            timed_retained_rebuild=False,
        ),
        3: SimpleNamespace(
            final_tokens=1,
            retained_tokens=0,
            target_prefix_tokens=0,
            timed_retained_rebuild=False,
        ),
    }
    selection = SimpleNamespace(
        migrate_ids=(0,),
        scheduled_exact_ids=(1,),
        natural_exact_ids=(2, 3),
    )

    requests = MODULE.build_requests(plans, selection, {0, 1})

    assert [value.cohort_id for value in requests] == [
        "migration",
        "scheduled-exact",
        "natural-prefix-exact",
        "natural-short-exact",
    ]
    assert [value.retained_tokens for value in requests] == [5, 4, 5, 0]
    assert requests[0].requested_action == "migrate"
    assert all(value.requested_action == "exact" for value in requests[1:])


def test_old_extent_is_loaded_once_for_multiple_records() -> None:
    lengths = torch.tensor([2, 2], dtype=torch.long)
    batch = JaggedMigratedKVBatch(
        record_ids=(1, 2),
        migration_anchor_version="theta0",
        served_kv_target="theta0",
        k=torch.arange(16, dtype=torch.float32).reshape(1, 4, 4),
        v=torch.arange(16, dtype=torch.float32).reshape(1, 4, 4),
        lengths=lengths,
        offsets=torch.tensor([0, 2, 4], dtype=torch.long),
    )

    class Destination:
        def __init__(self):
            self.calls = 0

        def load_extent(self, target_version, extent_id):
            assert target_version == "theta0"
            assert extent_id == "old-0"
            self.calls += 1
            return batch

    destination = Destination()
    manifest = SimpleNamespace(
        extents=(
            SimpleNamespace(
                extent_id="old-0",
                record_ids=(1, 2),
            ),
        )
    )

    loaded = MODULE._load_old_records(
        destination,
        manifest,
        (2, 1),
    )

    assert destination.calls == 1
    assert loaded.record_ids == (2, 1)
    assert tuple(int(value) for value in loaded.lengths) == (2, 2)


def test_direct_program_digest_changes_with_semantic_perturbation() -> None:
    program = DirectOldKVProgram(
        source_version="theta0",
        target_version="theta1",
        weights=torch.eye(4).reshape(1, 4, 4),
        biases=torch.zeros(1, 4),
    )
    perturbed = DirectOldKVProgram(
        source_version="theta0",
        target_version="theta1",
        weights=program.weights.clone(),
        biases=program.biases + 1024.0,
    )

    assert MODULE._direct_program_sha256(program) != (
        MODULE._direct_program_sha256(perturbed)
    )


def test_run_case_wiring_matches_live_helper_signatures() -> None:
    targets = {
        "build_cohorts": MODULE.build_cohorts,
        "_build_natural_prefix_once": (
            MODULE.stage49_formal._build_natural_prefix_once
        ),
        "_append_fresh_latest_once": (
            MODULE.stage49_formal._append_fresh_latest_once
        ),
        "_append_latest_once": MODULE.stage49_formal._append_latest_once,
        "_append_delta_once": MODULE.stage49_formal._append_delta_once,
    }
    expected_counts = {
        "build_cohorts": 1,
        "_build_natural_prefix_once": 1,
        "_append_fresh_latest_once": 1,
        "_append_latest_once": 2,
        "_append_delta_once": 1,
    }
    observed_counts = {value: 0 for value in expected_counts}
    tree = ast.parse(textwrap.dedent(inspect.getsource(MODULE.run_case)))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        if name not in targets:
            continue
        observed_counts[name] += 1
        kwargs = {
            value.arg: object()
            for value in node.keywords
            if value.arg is not None
        }
        inspect.signature(targets[name]).bind(
            *([object()] * len(node.args)),
            **kwargs,
        )
    assert observed_counts == expected_counts


def test_closure_validation_calls_jsonschema_and_cross_field(
    monkeypatch,
    tmp_path,
) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "properties": {
                    "stage5_closure": {"type": "object"}
                }
            }
        )
    )
    calls = []

    class Validator:
        @staticmethod
        def validate(instance, schema):
            calls.append(("jsonschema", instance, schema))

    monkeypatch.setattr(
        MODULE.freeze,
        "validate_stage5_closure_semantics",
        lambda value: calls.append(("semantic", value)),
    )
    closure = {"protocol": "test"}

    result = MODULE.validate_closure_artifact(
        closure,
        schema_path,
        jsonschema_module=Validator,
    )

    assert [value[0] for value in calls] == ["jsonschema", "semantic"]
    assert result["jsonschema_validated"] is True
    assert result["cross_field_validated"] is True
    assert result["validation_scope"] == "stage5_closure_subschema"


def test_atomic_save_preserves_old_output_on_write_failure(
    monkeypatch,
    tmp_path,
) -> None:
    output = tmp_path / "formal.json"
    original = b'{"status":"old"}\n'
    output.write_bytes(original)

    def fail_dump(*args, **kwargs):
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(MODULE.json, "dump", fail_dump)
    with pytest.raises(RuntimeError, match="injected write failure"):
        MODULE._atomic_save_json({"status": "new"}, output)

    assert output.read_bytes() == original
    assert list(tmp_path.glob(".formal.json.*.tmp")) == []


def test_atomic_save_replaces_complete_json(tmp_path) -> None:
    output = tmp_path / "formal.json"
    output.write_text('{"status":"old"}')
    value = {"status": "complete", "records": 682}

    MODULE._atomic_save_json(value, output)

    assert json.loads(output.read_text()) == value
    assert list(tmp_path.glob(".formal.json.*.tmp")) == []
    assert os.stat(output).st_size > 0


def test_static_smoke_is_non_scientific() -> None:
    payload = MODULE.smoke_payload(frozen_args())

    assert payload["scientific_result"] is False
    assert payload["formal_result_written"] is False
    assert all(payload["checks"].values())
