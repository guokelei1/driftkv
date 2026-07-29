from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from hstu_kvcache.migration.cohort_jagged import JaggedMigratedKVBatch
from hstu_kvcache.migration.design2_dev_wave import (
    D2DevWaveLineage,
    assemble_d2_dev_jagged,
    build_d2_dev_lineages,
    close_d2_dev_wave,
    d2_dev_record_payload_sha256,
)
from hstu_kvcache.migration.design2_plan import (
    D2ActionRecord,
    canonical_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LAUNCHER = _load(
    "cohortkv_design2_dev_c0_launcher_test",
    ROOT / "scripts/launch_cohortkv_design2_dev_c0.py",
)


def _action(
    record_id: int,
    requested_action: str,
    requested_reason: str,
) -> D2ActionRecord:
    natural = requested_reason == "natural_exact"
    retained = 0 if natural else 2
    delta = 3 if natural else 1
    return D2ActionRecord(
        record_id=record_id,
        prepared_user_id=record_id + 1,
        requested_action=requested_action,
        requested_reason=requested_reason,
        old_tokens=2,
        retained_start=2 if natural else 0,
        retained_tokens=retained,
        delta_start=retained,
        delta_tokens=delta,
        target_prefix_tokens=3,
        latest_tokens=1,
        final_tokens=4,
        last_exact_version=None if natural else "theta1",
        migration_depth=0,
        previous_cache_expected=not natural,
        previous_cache_present=not natural,
        old_history_sha256=None if natural else "a" * 64,
        target_history_sha256="b" * 64,
        retained_identity_sha256="c" * 64,
        delta_identity_sha256="d" * 64,
        target_prefix_identity_sha256="e" * 64,
    )


def _fragment(
    record_ids: tuple[int, ...],
    lengths: tuple[int, ...],
    offset: int,
) -> JaggedMigratedKVBatch:
    tokens = sum(lengths)
    values = torch.arange(
        offset,
        offset + 2 * tokens,
        dtype=torch.float16,
    ).reshape(1, tokens, 2)
    length_tensor = torch.tensor(lengths, dtype=torch.long)
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), length_tensor.cumsum(0))
    )
    return JaggedMigratedKVBatch(
        record_ids=record_ids,
        migration_anchor_version="theta2",
        served_kv_target="theta2",
        k=values,
        v=(values + 1).contiguous(),
        lengths=length_tensor,
        offsets=offsets,
    )


def test_build_lineages_preserves_route_semantics() -> None:
    actions = (
        _action(7, "exact", "natural_exact"),
        _action(3, "compiled", "migrate"),
        _action(5, "exact", "scheduled_exact"),
    )
    lineages = build_d2_dev_lineages(
        actions,
        {3: 0, 5: 1, 7: 0},
        2,
        "theta1",
        "theta2",
    )
    assert tuple(value.record_id for value in lineages) == (3, 5, 7)
    assert tuple(value.route for value in lineages) == (
        "compiled",
        "scheduled_exact",
        "natural_exact",
    )
    assert lineages[0].phase_tokens["source_old_kv_fixture"] == 2
    assert lineages[1].phase_tokens["scheduled_exact_retained"] == 2
    assert lineages[2].phase_tokens["delta_append"] == 0
    assert len({value.lineage_sha256 for value in lineages}) == 3


def test_assemble_and_close_wave_in_requested_order() -> None:
    first = _fragment((5,), (4,), 0)
    second = _fragment((3, 7), (4, 4), 20)
    assembled = assemble_d2_dev_jagged(
        (3, 5, 7),
        (first, None, second),
        "theta2",
        "theta2",
    )
    assert assembled is not None
    assert assembled.record_ids == (3, 5, 7)
    assert assembled.lengths.tolist() == [4, 4, 4]
    actions = (
        _action(3, "compiled", "migrate"),
        _action(5, "exact", "scheduled_exact"),
        _action(7, "exact", "natural_exact"),
    )
    lineages = build_d2_dev_lineages(
        actions,
        {3: 0, 5: 0, 7: 0},
        1,
        "theta1",
        "theta2",
    )
    closure = close_d2_dev_wave(
        assembled,
        lineages,
        0,
        1,
        "theta1",
        "theta2",
    )
    assert closure.passed
    assert closure.record_ids == (3, 5, 7)
    assert closure.token_count == 12
    assert len(closure.record_payload_sha256) == 3
    assert d2_dev_record_payload_sha256(assembled, 3) == (
        closure.record_payload_sha256[0]
    )


def test_assemble_rejects_duplicate_or_missing_coverage() -> None:
    fragment = _fragment((3,), (4,), 0)
    with pytest.raises(ValueError, match="appears twice"):
        assemble_d2_dev_jagged(
            (3,),
            (fragment, fragment),
            "theta2",
            "theta2",
        )
    with pytest.raises(ValueError, match="close coverage"):
        assemble_d2_dev_jagged(
            (3, 5),
            (fragment,),
            "theta2",
            "theta2",
        )


def test_empty_rank_closes_without_synthetic_fragment() -> None:
    closure = close_d2_dev_wave(
        None,
        (),
        2,
        3,
        "theta1",
        "theta2",
    )
    assert closure.passed
    assert closure.record_ids == ()
    assert closure.token_count == 0


def test_lineage_rejects_natural_delta_append() -> None:
    with pytest.raises(ValueError, match="lineage is invalid"):
        D2DevWaveLineage(
            record_id=3,
            owner_rank=0,
            world_size=1,
            route="natural_exact",
            source_version="theta1",
            target_version="theta2",
            old_tokens=2,
            retained_start=2,
            retained_tokens=0,
            delta_start=0,
            delta_tokens=2,
            target_prefix_tokens=3,
            latest_tokens=1,
            final_tokens=4,
            source_history_sha256=None,
            target_history_sha256="b" * 64,
        )


def test_c0_launcher_rejects_non_roundtrip_content_hash(
    tmp_path: Path,
) -> None:
    artifact = {
        "protocol": "cohortkv_d2_dev_c0_wave_v1",
        "status": "complete",
        "case": "normal",
        "scientific_result": False,
        "formal_stage_c": False,
        "configuration": {"world_size": 1},
        "epoch_integration": {"connected": True},
        "checks": {"closed": True},
        "scope": {
            "target_epoch_published": False,
            "development_target_pointer_published": True,
        },
        "rank_reports": [{"record_payloads": {2: "b", 10: "a"}}],
    }
    artifact["content_sha256"] = canonical_sha256(artifact)
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact, sort_keys=True))
    with pytest.raises(RuntimeError, match="failed validation"):
        LAUNCHER._validate(path, 1, "normal")


def test_c0_canonical_status_requires_complete_matrix() -> None:
    assert LAUNCHER._is_complete_matrix((1, 2, 3), (3,))
    assert not LAUNCHER._is_complete_matrix((3,), (3,))
    with pytest.raises(ValueError, match="complete matrix"):
        LAUNCHER.main(
            [
                "--world-sizes",
                "3",
                "--pre-commit-abort-world-sizes",
                "3",
                "--visible-devices",
                "0",
                "1",
                "3",
            ]
        )
