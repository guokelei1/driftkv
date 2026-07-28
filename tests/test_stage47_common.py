import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cohortkv_stage4_7_common.py"
SPEC = importlib.util.spec_from_file_location(
    "cohortkv_stage4_7_common",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

BASE_DATES = [f"202204{day:02d}" for day in range(8, 12)]
TARGET_DATES = [f"202204{day:02d}" for day in range(12, 24)]


def fake_plan(
    future_activity: object = None,
    reverse_user_map: bool = False,
) -> SimpleNamespace:
    pairs = [(100_000 + user_id, user_id) for user_id in range(1, 946)]
    if reverse_user_map:
        pairs.reverse()
    return SimpleNamespace(
        num_users=945,
        trace=SimpleNamespace(user_map=dict(pairs)),
        future_activity=future_activity,
    )


def fake_metadata() -> dict:
    return {
        "base_dates": BASE_DATES,
        "online_dates": TARGET_DATES,
    }


def fake_training(prepared_sha256: str = "") -> dict:
    return {
        "protocol": MODULE.training_protocol_for_base_days(4),
        "status": "complete",
        "args": {"seed": 0},
        "prepared_data": {"sha256": prepared_sha256},
        "model": {
            "num_items": 312_144,
            "num_behaviors": 9,
            "num_prediction_items": 50_000,
            "hidden_size": 512,
            "num_layers": 16,
            "num_heads": 8,
            "head_dim": 64,
            "max_seq_len": 2048,
        },
    }


def make_input_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    prepared = tmp_path / "prepared.npz"
    prepared.write_bytes(b"base-only-prepared-input")
    training = tmp_path / "training.json"
    training.write_text(
        json.dumps(fake_training(MODULE.sha256(prepared)))
    )
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for version in range(12):
        (checkpoints / f"theta_{version}.pt").write_bytes(
            f"theta{version}".encode()
        )
    return prepared, training, checkpoints


def test_base_only_organic_cohort_is_deterministic() -> None:
    first = MODULE.organic_user_ids(945)
    second = MODULE.organic_user_ids(945)
    digest = hashlib.sha256(
        json.dumps(first, separators=(",", ":")).encode()
    ).hexdigest()

    assert first == second
    assert first == tuple(sorted(first))
    assert len(first) == 682
    assert len(set(first)) == 682
    assert min(first) >= 1
    assert max(first) <= 945
    assert digest == (
        "3235bbd8745c91060a84e9560d84384e2e8b27a31ce324d2a7fd11253871eae3"
    )
    with pytest.raises(ValueError, match="smaller"):
        MODULE.organic_user_ids(681)


def test_role_assignment_is_disjoint_deterministic_and_frozen() -> None:
    user_ids = MODULE.organic_user_ids(945)
    first = MODULE.role_assignment(user_ids)
    second = MODULE.role_assignment(user_ids)

    assert first == second
    assert set(first) == set(user_ids)
    assert Counter(first.values()) == {
        "fit": 40,
        "program_selection": 60,
        "certificate": 60,
        "final_test": 522,
    }
    with pytest.raises(ValueError, match="differs"):
        MODULE.role_assignment(user_ids[:-1])


def test_manifest_hash_is_stable_and_ignores_future_activity() -> None:
    training = fake_training()
    first = MODULE.build_manifest(
        fake_plan(
            future_activity={"20220423": [3, 7, 9]},
            reverse_user_map=False,
        ),
        fake_metadata(),
        training,
    )
    second = MODULE.build_manifest(
        fake_plan(
            future_activity={"20220412": list(range(1, 946))},
            reverse_user_map=True,
        ),
        fake_metadata(),
        training,
    )
    unhashed = dict(first)
    digest = unhashed.pop("content_sha256")

    assert first == second
    assert first["selection_boundary"] == {
        "population": "base-only prepared cohort",
        "prepared_users": 945,
        "selected_records": 682,
        "selection": "lowest keyed SHA256 ranks over base-fitted internal user ids",
        "selection_key": MODULE.COHORT_KEY,
        "future_activity_used": False,
    }
    assert digest == MODULE.content_sha256(unhashed)
    assert len({record["user_id"] for record in first["records"]}) == 682
    assert first["roles"] == {
        "fit": 40,
        "program_selection": 60,
        "certificate": 60,
        "final_test": 522,
    }


def test_manifest_freezes_the_4plus12_prequential_timeline() -> None:
    manifest = MODULE.build_manifest(
        fake_plan(),
        fake_metadata(),
        fake_training(),
    )
    timeline = manifest["timeline"]

    assert timeline["base_dates"] == BASE_DATES
    assert timeline["target_dates"] == TARGET_DATES
    assert timeline["versions"] == [
        f"theta{version}" for version in range(12)
    ]
    assert list(
        zip(timeline["versions"], timeline["target_dates"], strict=True)
    ) == [
        (f"theta{version}", TARGET_DATES[version])
        for version in range(12)
    ]
    assert timeline["rule"] == (
        "theta_v uses history available before target_dates[v] and "
        "predicts that unseen date before ingestion"
    )


def test_load_inputs_validates_and_describes_all_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, training_path, checkpoint_dir = make_input_tree(tmp_path)
    plan = fake_plan()
    metadata = fake_metadata()
    validations = []
    monkeypatch.setattr(
        MODULE,
        "load_prepared_kuairand_plan",
        lambda path: (plan, metadata),
    )
    monkeypatch.setattr(
        MODULE,
        "validate_long_context_plan",
        lambda value, details, base_days: validations.append(
            (value, details, base_days)
        ),
    )

    (
        loaded_plan,
        loaded_metadata,
        loaded_training,
        cfg,
        manifest,
        checkpoints,
    ) = MODULE.load_inputs(
        prepared,
        training_path,
        checkpoint_dir,
    )

    assert loaded_plan is plan
    assert loaded_metadata is metadata
    assert loaded_training["status"] == "complete"
    assert validations == [(plan, metadata, 4)]
    assert cfg.num_layers == 16
    assert cfg.hidden_size == 512
    assert manifest["roles"]["final_test"] == 522
    assert [value["version"] for value in checkpoints] == [
        f"theta{version}" for version in range(12)
    ]
    assert all(value["bytes"] > 0 for value in checkpoints)
    assert all(
        value["sha256"] == MODULE.sha256(value["path"])
        for value in checkpoints
    )


def test_load_inputs_rejects_invalid_provenance_and_missing_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, training_path, checkpoint_dir = make_input_tree(tmp_path)
    monkeypatch.setattr(
        MODULE,
        "load_prepared_kuairand_plan",
        lambda path: (fake_plan(), fake_metadata()),
    )
    monkeypatch.setattr(
        MODULE,
        "validate_long_context_plan",
        lambda plan, metadata, base_days: None,
    )
    payload = json.loads(training_path.read_text())
    payload["prepared_data"]["sha256"] = "not-the-prepared-input"
    training_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="prepared input"):
        MODULE.load_inputs(prepared, training_path, checkpoint_dir)

    payload["prepared_data"]["sha256"] = MODULE.sha256(prepared)
    training_path.write_text(json.dumps(payload))
    (checkpoint_dir / "theta_11.pt").unlink()

    with pytest.raises(FileNotFoundError, match="theta_11"):
        MODULE.load_inputs(prepared, training_path, checkpoint_dir)
