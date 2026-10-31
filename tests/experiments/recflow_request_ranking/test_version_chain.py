import json

import numpy as np
import pytest

from experiments.recflow_request_ranking.code.data_common import json_save, sha256
from experiments.recflow_request_ranking.code.version_chain import (
    HERE,
    audit_evaluation,
    audit_update,
    checkpoint_info,
    execute,
    native_admission,
    selection_digest,
)


def update_fixture(tmp_path):
    records = {name: np.array(values) for name, values in {
        "uid": [1, 2], "rid": [8, 9], "day": [27, 27], "ts": [100, 110],
        "history_length": [1024, 1024],
    }.items()}
    np.savez(tmp_path / "request_selection.npz", **records)
    arguments = {"phase": "update", "days": [27], "scale": "max", "epochs": 1,
                 "global_batch": 64, "micro_batch": 16, "listwise_weight": .1,
                 "pilot": False, "request_limit": None, "max_steps": None,
                 "dtype": "bfloat16", "communication": "rows", "seed": 17,
                 "diagnostic_state": False, "stop_after_steps": None}
    metadata = {"scale": "max", "quality_use_permitted": True, "parent_checkpoint_sha256": "parent",
                "source_binding": {"code": "same"}, "proposal": {"fixture": True}, "phase": "update",
                "days": [27], "total_step": 1}
    path = tmp_path / "step_000001.pt"
    path.write_bytes(b"test metadata admission, not a model")
    json_save(path.with_suffix(".json"), {"status": "committed", "bytes": path.stat().st_size,
              "checkpoint_sha256": sha256(path), "metadata": metadata})
    json_save(tmp_path / "latest.json", {"checkpoint": path.name})
    json_save(tmp_path / "completion.json", {"status": "completed", "steps": 1, "next_epoch": 1, "next_step": 0})
    json_save(tmp_path / "plan.json", {"arguments": arguments, "world_size": 4, "requests": 2,
              "planned_steps": 1, "parent_checkpoint_sha256": "parent", "trainer_sha256": sha256(HERE / "train.py"),
              "source_binding": metadata["source_binding"], "request_selection_sha256": sha256(tmp_path / "request_selection.npz")})
    expected = {"requests": 2, "selection_sha256": selection_digest(records)}
    return path, expected, metadata


def test_chain_reuses_only_complete_update_with_actual_parent_and_input_order(tmp_path):
    path, expected, metadata = update_fixture(tmp_path)
    result = audit_update(tmp_path, "parent", 27, expected, metadata["proposal"], metadata["source_binding"])
    assert result["checkpoint"] == str(path)
    with pytest.raises(ValueError, match="actual training parent"):
        audit_update(tmp_path, "some_other_release", 27, expected, metadata["proposal"], metadata["source_binding"])
    with pytest.raises(ValueError, match="population/order"):
        audit_update(tmp_path, "parent", 27, dict(expected, selection_sha256="different"),
                     metadata["proposal"], metadata["source_binding"])


@pytest.mark.parametrize("change", ["partial", "diagnostic", "changed_bytes", "different_recipe"])
def test_chain_rejects_non_reusable_training_artifacts(tmp_path, change):
    path, expected, metadata = update_fixture(tmp_path)
    if change == "partial":
        json_save(tmp_path / "completion.json", {"status": "wall_budget_reached_checkpoint_saved", "steps": 1,
                  "next_epoch": 0, "next_step": 1})
    elif change == "diagnostic":
        value = json.loads(path.with_suffix(".json").read_text())
        value["metadata"]["quality_use_permitted"] = False
        json_save(path.with_suffix(".json"), value)
    elif change == "different_recipe":
        value = json.loads((tmp_path / "plan.json").read_text())
        value["arguments"]["epochs"] = 2
        json_save(tmp_path / "plan.json", value)
    else:
        path.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError):
        audit_update(tmp_path, "parent", 27, expected, metadata["proposal"], metadata["source_binding"])


def test_checkpoint_reuse_checks_file_hash_even_when_size_matches(tmp_path):
    path, _, _ = update_fixture(tmp_path)
    original = path.read_bytes()
    path.write_bytes(b"x" * len(original))
    with pytest.raises(ValueError, match="bytes changed"):
        checkpoint_info(path, verify_bytes=True)


def native_fixture(tmp_path, cache_difference):
    selected = {"epoch": 2, "checkpoint": "model.pt", "checkpoint_sha256": "selected", "native_validation_ndcg10": .04}
    first = dict(selected, epoch=1, native_validation_ndcg10=.03)
    json_save(tmp_path / "native_selection.json", {"selected": selected, "choices": [first, selected],
              "selection_before_cache_metrics": True, "confirmation_unopened": True})
    validation = {"status": "passed", "checkpoint_sha256": "selected", "role": 1, "days": [24, 25, 26],
                  "requests": 4096, "history_limit": 1024,
                  "metrics": {"current": {"ndcg@10": .04}, "random": {"ndcg@10": .015}, "popularity": {"ndcg@10": .02}}}
    json_save(tmp_path / "alignment_epoch2_validation/summary.json", validation)
    summary = {"paired_current_minus": {"random": {"ci95": [.01, .03]}, "popularity": {"ci95": [.01, .03]},
                                         "reuse": {"ci95": [cache_difference - .01, cache_difference + .01]}}}
    json_save(tmp_path / "m0_native_d27/summary.json", summary)
    return selected


@pytest.mark.parametrize("gap", [-.05, 0., .05])
def test_native_admission_does_not_choose_or_stop_by_cache_gap(tmp_path, gap):
    selected = native_fixture(tmp_path, gap)
    actual, evidence = native_admission(tmp_path)
    assert actual == selected
    assert evidence["cache_gap_used_for_admission"] is False


def test_positive_cache_loss_cannot_compensate_for_poor_native_learning(tmp_path):
    native_fixture(tmp_path, .1)
    path = tmp_path / "alignment_epoch2_validation/summary.json"
    value = json.loads(path.read_text())
    value["metrics"]["popularity"]["ndcg@10"] = .1
    json_save(path, value)
    with pytest.raises(ValueError, match="native validation"):
        native_admission(tmp_path)


def test_active_pilot_cannot_dispatch_or_open_confirmation(tmp_path):
    plan = {"data": str(tmp_path / "data"), "pilot": str(tmp_path / "active_pilot")}
    with pytest.raises(RuntimeError, match="still running"):
        execute(plan, tmp_path)
    assert not (tmp_path / "plan.json").exists()
    assert not (tmp_path / "data/formal_recipe_freeze.json").exists()


@pytest.mark.parametrize("change", [{"role": 3}, {"days": [29]}, {"history_limit": 128}, {"parent_sha256": "wrong"}])
def test_evaluation_reuse_requires_same_population_window_and_parent(tmp_path, change):
    summary = {"status": "passed", "checkpoint_sha256": "current", "parent_sha256": "parent",
               "days": [28], "role": 2, "first_only": False, "history_limit": 1024,
               "requests": 2, "arguments": {"limit": None}, "source_binding": {"same": True},
               "world_size": 4, "evaluator_sha256": sha256(HERE / "evaluate.py"),
               "precision": "float32_parameters_and_KV_tf32_disabled"}
    json_save(tmp_path / "summary.json", summary)
    args = (tmp_path, {"sha256": "current"}, {"sha256": "parent"}, 28, 2, 2, {"same": True})
    assert audit_evaluation(*args) == summary
    json_save(tmp_path / "summary.json", dict(summary, **change))
    with pytest.raises(ValueError, match="Evaluation has a different"):
        audit_evaluation(*args)
