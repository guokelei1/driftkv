"""CPU metadata and mocked-dispatch checks; never load a model or launch training."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import prepare_update as prepare
from . import run_train_pair as runner


ROOT = prepare.BASE / "training_runs"
PARENT = ROOT / "m6_d34_listwise48/training/step_002506.pt"
SOURCE = ROOT / "m5_d33_listwise12/training/step_002450.pt"
SIBLING = ROOT / "m6_d34_listwise24/training/step_002506.pt"


@pytest.fixture
def arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "BASE", tmp_path)
    return argparse.Namespace(parent_checkpoint=PARENT, day=35, listwise_weight=4.8,
                              name="metadata_check", source_label="publishedV6", target_label="next",
                              question=None, dry_run=True, launch=False)


def test_legacy_namespace_and_spec_keep_parent_source(arguments, tmp_path):
    result = prepare.prepare(arguments)
    spec = result["experiment"]
    assert spec["training_parent"] == spec["cache_source"] == str(PARENT)
    assert len(spec["cache_source_to_training_parent_path"]) == 1
    legacy = {key: value for key, value in spec.items()
              if key not in {"training_parent", "cache_source", "cache_source_to_training_parent_path"}}
    assert runner.validate_dispatch_spec(legacy) == (PARENT, PARENT)
    assert not list(tmp_path.iterdir())


def test_real_ancestor_and_explicit_equal_source(arguments, tmp_path):
    arguments.cache_source_checkpoint = SOURCE
    spec = prepare.prepare(arguments)["experiment"]
    assert runner.validate_dispatch_spec(spec) == (PARENT, SOURCE)
    assert [row["checkpoint"] for row in spec["cache_source_to_training_parent_path"]] == [str(SOURCE), str(PARENT)]
    assert spec["training_command"][spec["training_command"].index("--parent")+1] == str(PARENT)
    assert (spec["training_requests"], spec["training_steps"], spec["evaluation_day"]) == (154820, 2420, 36)
    assert spec["physical_gpus"] == [0, 1, 2, 3]
    assert spec["evaluation_selection_preview"]["requests"] == 197656
    arguments.cache_source_checkpoint = PARENT
    assert prepare.prepare(arguments)["experiment"]["cache_source"] == str(PARENT)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("source", [SIBLING,
    prepare.ORIGINAL / "runs/attempt_1/training/m9_d36/step_002487.pt",
    prepare.ORIGINAL / "runs/attempt_1/training/m5_d32/step_002494.pt"])
def test_real_sibling_future_and_unrelated_sources_rejected(arguments, source, tmp_path):
    assert source.is_file()
    arguments.cache_source_checkpoint = source
    with pytest.raises(ValueError, match="ancestor"):
        prepare.prepare(arguments)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("change", ["sibling", "parent_alias", "command_parent", "duplicate_parent", "path"])
def test_dispatch_rechecks_source_and_training_parent(arguments, change):
    arguments.cache_source_checkpoint = SOURCE
    spec = prepare.prepare(arguments)["experiment"]
    if change == "sibling":
        spec["cache_source"] = str(SIBLING)
    elif change == "parent_alias":
        spec["parent"] = str(SOURCE)
    elif change == "command_parent":
        spec["training_command"][spec["training_command"].index("--parent")+1] = str(SOURCE)
    elif change == "duplicate_parent":
        spec["training_command"].append("--parent=" + str(SOURCE))
    else:
        spec["cache_source_to_training_parent_path"][0]["train_end_day"] = 34
    with pytest.raises(ValueError):
        runner.validate_dispatch_spec(spec)


@pytest.mark.parametrize("wrong_endpoint_parent", [False, True])
def test_runner_routes_cache_source_but_checks_actual_endpoint_parent(arguments, tmp_path, monkeypatch,
                                                                    wrong_endpoint_parent):
    arguments.cache_source_checkpoint = SOURCE
    spec = prepare.prepare(arguments)["experiment"]
    run = tmp_path / "driver_check"
    run.mkdir()
    (run / "experiment.json").write_text(json.dumps(spec))
    calls = []

    def fake_popen(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            training = Path(spec["training_output"])
            training.mkdir(parents=True)
            endpoint = training / "step_002420.pt"
            endpoint.touch()
            meta = copy.deepcopy(prepare.read(PARENT.with_suffix(".json"))["metadata"])
            meta.update(days=[35])
            meta["arguments"].update(parent=str(SOURCE if wrong_endpoint_parent else PARENT), listwise_weight=4.8)
            endpoint.with_suffix(".json").write_text(json.dumps({"metadata": meta}))
            (training / "latest.json").write_text(json.dumps({"checkpoint": endpoint.name}))
            (training / "completion.json").write_text(json.dumps({"status": "completed", "next_epoch": 1, "next_step": 0}))
        elif len(calls) == 2:
            assert command[command.index("--source-checkpoint")+1] == str(SOURCE)
            assert command[command.index("--scenario")+1] == spec["publication_scenario"]
        else:
            evaluation = Path(spec["evaluation_directory"])
            evaluation.mkdir(parents=True)
            (evaluation / "completion.json").write_text(json.dumps({"status": "completed", "primary_result": {}}))
        return SimpleNamespace(pid=12345, wait=lambda: 0)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner.sys, "argv", ["run_train_pair", "--run", str(run)])
    if wrong_endpoint_parent:
        with pytest.raises(ValueError, match="saved endpoint"):
            runner.main()
        assert len(calls) == 1
    else:
        runner.main()
        complete = runner.read(run / "completion.json")
        assert len(calls) == 3
        assert complete["training_parent"] == str(PARENT)
        assert complete["cache_source"] == str(SOURCE)
