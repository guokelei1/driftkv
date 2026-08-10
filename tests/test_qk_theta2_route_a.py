from __future__ import annotations

import json
import runpy
from dataclasses import asdict
from pathlib import Path

from hstu_kvcache.streaming.qk_alignment_runner import (
    _validate_document as validate_alignment_document,
)
from hstu_kvcache.streaming.qk_protocol_sweep_runner import (
    _validate_document as validate_protocol_document,
)


def test_theta2_candidate_bindings_match_frozen_plan() -> None:
    plan = json.loads(
        Path(
            "configs/evokv_foundation/"
            "qk_theta2_route_a_sweep_two_gpu_v0.json"
        ).read_text()
    )
    bindings = [
        runpy.run_path(f"scripts/train_evokv_qk_{name}.py")["BINDING"]
        for name in (
            "theta2_route_a_e3_lr100",
            "theta2_route_a_e4_lr100",
            "theta2_route_a_e3_lr150",
        )
    ]
    observed = {
        binding.candidate_name: asdict(binding) for binding in bindings
    }
    assert set(observed) == {
        value["candidate_name"] for value in plan["candidates"]
    }
    for candidate in plan["candidates"]:
        binding = observed[candidate["candidate_name"]]
        assert binding["source_version"] == 1
        assert binding["target_version"] == 2
        assert binding["edge"] == 2
        for name in (
            "epochs",
            "dense_learning_rate",
            "projection_learning_rate",
            "embedding_learning_rate",
        ):
            assert binding[name] == candidate[name]


def test_alignment_and_protocol_documents_accept_edge_two() -> None:
    root = Path(
        "results/foundation_model/qk_theta1/"
        "qk_theta1_branch_a_e3_lr100_20260806_round1"
    )
    alignment = json.loads((root / "alignment/frozen_config.json").read_text())
    protocol = json.loads(
        (root / "protocol_sweep/frozen_config.json").read_text()
    )
    for document in (alignment, protocol):
        document["edge"] = {
            "source_version": 1,
            "target_version": 2,
            "edge": 2,
            "training_window": 2,
            "evaluation_window": 3,
        }
    validate_alignment_document(alignment)
    validate_protocol_document(protocol)
