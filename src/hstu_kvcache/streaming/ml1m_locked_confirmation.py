from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from .ml1m_opportunity import (
    _atomic_json,
    _evaluate,
    _save_checkpoint,
    _seed_everything,
    _state_delta,
    file_sha256,
    load_causal_records,
    load_config,
    make_model,
    summarize_evaluation,
)
from .ml1m_query_objective import (
    _base_examples,
    _candidate_maps,
    _train_query,
    _update_examples,
)

PROTOCOL_V0 = "evokv_ml1m_locked_popular50_confirmation_v0"
PROTOCOL_V1 = "evokv_ml1m_locked_popular50_metric_family_confirmation_v1"


def load_locked_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    protocol = document.get("protocol")
    expected_strategy = {
        "candidate_count": 50,
        "id": "popular_unseen_50",
        "source": "train_popularity",
    }
    expected_variant = {
        "architecture": "legacy",
        "id": "q1_legacy_normalized_query",
        "normalized_scoring": True,
    }
    training = document.get("training", {})
    gate = document.get("gate", {})
    common_invalid = (
        protocol not in (PROTOCOL_V0, PROTOCOL_V1)
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("variant") != expected_variant
        or document.get("evaluation", {}).get("strategy") != expected_strategy
        or training.get("base_epochs") != 3
        or training.get("update_epochs") != 3
        or training.get("query_loss_weight") != 16.0
        or gate.get("minimum_relative_percent") != 5.0
        or gate.get("bootstrap_ci_required") is not True
    )
    v0_valid = (
        protocol == PROTOCOL_V0
        and training.get("seed") == 23711
        and gate.get("ranking_metrics") == ["hit_rate_at_10", "ndcg_at_10"]
        and "minimum_passing_ranking_metrics" not in gate
    )
    v1_valid = (
        protocol == PROTOCOL_V1
        and training.get("seed") == 53117
        and gate.get("ranking_metrics")
        == ["hit_rate_at_5", "hit_rate_at_10", "mrr", "ndcg_at_5", "ndcg_at_10"]
        and gate.get("minimum_passing_ranking_metrics") == 2
    )
    if common_invalid or not (v0_valid or v1_valid):
        raise ValueError("ML1m locked confirmation config differs")
    parent = document.get("parent", {})
    for key in ("base_config", "query_config", "discovery_summary"):
        if file_sha256(parent.get(key, "")) != parent.get(f"{key}_sha256"):
            raise ValueError("ML1m locked confirmation parent binding differs")
    if protocol == PROTOCOL_V1 and file_sha256(parent.get("gate_audit", "")) != parent.get(
        "gate_audit_sha256"
    ):
        raise ValueError("ML1m locked confirmation gate audit differs")
    return document


def _locked_gate(summary: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    comparisons = summary["comparisons"]
    if document["protocol"] == PROTOCOL_V0:
        requirements = {
            "same_model_sanity": summary["sanity"]["passed"],
            "fresh_update_ranking": all(
                comparisons["fresh_update_value"][metric]["positive_direction_with_ci"]
                for metric in document["gate"]["ranking_metrics"]
            ),
            "history_ranking": all(
                comparisons["history_value"][metric]["positive_direction_with_ci"]
                for metric in document["gate"]["ranking_metrics"]
            ),
            "stale_cross_entropy": comparisons["recompute_over_reuse"][
                "candidate_cross_entropy"
            ]["positive_direction_with_ci"],
        }
        threshold = float(document["gate"]["minimum_relative_percent"])
        for metric in document["gate"]["ranking_metrics"]:
            value = comparisons["recompute_over_reuse"][metric]
            requirements[f"stale_{metric}_ci"] = value["positive_direction_with_ci"]
            requirements[f"stale_{metric}_effect"] = value["relative_percent"] >= threshold
        return {"requirements": requirements, "passed": all(requirements.values())}
    requirements = {
        "same_model_sanity": summary["sanity"]["passed"],
        "fresh_update_ranking": all(
            comparisons["fresh_update_value"][metric]["positive_direction_with_ci"]
            for metric in ("mrr", "ndcg_at_10")
        ),
        "history_ranking": all(
            comparisons["history_value"][metric]["positive_direction_with_ci"]
            for metric in ("mrr", "ndcg_at_10")
        ),
        "stale_cross_entropy": comparisons["recompute_over_reuse"][
            "candidate_cross_entropy"
        ]["positive_direction_with_ci"],
    }
    threshold = float(document["gate"]["minimum_relative_percent"])
    passing_metrics = []
    for metric in document["gate"]["ranking_metrics"]:
        value = comparisons["recompute_over_reuse"][metric]
        if value["positive_direction_with_ci"] and value["relative_percent"] >= threshold:
            passing_metrics.append(metric)
    minimum = int(
        document["gate"].get(
            "minimum_passing_ranking_metrics",
            len(document["gate"]["ranking_metrics"]),
        )
    )
    requirements["stale_ranking_family"] = len(passing_metrics) >= minimum
    return {
        "minimum_passing_ranking_metrics": minimum,
        "passing_ranking_metrics": passing_metrics,
        "requirements": requirements,
        "passed": all(requirements.values()),
    }


def _discovery_bindings(document: dict[str, Any]) -> list[dict[str, Any]]:
    source = json.loads(Path(document["parent"]["discovery_summary"]).read_text())
    variant = next(value for value in source["variants"] if value["id"] == document["variant"]["id"])
    output = []
    strategy_id = document["evaluation"]["strategy"]["id"]
    for seed in variant["seed_results"]:
        result = next(
            value for value in seed["strategy_results"] if value["strategy"]["id"] == strategy_id
        )
        output.append(
            {
                "seed": seed["seed"],
                "result_path": result["result_path"],
                "result_sha256": result["result_sha256"],
                "recompute_over_reuse": result["summary"]["comparisons"][
                    "recompute_over_reuse"
                ],
            }
        )
    return output


def run_locked_confirmation(config_path: str | Path) -> dict[str, Any]:
    document = load_locked_config(config_path)
    output_root = Path(document["outputs"]["root"])
    summary_path = output_root / "summary.json"
    if summary_path.is_file():
        result = json.loads(summary_path.read_text())
        validate_locked_summary(result, document)
        return result
    output_root.mkdir(parents=True, exist_ok=True)
    base_document = load_config(document["parent"]["base_config"])
    base_document["data"]["user_limit"] = int(document["data"]["user_limit"])
    base_document["training"].update(document["training"])
    causal = load_causal_records(base_document)
    selected = causal["selected_users"]
    max_seq_len = int(base_document["model"]["max_seq_len"])
    base_examples = _base_examples(causal["splits"]["train"], selected, max_seq_len)
    update_examples = _update_examples(causal["splits"]["dev"], selected, max_seq_len)
    candidate_maps = _candidate_maps(
        causal,
        selected,
        int(base_document["model"]["num_items"]),
        int(document["evaluation"]["candidate_seed"]),
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for ML1m locked confirmation")
    seed = int(document["training"]["seed"])
    started = time.monotonic()
    _seed_everything(seed)
    previous = make_model(base_document, document["variant"]["architecture"], device)
    base_training = _train_query(previous, base_examples, base_document, "base", seed + 1009)
    previous_state = {
        name: value.detach().cpu().clone() for name, value in previous.state_dict().items()
    }
    theta0 = _save_checkpoint(
        output_root / "checkpoints" / "theta0.pt",
        previous,
        document["variant"]["architecture"],
        {"training": base_training, "prediction_protocol": "learned_query_after_history"},
    )
    current = deepcopy(previous)
    update_training = _train_query(current, update_examples, base_document, "update", seed + 2003)
    current_state = {
        name: value.detach().cpu().clone() for name, value in current.state_dict().items()
    }
    theta1 = _save_checkpoint(
        output_root / "checkpoints" / "theta1.pt",
        current,
        document["variant"]["architecture"],
        {
            "training": update_training,
            "prediction_protocol": "learned_query_after_history",
            "parameter_delta": _state_delta(previous_state, current_state),
        },
    )
    strategy_id = document["evaluation"]["strategy"]["id"]
    evaluation = _evaluate(
        previous,
        current,
        causal["splits"]["test"],
        selected,
        base_document,
        True,
        candidate_map=candidate_maps[strategy_id],
        candidate_protocol_override=strategy_id,
        prediction_query=True,
    )
    compact = summarize_evaluation(evaluation, base_document)
    gate = _locked_gate(compact, document)
    records_path = output_root / "records.json"
    _atomic_json(records_path, {"records": evaluation["records"]})
    summary = {
        "protocol": document["protocol"],
        "round_id": document["round_id"],
        "status": "complete_development_confirmation",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "parent": document["parent"],
        "variant": document["variant"],
        "strategy": document["evaluation"]["strategy"],
        "seed": seed,
        "training": {"base": base_training, "update": update_training},
        "checkpoints": {"theta0": theta0, "theta1": theta1},
        "summary": compact,
        "locked_gate": gate,
        "discovery_seed_bindings": _discovery_bindings(document),
        "records": {"path": str(records_path), "sha256": file_sha256(records_path)},
        "decision": {
            "independent_seed_passed": gate["passed"],
            "positive_candidate_confirmed": gate["passed"],
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    validate_locked_summary(summary, document)
    _atomic_json(summary_path, summary)
    return summary


def validate_locked_summary(summary: dict[str, Any], document: dict[str, Any]) -> None:
    if (
        summary.get("protocol") != document["protocol"]
        or summary.get("round_id") != document["round_id"]
        or summary.get("status") != "complete_development_confirmation"
        or summary.get("scientific_result") is not False
        or summary.get("formal_result") is not False
        or summary.get("seed") != document["training"]["seed"]
        or summary.get("strategy") != document["evaluation"]["strategy"]
        or summary.get("config", {}).get("sha256") != file_sha256(summary["config"]["path"])
        or not summary.get("summary", {}).get("sanity", {}).get("passed")
    ):
        raise ValueError("ML1m locked confirmation summary differs")
    for binding in summary.get("checkpoints", {}).values():
        if file_sha256(binding.get("path", "")) != binding.get("sha256"):
            raise ValueError("ML1m locked confirmation checkpoint differs")
    records = summary.get("records", {})
    if file_sha256(records.get("path", "")) != records.get("sha256"):
        raise ValueError("ML1m locked confirmation records differ")
    expected = _locked_gate(summary["summary"], document)
    if summary.get("locked_gate") != expected:
        raise ValueError("ML1m locked confirmation decision differs")
    if summary.get("decision", {}).get("positive_candidate_confirmed") != expected["passed"]:
        raise ValueError("ML1m locked confirmation positive decision differs")
