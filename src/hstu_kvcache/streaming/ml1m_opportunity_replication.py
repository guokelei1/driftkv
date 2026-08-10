from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from .ml1m_candidate_robustness import _mechanism_summary, _strict_ranking_gate
from .ml1m_opportunity import (
    _atomic_json,
    _base_sequences,
    _evaluate,
    _save_checkpoint,
    _seed_everything,
    _state_delta,
    _train,
    _update_sequences,
    file_sha256,
    load_causal_records,
    load_config,
    make_model,
    summarize_evaluation,
)

PROTOCOL = "evokv_ml1m_opportunity_all_user_two_seed_v0"


def load_replication_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    parent = document.get("parent")
    replication = document.get("replication")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or not isinstance(parent, dict)
        or not isinstance(replication, dict)
        or replication.get("seeds") != [4217, 14929]
        or replication.get("user_limit") != 5923
        or replication.get("candidate_counts") != [50, 100, 3883]
        or replication.get("primary_candidate_count") != 50
        or replication.get("filter_seen") is not True
        or document.get("variants")
        != [
            {
                "architecture": "legacy",
                "id": "a1_legacy_normalized_full",
                "normalized_scoring": True,
                "update_scope": "full",
            },
            {
                "architecture": "dense_hstu_v2",
                "id": "a2_dense_normalized_full",
                "normalized_scoring": True,
                "update_scope": "full",
            },
        ]
    ):
        raise ValueError("ML1m opportunity replication config differs")
    for path_key, hash_key in (
        ("base_config", "base_config_sha256"),
        ("selection_summary", "selection_summary_sha256"),
    ):
        if file_sha256(parent[path_key]) != parent[hash_key]:
            raise ValueError("ML1m opportunity replication parent binding differs")
    return document


def _seed_decision(candidate_results: list[dict[str, Any]], primary_count: int):
    primary = next(
        value for value in candidate_results if value["candidate_count"] == primary_count
    )
    return {
        "primary_candidate_count": primary_count,
        "primary_gate": primary["strict_ranking_gate"],
        "passed": primary["strict_ranking_gate"]["passed"],
    }


def run_replication(config_path: str | Path) -> dict[str, Any]:
    document = load_replication_config(config_path)
    output_root = Path(document["outputs"]["root"])
    summary_path = output_root / "summary.json"
    if summary_path.is_file():
        result = json.loads(summary_path.read_text())
        validate_replication_summary(result, document)
        return result
    output_root.mkdir(parents=True, exist_ok=True)
    base_document = load_config(document["parent"]["base_config"])
    base_document["data"]["user_limit"] = int(document["replication"]["user_limit"])
    causal = load_causal_records(base_document)
    if causal["available_users"] != int(document["replication"]["user_limit"]):
        raise ValueError("ML1m opportunity replication user coverage differs")
    max_seq_len = int(base_document["model"]["max_seq_len"])
    selected = causal["selected_users"]
    base_examples = _base_sequences(causal["splits"]["train"], selected, max_seq_len)
    update_examples = _update_sequences(causal["splits"]["dev"], selected, max_seq_len)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for ML1m opportunity replication")
    started = time.monotonic()
    variants = []
    for variant_index, variant in enumerate(document["variants"]):
        seed_results = []
        for seed_index, seed in enumerate(document["replication"]["seeds"]):
            print(
                f"phase=ml1m_replication_start variant={variant['id']} seed={seed} users={len(selected)}",
                flush=True,
            )
            seed_document = deepcopy(base_document)
            seed_document["training"]["seed"] = int(seed)
            _seed_everything(int(seed))
            previous = make_model(seed_document, variant["architecture"], device)
            base_training = _train(
                previous,
                base_examples,
                seed_document,
                bool(variant["normalized_scoring"]),
                "base",
                "full",
                int(seed) + 1009,
            )
            previous_state = {
                name: value.detach().cpu().clone()
                for name, value in previous.state_dict().items()
            }
            theta0 = _save_checkpoint(
                output_root / "checkpoints" / variant["id"] / f"seed_{seed}" / "theta0.pt",
                previous,
                variant["architecture"],
                {
                    "training": base_training,
                    "selected_users_sha256": causal["selected_users_sha256"],
                },
            )
            current = deepcopy(previous)
            update_training = _train(
                current,
                update_examples,
                seed_document,
                bool(variant["normalized_scoring"]),
                "update",
                variant["update_scope"],
                int(seed) + 2003 + variant_index * 10007 + seed_index * 1000003,
            )
            current_state = {
                name: value.detach().cpu().clone()
                for name, value in current.state_dict().items()
            }
            parameter_delta = _state_delta(previous_state, current_state)
            theta1 = _save_checkpoint(
                output_root / "checkpoints" / variant["id"] / f"seed_{seed}" / "theta1.pt",
                current,
                variant["architecture"],
                {
                    "training": update_training,
                    "selected_users_sha256": causal["selected_users_sha256"],
                    "parameter_delta": parameter_delta,
                },
            )
            candidate_results = []
            mechanism = None
            for count in document["replication"]["candidate_counts"]:
                evaluation = _evaluate(
                    previous,
                    current,
                    causal["splits"]["test"],
                    selected,
                    seed_document,
                    bool(variant["normalized_scoring"]),
                    candidate_count=int(count),
                    candidate_seed=int(document["replication"]["candidate_seed"]),
                    filter_seen=True,
                )
                compact = summarize_evaluation(evaluation, seed_document)
                gate = _strict_ranking_gate(compact)
                result_path = (
                    output_root
                    / "variants"
                    / variant["id"]
                    / f"seed_{seed}"
                    / f"candidate_{count}.json"
                )
                _atomic_json(
                    result_path,
                    {
                        "variant": variant,
                        "seed": seed,
                        "candidate_count": count,
                        "summary": compact,
                        "strict_ranking_gate": gate,
                        "records": evaluation["records"],
                    },
                )
                candidate_results.append(
                    {
                        "candidate_count": count,
                        "result_path": str(result_path),
                        "result_sha256": file_sha256(result_path),
                        "summary": compact,
                        "strict_ranking_gate": gate,
                    }
                )
                if count == int(document["replication"]["primary_candidate_count"]):
                    mechanism = _mechanism_summary(evaluation["records"], seed_document)
                stale = compact["comparisons"]["recompute_over_reuse"]
                print(
                    f"phase=ml1m_replication_result variant={variant['id']} seed={seed} "
                    f"candidates={count} stale_ce={stale['candidate_cross_entropy']['absolute']:.6f} "
                    f"stale_mrr={stale['mrr']['absolute']:.6f} "
                    f"stale_ndcg10={stale['ndcg_at_10']['absolute']:.6f} passed={gate['passed']}",
                    flush=True,
                )
            seed_results.append(
                {
                    "seed": seed,
                    "base_training": base_training,
                    "update_training": update_training,
                    "parameter_delta": parameter_delta,
                    "checkpoints": {"theta0": theta0, "theta1": theta1},
                    "candidate_results": candidate_results,
                    "primary_mechanism": mechanism,
                    "decision": _seed_decision(
                        candidate_results,
                        int(document["replication"]["primary_candidate_count"]),
                    ),
                }
            )
            del previous, current
            torch.cuda.empty_cache()
        variants.append(
            {
                "id": variant["id"],
                "seed_results": seed_results,
                "all_seeds_passed": all(value["decision"]["passed"] for value in seed_results),
            }
        )
    passed = [value["id"] for value in variants if value["all_seeds_passed"]]
    summary = {
        "protocol": PROTOCOL,
        "round_id": document["round_id"],
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "parent": document["parent"],
        "data": {
            "available_users": causal["available_users"],
            "selected_users": len(selected),
            "selected_users_sha256": causal["selected_users_sha256"],
            "target_leakage": False,
        },
        "variants": variants,
        "decision": {
            "passed_variants": passed,
            "positive_candidate_replicated": bool(passed),
            "next": "mechanism_and_balance_derivation" if passed else "objective_or_data_followup",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    validate_replication_summary(summary, document)
    _atomic_json(summary_path, summary)
    return summary


def validate_replication_summary(summary: dict[str, Any], document: dict[str, Any]) -> None:
    variants = summary.get("variants")
    if (
        summary.get("protocol") != PROTOCOL
        or summary.get("round_id") != document["round_id"]
        or summary.get("status") != "complete"
        or summary.get("scientific_result") is not False
        or not isinstance(variants, list)
        or [value.get("id") for value in variants]
        != [value["id"] for value in document["variants"]]
    ):
        raise ValueError("ML1m opportunity replication summary differs")
    for variant in variants:
        seeds = variant.get("seed_results")
        if not isinstance(seeds, list) or [value.get("seed") for value in seeds] != document[
            "replication"
        ]["seeds"]:
            raise ValueError("ML1m opportunity replication seed coverage differs")
        for seed in seeds:
            results = seed.get("candidate_results")
            if not isinstance(results, list) or [value.get("candidate_count") for value in results] != document[
                "replication"
            ]["candidate_counts"]:
                raise ValueError("ML1m opportunity replication candidate coverage differs")
            for result in results:
                path = Path(result["result_path"])
                if not path.is_file() or file_sha256(path) != result["result_sha256"]:
                    raise ValueError("ML1m opportunity replication result binding differs")
                if not result.get("summary", {}).get("sanity", {}).get("passed"):
                    raise ValueError("ML1m opportunity replication sanity failed")
            for checkpoint in seed["checkpoints"].values():
                path = Path(checkpoint["path"])
                if not path.is_file() or file_sha256(path) != checkpoint["sha256"]:
                    raise ValueError("ML1m opportunity replication checkpoint binding differs")
