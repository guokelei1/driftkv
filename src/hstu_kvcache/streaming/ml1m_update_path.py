from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .ml1m_candidate_robustness import _strict_ranking_gate
from .ml1m_hard_candidate import (
    _base_candidates,
    _old_model_hard_map,
    _popular_map,
)
from .ml1m_opportunity import (
    _atomic_json,
    _evaluate,
    file_sha256,
    load_causal_records,
    load_config,
    make_model,
    summarize_evaluation,
)

PROTOCOL = "evokv_ml1m_update_path_interpolation_v0"


def load_update_path_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    parent = document.get("parent")
    sweep = document.get("sweep")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or not isinstance(parent, dict)
        or not isinstance(sweep, dict)
        or sweep.get("alphas") != [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        or sweep.get("variant") != "a1_legacy_normalized_full"
        or sweep.get("strategies")
        != [
            {"candidate_count": 20, "id": "benchmark_hard_20"},
            {"candidate_count": 50, "id": "popular_unseen_50"},
            {"candidate_count": 100, "id": "old_model_hard_100"},
        ]
    ):
        raise ValueError("ML1m update-path config differs")
    for path_key, hash_key in (
        ("base_config", "base_config_sha256"),
        ("replication_config", "replication_config_sha256"),
        ("replication_summary", "replication_summary_sha256"),
    ):
        if file_sha256(parent[path_key]) != parent[hash_key]:
            raise ValueError("ML1m update-path parent binding differs")
    return document


def _load_endpoint_models(
    seed_result: dict[str, Any],
    base_document: dict[str, Any],
    device: torch.device,
):
    models = []
    for version in ("theta0", "theta1"):
        binding = seed_result["checkpoints"][version]
        path = Path(binding["path"])
        if file_sha256(path) != binding["sha256"]:
            raise ValueError("ML1m update-path checkpoint differs")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = make_model(base_document, "legacy", device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
    return models[0], models[1]


def _interpolate(previous, endpoint, alpha: float):
    model = type(previous)(previous.cfg).to(next(previous.parameters()).device)
    previous_state = previous.state_dict()
    endpoint_state = endpoint.state_dict()
    model.load_state_dict(
        {
            name: old + alpha * (endpoint_state[name] - old)
            for name, old in previous_state.items()
        }
    )
    model.eval()
    return model


def run_update_path(config_path: str | Path) -> dict[str, Any]:
    document = load_update_path_config(config_path)
    output_root = Path(document["outputs"]["root"])
    summary_path = output_root / "summary.json"
    if summary_path.is_file():
        result = json.loads(summary_path.read_text())
        validate_update_path_summary(result, document)
        return result
    output_root.mkdir(parents=True, exist_ok=True)
    base_document = load_config(document["parent"]["base_config"])
    replication_config = json.loads(Path(document["parent"]["replication_config"]).read_text())
    base_document["data"]["user_limit"] = int(replication_config["replication"]["user_limit"])
    replication = json.loads(Path(document["parent"]["replication_summary"]).read_text())
    causal = load_causal_records(base_document)
    selected = causal["selected_users"]
    variant = next(
        value for value in replication["variants"] if value["id"] == document["sweep"]["variant"]
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for ML1m update-path interpolation")
    seed_outputs = []
    started = time.monotonic()
    for seed_result in variant["seed_results"]:
        seed = int(seed_result["seed"])
        previous, endpoint = _load_endpoint_models(seed_result, base_document, device)
        benchmark = {
            user_id: np.asarray(_base_candidates(causal["splits"]["test"][user_id]), dtype=np.int64)
            for user_id in selected
        }
        popular = _popular_map(
            causal["splits"]["train"],
            causal["splits"]["test"],
            selected,
            int(base_document["model"]["num_items"]),
            50,
        )
        old_hard = _old_model_hard_map(
            previous,
            causal["splits"]["test"],
            selected,
            base_document,
            100,
            True,
        )
        candidate_maps = {
            "benchmark_hard_20": benchmark,
            "popular_unseen_50": popular,
            "old_model_hard_100": old_hard,
        }
        alpha_results = []
        for alpha in document["sweep"]["alphas"]:
            print(f"phase=ml1m_update_path_start seed={seed} alpha={alpha}", flush=True)
            current = _interpolate(previous, endpoint, float(alpha))
            strategy_results = []
            for strategy in document["sweep"]["strategies"]:
                evaluation = _evaluate(
                    previous,
                    current,
                    causal["splits"]["test"],
                    selected,
                    base_document,
                    True,
                    candidate_map=candidate_maps[strategy["id"]],
                    candidate_protocol_override=strategy["id"],
                )
                compact = summarize_evaluation(evaluation, base_document)
                gate = _strict_ranking_gate(compact)
                strategy_results.append(
                    {
                        "strategy": strategy,
                        "summary": compact,
                        "strict_ranking_gate": gate,
                    }
                )
                stale = compact["comparisons"]["recompute_over_reuse"]
                print(
                    f"phase=ml1m_update_path_result seed={seed} alpha={alpha} "
                    f"strategy={strategy['id']} stale_ce={stale['candidate_cross_entropy']['absolute']:.6f} "
                    f"stale_mrr={stale['mrr']['absolute']:.6f} "
                    f"stale_ndcg10={stale['ndcg_at_10']['absolute']:.6f} passed={gate['passed']}",
                    flush=True,
                )
            alpha_results.append({"alpha": alpha, "strategy_results": strategy_results})
            del current
            torch.cuda.empty_cache()
        result_path = output_root / f"seed_{seed}.json"
        _atomic_json(result_path, {"seed": seed, "alpha_results": alpha_results})
        seed_outputs.append(
            {
                "seed": seed,
                "result_path": str(result_path),
                "result_sha256": file_sha256(result_path),
                "alpha_results": alpha_results,
            }
        )
        del previous, endpoint
        torch.cuda.empty_cache()
    stable = []
    for alpha in document["sweep"]["alphas"]:
        for strategy in document["sweep"]["strategies"]:
            if all(
                next(
                    value
                    for value in next(
                        item for item in seed["alpha_results"] if item["alpha"] == alpha
                    )["strategy_results"]
                    if value["strategy"]["id"] == strategy["id"]
                )["strict_ranking_gate"]["passed"]
                for seed in seed_outputs
            ):
                stable.append({"alpha": alpha, "strategy": strategy["id"]})
    summary = {
        "protocol": PROTOCOL,
        "round_id": document["round_id"],
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "parent": document["parent"],
        "seed_results": seed_outputs,
        "decision": {
            "stable_positive_cells": stable,
            "common_update_interval_found": bool(stable),
            "next": "real_training_binding" if stable else "parameter_group_and_loss_sweep",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    validate_update_path_summary(summary, document)
    _atomic_json(summary_path, summary)
    return summary


def validate_update_path_summary(summary: dict[str, Any], document: dict[str, Any]) -> None:
    seeds = summary.get("seed_results")
    if (
        summary.get("protocol") != PROTOCOL
        or summary.get("round_id") != document["round_id"]
        or summary.get("status") != "complete"
        or summary.get("scientific_result") is not False
        or not isinstance(seeds, list)
        or [value.get("seed") for value in seeds] != [4217, 14929]
    ):
        raise ValueError("ML1m update-path summary differs")
    for seed in seeds:
        path = Path(seed["result_path"])
        if not path.is_file() or file_sha256(path) != seed["result_sha256"]:
            raise ValueError("ML1m update-path result binding differs")
        if [value.get("alpha") for value in seed["alpha_results"]] != document["sweep"]["alphas"]:
            raise ValueError("ML1m update-path alpha coverage differs")
