from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from .kuairand_query_transition import (
    _atomic_json,
    _atomic_torch,
    _evaluate,
    _seed_everything,
    _summary,
    _train,
    build_workload,
    file_sha256,
    load_config,
    make_model,
)

PROTOCOL = "evokv_kuairand_query_multiversion_v0"


def load_chain_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    parent = document.get("parent")
    transitions = document.get("transitions")
    selection = document.get("selection")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not isinstance(parent, dict)
        or not isinstance(transitions, list)
        or len(transitions) < 2
        or not isinstance(selection, dict)
        or selection.get("metrics") != ["mrr", "ndcg_at_10", "hit_rate_at_10"]
        or float(selection.get("minimum_relative_percent", 0)) != 3.0
        or int(selection.get("minimum_metrics", 0)) != 2
        or not 0.0 < float(document.get("publish_alpha", 0)) <= 1.0
    ):
        raise ValueError("KuaiRand multiversion config differs")
    base = parent.get("base_config")
    checkpoints = parent.get("seeds")
    if not isinstance(base, dict) or not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("KuaiRand multiversion parent is absent")
    base_path = Path(base.get("path", ""))
    if not base_path.is_file() or file_sha256(base_path) != base.get("sha256"):
        raise ValueError("KuaiRand multiversion base config differs")
    for seed in checkpoints:
        if not 0.0 < float(seed.get("theta1_source_alpha", 0)) <= 1.0:
            raise ValueError("KuaiRand multiversion source alpha differs")
        for field in ("theta0", "theta1"):
            artifact = seed.get(field)
            artifact_path = Path(artifact.get("path", "")) if isinstance(artifact, dict) else Path()
            if (
                not isinstance(artifact, dict)
                or not artifact_path.is_file()
                or file_sha256(artifact_path) != artifact.get("sha256")
            ):
                raise ValueError("KuaiRand multiversion checkpoint differs")
    expected_source = 0
    for transition in transitions:
        if (
            int(transition.get("source_version", -1)) != expected_source
            or int(transition.get("target_version", -1)) != expected_source + 1
            or len(str(transition.get("update_date", ""))) != 8
            or len(str(transition.get("evaluation_date", ""))) != 8
        ):
            raise ValueError("KuaiRand multiversion transition differs")
        expected_source += 1
    return document


def _reconstruct_published(
    model,
    theta0: dict[str, torch.Tensor],
    source_theta1: dict[str, torch.Tensor],
    source_alpha: float,
    target_alpha: float,
) -> None:
    with torch.no_grad():
        for name, target in model.state_dict().items():
            previous = theta0[name].to(target.device)
            source = source_theta1[name].to(target.device)
            target.copy_(previous + (source - previous) * (target_alpha / source_alpha))


def _publish(previous, current, alpha: float) -> None:
    with torch.no_grad():
        previous_state = previous.state_dict()
        for name, target in current.state_dict().items():
            source = previous_state[name]
            target.copy_(source + (target - source) * alpha)


def _checkpoint(path: Path, model, metadata: dict[str, Any]) -> dict[str, Any]:
    _atomic_torch(
        path,
        {
            "protocol": PROTOCOL,
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "metadata": metadata,
        },
    )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _edge_config(base: dict[str, Any], transition: dict[str, Any], alpha: float):
    document = deepcopy(base)
    document["data"]["update_date"] = str(transition["update_date"])
    document["data"]["evaluation_date"] = str(transition["evaluation_date"])
    document["training"]["publish_alpha"] = alpha
    return document


def _passing_metrics(summary: dict[str, Any], selection: dict[str, Any]) -> list[str]:
    stale = summary["comparisons"]["recompute_over_reuse"]
    return [
        metric
        for metric in selection["metrics"]
        if stale[metric]["positive_direction_with_ci"]
        and stale[metric]["relative_percent"]
        >= float(selection["minimum_relative_percent"])
    ]


def _admitted(summary: dict[str, Any], passing: list[str], minimum: int) -> bool:
    gate = summary["gate"]
    return bool(
        gate["same_model_sanity"]
        and gate["fresh_update_ranking_positive"]
        and gate["history_ranking_positive"]
        and len(passing) >= minimum
    )


def run_multiversion_chain(config_path: str | Path) -> dict[str, Any]:
    document = load_chain_config(config_path)
    output_root = Path(document["outputs"]["root"])
    result_path = output_root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        validate_chain_result(result, document)
        return result
    base_config = load_config(document["parent"]["base_config"]["path"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("KuaiRand multiversion chain requires CUDA")
    alpha = float(document["publish_alpha"])
    minimum = int(document["selection"]["minimum_metrics"])
    seed_results = []
    started = time.monotonic()
    for parent in document["parent"]["seeds"]:
        seed = int(parent["seed"])
        _seed_everything(seed)
        theta0_payload = torch.load(
            parent["theta0"]["path"], map_location="cpu", weights_only=True
        )
        source_theta1_payload = torch.load(
            parent["theta1"]["path"], map_location="cpu", weights_only=True
        )
        first_config = _edge_config(base_config, document["transitions"][0], alpha)
        first_workload = build_workload(first_config)
        previous = make_model(
            first_config, int(first_workload["metadata"]["embedding_rows"]), device
        )
        previous.load_state_dict(theta0_payload["state_dict"])
        previous.eval()
        current = make_model(
            first_config, int(first_workload["metadata"]["embedding_rows"]), device
        )
        _reconstruct_published(
            current,
            theta0_payload["state_dict"],
            source_theta1_payload["state_dict"],
            float(parent["theta1_source_alpha"]),
            alpha,
        )
        current.eval()
        version_checkpoints = {
            "theta0": parent["theta0"],
            "theta1": _checkpoint(
                output_root / "checkpoints" / f"seed_{seed}" / "theta1.pt",
                current,
                {
                    "reconstructed_from": parent["theta1"],
                    "source_alpha": parent["theta1_source_alpha"],
                    "publish_alpha": alpha,
                },
            ),
        }
        edges = []
        for edge_index, transition in enumerate(document["transitions"]):
            edge_config = _edge_config(base_config, transition, alpha)
            workload = first_workload if edge_index == 0 else build_workload(edge_config)
            training = None
            if edge_index > 0:
                previous = current
                current = deepcopy(previous)
                negative_count = min(
                    int(edge_config["training"]["negative_pool_size"]),
                    len(workload["popular_ids"]),
                )
                training = _train(
                    current,
                    workload["update_examples"],
                    workload["popular_ids"][:negative_count],
                    workload["rank_by_item"],
                    workload["author_by_item"],
                    edge_config,
                    "update",
                    seed + 2003 + edge_index * 100003,
                )
                _publish(previous, current, alpha)
                current.eval()
                target = int(transition["target_version"])
                version_checkpoints[f"theta{target}"] = _checkpoint(
                    output_root
                    / "checkpoints"
                    / f"seed_{seed}"
                    / f"theta{target}.pt",
                    current,
                    {
                        "transition": transition,
                        "training": training,
                        "publish_alpha": alpha,
                    },
                )
            evaluation = _evaluate(previous, current, workload, edge_config)
            compact = _summary(evaluation, edge_config)
            passing = _passing_metrics(compact, document["selection"])
            cell = {
                "seed": seed,
                "edge_index": edge_index,
                "transition": transition,
                "training": training,
                "workload": workload["metadata"],
                "summary": compact,
                "passing_metrics": passing,
                "admitted": _admitted(compact, passing, minimum),
                "records": evaluation["records"],
            }
            cell_path = output_root / "cells" / f"seed_{seed}_edge_{edge_index}.json"
            _atomic_json(cell_path, cell)
            edges.append(
                {
                    key: value for key, value in cell.items() if key != "records"
                }
                | {"path": str(cell_path), "sha256": file_sha256(cell_path)}
            )
            stale = compact["comparisons"]["recompute_over_reuse"]
            print(
                f"phase=kuairand_query_chain seed={seed} edge={edge_index} "
                f"mrr={stale['mrr']['relative_percent']:.3f}% "
                f"ndcg10={stale['ndcg_at_10']['relative_percent']:.3f}% "
                f"hr10={stale['hit_rate_at_10']['relative_percent']:.3f}% "
                f"admitted={cell['admitted']}",
                flush=True,
            )
        seed_results.append(
            {
                "seed": seed,
                "edges": edges,
                "checkpoints": version_checkpoints,
                "all_edges_admitted": all(value["admitted"] for value in edges),
            }
        )
        del previous, current, theta0_payload, source_theta1_payload
        torch.cuda.empty_cache()
    all_admitted = all(value["all_edges_admitted"] for value in seed_results)
    result = {
        "protocol": PROTOCOL,
        "round_id": document["round_id"],
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "publish_alpha": alpha,
        "seed_results": seed_results,
        "decision": {
            "all_seeds_all_edges_admitted": all_admitted,
            "next": "capacity_scale" if all_admitted else "edge_specific_diagnosis",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    validate_chain_result(result, document)
    _atomic_json(result_path, result)
    return result


def validate_chain_result(result: dict[str, Any], document: dict[str, Any]) -> None:
    seeds = result.get("seed_results")
    if (
        result.get("protocol") != PROTOCOL
        or result.get("round_id") != document["round_id"]
        or result.get("status") != "complete"
        or result.get("scientific_result") is not False
        or result.get("formal_result") is not False
        or not isinstance(seeds, list)
        or len(seeds) != len(document["parent"]["seeds"])
    ):
        raise ValueError("KuaiRand multiversion result differs")
    for seed in seeds:
        if len(seed.get("edges", [])) != len(document["transitions"]):
            raise ValueError("KuaiRand multiversion edge count differs")
        for edge in seed["edges"]:
            path = Path(edge["path"])
            if (
                not path.is_file()
                or file_sha256(path) != edge["sha256"]
                or not edge.get("summary", {}).get("sanity", {}).get("passed")
            ):
                raise ValueError("KuaiRand multiversion edge binding differs")
        for checkpoint in seed.get("checkpoints", {}).values():
            path = Path(checkpoint["path"])
            if not path.is_file() or file_sha256(path) != checkpoint["sha256"]:
                raise ValueError("KuaiRand multiversion checkpoint binding differs")
