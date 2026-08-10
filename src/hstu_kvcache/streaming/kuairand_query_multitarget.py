from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .kuairand_query_multiversion import (
    _admitted,
    _edge_config,
    _passing_metrics,
    load_chain_config,
    validate_chain_result,
)
from .kuairand_query_transition import (
    _atomic_json,
    _evaluate,
    _summary,
    build_workload,
    file_sha256,
    load_config,
    make_model,
)

PROTOCOL = "evokv_kuairand_query_multitarget_reevaluation_v0"


def load_multitarget_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    parent = document.get("parent")
    selection = document.get("selection")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not isinstance(parent, dict)
        or not isinstance(selection, dict)
        or int(document.get("evaluation_targets_per_user", 0)) not in (2, 4, 8)
        or selection.get("metrics") != ["mrr", "ndcg_at_10", "hit_rate_at_10"]
        or float(selection.get("minimum_relative_percent", 0)) != 3.0
        or int(selection.get("minimum_metrics", 0)) != 2
    ):
        raise ValueError("KuaiRand multitarget config differs")
    for field in ("base_config", "chain_result"):
        artifact = parent.get(field)
        artifact_path = Path(artifact.get("path", "")) if isinstance(artifact, dict) else Path()
        if (
            not isinstance(artifact, dict)
            or not artifact_path.is_file()
            or file_sha256(artifact_path) != artifact.get("sha256")
        ):
            raise ValueError("KuaiRand multitarget parent differs")
    return document


def run_multitarget_reevaluation(config_path: str | Path) -> dict[str, Any]:
    document = load_multitarget_config(config_path)
    output_root = Path(document["outputs"]["root"])
    result_path = output_root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        validate_multitarget_result(result, document)
        return result
    chain = json.loads(Path(document["parent"]["chain_result"]["path"]).read_text())
    chain_document = load_chain_config(chain["config"]["path"])
    validate_chain_result(chain, chain_document)
    base_config = load_config(document["parent"]["base_config"]["path"])
    target_count = int(document["evaluation_targets_per_user"])
    edge_documents = []
    workloads = []
    for transition in chain_document["transitions"]:
        edge_document = _edge_config(
            base_config, transition, float(chain["publish_alpha"])
        )
        edge_document["data"]["evaluation_targets_per_user"] = target_count
        edge_documents.append(edge_document)
        workloads.append(build_workload(edge_document))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("KuaiRand multitarget reevaluation requires CUDA")
    seed_results = []
    started = time.monotonic()
    minimum = int(document["selection"]["minimum_metrics"])
    for seed_result in chain["seed_results"]:
        seed = int(seed_result["seed"])
        previous = make_model(
            edge_documents[0], int(workloads[0]["metadata"]["embedding_rows"]), device
        )
        current = make_model(
            edge_documents[0], int(workloads[0]["metadata"]["embedding_rows"]), device
        )
        edges = []
        for edge_index, transition in enumerate(chain_document["transitions"]):
            source_version = int(transition["source_version"])
            target_version = int(transition["target_version"])
            source_checkpoint = seed_result["checkpoints"][f"theta{source_version}"]
            target_checkpoint = seed_result["checkpoints"][f"theta{target_version}"]
            source_payload = torch.load(
                source_checkpoint["path"], map_location="cpu", weights_only=True
            )
            target_payload = torch.load(
                target_checkpoint["path"], map_location="cpu", weights_only=True
            )
            previous.load_state_dict(source_payload["state_dict"])
            current.load_state_dict(target_payload["state_dict"])
            previous.eval()
            current.eval()
            evaluation = _evaluate(
                previous, current, workloads[edge_index], edge_documents[edge_index]
            )
            compact = _summary(evaluation, edge_documents[edge_index])
            passing = _passing_metrics(compact, document["selection"])
            cell = {
                "seed": seed,
                "edge_index": edge_index,
                "transition": transition,
                "workload": workloads[edge_index]["metadata"],
                "source_checkpoint": source_checkpoint,
                "target_checkpoint": target_checkpoint,
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
                f"phase=kuairand_query_multitarget seed={seed} edge={edge_index} "
                f"records={compact['evaluation_records']} "
                f"mrr={stale['mrr']['relative_percent']:.3f}% "
                f"ndcg10={stale['ndcg_at_10']['relative_percent']:.3f}% "
                f"hr10={stale['hit_rate_at_10']['relative_percent']:.3f}% "
                f"admitted={cell['admitted']}",
                flush=True,
            )
            del source_payload, target_payload
        seed_results.append(
            {
                "seed": seed,
                "edges": edges,
                "all_edges_admitted": all(value["admitted"] for value in edges),
            }
        )
        del previous, current
        torch.cuda.empty_cache()
    all_admitted = all(value["all_edges_admitted"] for value in seed_results)
    result = {
        "protocol": PROTOCOL,
        "round_id": document["round_id"],
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "parent": document["parent"],
        "evaluation_targets_per_user": target_count,
        "seed_results": seed_results,
        "decision": {
            "all_seeds_all_edges_admitted": all_admitted,
            "next": "capacity_scale" if all_admitted else "chain_policy_revision",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    validate_multitarget_result(result, document)
    _atomic_json(result_path, result)
    return result


def validate_multitarget_result(
    result: dict[str, Any], document: dict[str, Any]
) -> None:
    seeds = result.get("seed_results")
    if (
        result.get("protocol") != PROTOCOL
        or result.get("round_id") != document["round_id"]
        or result.get("status") != "complete"
        or result.get("scientific_result") is not False
        or result.get("formal_result") is not False
        or not isinstance(seeds, list)
        or not seeds
    ):
        raise ValueError("KuaiRand multitarget result differs")
    for seed in seeds:
        for edge in seed.get("edges", []):
            path = Path(edge["path"])
            if (
                not path.is_file()
                or file_sha256(path) != edge["sha256"]
                or not edge.get("summary", {}).get("sanity", {}).get("passed")
            ):
                raise ValueError("KuaiRand multitarget cell binding differs")
