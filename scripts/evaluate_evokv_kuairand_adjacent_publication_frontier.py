from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from hstu_kvcache.streaming.kuairand_projected_persistent import (
    _capture_old,
    _distributed,
    _evaluate_captured,
    _evaluation_batches,
    _initialize_model,
    _lineage_holdout_summary,
    _load_checkpoint,
    _seed,
    _temperature_edge_document,
    load_persistent_config,
)
from hstu_kvcache.streaming.kuairand_query_multiversion import _edge_config
from hstu_kvcache.streaming.kuairand_query_transition import (
    _atomic_json,
    build_workload,
    file_sha256,
    load_config,
)

PROTOCOL = "evokv_kuairand_adjacent_publication_frontier_v0"


def _artifact(record: Any) -> Path:
    if not isinstance(record, dict):
        raise ValueError("KuaiRand publication-frontier artifact differs")
    path = Path(record.get("path", ""))
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise ValueError("KuaiRand publication-frontier artifact hash differs")
    return path


def load_frontier_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    evaluation = document.get("evaluation")
    execution = document.get("execution")
    outputs = document.get("outputs")
    parent = document.get("parent")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_user_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not isinstance(evaluation, dict)
        or evaluation.get("target_counts") != [1, 2, 4, 8]
        or int(evaluation.get("minimum_target_version", 0)) != 2
        or int(evaluation.get("maximum_target_version", 0)) != 10
        or evaluation.get("primary_metric") != "ndcg_at_5"
        or evaluation.get("partition") != "holdout"
        or not isinstance(execution, dict)
        or int(execution.get("world_size", 0)) != 2
        or execution.get("visible_devices") != [0, 1]
        or not isinstance(outputs, dict)
        or not isinstance(parent, dict)
    ):
        raise ValueError("KuaiRand publication-frontier config differs")
    for name in ("config", "result", "retention"):
        _artifact(parent.get(name))
    return document


def _broadcast(value: Any, rank: int) -> Any:
    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _validate_parent(
    document: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent_config_path = _artifact(document["parent"]["config"])
    chain_document = load_persistent_config(parent_config_path)
    chain_result = json.loads(_artifact(document["parent"]["result"]).read_text())
    retention = json.loads(_artifact(document["parent"]["retention"]).read_text())
    if (
        chain_result.get("status") != "complete"
        or int(chain_result.get("checkpoint_count", 0)) != 10
        or retention.get("status") != "complete"
        or retention.get("retained", {})
        .get("theta1_theta10", {})
        .get("versions")
        != list(range(1, 11))
    ):
        raise ValueError("KuaiRand publication-frontier parent state differs")
    checkpoint_root = Path(
        retention["retained"]["theta1_theta10"]["path"]
    )
    hashes = retention["retained"]["theta1_theta10"]["manifest_sha256"]
    for version in range(1, 11):
        manifest = checkpoint_root / f"theta_{version}" / "manifest.json"
        if not manifest.is_file() or file_sha256(manifest) != hashes[f"theta{version}"]:
            raise ValueError("KuaiRand publication-frontier checkpoint differs")
    return parent_config_path, chain_document, chain_result, retention


def preflight(path: str | Path) -> dict[str, Any]:
    document = load_frontier_config(path)
    _, chain_document, chain_result, retention = _validate_parent(document)
    return {
        "status": "ready",
        "world_size": chain_document["execution"]["world_size"],
        "target_counts": document["evaluation"]["target_counts"],
        "versions": chain_result["checkpoint_count"],
        "checkpoint_root": retention["retained"]["theta1_theta10"]["path"],
        "checkpoint_bytes": retention["retained"]["theta1_theta10"][
            "checkpoint_bytes"
        ],
        "writes_checkpoints": False,
    }


def _workloads(
    chain_document: dict[str, Any],
    base_config: dict[str, Any],
    target_counts: list[int],
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    maximum_count = max(target_counts)
    maximum_documents = []
    maximum_workloads = []
    for transition in chain_document["transitions"]:
        edge_document = _edge_config(base_config, transition, 1.0)
        if "update_dates" in transition:
            edge_document["data"]["update_dates"] = transition["update_dates"]
        edge_document["data"]["evaluation_targets_per_user"] = maximum_count
        edge_document["data"]["user_limit"] = chain_document["data"].get("user_limit")
        edge_document["evaluation"]["candidate_count"] = int(
            chain_document["evaluation"]["candidate_count"]
        )
        maximum_documents.append(edge_document)
        maximum_workloads.append(build_workload(edge_document))
    edge_documents: dict[int, list[dict[str, Any]]] = {}
    workloads: dict[int, list[dict[str, Any]]] = {}
    for target_count in target_counts:
        count_documents = []
        count_workloads = []
        for maximum_document, maximum_workload in zip(
            maximum_documents, maximum_workloads, strict=True
        ):
            edge_document = json.loads(json.dumps(maximum_document))
            edge_document["data"]["evaluation_targets_per_user"] = target_count
            keys = [
                key
                for key in maximum_workload["evaluation_keys"]
                if int(key[1]) < target_count
            ]
            workload = dict(maximum_workload)
            workload["evaluation"] = {
                key: maximum_workload["evaluation"][key] for key in keys
            }
            workload["candidate_maps"] = {
                key: maximum_workload["candidate_maps"][key] for key in keys
            }
            workload["evaluation_keys"] = keys
            workload["metadata"] = dict(maximum_workload["metadata"])
            workload["metadata"]["evaluation_records"] = len(keys)
            workload["metadata"]["evaluation_targets_per_user"] = target_count
            workload["metadata"]["evaluation_semantics"] = (
                f"first {target_count} eligible engaged actions on the next natural day "
                "from one nested eight-target candidate universe"
            )
            count_documents.append(edge_document)
            count_workloads.append(workload)
        edge_documents[target_count] = count_documents
        workloads[target_count] = count_workloads
    return edge_documents, workloads


def run(path: str | Path) -> dict[str, Any]:
    started = time.monotonic()
    config_path = Path(path)
    document = load_frontier_config(config_path)
    parent_config_path, chain_document, chain_result, retention = _validate_parent(
        document
    )
    rank, world_size, device = _distributed(chain_document)
    _seed(int(chain_document["training"]["seed"]))
    output_path = Path(document["outputs"]["result"])
    if output_path.is_file():
        result = json.loads(output_path.read_text()) if rank == 0 else None
        result = _broadcast(result, rank)
        dist.destroy_process_group()
        return result
    base_config = load_config(chain_document["parent"]["base_config"]["path"])
    target_counts = [int(value) for value in document["evaluation"]["target_counts"]]
    edge_documents, workloads = _workloads(chain_document, base_config, target_counts)
    reference_workload = workloads[target_counts[-1]][0]
    embedding_rows = int(reference_workload["metadata"]["embedding_rows"])
    dense, embedding, tracker, geometry = _initialize_model(
        chain_document, base_config, embedding_rows, rank, world_size, device
    )
    checkpoint_root = Path(retention["retained"]["theta1_theta10"]["path"])
    parent_hash = file_sha256(parent_config_path)
    _load_checkpoint(
        checkpoint_root,
        1,
        dense,
        embedding,
        tracker,
        chain_document,
        parent_hash,
        rank,
        False,
    )
    cells = []
    minimum = int(document["evaluation"]["minimum_target_version"])
    maximum = int(document["evaluation"]["maximum_target_version"])
    lineage = chain_document["lineage_selection"]
    accepted = chain_result["accepted_versions"]
    for target_version in range(minimum, maximum + 1):
        target_index = target_version - 1
        captures = {}
        for target_count in target_counts:
            batches = _evaluation_batches(
                workloads[target_count][target_index],
                int(chain_document["evaluation"]["local_batch_size"]),
                rank,
                world_size,
            )
            captures[target_count] = _capture_old(
                dense,
                embedding,
                batches,
                workloads[target_count][target_index],
                base_config,
                device,
            )
            del batches
        _load_checkpoint(
            checkpoint_root,
            target_version,
            dense,
            embedding,
            tracker,
            chain_document,
            parent_hash,
            rank,
            False,
        )
        evaluation_temperature = accepted[target_index]["candidate"].get(
            "evaluation_temperature"
        )
        for target_count in target_counts:
            edge_document = _temperature_edge_document(
                edge_documents[target_count][target_index], evaluation_temperature
            )
            compact, evaluation = _evaluate_captured(
                dense,
                embedding,
                captures.pop(target_count),
                workloads[target_count][target_index],
                edge_document,
                rank,
                world_size,
                device,
            )
            if rank == 0:
                if compact is None or evaluation is None:
                    raise RuntimeError("KuaiRand publication-frontier evaluation differs")
                holdout = _lineage_holdout_summary(
                    evaluation,
                    edge_document,
                    int(lineage["split_seed"]),
                    float(lineage["tuning_fraction"]),
                    int(lineage["tuning_bootstrap_samples"]),
                )
                cell = {
                    "source_version": target_version - 1,
                    "target_version": target_version,
                    "target_count": target_count,
                    "transition": chain_document["transitions"][target_index],
                    "workload": workloads[target_count][target_index]["metadata"],
                    "holdout": holdout,
                }
                cells.append(cell)
                stale = holdout["comparisons"]["recompute_over_reuse"]
                print(
                    f"phase=kuairand_adjacent_frontier target={target_version} "
                    f"targets_per_user={target_count} "
                    f"records={holdout['evaluation_records']} "
                    f"ndcg5={stale['ndcg_at_5']['relative_percent']:.3f}%",
                    flush=True,
                )
            del compact, evaluation
        gc.collect()
        torch.cuda.empty_cache()
    if rank == 0:
        summaries = {}
        for target_count in target_counts:
            selected = [cell for cell in cells if cell["target_count"] == target_count]
            values = [
                float(
                    cell["holdout"]["comparisons"]["recompute_over_reuse"][
                        "ndcg_at_5"
                    ]["relative_percent"]
                )
                for cell in selected
            ]
            update_values = [
                float(
                    cell["holdout"]["comparisons"]["fresh_update_value"][
                        "ndcg_at_5"
                    ]["relative_percent"]
                )
                for cell in selected
            ]
            summaries[str(target_count)] = {
                "edges": len(values),
                "positive_edges": sum(value > 0 for value in values),
                "above_one_percent_edges": sum(value >= 1 for value in values),
                "mean_relative_percent": float(np.mean(values)),
                "median_relative_percent": float(np.median(values)),
                "minimum_relative_percent": float(np.min(values)),
                "maximum_relative_percent": float(np.max(values)),
                "mean_fresh_update_relative_percent": float(np.mean(update_values)),
                "values_by_target_version": {
                    str(cell["target_version"]): value
                    for cell, value in zip(selected, values, strict=True)
                },
            }
        result = {
            "protocol": PROTOCOL,
            "round_id": document["round_id"],
            "status": "complete",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "parent": document["parent"],
            "geometry": geometry,
            "checkpoint_root": str(checkpoint_root),
            "cells": cells,
            "summaries": summaries,
            "decision": {
                "result_dependent_boundary": True,
                "next": "compare_frontier_before_training_revision",
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(output_path, result)
    else:
        result = None
    result = _broadcast(result, rank)
    dist.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = preflight(args.config) if args.preflight_only else run(args.config)
    print(json.dumps(result.get("summaries", result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
