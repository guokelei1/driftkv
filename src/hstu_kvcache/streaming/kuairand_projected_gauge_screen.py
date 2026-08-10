from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch.distributed as dist

from hstu_kvcache.models import apply_attention_coordinate_gauge_

from .kuairand_projected_persistent import (
    _build_workloads,
    _distributed,
    _evaluation_batches,
    _initialize_model,
    _load_checkpoint,
    load_persistent_config,
)
from .kuairand_projected_scale import _capture_old, _evaluate_captured
from .kuairand_query_transition import _atomic_json, _summary, file_sha256, load_config

PROTOCOL = "evokv_kuairand_projected_gauge_screen_v0"
EXPECTED_ANGLE_SETS = (
    [0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18],
    [0.0, 0.25, 0.35, 0.5, 0.7, 1.0],
)


def load_projected_gauge_screen_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    source = document.get("source", {})
    selection = document.get("selection", {})
    output = document.get("output")
    source_path = Path(source.get("config", {}).get("path", ""))
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or source.get("target_version") != 2
        or source.get("source_version") != 1
        or not source_path.is_file()
        or file_sha256(source_path) != source.get("config", {}).get("sha256")
        or document.get("angles_radians") not in EXPECTED_ANGLE_SETS
        or int(document.get("split_seed", 0)) < 1
        or float(document.get("tuning_fraction", 0.0)) != 0.25
        or selection
        != {
            "metrics": ["mrr", "ndcg_at_5"],
            "minimum_mean_relative_percent": 3.0,
            "maximum_mean_relative_percent": 15.0,
            "target_mean_relative_percent": 8.0,
        }
        or not isinstance(output, str)
    ):
        raise ValueError("KuaiRand projected gauge screen config differs")
    return document


def _user_partition(user_id: int, seed: int, tuning_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{user_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "little") / float(1 << 64)
    return "tuning" if value < tuning_fraction else "holdout"


def _partition_summary(
    evaluation: dict[str, Any],
    document: dict[str, Any],
    split_seed: int,
    tuning_fraction: float,
) -> dict[str, Any]:
    output = {}
    for split in ("tuning", "holdout"):
        records = [
            value
            for value in evaluation["records"]
            if _user_partition(int(value["user_id"]), split_seed, tuning_fraction)
            == split
        ]
        if not records:
            raise RuntimeError("KuaiRand projected gauge split is empty")
        output[split] = _summary(
            {"records": records, "sanity": evaluation["sanity"]},
            document,
        )
    return output


def _selection_value(value: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    stale = value["tuning"]["comparisons"]["recompute_over_reuse"]
    metrics = config["selection"]["metrics"]
    relative = [float(stale[metric]["relative_percent"]) for metric in metrics]
    mean = sum(relative) / len(relative)
    positive_ci = all(stale[metric]["positive_direction_with_ci"] for metric in metrics)
    minimum = float(config["selection"]["minimum_mean_relative_percent"])
    maximum = float(config["selection"]["maximum_mean_relative_percent"])
    return {
        "mean_relative_percent": mean,
        "relative_percent_by_metric": dict(zip(metrics, relative, strict=True)),
        "positive_ci_all_metrics": positive_ci,
        "inside_predeclared_band": bool(positive_ci and minimum <= mean <= maximum),
    }


def _select_angle(values: list[dict[str, Any]], config: dict[str, Any]) -> float | None:
    eligible = [value for value in values if value["selection"]["inside_predeclared_band"]]
    if not eligible:
        return None
    target = float(config["selection"]["target_mean_relative_percent"])
    selected = min(
        eligible,
        key=lambda value: (
            abs(value["selection"]["mean_relative_percent"] - target),
            value["angle_radians"],
        ),
    )
    return float(selected["angle_radians"])


def run_projected_gauge_screen(config_path: str | Path) -> dict[str, Any] | None:
    path = Path(config_path)
    config = load_projected_gauge_screen_config(path)
    output_path = Path(config["output"])
    if output_path.is_file():
        result = json.loads(output_path.read_text())
        return result if int(__import__("os").environ.get("RANK", "0")) == 0 else None
    source_path = Path(config["source"]["config"]["path"])
    document = load_persistent_config(source_path)
    document["config_path"] = str(source_path)
    config_sha256 = file_sha256(source_path)
    rank, world_size, device = _distributed(document)
    started = time.monotonic()
    try:
        base_config = load_config(document["parent"]["base_config"]["path"])
        edge_documents, workloads = _build_workloads(document, base_config, rank, 2)
        dense, embedding, tracker, geometry = _initialize_model(
            document,
            base_config,
            int(workloads[0]["metadata"]["embedding_rows"]),
            rank,
            world_size,
            device,
        )
        root = Path(document["outputs"]["checkpoint_root"])
        _load_checkpoint(
            root,
            1,
            dense,
            embedding,
            tracker,
            document,
            config_sha256,
            rank,
        )
        batches = _evaluation_batches(
            workloads[1],
            int(document["evaluation"]["local_batch_size"]),
            rank,
            world_size,
        )
        captured = _capture_old(
            dense,
            embedding,
            batches,
            workloads[1],
            base_config,
            device,
        )
        _load_checkpoint(
            root,
            2,
            dense,
            embedding,
            tracker,
            document,
            config_sha256,
            rank,
        )
        baseline_state = {
            name: value.detach().clone() for name, value in dense.state_dict().items()
        }
        values = []
        baseline_endpoints = None
        for angle in config["angles_radians"]:
            dense.load_state_dict(baseline_state)
            certificate = apply_attention_coordinate_gauge_(dense.core, float(angle))
            summary, evaluation = _evaluate_captured(
                dense,
                embedding,
                captured,
                workloads[1],
                edge_documents[1],
                rank,
                world_size,
                device,
            )
            if rank == 0:
                assert summary is not None and evaluation is not None
                partitions = _partition_summary(
                    evaluation,
                    edge_documents[1],
                    int(config["split_seed"]),
                    float(config["tuning_fraction"]),
                )
                fresh = summary["endpoints"]["recompute"]
                if baseline_endpoints is None:
                    baseline_endpoints = fresh
                maximum_fresh_difference = max(
                    abs(float(fresh[name]) - float(baseline_endpoints[name]))
                    for name in fresh
                )
                value = {
                    "angle_radians": float(angle),
                    "gauge_certificate": certificate,
                    "maximum_fresh_metric_difference_from_angle_zero": maximum_fresh_difference,
                    "all_users": summary,
                    **partitions,
                }
                value["selection"] = _selection_value(value, config)
                values.append(value)
                stale = partitions["holdout"]["comparisons"]["recompute_over_reuse"]
                print(
                    f"phase=kuairand_projected_gauge angle={angle:.3f} "
                    f"holdout_mrr={stale['mrr']['relative_percent']:.3f}% "
                    f"holdout_ndcg5={stale['ndcg_at_5']['relative_percent']:.3f}%",
                    flush=True,
                )
        if rank != 0:
            return None
        result = {
            "protocol": PROTOCOL,
            "status": "complete_development_screen",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(path), "sha256": file_sha256(path)},
            "source": config["source"],
            "geometry": geometry,
            "values": values,
            "selected_angle_radians": _select_angle(values, config),
            "selection_uses_holdout": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(output_path, result)
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
