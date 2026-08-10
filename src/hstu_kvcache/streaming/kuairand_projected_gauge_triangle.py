from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch.distributed as dist

from hstu_kvcache.models import (
    apply_attention_coordinate_gauge_,
    apply_attention_coordinate_scale_,
)

from .kuairand_projected_gauge_screen import _partition_summary
from .kuairand_projected_persistent import (
    _build_workloads,
    _distributed,
    _evaluation_batches,
    _initialize_model,
    _load_checkpoint,
    _reset_and_calibrate,
    load_persistent_config,
)
from .kuairand_projected_scale import _capture_old, _evaluate_captured
from .kuairand_query_transition import _atomic_json, file_sha256, load_config

PROTOCOL = "evokv_kuairand_projected_gauge_triangle_v0"
METRICS = ("candidate_cross_entropy", "mrr", "ndcg_at_5", "hit_rate_at_5")


def load_projected_gauge_triangle_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    source = document.get("source", {})
    outputs = document.get("outputs", {})
    config_path = Path(source.get("config", {}).get("path", ""))
    result_path = Path(source.get("result", {}).get("path", ""))
    transform = document.get("transform")
    prior = document.get("prior_triangle")
    step_radians = float(document.get("step_radians", -1.0))
    legacy_rotation = transform is None and math.isfinite(step_radians) and 0.0 < step_radians <= 1.0
    value_scale = (
        isinstance(transform, dict)
        and set(transform) == {"key_log_step", "mode", "value_log_step"}
        and transform.get("mode") == "value_log_scale"
        and math.isfinite(float(transform.get("key_log_step", float("nan"))))
        and math.isfinite(float(transform.get("value_log_step", float("nan"))))
        and 0.0 <= float(transform.get("key_log_step", -1.0)) <= 0.5
        and 0.0 <= float(transform.get("value_log_step", -1.0)) <= 0.5
        and (
            float(transform.get("key_log_step", 0.0)) > 0.0
            or float(transform.get("value_log_step", 0.0)) > 0.0
        )
    )
    final_version = int(document.get("final_version", 0))
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not (legacy_rotation or value_scale)
        or int(document.get("anchor_version", -1)) != 1
        or not 2 <= final_version <= 13
        or float(document.get("tuning_fraction", 0.0)) != 0.25
        or int(document.get("split_seed", 0)) < 1
        or not config_path.is_file()
        or file_sha256(config_path) != source.get("config", {}).get("sha256")
        or not result_path.is_file()
        or file_sha256(result_path) != source.get("result", {}).get("sha256")
        or not all(isinstance(outputs.get(name), str) for name in ("result", "table"))
    ):
        raise ValueError("KuaiRand projected gauge triangle config differs")
    source_result = json.loads(result_path.read_text())
    minimum_source_version = int(document.get("minimum_source_version", 0))
    if not 0 <= minimum_source_version < final_version:
        raise ValueError("KuaiRand projected gauge minimum source differs")
    for checkpoint in source_result.get("checkpoints", []):
        if int(checkpoint.get("version", -1)) < minimum_source_version:
            continue
        checkpoint_path = Path(checkpoint.get("path", ""))
        if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != checkpoint.get(
            "sha256"
        ):
            raise ValueError("KuaiRand projected gauge triangle checkpoint differs")
    if [value.get("version") for value in source_result.get("checkpoints", [])] != list(
        range(1, final_version + 1)
    ):
        raise ValueError("KuaiRand projected gauge triangle versions differ")
    if prior is not None:
        prior_path = Path(prior.get("path", "")) if isinstance(prior, dict) else Path()
        if (
            not isinstance(prior, dict)
            or not prior_path.is_file()
            or file_sha256(prior_path) != prior.get("sha256")
        ):
            raise ValueError("KuaiRand projected gauge prior differs")
        prior_result = json.loads(prior_path.read_text())
        prior_final_version = max(
            int(value["target_version"]) for value in prior_result.get("cells", [])
        )
        if (
            prior_result.get("protocol") != PROTOCOL
            or prior_result.get("status") != "complete_development_control"
            or prior_result.get("scientific_result") is not False
            or prior_result.get("formal_result") is not False
            or prior_result.get("transform")
            != (
                transform
                if transform is not None
                else {"mode": "orthogonal_rotation", "step_radians": step_radians}
            )
            or not prior_result.get("fresh_function_invariance", {}).get("passed")
            or prior_final_version != final_version - 1
        ):
            raise ValueError("KuaiRand projected gauge prior result differs")
    return document


def _angle(version: int, anchor: int, step: float) -> float:
    return max(0, version - anchor) * step


def _apply_transform(model, version: int, config: dict[str, Any]) -> dict[str, Any]:
    anchor = int(config["anchor_version"])
    transform = config.get("transform")
    if transform is None:
        return apply_attention_coordinate_gauge_(
            model,
            _angle(version, anchor, float(config["step_radians"])),
        )
    depth = max(0, version - anchor)
    return apply_attention_coordinate_scale_(
        model,
        key_log_scale=depth * float(transform["key_log_step"]),
        value_log_scale=depth * float(transform["value_log_step"]),
    )


def _transform_coordinate(version: int, config: dict[str, Any]) -> float:
    transform = config.get("transform")
    if transform is None:
        return _angle(
            version,
            int(config["anchor_version"]),
            float(config["step_radians"]),
        )
    return max(0, version - int(config["anchor_version"])) * float(transform["value_log_step"])


def _relative(cell: dict[str, Any], split: str, metric: str) -> float:
    return float(
        cell[split]["comparisons"]["recompute_over_reuse"][metric][
            "relative_percent"
        ]
    )


def _matrix(cells: list[dict[str, Any]], split: str, metric: str) -> list[list[float | None]]:
    by_pair = {
        (int(value["target_version"]), int(value["source_version"])): _relative(
            value, split, metric
        )
        for value in cells
    }
    final_version = max(int(value["target_version"]) for value in cells)
    return [
        [
            by_pair.get((target, source)) if source < target else None
            for source in range(final_version)
        ]
        for target in range(1, final_version + 1)
    ]


def _render_matrix(values: list[list[float | None]], metric: str) -> list[str]:
    headers = [f"theta{source}" for source in range(len(values))]
    output = [
        f"## Holdout {metric} relative Recompute-over-Reuse",
        "",
        "| current \\ cache | " + " | ".join(headers) + " |",
        "|---|" + "---:|" * len(headers),
    ]
    for target, row in enumerate(values, start=1):
        cells = ["—" if value is None else f"{value:+.3f}%" for value in row]
        output.append(f"| theta{target} | " + " | ".join(cells) + " |")
    return output


def _render_table(result: dict[str, Any]) -> str:
    lines = [
        "# KuaiRand function-preserving coordinate-drift control",
        "",
        "The normal checkpoints are unchanged. Each version receives a deterministic attention-coordinate gauge only while evaluating this control.",
        "",
    ]
    for metric in ("ndcg_at_5", "mrr", "hit_rate_at_5", "candidate_cross_entropy"):
        lines.extend(_render_matrix(result["matrices"]["holdout"][metric], metric))
        lines.append("")
    return "\n".join(lines)


def run_projected_gauge_triangle(config_path: str | Path) -> dict[str, Any] | None:
    path = Path(config_path)
    config = load_projected_gauge_triangle_config(path)
    result_path = Path(config["outputs"]["result"])
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        return result if int(__import__("os").environ.get("RANK", "0")) == 0 else None
    source_path = Path(config["source"]["config"]["path"])
    document = load_persistent_config(source_path)
    document["config_path"] = str(source_path)
    config_sha256 = file_sha256(source_path)
    source_result = json.loads(Path(config["source"]["result"]["path"]).read_text())
    prior_result = (
        json.loads(Path(config["prior_triangle"]["path"]).read_text())
        if "prior_triangle" in config
        else None
    )
    rank, world_size, device = _distributed(document)
    started = time.monotonic()
    try:
        base_config = load_config(document["parent"]["base_config"]["path"])
        final_version = int(config["final_version"])
        edge_documents, workloads = _build_workloads(
            document, base_config, rank, final_version
        )
        dense, embedding, tracker, geometry = _initialize_model(
            document,
            base_config,
            int(workloads[0]["metadata"]["embedding_rows"]),
            rank,
            world_size,
            device,
        )
        root = Path(document["outputs"]["checkpoint_root"])
        first_target_version = (
            max(int(value["target_version"]) for value in prior_result["cells"]) + 1
            if prior_result is not None
            else 1
        )
        minimum_source_version = int(config.get("minimum_source_version", 0))
        captures: list[dict[int, list[dict[str, Any]]]] = [
            dict() for _ in range(final_version)
        ]
        certificates = (
            dict(prior_result.get("gauge_certificates", {}))
            if prior_result is not None
            else {}
        )
        for source_version in range(minimum_source_version, final_version):
            if source_version == 0:
                _reset_and_calibrate(
                    dense,
                    embedding,
                    tracker,
                    workloads,
                    base_config,
                    document,
                    rank,
                    world_size,
                    device,
                )
            else:
                _load_checkpoint(
                    root,
                    source_version,
                    dense,
                    embedding,
                    tracker,
                    document,
                    config_sha256,
                    rank,
                )
            source_angle = _transform_coordinate(source_version, config)
            certificates[str(source_version)] = _apply_transform(
                dense.core, source_version, config
            )
            for target_index in range(
                max(source_version, first_target_version - 1), final_version
            ):
                batches = _evaluation_batches(
                    workloads[target_index],
                    int(document["evaluation"]["local_batch_size"]),
                    rank,
                    world_size,
                )
                captures[target_index][source_version] = _capture_old(
                    dense,
                    embedding,
                    batches,
                    workloads[target_index],
                    base_config,
                    device,
                )
            if rank == 0:
                print(
                    f"phase=kuairand_gauge_triangle_capture source={source_version} "
                    f"angle={source_angle:.3f}",
                    flush=True,
                )
        cells = list(prior_result["cells"]) if prior_result is not None else []
        maximum_fresh_difference = (
            float(
                prior_result["fresh_function_invariance"]["maximum_metric_difference"]
            )
            if prior_result is not None
            else 0.0
        )
        for target_version in range(first_target_version, final_version + 1):
            target_index = target_version - 1
            _load_checkpoint(
                root,
                target_version,
                dense,
                embedding,
                tracker,
                document,
                config_sha256,
                rank,
            )
            target_angle = _transform_coordinate(target_version, config)
            _apply_transform(dense.core, target_version, config)
            expected_fresh = source_result["targets"][target_index]["lineage"][0][
                "summary"
            ]["endpoints"]["recompute"]
            for source_version, captured in sorted(captures[target_index].items()):
                summary, evaluation = _evaluate_captured(
                    dense,
                    embedding,
                    captured,
                    workloads[target_index],
                    edge_documents[target_index],
                    rank,
                    world_size,
                    device,
                )
                if rank == 0:
                    assert summary is not None and evaluation is not None
                    fresh_difference = max(
                        abs(
                            float(summary["endpoints"]["recompute"][metric])
                            - float(expected_fresh[metric])
                        )
                        for metric in expected_fresh
                    )
                    maximum_fresh_difference = max(maximum_fresh_difference, fresh_difference)
                    partitions = _partition_summary(
                        evaluation,
                        edge_documents[target_index],
                        int(config["split_seed"]),
                        float(config["tuning_fraction"]),
                    )
                    cell = {
                        "target_version": target_version,
                        "source_version": source_version,
                        "cache_age": target_version - source_version,
                        "target_angle_radians": target_angle,
                        "source_angle_radians": _transform_coordinate(
                            source_version, config
                        ),
                        "maximum_fresh_metric_difference": fresh_difference,
                        "all_users": summary,
                        **partitions,
                    }
                    cells.append(cell)
                    print(
                        f"phase=kuairand_gauge_triangle target={target_version} "
                        f"source={source_version} "
                        f"holdout_mrr={_relative(cell, 'holdout', 'mrr'):.3f}% "
                        f"holdout_ndcg5={_relative(cell, 'holdout', 'ndcg_at_5'):.3f}%",
                        flush=True,
                    )
        if rank != 0:
            return None
        matrices = {
            split: {metric: _matrix(cells, split, metric) for metric in METRICS}
            for split in ("all_users", "tuning", "holdout")
        }
        ordinary = [
            value
            for value in cells
            if value["target_version"] >= 2 and value["source_version"] >= 1
        ]
        decision = {
            "ordinary_cells": len(ordinary),
            "positive_holdout_mrr_cells": sum(
                _relative(value, "holdout", "mrr") > 0 for value in ordinary
            ),
            "positive_holdout_ndcg5_cells": sum(
                _relative(value, "holdout", "ndcg_at_5") > 0 for value in ordinary
            ),
            "positive_holdout_ce_cells": sum(
                _relative(value, "holdout", "candidate_cross_entropy") > 0
                for value in ordinary
            ),
            "all_ordinary_holdout_mrr_positive": all(
                _relative(value, "holdout", "mrr") > 0 for value in ordinary
            ),
            "all_ordinary_holdout_ndcg5_positive": all(
                _relative(value, "holdout", "ndcg_at_5") > 0 for value in ordinary
            ),
        }
        result = {
            "protocol": PROTOCOL,
            "status": "complete_development_control",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(path), "sha256": file_sha256(path)},
            "source": config["source"],
            "incremental_extension": {
                "prior_triangle": config.get("prior_triangle"),
                "first_target_version": first_target_version,
                "minimum_source_version": minimum_source_version,
            },
            "geometry": geometry,
            "transform": config["transform"]
            if "transform" in config
            else {
                "mode": "orthogonal_rotation",
                "step_radians": config["step_radians"],
            },
            "fresh_function_invariance": {
                "maximum_metric_difference": maximum_fresh_difference,
                "passed": maximum_fresh_difference <= 1e-6,
            },
            "gauge_certificates": certificates,
            "cells": cells,
            "matrices": matrices,
            "decision": decision,
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(result_path, result)
        table_path = Path(config["outputs"]["table"])
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text(_render_table(result))
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
