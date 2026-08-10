from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch.distributed as dist

from hstu_kvcache.models import apply_attention_coordinate_scale_

from .kuairand_projected_gauge_screen import _user_partition
from .kuairand_projected_persistent import (
    _distributed,
    _evaluation_batches,
    _initialize_model,
    _load_checkpoint,
    load_persistent_config,
)
from .kuairand_projected_scale import _capture_old, _evaluate_captured
from .kuairand_query_multiversion import _edge_config
from .kuairand_query_transition import (
    _atomic_json,
    _summary,
    build_workload,
    file_sha256,
    load_config,
)

PROTOCOL = "evokv_kuairand_stationary_coordinate_control_v0"
FIDELITY_METRICS = (
    "hidden_cosine",
    "score_cosine",
    "score_relative_error",
    "score_kl_from_fresh",
    "top10_overlap_with_fresh",
)
RANKING_METRICS = ("mrr", "ndcg_at_5", "hit_rate_at_5")


def load_stationary_coordinate_control_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    source = document.get("source")
    evaluation = document.get("evaluation")
    selection = document.get("selection")
    outputs = document.get("outputs")
    transforms = document.get("transform_candidates")
    config_path = Path(source.get("config", {}).get("path", "")) if isinstance(source, dict) else Path()
    manifest_path = Path(source.get("anchor_manifest", {}).get("path", "")) if isinstance(source, dict) else Path()
    accepted_path = Path(source.get("anchor_accepted", {}).get("path", "")) if isinstance(source, dict) else Path()
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not all(isinstance(value, dict) for value in (source, evaluation, selection, outputs))
        or not config_path.is_file()
        or file_sha256(config_path) != source.get("config", {}).get("sha256")
        or not manifest_path.is_file()
        or file_sha256(manifest_path) != source.get("anchor_manifest", {}).get("sha256")
        or not accepted_path.is_file()
        or file_sha256(accepted_path) != source.get("anchor_accepted", {}).get("sha256")
        or int(source.get("anchor_version", 0)) < 1
        or int(evaluation.get("workload_transition_version", 0)) < 2
        or int(evaluation.get("virtual_versions", 0)) != 8
        or int(evaluation.get("candidate_count", 0)) != 100
        or int(evaluation.get("targets_per_user", 0)) != 8
        or float(evaluation.get("tuning_fraction", 0.0)) != 0.25
        or int(evaluation.get("split_seed", 0)) < 1
        or selection
        != {
            "metric": "top10_changed_fraction",
            "minimum": 0.05,
            "maximum": 0.15,
            "target": 0.1,
        }
        or not isinstance(transforms, list)
        or len(transforms) < 3
        or not all(isinstance(outputs.get(name), str) for name in ("screen", "result", "table"))
    ):
        raise ValueError("KuaiRand stationary coordinate-control config differs")
    names = []
    coordinates = []
    for transform in transforms:
        if not isinstance(transform, dict) or set(transform) != {
            "key_log_step",
            "name",
            "value_log_step",
        }:
            raise ValueError("KuaiRand stationary coordinate transform differs")
        key_step = float(transform["key_log_step"])
        value_step = float(transform["value_log_step"])
        if (
            not isinstance(transform["name"], str)
            or not math.isfinite(key_step)
            or not math.isfinite(value_step)
            or not 0.0 <= key_step <= 0.5
            or not 0.0 < value_step <= 1.2
        ):
            raise ValueError("KuaiRand stationary coordinate transform differs")
        names.append(transform["name"])
        coordinates.append((key_step, value_step))
    if len(names) != len(set(names)) or coordinates != sorted(set(coordinates)):
        raise ValueError("KuaiRand stationary coordinate transform ordering differs")
    source_document = load_persistent_config(config_path)
    anchor_version = int(source["anchor_version"])
    workload_version = int(evaluation["workload_transition_version"])
    if (
        anchor_version >= workload_version
        or workload_version > int(source_document["checkpoint"]["versions"])
        or int(json.loads(manifest_path.read_text()).get("version", -1)) != anchor_version
        or int(json.loads(accepted_path.read_text()).get("version", -1)) != anchor_version
    ):
        raise ValueError("KuaiRand stationary coordinate source boundary differs")
    return document


def _records_for_split(
    evaluation: dict[str, Any], split: str, split_seed: int, tuning_fraction: float
) -> list[dict[str, Any]]:
    records = [
        record
        for record in evaluation["records"]
        if _user_partition(int(record["user_id"]), split_seed, tuning_fraction) == split
    ]
    if not records:
        raise RuntimeError("KuaiRand stationary coordinate split is empty")
    return records


def _fidelity_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for record in records:
        grouped[int(record["user_id"])].append(record["fidelity"])
    metrics = {}
    for metric in FIDELITY_METRICS:
        values = np.asarray(
            [
                statistics.fmean(float(record[metric]) for record in user_records)
                for user_records in grouped.values()
            ],
            dtype=np.float64,
        )
        metrics[metric] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
        }
    overlap = metrics["top10_overlap_with_fresh"]["mean"]
    return {
        "records": len(records),
        "users": len(grouped),
        "user_mean": metrics,
        "top10_changed_fraction": 1.0 - overlap,
    }


def _ranking_summary(
    records: list[dict[str, Any]], edge_document: dict[str, Any], sanity: dict[str, Any]
) -> dict[str, Any]:
    return _summary({"records": records, "sanity": sanity}, edge_document)


def _select_transform(
    values: list[dict[str, Any]], selection: dict[str, Any]
) -> dict[str, Any]:
    minimum = float(selection["minimum"])
    maximum = float(selection["maximum"])
    target = float(selection["target"])
    eligible = [
        value
        for value in values
        if minimum
        <= float(value["tuning_fidelity"]["top10_changed_fraction"])
        <= maximum
        and float(value["maximum_fresh_metric_difference"]) <= 1e-6
    ]
    candidates = eligible if eligible else values
    selected = min(
        candidates,
        key=lambda value: (
            abs(float(value["tuning_fidelity"]["top10_changed_fraction"]) - target),
            float(value["transform"]["value_log_step"]),
            float(value["transform"]["key_log_step"]),
        ),
    )
    return {
        "name": selected["transform"]["name"],
        "inside_predeclared_band": selected in eligible,
        "observed_tuning_top10_changed_fraction": selected["tuning_fidelity"][
            "top10_changed_fraction"
        ],
        "transform": selected["transform"],
    }


def _render_matrix(
    matrix: list[list[float | None]], title: str, percent: bool = True
) -> list[str]:
    versions = len(matrix)
    lines = [
        f"## {title}",
        "",
        "| current \\ cache | " + " | ".join(f"v{value}" for value in range(1, versions + 1)) + " |",
        "|---|" + "---:|" * versions,
    ]
    for target, row in enumerate(matrix, start=1):
        rendered = []
        for value in row:
            if value is None:
                rendered.append("—")
            elif percent:
                rendered.append(f"{value:+.2f}%")
            else:
                rendered.append(f"{value:.6f}")
        lines.append(f"| v{target} | " + " | ".join(rendered) + " |")
    return lines


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# KuaiRand stationary coordinate-drift causal control",
        "",
        "All eight logical versions share one trained 45.3-GiB anchor function. Only the attention K/V coordinate system changes, with an exact inverse on Q/output. Fresh invariance is mandatory; labels do not select the transform.",
        "",
    ]
    for metric, label in (
        ("holdout_ndcg_loss_percent", "NDCG@5"),
        ("holdout_mrr_loss_percent", "MRR"),
        ("holdout_hr5_loss_percent", "HR@5"),
    ):
        lines.extend(
            _render_matrix(
                result["matrices"][metric],
                f"Holdout {label} loss: 100 × (Fresh − Reuse) / Fresh",
            )
        )
        lines.append("")
    lines.extend(
        _render_matrix(
            result["matrices"]["holdout_top10_changed_percent"],
            "Label-free Top-10 recommendation change versus Fresh",
        )
    )
    lines.append("")
    lines.extend(
        _render_matrix(
            result["matrices"]["holdout_score_cosine_loss_percent"],
            "Label-free score cosine loss versus Fresh",
        )
    )
    lines.append("")
    lines.extend(
        _render_matrix(
            result["matrices"]["holdout_hidden_relative_error_percent"],
            "Label-free hidden relative error versus Fresh",
        )
    )
    lines.extend(
        [
            "",
            "## Fresh absolute endpoint",
            "",
            "| MRR | NDCG@5 | HR@5 |",
            "|---:|---:|---:|",
            "| {mrr:.6f} | {ndcg:.6f} | {hr:.6f} |".format(
                mrr=result["fresh_endpoint"]["mrr"],
                ndcg=result["fresh_endpoint"]["ndcg_at_5"],
                hr=result["fresh_endpoint"]["hit_rate_at_5"],
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _apply_transform(model, depth: int, transform: dict[str, Any]) -> dict[str, Any]:
    return apply_attention_coordinate_scale_(
        model,
        key_log_scale=depth * float(transform["key_log_step"]),
        value_log_scale=depth * float(transform["value_log_step"]),
    )


def _broadcast(value: Any, rank: int) -> Any:
    if not dist.is_initialized():
        return value
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _matrix_from_derived(
    cells: list[dict[str, Any]], metric: str, versions: int
) -> list[list[float | None]]:
    values = {
        (int(cell["target_version"]), int(cell["source_version"])): float(
            cell["holdout"]["derived"][metric]
        )
        for cell in cells
    }
    return [
        [
            0.0
            if source == target
            else values[(target, source)]
            if source < target
            else None
            for source in range(1, versions + 1)
        ]
        for target in range(1, versions + 1)
    ]


def _age_summary(cells: list[dict[str, Any]], versions: int) -> list[dict[str, Any]]:
    output = []
    for age in range(1, versions):
        selected = [cell for cell in cells if int(cell["cache_age"]) == age]
        output.append(
            {
                "cache_age": age,
                "cells": len(selected),
                **{
                    metric: statistics.fmean(
                        float(cell["holdout"]["derived"][metric]) for cell in selected
                    )
                    for metric in (
                        "mrr_loss_percent",
                        "ndcg_at_5_loss_percent",
                        "hit_rate_at_5_loss_percent",
                        "top10_changed_percent",
                        "score_cosine_loss_percent",
                        "hidden_relative_error_percent",
                    )
                },
            }
        )
    return output


def render_stationary_coordinate_control(result_path: str | Path) -> dict[str, Any]:
    path = Path(result_path)
    result = json.loads(path.read_text())
    if result.get("status") != "complete_development_causal_control":
        raise ValueError("KuaiRand stationary coordinate result differs")
    versions = int(result["config"].get("virtual_versions", 0))
    if versions == 0:
        versions = max(int(cell["target_version"]) for cell in result["cells"])
    cells = result["cells"]
    result["matrices"] = {
        "holdout_ndcg_loss_percent": _matrix_from_derived(
            cells, "ndcg_at_5_loss_percent", versions
        ),
        "holdout_mrr_loss_percent": _matrix_from_derived(
            cells, "mrr_loss_percent", versions
        ),
        "holdout_hr5_loss_percent": _matrix_from_derived(
            cells, "hit_rate_at_5_loss_percent", versions
        ),
        "holdout_top10_changed_percent": _matrix_from_derived(
            cells, "top10_changed_percent", versions
        ),
        "holdout_score_cosine_loss_percent": _matrix_from_derived(
            cells, "score_cosine_loss_percent", versions
        ),
        "holdout_hidden_relative_error_percent": _matrix_from_derived(
            cells, "hidden_relative_error_percent", versions
        ),
    }
    result["age_summary"] = _age_summary(cells, versions)
    _atomic_json(path, result)
    config = load_stationary_coordinate_control_config(result["config"]["path"])
    table_path = Path(config["outputs"]["table"])
    table_path.write_text(_render(result))
    return result


def run_stationary_coordinate_control(config_path: str | Path) -> dict[str, Any] | None:
    path = Path(config_path)
    config = load_stationary_coordinate_control_config(path)
    result_path = Path(config["outputs"]["result"])
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        return result if int(os.environ.get("RANK", "0")) == 0 else None
    source_path = Path(config["source"]["config"]["path"])
    document = load_persistent_config(source_path)
    document["config_path"] = str(source_path)
    config_sha256 = file_sha256(source_path)
    rank, world_size, device = _distributed(document)
    started = time.monotonic()
    try:
        base_config = load_config(document["parent"]["base_config"]["path"])
        workload_version = int(config["evaluation"]["workload_transition_version"])
        transition = document["transitions"][workload_version - 1]
        edge_document = _edge_config(base_config, transition, 1.0)
        edge_document["data"]["update_dates"] = transition["update_dates"]
        edge_document["data"]["evaluation_targets_per_user"] = int(
            config["evaluation"]["targets_per_user"]
        )
        edge_document["data"]["user_limit"] = document["data"].get("user_limit")
        edge_document["evaluation"]["candidate_count"] = int(
            config["evaluation"]["candidate_count"]
        )
        workload = build_workload(edge_document)
        dense, embedding, tracker, geometry = _initialize_model(
            document,
            base_config,
            int(workload["metadata"]["embedding_rows"]),
            rank,
            world_size,
            device,
        )
        root = Path(document["outputs"]["checkpoint_root"])
        _load_checkpoint(
            root,
            int(config["source"]["anchor_version"]),
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
        batches = _evaluation_batches(
            workload,
            int(document["evaluation"]["local_batch_size"]),
            rank,
            world_size,
        )
        source_zero = _capture_old(
            dense, embedding, batches, workload, base_config, device
        )
        baseline_summary, baseline_evaluation = _evaluate_captured(
            dense,
            embedding,
            source_zero,
            workload,
            edge_document,
            rank,
            world_size,
            device,
        )
        baseline_fresh = (
            baseline_summary["endpoints"]["recompute"] if rank == 0 else None
        )
        split_seed = int(config["evaluation"]["split_seed"])
        tuning_fraction = float(config["evaluation"]["tuning_fraction"])
        screen_values = []
        for transform in config["transform_candidates"]:
            dense.load_state_dict(baseline_state)
            certificate = _apply_transform(dense.core, 1, transform)
            summary, evaluation = _evaluate_captured(
                dense,
                embedding,
                source_zero,
                workload,
                edge_document,
                rank,
                world_size,
                device,
            )
            if rank == 0:
                assert summary is not None and evaluation is not None and baseline_fresh is not None
                tuning_records = _records_for_split(
                    evaluation, "tuning", split_seed, tuning_fraction
                )
                difference = max(
                    abs(float(summary["endpoints"]["recompute"][metric]) - float(baseline_fresh[metric]))
                    for metric in baseline_fresh
                )
                value = {
                    "transform": transform,
                    "certificate": certificate,
                    "maximum_fresh_metric_difference": difference,
                    "tuning_fidelity": _fidelity_summary(tuning_records),
                }
                screen_values.append(value)
                print(
                    f"phase=kuairand_stationary_screen candidate={transform['name']} "
                    f"top10_changed={value['tuning_fidelity']['top10_changed_fraction']:.6f} "
                    f"score_cosine={value['tuning_fidelity']['user_mean']['score_cosine']['mean']:.6f}",
                    flush=True,
                )
        if rank == 0:
            selected = _select_transform(screen_values, config["selection"])
            screen_result = {
                "protocol": PROTOCOL,
                "status": "complete_label_free_tuning",
                "scientific_result": False,
                "formal_result": False,
                "config": {"path": str(path), "sha256": file_sha256(path)},
                "selection": config["selection"],
                "values": screen_values,
                "selected": selected,
            }
            _atomic_json(Path(config["outputs"]["screen"]), screen_result)
        else:
            selected = None
        selected = _broadcast(selected, rank)
        transform = selected["transform"]
        versions = int(config["evaluation"]["virtual_versions"])
        captures = {0: source_zero}
        dense.load_state_dict(baseline_state)
        certificates = {"0": _apply_transform(dense.core, 0, transform)}
        for source_depth in range(1, versions - 1):
            dense.load_state_dict(baseline_state)
            certificates[str(source_depth)] = _apply_transform(
                dense.core, source_depth, transform
            )
            captures[source_depth] = _capture_old(
                dense, embedding, batches, workload, base_config, device
            )
            if rank == 0:
                print(
                    f"phase=kuairand_stationary_capture source=v{source_depth + 1}",
                    flush=True,
                )
        cells = []
        maximum_fresh_difference = 0.0
        for target_depth in range(1, versions):
            dense.load_state_dict(baseline_state)
            certificates[str(target_depth)] = _apply_transform(
                dense.core, target_depth, transform
            )
            for source_depth in range(target_depth):
                summary, evaluation = _evaluate_captured(
                    dense,
                    embedding,
                    captures[source_depth],
                    workload,
                    edge_document,
                    rank,
                    world_size,
                    device,
                )
                if rank == 0:
                    assert summary is not None and evaluation is not None and baseline_fresh is not None
                    difference = max(
                        abs(float(summary["endpoints"]["recompute"][metric]) - float(baseline_fresh[metric]))
                        for metric in baseline_fresh
                    )
                    maximum_fresh_difference = max(maximum_fresh_difference, difference)
                    partitions = {}
                    for split in ("tuning", "holdout"):
                        records = _records_for_split(
                            evaluation, split, split_seed, tuning_fraction
                        )
                        ranking = _ranking_summary(records, edge_document, evaluation["sanity"])
                        fresh = ranking["endpoints"]["recompute"]
                        reuse = ranking["endpoints"]["reuse"]
                        partitions[split] = {
                            "ranking": ranking,
                            "fidelity": _fidelity_summary(records),
                            "derived": {
                                "mrr_loss_percent": 100.0 * (fresh["mrr"] - reuse["mrr"]) / fresh["mrr"],
                                "ndcg_at_5_loss_percent": 100.0 * (fresh["ndcg_at_5"] - reuse["ndcg_at_5"]) / fresh["ndcg_at_5"],
                                "hit_rate_at_5_loss_percent": 100.0 * (fresh["hit_rate_at_5"] - reuse["hit_rate_at_5"]) / fresh["hit_rate_at_5"],
                            },
                        }
                        partitions[split]["derived"]["top10_changed_percent"] = 100.0 * partitions[split]["fidelity"]["top10_changed_fraction"]
                        partitions[split]["derived"]["score_cosine_loss_percent"] = 100.0 * (
                            1.0 - partitions[split]["fidelity"]["user_mean"]["score_cosine"]["mean"]
                        )
                        partitions[split]["derived"]["hidden_relative_error_percent"] = 100.0 * statistics.fmean(
                            float(record["hidden_relative_error"]) for record in records
                        )
                    cell = {
                        "target_version": target_depth + 1,
                        "source_version": source_depth + 1,
                        "cache_age": target_depth - source_depth,
                        "maximum_fresh_metric_difference": difference,
                        **partitions,
                    }
                    cells.append(cell)
                    print(
                        f"phase=kuairand_stationary_cell target=v{target_depth + 1} "
                        f"source=v{source_depth + 1} "
                        f"holdout_ndcg={cell['holdout']['derived']['ndcg_at_5_loss_percent']:.3f}% "
                        f"top10_changed={cell['holdout']['derived']['top10_changed_percent']:.3f}%",
                        flush=True,
                    )
        if rank != 0:
            return None
        holdout_cells = [cell["holdout"]["derived"] for cell in cells]
        matrices = {
            "holdout_ndcg_loss_percent": _matrix_from_derived(
                cells, "ndcg_at_5_loss_percent", versions
            ),
            "holdout_mrr_loss_percent": _matrix_from_derived(
                cells, "mrr_loss_percent", versions
            ),
            "holdout_hr5_loss_percent": _matrix_from_derived(
                cells, "hit_rate_at_5_loss_percent", versions
            ),
            "holdout_top10_changed_percent": _matrix_from_derived(
                cells, "top10_changed_percent", versions
            ),
            "holdout_score_cosine_loss_percent": _matrix_from_derived(
                cells, "score_cosine_loss_percent", versions
            ),
            "holdout_hidden_relative_error_percent": _matrix_from_derived(
                cells, "hidden_relative_error_percent", versions
            ),
        }
        holdout_records = _records_for_split(
            baseline_evaluation, "holdout", split_seed, tuning_fraction
        )
        fresh_endpoint = _ranking_summary(
            holdout_records, edge_document, baseline_evaluation["sanity"]
        )["endpoints"]["recompute"]
        result = {
            "protocol": PROTOCOL,
            "status": "complete_development_causal_control",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(path), "sha256": file_sha256(path)},
            "source": config["source"],
            "geometry": geometry,
            "selected": selected,
            "fresh_function_invariance": {
                "maximum_metric_difference": maximum_fresh_difference,
                "passed": maximum_fresh_difference <= 1e-6,
            },
            "cells": cells,
            "matrices": matrices,
            "age_summary": _age_summary(cells, versions),
            "fresh_endpoint": fresh_endpoint,
            "decision": {
                "ordinary_cells": len(cells),
                "positive_holdout_ndcg_cells": sum(
                    value["ndcg_at_5_loss_percent"] > 0.0 for value in holdout_cells
                ),
                "mean_adjacent_holdout_ndcg_percent": statistics.fmean(
                    cell["holdout"]["derived"]["ndcg_at_5_loss_percent"]
                    for cell in cells
                    if int(cell["cache_age"]) == 1
                ),
                "label_free_top10_change_increases_with_age": all(
                    statistics.fmean(
                        cell["holdout"]["derived"]["top10_changed_percent"]
                        for cell in cells
                        if int(cell["cache_age"]) == age
                    )
                    < statistics.fmean(
                        cell["holdout"]["derived"]["top10_changed_percent"]
                        for cell in cells
                        if int(cell["cache_age"]) == age + 1
                    )
                    for age in range(1, versions - 1)
                ),
            },
            "gauge_certificates": certificates,
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(result_path, result)
        table_path = Path(config["outputs"]["table"])
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text(_render(result))
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
