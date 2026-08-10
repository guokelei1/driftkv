from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from hstu_kvcache.models import apply_attention_coordinate_scale_

from .kuairand_projected_gauge_screen import _user_partition
from .kuairand_projected_persistent import (
    CHECKPOINT_SCHEMA,
    PROTOCOL,
    _accepted_path,
    _build_workloads,
    _candidate_path,
    _completed_prefix,
    _disk_preflight,
    _distributed,
    _evaluation_batches,
    _global_new_rows,
    _initialize_model,
    _load_checkpoint,
    _passing,
    _save_checkpoint,
    _train_candidate,
    _training_document,
    load_persistent_config,
)
from .kuairand_projected_scale import _capture_old, _evaluate_captured
from .kuairand_query_transition import _atomic_json, file_sha256, load_config

LINEAGE_PROTOCOL = "evokv_kuairand_lineage_retrain_v0"


def _coordinate_drift_config(document: dict[str, Any]) -> dict[str, Any] | None:
    value = document.get("coordinate_drift")
    if value is None:
        return None
    calibration = value.get("calibration") if isinstance(value, dict) else None
    calibration_path = (
        Path(calibration.get("path", "")) if isinstance(calibration, dict) else Path()
    )
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "calibration",
            "canonicalize_before_update",
            "key_log_step",
            "mode",
            "selection_basis",
            "value_log_step",
        }
        or value.get("mode") != "cumulative_function_preserving_scale"
        or value.get("canonicalize_before_update") is not True
        or value.get("selection_basis")
        not in (
            "label_free_top10_change_band",
            "label_free_top10_change_target_extrapolation",
        )
        or not 0.0 < float(value.get("key_log_step", 0.0)) <= 0.75
        or not 0.0 < float(value.get("value_log_step", 0.0)) <= 2.0
        or not calibration_path.is_file()
        or file_sha256(calibration_path) != calibration.get("sha256")
    ):
        raise ValueError("KuaiRand lineage coordinate-drift config differs")
    return value


def _apply_coordinate_depth(
    dense,
    coordinate: dict[str, Any] | None,
    depth: int,
    direction: int = 1,
) -> dict[str, Any] | None:
    if coordinate is None:
        return None
    if depth < 0 or direction not in (-1, 1):
        raise ValueError("KuaiRand lineage coordinate depth differs")
    model = dense.core if hasattr(dense, "core") else dense
    return apply_attention_coordinate_scale_(
        model,
        key_log_scale=direction * depth * float(coordinate["key_log_step"]),
        value_log_scale=direction * depth * float(coordinate["value_log_step"]),
    )


def _lineage_workload_limit(
    document: dict[str, Any], stop_after_version: int | None = None
) -> int:
    configured = int(document["checkpoint"]["versions"])
    return configured if stop_after_version is None else stop_after_version


def load_lineage_retrain_config(path: str | Path) -> dict[str, Any]:
    document = load_persistent_config(path)
    _coordinate_drift_config(document)
    selection = document.get("lineage_selection", {})
    checkpoint = document["checkpoint"]
    imported_prefix_versions = int(checkpoint.get("imported_prefix_versions", 0))
    expected_versions = list(range(imported_prefix_versions + 1, int(checkpoint["versions"]) + 1))
    candidate_names = [value["name"] for value in document["training"]["candidate_ladder"]]
    primary_metric = selection.get("primary_metric")
    positive_ranking_metrics = selection.get("positive_ranking_metrics")
    lineage_mean_metrics = selection.get("lineage_mean_metrics")
    ranking_metrics = {"mrr", "ndcg_at_5", "hit_rate_at_5"}
    minimum_fresh_metrics = selection.get("minimum_fresh_metrics", {})
    require_candidate_ce_positive = selection.get("require_candidate_ce_positive", True)
    require_fresh_update_ranking_positive = selection.get(
        "require_fresh_update_ranking_positive", True
    )
    minimum_lineage_positive_fraction = float(
        selection.get("minimum_lineage_positive_fraction", 1.0)
    )
    maximum_lineage_relative_percent = float(
        selection.get("maximum_lineage_relative_percent", float("inf"))
    )
    minimum_source_version = int(selection.get("minimum_source_version", 1))
    if (
        not expected_versions
        or selection.get("versions") != expected_versions
        or not 0.0 < float(selection.get("tuning_fraction", 0.0)) < 1.0
        or int(selection.get("split_seed", 0)) < 1
        or int(selection.get("tuning_bootstrap_samples", 0)) != 200
        or float(selection.get("minimum_adjacent_relative_percent", 0.0)) < 0.5
        or float(selection.get("minimum_lineage_mean_relative_percent", 0.0)) < 1.0
        or not 0.75 <= minimum_lineage_positive_fraction <= 1.0
        or maximum_lineage_relative_percent < 10.0
        or not 1 <= minimum_source_version <= imported_prefix_versions
        or selection.get("candidate_order") != candidate_names
        or not isinstance(minimum_fresh_metrics, dict)
        or not set(minimum_fresh_metrics).issubset({"mrr", "ndcg_at_5", "hit_rate_at_5"})
        or any(not 0.0 <= float(value) <= 1.0 for value in minimum_fresh_metrics.values())
        or not isinstance(require_candidate_ce_positive, bool)
        or not isinstance(require_fresh_update_ranking_positive, bool)
        or (
            primary_metric is not None
            and (
                primary_metric != "ndcg_at_5"
                or not isinstance(positive_ranking_metrics, list)
                or not positive_ranking_metrics
                or len(positive_ranking_metrics) != len(set(positive_ranking_metrics))
                or primary_metric not in positive_ranking_metrics
                or not set(positive_ranking_metrics).issubset(ranking_metrics)
                or (
                    lineage_mean_metrics is not None
                    and (
                        not isinstance(lineage_mean_metrics, list)
                        or not lineage_mean_metrics
                        or not set(lineage_mean_metrics).issubset(set(positive_ranking_metrics))
                    )
                )
            )
        )
    ):
        raise ValueError("KuaiRand lineage retrain config differs")
    return document


def _subset_workload(
    workload: dict[str, Any], split: str, seed: int, fraction: float
) -> dict[str, Any]:
    keys = [
        key
        for key in workload["evaluation_keys"]
        if _user_partition(int(workload["evaluation"][key]["user_id"]), seed, fraction) == split
    ]
    if not keys:
        raise RuntimeError("KuaiRand lineage retrain split is empty")
    output = workload.copy()
    output["evaluation_keys"] = keys
    output["evaluation"] = {key: workload["evaluation"][key] for key in keys}
    output["candidate_maps"] = {key: workload["candidate_maps"][key] for key in keys}
    output["metadata"] = copy.deepcopy(workload["metadata"])
    output["metadata"]["evaluation_records"] = len(keys)
    output["metadata"]["selected_users"] = len(
        {int(output["evaluation"][key]["user_id"]) for key in keys}
    )
    return output


def _lineage_gate(
    summaries: dict[int, dict[str, Any]], version: int, document: dict[str, Any]
) -> dict[str, Any]:
    selection = document["lineage_selection"]
    primary_metric = selection.get("primary_metric")
    metrics = tuple(selection.get("positive_ranking_metrics", ("mrr", "ndcg_at_5")))
    relative = {
        source: {
            metric: float(
                summary["comparisons"]["recompute_over_reuse"][metric]["relative_percent"]
            )
            for metric in metrics
        }
        for source, summary in summaries.items()
    }
    ce = {
        source: float(
            summary["comparisons"]["recompute_over_reuse"]["candidate_cross_entropy"][
                "relative_percent"
            ]
        )
        for source, summary in summaries.items()
    }
    adjacent = relative[version - 1]
    means = {
        metric: float(np.mean([value[metric] for value in relative.values()])) for metric in metrics
    }
    fresh = summaries[version - 1]["comparisons"]["fresh_update_value"]
    fresh_positive = all(float(fresh[metric]["relative_percent"]) > 0 for metric in metrics)
    fresh_update_gate_pass = bool(
        fresh_positive or not selection.get("require_fresh_update_ranking_positive", True)
    )
    fresh_endpoints = summaries[version - 1]["endpoints"]["recompute"]
    minimum_fresh_metrics = selection.get("minimum_fresh_metrics", {})
    fresh_absolute_pass = all(
        float(fresh_endpoints[metric]) >= float(minimum)
        for metric, minimum in minimum_fresh_metrics.items()
    )
    ranking_values = [value[metric] for value in relative.values() for metric in metrics]
    ranking_positive_fraction = float(
        sum(value > 0 for value in ranking_values) / len(ranking_values)
    )
    all_ranking_positive = ranking_positive_fraction == 1.0
    minimum_lineage_positive_fraction = float(
        selection.get("minimum_lineage_positive_fraction", 1.0)
    )
    ranking_fraction_pass = ranking_positive_fraction >= minimum_lineage_positive_fraction
    all_ce_positive = all(value > 0 for value in ce.values())
    candidate_ce_pass = bool(
        all_ce_positive or not selection.get("require_candidate_ce_positive", True)
    )
    threshold_metrics = (primary_metric,) if primary_metric is not None else metrics
    maximum_lineage_relative_percent = float(
        selection.get("maximum_lineage_relative_percent", float("inf"))
    )
    maximum_observed_relative_percent = max(
        relative[source][metric] for source in relative for metric in threshold_metrics
    )
    maximum_lineage_pass = (
        maximum_observed_relative_percent <= maximum_lineage_relative_percent
    )
    adjacent_pass = all(
        adjacent[metric] >= float(selection["minimum_adjacent_relative_percent"])
        for metric in threshold_metrics
    )
    lineage_mean_metrics = selection.get("lineage_mean_metrics")
    if lineage_mean_metrics is None:
        threshold_lineage_mean = min(means[metric] for metric in threshold_metrics)
    else:
        threshold_lineage_mean = float(np.mean([means[metric] for metric in lineage_mean_metrics]))
    mean_pass = threshold_lineage_mean >= float(selection["minimum_lineage_mean_relative_percent"])
    sanity = all(value["sanity"]["passed"] for value in summaries.values())
    return {
        "relative_percent": relative,
        "candidate_ce_relative_percent": ce,
        "lineage_mean_relative_percent": means,
        "threshold_lineage_mean_relative_percent": threshold_lineage_mean,
        "fresh_update_ranking_positive": fresh_positive,
        "fresh_update_gate_pass": fresh_update_gate_pass,
        "fresh_absolute_metrics": {
            metric: float(fresh_endpoints[metric]) for metric in minimum_fresh_metrics
        },
        "fresh_absolute_pass": fresh_absolute_pass,
        "all_lineage_ranking_positive": all_ranking_positive,
        "lineage_ranking_positive_fraction": ranking_positive_fraction,
        "minimum_lineage_positive_fraction": minimum_lineage_positive_fraction,
        "lineage_ranking_fraction_pass": ranking_fraction_pass,
        "maximum_lineage_relative_percent": maximum_lineage_relative_percent,
        "maximum_observed_relative_percent": maximum_observed_relative_percent,
        "maximum_lineage_pass": maximum_lineage_pass,
        "all_lineage_candidate_ce_positive": all_ce_positive,
        "candidate_ce_gate_pass": candidate_ce_pass,
        "adjacent_pass": adjacent_pass,
        "lineage_mean_pass": mean_pass,
        "sanity_pass": sanity,
        "passed": bool(
            fresh_update_gate_pass
            and fresh_absolute_pass
            and ranking_fraction_pass
            and maximum_lineage_pass
            and candidate_ce_pass
            and adjacent_pass
            and mean_pass
            and sanity
        ),
    }


def _capture_sources(
    dense,
    embedding,
    tracker,
    workload: dict[str, Any],
    base_config: dict[str, Any],
    document: dict[str, Any],
    config_sha256: str,
    checkpoint_root: Path,
    version: int,
    minimum_source_version: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[int, list[dict[str, Any]]]:
    output = {}
    for source in range(minimum_source_version, version):
        _load_checkpoint(
            checkpoint_root,
            source,
            dense,
            embedding,
            tracker,
            document,
            config_sha256,
            rank,
        )
        batches = _evaluation_batches(
            workload,
            int(document["evaluation"]["local_batch_size"]),
            rank,
            world_size,
        )
        output[source] = _capture_old(dense, embedding, batches, workload, base_config, device)
    return output


def _evaluate_sources(
    dense,
    embedding,
    captures: dict[int, list[dict[str, Any]]],
    workload: dict[str, Any],
    edge_document: dict[str, Any],
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[int, dict[str, Any]] | None:
    output = {}
    for source, captured in sorted(captures.items()):
        summary, _ = _evaluate_captured(
            dense,
            embedding,
            captured,
            workload,
            edge_document,
            rank,
            world_size,
            device,
        )
        if rank == 0:
            assert summary is not None
            output[source] = summary
    return output if rank == 0 else None


def _temperature_document(document: dict[str, Any], temperature: float) -> dict[str, Any]:
    output = copy.deepcopy(document)
    output["training"]["temperature"] = temperature
    return output


def _trial_priority(trial: dict[str, Any]) -> tuple[Any, ...]:
    gate = trial["gate"]
    relative = [value for source in gate["relative_percent"].values() for value in source.values()]
    return (
        bool(gate["passed"]),
        bool(gate["all_lineage_ranking_positive"]),
        bool(gate["adjacent_pass"]),
        bool(gate["lineage_mean_pass"]),
        bool(gate["maximum_lineage_pass"]),
        bool(gate["all_lineage_candidate_ce_positive"]),
        min(relative),
        min(gate["candidate_ce_relative_percent"].values()),
        -float(trial["temperature"]),
    )


def _trial_fresh_cross_entropy(trial: dict[str, Any]) -> float:
    return float(
        np.mean(
            [
                summary["endpoints"]["recompute"]["candidate_cross_entropy"]
                for summary in trial["summaries"].values()
            ]
        )
    )


def run_lineage_retrain(
    config_path: str | Path, stop_after_version: int | None = None
) -> dict[str, Any] | None:
    path = Path(config_path)
    document = load_lineage_retrain_config(path)
    document["config_path"] = str(path)
    config_sha256 = file_sha256(path)
    output_root = Path(document["outputs"]["root"])
    checkpoint_root = Path(document["outputs"]["checkpoint_root"])
    result_path = output_root / "lineage_retrain.json"
    rank, world_size, device = _distributed(document)
    started = time.monotonic()
    try:
        base_config = load_config(document["parent"]["base_config"]["path"])
        edge_documents, workloads = _build_workloads(
            document,
            base_config,
            rank,
            _lineage_workload_limit(document, stop_after_version),
        )
        dense, embedding, tracker, geometry = _initialize_model(
            document,
            base_config,
            int(workloads[0]["metadata"]["embedding_rows"]),
            rank,
            world_size,
            device,
        )
        completed = _completed_prefix(checkpoint_root, output_root, document, config_sha256)
        imported_prefix_versions = int(document["checkpoint"].get("imported_prefix_versions", 0))
        coordinate = _coordinate_drift_config(document)
        selected_versions = document["lineage_selection"]["versions"]
        final_selected_version = int(selected_versions[-1])
        if stop_after_version is None:
            stop_after_version = final_selected_version
        if (
            stop_after_version not in selected_versions
            or not imported_prefix_versions < stop_after_version <= final_selected_version
            or completed > stop_after_version
        ):
            raise ValueError("KuaiRand lineage stop-after version differs")
        if completed not in range(
            imported_prefix_versions,
            int(document["checkpoint"]["versions"]) + 1,
        ):
            raise RuntimeError("KuaiRand lineage retrain prefix differs")
        disk = _disk_preflight(document, checkpoint_root, completed)
        selected_records = []
        for version in selected_versions:
            if version > stop_after_version:
                break
            if version <= completed:
                selected_records.append(
                    json.loads(_accepted_path(output_root, version).read_text())
                )
                continue
            workload = workloads[version - 1]
            tuning_workload = _subset_workload(
                workload,
                "tuning",
                int(document["lineage_selection"]["split_seed"]),
                float(document["lineage_selection"]["tuning_fraction"]),
            )
            tuning_document = copy.deepcopy(edge_documents[version - 1])
            tuning_document["evaluation"]["bootstrap_samples"] = int(
                document["lineage_selection"]["tuning_bootstrap_samples"]
            )
            tuning_captures = _capture_sources(
                dense,
                embedding,
                tracker,
                tuning_workload,
                base_config,
                document,
                config_sha256,
                checkpoint_root,
                version,
                int(document["lineage_selection"].get("minimum_source_version", 1)),
                rank,
                world_size,
                device,
            )
            _load_checkpoint(
                checkpoint_root,
                version - 1,
                dense,
                embedding,
                tracker,
                document,
                config_sha256,
                rank,
            )
            full_batches = _evaluation_batches(
                workload,
                int(document["evaluation"]["local_batch_size"]),
                rank,
                world_size,
            )
            full_adjacent_capture = _capture_old(
                dense, embedding, full_batches, workload, base_config, device
            )
            accepted = None
            for candidate_index, candidate in enumerate(document["training"]["candidate_ladder"]):
                candidate_path = _candidate_path(output_root, version, candidate_index, candidate)
                if candidate_path.is_file():
                    cached = json.loads(candidate_path.read_text())
                    if (
                        cached.get("lineage_protocol") != LINEAGE_PROTOCOL
                        or cached.get("config_sha256") != config_sha256
                        or cached.get("candidate") != candidate
                    ):
                        raise RuntimeError("KuaiRand lineage candidate cache differs")
                    if not cached["tuning_lineage_gate"]["passed"]:
                        continue
                _load_checkpoint(
                    checkpoint_root,
                    version - 1,
                    dense,
                    embedding,
                    tracker,
                    document,
                    config_sha256,
                    rank,
                )
                canonicalization = _apply_coordinate_depth(
                    dense,
                    coordinate,
                    version - 1 - imported_prefix_versions,
                    -1,
                )
                before = tracker.local_bitmap.clone()
                counts_before = tracker.local_update_counts.clone()
                training_temperature = float(
                    candidate.get(
                        "training_temperature",
                        base_config["training"]["temperature"],
                    )
                )
                training = _train_candidate(
                    dense,
                    embedding,
                    tracker,
                    workload,
                    _temperature_document(base_config, training_temperature),
                    _training_document(document, candidate),
                    candidate,
                    rank,
                    world_size,
                    device,
                    int(document["training"]["seed"]) + 2003 + (version - 1) * 100003,
                )
                publication_transform = _apply_coordinate_depth(
                    dense,
                    coordinate,
                    version - imported_prefix_versions,
                )
                temperatures = candidate.get(
                    "evaluation_temperatures",
                    [float(base_config["training"]["temperature"])],
                )
                trials = []
                selected_trial = None
                for temperature in temperatures:
                    tuning_summaries = _evaluate_sources(
                        dense,
                        embedding,
                        tuning_captures,
                        tuning_workload,
                        _temperature_document(tuning_document, float(temperature)),
                        rank,
                        world_size,
                        device,
                    )
                    if rank == 0:
                        assert tuning_summaries is not None
                        gate = _lineage_gate(tuning_summaries, version, document)
                        trial = {
                            "temperature": float(temperature),
                            "gate": gate,
                            "summaries": tuning_summaries,
                        }
                        trials.append(trial)
                        trial_passed = bool(gate["passed"])
                        temperature_search_eligible = bool(
                            gate["fresh_update_gate_pass"]
                            and gate["lineage_ranking_fraction_pass"]
                            and gate["adjacent_pass"]
                            and gate["lineage_mean_pass"]
                            and gate["maximum_lineage_pass"]
                            and gate["sanity_pass"]
                        )
                        print(
                            f"phase=kuairand_lineage_temperature version={version} "
                            f"candidate={candidate['name']} temperature={temperature} "
                            f"passed={trial_passed} "
                            + " ".join(
                                f"{metric}_mean={value:.3f}%"
                                for metric, value in gate["lineage_mean_relative_percent"].items()
                            ),
                            flush=True,
                        )
                    else:
                        trial_passed = None
                        temperature_search_eligible = None
                    trial_payload = [trial_passed, temperature_search_eligible]
                    dist.broadcast_object_list(trial_payload, src=0)
                    if not bool(trial_payload[1]):
                        break
                new_rows, cumulative_rows = _global_new_rows(tracker, before, device)
                embedding_delta_rows = torch.nonzero(
                    tracker.local_update_counts != counts_before,
                    as_tuple=False,
                ).flatten()
                if rank == 0:
                    passing_trials = [trial for trial in trials if trial["gate"]["passed"]]
                    if passing_trials:
                        selected_trial = min(
                            passing_trials,
                            key=lambda trial: (
                                _trial_fresh_cross_entropy(trial),
                                trial["temperature"],
                            ),
                        )
                    else:
                        selected_trial = max(trials, key=_trial_priority)
                    gate = selected_trial["gate"]
                    cell = {
                        "protocol": PROTOCOL,
                        "lineage_protocol": LINEAGE_PROTOCOL,
                        "config_sha256": config_sha256,
                        "version": version,
                        "source_version": version - 1,
                        "transition": document["transitions"][version - 1],
                        "candidate_index": candidate_index,
                        "candidate": candidate,
                        "training": training,
                        "coordinate_drift": {
                            "canonicalization": canonicalization,
                            "publication_transform": publication_transform,
                            "source_depth": version - 1 - imported_prefix_versions,
                            "target_depth": version - imported_prefix_versions,
                        }
                        if coordinate is not None
                        else None,
                        "evaluation_temperature": selected_trial["temperature"],
                        "temperature_trials": [
                            {
                                "temperature": trial["temperature"],
                                "gate": trial["gate"],
                            }
                            for trial in trials
                        ],
                        "tuning_lineage": selected_trial["summaries"],
                        "tuning_lineage_gate": gate,
                        "new_optimizer_active_rows": new_rows,
                        "cumulative_optimizer_active_rows": cumulative_rows,
                        "local_embedding_delta_rows": int(embedding_delta_rows.numel()),
                        "scientific_result": False,
                        "formal_result": False,
                    }
                    _atomic_json(candidate_path, cell)
                    print(
                        f"phase=kuairand_lineage_candidate version={version} "
                        f"candidate={candidate['name']} passed={gate['passed']} "
                        + " ".join(
                            f"{metric}_mean={value:.3f}%"
                            for metric, value in gate["lineage_mean_relative_percent"].items()
                        ),
                        flush=True,
                    )
                    passed = bool(gate["passed"])
                else:
                    cell = None
                    passed = None
                payload = [passed]
                dist.broadcast_object_list(payload, src=0)
                if not bool(payload[0]):
                    continue
                evaluation_temperature = (
                    float(cell["evaluation_temperature"]) if rank == 0 else None
                )
                temperature_payload = [evaluation_temperature]
                dist.broadcast_object_list(temperature_payload, src=0)
                evaluation_temperature = float(temperature_payload[0])
                full_summary, _ = _evaluate_captured(
                    dense,
                    embedding,
                    full_adjacent_capture,
                    workload,
                    _temperature_document(edge_documents[version - 1], evaluation_temperature),
                    rank,
                    world_size,
                    device,
                )
                if rank == 0:
                    assert cell is not None and full_summary is not None
                    cell["summary"] = full_summary
                    cell["admitted"] = True
                    cell["quality_admitted"] = True
                    cell["passing_metrics"] = _passing(full_summary, document)
                    _atomic_json(candidate_path, cell)
                    candidate_record = {
                        "path": str(candidate_path),
                        "sha256": file_sha256(candidate_path),
                        "candidate": candidate,
                        "evaluation_temperature": evaluation_temperature,
                        "summary": full_summary,
                        "passing_metrics": cell["passing_metrics"],
                        "admitted": True,
                        "quality_admitted": True,
                        "new_optimizer_active_rows": new_rows,
                        "cumulative_optimizer_active_rows": cumulative_rows,
                        "local_embedding_delta_rows": int(embedding_delta_rows.numel()),
                    }
                else:
                    candidate_record = None
                records = [candidate_record]
                dist.broadcast_object_list(records, src=0)
                candidate_record = records[0]
                manifest = _save_checkpoint(
                    checkpoint_root,
                    version,
                    dense,
                    embedding,
                    tracker,
                    geometry,
                    document,
                    config_sha256,
                    {
                        "round_id": document["round_id"],
                        "config": {"path": str(path), "sha256": config_sha256},
                        "source_version": version - 1,
                        "transition": document["transitions"][version - 1],
                        "accepted_candidate": {
                            "path": candidate_record["path"],
                            "sha256": candidate_record["sha256"],
                            "name": candidate["name"],
                            "evaluation_temperature": candidate_record["evaluation_temperature"],
                        },
                        "checkpoint_policy": "lineage_tuning_gate",
                        "coordinate_drift": cell.get("coordinate_drift")
                        if rank == 0
                        else None,
                    },
                    rank,
                    world_size,
                    embedding_delta_rows,
                )
                if rank == 0:
                    accepted = {
                        "protocol": PROTOCOL,
                        "version": version,
                        "source_version": version - 1,
                        "status": "accepted",
                        "candidate": candidate_record,
                        "checkpoint": {
                            "path": str(checkpoint_root / f"theta_{version}" / "manifest.json"),
                            "sha256": file_sha256(
                                checkpoint_root / f"theta_{version}" / "manifest.json"
                            ),
                            "bytes": int(manifest["checkpoint_bytes"]),
                        },
                        "selection": document["selection"],
                        "checkpoint_policy": "lineage_tuning_gate",
                        "evaluation_temperature": candidate_record["evaluation_temperature"],
                        "scientific_result": False,
                        "formal_result": False,
                    }
                    _atomic_json(_accepted_path(output_root, version), accepted)
                accepted_payload = [accepted]
                dist.broadcast_object_list(accepted_payload, src=0)
                accepted = accepted_payload[0]
                selected_records.append(accepted)
                break
            if accepted is None:
                raise RuntimeError(f"KuaiRand theta{version} has no lineage-safe candidate")
            completed = version
        if rank != 0:
            return None
        result = {
            "lineage_protocol": LINEAGE_PROTOCOL,
            "status": "complete_selected_lineage_versions"
            if stop_after_version == final_selected_version
            else "partial_selected_lineage_versions",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(path), "sha256": config_sha256},
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "selected": selected_records,
            "geometry": geometry,
            "disk": disk,
            "stop_after_version": stop_after_version,
            "elapsed_seconds": time.monotonic() - started,
            "next": "run_persistent_chain_for_full_holdout_triangle"
            if stop_after_version == final_selected_version
            else f"train_theta{stop_after_version + 1}",
        }
        _atomic_json(result_path, result)
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
