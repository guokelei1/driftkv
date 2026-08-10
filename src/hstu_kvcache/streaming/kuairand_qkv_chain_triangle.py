from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .kuairand_lineage_retrain import load_lineage_retrain_config
from .kuairand_projected_persistent import (
    _accepted_path,
    _build_workloads,
    _distributed,
    _evaluation_batches,
    _initialize_model,
    _lineage_partition_summary,
    _load_checkpoint,
    _temperature_edge_document,
)
from .kuairand_projected_scale import _capture_old, _evaluate_captured
from .kuairand_query_transition import _atomic_json, file_sha256, load_config

PROTOCOL = "evokv_kuairand_qkv_chain_triangle_v0"
METRICS = ("ndcg_at_5", "mrr", "hit_rate_at_5")


def load_qkv_chain_config(path: str | Path) -> dict[str, Any]:
    document = load_lineage_retrain_config(path)
    selection = document["lineage_selection"]
    candidates = document["training"]["candidate_ladder"]
    coordinate = document.get("coordinate_drift", {})
    if (
        int(document["checkpoint"]["imported_prefix_versions"]) != 5
        or int(document["checkpoint"]["versions"]) != 12
        or document["checkpoint"]["embedding_storage"]
        != "sparse_delta_after_imported_prefix"
        or selection["versions"] != list(range(6, 13))
        or int(selection["minimum_source_version"]) != 5
        or not candidates
        or any(candidate.get("dense_update_scope") != "frozen" for candidate in candidates)
        or coordinate.get("mode") != "cumulative_function_preserving_scale"
        or coordinate.get("canonicalize_before_update") is not True
    ):
        raise ValueError("KuaiRand QKV chain config differs")
    return document


def _relative(cell: dict[str, Any], split: str, metric: str) -> float:
    return float(
        cell[split]["comparisons"]["recompute_over_reuse"][metric][
            "relative_percent"
        ]
    )


def _render_matrix(
    cells: list[dict[str, Any]], versions: list[int], metric: str
) -> list[str]:
    by_pair = {
        (int(cell["target_version"]), int(cell["source_version"])): _relative(
            cell, "holdout", metric
        )
        for cell in cells
    }
    lines = [
        f"## Holdout {metric}",
        "",
        "| current \\ cache | "
        + " | ".join(f"M{version - versions[0]}" for version in versions)
        + " |",
        "|---|" + "---:|" * len(versions),
    ]
    for target in versions:
        values = []
        for source in versions:
            if source > target:
                values.append("—")
            elif source == target:
                values.append("+0.000%")
            else:
                values.append(f"{by_pair[(target, source)]:+.3f}%")
        lines.append(f"| M{target - versions[0]} | " + " | ".join(values) + " |")
    return lines


def _render(result: dict[str, Any]) -> str:
    versions = [int(value) for value in result["versions"]]
    lines = [
        "# KuaiRand coordinate-aligned M0–M7 Reuse–Recompute matrix",
        "",
        "Positive means Recompute is better. M0 is one retained bootstrap; M1–M7 are real sequential next-item updates with a label-free-calibrated, function-preserving K/V-coordinate publication step. The steady-state publication setting is frozen from M2 onward.",
        "Relative percent is 100 × (Fresh - Reuse) / Reuse.",
        "",
    ]
    for metric in METRICS:
        lines.extend(_render_matrix(result["cells"], versions, metric))
        lines.append("")
    lines.extend(
        [
            "## Adjacent holdout summary",
            "",
            "| current | cache | MRR | NDCG@5 | HR@5 | Fresh NDCG@5 | Reuse NDCG@5 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    adjacent = [
        cell
        for cell in result["cells"]
        if int(cell["source_version"]) + 1 == int(cell["target_version"])
    ]
    for cell in adjacent:
        endpoints = cell["holdout"]["endpoints"]
        lines.append(
            "| M{target} | M{source} | {mrr:+.3f}% | {ndcg:+.3f}% | {hr:+.3f}% | {fresh:.6f} | {reuse:.6f} |".format(
                target=int(cell["target_version"]) - versions[0],
                source=int(cell["source_version"]) - versions[0],
                mrr=_relative(cell, "holdout", "mrr"),
                ndcg=_relative(cell, "holdout", "ndcg_at_5"),
                hr=_relative(cell, "holdout", "hit_rate_at_5"),
                fresh=endpoints["recompute"]["ndcg_at_5"],
                reuse=endpoints["reuse"]["ndcg_at_5"],
            )
        )
    return "\n".join(lines) + "\n"


def _decision(cells: list[dict[str, Any]]) -> dict[str, Any]:
    adjacent = [
        cell
        for cell in cells
        if int(cell["source_version"]) + 1 == int(cell["target_version"])
    ]
    positive = {
        metric: sum(_relative(cell, "holdout", metric) > 0.0 for cell in cells)
        for metric in METRICS
    }
    adjacent_values = {
        metric: [_relative(cell, "holdout", metric) for cell in adjacent]
        for metric in METRICS
    }
    accumulation_rows = 0
    for target in range(7, 13):
        row = [cell for cell in cells if int(cell["target_version"]) == target]
        adjacent_cell = next(
            cell for cell in row if int(cell["source_version"]) == target - 1
        )
        older = [
            _relative(cell, "holdout", "ndcg_at_5")
            for cell in row
            if int(cell["source_version"]) < target - 1
        ]
        if older and max(older) > _relative(adjacent_cell, "holdout", "ndcg_at_5"):
            accumulation_rows += 1
    return {
        "matrix_versions": 8,
        "off_diagonal_cells": len(cells),
        "positive_cells": positive,
        "adjacent_cells": len(adjacent),
        "adjacent_mean_relative_percent": {
            metric: float(np.mean(values)) for metric, values in adjacent_values.items()
        },
        "adjacent_minimum_relative_percent": {
            metric: float(np.min(values)) for metric, values in adjacent_values.items()
        },
        "rows_with_older_ndcg_stronger_than_adjacent": accumulation_rows,
        "same_model_sanity_passed": all(
            cell["all_users"]["sanity"]["passed"] for cell in cells
        ),
    }


def run_qkv_chain_triangle(config_path: str | Path) -> dict[str, Any] | None:
    path = Path(config_path)
    document = load_qkv_chain_config(path)
    output_root = Path(document["outputs"]["root"])
    result_path = output_root / "qkv_chain_triangle.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        return result if int(os.environ.get("RANK", "0")) == 0 else None
    document["config_path"] = str(path)
    config_sha256 = file_sha256(path)
    rank, world_size, device = _distributed(document)
    started = time.monotonic()
    try:
        base_config = load_config(document["parent"]["base_config"]["path"])
        edge_documents, workloads = _build_workloads(document, base_config, rank, 12)
        dense, embedding, tracker, geometry = _initialize_model(
            document,
            base_config,
            int(workloads[0]["metadata"]["embedding_rows"]),
            rank,
            world_size,
            device,
        )
        checkpoint_root = Path(document["outputs"]["checkpoint_root"])
        captures: dict[int, dict[int, list[dict[str, Any]]]] = {
            target: {} for target in range(6, 13)
        }
        for source in range(5, 12):
            _load_checkpoint(
                checkpoint_root,
                source,
                dense,
                embedding,
                tracker,
                document,
                config_sha256,
                rank,
                verify_hash=True,
            )
            for target in range(source + 1, 13):
                workload = workloads[target - 1]
                batches = _evaluation_batches(
                    workload,
                    int(document["evaluation"]["local_batch_size"]),
                    rank,
                    world_size,
                )
                captures[target][source] = _capture_old(
                    dense, embedding, batches, workload, base_config, device
                )
                del batches
            if rank == 0:
                print(
                    f"phase=kuairand_qkv_triangle_capture source={source} "
                    f"targets={12 - source}",
                    flush=True,
                )
            torch.cuda.empty_cache()
        cells = []
        selection = document["lineage_selection"]
        for target in range(6, 13):
            accepted = json.loads(_accepted_path(output_root, target).read_text())
            if accepted["candidate"]["candidate"].get("dense_update_scope") != "frozen":
                raise RuntimeError("KuaiRand QKV chain accepted scope differs")
            temperature = float(accepted.get("evaluation_temperature", 0.05))
            evaluation_document = _temperature_edge_document(
                edge_documents[target - 1], temperature
            )
            _load_checkpoint(
                checkpoint_root,
                target,
                dense,
                embedding,
                tracker,
                document,
                config_sha256,
                rank,
                verify_hash=True,
            )
            for source, captured in sorted(captures[target].items()):
                compact, evaluation = _evaluate_captured(
                    dense,
                    embedding,
                    captured,
                    workloads[target - 1],
                    evaluation_document,
                    rank,
                    world_size,
                    device,
                )
                if rank == 0:
                    assert compact is not None and evaluation is not None
                    cell = {
                        "target_version": target,
                        "source_version": source,
                        "cache_age": target - source,
                        "all_users": compact,
                        "tuning": _lineage_partition_summary(
                            evaluation,
                            evaluation_document,
                            int(selection["split_seed"]),
                            float(selection["tuning_fraction"]),
                            int(selection["tuning_bootstrap_samples"]),
                            "tuning",
                        ),
                        "holdout": _lineage_partition_summary(
                            evaluation,
                            evaluation_document,
                            int(selection["split_seed"]),
                            float(selection["tuning_fraction"]),
                            int(selection["tuning_bootstrap_samples"]),
                            "holdout",
                        ),
                    }
                    if source == target - 1:
                        expected = accepted["candidate"]["summary"]
                        for metric in METRICS:
                            observed_value = compact["comparisons"][
                                "recompute_over_reuse"
                            ][metric]["relative_percent"]
                            expected_value = expected["comparisons"][
                                "recompute_over_reuse"
                            ][metric]["relative_percent"]
                            if not np.isclose(
                                observed_value, expected_value, rtol=0.0, atol=1e-6
                            ):
                                raise RuntimeError(
                                    "KuaiRand QKV adjacent checkpoint replay differs"
                                )
                    cells.append(cell)
                del captured
            captures[target].clear()
            gc.collect()
            torch.cuda.empty_cache()
            if rank == 0:
                print(
                    f"phase=kuairand_qkv_triangle_target target={target} "
                    f"sources={target - 5}",
                    flush=True,
                )
        if rank != 0:
            return None
        checkpoints = []
        for version in range(5, 13):
            manifest_path = checkpoint_root / f"theta_{version}" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            checkpoints.append(
                {
                    "version": version,
                    "path": str(manifest_path),
                    "sha256": file_sha256(manifest_path),
                    "bytes": int(manifest["checkpoint_bytes"]),
                    "embedding_storage": manifest.get("embedding_storage", "full"),
                }
            )
        if len(cells) != 28:
            raise RuntimeError("KuaiRand QKV triangle cell count differs")
        result = {
            "protocol": PROTOCOL,
            "status": "complete_development_qkv_chain_triangle",
            "scientific_result": False,
            "formal_result": False,
            "config": {"path": str(path), "sha256": config_sha256},
            "versions": list(range(5, 13)),
            "display_versions": list(range(8)),
            "coordinate_drift": document["coordinate_drift"],
            "geometry": geometry,
            "checkpoints": checkpoints,
            "cells": cells,
            "decision": _decision(cells),
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(result_path, result)
        table_path = output_root / "qkv_chain_triangle.md"
        temporary = table_path.with_suffix(f".md.tmp.{os.getpid()}")
        temporary.write_text(_render(result))
        os.replace(temporary, table_path)
        result["table"] = {"path": str(table_path), "sha256": file_sha256(table_path)}
        _atomic_json(result_path, result)
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
