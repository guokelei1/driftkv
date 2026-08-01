from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL = "evokv_xp_reuse_exact_suffix_diagnostic_development_v0"
PREFLIGHT_SCHEMA = "evokv_reuse_exact_opportunity_screen_preflight_v0"
SUMMARY_SCHEMA = "evokv_reuse_exact_opportunity_screen_summary_v0"
METRICS = (
    "sampled_cross_entropy",
    "hit_rate_at_10",
    "ndcg_at_10",
    "mean_reciprocal_rank",
)


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    category: str
    source_version: int
    target_version: int
    history_end: int
    update_end: int
    training_history_end: int | None
    training_update_end: int | None
    qualification_role: str
    evaluation_kind: str


def screen_matrix() -> tuple[CellSpec, ...]:
    cells = []
    for source_version in range(3):
        target_version = source_version + 1
        history_end = 72 + source_version * 8
        cells.append(
            CellSpec(
                cell_id=(
                    f"preq_theta{source_version}_to_theta{target_version}_"
                    f"h{history_end}"
                ),
                category="prequential",
                source_version=source_version,
                target_version=target_version,
                history_end=history_end,
                update_end=history_end + 8,
                training_history_end=64 + source_version * 8,
                training_update_end=72 + source_version * 8,
                qualification_role="qualification",
                evaluation_kind="prequential",
            )
        )
        for history_end in (145, 396):
            cells.append(
                CellSpec(
                    cell_id=(
                        f"long_h{history_end}_theta{source_version}_to_"
                        f"theta{target_version}"
                    ),
                    category="long_context_characterization",
                    source_version=source_version,
                    target_version=target_version,
                    history_end=history_end,
                    update_end=history_end + 8,
                    training_history_end=None,
                    training_update_end=None,
                    qualification_role="theta12",
                    evaluation_kind="long_context_characterization",
                )
            )
    return tuple(cells)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-label", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--base-checkpoint-root", type=Path, required=True)
    parser.add_argument("--target-checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def resolved_equal(left: object, right: Path) -> bool:
    return isinstance(left, str) and Path(left).resolve() == right.resolve()


def checkpoint_manifest(
    spec: CellSpec,
    root: Path,
    base_root: Path,
) -> Path:
    selected = base_root if spec.source_version == 0 else root
    return selected / f"theta_{spec.source_version}" / "manifest.json"


def target_manifest(spec: CellSpec, root: Path) -> Path:
    return root / f"theta_{spec.target_version}" / "manifest.json"


def metric_contributions(
    ranks: np.ndarray,
    cross_entropy: np.ndarray,
) -> dict[str, np.ndarray]:
    hit = (ranks <= 10).astype(np.float64)
    ndcg = np.where(
        ranks <= 10,
        1.0 / np.log2(ranks.astype(np.float64) + 1.0),
        0.0,
    )
    return {
        "sampled_cross_entropy": cross_entropy,
        "hit_rate_at_10": hit,
        "ndcg_at_10": ndcg,
        "mean_reciprocal_rank": 1.0 / ranks.astype(np.float64),
    }


def paired_gains(paired: dict[str, Any]) -> dict[str, np.ndarray]:
    reuse_ranks = np.asarray(paired["all_reuse"]["ranks"], dtype=np.int64)
    exact_ranks = np.asarray(paired["all_exact"]["ranks"], dtype=np.int64)
    reuse_ce = np.asarray(
        paired["all_reuse"]["sampled_cross_entropy"],
        dtype=np.float64,
    )
    exact_ce = np.asarray(
        paired["all_exact"]["sampled_cross_entropy"],
        dtype=np.float64,
    )
    reuse = metric_contributions(reuse_ranks, reuse_ce)
    exact = metric_contributions(exact_ranks, exact_ce)
    return {
        metric: (
            reuse[metric] - exact[metric]
            if metric == "sampled_cross_entropy"
            else exact[metric] - reuse[metric]
        )
        for metric in METRICS
    }


def cluster_bootstrap(
    record_ids: np.ndarray,
    gains: dict[str, np.ndarray],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if replicates < 1 or record_ids.size < 2:
        raise ValueError("record-cluster bootstrap request differs")
    unique, inverse = np.unique(record_ids, return_inverse=True)
    cluster_counts = np.bincount(inverse).astype(np.float64)
    cluster_sums = {}
    for metric, values in gains.items():
        sums = np.zeros(unique.size, dtype=np.float64)
        np.add.at(sums, inverse, values)
        cluster_sums[metric] = sums
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in METRICS}
    remaining = replicates
    while remaining:
        current = min(remaining, 256)
        selected = rng.integers(
            0,
            unique.size,
            size=(current, unique.size),
        )
        denominators = cluster_counts[selected].sum(axis=1)
        for metric in METRICS:
            samples[metric].append(
                cluster_sums[metric][selected].sum(axis=1) / denominators
            )
        remaining -= current
    return {
        metric: {
            "lower_95": float(
                np.quantile(np.concatenate(samples[metric]), 0.025)
            ),
            "upper_95": float(
                np.quantile(np.concatenate(samples[metric]), 0.975)
            ),
        }
        for metric in METRICS
    }


def validate_checkpoint_binding(
    binding: dict[str, Any],
    manifest: Path,
    version: int,
) -> None:
    if (
        not manifest.is_file()
        or not resolved_equal(binding.get("path"), manifest)
        or binding.get("sha256") != file_sha256(manifest)
        or binding.get("version") != version
    ):
        raise ValueError(f"checkpoint binding differs: {manifest}")


def validate_paired(
    quality: dict[str, Any],
    path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    paired = quality.get("paired_target_contributions")
    if not isinstance(paired, dict):
        raise ValueError(f"paired contributions absent: {path}")
    targets = int(paired.get("targets", -1))
    arrays = [
        paired.get("record_ids"),
        paired.get("suffix_offsets"),
        paired.get("all_reuse", {}).get("ranks"),
        paired.get("all_reuse", {}).get("sampled_cross_entropy"),
        paired.get("all_exact", {}).get("ranks"),
        paired.get("all_exact", {}).get("sampled_cross_entropy"),
    ]
    if (
        targets < 1
        or any(not isinstance(values, list) for values in arrays)
        or any(len(values) != targets for values in arrays)
    ):
        raise ValueError(f"paired contribution lengths differ: {path}")
    record_ids = np.asarray(arrays[0], dtype=np.int64)
    offsets = np.asarray(arrays[1], dtype=np.int64)
    keys = np.stack((record_ids, offsets), axis=1)
    if (
        np.unique(keys, axis=0).shape[0] != targets
        or np.any((offsets < 1) | (offsets > 8))
        or paired.get("pair_key_sha256")
        != canonical_sha256(
            {
                "pairs": [
                    {
                        "record_id": int(record_id),
                        "suffix_offset": int(offset),
                    }
                    for record_id, offset in keys
                ]
            }
        )
    ):
        raise ValueError(f"paired contribution keys differ: {path}")
    gains = paired_gains(paired)
    methods = quality["methods"]
    for method in ("all_reuse", "all_exact"):
        ranks = np.asarray(paired[method]["ranks"], dtype=np.int64)
        ce = np.asarray(
            paired[method]["sampled_cross_entropy"],
            dtype=np.float64,
        )
        if np.any(ranks < 1) or not np.all(np.isfinite(ce)):
            raise ValueError(f"paired contribution values differ: {path}")
        observed = metric_contributions(ranks, ce)
        recommendation = methods[method]["recommendation"]
        for metric in METRICS:
            if not math.isclose(
                float(np.mean(observed[metric])),
                float(recommendation[metric]),
                rel_tol=1e-8,
                abs_tol=1e-7,
            ):
                raise ValueError(f"paired aggregate differs: {path}: {metric}")
    return {
        "record_ids": record_ids,
        "suffix_offsets": offsets,
        "targets": targets,
    }, gains


def validate_cell(
    spec: CellSpec,
    path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    value = load_json(path)
    edge = value.get("edge", {})
    expected_semantics = (
        "next_unseen_window"
        if spec.category == "prequential"
        else "nonprequential_long_context_characterization"
    )
    if (
        value.get("protocol") != PROTOCOL
        or value.get("status") != "complete"
        or value.get("scientific_result") is not False
        or value.get("formal_result") is not False
        or value.get("world_size") != 2
        or value.get("evaluation_kind") != spec.evaluation_kind
        or edge.get("source_version") != spec.source_version
        or edge.get("target_version") != spec.target_version
        or edge.get("history_end") != spec.history_end
        or edge.get("update_end") != spec.update_end
        or edge.get("training_window")
        != (
            None
            if spec.training_history_end is None
            else {
                "history_end": spec.training_history_end,
                "update_end": spec.training_update_end,
            }
        )
        or edge.get("evaluation_window", {}).get("semantics")
        != expected_semantics
        or edge.get("evaluation_window", {}).get("suffix_offsets")
        != list(range(1, 9))
        or value.get("role", {}).get("source_role")
        != spec.qualification_role
        or value.get("recommendation_contract", {}).get(
            "negative_candidates"
        )
        != [999]
        or value.get("recommendation_contract", {}).get(
            "common_cache_endpoint"
        )
        != {
            "storage_dtype": "torch.float16",
            "consumption_dtype": "torch.float32",
            "exact_path": "fp32_compute_to_fp16_storage_to_fp32_consume",
            "reuse_path": "fp16_storage_to_fp32_consume",
        }
    ):
        raise ValueError(f"opportunity cell contract differs: {path}")
    run_args = value.get("args", {})
    split = value.get("bindings", {}).get("evaluation_split", {}).get(
        "roles", {}
    )
    if (
        run_args.get("qualification_role") != spec.qualification_role
        or run_args.get("batch_size_per_rank") != 4
        or run_args.get("diagnostic_negative_counts") != "999"
        or run_args.get("reuse_exact_suffix_offsets") is not True
        or run_args.get("diagnostic_evaluation_kind")
        != spec.evaluation_kind
        or not resolved_equal(run_args.get("config"), args.benchmark_config)
        or split.get("fit", {}).get("records") != 0
        or split.get("probe", {}).get("records") != 0
        or split.get("qualification_test", {}).get("records", 0) < 1
    ):
        raise ValueError(f"opportunity cell execution binding differs: {path}")
    source_manifest = checkpoint_manifest(
        spec,
        args.target_checkpoint_root,
        args.base_checkpoint_root,
    )
    target_path = target_manifest(spec, args.target_checkpoint_root)
    validate_checkpoint_binding(
        value["bindings"]["source_checkpoint"],
        source_manifest,
        spec.source_version,
    )
    validate_checkpoint_binding(
        value["bindings"]["target_checkpoint"],
        target_path,
        spec.target_version,
    )
    quality_map = value.get("quality_by_negative_count")
    if not isinstance(quality_map, dict) or set(quality_map) != {"999"}:
        raise ValueError(f"opportunity candidate sets differ: {path}")
    quality = quality_map["999"]
    methods = quality.get("methods", {})
    if set(methods) != {"all_reuse", "all_exact"}:
        raise ValueError(f"opportunity methods differ: {path}")
    for method in ("all_reuse", "all_exact"):
        offsets = methods[method].get("recommendation_by_suffix_offset", {})
        if set(offsets) != {str(value) for value in range(1, 9)}:
            raise ValueError(f"opportunity suffix offsets differ: {path}")
    paired, gains = validate_paired(quality, path)
    if paired["targets"] != methods["all_exact"]["recommendation"][
        "positive_targets"
    ]:
        raise ValueError(f"opportunity positive target count differs: {path}")
    offset_one = paired["suffix_offsets"] == 1
    if not bool(np.any(offset_one)):
        raise ValueError(f"opportunity offset one is empty: {path}")
    seed_prefix = int(
        hashlib.sha256(spec.cell_id.encode()).hexdigest()[:16],
        16,
    )
    bootstrap_all = cluster_bootstrap(
        paired["record_ids"],
        gains,
        args.bootstrap_replicates,
        args.bootstrap_seed ^ seed_prefix,
    )
    bootstrap_offset_one = cluster_bootstrap(
        paired["record_ids"][offset_one],
        {metric: values[offset_one] for metric, values in gains.items()},
        args.bootstrap_replicates,
        args.bootstrap_seed ^ seed_prefix ^ 0x51A7,
    )
    reuse = methods["all_reuse"]["recommendation"]
    exact = methods["all_exact"]["recommendation"]
    reuse_offset = methods["all_reuse"]["recommendation_by_suffix_offset"][
        "1"
    ]
    exact_offset = methods["all_exact"]["recommendation_by_suffix_offset"][
        "1"
    ]
    deltas = {
        metric: (
            float(reuse[metric]) - float(exact[metric])
            if metric == "sampled_cross_entropy"
            else float(exact[metric]) - float(reuse[metric])
        )
        for metric in METRICS
    }
    offset_deltas = {
        metric: (
            float(reuse_offset[metric]) - float(exact_offset[metric])
            if metric == "sampled_cross_entropy"
            else float(exact_offset[metric]) - float(reuse_offset[metric])
        )
        for metric in METRICS
    }
    return {
        "cell_id": spec.cell_id,
        "category": spec.category,
        "characterization_only": spec.category != "prequential",
        "source_version": spec.source_version,
        "target_version": spec.target_version,
        "training_history_end": spec.training_history_end,
        "training_update_end": spec.training_update_end,
        "evaluation_history_end": spec.history_end,
        "evaluation_end": spec.update_end,
        "qualification_role": spec.qualification_role,
        "records": quality["records"],
        "positive_targets": paired["targets"],
        "offset1_positive_targets": int(np.sum(offset_one)),
        "negative_count": 999,
        "all_reuse": {metric: float(reuse[metric]) for metric in METRICS},
        "all_exact": {metric: float(exact[metric]) for metric in METRICS},
        "exact_gain": deltas,
        "offset1_all_reuse": {
            metric: float(reuse_offset[metric]) for metric in METRICS
        },
        "offset1_all_exact": {
            metric: float(exact_offset[metric]) for metric in METRICS
        },
        "offset1_exact_gain": offset_deltas,
        "record_cluster_bootstrap_95": {
            "overall": bootstrap_all,
            "offset1": bootstrap_offset_one,
            "replicates": args.bootstrap_replicates,
        },
        "artifact": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        },
    }


def validate_preflight(args: argparse.Namespace) -> dict[str, Any]:
    value = load_json(args.preflight)
    if (
        value.get("schema") != PREFLIGHT_SCHEMA
        or value.get("status") != "pass"
        or value.get("round_label") != args.round_label
        or not resolved_equal(
            value.get("benchmark_config", {}).get("path"),
            args.benchmark_config,
        )
        or value.get("benchmark_config", {}).get("sha256")
        != file_sha256(args.benchmark_config)
        or len(value.get("checkpoint_manifests", [])) != 4
    ):
        raise ValueError("opportunity screen preflight differs")
    expected = [
        args.base_checkpoint_root / "theta_0" / "manifest.json",
        *(
            args.target_checkpoint_root / f"theta_{version}" / "manifest.json"
            for version in (1, 2, 3)
        ),
    ]
    for descriptor, path in zip(
        value["checkpoint_manifests"],
        expected,
        strict=True,
    ):
        if (
            not resolved_equal(descriptor.get("path"), path)
            or descriptor.get("sha256") != file_sha256(path)
            or descriptor.get("artifacts_verified") is not True
        ):
            raise ValueError(f"preflight checkpoint differs: {path}")
    return value


def tsv_rows(cells: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    fields = [
        "cell_id",
        "category",
        "characterization_only",
        "source_version",
        "target_version",
        "training_history_end",
        "training_update_end",
        "evaluation_history_end",
        "evaluation_end",
        "qualification_role",
        "records",
        "positive_targets",
        "offset1_positive_targets",
        "negative_count",
    ]
    for scope in ("overall", "offset1"):
        for metric in METRICS:
            fields.extend(
                (
                    f"{scope}_reuse_{metric}",
                    f"{scope}_exact_{metric}",
                    f"{scope}_exact_gain_{metric}",
                    f"{scope}_gain_ci95_low_{metric}",
                    f"{scope}_gain_ci95_high_{metric}",
                )
            )
    rows = []
    for cell in cells:
        row = [cell[field] for field in fields[:14]]
        for scope in ("overall", "offset1"):
            prefix = "" if scope == "overall" else "offset1_"
            for metric in METRICS:
                interval = cell["record_cluster_bootstrap_95"][scope][metric]
                row.extend(
                    (
                        cell[f"{prefix}all_reuse"][metric],
                        cell[f"{prefix}all_exact"][metric],
                        cell[f"{prefix}exact_gain"][metric],
                        interval["lower_95"],
                        interval["upper_95"],
                    )
                )
        rows.append(row)
    return fields, rows


def write_outputs(
    summary: dict[str, Any],
    cells: list[dict[str, Any]],
    output_json: Path,
    output_tsv: Path,
) -> None:
    if output_json.exists() or output_tsv.exists():
        raise FileExistsError("refusing to overwrite opportunity summary")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    json_temporary = output_json.with_name(f".{output_json.name}.{os.getpid()}.tmp")
    tsv_temporary = output_tsv.with_name(f".{output_tsv.name}.{os.getpid()}.tmp")
    fields, rows = tsv_rows(cells)
    try:
        json_temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        with tsv_temporary.open("w", newline="") as destination:
            writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
            writer.writerow(fields)
            writer.writerows(rows)
        os.replace(tsv_temporary, output_tsv)
        os.replace(json_temporary, output_json)
    finally:
        json_temporary.unlink(missing_ok=True)
        tsv_temporary.unlink(missing_ok=True)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_replicates < 1 or args.bootstrap_seed < 0:
        raise ValueError("bootstrap configuration differs")
    preflight = validate_preflight(args)
    cells = [
        validate_cell(
            spec,
            args.result_root / "cells" / f"{spec.cell_id}.json",
            args,
        )
        for spec in screen_matrix()
    ]
    observed = sorted((args.result_root / "cells").glob("*.json"))
    expected = sorted(
        args.result_root / "cells" / f"{spec.cell_id}.json"
        for spec in screen_matrix()
    )
    if observed != expected:
        raise ValueError("opportunity screen cell set differs")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "round_label": args.round_label,
        "protocol": PROTOCOL,
        "matrix_complete": True,
        "cells_reported": len(cells),
        "selection_policy": "all_nine_cells_reported_without_selection",
        "common_cache_endpoint": {
            "storage_dtype": "torch.float16",
            "consumption_dtype": "torch.float32",
        },
        "negative_count": 999,
        "gain_direction": "positive_values_always_mean_all_exact_is_better",
        "bootstrap": {
            "unit": "record_id_cluster",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "interval": "percentile_95",
        },
        "claim_boundary": {
            "prequential": "baseline opportunity screen",
            "long_context_characterization": (
                "nonprequential development characterization only"
            ),
        },
        "preflight": {
            "path": str(args.preflight),
            "sha256": file_sha256(args.preflight),
            "gpu_uuids": preflight["gpu_uuids"],
        },
        "cells": cells,
    }
    write_outputs(summary, cells, args.output_json, args.output_tsv)
    return summary


def main() -> None:
    args = parse_args()
    summary = summarize(args)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_tsv": str(args.output_tsv),
                "cells": summary["cells_reported"],
                "status": summary["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
