from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "mrr",
    "ndcg@10",
    "ndcg@100",
    "hit@10",
    "hit@100",
    "best_rank",
    "mean_rank",
    "rank_utility",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
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


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"table is empty: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=list(rows[0]),
            dialect="excel-tab",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def metric_columns(
    prefix: str,
    values: dict[str, Any],
) -> dict[str, object]:
    return {
        f"{prefix}_{name.replace('@', '_at_')}": values.get(name)
        for name in METRICS
    }


def semantic_rows(summary: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    for cell_name, cell in summary["cells"].items():
        model = cell["data_and_model"]
        for relative in cell["source_files"]["control"]:
            source = load(ROOT / relative)
            if source.get("protocol") != "motivation_capacity_v2_streaming_control":
                raise ValueError(f"streaming control differs: {relative}")
            for pair in source["pairs"]:
                conditions = pair["summary"]["conditions"]
                row: dict[str, object] = {
                    "cell": cell_name,
                    "dataset": cell["dataset"],
                    "tier": cell["tier"],
                    "seed": source["seed"],
                    "model_version": pair["model_t"],
                    "cache_version": pair["frozen_t"],
                    "model_delta_relative": pair["dtheta_rel"],
                    "evaluation_users": pair["n_users"],
                    "num_users": model["num_users"],
                    "num_items": model["num_items"],
                    "num_layers": model["num_layers"],
                    "hidden_size": model["hidden_size"],
                    "num_parameters": model["num_parameters"],
                }
                row.update(metric_columns("reuse", conditions["full_reuse"]))
                row.update(metric_columns("exact", conditions["full_compute"]))
                row.update(metric_columns("frozen", conditions["frozen"]))
                rows.append(row)
    return rows


def cache_age_rows(summary: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    for cell_name, cell in summary["cells"].items():
        for relative in cell["source_files"]["matrix"]:
            source = load(ROOT / relative)
            if source.get("protocol") != "motivation_capacity_v2_cache_version_matrix":
                raise ValueError(f"cache-age matrix differs: {relative}")
            for point in source["points"]:
                conditions = point["summary"]["conditions"]
                row: dict[str, object] = {
                    "cell": cell_name,
                    "dataset": cell["dataset"],
                    "tier": cell["tier"],
                    "seed": source["seed"],
                    "current_model_version": source["current_t"],
                    "cache_version": point["stale_t"],
                    "cache_age": point["cache_age"],
                    "model_delta_relative": point["dtheta_rel"],
                    "evaluation_users": point["n_users"],
                }
                row.update(metric_columns("reuse", conditions["reuse"]))
                row.update(metric_columns("exact", conditions["full_compute"]))
                row.update(metric_columns("frozen", conditions["frozen"]))
                rows.append(row)
    return rows


def d1_rows(summary: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    for cell_name, cell in summary["cells"].items():
        source_files = cell["source_files"]
        paths = [source_files["discovery"], *source_files["heldout_seed_replication"]]
        for relative in paths:
            source = load(ROOT / relative)
            target = source["selection"]["fidelity_targets"]["0.5"]
            selected_name = target["selected"]
            selected = source["test"]["configs"][selected_name]
            reuse = source["test"]["configs"]["reuse"]
            exact = source["test"]["configs"]["recompute"]
            row: dict[str, object] = {
                "cell": cell_name,
                "seed": source["seed"],
                "study_stage": source["study_stage"],
                "source_version": source["stale_t"],
                "target_version": source["model_t"],
                "selected_action": selected_name,
                "fidelity_target": 0.5,
                "cache_fidelity_recovery": selected[
                    "cache_fidelity_recovery"
                ],
                "measured_cost_ratio_to_exact": selected[
                    "migration_ratio_to_recompute"
                ],
                "measured_migration_ms_per_user": selected[
                    "migration_ms_per_user"
                ],
                "exact_migration_ms_per_user": exact[
                    "migration_ms_per_user"
                ],
            }
            row.update(metric_columns("reuse", reuse["metrics"]))
            row.update(metric_columns("d1", selected["metrics"]))
            row.update(metric_columns("exact", exact["metrics"]))
            rows.append(row)
    return rows


def structural_rows(summary: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    for cell in summary["cells"]:
        for family, result in cell["families"].items():
            rows.append(
                {
                    "cell": cell["cell"],
                    "seed": cell["seed"],
                    "family": family,
                    "selected_action": result["selected"],
                    "exact_fallback": result["exact_fallback"],
                    "probe_recovery": result[
                        "probe_cache_fidelity_recovery"
                    ],
                    "test_recovery": result[
                        "test_cache_fidelity_recovery"
                    ],
                    "probe_cost_ratio_to_exact": result[
                        "probe_cost_ratio_to_exact"
                    ],
                    "test_cost_ratio_to_exact": result[
                        "test_cost_ratio_to_exact"
                    ],
                    "probe_users": cell["probe_users"],
                    "test_users": cell["test_users"],
                    "probe_test_disjoint": cell["probe_test_disjoint"],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    capacity = load(
        ROOT / "results/motivation_scale/capacity_v2_summary.json"
    )
    d1 = load(
        ROOT
        / "results/motivation_scale/"
        "cohort_tiered_migration_v1_summary.json"
    )
    structural = load(
        ROOT
        / "results/baseline_foundation/"
        "d1_same_sla_development_v0_summary.json"
    )
    tables = {
        "m1_streaming_versions.tsv": semantic_rows(capacity),
        "m1_cache_age.tsv": cache_age_rows(capacity),
        "d1_cost_quality.tsv": d1_rows(d1),
        "d1_same_sla_structural.tsv": structural_rows(structural),
    }
    for name, rows in tables.items():
        write_tsv(args.output_dir / name, rows)
    manifest = {
        "schema": "evokv_semantic_baseline_tables_v0",
        "status": "pass",
        "scientific_result": False,
        "artifact_role": "table_export_from_existing_protocol_results",
        "tables": {
            name: {
                "path": str(args.output_dir / name),
                "rows": len(rows),
                "bytes": (args.output_dir / name).stat().st_size,
                "sha256": file_sha256(args.output_dir / name),
            }
            for name, rows in tables.items()
        },
        "boundaries": {
            "m1_streaming_versions": (
                "3 datasets x 3 model tiers x 4 seeds x 6 streamed model versions"
            ),
            "m1_cache_age": (
                "theta11 current model against 11 theta0..theta10 cache ages"
            ),
            "d1_cost_quality": (
                "theta0-to-theta11 primary 50-percent cache-fidelity target"
            ),
            "d1_same_sla_structural": (
                "nine discovery checkpoints; development comparator foundation"
            ),
        },
    }
    path = args.output_dir / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
