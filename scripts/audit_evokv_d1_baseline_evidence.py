from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
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


def artifact(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size < 1:
        raise FileNotFoundError(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def source_paths(cell: dict[str, Any]) -> list[Path]:
    source = cell.get("source_files")
    if not isinstance(source, dict):
        raise ValueError("summary cell source_files is absent")
    values: list[str] = []
    for item in source.values():
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, list) and all(
            isinstance(value, str) for value in item
        ):
            values.extend(item)
        else:
            raise ValueError("summary source_files differs")
    return [ROOT / value for value in values]


def audit_summary(
    relative_path: str,
    protocol: str,
    expected_cells: int,
    expected_artifacts: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    path = ROOT / relative_path
    summary = load_json(path)
    cells = summary.get("cells")
    if (
        summary.get("protocol") != protocol
        or not isinstance(cells, dict)
        or len(cells) != expected_cells
    ):
        raise ValueError(f"summary contract differs: {path}")
    paths = [
        source
        for cell in cells.values()
        for source in source_paths(cell)
    ]
    unique = sorted(set(paths))
    if len(paths) != expected_artifacts or len(unique) != expected_artifacts:
        raise ValueError(f"summary artifact count differs: {path}")
    artifacts = [artifact(source) for source in unique]
    return artifact(path), artifacts


def audit_protocols(
    records: list[dict[str, object]],
    expected: dict[str, int],
) -> dict[str, int]:
    observed: dict[str, int] = {}
    for record in records:
        value = load_json(ROOT / str(record["path"]))
        protocol = value.get("protocol")
        if not isinstance(protocol, str):
            raise ValueError(f"protocol is absent: {record['path']}")
        observed[protocol] = observed.get(protocol, 0) + 1
    if observed != expected:
        raise ValueError(
            f"source protocol counts differ: {observed} != {expected}"
        )
    return observed


def audit_checkpoints() -> dict[str, object]:
    root = ROOT / "checkpoints/motivation_capacity_v2"
    cells = [
        f"{dataset}_{tier}"
        for dataset in ("kuai", "qb", "qk")
        for tier in ("small", "medium", "large")
    ]
    paths = [
        root / f"{cell}_seed{seed}" / f"theta_{version}.pt"
        for cell in cells
        for seed in range(4)
        for version in range(12)
    ]
    records = [artifact(path) for path in paths]
    digest = hashlib.sha256()
    total_bytes = 0
    for record in records:
        total_bytes += int(record["bytes"])
        digest.update(str(record["path"]).encode())
        digest.update(str(record["sha256"]).encode())
    return {
        "chains": 36,
        "versions_per_chain": 12,
        "files": len(records),
        "bytes": total_bytes,
        "path_and_content_sha256": digest.hexdigest(),
    }


def audit_same_sla() -> dict[str, object]:
    path = (
        ROOT
        / "results/baseline_foundation/"
        "d1_same_sla_development_v0_summary.json"
    )
    summary = load_json(path)
    cells = summary.get("cells")
    aggregate = summary.get("aggregate")
    if (
        summary.get("protocol")
        != "d1_same_sla_baseline_development_v0_summary"
        or summary.get("scientific_result") is not False
        or summary.get("formal_result") is not False
        or not isinstance(cells, list)
        or len(cells) != 9
        or aggregate
        != {
            "cells": 9,
            "exact_fallbacks": 23,
            "family_cells": 36,
            "non_exact_selections": 13,
        }
    ):
        raise ValueError("D1 same-SLA summary contract differs")
    records = []
    for cell in cells:
        descriptor = cell.get("artifact")
        if not isinstance(descriptor, dict):
            raise ValueError("D1 same-SLA artifact is absent")
        source = ROOT / str(descriptor["path"])
        observed = artifact(source)
        if (
            observed["bytes"] != descriptor.get("bytes")
            or observed["sha256"] != descriptor.get("sha256")
        ):
            raise ValueError(f"D1 same-SLA binding differs: {source}")
        raw = load_json(source)
        if (
            raw.get("protocol")
            != "d1_same_sla_baseline_development_v0"
            or raw.get("scientific_result") is not False
            or raw.get("formal_result") is not False
        ):
            raise ValueError(f"D1 same-SLA source differs: {source}")
        records.append(observed)
    return {
        "summary": artifact(path),
        "artifacts": records,
        "aggregate": aggregate,
        "evidence_status": "development_comparator_foundation",
    }


def main() -> None:
    args = parse_args()
    motivation_summary, motivation_sources = audit_summary(
        "results/motivation_scale/capacity_v2_summary.json",
        "motivation_capacity_v2_seed_summary",
        9,
        117,
    )
    d1_summary, d1_sources = audit_summary(
        "results/motivation_scale/cohort_tiered_migration_v1_summary.json",
        "cohort_tiered_migration_v1_seed_summary",
        9,
        36,
    )
    prefix_summary, prefix_sources = audit_summary(
        "results/motivation_scale/progressive_prefix_replay_v1_summary.json",
        "progressive_prefix_replay_v1_seed_summary",
        9,
        36,
    )
    result = {
        "schema": "evokv_d1_baseline_evidence_audit_v0",
        "status": "pass",
        "scientific_result": False,
        "artifact_role": "integrity_and_reuse_ledger",
        "families": {
            "m1_reuse_exact": {
                "summary": motivation_summary,
                "artifacts": motivation_sources,
                "protocol_counts": audit_protocols(
                    motivation_sources,
                    {
                        "motivation_capacity_v2_training": 36,
                        "motivation_capacity_v2_streaming_control": 36,
                        "motivation_capacity_v2_cache_version_matrix": 36,
                        "motivation_capacity_v2_operator_cost": 9,
                    },
                ),
                "training_chains": 36,
                "seeds_per_cell": 4,
            },
            "d1_active": {
                "summary": d1_summary,
                "artifacts": d1_sources,
                "protocol_counts": audit_protocols(
                    d1_sources,
                    {
                        "cohort_tiered_migration_discovery_v1": 9,
                        "cohort_tiered_migration_v1": 27,
                    },
                ),
            },
            "d1_progressive_prefix": {
                "summary": prefix_summary,
                "artifacts": prefix_sources,
                "protocol_counts": audit_protocols(
                    prefix_sources,
                    {"progressive_prefix_replay_v1": 36},
                ),
            },
            "d1_same_sla_structural": audit_same_sla(),
        },
        "checkpoints": audit_checkpoints(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "families": sorted(result["families"]),
                "checkpoint_files": result["checkpoints"]["files"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
