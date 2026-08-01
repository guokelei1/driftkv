from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

TRAINING_PROTOCOL = "evokv_xp_prequential_stream_training_development_v1"
D1_PROTOCOL = "evokv_xp_d1_quality_development_v0"
ACTION_PLAN_PROTOCOL = "evokv_xp_d1_action_plan_v2_development_v0"
M2_PROTOCOL = "evokv_xp_m2_append_aware_lookup_development_v0"
D1_METHODS = (
    "all_reuse",
    "compiled_direct_oldkv",
    "mixed_fixed20",
    "all_exact",
)


def recommendation_recovery(
    methods: dict[str, object],
) -> dict[str, object]:
    exact = methods["all_exact"]["recommendation"]
    reuse = methods["all_reuse"]["recommendation"]
    directions = {
        "sampled_cross_entropy": -1.0,
        "hit_rate_at_10": 1.0,
        "ndcg_at_10": 1.0,
        "mean_reciprocal_rank": 1.0,
    }
    output = {}
    for method, values in methods.items():
        metrics = values["recommendation"]
        rows = {}
        for metric, direction in directions.items():
            denominator = direction * (
                float(exact[metric]) - float(reuse[metric])
            )
            numerator = direction * (
                float(metrics[metric]) - float(reuse[metric])
            )
            rows[metric] = {
                "method_minus_exact": float(metrics[metric])
                - float(exact[metric]),
                "reuse_to_exact_recovery": (
                    None if denominator == 0.0 else numerator / denominator
                ),
                "recovery_denominator_identifiable": (
                    denominator != 0.0
                ),
                "development_arithmetic_only": True,
            }
        output[method] = rows
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-label", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def directory_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in path.rglob("*")
        if candidate.is_file()
    )


def relative_or_resolved(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def artifact(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size < 1:
        raise FileNotFoundError(path)
    return {
        "path": relative_or_resolved(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def resolve_bound_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("bound artifact path is invalid")
    return Path(value)


def summarize_training(path: Path) -> dict[str, object]:
    value = load_json(path)
    updates = value.get("updates")
    if (
        value.get("protocol") != TRAINING_PROTOCOL
        or value.get("status") != "complete"
        or value.get("downstream_d1_d2_gate_passed") is not True
        or not isinstance(updates, list)
        or len(updates) != 3
        or len(value.get("prequential_evaluations", [])) != 4
        or any(update.get("target_checkpoint_committed") is not True for update in updates)
    ):
        raise ValueError("XP multiversion training result differs")
    edge_rows = []
    for update in updates:
        quality = update["target_prequential_evaluation"][
            "quality_observation"
        ]["metrics"]
        diagnostic = update["target_prequential_evaluation"][
            "tuning_observation"
        ]["metrics"]
        training_window = update["training_window"]
        edge_rows.append(
            {
                "source_version": update["source_version"],
                "target_version": update["target_version"],
                "training_history_end": training_window["history_end"],
                "training_update_end": training_window["update_end"],
                "evaluation_history_end": update[
                    "target_prequential_evaluation"
                ]["history_end"],
                "evaluation_end": update[
                    "target_prequential_evaluation"
                ]["evaluation_end"],
                "training_targets": sum(
                    int(epoch["global_targets"])
                    for epoch in update["training"]["epochs"]
                ),
                "training_wall_seconds": update["training"]["wall_seconds"],
                "diagnostic_sampled_cross_entropy": diagnostic[
                    "sampled_cross_entropy"
                ],
                "quality_sampled_cross_entropy": quality[
                    "sampled_cross_entropy"
                ],
                "quality_ndcg_at_10": quality["ndcg_at_10"],
                "quality_hit_rate_at_10": quality["hit_rate_at_10"],
                "quality_used_for_selection_or_gate": False,
                "checkpoint_admission": update["checkpoint_admission"],
                "checkpoint": update["checkpoint"],
            }
        )
    return {
        "artifact": artifact(path),
        "stack_identity": value["stack_identity"],
        "model": value["model"],
        "learning_rate_policy": value["learning_rate_policy"],
        "prequential_evaluations": value["prequential_evaluations"],
        "edges": edge_rows,
        "total_wall_seconds": value["execution"]["total_wall_seconds"],
    }


def summarize_d1(result_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    edges = []
    retained = []
    for source_version in range(3):
        target_version = source_version + 1
        name = f"theta{source_version}_to_theta{target_version}"
        path = result_root / "d1" / f"{name}.json"
        value = load_json(path)
        edge = value.get("edge")
        if (
            value.get("protocol") != D1_PROTOCOL
            or value.get("status") != "complete"
            or not isinstance(edge, dict)
            or edge.get("source_version") != source_version
            or edge.get("target_version") != target_version
            or edge.get("history_end") != 72 + source_version * 8
            or edge.get("update_end") != 80 + source_version * 8
            or edge.get("training_window", {}).get("history_end")
            != 64 + source_version * 8
            or edge.get("training_window", {}).get("update_end")
            != 72 + source_version * 8
        ):
            raise ValueError(f"XP D1 edge result differs: {path}")
        for role in ("probe", "qualification_test"):
            audit = value["roles"][role]["audit"]
            if (
                audit.get("history_end") != edge["history_end"]
                or audit.get("update_end") != edge["update_end"]
            ):
                raise ValueError(
                    f"XP D1 {role} window binding differs: {path}"
                )
        quality = value["quality"]["qualification_test"]
        methods = quality["methods"]
        if set(methods) != set(D1_METHODS):
            raise ValueError(f"XP D1 method order differs: {path}")
        action_binding = value["bindings"]["action_plan"]
        action_path = resolve_bound_path(action_binding["path"])
        action = load_json(action_path)
        if (
            action.get("protocol") != ACTION_PLAN_PROTOCOL
            or action.get("source_version") != source_version
            or action.get("target_version") != target_version
            or file_sha256(action_path) != action_binding["sha256"]
            or action.get("records_sha256") != action_binding["records_sha256"]
        ):
            raise ValueError(f"XP D1 ActionPlan binding differs: {action_path}")
        method_rows = {}
        for method in D1_METHODS:
            current = methods[method]
            method_rows[method] = {
                "recommendation": current["recommendation"],
                "cache_fidelity": current["cache_fidelity"],
                "gpu_cost": current["gpu_cost"],
            }
        quality_selection = value["roles"]["qualification_test"][
            "mixed_fixed20_selection"
        ]
        logical_exact_fraction = {
            "all_reuse": 0.0,
            "compiled_direct_oldkv": 0.0,
            "mixed_fixed20": quality_selection[
                "actual_retained_token_fraction"
            ],
            "all_exact": 1.0,
        }
        program_path = resolve_bound_path(value["bindings"]["program"]["path"])
        if file_sha256(program_path) != value["bindings"]["program"]["sha256"]:
            raise ValueError(f"XP D1 program binding differs: {program_path}")
        retained.extend((artifact(program_path), artifact(action_path)))
        edges.append(
            {
                "edge": name,
                "artifact": artifact(path),
                "records": quality["records"],
                "methods": method_rows,
                "quality_mixed_selection": quality_selection,
                "logical_exact_recompute_fraction": logical_exact_fraction,
                "recommendation_recovery": recommendation_recovery(
                    method_rows
                ),
                "action_plan_selection": action["selection"],
                "action_plan_records": action["record_count"],
                "mixed_timing_is_end_to_end": value["gpu_cost"][
                    "mixed_cost_is_end_to_end"
                ],
                "claim_boundary": (
                    "large-model analytic direct-old-K/V bridge diagnostic; "
                    "cross-dataset fitted D1 remains the primary D1 evidence"
                ),
            }
        )
    return edges, retained


def summarize_m2(path: Path) -> dict[str, object]:
    value = load_json(path)
    cells = value.get("cells")
    if (
        value.get("protocol") != M2_PROTOCOL
        or value.get("world_size") != 2
        or value.get("append_tokens_per_record") != 32
        or not isinstance(cells, list)
        or len(cells) != 4
    ):
        raise ValueError("XP M2 lookup characterization differs")
    rows = []
    for cell in cells:
        selection = cell["retained_budget"]
        retained = cell["retained_only"]
        complete = cell["complete_wave"]
        rows.append(
            {
                "fraction_requested": selection["fraction_requested"],
                "fraction_realized": selection["fraction_realized"],
                "selected_records": selection["selected_records"],
                "selected_retained_tokens": selection[
                    "selected_retained_tokens"
                ],
                "retained_requested_tokens": retained["requested_tokens"],
                "retained_remote_tokens": retained["remote_tokens"],
                "retained_response_bytes": retained[
                    "h1536_fp32_response_bytes"
                ],
                "retained_collectives_per_rank": retained[
                    "all_to_all_collective_invocations_per_rank"
                ],
                "retained_median_max_rank_seconds": retained["timing"][
                    "median_max_rank_seconds"
                ],
                "complete_requested_tokens": complete["requested_tokens"],
                "complete_remote_tokens": complete["remote_tokens"],
                "complete_response_bytes": complete[
                    "h1536_fp32_response_bytes"
                ],
                "complete_collectives_per_rank": complete[
                    "all_to_all_collective_invocations_per_rank"
                ],
                "complete_median_max_rank_seconds": complete["timing"][
                    "median_max_rank_seconds"
                ],
            }
        )
    return {
        "artifact": artifact(path),
        "benchmark_id": value["benchmark_id"],
        "records": value["records"],
        "all_exact_target_lookup_tokens": value[
            "all_exact_target_lookup_tokens"
        ],
        "append_only_common": value["append_only_common"],
        "cells": rows,
        "claim_boundary": value["claim_boundary"],
    }


def retention_ledger(
    result_root: Path,
    checkpoint_root: Path,
    log_root: Path,
    output: Path,
) -> dict[str, object]:
    root_files = {
        "input_hashes.tsv",
        "preflight.json",
        "preflight_latest.json",
        "semantic_evidence_audit.json",
        "xp_multiversion_training.json",
        "m2_append_aware_lookup.json",
    }
    table_files = {
        "manifest.json",
        "m1_streaming_versions.tsv",
        "m1_cache_age.tsv",
        "d1_cost_quality.tsv",
        "d1_same_sla_structural.tsv",
    }
    unknown = []
    for path in result_root.rglob("*"):
        if not path.is_file() or path.resolve() == output.resolve():
            continue
        relative = path.relative_to(result_root)
        parts = relative.parts
        admitted = (
            (len(parts) == 1 and parts[0] in root_files)
            or (
                len(parts) == 2
                and parts[0] == "semantic_tables"
                and parts[1] in table_files
            )
            or (
                len(parts) == 2
                and parts[0] == "xp_multiversion_ledgers"
                and parts[1].startswith("version_")
                and parts[1].endswith(".json")
            )
            or (
                len(parts) == 2
                and parts[0] == "d1"
                and parts[1].startswith("theta")
                and parts[1].endswith((".json", ".pt"))
            )
        )
        if not admitted:
            unknown.append(relative_or_resolved(path))
    checkpoint_files = []
    admitted_checkpoint_names = {
        "manifest.json",
        "dense.pt",
        "projection.pt",
        "embedding_rank_00000.pt",
        "embedding_rank_00001.pt",
        "active_bitmap_rank_00000.pt",
        "active_bitmap_rank_00001.pt",
    }
    for path in checkpoint_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(checkpoint_root)
        if (
            len(relative.parts) != 2
            or relative.parts[0] not in {"theta_1", "theta_2", "theta_3"}
            or relative.parts[1] not in admitted_checkpoint_names
        ):
            unknown.append(relative_or_resolved(path))
        checkpoint_files.append(relative_or_resolved(path))
    log_files = []
    if log_root.exists():
        for path in log_root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix != ".log":
                unknown.append(relative_or_resolved(path))
            log_files.append(relative_or_resolved(path))
    if unknown:
        raise ValueError(f"unclassified durable round artifacts: {unknown}")
    return {
        "allowlist_validation": "pass",
        "checkpoint_files": checkpoint_files,
        "log_files": log_files,
        "unclassified_files": [],
    }


def main() -> None:
    args = parse_args()
    result_root = args.result_root
    checkpoint_root = args.checkpoint_root
    log_root = args.log_root
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    audit_path = result_root / "semantic_evidence_audit.json"
    table_manifest_path = result_root / "semantic_tables" / "manifest.json"
    training_path = result_root / "xp_multiversion_training.json"
    m2_path = result_root / "m2_append_aware_lookup.json"
    audit = load_json(audit_path)
    tables = load_json(table_manifest_path)
    if (
        audit.get("schema") != "evokv_d1_baseline_evidence_audit_v0"
        or audit.get("status") != "pass"
        or tables.get("schema") != "evokv_semantic_baseline_tables_v0"
        or tables.get("status") != "pass"
    ):
        raise ValueError("semantic evidence package differs")
    d1_edges, d1_retained = summarize_d1(result_root)
    retained_layout = retention_ledger(
        result_root,
        checkpoint_root,
        log_root,
        args.output,
    )
    checkpoint_bytes = directory_bytes(checkpoint_root)
    result_bytes_before_summary = directory_bytes(result_root)
    log_bytes = directory_bytes(log_root) if log_root.exists() else 0
    output = {
        "schema": "evokv_d1_d2_baseline_round_summary_v0",
        "status": "complete",
        "scientific_result": False,
        "formal_design2_result": False,
        "round_label": args.round_label,
        "semantic_evidence": {
            "audit": artifact(audit_path),
            "tables": {
                "manifest": artifact(table_manifest_path),
                "rows": {
                    name: descriptor["rows"]
                    for name, descriptor in tables["tables"].items()
                },
            },
            "training_chains": audit["checkpoints"]["chains"],
            "checkpoint_files_audited": audit["checkpoints"]["files"],
        },
        "xp_multiversion": summarize_training(training_path),
        "xp_d1_bridge": d1_edges,
        "m2_append_aware_lookup": summarize_m2(m2_path),
        "retention": {
            "policy": "keep_irreplaceable_or_high_reuse_value_assets_only",
            "checkpoint_root": relative_or_resolved(checkpoint_root),
            "checkpoint_bytes": checkpoint_bytes,
            "compact_result_root": relative_or_resolved(result_root),
            "result_bytes_before_summary": result_bytes_before_summary,
            "log_root": relative_or_resolved(log_root),
            "log_bytes": log_bytes,
            "retained_d1_programs_and_plans": d1_retained,
            "full_kv_payloads_retained": 0,
            "candidate_checkpoint_copies_retained": 0,
            "fit_probe_tensor_dumps_retained": 0,
            "reconstructible_staging_layouts_retained": 0,
            "durable_layout": retained_layout,
        },
        "next_dependency": (
            "freeze the formal D2 mechanism matrix against these ActionPlans; "
            "do not compare it with an older stack revision"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": output["status"],
                "d1_edges": len(d1_edges),
                "m2_cells": len(output["m2_append_aware_lookup"]["cells"]),
                "checkpoint_gib": checkpoint_bytes / 1024**3,
                "full_kv_payloads_retained": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
