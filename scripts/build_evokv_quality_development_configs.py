from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-input", type=Path, required=True)
    parser.add_argument("--edge-summary", type=Path, required=True)
    parser.add_argument("--quality-roles", type=Path, required=True)
    parser.add_argument("--schedule-output", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("three_edge", "warmup_plus_three"),
        default="three_edge",
    )
    parser.add_argument(
        "--schedule-template",
        type=Path,
        default=Path(
            "configs/evokv_foundation/"
            "xp_qk_multiversion_prequential3_development_v1.json"
        ),
    )
    parser.add_argument(
        "--benchmark-template",
        type=Path,
        default=Path(
            "configs/evokv_baselines/"
            "x_qk_xp_multiversion_two_gpu_baseline_v1.json"
        ),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path, parent: Path) -> str:
    return os.path.relpath(path.resolve(), parent.resolve())


def atomic_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"generated config differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    required = (
        args.edge_input,
        args.edge_summary,
        args.quality_roles,
        args.schedule_template,
        args.benchmark_template,
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("quality development config input is absent")
    edge_summary = json.loads(args.edge_summary.read_text())
    role_document = json.loads(args.quality_roles.read_text())
    if (
        edge_summary.get("status") != "pass"
        or edge_summary.get("scientific_result") is not False
        or edge_summary.get("artifact", {}).get("file_sha256")
        != file_sha256(args.edge_input)
        or role_document.get("scientific_result") is not False
    ):
        raise ValueError("quality development corpus binding differs")
    schedule = json.loads(args.schedule_template.read_text())
    training_users = int(role_document["roles"]["theta01"]["count"])
    qualification_users = int(
        role_document["roles"]["qualification"]["count"]
    )
    schedule["edge_inputs"] = relative_path(
        args.edge_input, args.schedule_output.parent
    )
    schedule["edge_summary"] = relative_path(
        args.edge_summary, args.schedule_output.parent
    )
    schedule["stack_identity"] = (
        f"xp_qk_prequential3_train{training_users}_qual"
        f"{qualification_users}_e1_fixed010_development_v0"
    )
    if args.mode == "warmup_plus_three":
        schedule["updates"].append(
            {
                "history_end": 88,
                "source_version": 3,
                "target_version": 4,
                "update_end": 96,
            }
        )
        schedule["prequential_evaluations"].append(
            {
                "evaluation_end": 104,
                "history_end": 96,
                "model_version": 4,
            }
        )
        schedule["stack_identity"] = (
            f"xp_qk_stream_aligned_warmup_train{training_users}_qual"
            f"{qualification_users}_e1_fixed010_development_v1"
        )
    schedule["training"]["epochs_per_update"] = 1
    schedule["training"]["optimizer_state_continuity"] = (
        "continuous_within_round"
    )
    schedule["training_data_revision"] = {
        "theta01_users": int(role_document["roles"]["theta01"]["count"]),
        "qualification_users": int(
            role_document["roles"]["qualification"]["count"]
        ),
        "quality_roles_sha256": file_sha256(args.quality_roles),
        "purpose": (
            "hold update-step scale approximately fixed while replacing "
            "three repeated passes with one pass over more independent users"
        ),
    }
    benchmark = copy.deepcopy(
        json.loads(args.benchmark_template.read_text())
    )
    benchmark["benchmark_id"] = (
        f"x_qk_xp_quality_train{training_users}_qual{qualification_users}"
        "_e1_two_gpu_development_v0"
    )
    benchmark["date"] = "2026-08-01"
    benchmark["purpose"] = (
        "diagnose and establish a generalizing four-version D1 quality "
        "chain before integrated D1 and D2 evaluation"
    )
    benchmark["status"] = "quality_chain_candidate_ready"
    if args.mode == "warmup_plus_three":
        benchmark["benchmark_id"] = (
            f"x_qk_xp_quality_stream_aligned_train{training_users}_qual"
            f"{qualification_users}_e1_two_gpu_development_v1"
        )
        benchmark["purpose"] = (
            "evaluate three ordinary version edges after one explicit "
            "bootstrap-to-streaming-objective warmup edge"
        )
        benchmark["status"] = "stream_aligned_quality_chain_candidate_ready"
    benchmark["data"]["quality_roles"] = {
        "path": relative_path(args.quality_roles, Path.cwd()),
        "sha256": file_sha256(args.quality_roles),
        "train_role": "theta01",
        "quality_role": "qualification",
        "post_base_users_disjoint": True,
        "disjoint_from_system_het_roles": True,
    }
    benchmark["data"]["fixed_edge_inputs"] = {
        "path": relative_path(args.edge_input, Path.cwd()),
        "sha256": file_sha256(args.edge_input),
        "summary_path": relative_path(args.edge_summary, Path.cwd()),
        "summary_sha256": file_sha256(args.edge_summary),
    }
    benchmark["quality_chain"] = {
        "schedule_path": relative_path(
            args.schedule_output, Path.cwd()
        ),
        "epochs_per_update": 1,
        "training_users": int(
            role_document["roles"]["theta01"]["count"]
        ),
        "qualification_users": int(
            role_document["roles"]["qualification"]["count"]
        ),
        "candidate_only": True,
        "method_selection_allowed": False,
    }
    if args.mode == "warmup_plus_three":
        benchmark["model"]["checkpoint_versions"] = [0, 1, 2, 3, 4]
        benchmark["training_edges"] = copy.deepcopy(schedule["updates"])
        benchmark["prequential_windows"] = [
            {
                "model_version": value["model_version"],
                "history_end": value["history_end"],
                "evaluation_end": value["evaluation_end"],
            }
            for value in schedule["prequential_evaluations"]
        ]
        benchmark["d1_edge_evaluation_windows"] = [
            {
                "source_version": 1,
                "target_version": 2,
                "history_end": 80,
                "evaluation_end": 88,
            },
            {
                "source_version": 2,
                "target_version": 3,
                "history_end": 88,
                "evaluation_end": 96,
            },
            {
                "source_version": 3,
                "target_version": 4,
                "history_end": 96,
                "evaluation_end": 104,
            },
        ]
        benchmark["quality_chain"].update(
            {
                "bootstrap_checkpoint_version": 0,
                "warmup_edge": {
                    "source_version": 0,
                    "target_version": 1,
                    "history_end": 64,
                    "update_end": 72,
                    "included_in_d1_evidence": False,
                    "reason": (
                        "align the cooccurrence-expanded bootstrap with "
                        "the next-item streaming objective"
                    ),
                },
                "evaluated_edges": [[1, 2], [2, 3], [3, 4]],
            }
        )
    atomic_json(args.schedule_output, schedule)
    atomic_json(args.benchmark_output, benchmark)
    print(
        json.dumps(
            {
                "benchmark": str(args.benchmark_output),
                "schedule": str(args.schedule_output),
                "status": "complete",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
