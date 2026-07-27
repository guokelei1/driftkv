from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL = "cohortkv_single_config_stage1_frozen_v1"
SOURCE_PROTOCOL = "cohortkv_single_config_stage1_frontier_v1"
SOURCE_RESULT = Path(
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage1_frontier_seed0.json"
)
OUTPUT = Path("configs/cohortkv_single_config_v1/stage1_frontier_summary.json")
SOURCE_VERSIONS = ("theta0", "theta4", "theta10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", default=str(SOURCE_RESULT))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pair(pair: dict) -> None:
    if len(pair["selection_points"]) != 59:
        raise ValueError("each source pair must contain 59 selection points")
    selective = [
        point
        for point in pair["selection_points"]
        if point["method"] == "selective_contiguous"
    ]
    intervals = {
        (
            point["configuration"]["start_layer"],
            point["configuration"]["end_layer"],
        )
        for point in selective
    }
    if len(selective) != 53 or len(intervals) != 53:
        raise ValueError("selective interval grid is incomplete")
    if len(pair["selection"]["per_user"]) != 60:
        raise ValueError("program-selection role must contain 60 users")
    if len(pair["certificate"]["per_user"]) != 60:
        raise ValueError("certificate role must contain 60 users")
    if any(
        record["evaluation_role"] != "program_selection"
        for record in pair["selection"]["per_user"]
    ):
        raise ValueError("selection records contain the wrong role")
    if any(
        record["evaluation_role"] != "certificate"
        for record in pair["certificate"]["per_user"]
    ):
        raise ValueError("certificate records contain the wrong role")


def summarize_pair(pair: dict) -> dict:
    validate_pair(pair)
    selection = pair["selection"]["summary"]
    selective = [
        (name, values)
        for name, values in selection.items()
        if name.startswith("selective_")
    ]
    def selection_key(item: tuple[str, dict]) -> tuple[float, float, int, str]:
        name, values = item
        start = int(name.split("_")[2].removeprefix("s"))
        return (
            values["worst_view_recovery"],
            -values["cost_ratio_to_exact"],
            -start,
            name,
        )

    profiled_name, profiled = max(
        selective,
        key=selection_key,
    )
    profiled_point = next(
        point
        for point in pair["selection_points"]
        if point["method"] == "selective_contiguous"
        and point["configuration"]["m"]
        == int(profiled_name.split("_")[1].removeprefix("m"))
        and point["configuration"]["start_layer"]
        == int(profiled_name.split("_")[2].removeprefix("s"))
        and point["configuration"]["end_layer"]
        == int(profiled_name.split("_")[3].removeprefix("e"))
    )
    compiled = selection["compiled"]
    dominates = all(
        compiled["cost_ratio_to_exact"] < values["cost_ratio_to_exact"]
        and compiled["worst_view_recovery"] > values["worst_view_recovery"]
        for _, values in selective
    )
    certificate = pair["certified_selective_action"]
    if certificate["certificate_passed"]:
        raise ValueError("Stage 1 result unexpectedly certified a selective action")
    if certificate["action_name"] != "recompute":
        raise ValueError("failed selective certification must fall back to exact")
    if profiled_point["configuration"]["start_layer"] != 0:
        source_representations = [
            "old_kv_fp16",
            "transition_hidden_fp16",
            "raw_history",
        ]
    else:
        source_representations = ["old_kv_fp16", "raw_history"]
    return {
        "source_version": pair["source_version"],
        "target_version": pair["target_version"],
        "compiled": {
            "cost_ratio_to_exact": compiled["cost_ratio_to_exact"],
            "cache_recovery": compiled["cache_recovery"],
            "score_recovery": compiled["score_recovery"],
            "top100_recovery": compiled["top100_recovery"],
            "worst_view_recovery": compiled["worst_view_recovery"],
        },
        "profiled_selective_action": {
            "name": profiled_name,
            "configuration": profiled_point["configuration"],
            "cost_ratio_to_exact": profiled["cost_ratio_to_exact"],
            "cache_recovery": profiled["cache_recovery"],
            "score_recovery": profiled["score_recovery"],
            "top100_recovery": profiled["top100_recovery"],
            "worst_view_recovery": profiled["worst_view_recovery"],
            "source_representations": source_representations,
            "publishable_sync_action": False,
            "system_role": "frozen_diagnostic_external_baseline",
        },
        "certificate": {
            "passed": False,
            "published_action": "recompute",
            "reason": certificate["selection_reason"],
        },
        "compiled_strictly_dominates_all_53_selective_points": dominates,
    }


def build_summary(source_path: Path, source: dict) -> dict:
    if source.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError("Stage 1 source protocol mismatch")
    if source.get("status") != "stage1_complete":
        raise ValueError("Stage 1 source result is incomplete")
    if source.get("labels_used") is not False:
        raise ValueError("Stage 1 result must be label-free")
    if source.get("final_test_evaluated") is not False:
        raise ValueError("Stage 1 must not evaluate final-test users")
    if source.get("role_counts") != {
        "fit": 40,
        "program_selection": 60,
        "certificate": 60,
        "final_test": 522,
    }:
        raise ValueError("Stage 1 role counts differ from the frozen split")
    if len(source["rq3_frontier"]["selection_points"]) != 177:
        raise ValueError("Stage 1 aggregate must contain 177 points")
    pairs = sorted(
        (summarize_pair(pair) for pair in source["pairs"]),
        key=lambda pair: SOURCE_VERSIONS.index(pair["source_version"]),
    )
    if tuple(pair["source_version"] for pair in pairs) != SOURCE_VERSIONS:
        raise ValueError("Stage 1 source-version coverage is incomplete")
    if not all(
        pair["compiled_strictly_dominates_all_53_selective_points"]
        for pair in pairs
    ):
        raise ValueError("compiled does not dominate every selective point")
    configurations = {
        tuple(
            pair["profiled_selective_action"]["configuration"][key]
            for key in ("m", "start_layer", "end_layer")
        )
        for pair in pairs
    }
    if configurations != {(12, 0, 11)}:
        raise ValueError("profiled system actions differ across source pairs")
    return {
        "protocol": PROTOCOL,
        "status": "stage1_frozen",
        "study_stage": "single_configuration_seed0_development",
        "source_result": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "protocol": SOURCE_PROTOCOL,
        },
        "parent_blueprint": {
            **source["blueprint"],
            "hash_scope": (
                "blueprint bytes used by the measurement before the "
                "post-Stage-1 downstream amendment"
            ),
        },
        "workload_content_sha256": source["workload_manifest"][
            "content_sha256"
        ],
        "measurement_boundary": {
            "execution": "resident FP32 algorithmic reference",
            "selection_users": 60,
            "certificate_users": 60,
            "final_test_evaluated": False,
            "labels_used": False,
            "selection_points": 177,
            "selective_intervals_per_pair": 53,
        },
        "pairs": pairs,
        "downstream_rule": {
            "publication_action": (
                "no selective interval passed the primary semantic contract; "
                "exact is the publishable fallback"
            ),
            "system_baseline_action": (
                "Stage 4 still measures the highest-worst-view frozen "
                "program-selection action m12/layers0-11 as a diagnostic "
                "external baseline through the common destination transaction"
            ),
            "system_baseline_source_representations": [
                "old_kv_fp16",
                "raw_history",
            ],
            "transition_hidden_bytes_for_frozen_system_baseline": 0,
            "claim_boundary": (
                "the failed certificate is reported; the diagnostic "
                "selective row is never called a publishable synchronized "
                "target or a certified action"
            ),
        },
        "open_before_stage4": [
            "reapply numeric correctness to serialized FP16 old K/V and FP16 output",
            "independently tune the profiled selective runtime per destination and GPU count",
            "retain exact as the publication fallback",
        ],
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    source_path = root / args.source_result
    output_path = root / args.output
    source = json.loads(source_path.read_text())
    payload = canonical_json_bytes(build_summary(Path(args.source_result), source))
    if args.check:
        if not output_path.is_file() or output_path.read_bytes() != payload:
            raise RuntimeError("Stage 1 frozen summary differs from the source result")
        status = "verified"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
        status = "frozen"
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "status": status,
                "output": str(Path(args.output)),
                "sha256": sha256_bytes(payload),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
