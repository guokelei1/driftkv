from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import freeze_cohortkv_single_config_v1 as freeze
import jsonschema

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path("configs/cohortkv_single_config_v1")
RESULT_DIR = Path(
    "results/system/cohortkv_single_config_full_chain_v1"
)
MANUSCRIPT = Path("paper/cohortkv/manuscript_v3_target_en.md")
STAGE1_RAW = RESULT_DIR / "stage1_frontier_seed0.json"
STAGE2_RAW = RESULT_DIR / "stage2_compiler_seed0.json"
STAGE3_RAW = RESULT_DIR / "stage3_operator_seed0.json"
STAGE4_RAW = RESULT_DIR / "stage4_system_seed0.json"
STAGE5_RAW = RESULT_DIR / "stage5_full_cow_theta0_theta1_seed0.json"
ACCOUNTING_RAW = (
    RESULT_DIR / "stage5_source_state_accounting_seed0.json"
)
STAGE49_RAW = (
    RESULT_DIR / "stage4_9_same_device_confirmation_seed0.json"
)
STAGE49_CANDIDATES = {
    freeze.STAGE49_COST_ENDPOINT: (
        RESULT_DIR / "stage4_9_token_debt_total10_seed0.json"
    ),
    freeze.STAGE49_SELECTED_CANDIDATE: (
        RESULT_DIR / "stage4_9_staggered_renewal_h12_seed0.json"
    ),
}
FROZEN_CONFIGS = {
    "stage1": CONFIG_DIR / "stage1_frontier_summary.json",
    "stage2": CONFIG_DIR / "stage2_compiler_summary.json",
    "stage3": CONFIG_DIR / "stage3_operator_summary.json",
    "stage4": CONFIG_DIR / "stage4_system_summary.json",
    "stage4_5": CONFIG_DIR / "stage4_5_source_plan_summary.json",
    "stage4_6_policy": (
        CONFIG_DIR / "stage4_6_lifecycle_policy.json"
    ),
    "stage4_6": CONFIG_DIR / "stage4_6_lifecycle_summary.json",
    "stage4_7": CONFIG_DIR / "stage4_7_organic_summary.json",
    "stage4_8_exact": (
        CONFIG_DIR / "stage4_8_exact_baseline.json"
    ),
}
OUTPUTS = {
    "correctness_report": (
        RESULT_DIR / "stage6_correctness_report_seed0.json"
    ),
    "timing_memory_report": (
        RESULT_DIR / "stage6_timing_memory_report_seed0.json"
    ),
    "paper_tables": RESULT_DIR / "stage6_paper_tables_seed0.json",
    "paper_figures": RESULT_DIR / "stage6_paper_figures_seed0.json",
    "artifact_to_claim": (
        RESULT_DIR / "stage6_artifact_to_claim_seed0.json"
    ),
    "negative_results_log": (
        RESULT_DIR / "stage6_negative_results_seed0.json"
    ),
    "tbd_disposition": (
        RESULT_DIR / "stage6_tbd_disposition_seed0.json"
    ),
    "code_snapshot_manifest": (
        RESULT_DIR / "stage6_code_snapshot_seed0.json"
    ),
}
FINAL_OUTPUT = RESULT_DIR / "final_summary_seed0.json"
SCHEMA = CONFIG_DIR / "result.schema.json"
BLUEPRINT = CONFIG_DIR / "blueprint.json"
WORKLOAD = CONFIG_DIR / "workload_manifest.json"
CODE_PATHS = (
    Path("scripts/freeze_cohortkv_single_config_v1.py"),
    Path("scripts/freeze_cohortkv_stage6.py"),
    Path("scripts/run_cohortkv_stage4_9_formal_confirmation.py"),
    Path("scripts/run_cohortkv_stage5_full_cow.py"),
    Path("src/hstu_kvcache/migration/destination.py"),
    Path("src/hstu_kvcache/migration/organic.py"),
    Path("src/hstu_kvcache/migration/organic_schedulers.py"),
    Path("src/hstu_kvcache/migration/rollout.py"),
    Path("src/hstu_kvcache/migration/stage45_oldkv.py"),
    Path("src/hstu_kvcache/migration/stage5_accounting.py"),
    Path("src/hstu_kvcache/migration/stage5_closure.py"),
    Path("tests/test_single_config_stage0.py"),
    Path("tests/test_stage5_full_cow.py"),
    Path("tests/test_stage6_freeze.py"),
)
STAGE6_PROTOCOL = freeze.STAGE6_PROTOCOL
SELECTED_CANDIDATE = freeze.STAGE49_SELECTED_CANDIDATE
COST_ENDPOINT = freeze.STAGE49_COST_ENDPOINT
EXPECTED_CLAIMS = {
    "rq2_deployed_compiler",
    "rq3_selective_frontier",
    "stage3_operator_correctness",
    "stage4_normalized_source_negative",
    "stage4_5_direct_oldkv_hot_hbm",
    "stage4_6_fixed_history_lifecycle",
    "stage4_9_corrected_growing_history",
    "stage5_transactional_closure",
    "source_state_accounting",
}
EXPECTED_NEGATIVE_SLOTS = {
    "selective_certificate",
    "normalized_capsule_system",
    "per_cache_threshold_router",
    "stage4_7_cache_fidelity",
    "stage4_9_state_movement",
    "cold_storage_scope",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage5-result", type=Path, default=STAGE5_RAW)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def relative_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"artifact is outside the repository: {path}") from error


def load_json(path: Path) -> dict[str, Any]:
    resolved = repo_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"required Stage 6 input is missing: {path}")
    value = json.loads(resolved.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Stage 6 input is not an object: {path}")
    return value


def descriptor(
    path: Path,
    protocol: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    relative = relative_path(repo_path(path))
    value = load_json(relative)
    observed_protocol = value.get("protocol")
    observed_status = value.get("status")
    if (
        not isinstance(observed_protocol, str)
        or not isinstance(observed_status, str)
        or (protocol is not None and observed_protocol != protocol)
        or (status is not None and observed_status != status)
    ):
        raise ValueError(f"artifact protocol or status differs: {relative}")
    resolved = repo_path(relative)
    return {
        "path": str(relative),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "protocol": observed_protocol,
        "status": observed_status,
    }


def output_descriptor(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def repository_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_code_snapshot() -> dict[str, Any]:
    files = []
    for path in CODE_PATHS:
        resolved = repo_path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"snapshot input is missing: {path}")
        files.append(
            {
                "path": str(path),
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    content_sha256 = sha256_bytes(canonical_json_bytes(files))
    return {
        "protocol": "cohortkv_single_config_stage6_code_snapshot_v1",
        "status": "frozen",
        "repository_commit": repository_commit(),
        "content_sha256": content_sha256,
        "files": files,
    }


def validate_frozen_inputs(
    values: dict[str, dict[str, Any]],
) -> None:
    expected = {
        "stage1": (
            "cohortkv_single_config_stage1_frozen_v1",
            "stage1_frozen",
        ),
        "stage2": (
            "cohortkv_single_config_stage2_frozen_v1",
            "stage2_frozen",
        ),
        "stage3": (
            "cohortkv_single_config_stage3_frozen_v1",
            "stage3_frozen",
        ),
        "stage4": (
            "cohortkv_single_config_stage4_frozen_v1",
            "stage4_frozen",
        ),
        "stage4_5": (
            "cohortkv_single_config_stage4_5_frozen_v1",
            "stage4_5_source_plan_frozen",
        ),
        "stage4_6_policy": (
            "cohortkv_single_config_stage4_6_policy_frozen_v1",
            "stage4_6_lifecycle_policy_frozen",
        ),
        "stage4_6": (
            "cohortkv_single_config_stage4_6_frozen_v1",
            "stage4_6_lifecycle_frozen",
        ),
        "stage4_7": (
            "cohortkv_single_config_organic_lifecycle_v1",
            "evidence_complete_development_gates_not_fully_passed",
        ),
        "stage4_8_exact": (
            "cohortkv_single_config_stage4_8_external_exact_baseline_v1",
            "complete",
        ),
    }
    for name, (protocol, status) in expected.items():
        value = values[name]
        if value.get("protocol") != protocol or value.get("status") != status:
            raise ValueError(f"frozen {name} identity differs")
    for name in ("stage1", "stage2", "stage3", "stage4"):
        source = values[name].get("source_result")
        if not isinstance(source, dict):
            raise ValueError(f"frozen {name} lacks source result")
        path = Path(str(source["path"]))
        if (
            sha256_file(repo_path(path)) != source.get("sha256")
            or load_json(path).get("protocol") != source.get("protocol")
        ):
            raise ValueError(f"frozen {name} raw result differs")


def build_rq3(
    raw: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    value = dict(raw["rq3_frontier"])
    value["profiled_selective_actions"] = [
        {
            "source_version": pair["source_version"],
            "action": pair["profiled_selective_action"]["configuration"],
            "certificate_passed": False,
            "publishable_sync_action": False,
            "system_role": pair["profiled_selective_action"][
                "system_role"
            ],
            "source_representations": pair[
                "profiled_selective_action"
            ]["source_representations"],
        }
        for pair in summary["pairs"]
    ]
    return value


def candidate_summary(
    candidate_name: str,
    value: dict[str, Any],
    artifact: dict[str, Any],
) -> dict[str, Any]:
    action_totals = Counter()
    depths = []
    for step in value["steps"]:
        action_totals.update(
            {
                "scheduled_exact": int(
                    step["actions"]["scheduled_exact"]
                ),
                "reusable_records": int(
                    step["actions"]["reusable_records"]
                ),
            }
        )
        depths.extend(
            int(row["migration_depth_after"])
            for row in step["lineage"]
            if row["migration_depth_after"] is not None
        )
    return {
        "candidate_name": candidate_name,
        "artifact": artifact,
        "primary_sum_u_over_sum_e": value["cumulative_gpu_cost"][
            "primary_sum_u_over_sum_e"
        ],
        "record_weighted_task_ratio": value[
            "record_weighted_task"
        ]["mixed_over_fresh_exact"],
        "scheduled_exact_records": action_totals["scheduled_exact"],
        "reusable_records": action_totals["reusable_records"],
        "maximum_observed_migration_depth": max(depths),
        "checks_passed": value["checks"]["all_passed"],
    }


def build_lifecycle(
    frozen: dict[str, dict[str, Any]],
    stage49_summary: dict[str, Any],
    stage49_values: dict[str, dict[str, Any]],
    stage49_descriptors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if (
        stage49_summary.get("protocol")
        != "cohortkv_single_config_stage4_9_same_device_confirmation_v2"
        or stage49_summary.get("status") != "complete"
        or stage49_summary.get("scientific_result") is not True
        or stage49_summary.get("checks", {}).get("all_passed") is not True
    ):
        raise ValueError("Stage 4.9 aggregate is invalid")
    entries = {
        value["candidate_name"]: value
        for value in stage49_summary["results"]
    }
    if set(entries) != set(STAGE49_CANDIDATES):
        raise ValueError("Stage 4.9 candidate coverage differs")
    for name, entry in entries.items():
        expected = stage49_descriptors[name]
        if (
            entry["path"] != expected["path"]
            or entry["sha256"] != expected["sha256"]
            or stage49_values[name].get("candidate_name") != name
            or stage49_values[name].get("checks", {}).get("all_passed")
            is not True
        ):
            raise ValueError(f"Stage 4.9 {name} binding differs")
    stage46 = frozen["stage4_6"]
    complete = stage46["complete_recursive_chain"]
    return {
        "fixed_history": {
            "summary_artifact": descriptor(
                FROZEN_CONFIGS["stage4_6"]
            ),
            "policy_artifact": descriptor(
                FROZEN_CONFIGS["stage4_6_policy"]
            ),
            "updates": complete["updates"],
            "records": complete["records"],
            "maximum_migration_depth": stage46["gate"][
                "maximum_migration_depth"
            ],
            "cumulative_gpu_cost_ratio": complete[
                "cumulative_gpu_cost"
            ]["ratio_to_all_exact"],
            "certificate_passed": stage46[
                "independent_certificate"
            ]["certificate"]["passed"],
            "scope": "fixed_history_hot_hbm_single_seed_development",
        },
        "corrected_growing_history": {
            "summary_artifact": descriptor(STAGE49_RAW),
            "selected_candidate": SELECTED_CANDIDATE,
            "cost_endpoint": COST_ENDPOINT,
            "selection_basis": (
                "freeze the preregistered bounded-renewal candidate "
                "without using recommendation labels; retain token debt "
                "only as the cost endpoint"
            ),
            "candidates": [
                candidate_summary(
                    name,
                    stage49_values[name],
                    stage49_descriptors[name],
                )
                for name in (COST_ENDPOINT, SELECTED_CANDIDATE)
            ],
            "target_append_excluded": stage49_summary[
                "measurement_boundary"
            ]["target_append_excluded"],
            "groupwise_host_staging": stage49_summary[
                "measurement_boundary"
            ]["groupwise_device_staging"],
            "state_movement_reported_separately": all(
                value["state_movement_outside_primary"][
                    "reported_separately"
                ]
                for value in stage49_summary["results"]
            ),
            "full_cohort_hbm_claim": stage49_summary[
                "measurement_boundary"
            ]["full_cohort_hbm_claim"],
            "end_to_end_state_movement_claim": stage49_summary[
                "measurement_boundary"
            ]["end_to_end_state_movement_claim"],
            "checks_passed": stage49_summary["checks"]["all_passed"],
        },
    }


def tbd_rules() -> dict[str, tuple[str, ...]]:
    return {
        "所有**尚未测得**的数值": ("editorial_disclosure",),
        "stable in the ⟨TBD⟩": (
            "measured_in_stage2_seed0",
            "measured_in_stage2_seed0",
        ),
        "frozen compiler is then run end-to-end": (
            "still_open_stage7",
            "delete_or_rewrite_editorial_placeholder",
        ),
        "| ⟨TBD⟩ | ⟨TBD⟩": (
            "still_open_stage7",
            "still_open_stage7",
            "still_open_stage7",
            "still_open_stage7",
            "still_open_stage7",
            "still_open_stage7",
        ),
        "Across seeds, the certificate selects": (
            "still_open_stage7",
            "still_open_stage7",
            "measured_in_stage2_seed0",
            "measured_in_stage2_seed0",
        ),
        "Cross-seed, cross-capacity": ("still_open_stage7",),
        "remains ⟨TBD after replication⟩": (
            "still_open_stage7",
        ),
        "capture/INT8 controls are still": (
            "deferred_optional_post_v1",
        ),
        "Artifact-to-claim map": (
            "replaced_by_stage6_artifact_map",
        ),
        "Frozen compiler replication": (
            "still_open_stage7",
            "still_open_stage7",
        ),
        "Selective-layer frontier replication": (
            "still_open_stage7",
            "still_open_stage7",
        ),
        "| SSD endpoint |": (
            "deferred_optional_post_v1",
            "deferred_optional_post_v1",
        ),
        "| Capsule economics |": (
            "deferred_optional_post_v1",
            "deferred_optional_post_v1",
        ),
        "| Escalation and failure injection |": (
            "measured_in_stage5",
            "measured_in_stage5",
        ),
    }


def build_tbd_disposition() -> dict[str, Any]:
    lines = repo_path(MANUSCRIPT).read_text().splitlines()
    rules = tbd_rules()
    entries = []
    for line_number, line in enumerate(lines, start=1):
        count = line.count("⟨TBD")
        if count == 0:
            continue
        matches = [
            statuses
            for pattern, statuses in rules.items()
            if pattern in line
        ]
        if len(matches) != 1 or len(matches[0]) != count:
            raise ValueError(
                f"unclassified manuscript TBD markers at line {line_number}"
            )
        entries.append(
            {
                "line": line_number,
                "context_sha256": sha256_bytes(
                    line.encode("utf-8")
                ),
                "marker_count": count,
                "dispositions": list(matches[0]),
            }
        )
    marker_count = sum(value["marker_count"] for value in entries)
    if marker_count != 0:
        raise ValueError(
            f"rewritten manuscript contains TBD markers: {marker_count}"
        )
    counts = Counter(
        disposition
        for entry in entries
        for disposition in entry["dispositions"]
    )
    return {
        "protocol": "cohortkv_single_config_stage6_manuscript_disposition_v2",
        "status": "complete",
        "target_manuscript": str(MANUSCRIPT),
        "target_manuscript_sha256": sha256_file(
            repo_path(MANUSCRIPT)
        ),
        "markers": marker_count,
        "disposition_counts": dict(sorted(counts.items())),
        "entries": entries,
        "all_markers_disposed": True,
    }


def build_negative_results() -> list[dict[str, str]]:
    return [
        {
            "slot": "selective_certificate",
            "observation": (
                "no DroidSpeak-adapted selective interval passes the "
                "frozen three-view certificate"
            ),
            "decision": (
                "retain the strongest interval as a non-publishable "
                "diagnostic and use exact as its fallback"
            ),
        },
        {
            "slot": "normalized_capsule_system",
            "observation": (
                "the 17.82-GB FP16 normalized-capsule source loses to "
                "paired exact at all six matched Stage-4 endpoints"
            ),
            "decision": (
                "retain it as a negative source-path result and use "
                "direct old K/V only in the declared hot-HBM regime"
            ),
        },
        {
            "slot": "per_cache_threshold_router",
            "observation": (
                "the per-cache threshold diagnostic creates exact-refresh "
                "waves from 0.15% to 65.10%"
            ),
            "decision": (
                "do not recover an adaptive-risk claim; retain the "
                "bounded scheduling negative result"
            ),
        },
        {
            "slot": "stage4_7_cache_fidelity",
            "observation": (
                "the canonical-date growing-history control misses its "
                "0.90 q90 cache-fidelity gate"
            ),
            "decision": (
                "retain it as completed mixed development evidence and "
                "do not use norm shift as a safety oracle"
            ),
        },
        {
            "slot": "stage4_9_state_movement",
            "observation": (
                "single-GPU formal confirmation requires separately "
                "reported groupwise host staging"
            ),
            "decision": (
                "claim only device-resident retained-prefix U/E; do not "
                "claim an 11-edge end-to-end state-movement result"
            ),
        },
        {
            "slot": "cold_storage_scope",
            "observation": (
                "no physical SSD, GDS, capture, or quantized full-cohort "
                "performance experiment is part of v1"
            ),
            "decision": (
                "mark those paths optional post-v1 and make no cold "
                "storage performance claim"
            ),
        },
    ]


def source_descriptor_map(
    frozen: dict[str, dict[str, Any]],
    stage5_path: Path,
) -> dict[str, dict[str, Any]]:
    values = {
        name: descriptor(path)
        for name, path in FROZEN_CONFIGS.items()
    }
    values.update(
        {
            "stage1_raw": descriptor(STAGE1_RAW),
            "stage2_raw": descriptor(STAGE2_RAW),
            "stage3_raw": descriptor(STAGE3_RAW),
            "stage4_raw": descriptor(STAGE4_RAW),
            "stage4_9_summary": descriptor(STAGE49_RAW),
            "stage5": descriptor(stage5_path),
            "accounting": descriptor(ACCOUNTING_RAW),
        }
    )
    for name, path in STAGE49_CANDIDATES.items():
        values[f"stage4_9_{name}"] = descriptor(path)
    if len({value["path"] for value in values.values()}) != len(values):
        raise ValueError("Stage 6 source artifact paths are duplicated")
    return values


def build_correctness_report(
    frozen: dict[str, dict[str, Any]],
    stage1_raw: dict[str, Any],
    stage2_raw: dict[str, Any],
    stage3_raw: dict[str, Any],
    stage4_raw: dict[str, Any],
    stage49_summary: dict[str, Any],
    stage5: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "stage1_complete_frontier": (
            len(stage1_raw["rq3_frontier"]["selection_points"]) >= 177
            and all(
                value["complete"]
                for value in stage1_raw["rq3_frontier"][
                    "selective_grid_audit"
                ]
            )
        ),
        "stage2_deployed_certificates": all(
            value["primary_certificate"]["passed"]
            for value in stage2_raw["pairs"]
        ),
        "stage3_transport": (
            frozen["stage3"]["correctness"][
                "transport_mismatched_elements"
            ]
            == 0
        ),
        "stage4_complete_manifests": all(
            value["correctness"]["allclose"]
            and value["correctness"]["finite"]
            and value["correctness"]["lengths_offsets_valid"]
            and value["correctness"]["record_order_valid"]
            and value["manifest"]["complete"]
            and value["manifest"]["duplicate_free"]
            and value["manifest"]["record_count"]
            == freeze.EXPECTED_RECORDS
            for value in stage4_raw["runs"]
        ),
        "stage4_5_gate": frozen["stage4_5"]["gate"][
            "stage5_admitted"
        ],
        "stage4_6_chain": frozen["stage4_6"]["gate"][
            "complete_chain_cost_and_fidelity_passed"
        ],
        "stage4_9_chain": stage49_summary["checks"]["all_passed"],
        "stage5_closure": stage5["checks"]["all_passed"],
    }
    if not all(checks.values()):
        raise ValueError(f"Stage 6 correctness report failed: {checks}")
    return {
        "protocol": "cohortkv_single_config_stage6_correctness_v1",
        "status": "complete",
        "checks": {**checks, "all_passed": True},
    }


def build_timing_memory_report(
    stage4_raw: dict[str, Any],
    accounting: dict[str, Any],
    lifecycle: dict[str, Any],
    stage5: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": "cohortkv_single_config_stage6_timing_memory_v1",
        "status": "complete",
        "stage4_normal_path_runs": stage4_raw["runs"],
        "direct_oldkv_hot_hbm": accounting["active_direct_oldkv"],
        "offline_setup": accounting["offline_setup"],
        "rejected_fp16_normalized_capsule": accounting[
            "rejected_fp16_normalized_capsule"
        ],
        "lifecycle": lifecycle,
        "stage5_copy_on_write_capacity": stage5["stage5_closure"][
            "copy_on_write_capacity"
        ],
        "claim_boundary": {
            "stage4_9_primary_is_device_resident_u_over_e": True,
            "stage4_9_host_state_movement_excluded_and_reported": True,
            "stage4_9_end_to_end_state_movement_claim": False,
            "stage5_is_theta0_theta1_correctness_closure": True,
        },
    }


def build_paper_tables(
    stage1_raw: dict[str, Any],
    stage2_raw: dict[str, Any],
    stage3_raw: dict[str, Any],
    stage4_raw: dict[str, Any],
    accounting: dict[str, Any],
    lifecycle: dict[str, Any],
    stage5: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": "cohortkv_single_config_stage6_paper_tables_v1",
        "status": "complete",
        "rq2_compiler": stage2_raw["rq2_compiler"],
        "rq3_frontier": {
            "selection_points": stage1_raw["rq3_frontier"][
                "selection_points"
            ],
            "certified_selective_actions": stage1_raw[
                "rq3_frontier"
            ]["certified_selective_actions"],
        },
        "stage3_operator": {
            "selection": stage3_raw["selection"],
            "correctness": stage3_raw["correctness_by_layout"],
        },
        "stage4_normalized_source": stage4_raw["runs"],
        "stage4_5_direct_oldkv": accounting[
            "active_direct_oldkv"
        ]["normal_path_points"],
        "lifecycle": lifecycle,
        "stage5": {
            "normal_job": stage5["stage5_closure"]["normal_job"],
            "semantic_fallback_job": stage5["stage5_closure"][
                "semantic_fallback_job"
            ],
            "abort_jobs": stage5["stage5_closure"]["abort_jobs"],
        },
    }


def build_paper_figures(
    stage1_raw: dict[str, Any],
    stage4_raw: dict[str, Any],
    accounting: dict[str, Any],
    lifecycle: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": "cohortkv_single_config_stage6_paper_figures_v1",
        "status": "complete",
        "figure5_threshold_source": (
            "results/system/cohortkv_single_config_full_chain_v1/"
            "stage2_compiler_seed0.json"
        ),
        "figure6_frontier_points": stage1_raw["rq3_frontier"][
            "selection_points"
        ],
        "figure7_normalized_source_runs": stage4_raw["runs"],
        "figure8_direct_hot_hbm_points": accounting[
            "active_direct_oldkv"
        ]["normal_path_points"],
        "lifecycle_candidates": lifecycle[
            "corrected_growing_history"
        ]["candidates"],
        "rendering_scope": "data_only_reproducible_inputs",
    }


def build_artifact_to_claim(
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    mapping = [
        ("rq2_deployed_compiler", ["stage2"]),
        ("rq3_selective_frontier", ["stage1"]),
        ("stage3_operator_correctness", ["stage3"]),
        ("stage4_normalized_source_negative", ["stage4"]),
        (
            "stage4_5_direct_oldkv_hot_hbm",
            ["stage4_5", "accounting"],
        ),
        (
            "stage4_6_fixed_history_lifecycle",
            ["stage4_6_policy", "stage4_6"],
        ),
        (
            "stage4_9_corrected_growing_history",
            [
                "stage4_9_summary",
                f"stage4_9_{COST_ENDPOINT}",
                f"stage4_9_{SELECTED_CANDIDATE}",
            ],
        ),
        (
            "stage5_transactional_closure",
            ["stage5"],
        ),
        ("source_state_accounting", ["accounting"]),
    ]
    claims = [
        {
            "claim_id": claim_id,
            "evidence": [sources[name] for name in evidence],
        }
        for claim_id, evidence in mapping
    ]
    if {value["claim_id"] for value in claims} != EXPECTED_CLAIMS:
        raise ValueError("Stage 6 claim coverage differs")
    return {
        "protocol": "cohortkv_single_config_stage6_artifact_claim_v1",
        "status": "complete",
        "claims": claims,
        "all_claims_bound": True,
    }


def normalized_environment(
    stage4_raw: dict[str, Any],
    code_snapshot: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    environment = stage4_raw["environment"]
    page_cache = schema["properties"]["environment"]["properties"][
        "page_cache_condition"
    ]["const"]
    return {
        "gpus": [
            {
                "index": value["index"],
                "name": value["name"],
                "total_memory_bytes": value["total_bytes"],
            }
            for value in environment["gpus"]
        ],
        "software": {
            "python": environment["python"],
            "torch": environment["torch"],
            "cuda_runtime": environment["cuda_runtime"],
            "repository_commit": code_snapshot["repository_commit"],
            "code_snapshot_sha256": code_snapshot[
                "content_sha256"
            ],
            "assembler_python": platform.python_version(),
        },
        "source_storage": environment["source_storage"],
        "page_cache_condition": page_cache,
    }


def build_stage6_closure(
    sources: dict[str, dict[str, Any]],
    sidecar_payloads: dict[str, bytes],
) -> dict[str, Any]:
    return {
        "protocol": STAGE6_PROTOCOL,
        "status": "single_configuration_v1_frozen",
        "selected_candidate": SELECTED_CANDIDATE,
        "old_gpu_matrix_rerun": False,
        "source_artifacts": [
            sources[name] for name in sorted(sources)
        ],
        "outputs": {
            name: output_descriptor(OUTPUTS[name], sidecar_payloads[name])
            for name in OUTPUTS
        },
        "checks": {
            "all_source_hashes": True,
            "whole_aggregate_semantics": True,
            "jsonschema": True,
            "stage5_semantics": True,
            "candidate_binding": True,
            "all_tbd_markers_disposed": True,
            "all_claims_bound": True,
            "all_passed": True,
        },
    }


def validate_stage5_binding(
    stage5: dict[str, Any],
    stage49_summary_descriptor: dict[str, Any],
    selected_descriptor: dict[str, Any],
) -> None:
    gate = stage5.get("formal_stage4_9_gate", {})
    if (
        stage5.get("protocol")
        != "cohortkv_single_config_stage5_full_cow_integration_v1"
        or stage5.get("status") != "complete"
        or stage5.get("scientific_result") is not True
        or stage5.get("configuration", {}).get("candidate_name")
        != SELECTED_CANDIDATE
        or gate.get("summary_sha256")
        != stage49_summary_descriptor["sha256"]
        or gate.get("candidate_result_sha256")
        != selected_descriptor["sha256"]
        or stage5.get("checks", {}).get("all_passed") is not True
    ):
        raise ValueError("Stage 5 formal candidate binding differs")
    freeze.validate_stage5_closure_semantics(
        stage5["stage5_closure"]
    )


def validate_aggregate_semantics(
    final: dict[str, Any],
    sidecars: dict[str, dict[str, Any]],
    sidecar_payloads: dict[str, bytes],
) -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    closure = final["stage6_closure"]
    source_artifacts = closure["source_artifacts"]
    require(
        len({value["path"] for value in source_artifacts})
        == len(source_artifacts),
        "Stage 6 source artifact paths are not unique",
    )
    for value in source_artifacts:
        path = repo_path(Path(value["path"]))
        require(path.is_file(), "Stage 6 source artifact is missing")
        require(
            path.stat().st_size == value["bytes"]
            and sha256_file(path) == value["sha256"],
            "Stage 6 source artifact hash differs",
        )
    for name, descriptor_value in closure["outputs"].items():
        require(
            descriptor_value
            == output_descriptor(OUTPUTS[name], sidecar_payloads[name]),
            "Stage 6 report descriptor differs",
        )
    rq4_runs = final["rq4_system"]["runs"]
    expected_keys = {
        (method, destination, gpu_count)
        for method in (
            "compiled",
            "selective_contiguous",
            "exact",
            "residual_p",
            "no_transform",
        )
        for destination in freeze.DESTINATIONS
        for gpu_count in freeze.GPU_COUNTS
    }
    require(
        len(rq4_runs) == 30
        and {
            (
                value["method"],
                value["destination"],
                value["gpu_count"],
            )
            for value in rq4_runs
        }
        == expected_keys,
        "Stage 6 RQ4 coverage differs",
    )
    lifecycle = final["lifecycle"]["corrected_growing_history"]
    require(
        lifecycle["selected_candidate"] == SELECTED_CANDIDATE
        and lifecycle["cost_endpoint"] == COST_ENDPOINT
        and lifecycle["full_cohort_hbm_claim"] is False
        and lifecycle["end_to_end_state_movement_claim"] is False,
        "Stage 6 lifecycle boundary differs",
    )
    require(
        {
            value["candidate_name"]
            for value in lifecycle["candidates"]
        }
        == {SELECTED_CANDIDATE, COST_ENDPOINT},
        "Stage 6 lifecycle candidates differ",
    )
    require(
        {
            value["claim_id"]
            for value in sidecars["artifact_to_claim"]["claims"]
        }
        == EXPECTED_CLAIMS,
        "Stage 6 claim coverage differs",
    )
    require(
        {
            value["slot"]
            for value in final["negative_results"]
        }
        == EXPECTED_NEGATIVE_SLOTS,
        "Stage 6 negative-result coverage differs",
    )
    require(
        sidecars["tbd_disposition"]["all_markers_disposed"] is True
        and sidecars["tbd_disposition"]["markers"] == 0,
        "Stage 6 TBD disposition differs",
    )
    require(
        final["blueprint_sha256"] == sha256_file(repo_path(BLUEPRINT)),
        "Stage 6 blueprint hash differs",
    )
    require(
        final["workload_content_sha256"]
        == load_json(WORKLOAD)["content_sha256"],
        "Stage 6 workload hash differs",
    )
    freeze.validate_stage5_closure_semantics(final["stage5_closure"])


def build_outputs(
    stage5_path: Path = STAGE5_RAW,
) -> dict[Path, bytes]:
    schema = load_json(SCHEMA)
    jsonschema.Draft202012Validator.check_schema(schema)
    frozen = {
        name: load_json(path) for name, path in FROZEN_CONFIGS.items()
    }
    validate_frozen_inputs(frozen)
    stage1_raw = load_json(STAGE1_RAW)
    stage2_raw = load_json(STAGE2_RAW)
    stage3_raw = load_json(STAGE3_RAW)
    stage4_raw = load_json(STAGE4_RAW)
    stage49_summary = load_json(STAGE49_RAW)
    stage49_values = {
        name: load_json(path)
        for name, path in STAGE49_CANDIDATES.items()
    }
    stage49_descriptors = {
        name: descriptor(path)
        for name, path in STAGE49_CANDIDATES.items()
    }
    lifecycle = build_lifecycle(
        frozen,
        stage49_summary,
        stage49_values,
        stage49_descriptors,
    )
    stage5 = load_json(stage5_path)
    accounting = load_json(ACCOUNTING_RAW)
    sources = source_descriptor_map(frozen, stage5_path)
    validate_stage5_binding(
        stage5,
        sources["stage4_9_summary"],
        sources[f"stage4_9_{SELECTED_CANDIDATE}"],
    )
    code_snapshot = build_code_snapshot()
    tbd_disposition = build_tbd_disposition()
    negative_results = build_negative_results()
    sidecars = {
        "correctness_report": build_correctness_report(
            frozen,
            stage1_raw,
            stage2_raw,
            stage3_raw,
            stage4_raw,
            stage49_summary,
            stage5,
        ),
        "timing_memory_report": build_timing_memory_report(
            stage4_raw,
            accounting,
            lifecycle,
            stage5,
        ),
        "paper_tables": build_paper_tables(
            stage1_raw,
            stage2_raw,
            stage3_raw,
            stage4_raw,
            accounting,
            lifecycle,
            stage5,
        ),
        "paper_figures": build_paper_figures(
            stage1_raw,
            stage4_raw,
            accounting,
            lifecycle,
        ),
        "artifact_to_claim": build_artifact_to_claim(sources),
        "negative_results_log": {
            "protocol": (
                "cohortkv_single_config_stage6_negative_results_v1"
            ),
            "status": "complete",
            "results": negative_results,
        },
        "tbd_disposition": tbd_disposition,
        "code_snapshot_manifest": code_snapshot,
    }
    sidecar_payloads = {
        name: canonical_json_bytes(value)
        for name, value in sidecars.items()
    }
    final = {
        "protocol": freeze.PROTOCOL,
        "status": "development_complete",
        "study_stage": "adaptive_seed0_development",
        "seed": freeze.TRAINING_SEED,
        "blueprint_sha256": sha256_file(repo_path(BLUEPRINT)),
        "workload_content_sha256": load_json(WORKLOAD)[
            "content_sha256"
        ],
        "environment": normalized_environment(
            stage4_raw,
            code_snapshot,
            schema,
        ),
        "rq2_compiler": stage2_raw["rq2_compiler"],
        "rq3_frontier": build_rq3(stage1_raw, frozen["stage1"]),
        "rq4_system": {"runs": stage4_raw["runs"]},
        "lifecycle": lifecycle,
        "stage5_closure": stage5["stage5_closure"],
        "source_state_accounting": accounting,
        "stage6_closure": build_stage6_closure(
            sources,
            sidecar_payloads,
        ),
        "negative_results": negative_results,
    }
    validate_aggregate_semantics(
        final,
        sidecars,
        sidecar_payloads,
    )
    jsonschema.validate(instance=final, schema=schema)
    return {
        **{
            OUTPUTS[name]: payload
            for name, payload in sidecar_payloads.items()
        },
        FINAL_OUTPUT: canonical_json_bytes(final),
    }


def check_outputs(outputs: dict[Path, bytes]) -> None:
    mismatches = []
    for path, expected in outputs.items():
        resolved = repo_path(path)
        if not resolved.is_file():
            mismatches.append(f"missing {path}")
        elif resolved.read_bytes() != expected:
            mismatches.append(f"content differs for {path}")
    if mismatches:
        raise RuntimeError("; ".join(mismatches))


def write_atomic(path: Path, payload: bytes) -> None:
    resolved = repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(
        f".{resolved.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
        descriptor_value = os.open(resolved.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor_value)
        finally:
            os.close(descriptor_value)
    finally:
        temporary.unlink(missing_ok=True)


def write_outputs(outputs: dict[Path, bytes]) -> None:
    for path, payload in outputs.items():
        if path != FINAL_OUTPUT:
            write_atomic(path, payload)
    write_atomic(FINAL_OUTPUT, outputs[FINAL_OUTPUT])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = build_outputs(args.stage5_result)
    if args.check:
        check_outputs(outputs)
        status = "verified"
    else:
        write_outputs(outputs)
        status = "single_configuration_v1_frozen"
    print(
        json.dumps(
            {
                "protocol": STAGE6_PROTOCOL,
                "status": status,
                "selected_candidate": SELECTED_CANDIDATE,
                "old_gpu_matrix_rerun": False,
                "outputs": [str(path) for path in outputs],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
