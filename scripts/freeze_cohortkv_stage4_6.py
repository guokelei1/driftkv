from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from hstu_kvcache.migration import (
    BalancedLifecyclePolicy,
    CacheLifecycleState,
    sha256_file,
)

POLICY_PROTOCOL = "cohortkv_single_config_stage4_6_policy_frozen_v1"
SUMMARY_PROTOCOL = "cohortkv_single_config_stage4_6_frozen_v1"
EXPERIMENT_PROTOCOL = (
    "cohortkv_single_config_stage4_6_lifecycle_development_v1"
)
COMPILER_PROTOCOL = (
    "cohortkv_single_config_stage4_6_adjacent_compiler_v1"
)
SEARCH_PROTOCOL = "cohortkv_single_config_stage4_6_lifecycle_search_v1"
CHAIN_PROTOCOL = "cohortkv_single_config_stage4_6_recursive_chain_v1"
ROOT_RESULT = Path(
    "results/system/cohortkv_single_config_full_chain_v1"
)
COMPILER = ROOT_RESULT / "stage4_6_adjacent_compiler_seed0.json"
FIT = ROOT_RESULT / "stage4_6_fit_trajectory_seed0.json"
FIT_TRANSITIONS = ROOT_RESULT / "stage4_6_fit_transitions_seed0.json"
SELECTION_TRANSITIONS = (
    ROOT_RESULT / "stage4_6_selection_transitions_seed0.json"
)
SEARCH = ROOT_RESULT / "stage4_6_policy_search_seed0.json"
CERTIFICATE = ROOT_RESULT / "stage4_6_certificate_chain_seed0.json"
FULL = ROOT_RESULT / "stage4_6_full_chain_seed0.json"
THRESHOLD_CERTIFICATE = (
    ROOT_RESULT / "stage4_6_threshold_certificate_diagnostic_seed0.json"
)
THRESHOLD_FULL = (
    ROOT_RESULT / "stage4_6_threshold_full_chain_diagnostic_seed0.json"
)
POLICY_OUTPUT = Path(
    "configs/cohortkv_single_config_v1/stage4_6_lifecycle_policy.json"
)
SUMMARY_OUTPUT = Path(
    "configs/cohortkv_single_config_v1/stage4_6_lifecycle_summary.json"
)
IMPLEMENTATION_FILES = (
    Path("src/hstu_kvcache/migration/lifecycle.py"),
    Path("src/hstu_kvcache/migration/stage46_chain.py"),
    Path("scripts/compile_cohortkv_stage4_6_edges.py"),
    Path("scripts/evaluate_cohortkv_stage4_6_lifecycle.py"),
    Path("scripts/run_cohortkv_stage4_6_full_chain.py"),
    Path("tests/test_lifecycle.py"),
    Path("tests/test_stage46_chain.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(root: Path, path: Path) -> dict:
    return json.loads((root / path).read_text())


def descriptor(root: Path, path: Path, protocol: str | None = None) -> dict:
    resolved = root / path
    value = {
        "path": str(path),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if protocol is not None:
        value["protocol"] = protocol
    return value


def validate_common(
    value: dict,
    protocol: str,
    status: str = "complete",
) -> None:
    require(value.get("protocol") == protocol, "artifact protocol differs")
    require(
        value.get("experiment_protocol") == EXPERIMENT_PROTOCOL,
        "experiment protocol differs",
    )
    require(value.get("status") == status, "artifact status differs")


def validate_compiler(root: Path, value: dict) -> list[dict]:
    validate_common(value, COMPILER_PROTOCOL)
    pairs = value.get("pairs", [])
    require(len(pairs) == 11, "adjacent program coverage differs")
    compact = []
    for source_version, pair in enumerate(pairs):
        target_version = source_version + 1
        require(
            pair.get("source_version") == f"theta{source_version}"
            and pair.get("target_version") == f"theta{target_version}",
            "adjacent program edge differs",
        )
        program = pair.get("direct_program", {})
        path = Path(str(program.get("path")))
        require(not path.is_absolute(), "adjacent program path is absolute")
        require((root / path).is_file(), "adjacent program is missing")
        require(
            program.get("sha256") == sha256_file(root / path)
            and program.get("bytes") == (root / path).stat().st_size,
            "adjacent program descriptor differs",
        )
        require(
            pair.get("fit", {}).get("labels_used") is False
            and pair.get("load_validation", {}).get("passed") is True
            and pair["load_validation"]["provenance"].get("labels_used")
            is False,
            "adjacent program fit or reload differs",
        )
        compact.append(
            {
                "source_version": source_version,
                "target_version": target_version,
                "path": str(path),
                "bytes": program["bytes"],
                "sha256": program["sha256"],
                "condition_number_min": program["compile_metrics"][
                    "condition_number_min"
                ],
                "condition_number_max": program["compile_metrics"][
                    "condition_number_max"
                ],
            }
        )
    return compact


def validate_transition_artifact(
    value: dict,
    role: str,
    records: int,
    transitions: int,
) -> None:
    validate_common(value, SEARCH_PROTOCOL)
    require(value.get("labels_used") is False, "transition used labels")
    require(
        value.get("role") == role
        and value.get("records") == records
        and value.get("edges") == 11
        and value.get("maximum_candidate_depth") == 11
        and len(value.get("transitions", [])) == transitions
        and len(value.get("exact_costs", [])) == records * 11,
        "transition coverage differs",
    )


def derive_edge_schedule(
    fit_transitions: dict,
    policy: BalancedLifecyclePolicy,
) -> list[dict]:
    severities = []
    for source_version in range(11):
        values = [
            float(value["cache_error"]["q090"])
            for value in fit_transitions["transitions"]
            if int(value["source_version"]) == source_version
            and int(value["migration_depth_after"]) == 1
        ]
        require(len(values) == 40, "one-hop edge calibration differs")
        severities.append(float(statistics.median(values)))
    for actual, expected in zip(
        policy.edge_severities,
        severities,
        strict=True,
    ):
        require(
            math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            "edge severity differs from fit records",
        )
    order = {
        index: rank
        for rank, index in enumerate(
            sorted(range(11), key=lambda index: severities[index])
        )
    }
    expected_fractions = [
        min(
            0.25,
            max(
                0.15,
                0.20 + 0.05 * (2 * order[index] / 10 - 1),
            ),
        )
        for index in range(11)
    ]
    for actual, expected in zip(
        policy.exact_fractions,
        expected_fractions,
        strict=True,
    ):
        require(
            math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            "edge exact fraction differs from severity rank",
        )
    return [
        {
            "source_version": source_version,
            "target_version": source_version + 1,
            "fit_median_one_hop_cache_error_q090": severities[
                source_version
            ],
            "configured_exact_fraction": policy.exact_fractions[
                source_version
            ],
        }
        for source_version in range(11)
    ]


def validate_search(
    value: dict,
    fit_transitions: dict,
) -> tuple[BalancedLifecyclePolicy, dict, dict]:
    validate_common(value, SEARCH_PROTOCOL)
    require(
        value.get("phase") == "policy-search"
        and value.get("labels_used") is False
        and value.get("roles")
        == {"fit": "fit", "selection": "program_selection"},
        "policy-search role boundary differs",
    )
    recommended = value.get("recommended", {})
    require(
        recommended.get("selector") == "balanced_age_severity_quota"
        and recommended.get("configuration", {}).get("name")
        == "severity_bounded_0.05",
        "recommended lifecycle selector differs",
    )
    policy = BalancedLifecyclePolicy.from_dict(recommended["policy"])
    require(
        policy.max_migration_depth == 4
        and policy.scheduler_seed == 0
        and len(policy.exact_fractions) == 11,
        "balanced lifecycle policy differs",
    )
    result = recommended.get("result", {})
    balance = result.get("balance", {})
    require(
        result.get("cost_ratio_to_all_exact", 1) <= 0.25
        and result.get("worst_view_fidelity", 0) >= 0.95
        and balance.get("minimum_step_exact_fraction", 0) >= 0.15
        and balance.get("maximum_step_exact_fraction", 1) <= 0.25
        and balance.get("step_exact_fraction_range", 1) <= 0.10
        and result.get("discarded_candidate_records") == 0,
        "balanced selection gate failed",
    )
    eligible = [
        candidate
        for candidate in value.get("balanced_candidates", [])
        if candidate["result"]["cost_ratio_to_all_exact"] <= 0.25
        and candidate["result"]["worst_view_fidelity"] >= 0.95
        and candidate["result"]["balance"][
            "maximum_step_exact_fraction"
        ]
        <= 0.25
        and candidate["result"]["balance"][
            "step_exact_fraction_range"
        ]
        <= 0.10
    ]
    selected = max(
        eligible,
        key=lambda candidate: (
            candidate["result"]["worst_view_fidelity"],
            -candidate["result"]["cost_ratio_to_all_exact"],
            -candidate["result"]["balance"][
                "step_exact_fraction_range"
            ],
            candidate["configuration"]["name"],
        ),
    )
    require(selected == recommended, "balanced selection cannot be rebuilt")
    diagnostic = value.get("selected_threshold_diagnostic", {})
    diagnostic_counts = [
        int(step["exact_records"])
        for step in diagnostic.get("result", {}).get("steps", [])
    ]
    require(
        len(diagnostic_counts) == 11
        and min(diagnostic_counts) == 0
        and max(diagnostic_counts) >= 36
        and "not frozen" in value.get("risk_selector_status", ""),
        "threshold-wave diagnostic differs",
    )
    schedule = derive_edge_schedule(fit_transitions, policy)
    return policy, recommended, {
        "selector": diagnostic["selector"],
        "selection_exact_records_by_step": diagnostic_counts,
        "selection_minimum_exact_fraction": min(diagnostic_counts) / 60,
        "selection_maximum_exact_fraction": max(diagnostic_counts) / 60,
        "selection_cost_ratio_to_all_exact": diagnostic["result"][
            "cost_ratio_to_all_exact"
        ],
        "selection_worst_view_fidelity": diagnostic["result"][
            "worst_view_fidelity"
        ],
        "status": value["risk_selector_status"],
        "replacement_reason": value["fallback_reason"],
        "edge_schedule": schedule,
    }


def state_from_dict(value: dict) -> CacheLifecycleState:
    return CacheLifecycleState(
        record_id=int(value["record_id"]),
        served_version=int(value["served_version"]),
        last_exact_version=int(value["last_exact_version"]),
        migration_depth=int(value["migration_depth"]),
        risk_score=float(value["risk_score"]),
        state_kind=str(value["state_kind"]),
    )


def validate_lineage(
    value: dict,
    policy: BalancedLifecyclePolicy,
    program_hashes: dict[int, str],
) -> dict:
    previous = {}
    record_ids = None
    exact_counts = []
    reason_counts = {
        "balanced_exact_quota": 0,
        "max_migration_depth": 0,
        "balanced_migrate": 0,
    }
    for target_version, step in enumerate(value["steps"], start=1):
        lineage = step.get("lineage", [])
        require(
            len(lineage) == value["records"],
            "lineage coverage differs",
        )
        by_record = {
            int(item["record_id"]): item for item in lineage
        }
        require(
            len(by_record) == value["records"],
            "lineage record IDs repeat",
        )
        current_ids = tuple(sorted(by_record))
        if record_ids is None:
            record_ids = current_ids
        require(current_ids == record_ids, "lineage record set changed")
        before_states = tuple(
            state_from_dict(by_record[record_id]["state_before"])
            for record_id in current_ids
        )
        expected = {
            decision.record_id: decision
            for decision in policy.plan(before_states, target_version)
        }
        observed_exact = 0
        for record_id in current_ids:
            item = by_record[record_id]
            before = item["state_before"]
            after = item["state_after"]
            decision = item["decision"]
            require(
                int(before["record_id"]) == record_id
                and int(after["record_id"]) == record_id
                and int(decision["record_id"]) == record_id,
                "lineage record identity differs",
            )
            if target_version > 1:
                require(
                    before == previous[record_id],
                    "lineage does not consume the previous output",
                )
            planned = expected[record_id]
            require(
                decision["action"] == planned.action
                and decision["reason"] == planned.reason
                and decision["candidate_evaluated"] is False
                and int(decision["source_version"]) == target_version - 1
                and int(decision["target_version"]) == target_version
                and decision["program_sha256"]
                == program_hashes[target_version - 1],
                "lineage decision differs from frozen policy",
            )
            reason_counts[decision["reason"]] += 1
            require(
                int(after["served_version"]) == target_version,
                "lineage target version differs",
            )
            if decision["action"] == "exact":
                observed_exact += 1
                require(
                    after["state_kind"] == "exact"
                    and int(after["last_exact_version"]) == target_version
                    and int(after["migration_depth"]) == 0
                    and float(after["risk_score"]) == 0,
                    "exact refresh did not reset lifecycle state",
                )
            else:
                require(
                    after["state_kind"] == "migrated"
                    and int(after["last_exact_version"])
                    == int(before["last_exact_version"])
                    and int(after["migration_depth"])
                    == int(before["migration_depth"]) + 1
                    and int(after["migration_depth"])
                    <= policy.max_migration_depth,
                    "migration lifecycle advance differs",
                )
            previous[record_id] = after
        require(
            observed_exact == step["actions"]["exact"]
            and step["actions"]["discarded_migration_candidates"] == 0,
            "lineage action summary differs",
        )
        exact_counts.append(observed_exact)
    require(record_ids is not None, "lineage is empty")
    terminal = value["terminal_state"]
    terminal_kinds = {
        kind: sum(state["state_kind"] == kind for state in previous.values())
        for kind in ("exact", "migrated")
    }
    require(
        terminal.get("served_versions") == [11]
        and terminal.get("state_kind_counts") == terminal_kinds
        and terminal.get("maximum_migration_depth")
        == max(int(state["migration_depth"]) for state in previous.values()),
        "terminal lifecycle summary differs",
    )
    return {
        "records": len(record_ids),
        "updates": 11,
        "lineage_rows": len(record_ids) * 11,
        "exact_records_by_step": exact_counts,
        "reason_counts": reason_counts,
        "all_steps_consume_previous_actual_output": True,
        "all_exact_actions_reset_state": True,
        "all_migration_depths_bounded": True,
        "all_decisions_rebuilt_from_frozen_policy": True,
        "all_program_hashes_match_adjacent_edges": True,
    }


def task_summary(steps: list[dict]) -> dict:
    metrics = ("mean_rank", "catalog_auc", "ndcg@100", "hit@100")
    paired = {
        metric: [
            float(
                step["task_metrics"][
                    "paired_difference_mixed_minus_exact"
                ][metric]
            )
            for step in steps
        ]
        for metric in metrics
    }
    require(
        all(step.get("task_metrics", {}).get("records") == 522 for step in steps)
        and all(math.isfinite(value) for values in paired.values() for value in values),
        "final-test task metrics differ",
    )
    return {
        "records": 522,
        "labels_used_for_routing": False,
        "paired_difference_mixed_minus_exact_by_step": paired,
        "mean_paired_difference_mixed_minus_exact": {
            metric: statistics.fmean(values)
            for metric, values in paired.items()
        },
        "maximum_absolute_paired_difference": {
            metric: max(abs(value) for value in values)
            for metric, values in paired.items()
        },
        "recovery_disclosure": (
            "reuse-to-exact recovery is diagnostic and may be unstable "
            "when its denominator is near zero"
        ),
    }


def validate_chain(
    value: dict,
    role: str,
    records: int,
    policy: BalancedLifecyclePolicy,
    program_hashes: dict[int, str],
) -> dict:
    validate_common(value, CHAIN_PROTOCOL)
    require(
        value.get("role") == role
        and value.get("records") == records
        and value.get("edges") == 11
        and value.get("labels_used_for_routing") is False
        and value.get("policy") == policy.to_dict()
        and len(value.get("steps", [])) == 11,
        "recursive chain identity differs",
    )
    lineage = validate_lineage(value, policy, program_hashes)
    exact_fractions = [
        step["actions"]["exact"] / records for step in value["steps"]
    ]
    rounding = 1 / records
    worst = {
        "cache_fidelity_q090": min(
            step["label_free_metrics"]["cache_fidelity_q090"]
            for step in value["steps"]
        ),
        "score_cosine": min(
            step["label_free_metrics"]["score_cosine"]
            for step in value["steps"]
        ),
        "top100_overlap": min(
            step["label_free_metrics"]["top100_overlap"]
            for step in value["steps"]
        ),
    }
    maximum_step_cost = max(
        step["gpu_cost_ms"]["ratio_to_all_exact"]
        for step in value["steps"]
    )
    require(
        min(exact_fractions) >= 0.15 - rounding
        and max(exact_fractions) <= 0.25 + rounding
        and max(exact_fractions) - min(exact_fractions)
        <= 0.10 + 2 * rounding,
        "recursive chain refresh balance failed",
    )
    require(
        value["cumulative_gpu_cost"]["ratio_to_all_exact"] <= 0.30
        and maximum_step_cost <= 0.35
        and worst["cache_fidelity_q090"] >= 0.90
        and worst["score_cosine"] >= 0.995
        and worst["top100_overlap"] >= 0.95,
        "recursive chain cost or fidelity gate failed",
    )
    scheduler_values = [
        float(step["scheduler_cpu_ms"]) for step in value["steps"]
    ]
    require(
        math.isclose(
            sum(scheduler_values),
            float(value["scheduler_cpu_ms"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "scheduler timing summary differs",
    )
    output = {
        "role": role,
        "records": records,
        "updates": 11,
        "cumulative_gpu_cost": value["cumulative_gpu_cost"],
        "maximum_step_gpu_cost_ratio": maximum_step_cost,
        "minimum_step_exact_fraction": min(exact_fractions),
        "maximum_step_exact_fraction": max(exact_fractions),
        "step_exact_fraction_range": (
            max(exact_fractions) - min(exact_fractions)
        ),
        "nearest_record_rounding_tolerance": rounding,
        "worst_step_label_free_metrics": worst,
        "scheduler_cpu_ms": value["scheduler_cpu_ms"],
        "lineage": lineage,
        "measurement_boundary": (
            "one A40; existing old K/V and raw history treated as "
            "hot-HBM sources; GPU migration, exact subset gather/replay, "
            "publication measured; offline all-exact evaluation excluded "
            "from mixed-policy cost"
        ),
    }
    if role == "certificate":
        require(
            value.get("certificate", {}).get("passed") is True
            and all(
                value["certificate"]["checks"].values()
            ),
            "independent certificate failed",
        )
        output["certificate"] = value["certificate"]
    else:
        require(value.get("certificate") is None, "full chain certificate differs")
        output["task_metrics"] = task_summary(value["steps"])
    return output


def validate_threshold_diagnostics(
    root: Path,
    certificate: dict,
    full: dict,
    search: dict,
) -> dict:
    validate_common(certificate, CHAIN_PROTOCOL)
    validate_common(full, CHAIN_PROTOCOL)
    threshold_policy = search["selected_threshold_diagnostic"]["policy"]
    require(
        certificate.get("role") == "certificate"
        and full.get("role") == "all"
        and certificate.get("policy") == threshold_policy
        and full.get("policy") == threshold_policy,
        "threshold diagnostic policy differs",
    )
    counts = [int(step["actions"]["exact"]) for step in full["steps"]]
    require(
        min(counts) <= 3
        and max(counts) >= 430,
        "threshold full-chain refresh waves differ",
    )
    return {
        "certificate_artifact": descriptor(
            root,
            THRESHOLD_CERTIFICATE,
            CHAIN_PROTOCOL,
        ),
        "full_chain_artifact": descriptor(
            root,
            THRESHOLD_FULL,
            CHAIN_PROTOCOL,
        ),
        "full_chain_exact_records_by_step": counts,
        "full_chain_minimum_exact_fraction": min(counts) / 682,
        "full_chain_maximum_exact_fraction": max(counts) / 682,
        "full_chain_step_exact_fraction_range": (
            max(counts) - min(counts)
        )
        / 682,
        "full_chain_cumulative_gpu_cost_ratio": full[
            "cumulative_gpu_cost"
        ]["ratio_to_all_exact"],
        "status": (
            "rejected as the frozen policy because it omits a per-step "
            "peak-refresh objective and produces synchronized waves"
        ),
    }


def implementation_snapshot(root: Path) -> list[dict]:
    return [descriptor(root, path) for path in IMPLEMENTATION_FILES]


def build(root: Path) -> tuple[dict, dict]:
    compiler = load(root, COMPILER)
    fit = load(root, FIT)
    fit_transitions = load(root, FIT_TRANSITIONS)
    selection_transitions = load(root, SELECTION_TRANSITIONS)
    search = load(root, SEARCH)
    certificate = load(root, CERTIFICATE)
    full = load(root, FULL)
    threshold_certificate = load(root, THRESHOLD_CERTIFICATE)
    threshold_full = load(root, THRESHOLD_FULL)
    programs = validate_compiler(root, compiler)
    validate_common(fit, SEARCH_PROTOCOL)
    require(
        fit.get("phase") == "fit"
        and fit.get("role") == "fit"
        and fit.get("records") == 40
        and fit.get("labels_used") is False,
        "fit trajectory differs",
    )
    validate_transition_artifact(
        fit_transitions,
        "fit",
        40,
        1800,
    )
    validate_transition_artifact(
        selection_transitions,
        "program_selection",
        60,
        2700,
    )
    policy, recommended, search_diagnostic = validate_search(
        search,
        fit_transitions,
    )
    program_hashes = {
        program["source_version"]: program["sha256"]
        for program in programs
    }
    certificate_summary = validate_chain(
        certificate,
        "certificate",
        60,
        policy,
        program_hashes,
    )
    full_summary = validate_chain(
        full,
        "all",
        682,
        policy,
        program_hashes,
    )
    threshold_diagnostics = validate_threshold_diagnostics(
        root,
        threshold_certificate,
        threshold_full,
        search,
    )
    policy_payload = {
        "protocol": POLICY_PROTOCOL,
        "status": "stage4_6_lifecycle_policy_frozen",
        "study_stage": "single_configuration_seed0_development",
        "frozen_date": "2026-07-27",
        "selector": recommended["selector"],
        "configuration": recommended["configuration"],
        "policy": policy.to_dict(),
        "edge_schedule": search_diagnostic.pop("edge_schedule"),
        "decision_rule": {
            "mandatory_exact": (
                "migration_depth >= 4 before the adjacent update"
            ),
            "per_edge_exact_budget": (
                "15%-25%, ranked by label-free fit one-hop edge severity"
            ),
            "within_edge_priority": (
                "greater migration depth first, then stable SHA256 tie-break"
            ),
            "normal_actions": ["migrate", "exact"],
            "reuse_action": False,
            "speculative_candidate_for_routing": False,
        },
        "selection_evidence": {
            "records": 60,
            "cumulative_gpu_cost_ratio": recommended["result"][
                "cost_ratio_to_all_exact"
            ],
            "worst_view_fidelity": recommended["result"][
                "worst_view_fidelity"
            ],
            "exact_records_by_step": recommended["result"]["balance"][
                "exact_records_by_step"
            ],
            "minimum_step_exact_fraction": recommended["result"]["balance"][
                "minimum_step_exact_fraction"
            ],
            "maximum_step_exact_fraction": recommended["result"]["balance"][
                "maximum_step_exact_fraction"
            ],
            "step_exact_fraction_range": recommended["result"]["balance"][
                "step_exact_fraction_range"
            ],
            "labels_used": False,
        },
        "independent_certificate": certificate_summary,
        "policy_status": (
            "bounded deterministic development heuristic; empirically "
            "selected on the frozen program-selection role; no global "
            "optimality claim"
        ),
        "amendment": {
            "original_candidate": "per-cache norm-sketch risk threshold",
            "reason": (
                "the original cumulative-only objective produced severe "
                "per-step exact-refresh waves on a diagnostic full chain"
            ),
            "quality_labels_used_to_amend_or_select": False,
            "final_full_chain_used_to_select_numeric_policy": False,
            "replacement": (
                "balanced age/deadline scheduling with program-level "
                "label-free edge severity"
            ),
        },
        "inputs": {
            "compiler": descriptor(root, COMPILER, COMPILER_PROTOCOL),
            "fit_trajectory": descriptor(root, FIT, SEARCH_PROTOCOL),
            "fit_transitions": descriptor(
                root,
                FIT_TRANSITIONS,
                SEARCH_PROTOCOL,
            ),
            "selection_transitions": descriptor(
                root,
                SELECTION_TRANSITIONS,
                SEARCH_PROTOCOL,
            ),
            "policy_search": descriptor(root, SEARCH, SEARCH_PROTOCOL),
            "certificate": descriptor(
                root,
                CERTIFICATE,
                CHAIN_PROTOCOL,
            ),
        },
    }
    policy_bytes = canonical_json_bytes(policy_payload)
    policy_descriptor = {
        "path": str(POLICY_OUTPUT),
        "bytes": len(policy_bytes),
        "sha256": sha256_bytes(policy_bytes),
        "protocol": POLICY_PROTOCOL,
    }
    summary_payload = {
        "protocol": SUMMARY_PROTOCOL,
        "status": "stage4_6_lifecycle_frozen",
        "study_stage": "single_configuration_seed0_development",
        "frozen_date": "2026-07-27",
        "policy_artifact": policy_descriptor,
        "evidence_artifacts": {
            "compiler": descriptor(root, COMPILER, COMPILER_PROTOCOL),
            "fit_trajectory": descriptor(root, FIT, SEARCH_PROTOCOL),
            "fit_transitions": descriptor(
                root,
                FIT_TRANSITIONS,
                SEARCH_PROTOCOL,
            ),
            "selection_transitions": descriptor(
                root,
                SELECTION_TRANSITIONS,
                SEARCH_PROTOCOL,
            ),
            "policy_search": descriptor(root, SEARCH, SEARCH_PROTOCOL),
            "certificate_chain": descriptor(
                root,
                CERTIFICATE,
                CHAIN_PROTOCOL,
            ),
            "full_chain": descriptor(root, FULL, CHAIN_PROTOCOL),
        },
        "implementation": {
            "files": implementation_snapshot(root),
        },
        "selector_search": {
            "recommended": recommended["selector"],
            "configuration": recommended["configuration"],
            "balanced_selection_result": policy_payload[
                "selection_evidence"
            ],
            "threshold_diagnostic": search_diagnostic,
        },
        "rejected_threshold_full_chain": threshold_diagnostics,
        "independent_certificate": certificate_summary,
        "complete_recursive_chain": full_summary,
        "gate": {
            "all_11_updates_complete": True,
            "actual_previous_outputs_consumed": True,
            "lineage_rebuilt": True,
            "maximum_migration_depth": 4,
            "per_step_refresh_balanced_with_record_rounding": True,
            "certificate_passed": True,
            "complete_chain_cost_and_fidelity_passed": True,
            "recommendation_labels_used_for_routing": False,
            "stage5_admitted": True,
        },
        "claim_boundary": {
            "configuration": (
                "KuaiRand 4+12, seed 0, 16L/H512, history 2048, one A40"
            ),
            "trajectory": "fixed histories, theta0 exact through 11 updates",
            "source_and_target": "hot HBM",
            "evidence_level": "single-configuration development",
            "excluded_claims": [
                "global or analytical selector optimality",
                "organic request arrivals or learned user hotness",
                "new-seed or cross-dataset replication",
                "DRAM, filesystem, SSD, or remote lifecycle performance",
                "failure-safe automatic fallback and transactional rework",
            ],
        },
        "next_stage": {
            "stage": 5,
            "objective": (
                "connect the frozen lifecycle policy to guard, automatic "
                "fallback, transactional rework, and failure visibility"
            ),
            "lifecycle_policy_may_change": False,
        },
    }
    return policy_payload, summary_payload


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    policy, summary = build(root)
    payloads = (
        (POLICY_OUTPUT, canonical_json_bytes(policy)),
        (SUMMARY_OUTPUT, canonical_json_bytes(summary)),
    )
    if args.check:
        for path, payload in payloads:
            resolved = root / path
            if not resolved.is_file() or resolved.read_bytes() != payload:
                raise RuntimeError(f"{path} differs from Stage 4.6 evidence")
        status = "verified"
    else:
        for path, payload in payloads:
            resolved = root / path
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_bytes(payload)
        status = "frozen"
    print(
        json.dumps(
            {
                "protocol": SUMMARY_PROTOCOL,
                "status": status,
                "outputs": [
                    {
                        "path": str(path),
                        "sha256": sha256_bytes(payload),
                    }
                    for path, payload in payloads
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
