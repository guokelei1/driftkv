from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from hstu_kvcache.migration import (
    FidelityContract,
    MigrationActionSpec,
    compile_verified_plan,
    load_executable_plan,
)

PROTOCOL = "cohortkv_single_config_stage2_frozen_v1"
SOURCE_PROTOCOL = "cohortkv_single_config_stage2_compiler_v1"
PARENT_PROTOCOL = "cohortkv_single_config_full_chain_development_v1"
SOURCE_RESULT = Path(
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage2_compiler_seed0.json"
)
BLUEPRINT = Path("configs/cohortkv_single_config_v1/blueprint.json")
WORKLOAD = Path("configs/cohortkv_single_config_v1/workload_manifest.json")
OUTPUT = Path("configs/cohortkv_single_config_v1/stage2_compiler_summary.json")
SOURCE_VERSIONS = ("theta0", "theta4", "theta10")
TARGET_VERSION = "theta11"
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
ACTION_NAMES = (
    "reuse",
    "projection_only",
    "compiled_full_affine",
    "structural_p4",
    "structural_p8",
    "recompute",
)
EXPECTED_FALLBACKS = {
    "theta0": ("structural_p8", "recompute"),
    "theta4": ("recompute",),
    "theta10": ("structural_p8", "recompute"),
}


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(source: dict, workload: dict) -> None:
    if (
        source.get("protocol") != SOURCE_PROTOCOL
        or source.get("parent_protocol") != PARENT_PROTOCOL
        or source.get("status") != "stage2_complete"
        or source.get("study_stage")
        != "single_configuration_seed0_development"
        or source.get("seed") != 0
        or source.get("labels_used") is not False
        or source.get("final_test_evaluated") is not False
    ):
        raise ValueError("Stage 2 source result identity is invalid")
    if source.get("role_counts") != {
        "fit": 40,
        "program_selection": 60,
        "certificate": 60,
        "final_test": 522,
    }:
        raise ValueError("Stage 2 role counts differ from the frozen split")
    blueprint = source.get("blueprint", {})
    workload_descriptor = source.get("workload_manifest", {})
    if (
        blueprint.get("path") != str(BLUEPRINT)
        or blueprint.get("protocol") != PARENT_PROTOCOL
        or not isinstance(blueprint.get("sha256"), str)
        or len(blueprint["sha256"]) != 64
        or workload_descriptor.get("path") != str(WORKLOAD)
        or workload_descriptor.get("protocol")
        != "cohortkv_single_config_workload_v1"
        or workload_descriptor.get("content_sha256")
        != workload["content_sha256"]
        or workload_descriptor.get("sha256")
        != sha256_file(Path(__file__).resolve().parents[1] / WORKLOAD)
    ):
        raise ValueError("Stage 2 parent artifact descriptor is invalid")
    if (
        source.get("workload_content_sha256")
        != workload["content_sha256"]
    ):
        raise ValueError("Stage 2 workload content hash mismatch")
    frozen = source.get("frozen_hyperparameters", {})
    if (
        frozen.get("attention_mix") != 1.0
        or frozen.get("ridge") != 0.001
        or frozen.get("fit_users") != 40
        or frozen.get("sampled_tokens_per_layer") != [8192] * 16
        or frozen.get("primary_recovery_target") != 0.7
        or frozen.get("threshold_sweep") != list(THRESHOLDS)
        or frozen.get("action_library")
        != [
            "projection_only",
            "compiled_full_affine",
            "structural_p4",
            "structural_p8",
            "recompute",
        ]
    ):
        raise ValueError("Stage 2 frozen hyperparameters changed")
    storage = source.get("source_storage_preflight", {})
    if (
        storage.get("mount") != "/data"
        or storage.get("device") != "/dev/nvme2n1p1"
        or storage.get("filesystem") != "ext4"
        or storage.get("free_bytes_before_materialization", 0)
        < storage.get("minimum_required_bytes", 1)
        or storage.get("temporary_shards_retained") is not False
    ):
        raise ValueError("Stage 2 source storage preflight is invalid")


def certificate_records(
    pair: dict,
    expected_records: dict[int, dict],
) -> None:
    records = pair.get("certificate_records", [])
    if (
        len(records) != len(expected_records)
        or {record["record_id"] for record in records}
        != set(expected_records)
    ):
        raise ValueError("Stage 2 certificate record coverage is invalid")
    for record in records:
        expected = expected_records[record["record_id"]]
        if (
            record.get("evaluation_role") != "certificate"
            or record.get("user_id") != expected["user_id"]
            or record.get("prefix_tokens") != expected["prefix_tokens"]
            or set(record.get("configs", {})) != set(ACTION_NAMES)
        ):
            raise ValueError("Stage 2 certificate record identity is invalid")
        for values in record["configs"].values():
            if set(values) != {
                "cache_error_rel",
                "hidden_cosine",
                "score_cosine",
                "top100_overlap",
            } or any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values.values()
            ) or (
                values["cache_error_rel"] < 0
                or not -1.000001 <= values["hidden_cosine"] <= 1.000001
                or not -1.000001 <= values["score_cosine"] <= 1.000001
                or not 0 <= values["top100_overlap"] <= 1.000001
            ):
                raise ValueError("Stage 2 certificate metric is invalid")


def metric_error(record: dict, action: str, metric: str) -> float:
    values = record["configs"][action]
    if metric == "cache":
        return values["cache_error_rel"]
    if metric == "score":
        return max(0.0, 1.0 - values["score_cosine"])
    if metric == "top100":
        return max(0.0, 1.0 - values["top100_overlap"])
    raise ValueError("unsupported Stage 2 certificate metric")


def recovery(records: list[dict], action: str, metric: str) -> float:
    reuse = sum(
        metric_error(record, "reuse", metric)
        for record in records
    ) / len(records)
    action_value = sum(
        metric_error(record, action, metric)
        for record in records
    ) / len(records)
    exact = sum(
        metric_error(record, "recompute", metric)
        for record in records
    ) / len(records)
    denominator = reuse - exact
    if denominator <= 1e-12:
        return math.nan
    return (reuse - action_value) / denominator


def validate_action_derivations(pair: dict) -> None:
    records = pair["certificate_records"]
    exact_ms = pair["action_summary"]["recompute"][
        "migration_ms_per_user"
    ]
    for name in ACTION_NAMES:
        values = pair["action_summary"][name]
        expected = {
            "cache_error_rel": sum(
                record["configs"][name]["cache_error_rel"]
                for record in records
            )
            / len(records),
            "hidden_cosine": sum(
                record["configs"][name]["hidden_cosine"]
                for record in records
            )
            / len(records),
            "score_cosine": sum(
                record["configs"][name]["score_cosine"]
                for record in records
            )
            / len(records),
            "top100_overlap": sum(
                record["configs"][name]["top100_overlap"]
                for record in records
            )
            / len(records),
            "cache_recovery": recovery(records, name, "cache"),
            "score_recovery": recovery(records, name, "score"),
            "top100_recovery": recovery(records, name, "top100"),
            "cost_ratio_to_exact": (
                values["migration_ms_per_user"] / exact_ms
            ),
        }
        expected["worst_view_recovery"] = min(
            expected["cache_recovery"],
            expected["score_recovery"],
            expected["top100_recovery"],
        )
        if any(
            not math.isclose(
                values.get(field, math.nan),
                result,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for field, result in expected.items()
        ):
            raise ValueError("Stage 2 action summary is not derived from records")


def validate_certificate_derivations(pair: dict) -> None:
    source_version = pair["source_version"]
    source_index = int(source_version.removeprefix("theta"))
    actions = tuple(
        MigrationActionSpec(
            name=value["name"],
            kind=value["kind"],
            required_state=value["required_state"],
            program_path=value.get("program_path"),
            replay_depth=value.get("replay_depth"),
        )
        for value in pair["actions"]
    )
    costs = {
        action.name: pair["action_summary"][action.name][
            "cost_ratio_to_exact"
        ]
        for action in actions
    }
    derived = []
    for threshold in THRESHOLDS:
        plan = compile_verified_plan(
            protocol=SOURCE_PROTOCOL,
            source_version=source_version,
            target_version=TARGET_VERSION,
            actions=actions,
            records=pair["certificate_records"],
            cost_ratios=costs,
            contract=FidelityContract(
                recovery_target=threshold,
                minimum_coverage=0.8,
                confidence_level=0.9,
                max_cost_ratio=0.3,
                bootstrap_samples=1000,
                minimum_probe_users=50,
            ),
            seed=source_index * 10007,
        ).to_dict()
        derived.append(
            {
                "recovery_target": threshold,
                "selected_action": plan["selected_action"],
                "selection_reason": plan["selection_reason"],
                "fallback_actions": plan["fallback_actions"],
                "certificates": plan["certificates"],
            }
        )
    normalized = json.loads(json.dumps(derived))
    if normalized != pair["threshold_sweep"]:
        raise ValueError("Stage 2 certificates are not derived from raw records")
    primary = normalized[THRESHOLDS.index(0.7)]
    selected = next(
        value
        for value in primary["certificates"]
        if value["action_name"] == primary["selected_action"]
    )
    if (
        primary["selected_action"] != pair["selected_action"]
        or primary["fallback_actions"] != pair["fallback_actions"]
        or selected != pair["primary_certificate"]["selected_certificate"]
    ):
        raise ValueError("Stage 2 primary certificate differs from the sweep")


def validate_threshold_sweep(pair: dict) -> list[dict]:
    sweep = pair.get("threshold_sweep", [])
    if [value.get("recovery_target") for value in sweep] != list(THRESHOLDS):
        raise ValueError("Stage 2 threshold sweep is incomplete")
    if [value.get("selected_action") for value in sweep] != [
        "compiled_full_affine",
        "compiled_full_affine",
        "compiled_full_affine",
        "compiled_full_affine",
        "recompute",
    ]:
        raise ValueError("Stage 2 threshold actions differ from the frozen result")
    output = []
    for value in sweep:
        if (
            value.get("selection_reason") not in {
                "minimum_cost_certified_within_budget",
                "minimum_cost_certified_budget_overflow",
            }
            or not isinstance(value.get("fallback_actions"), list)
            or len(value.get("certificates", [])) != 5
        ):
            raise ValueError("Stage 2 threshold plan is invalid")
        output.append(
            {
                "recovery_target": value["recovery_target"],
                "selected_action": value["selected_action"],
                "selection_reason": value["selection_reason"],
                "fallback_actions": value["fallback_actions"],
            }
        )
    return output


def validate_pair(
    root: Path,
    pair: dict,
    expected_records: dict[int, dict],
    workload_hash: str,
) -> dict:
    source_version = pair.get("source_version")
    fallback_actions = tuple(pair.get("fallback_actions", ()))
    if (
        source_version not in SOURCE_VERSIONS
        or pair.get("target_version") != TARGET_VERSION
        or pair.get("selected_action") != "compiled_full_affine"
        or pair.get("selection_reason")
        != "minimum_cost_certified_within_budget"
        or fallback_actions != EXPECTED_FALLBACKS[source_version]
        or tuple(pair.get("executable_fallback_actions", ()))
        != fallback_actions
        or set(pair.get("action_summary", {})) != set(ACTION_NAMES)
    ):
        raise ValueError("Stage 2 primary pair decision is invalid")
    certificate_records(pair, expected_records)
    validate_action_derivations(pair)
    validate_certificate_derivations(pair)
    certificate = pair.get("primary_certificate", {})
    selected_certificate = certificate.get("selected_certificate", {})
    selected_summary = certificate.get("selected_summary", {})
    serialized = certificate.get("serialized_source", {})
    if (
        certificate.get("source_dtype") != "float16"
        or certificate.get("residual_hidden_suffix_dtype") != "bfloat16"
        or certificate.get("program_dtype") != "float16"
        or certificate.get("output_dtype") != "float16"
        or certificate.get("passed") is not True
        or certificate.get("certificate_users") != 60
        or certificate.get("views") != ["cache", "score", "top100"]
        or selected_certificate.get("action_name")
        != "compiled_full_affine"
        or selected_certificate.get("fidelity_passed") is not True
        or selected_certificate.get("budget_passed") is not True
        or selected_certificate.get("cost_ratio", 1) > 0.3
    ):
        raise ValueError("Stage 2 deployed certificate is invalid")
    metrics = selected_certificate.get("metrics", [])
    if (
        [value.get("metric") for value in metrics]
        != ["cache", "score", "top100"]
        or any(value.get("passed") is not True for value in metrics)
        or selected_certificate.get("worst_recovery_lower_bound", 0) < 0.7
        or selected_certificate.get("worst_coverage_lower_bound", 0) < 0.8
    ):
        raise ValueError("Stage 2 primary fidelity contract did not pass")
    expected_tokens = sum(
        record["prefix_tokens"] for record in expected_records.values()
    )
    if (
        serialized.get("protocol")
        != "cohortkv_stage2_certificate_shard_v1"
        or serialized.get("batches") != 15
        or serialized.get("records") != 60
        or serialized.get("valid_prefix_tokens") != expected_tokens
        or serialized.get("logical_tensor_bytes", 0) < 1
        or serialized.get("physical_bytes", 0) < 1
        or serialized.get("residual_hidden_suffix_dtype") != "bfloat16"
        or serialized.get("residual_hidden_suffix_absmax", 0)
        <= 65_504
        or serialized.get(
            "residual_hidden_suffix_fp16_overflow_values",
            0,
        )
        < 1
        or serialized.get("temporary_shards_retained") is not False
    ):
        raise ValueError("Stage 2 serialized certificate source is invalid")
    plan_descriptor = pair.get("executable_plan", {})
    plan_path = root / plan_descriptor["path"]
    if (
        sha256_file(plan_path) != plan_descriptor.get("sha256")
        or plan_descriptor.get("protocol")
        != "cohortkv_executable_migration_plan_v1"
    ):
        raise ValueError("Stage 2 executable plan descriptor is invalid")
    plan = load_executable_plan(
        plan_path,
        repository_root=root,
        expected_sha256=plan_descriptor["sha256"],
    )
    expected_contract = {
        "recovery_target": 0.7,
        "minimum_coverage": 0.8,
        "confidence_level": 0.9,
        "max_cost_ratio": 0.3,
        "bootstrap_samples": 1000,
        "minimum_probe_users": 50,
        "metrics": ["cache", "score", "top100"],
    }
    primary_sweep = pair["threshold_sweep"][THRESHOLDS.index(0.7)]
    if (
        plan.source_version != source_version
        or plan.target_version != TARGET_VERSION
        or plan.action_chain
        != ("compiled_full_affine", *fallback_actions)
        or plan.payload.get("parent_protocol") != SOURCE_PROTOCOL
        or plan.payload.get("contract") != expected_contract
        or plan.payload.get("actions") != pair["actions"]
        or plan.payload.get("certificates")
        != primary_sweep["certificates"]
        or plan.payload.get("selection_reason")
        != pair["selection_reason"]
        or plan.payload.get("threshold_sweep")
        != pair["threshold_sweep"]
        or plan.payload.get("deployed_representation_certificate")
        != pair["primary_certificate"]
        or plan.payload.get("runtime_program")
        != pair["runtime_program"]
        or plan.payload.get("compiler_cost") != pair["compiler_cost"]
        or plan.payload.get("workload_content_sha256") != workload_hash
        or plan.required_representations("structural_p4")
        != ("raw_history", "residual_hidden_suffix_p4_bf16")
        or plan.required_representations("structural_p8")
        != ("raw_history", "residual_hidden_suffix_p8_bf16")
    ):
        raise ValueError("Stage 2 executable plan does not match the result")
    runtime = pair.get("runtime_program", {})
    if (
        runtime.get("sha256") != plan.program_descriptor["sha256"]
        or runtime.get("bytes") != plan.program_descriptor["bytes"]
        or runtime.get("dtype") != "float16"
        or runtime.get("weights_shape") != [16, 512, 1024]
        or runtime.get("biases_shape") != [16, 1024]
    ):
        raise ValueError("Stage 2 runtime program descriptor is invalid")
    threshold_sweep = validate_threshold_sweep(pair)
    action_summary = {}
    for name in ACTION_NAMES:
        values = pair["action_summary"][name]
        fields = (
            "cache_recovery",
            "score_recovery",
            "top100_recovery",
            "worst_view_recovery",
            "migration_ms_per_user",
            "cost_ratio_to_exact",
        )
        if any(
            not isinstance(values.get(field), (int, float))
            or not math.isfinite(values[field])
            for field in fields
        ):
            raise ValueError("Stage 2 action summary contains invalid values")
        action_summary[name] = {field: values[field] for field in fields}
    compiler_cost = pair.get("compiler_cost", {})
    if any(
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in compiler_cost.values()
    ):
        raise ValueError("Stage 2 compiler cost is invalid")
    amortization = pair.get("amortization", {})
    if (
        amortization.get("one_time_seconds", 0) <= 0
        or amortization.get("resident_seconds_saved_per_record", 0) <= 0
        or amortization.get("resident_break_even_records", 0) < 1
    ):
        raise ValueError("Stage 2 amortization floor is invalid")
    return {
        "source_version": source_version,
        "target_version": TARGET_VERSION,
        "selected_action": pair["selected_action"],
        "selection_reason": pair["selection_reason"],
        "fallback_actions": list(fallback_actions),
        "executable_fallback_actions": list(fallback_actions),
        "runtime_program": runtime,
        "executable_plan": plan_descriptor,
        "selected_certificate": {
            "cost_ratio_to_exact": selected_summary[
                "cost_ratio_to_exact"
            ],
            "cache_recovery": selected_summary["cache_recovery"],
            "score_recovery": selected_summary["score_recovery"],
            "top100_recovery": selected_summary["top100_recovery"],
            "worst_view_recovery": selected_summary[
                "worst_view_recovery"
            ],
            "worst_recovery_lower_bound": selected_certificate[
                "worst_recovery_lower_bound"
            ],
            "worst_coverage_lower_bound": selected_certificate[
                "worst_coverage_lower_bound"
            ],
            "certificate_passed": True,
        },
        "action_summary": action_summary,
        "threshold_sweep": threshold_sweep,
        "serialized_certificate_source": serialized,
        "compiler_cost": compiler_cost,
        "amortization": amortization,
    }


def build_summary(
    root: Path,
    source_path: Path,
    source: dict,
    workload: dict,
) -> dict:
    validate_source(source, workload)
    expected_records = {
        record["record_id"]: record
        for record in workload["records"]
        if record["evaluation_role"] == "certificate"
    }
    pairs = sorted(
        (
            validate_pair(
                root,
                pair,
                expected_records,
                workload["content_sha256"],
            )
            for pair in source.get("pairs", [])
        ),
        key=lambda pair: SOURCE_VERSIONS.index(pair["source_version"]),
    )
    if [pair["source_version"] for pair in pairs] != list(SOURCE_VERSIONS):
        raise ValueError("Stage 2 source-version coverage is incomplete")
    rq2 = source.get("rq2_compiler", {})
    cohorts = rq2.get("cohorts", [])
    if (
        [value.get("source_version") for value in cohorts]
        != list(SOURCE_VERSIONS)
        or any(
            value.get("selected_action") != "compiled_full_affine"
            or value.get("fallback_actions")
            != next(
                pair["fallback_actions"]
                for pair in pairs
                if pair["source_version"] == value["source_version"]
            )
            or value.get("deployed_representation_certificate", {}).get(
                "passed"
            )
            is not True
            for value in cohorts
        )
        or [
            value.get("recovery_target")
            for value in rq2.get("threshold_sweep", [])
        ]
        != list(THRESHOLDS)
        or any(
            action.get("selected_action") != "compiled_full_affine"
            for threshold in rq2["threshold_sweep"][:4]
            for action in threshold["cohort_actions"]
        )
        or any(
            action.get("selected_action") != "recompute"
            for action in rq2["threshold_sweep"][4]["cohort_actions"]
        )
        or not math.isclose(
            rq2.get("compile_seconds", -1),
            sum(
                pair["compiler_cost"]["runtime_prepare_seconds"]
                for pair in pairs
            ),
        )
        or not math.isclose(
            rq2.get("certificate_seconds", -1),
            sum(
                pair["compiler_cost"]["certificate_seconds"]
                for pair in pairs
            ),
        )
    ):
        raise ValueError("Stage 2 aggregate RQ2 result is invalid")
    return {
        "protocol": PROTOCOL,
        "status": "stage2_frozen",
        "study_stage": "single_configuration_seed0_development",
        "source_result": {
            "path": str(source_path),
            "sha256": sha256_file(root / source_path),
            "protocol": SOURCE_PROTOCOL,
        },
        "parent_blueprint": {
            **source["blueprint"],
            "hash_scope": (
                "blueprint bytes used by Stage 2 before the downstream "
                "Stage-2 completion amendment"
            ),
        },
        "workload": {
            "path": str(WORKLOAD),
            "file_sha256": sha256_file(root / WORKLOAD),
            "content_sha256": workload["content_sha256"],
            "certificate_records": len(expected_records),
            "certificate_prefix_tokens": sum(
                record["prefix_tokens"]
                for record in expected_records.values()
            ),
        },
        "measurement_boundary": {
            "execution": (
                "serialized certificate shards reloaded into deployed "
                "numeric representations"
            ),
            "primary_source_dtype": "float16",
            "runtime_program_dtype": "float16",
            "output_dtype": "float16",
            "residual_hidden_suffix_dtype": "bfloat16",
            "certificate_users_per_pair": 60,
            "final_test_users_evaluated": False,
            "recommendation_labels_used": False,
            "world_size": source["environment"]["world_size"],
            "gpu_name": source["environment"]["gpu_name"],
        },
        "frozen_hyperparameters": source["frozen_hyperparameters"],
        "pairs": pairs,
        "aggregate": {
            "compile_seconds": rq2["compile_seconds"],
            "certificate_seconds": rq2["certificate_seconds"],
            "historical_fit_seconds": rq2["historical_fit_seconds"],
            "full_catalog_score_seconds": rq2[
                "full_catalog_score_seconds"
            ],
            "amortized_seconds_per_record_at_682": rq2[
                "amortized_seconds_per_record"
            ],
            "amortization_curve": rq2["amortization_curve"],
            "program_bytes": sum(
                pair["runtime_program"]["bytes"] for pair in pairs
            ),
            "threshold_actions": [
                {
                    "recovery_target": value["recovery_target"],
                    "cohort_actions": value["cohort_actions"],
                }
                for value in rq2["threshold_sweep"]
            ],
            "boundary": rq2["boundary"],
        },
        "design_correction": {
            "rejected_representation": "residual_hidden_suffix_fp16",
            "observed_failure": (
                "unnormalized old hidden suffix exceeds the FP16 finite "
                "range in every source cohort"
            ),
            "replacement": "residual_hidden_suffix_bf16",
            "bytes_per_element_unchanged": True,
            "primary_compiled_path_changed": False,
            "logical_goal_changed": False,
        },
        "downstream_rule": {
            "stage3_operator_input": "compiled FP16 capsule and program",
            "stage4_plan_loader": (
                "load the checked executable plan and preserve its ordered "
                "fallback chain"
            ),
            "residual_fallback_requirement": (
                "theta0/theta10 p8 fallback is executable only when its "
                "BF16 old hidden suffix shard is retained"
            ),
            "amortization_boundary": (
                "the Stage 2 number is a resident compiler floor; Stage 4 "
                "must add source reads, destination writes, and commit"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    source_path = Path(args.source_result)
    output_path = Path(args.output)
    source = json.loads((root / source_path).read_text())
    workload = json.loads((root / WORKLOAD).read_text())
    payload = canonical_json_bytes(
        build_summary(root, source_path, source, workload)
    )
    resolved = root / output_path
    if args.check:
        if not resolved.is_file() or resolved.read_bytes() != payload:
            raise RuntimeError("Stage 2 frozen summary differs from source result")
        status = "verified"
    else:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(payload)
        status = "frozen"
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "status": status,
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
