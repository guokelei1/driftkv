from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from hstu_kvcache.migration import (
    D2ActionPlan,
    build_d2_record_owner_map,
    canonical_sha256,
    d2_record_owner_map_sha256,
)
from hstu_kvcache.migration.design2_plan import file_sha256

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cohortkv_d2_stage_b_frozen_v1"
NORMAL_PROTOCOL = "cohortkv_d2_stage_b_distributed_primitives_v1"
FAILURE_PROTOCOL = "cohortkv_d2_stage_b_hard_failure_v1"
OUTPUT = Path("configs/cohortkv_d2/stage_b_summary.json")
ACTION_PLAN = Path(
    "configs/cohortkv_d2/"
    "action_plan_theta1_theta2_staggered_renewal_h12.json"
)
STAGE_A = Path("configs/cohortkv_d2/stage_a_summary.json")
SAMPLE = Path("configs/cohortkv_d2/stage_b_sample_inputs.json")
NORMAL = {
    world_size: Path(
        f"configs/cohortkv_d2/stage_b_w{world_size}_primitives.json"
    )
    for world_size in (1, 2, 4)
}
W1_REPEAT = Path(
    "configs/cohortkv_d2/stage_b_w1_repeat_primitives.json"
)
W2_CROSS_ISLAND = Path(
    "configs/cohortkv_d2/stage_b_w2_cross_island_primitives.json"
)
FAILURES = {
    world_size: Path(
        f"configs/cohortkv_d2/stage_b_w{world_size}_hard_failure.json"
    )
    for world_size in (2, 4)
}
IMPLEMENTATION_FILES = (
    Path("src/hstu_kvcache/migration/__init__.py"),
    Path("src/hstu_kvcache/migration/design2_distributed.py"),
    Path("src/hstu_kvcache/migration/design2_embedding.py"),
    Path("src/hstu_kvcache/migration/design2_owner.py"),
    Path("src/hstu_kvcache/migration/design2_transaction.py"),
    Path("scripts/materialize_cohortkv_design2_stage_b_inputs.py"),
    Path("scripts/run_cohortkv_design2_distributed_tests.py"),
    Path("scripts/launch_cohortkv_design2_stage_b.py"),
    Path("scripts/freeze_cohortkv_design2_stage_b.py"),
    Path("tests/test_design2_sharded_embedding.py"),
    Path("tests/test_design2_owner_compute.py"),
    Path("tests/test_design2_transaction.py"),
    Path("tests/test_design2_faults.py"),
)
EXPECTED_LEDGER = {
    "all_exact_retained_tokens": 637954,
    "mixed_scheduled_retained_tokens": 50099,
    "natural_exact_prefix_tokens": 82612,
    "delta_tokens": 213669,
    "latest_tokens": 682,
    "all_exact_full_wave_tokens": 934917,
    "mixed_full_wave_tokens": 347062,
}
ZERO_COMPILED_COUNTERS = {
    "item_lookup_calls": 0,
    "embedding_collective_count": 0,
    "embedding_collective_bytes": 0,
    "old_kv_p2p_bytes": 0,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> dict[str, object]:
    resolved = ROOT / path
    require(resolved.is_file(), f"missing Stage B input: {path}")
    return json.loads(resolved.read_text())


def descriptor(path: Path, protocol: str | None = None) -> dict[str, object]:
    resolved = ROOT / path
    value: dict[str, object] = {
        "path": str(path),
        "sha256": file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if protocol is not None:
        value["protocol"] = protocol
    return value


def _content_sha256(value: dict[str, object]) -> str:
    return canonical_sha256(
        {
            key: item
            for key, item in value.items()
            if key != "content_sha256"
        }
    )


def _validate_stage_a_lock(stage_a: dict[str, object]) -> None:
    require(
        stage_a["status"] == "complete"
        and stage_a["stage_b_entry"] == "go",
        "Stage A is not a frozen GO",
    )
    for value in stage_a["implementation"]:
        path = ROOT / value["path"]
        require(path.is_file(), "Stage A implementation file is missing")
        require(
            file_sha256(path) == value["sha256"]
            and path.stat().st_size == value["bytes"],
            f"Stage A implementation lock differs: {value['path']}",
        )


def _validate_model_inputs(
    value: dict[str, object],
) -> None:
    for name in ("source_checkpoint", "target_checkpoint", "program"):
        current = value["model_inputs"][name]
        path = Path(current["path"])
        resolved = path if path.is_absolute() else ROOT / path
        require(
            resolved.is_file()
            and file_sha256(resolved) == current["sha256"],
            f"Stage B {name} hash differs",
        )


def _validate_lookup(
    lookup: dict[str, object],
    world_size: int,
) -> None:
    require(
        all(lookup["checks"].values()),
        "Stage B lookup reconstruction differs",
    )
    require(
        len(lookup["remote_send_matrix"]) == world_size
        and len(lookup["remote_receive_matrix"]) == world_size
        and len(lookup["rank_id_evidence"]) == world_size,
        "Stage B lookup rank coverage differs",
    )
    require(
        lookup["actual_collective_tensor_payload_bytes"]
        == lookup["counts_collective_input_bytes"]
        + lookup["counts_collective_output_bytes"]
        + lookup["id_collective_input_bytes"]
        + lookup["id_collective_output_bytes"]
        + lookup["vector_collective_input_bytes"]
        + lookup["vector_collective_output_bytes"],
        "Stage B collective payload does not reconstruct",
    )
    for evidence in lookup["rank_id_evidence"]:
        require(
            all(_SHA256.fullmatch(item) for item in evidence.values()),
            "Stage B lookup ID evidence is malformed",
        )


def _validate_rank(
    report: dict[str, object],
    world_size: int,
) -> None:
    require(
        0 <= report["rank"] < world_size,
        "Stage B rank identity differs",
    )
    require(
        report["preflight"]["passed"],
        "Stage B rank preflight failed",
    )
    require(
        report["compiled_retained"]["metrics"]["phase_counters"]
        == ZERO_COMPILED_COUNTERS,
        "compiled retained phase touched a forbidden resource",
    )
    require(
        report["compiled_retained"]["metadata"]["owner_rank"]
        == report["rank"],
        "compiled retained ran on a non-owner",
    )
    require(
        report["compiled_delta_append"]["lookup"]["requested_tokens"]
        == report["compiled_delta_append"]["tokens"]
        and report["compiled_latest_append"]["lookup"][
            "requested_tokens"
        ]
        == report["compiled_latest_append"]["tokens"],
        "append lookup accounting differs",
    )
    require(
        report["transaction_ready"]["status"] == "ready"
        and not report["transaction_ready"]["publishes_target_epoch"]
        and report["transaction_abort"]["status"] == "abort"
        and not report["transaction_abort"]["publishes_target_epoch"],
        "Stage B private transaction boundary differs",
    )
    require(
        report["cooperative_failure"]["isolated_probe"]
        and report["cooperative_failure"][
            "does_not_gate_normal_private_fragment"
        ]
        and not report["cooperative_failure"]["passed"],
        "Stage B isolated cooperative failure differs",
    )
    private = report["private_fragment"]
    require(
        private["dtype"] == "float16"
        and private["payload_bytes"] > 0
        and _SHA256.fullmatch(private["sha256"])
        and all(
            _SHA256.fullmatch(value)
            for value in private["component_sha256"].values()
        ),
        "Stage B private payload checksum differs",
    )
    require(
        all(
            evidence["passed"]
            and evidence["expected"] == evidence["observed"]
            for evidence in report["lookup_input_evidence"].values()
        ),
        "Stage B rank ID reconstruction differs",
    )
    require(
        report["placement_ring"]["returned_output_bitwise"],
        "Stage B placement return differs",
    )


def _validate_normal(
    value: dict[str, object],
    world_size: int,
    plan: D2ActionPlan,
    action_file_sha256: str,
    stage_a_sha256: str,
    sample_sha256: str,
    sample: dict[str, object],
    topology: dict[str, object],
) -> dict[str, object]:
    require(value["protocol"] == NORMAL_PROTOCOL, "normal protocol differs")
    require(
        value["status"] == "complete"
        and value["scientific_result"] is False
        and all(value["checks"].values()),
        "normal Stage B artifact is incomplete",
    )
    require(
        value["action_plan"]["content_sha256"] == plan.content_sha256
        and value["action_plan"]["file_sha256"] == action_file_sha256
        and value["stage_a_summary"]["sha256"] == stage_a_sha256
        and value["sample_inputs"]["sha256"] == sample_sha256,
        "normal Stage B provenance differs",
    )
    configuration = value["configuration"]
    owner_map = build_d2_record_owner_map(
        plan,
        world_size,
        "strict_cow_lpt",
    )
    require(
        configuration["world_size"] == world_size
        and configuration["backend"] == "nccl"
        and configuration["embedding_owner"] == "item_id_mod_world_size"
        and configuration["record_owner"] == "strict_cow_lpt"
        and configuration["embedding_transport_dtype"] == "float32"
        and configuration["publication_dtype"] == "float16"
        and configuration["record_owner_map_sha256"]
        == d2_record_owner_map_sha256(owner_map)
        == sample["selections"][str(world_size)]["owner_map_sha256"],
        "normal Stage B configuration differs",
    )
    scope = value["scope"]
    require(
        scope["primitive_correctness_only"]
        and scope["actual_collective_tensor_payload_recorded"]
        and scope["old_kv_source_materialized_as_test_fixture"]
        and scope["full_plan_ledger_replayed_without_full_wave_execution"]
        and not scope["full_mixed_wave_executed"]
        and not scope["target_epoch_published"]
        and not scope["nccl_wire_bytes_observed"],
        "normal Stage B scope differs",
    )
    require(
        value["stage_a_ledger_recheck"]["expected"] == EXPECTED_LEDGER
        and all(
            value["stage_a_ledger_recheck"]["observed"][key] == item
            for key, item in EXPECTED_LEDGER.items()
        ),
        "Stage A phase ledger continuity differs",
    )
    require(
        len(value["rank_reports"]) == world_size
        and tuple(
            report["rank"] for report in value["rank_reports"]
        )
        == tuple(range(world_size)),
        "normal Stage B rank coverage differs",
    )
    for report in value["rank_reports"]:
        _validate_rank(report, world_size)
    for lookup in value["lookups"].values():
        _validate_lookup(lookup, world_size)
    projected = [
        report["capacity"]["projected_admitted"]
        for report in value["rank_reports"]
    ]
    require(
        projected == ([False] if world_size == 1 else [True] * world_size),
        "Stage B projected strict-COW boundary differs",
    )
    if world_size > 1:
        require(
            all(
                report["capacity"]["projected_required_bytes"]
                < report["capacity"]["device_total_bytes"]
                for report in value["rank_reports"]
            ),
            "Stage B projected capacity has no positive margin",
        )
    if world_size == 4:
        expected_uuids = [
            item["uuid"] for item in topology["devices"]
        ]
        require(
            [item["device_uuid"] for item in value["rank_reports"]]
            == expected_uuids,
            "W4 physical device order differs from Stage A topology",
        )
        require(
            value["lookups"]["synthetic_routing"][
                "cross_island_tokens"
            ]
            > 0
            and any(
                report["placement_ring"]["cross_island_edge"]
                and report["placement_ring"]["old_kv_send_bytes"] > 0
                for report in value["rank_reports"]
            ),
            "W4 did not exercise a cross-island route",
        )
    _validate_model_inputs(value)
    return {
        "world_size": world_size,
        "backend": configuration["backend"],
        "visible_devices": configuration["cuda_visible_devices"],
        "record_owner_map_sha256": configuration[
            "record_owner_map_sha256"
        ],
        "embedding_owner_sha256": configuration[
            "embedding_owner_sha256"
        ],
        "device_uuids": [
            report["device_uuid"] for report in value["rank_reports"]
        ],
        "projected_capacity": [
            {
                "rank": report["rank"],
                "required_bytes": report["capacity"][
                    "projected_required_bytes"
                ],
                "capacity_bytes": report["capacity"][
                    "device_total_bytes"
                ],
                "margin_bytes": report["capacity"][
                    "device_total_bytes"
                ]
                - report["capacity"]["projected_required_bytes"],
                "admitted": report["capacity"]["projected_admitted"],
                "measured_peak_bytes": report["capacity"][
                    "torch_peak_allocated_bytes"
                ],
            }
            for report in value["rank_reports"]
        ],
        "collective_tensor_payload_bytes": {
            name: lookup["actual_collective_tensor_payload_bytes"]
            for name, lookup in value["lookups"].items()
        },
        "off_diagonal_bytes": {
            name: lookup["off_diagonal_bytes"]
            for name, lookup in value["lookups"].items()
        },
        "private_fragment_set_sha256": value["rank_reports"][0][
            "transaction_ready"
        ]["fragment_set_sha256"],
    }


def _validate_failure(
    value: dict[str, object],
    world_size: int,
    expected_device_uuids: list[str],
) -> dict[str, object]:
    require(
        value["protocol"] == FAILURE_PROTOCOL
        and value["status"] == "complete"
        and value["scientific_result"] is False
        and value["world_size"] == world_size
        and value["case"] == "hard_failure"
        and all(value["checks"].values()),
        "Stage B hard-failure artifact differs",
    )
    require(
        value["injected_failure"]
        == {"rank": 1, "mechanism": "os._exit", "exit_code": 23}
        and value["launch"]["backend"] == "nccl"
        and value["launch"]["device_uuids"] == expected_device_uuids
        and len(value["launch"]["visible_devices"]) == world_size
        and value["launch"]["returncode"] != 0
        and not value["launch"]["timed_out"]
        and not value["launch"]["process_group_alive_after_cleanup"],
        "Stage B hard-failure propagation differs",
    )
    require(
        value["content_sha256"] == _content_sha256(value),
        "Stage B hard-failure content hash differs",
    )
    return {
        "world_size": world_size,
        "backend": value["launch"]["backend"],
        "device_uuids": value["launch"]["device_uuids"],
        "elapsed_seconds": value["launch"]["elapsed_seconds"],
        "launcher_returncode": value["launch"]["returncode"],
        "rank1_exit_code": value["injected_failure"]["exit_code"],
        "bounded": value["checks"]["peer_termination_bounded"],
    }


def _validate_w1_repeat(
    primary: dict[str, object],
    repeat: dict[str, object],
) -> None:
    require(
        repeat["status"] == "complete"
        and repeat["configuration"]["world_size"] == 1
        and all(repeat["checks"].values()),
        "W1 repeat artifact differs",
    )
    for first, second in zip(
        primary["rank_reports"],
        repeat["rank_reports"],
        strict=True,
    ):
        require(
            first["private_fragment"]["component_sha256"]
            == second["private_fragment"]["component_sha256"]
            and first["private_fragment"]["sha256"]
            == second["private_fragment"]["sha256"]
            and first["transaction_ready"]["fragment_set_sha256"]
            == second["transaction_ready"]["fragment_set_sha256"],
            "W1 private payload checksum is not deterministic",
        )


def derive_summary() -> dict[str, object]:
    plan = D2ActionPlan.load(ROOT / ACTION_PLAN)
    action_file_sha256 = file_sha256(ROOT / ACTION_PLAN)
    stage_a = load(STAGE_A)
    sample = load(SAMPLE)
    _validate_stage_a_lock(stage_a)
    require(
        plan.counts.to_dict()
        == {
            "compiled": 548,
            "scheduled_exact": 46,
            "natural_exact": 88,
            "records": 682,
        }
        and plan.content_sha256
        == stage_a["action_plan"]["content_sha256"]
        and action_file_sha256
        == stage_a["action_plan"]["file_sha256"],
        "Stage B action-plan lock differs",
    )
    require(
        sample["status"] == "complete"
        and sample["scientific_result"] is False
        and sample["content_sha256"] == _content_sha256(sample)
        and sample["action_plan"]["content_sha256"]
        == plan.content_sha256
        and sample["action_plan"]["file_sha256"] == action_file_sha256
        and sample["stage_a_summary"]["sha256"]
        == file_sha256(ROOT / STAGE_A),
        "Stage B sample input binding differs",
    )
    topology_path = Path(stage_a["artifacts"]["p2p"]["path"])
    topology = load(topology_path)
    require(
        file_sha256(ROOT / topology_path)
        == stage_a["artifacts"]["p2p"]["sha256"],
        "Stage A topology hash differs",
    )
    normal_values = {
        world_size: load(path)
        for world_size, path in NORMAL.items()
    }
    normal_summaries = {
        world_size: _validate_normal(
            value,
            world_size,
            plan,
            action_file_sha256,
            file_sha256(ROOT / STAGE_A),
            file_sha256(ROOT / SAMPLE),
            sample,
            topology,
        )
        for world_size, value in normal_values.items()
    }
    repeat = load(W1_REPEAT)
    _validate_w1_repeat(normal_values[1], repeat)
    cross_island_value = load(W2_CROSS_ISLAND)
    cross_island_summary = _validate_normal(
        cross_island_value,
        2,
        plan,
        action_file_sha256,
        file_sha256(ROOT / STAGE_A),
        file_sha256(ROOT / SAMPLE),
        sample,
        topology,
    )
    require(
        cross_island_summary["device_uuids"]
        == [
            topology["devices"][1]["uuid"],
            topology["devices"][3]["uuid"],
        ]
        and any(
            value > 0
            for value in cross_island_summary[
                "off_diagonal_bytes"
            ].values()
        ),
        "supplemental W2 did not exercise the declared cross-island pair",
    )
    failure_summaries = {
        world_size: _validate_failure(
            load(path),
            world_size,
            normal_summaries[world_size]["device_uuids"],
        )
        for world_size, path in FAILURES.items()
    }
    return {
        "protocol": PROTOCOL,
        "status": "complete",
        "scientific_result": False,
        "stage": "B",
        "stage_c_entry": "go",
        "action_plan": {
            "content_sha256": plan.content_sha256,
            "file_sha256": action_file_sha256,
            "counts": plan.counts.to_dict(),
            "source_version": plan.source_version,
            "target_version": plan.target_version,
        },
        "gates": {
            "stage_a_reverse_audit": "pass",
            "g2_distributed_exact_and_append": "pass",
            "g3_distributed_resource_and_communication": "pass",
            "deterministic_collective_order": "pass",
            "private_ready_abort_no_publication": "pass",
            "hard_failure_propagation": "pass",
            "w2_w4_projected_capacity_admission": "pass",
            "g4_integrated_transaction": "deferred_to_stage_c",
            "g7_capacity_claim": "not_passed",
            "paper_performance_claim": "not_evaluated",
        },
        "normal_matrix": [
            normal_summaries[world_size]
            for world_size in (1, 2, 4)
        ],
        "hard_failure_matrix": [
            failure_summaries[world_size]
            for world_size in (2, 4)
        ],
        "supplemental_cross_island_w2": {
            **cross_island_summary,
            "does_not_replace_w4_gate": True,
        },
        "stage_a_ledger": EXPECTED_LEDGER,
        "decisions": {
            "embedding_transport": "fp32_correctness_baseline",
            "embedding_layout": "modulo_row_sharded",
            "record_layout": "strict_cow_lpt",
            "compiled_retained": "owner_local_zero_embedding_zero_oldkv_p2p",
            "append": "method_independent_sharded_lookup_counted",
            "dedup": "deferred_until_stage_c_full_wave_exposure",
            "publication": "private_fragments_only_until_stage_c",
            "single_rank_strict_cow": "rejected",
            "w2_w4_strict_cow": "projected_admission_not_g7_claim",
        },
        "scope": {
            "primitive_correctness": True,
            "real_theta1_theta2_samples": True,
            "full_682_record_phase_ledger": True,
            "full_682_record_mixed_wave_executed": False,
            "target_epoch_published": False,
            "actual_collective_tensor_payload": True,
            "nccl_wire_bytes": False,
            "timings_are_paper_results": False,
        },
        "unsupported_claims": [
            "an integrated mixed wave or target epoch publication",
            "end-to-end speedup or a paper performance result",
            "NCCL wire-byte measurement",
            "G7 full-cohort physical admission",
            "foreground embedding isolation",
            "dedup, overlap, or topology-aware placement benefit",
            "a model that cannot fit on one A40",
        ],
        "artifacts": {
            "action_plan": descriptor(ACTION_PLAN, plan.protocol),
            "stage_a": descriptor(STAGE_A, stage_a["protocol"]),
            "sample_inputs": descriptor(SAMPLE, sample["protocol"]),
            "w1": descriptor(NORMAL[1], NORMAL_PROTOCOL),
            "w1_repeat": descriptor(W1_REPEAT, NORMAL_PROTOCOL),
            "w2": descriptor(NORMAL[2], NORMAL_PROTOCOL),
            "w2_cross_island": descriptor(
                W2_CROSS_ISLAND,
                NORMAL_PROTOCOL,
            ),
            "w4": descriptor(NORMAL[4], NORMAL_PROTOCOL),
            "w2_hard_failure": descriptor(
                FAILURES[2],
                FAILURE_PROTOCOL,
            ),
            "w4_hard_failure": descriptor(
                FAILURES[4],
                FAILURE_PROTOCOL,
            ),
        },
        "implementation": [
            descriptor(path) for path in IMPLEMENTATION_FILES
        ],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    summary = derive_summary()
    output_path = ROOT / OUTPUT
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.check:
        require(output_path.is_file(), "Stage B summary is missing")
        require(
            output_path.read_text() == payload,
            "Stage B frozen summary differs",
        )
        return summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, output_path)
    return summary


def main() -> None:
    summary = run(parse_args())
    print(
        json.dumps(
            {
                "status": summary["status"],
                "stage": summary["stage"],
                "stage_c_entry": summary["stage_c_entry"],
                "scientific_result": summary["scientific_result"],
                "gates": summary["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
