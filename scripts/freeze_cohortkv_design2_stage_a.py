from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.migration import (
    D2ActionPlan,
    export_stage49_h12_action_plan,
)
from hstu_kvcache.migration.design2_plan import file_sha256

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cohortkv_d2_stage_a_frozen_v1"
OUTPUT = Path("configs/cohortkv_d2/stage_a_summary.json")
ACTION_PLAN = Path(
    "configs/cohortkv_d2/"
    "action_plan_theta1_theta2_staggered_renewal_h12.json"
)
ARTIFACTS = {
    "plan_adapter": (
        Path("configs/cohortkv_d2/stage_a_plan_adapter_summary.json"),
        "cohortkv_d2_stage_a_plan_adapter_v1",
    ),
    "exact_frontend": (
        Path(
            "configs/cohortkv_d2/"
            "stage_a_exact_frontend_validation.json"
        ),
        "cohortkv_d2_stage_a_exact_frontend_v1",
    ),
    "stage5_adapter": (
        Path(
            "configs/cohortkv_d2/"
            "stage_a_stage5_adapter_validation.json"
        ),
        "cohortkv_d2_stage_a_stage5_adapter_v1",
    ),
    "requests": (
        Path(
            "configs/cohortkv_d2/"
            "stage_a_request_characterization.json"
        ),
        "cohortkv_d2_stage_a_request_characterization_v1",
    ),
    "capacity": (
        Path(
            "configs/cohortkv_d2/"
            "stage_a_capacity_characterization.json"
        ),
        "cohortkv_d2_stage_a_capacity_characterization_v1",
    ),
    "p2p": (
        Path("configs/cohortkv_d2/stage_a_p2p_topology.json"),
        "cohortkv_d2_stage_a_p2p_topology_v2",
    ),
}
IMPLEMENTATION_FILES = (
    Path("src/hstu_kvcache/models/hstu.py"),
    Path("src/hstu_kvcache/migration/recompute.py"),
    Path("src/hstu_kvcache/migration/design2_plan.py"),
    Path("src/hstu_kvcache/migration/design2_runtime.py"),
    Path("src/hstu_kvcache/migration/design2_metrics.py"),
    Path("scripts/export_cohortkv_d2_action_plan.py"),
    Path("scripts/characterize_cohortkv_d2_requests.py"),
    Path("scripts/characterize_cohortkv_d2_capacity.py"),
    Path("scripts/benchmark_cohortkv_design2_p2p.py"),
    Path("scripts/validate_cohortkv_d2_exact_frontend.py"),
    Path("scripts/validate_cohortkv_d2_stage5_adapter.py"),
    Path("scripts/freeze_cohortkv_design2_stage_a.py"),
    Path("tests/test_design2_plan.py"),
    Path("tests/test_design2_exact_frontend.py"),
    Path("tests/test_design2_synthetic.py"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> dict:
    return json.loads((ROOT / path).read_text())


def descriptor(path: Path, protocol: str | None = None) -> dict:
    resolved = ROOT / path
    value = {
        "path": str(path),
        "sha256": file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if protocol is not None:
        value["protocol"] = protocol
    return value


def validate_artifact(
    value: dict,
    protocol: str,
    plan: D2ActionPlan,
    action_plan_file_sha256: str,
) -> None:
    require(value.get("protocol") == protocol, "artifact protocol differs")
    require(value.get("status") == "complete", "artifact status differs")
    require(
        value.get("scientific_result") is False,
        "Stage A artifact became a scientific result",
    )
    action = value.get("action_plan", {})
    require(
        action.get("content_sha256") == plan.content_sha256
        and action.get("file_sha256") == action_plan_file_sha256,
        "artifact action-plan binding differs",
    )
    checks = value.get("checks", {})
    require(
        isinstance(checks, dict)
        and checks
        and all(checks.values()),
        "artifact checks differ",
    )


def derive_summary() -> dict:
    plan_path = ROOT / ACTION_PLAN
    plan = D2ActionPlan.load(plan_path)
    action_plan_file_sha256 = file_sha256(plan_path)
    require(
        plan.counts.to_dict()
        == {
            "compiled": 548,
            "scheduled_exact": 46,
            "natural_exact": 88,
            "records": 682,
        },
        "action counts differ",
    )
    require(
        tuple(value.record_id for value in plan.records)
        == tuple(range(682)),
        "action record coverage differs",
    )
    reexported = export_stage49_h12_action_plan(
        plan.provenance.artifact,
        step_index=plan.provenance.step_index,
    )
    require(reexported == plan, "action plan no longer reexports exactly")
    require(
        file_sha256(ROOT / plan.provenance.artifact)
        == plan.provenance.artifact_sha256,
        "upstream H12 artifact hash differs",
    )
    loaded = {
        name: load(path)
        for name, (path, _) in ARTIFACTS.items()
    }
    for name, (path, protocol) in ARTIFACTS.items():
        validate_artifact(
            loaded[name],
            protocol,
            plan,
            action_plan_file_sha256,
        )
        require((ROOT / path).is_file(), "Stage A artifact is missing")

    ledger = loaded["plan_adapter"]["wave_report"]["phase_ledger"]
    mixed = {value["phase"]: value for value in ledger["mixed"]}
    require(
        {
            phase: (
                mixed[phase]["compute_tokens"],
                mixed[phase]["lookup_tokens"],
            )
            for phase in mixed
        }
        == {
            "compiled_retained": (587855, 0),
            "scheduled_exact_retained": (50099, 50099),
            "natural_exact_target_prefix": (82612, 82612),
            "delta_append": (213669, 213669),
            "latest_append": (682, 682),
        },
        "phase ledger differs",
    )
    require(
        all(
            value["physical_collective_bytes"] is None
            and value["collective_calls"] is None
            for value in ledger["mixed"]
        ),
        "static ledger invented physical collectives",
    )
    require(
        ledger["boundaries"]["retained_prefix"][
            "all_exact_lookup_tokens"
        ]
        == 637954
        and ledger["boundaries"]["retained_prefix"][
            "mixed_lookup_tokens"
        ]
        == 50099
        and ledger["boundaries"]["integrated_post_append"][
            "all_exact_lookup_tokens"
        ]
        == 934917
        and ledger["boundaries"]["integrated_post_append"][
            "mixed_lookup_tokens"
        ]
        == 347062,
        "phase boundary totals differ",
    )

    exact = loaded["exact_frontend"]
    replay = exact["phase_lookup_instrumentation"][
        "full_plan_request_replay"
    ]
    require(
        {
            phase: value["logical_lookup_tokens"]
            for phase, value in replay.items()
        }
        == {
            "compiled_retained": 0,
            "scheduled_exact_retained": 50099,
            "natural_exact_target_prefix": 82612,
            "delta_append": 213669,
            "latest_append": 682,
        },
        "observed lookup replay differs from ledger",
    )
    require(
        replay["compiled_retained"]["lookup_calls"] == 0
        and all(
            replay[phase]["lookup_calls"] > 0
            for phase in replay
            if phase != "compiled_retained"
        ),
        "phase lookup calls differ",
    )
    require(
        exact["configuration"]["frontend_item_vector_dtype"]
        == "float32"
        and exact["configuration"][
            "frontend_item_vector_element_bytes"
        ]
        == 4
        and all(
            value["mechanical_equivalence"] is False
            for value in exact[
                "transport_dtype_characterization"
            ].values()
        ),
        "transport dtype boundary differs",
    )

    stage5 = loaded["stage5_adapter"]
    require(
        stage5["scope"]["full_frozen_h12_action_plan"]
        and stage5["scope"]["actual_stage5_transaction_engine"]
        and stage5["scope"]["synthetic_kv_payload"]
        and not stage5["scope"]["real_edge_performance_measured"],
        "Stage 5 adapter scope differs",
    )
    require(
        stage5["scenarios"]["normal_commit"]["target_records"] == 682
        and stage5["scenarios"]["semantic_fallback"][
            "final_counts"
        ]
        == {"migrate": 0, "exact": 682}
        and stage5["scenarios"]["mid_job_abort"]["outcome"]
        == "aborted"
        and stage5["scenarios"]["pre_commit_abort"]["outcome"]
        == "aborted",
        "Stage 5 adapter behavior differs",
    )

    requests = loaded["requests"]
    exact_ceiling = requests["coalescing_ceilings"][
        "exact_retained_or_natural"
    ]
    require(
        exact_ceiling["requested_ids"] == 132711
        and exact_ceiling["unique_ids"] == 96844,
        "combined exact-prefix ceiling differs",
    )
    points = requests["scoped_dedup"]["points"]

    def maximum_reduction(
        world_size: int,
        batches: set[int] | None = None,
    ) -> float:
        values = [
            float(
                value["exact_prefix_coalesced"][
                    "remote_return_reduction"
                ]
            )
            for value in points
            if value["world_size"] == world_size
            and (
                batches is None
                or value["batch_records"] in batches
            )
        ]
        require(bool(values), "dedup scope is missing")
        return max(values)

    dedup = {
        "decision": "defer_until_stage_b_actual_collective_bytes",
        "implement_in_stage_b_baseline": False,
        "retain_as_p5_candidate": True,
        "gate": 0.10,
        "maximum_remote_return_reduction": {
            "w2_all_scopes": maximum_reduction(2),
            "w4_all_scopes": maximum_reduction(4),
            "w2_batch4": maximum_reduction(2, {4}),
            "w4_batch4": maximum_reduction(4, {4}),
        },
        "reason": (
            "large planned coalescing scopes exceed the static gate, "
            "while batch-4 scopes do not; physical and exposed bytes "
            "remain unmeasured"
        ),
    }
    require(
        dedup["maximum_remote_return_reduction"][
            "w2_all_scopes"
        ]
        >= 0.10
        and dedup["maximum_remote_return_reduction"][
            "w4_all_scopes"
        ]
        >= 0.10
        and dedup["maximum_remote_return_reduction"]["w2_batch4"]
        < 0.10
        and dedup["maximum_remote_return_reduction"]["w4_batch4"]
        < 0.10,
        "dedup decision boundary differs",
    )

    capacity = loaded["capacity"]
    cohort = capacity["cohort"]
    require(
        cohort["old_kv_bytes"] == 28383969280
        and cohort["complete_new_kv_bytes"] == 30635360256
        and cohort["strict_cow_kv_bytes"] == 59019329536,
        "capacity cohort bytes differ",
    )
    require(
        capacity["program"]["bytes"] == 33592613
        and capacity["program"]["tensor_bytes"] == 33587200,
        "program file and tensor bytes differ",
    )
    layouts = capacity["layouts"]
    w1 = [
        value
        for value in layouts
        if value["world_size"] == 1
    ]
    require(
        w1
        and not any(
            value["all_full_model_total_capacity_admitted"]
            for value in w1
        ),
        "single-rank strict-COW admission differs",
    )
    require(
        all(
            any(
                value["world_size"] == world_size
                and value["all_full_model_total_capacity_admitted"]
                for value in layouts
            )
            for world_size in (2, 4)
        ),
        "multi-rank static admission candidates differ",
    )

    capacity_owner_hashes = {
        (
            int(name.rsplit("_w", 1)[1]),
            name.rsplit("_w", 1)[0],
        ): value["sha256"]
        for name, value in capacity["owner_maps"].items()
    }
    for value in points:
        key = (
            value["world_size"],
            value["record_owner_strategy"],
        )
        require(
            capacity_owner_hashes[key]
            == value["record_owner_map_sha256"],
            "request and capacity owner maps differ",
        )

    p2p = loaded["p2p"]
    matrix = {
        (
            value["source_index"],
            value["destination_index"],
        ): value["direct_peer_supported"]
        for value in p2p["peer_matrix"]
    }
    require(
        all(
            matrix[(source, destination)]
            == matrix[(destination, source)]
            for source, destination in matrix
        ),
        "P2P peer matrix is asymmetric",
    )
    supported = {
        (source, destination)
        for (source, destination), enabled in matrix.items()
        if enabled and source != destination
    }
    measured = {
        (
            value["source_index"],
            value["destination_index"],
        )
        for value in p2p["direct_peer_measurements"]
    }
    require(
        supported == measured
        and not p2p["unmeasured_supported_pairs"],
        "direct peer measurement coverage differs",
    )
    require(
        p2p["scope"]["full_four_gpu_direct_peer_topology_measured"]
        and p2p["scope"]["concurrent_copy_measured"]
        and p2p["scope"]["real_compiled_compute_overlap_measured"]
        and p2p["scope"][
            "balanced_and_injected_owner_execution_measured"
        ]
        and not p2p["scope"]["cross_island_route_measured"]
        and not p2p["scope"]["nccl_send_recv_measured"],
        "P2P scope differs",
    )

    artifacts = {
        "action_plan": descriptor(
            ACTION_PLAN,
            plan.protocol,
        ),
        **{
            name: descriptor(path, protocol)
            for name, (path, protocol) in ARTIFACTS.items()
        },
    }
    return {
        "protocol": PROTOCOL,
        "status": "complete",
        "scientific_result": False,
        "stage": "A",
        "stage_b_entry": "go",
        "action_plan": {
            "content_sha256": plan.content_sha256,
            "file_sha256": action_plan_file_sha256,
            "upstream_artifact_sha256": (
                plan.provenance.artifact_sha256
            ),
            "action_partition_sha256": (
                plan.provenance.action_partition_sha256
            ),
            "prepared_data_sha256": (
                plan.provenance.prepared_data_sha256
            ),
            "counts": plan.counts.to_dict(),
        },
        "gates": {
            "g0_action_plan": "pass",
            "g1_mechanical_refactor": "pass",
            "p0_2_stage5_adapter_contract": "pass",
            "p0_3_phase_lookup": "pass",
            "p0_4_topology_and_microbench": "pass",
            "p0_5_request_characterization": "pass",
            "g2_distributed_exact": "not_started",
            "g3_distributed_communication": "not_started",
            "g7_capacity_claim": "not_passed",
            "paper_performance_claim": "not_evaluated",
        },
        "decisions": {
            "dedup": dedup,
            "strict_cow_single_rank": "no_go",
            "strict_cow_w2_w4": (
                "static_admission_candidate_only"
            ),
            "embedding_vector_transport": (
                "fp32_mechanical_baseline"
            ),
            "lower_precision_transport": (
                "candidate_requires_stage_b_correctness"
            ),
            "cross_island_transport": (
                "stage_b_nccl_route_required"
            ),
        },
        "falsified_hypotheses": [
            "compiled whole records are embedding-free",
            "single-A40 strict-COW admission is feasible",
            "the four A40s form one uniform direct-peer fabric",
            "two-byte embedding transport is mechanically equivalent",
            "dedup is a universal static no-go",
        ],
        "provisional_assumptions": [
            "W2/W4 capacity is modeled without an actual HBM source manifest",
            "CUDA context remains a two-GiB margin rather than a measured rank cost",
            "owner imbalance execution is a sampled microbenchmark",
            "cross-island NCCL routing and physical collective bytes are unmeasured",
            "FP32 remains the only mechanically equivalent item-vector path",
        ],
        "unsupported_claims": [
            "multi-rank owner-compute correctness",
            "row-sharded exact correctness or communication savings",
            "integrated mixed-wave speedup",
            "capacity G7 or a model-too-large-for-one-GPU claim",
            "foreground embedding-tier isolation",
            "any paper performance result from Stage A",
        ],
        "stage_b_first_diagnostics": [
            "world-size-1 SPMD must reproduce Stage A exact and phase ledgers",
            "2-rank sharded exact must report actual local remote IDs and collective bytes",
            "4-rank NCCL must exercise both NVLink islands and a cross-island route before optimization",
        ],
        "artifacts": artifacts,
        "implementation": [
            descriptor(path) for path in IMPLEMENTATION_FILES
        ],
    }


def main() -> None:
    args = parse_args()
    summary = derive_summary()
    output = ROOT / OUTPUT
    if args.check:
        require(output.is_file(), "Stage A summary is missing")
        require(load(OUTPUT) == summary, "Stage A summary differs")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
