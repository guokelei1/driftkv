from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from hstu_kvcache.migration.recursive_d1 import (
    RECURSIVE_ACTION_PLAN_PROTOCOL,
    RECURSIVE_D1_PROTOCOL,
)
from hstu_kvcache.migration.xp_exact_baseline import file_sha256

COMPARISONS = {
    "all_reuse_from_theta1": ("reuse_exact_baselines", "post", "pre"),
    "all_exact_every_edge": ("reuse_exact_baselines", "exact", "pre"),
    "edge_local_exact_source_rank16_oracle": (
        "incumbent_rank16_recursive",
        "oracle",
        "source",
    ),
    "incumbent_rank16_true_recursive": (
        "incumbent_rank16_recursive",
        "post",
        "pre",
    ),
    "rollout_conditioned_without_contraction_exact0": (
        "rollout_only_exact0",
        "post",
        "pre",
    ),
    "ract_kv_exact0": ("ract_kv_exact0", "post", "pre"),
    "ract_kv_exact10": ("ract_kv_exact10", "post", "pre"),
    "ract_kv_exact20": ("ract_kv_exact20", "post", "pre"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--return-manifest", type=Path, required=True)
    return parser.parse_args()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != value:
            raise FileExistsError(f"recursive D1 summary differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_methods(
    result_root: Path,
    config: dict[str, object],
    config_path: Path,
) -> dict[str, dict[str, object]]:
    methods = {}
    expected_edges = [
        f"theta{value['source_version']}_to_theta{value['target_version']}"
        for value in config["edges"]
    ]
    role_binding = None
    for method in config["methods"]:
        root = result_root / "methods" / str(method)
        summary_path = root / "method_summary.json"
        summary = load_json(summary_path)
        if (
            summary.get("protocol") != RECURSIVE_D1_PROTOCOL
            or summary.get("status") != "complete"
            or summary.get("method") != method
            or summary.get("world_size") != 2
            or summary.get("single_current_serving_model") is not True
            or summary.get("true_recursive_handoff") is not True
            or summary.get("hidden_exact_reset") is not False
            or summary.get("admissible_full_round") is not True
            or summary.get("full_kv_payloads_persisted") != 0
            or [value["edge"] for value in summary.get("edges", [])]
            != expected_edges
            or summary.get("round_config", {}).get("sha256")
            != file_sha256(config_path)
        ):
            raise ValueError(f"recursive D1 method summary differs: {method}")
        if role_binding is None:
            role_binding = summary["role_bindings"]
        elif summary["role_bindings"] != role_binding:
            raise ValueError("recursive D1 method role bindings differ")
        edges = {}
        previous_lineage = None
        previous_cache_state = None
        for descriptor in summary["edges"]:
            path = Path(str(descriptor["path"]))
            if (
                not path.is_file()
                or file_sha256(path) != descriptor["sha256"]
            ):
                raise ValueError(
                    f"recursive D1 edge descriptor differs: {path}"
                )
            edge = load_json(path)
            action_path = Path(str(descriptor["action_plan_path"]))
            if (
                not action_path.is_file()
                or file_sha256(action_path)
                != descriptor["action_plan_sha256"]
            ):
                raise ValueError(
                    f"recursive D1 action descriptor differs: {action_path}"
                )
            action = load_json(action_path)
            if (
                edge.get("protocol") != RECURSIVE_D1_PROTOCOL
                or edge.get("status") != "complete"
                or edge.get("method") != method
                or edge.get("edge") != descriptor["edge"]
                or edge.get("single_current_serving_model") is not True
                or edge.get("full_kv_payloads_persisted") != 0
                or edge.get("recursive_handoff", {}).get(
                    "hidden_exact_reset"
                )
                is not False
                or action.get("protocol")
                != RECURSIVE_ACTION_PLAN_PROTOCOL
                or action.get("method") != method
                or action.get("source_version") != edge.get("source_version")
                or action.get("target_version") != edge.get("target_version")
                or action.get("input_lineage_sha256")
                != edge.get("recursive_handoff", {}).get(
                    "input_lineage_sha256"
                )
                or action.get("output_lineage_sha256")
                != edge.get("recursive_handoff", {}).get(
                    "output_lineage_sha256"
                )
                or action.get("output_cache_state_sha256")
                != edge.get("recursive_handoff", {})
                .get("output_cache_state", {})
                .get("sha256")
                or edge.get("bindings", {})
                .get("action_plan", {})
                .get("sha256")
                != descriptor["action_plan_sha256"]
            ):
                raise ValueError(
                    f"recursive D1 edge result differs: {method} {path}"
                )
            handoff = edge["recursive_handoff"]
            if (
                previous_lineage is not None
                and handoff["input_lineage_sha256"] != previous_lineage
            ):
                raise ValueError(
                    f"recursive D1 lineage is discontinuous: {method}"
                )
            previous_lineage = handoff["output_lineage_sha256"]
            if (
                previous_cache_state is not None
                and handoff.get("input_cache_state", {}).get("sha256")
                != previous_cache_state
            ):
                raise ValueError(
                    f"recursive D1 cache state is discontinuous: {method}"
                )
            previous_cache_state = handoff["output_cache_state"]["sha256"]
            edges[str(edge["edge"])] = edge
        methods[str(method)] = {
            "root": root,
            "summary_path": summary_path,
            "summary": summary,
            "edges": edges,
        }
    expected_roles = {
        "fit": {
            "records": int(config["roles"]["fit_records_global"]),
            "record_ids_sha256": config["roles"][
                "fit_record_ids_sha256"
            ],
        },
        "stability_probe": {
            "records": int(
                config["roles"]["stability_probe_records_global"]
            ),
            "record_ids_sha256": config["roles"][
                "stability_probe_record_ids_sha256"
            ],
        },
        "qualification": {
            "records": int(
                config["roles"]["qualification_records_global"]
            ),
            "record_ids_sha256": config["roles"][
                "qualification_record_ids_sha256"
            ],
        },
    }
    if role_binding != expected_roles:
        raise ValueError("recursive D1 role binding differs from config")
    return methods


def contribution_map(edge: dict[str, object]) -> dict[tuple[int, int], dict[str, object]]:
    rows = edge["quality"]["paired_target_contributions"]
    result = {
        (int(value["record_id"]), int(value["suffix_offset"])): value
        for value in rows
    }
    if len(result) != len(rows):
        raise ValueError("recursive D1 target contributions overlap")
    return result


def cluster_bootstrap_recovery(
    rows: dict[tuple[int, int], dict[str, object]],
    *,
    method_field: str,
    pre_field: str,
    seed: int,
    samples: int = 1000,
) -> dict[str, object] | None:
    by_record: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for (record_id, _), value in rows.items():
        by_record[record_id].append(
            (
                float(value[f"{pre_field}_sampled_cross_entropy"]),
                float(value[f"{method_field}_sampled_cross_entropy"]),
                float(value["exact_sampled_cross_entropy"]),
            )
        )
    record_ids = sorted(by_record)
    if len(record_ids) < 2:
        return None
    record_sums = np.asarray(
        [
            [
                sum(value[index] for value in by_record[record_id])
                for index in range(3)
            ]
            + [len(by_record[record_id])]
            for record_id in record_ids
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    recoveries = []
    for _ in range(samples):
        selected = generator.integers(0, len(record_ids), len(record_ids))
        aggregate = record_sums[selected].sum(axis=0)
        count = aggregate[3]
        pre, method, exact = aggregate[:3] / count
        denominator = pre - exact
        if denominator > 0:
            recoveries.append((pre - method) / denominator)
    if not recoveries:
        return None
    values = np.asarray(recoveries, dtype=np.float64)
    return {
        "replication_unit": "record_cluster_within_one_trained_model_diagnostic",
        "samples_requested": samples,
        "samples_valid": len(values),
        "seed": seed,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def metric_for_field(
    edge: dict[str, object], field: str
) -> tuple[dict[str, object], dict[str, object]]:
    quality = edge["quality"]
    return (
        quality["recommendation"][field],
        quality["cache_fidelity"][field],
    )


def recovery(pre: float, method: float, exact: float) -> float | None:
    denominator = pre - exact
    return None if denominator <= 0 else (pre - method) / denominator


def maximum_certificate_ratio(edge: dict[str, object]) -> float | None:
    certificate = edge.get("stability_certificate")
    if certificate is None:
        return None
    ratios = [
        float(value["recurrence_bound_over_stale_reuse_error"])
        for value in certificate["rows"]
    ]
    return None if not ratios else max(ratios)


def comparison_rows(
    methods: dict[str, dict[str, object]],
    config: dict[str, object],
) -> list[dict[str, object]]:
    baseline_edges = methods["reuse_exact_baselines"]["edges"]
    expected_keys = {
        edge_name: contribution_map(edge)
        for edge_name, edge in baseline_edges.items()
    }
    rows = []
    for comparison, (method, field, pre_field) in COMPARISONS.items():
        method_edges = methods[method]["edges"]
        edge_rows = []
        for edge_ordinal, edge_config in enumerate(config["edges"]):
            edge_name = (
                f"theta{edge_config['source_version']}_to_"
                f"theta{edge_config['target_version']}"
            )
            edge = method_edges[edge_name]
            baseline = baseline_edges[edge_name]
            observed = contribution_map(edge)
            if set(observed) != set(expected_keys[edge_name]):
                raise ValueError(
                    f"recursive D1 paired targets differ: {comparison} {edge_name}"
                )
            baseline_exact = baseline["quality"]["recommendation"]["exact"]
            current_exact = edge["quality"]["recommendation"]["exact"]
            if (
                edge["quality"]["candidate_sha256_per_rank"]
                != baseline["quality"]["candidate_sha256_per_rank"]
                or not math.isclose(
                    float(baseline_exact["sampled_cross_entropy"]),
                    float(current_exact["sampled_cross_entropy"]),
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
            ):
                raise ValueError(
                    f"recursive D1 exact endpoints differ: {comparison} {edge_name}"
                )
            recommendation, fidelity = metric_for_field(edge, field)
            pre = edge["quality"]["recommendation"][pre_field]
            exact = edge["quality"]["recommendation"]["exact"]
            edge_recovery = recovery(
                float(pre["sampled_cross_entropy"]),
                float(recommendation["sampled_cross_entropy"]),
                float(exact["sampled_cross_entropy"]),
            )
            baseline_reuse = baseline["quality"]["recommendation"]["post"]
            cumulative_recovery = recovery(
                float(baseline_reuse["sampled_cross_entropy"]),
                float(recommendation["sampled_cross_entropy"]),
                float(baseline_exact["sampled_cross_entropy"]),
            )
            pre_fidelity = float(
                edge["quality"]["cache_fidelity"][pre_field][
                    "relative_error_mean"
                ]
            )
            mean_fidelity_recovery = (
                None
                if pre_fidelity <= 0
                else 1.0
                - float(fidelity["relative_error_mean"]) / pre_fidelity
            )
            oracle_recovery = edge["quality"]["oracle_reset_ce_recovery"]
            oracle_gap = (
                None
                if field != "post"
                or edge_recovery is None
                or oracle_recovery is None
                else 100.0 * (float(oracle_recovery) - edge_recovery)
            )
            work = edge["logical_work"]["qualification"]
            if comparison == "all_exact_every_edge":
                logical_fraction = 1.0
                logical_budget_admitted = True
                fallback_exact_records = 0
            else:
                logical_fraction = work[
                    "total_d1_exact_valid_token_fraction"
                ]
                logical_budget_admitted = work["budget_admitted"]
                fallback_exact_records = work["fallback_exact_records"]
            edge_rows.append(
                {
                    "edge": edge_name,
                    "edge_ordinal": edge_ordinal,
                    "positive_targets": recommendation["positive_targets"],
                    "sampled_cross_entropy": recommendation[
                        "sampled_cross_entropy"
                    ],
                    "edge_ce_recovery": edge_recovery,
                    "cumulative_ce_recovery_from_initial_reuse": (
                        cumulative_recovery
                    ),
                    "oracle_reset_gap_percentage_points": oracle_gap,
                    "mean_kv_fidelity_recovery": mean_fidelity_recovery,
                    "logical_exact_valid_token_fraction": logical_fraction,
                    "logical_budget_admitted": logical_budget_admitted,
                    "fallback_exact_records": fallback_exact_records,
                    "stability_certificate_maximum_ratio": (
                        maximum_certificate_ratio(edge)
                    ),
                    "stability_certificate_target_pass": (
                        None
                        if edge.get("stability_certificate") is None
                        else edge["stability_certificate"]["target_pass"]
                    ),
                    "stability_certificate_hard_failure": (
                        None
                        if edge.get("stability_certificate") is None
                        else edge["stability_certificate"]["hard_failure"]
                    ),
                    "record_cluster_ce_recovery_95": (
                        cluster_bootstrap_recovery(
                            observed,
                            method_field=field,
                            pre_field=pre_field,
                            seed=2026080500 + edge_ordinal,
                        )
                    ),
                }
            )
        rows.append(
            {
                "comparison": comparison,
                "execution_method": method,
                "quality_field": field,
                "quality_pre_field": pre_field,
                "edges": edge_rows,
            }
        )
    return rows


def policy_gate(
    row: dict[str, object], gates: dict[str, object]
) -> dict[str, object]:
    edges = row["edges"]
    edge_recoveries = [value["edge_ce_recovery"] for value in edges]
    fidelity_recoveries = [
        value["mean_kv_fidelity_recovery"] for value in edges
    ]
    oracle_gaps = [
        value["oracle_reset_gap_percentage_points"] for value in edges
    ]
    target_checks = {
        "every_edge_ce_recovery": all(
            value is not None
            and float(value) >= float(gates["ce_recovery_target"])
            for value in edge_recoveries
        ),
        "final_cumulative_ce_recovery": (
            edges[-1]["cumulative_ce_recovery_from_initial_reuse"]
            is not None
            and float(
                edges[-1]["cumulative_ce_recovery_from_initial_reuse"]
            )
            >= float(gates["final_cumulative_ce_recovery_target"])
        ),
        "every_edge_oracle_gap": all(
            value is not None
            and abs(float(value))
            <= float(gates["oracle_reset_gap_target_percentage_points"])
            for value in oracle_gaps
        ),
        "every_edge_kv_fidelity": all(
            value is not None
            and float(value) >= float(gates["kv_fidelity_recovery_target"])
            for value in fidelity_recoveries
        ),
        "every_edge_stability_certificate": all(
            value["stability_certificate_target_pass"] is True
            for value in edges
        ),
        "every_edge_logical_budget": all(
            value["logical_budget_admitted"] is True for value in edges
        ),
    }
    hard_checks = {
        "every_edge_ce_recovery": all(
            value is not None
            and float(value) >= float(gates["ce_recovery_hard_floor"])
            for value in edge_recoveries
        ),
        "every_edge_oracle_gap": all(
            value is not None
            and abs(float(value))
            <= float(gates["oracle_reset_gap_hard_limit_percentage_points"])
            for value in oracle_gaps
        ),
        "every_edge_kv_fidelity": all(
            value is not None
            and float(value)
            >= float(gates["kv_fidelity_recovery_hard_floor"])
            for value in fidelity_recoveries
        ),
        "no_certificate_hard_failure": all(
            value["stability_certificate_hard_failure"] is False
            for value in edges
        ),
        "every_edge_logical_budget": all(
            value["logical_budget_admitted"] is True for value in edges
        ),
    }
    return {
        "target_checks": target_checks,
        "target_pass": all(target_checks.values()),
        "hard_checks": hard_checks,
        "hard_pass": all(hard_checks.values()),
    }


def build_summary(
    result_root: Path,
    config_path: Path,
) -> tuple[dict[str, object], str, dict[str, object]]:
    config = load_json(config_path)
    methods = load_methods(result_root, config, config_path)
    rows = comparison_rows(methods, config)
    by_name = {value["comparison"]: value for value in rows}
    gates = {
        name: policy_gate(by_name[name], config["gates"])
        for name in ("ract_kv_exact10", "ract_kv_exact20")
    }
    if gates["ract_kv_exact10"]["target_pass"]:
        selected = "ract_kv_exact10"
    elif gates["ract_kv_exact20"]["target_pass"]:
        selected = "ract_kv_exact20"
    else:
        selected = None
    summary = {
        "schema": "evokv_qk_recursive_d1_round_a_summary_development_v0",
        "protocol": RECURSIVE_D1_PROTOCOL,
        "scientific_result": False,
        "formal_result": False,
        "status": (
            "complete_selected_policy"
            if selected is not None
            else "complete_no_admitted_policy"
        ),
        "single_current_serving_model": True,
        "world_size": 2,
        "config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
        },
        "comparisons": rows,
        "selection": {
            "rule": "prefer_exact10_if_all_target_gates_pass_else_exact20",
            "selected_policy": selected,
            "gates": gates,
            "qb_round_may_start": selected is not None,
        },
        "physical_gpu_speedup_claimed": False,
        "full_kv_payloads_persisted": 0,
        "next_boundary": (
            "interpret_qk_then_build_locked_qb_round"
            if selected is not None
            else "interpret_negative_qk_mechanism_result"
        ),
    }
    header = [
        "comparison",
        "edge",
        "edge_ce_recovery",
        "cumulative_ce_recovery",
        "oracle_gap_pp",
        "mean_kv_fidelity_recovery",
        "logical_exact_valid_token_fraction",
        "certificate_maximum_ratio",
    ]
    lines = ["\t".join(header)]
    for comparison in rows:
        for edge in comparison["edges"]:
            lines.append(
                "\t".join(
                    str(value)
                    for value in (
                        comparison["comparison"],
                        edge["edge"],
                        edge["edge_ce_recovery"],
                        edge[
                            "cumulative_ce_recovery_from_initial_reuse"
                        ],
                        edge["oracle_reset_gap_percentage_points"],
                        edge["mean_kv_fidelity_recovery"],
                        edge["logical_exact_valid_token_fraction"],
                        edge["stability_certificate_maximum_ratio"],
                    )
                )
            )
    compact_artifacts = []
    for method in config["methods"]:
        method_root = result_root / "methods" / str(method)
        for path in sorted(method_root.rglob("*.json")):
            compact_artifacts.append(
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    return_manifest = {
        "schema": "evokv_qk_recursive_d1_round_a_return_manifest_v0",
        "status": "complete",
        "return_first": [
            str(result_root / "round_summary.json"),
            str(result_root / "round_summary.tsv"),
            str(result_root / "return_manifest.json"),
        ],
        "compact_method_artifacts": compact_artifacts,
        "program_payloads_excluded_from_return": True,
        "full_kv_payloads_persisted": 0,
    }
    return summary, "\n".join(lines) + "\n", return_manifest


def main() -> None:
    args = parse_args()
    summary, table, return_manifest = build_summary(
        args.result_root, args.config
    )
    atomic_text(args.output, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_text(args.tsv, table)
    atomic_text(
        args.return_manifest,
        json.dumps(return_manifest, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_policy": summary["selection"]["selected_policy"],
                "status": summary["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
