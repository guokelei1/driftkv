import json
import subprocess
import sys
from pathlib import Path

from hstu_kvcache.migration.recursive_d1 import RECURSIVE_D1_PROTOCOL
from hstu_kvcache.migration.xp_exact_baseline import file_sha256

METHODS = [
    "reuse_exact_baselines",
    "incumbent_rank16_recursive",
    "rollout_only_exact0",
    "ract_kv_exact0",
    "ract_kv_exact10",
    "ract_kv_exact20",
]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _metric(cross_entropy: float) -> dict[str, object]:
    return {
        "positive_targets": 2,
        "sampled_cross_entropy": cross_entropy,
    }


def _edge(
    method: str,
    edge_name: str,
    ordinal: int,
) -> dict[str, object]:
    post = 2.0 if method == "reuse_exact_baselines" else 1.05
    source = 2.0 if method == "reuse_exact_baselines" else 1.8
    certificate = None
    if method in {
        "rollout_only_exact0",
        "ract_kv_exact0",
        "ract_kv_exact10",
        "ract_kv_exact20",
    }:
        certificate = {
            "rows": [{"recurrence_bound_over_stale_reuse_error": 0.05}],
            "target_pass": True,
            "hard_failure": False,
        }
    contributions = []
    for record_id in (10, 11):
        contributions.append(
            {
                "record_id": record_id,
                "suffix_offset": 1,
                "pre_sampled_cross_entropy": 2.0,
                "source_sampled_cross_entropy": source,
                "post_sampled_cross_entropy": post,
                "oracle_sampled_cross_entropy": 1.04,
                "exact_sampled_cross_entropy": 1.0,
            }
        )
    return {
        "protocol": RECURSIVE_D1_PROTOCOL,
        "status": "complete",
        "method": method,
        "edge": edge_name,
        "single_current_serving_model": True,
        "full_kv_payloads_persisted": 0,
        "recursive_handoff": {
            "input_lineage_sha256": f"lineage-{ordinal}",
            "output_lineage_sha256": f"lineage-{ordinal + 1}",
            "input_cache_state": {"sha256": f"cache-{ordinal}"},
            "output_cache_state": {
                "sha256": f"cache-{ordinal + 1}"
            },
            "hidden_exact_reset": False,
        },
        "bindings": {},
        "quality": {
            "recommendation": {
                "pre": _metric(2.0),
                "source": _metric(source),
                "post": _metric(post),
                "oracle": _metric(1.04),
                "exact": _metric(1.0),
            },
            "cache_fidelity": {
                "pre": {"relative_error_mean": 1.0},
                "source": {"relative_error_mean": 0.8},
                "post": {"relative_error_mean": 0.05},
                "oracle": {"relative_error_mean": 0.04},
                "exact": {"relative_error_mean": 0.0},
            },
            "oracle_reset_ce_recovery": 0.96,
            "paired_target_contributions": contributions,
            "candidate_sha256_per_rank": ["rank0", "rank1"],
        },
        "logical_work": {
            "qualification": {
                "total_d1_exact_valid_token_fraction": 0.1,
                "budget_admitted": True,
                "fallback_exact_records": 0,
            }
        },
        "stability_certificate": certificate,
    }


def test_recursive_round_summary_selects_exact10(tmp_path: Path) -> None:
    result_root = tmp_path / "round"
    edges = [
        {"source_version": source, "target_version": source + 1}
        for source in (1, 2, 3)
    ]
    config = {
        "methods": METHODS,
        "edges": edges,
        "gates": {
            "ce_recovery_hard_floor": 0.8,
            "ce_recovery_target": 0.9,
            "final_cumulative_ce_recovery_target": 0.9,
            "kv_fidelity_recovery_hard_floor": 0.8,
            "kv_fidelity_recovery_target": 0.9,
            "oracle_reset_gap_hard_limit_percentage_points": 10.0,
            "oracle_reset_gap_target_percentage_points": 5.0,
        },
        "roles": {
            "fit_records_global": 1,
            "fit_record_ids_sha256": "fit",
            "stability_probe_records_global": 1,
            "stability_probe_record_ids_sha256": "probe",
            "qualification_records_global": 2,
            "qualification_record_ids_sha256": "qualification",
        },
    }
    config_path = tmp_path / "config.json"
    _write(config_path, config)
    for method in METHODS:
        root = result_root / "methods" / method
        descriptors = []
        for ordinal, edge in enumerate(edges):
            edge_name = (
                f"theta{edge['source_version']}_to_"
                f"theta{edge['target_version']}"
            )
            path = root / "edges" / f"{edge_name}.json"
            action_path = root / "action_plans" / f"{edge_name}.json"
            action = {
                "protocol": "evokv_qk_recursive_d1_action_plan_development_v0",
                "method": method,
                "source_version": edge["source_version"],
                "target_version": edge["target_version"],
                "input_lineage_sha256": f"lineage-{ordinal}",
                "output_lineage_sha256": f"lineage-{ordinal + 1}",
                "output_cache_state_sha256": f"cache-{ordinal + 1}",
            }
            _write(action_path, action)
            edge_value = _edge(method, edge_name, ordinal)
            edge_value["source_version"] = edge["source_version"]
            edge_value["target_version"] = edge["target_version"]
            edge_value["bindings"] = {
                "action_plan": {"sha256": file_sha256(action_path)}
            }
            _write(path, edge_value)
            descriptors.append(
                {
                    "edge": edge_name,
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "action_plan_path": str(action_path),
                    "action_plan_sha256": file_sha256(action_path),
                }
            )
        _write(
            root / "method_summary.json",
            {
                "protocol": RECURSIVE_D1_PROTOCOL,
                "status": "complete",
                "method": method,
                "world_size": 2,
                "single_current_serving_model": True,
                "true_recursive_handoff": True,
                "hidden_exact_reset": False,
                "admissible_full_round": True,
                "full_kv_payloads_persisted": 0,
                "round_config": {
                    "path": str(config_path),
                    "sha256": file_sha256(config_path),
                },
                "role_bindings": {
                    "fit": {
                        "records": 1,
                        "record_ids_sha256": "fit",
                    },
                    "stability_probe": {
                        "records": 1,
                        "record_ids_sha256": "probe",
                    },
                    "qualification": {
                        "records": 2,
                        "record_ids_sha256": "qualification",
                    },
                },
                "edges": descriptors,
            },
        )
    summary_path = result_root / "round_summary.json"
    table_path = result_root / "round_summary.tsv"
    manifest_path = result_root / "return_manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts/summarize_evokv_qk_recursive_d1.py"
            ),
            "--result-root",
            str(result_root),
            "--config",
            str(config_path),
            "--output",
            str(summary_path),
            "--tsv",
            str(table_path),
            "--return-manifest",
            str(manifest_path),
        ],
        check=True,
    )
    summary = json.loads(summary_path.read_text())
    table = table_path.read_text()
    manifest = json.loads(manifest_path.read_text())
    assert summary["status"] == "complete_selected_policy"
    assert summary["selection"]["selected_policy"] == "ract_kv_exact10"
    assert len(summary["comparisons"]) == 8
    comparisons = {
        value["comparison"]: value for value in summary["comparisons"]
    }
    assert comparisons["all_exact_every_edge"]["edges"][0][
        "logical_exact_valid_token_fraction"
    ] == 1.0
    assert abs(
        comparisons["edge_local_exact_source_rank16_oracle"]["edges"][0][
            "edge_ce_recovery"
        ]
        - 0.95
    ) < 1e-12
    assert "ract_kv_exact20" in table
    assert manifest["full_kv_payloads_persisted"] == 0
