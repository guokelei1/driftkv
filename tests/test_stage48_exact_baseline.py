import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = (
    ROOT
    / "configs"
    / "cohortkv_single_config_v1"
    / "stage4_8_exact_baseline.json"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted(rows: list[dict], metric: str) -> float:
    records = sum(row["records"] for row in rows)
    return sum(row["records"] * row[metric] for row in rows) / records


def compiler_pair(pair: dict) -> dict:
    program = pair["direct_program"]
    provenance = pair["load_validation"]["provenance"]
    return {
        "source_version": pair["source_version"],
        "target_version": pair["target_version"],
        "history_target_date": pair["history_target_date"],
        "history_view_sha256": pair["history_view_sha256"],
        "direct_program_path": program["path"],
        "direct_program_sha256": program["sha256"],
        "direct_program_bytes": program["bytes"],
        "source_checkpoint_sha256": provenance[
            "source_checkpoint_sha256"
        ],
        "target_checkpoint_sha256": provenance[
            "target_checkpoint_sha256"
        ],
        "labels_used": provenance["labels_used"],
        "future_history_used": provenance["future_history_used"],
    }


def test_stage48_exact_baseline_schema() -> None:
    baseline = load_json(BASELINE_PATH)
    assert set(baseline) == {
        "protocol",
        "status",
        "purpose",
        "frozen_date",
        "source_artifacts",
        "configuration",
        "measurement_contract",
        "input_provenance",
        "edge_exact_gpu_denominators",
        "cumulative_exact_gpu_denominator_ms",
        "endpoint_exact_task",
        "record_weighted_exact_task",
        "incumbent_stage4_7",
    }
    assert (
        baseline["protocol"]
        == "cohortkv_single_config_stage4_8_external_exact_baseline_v1"
    )
    assert baseline["status"] == "complete"
    assert set(baseline["source_artifacts"]) == {
        "stage4_7_chain",
        "stage4_7_compiler",
        "stage4_7_summary",
    }
    for artifact in baseline["source_artifacts"].values():
        assert len(artifact["sha256"]) == 64
        assert all(character in "0123456789abcdef" for character in artifact["sha256"])
    configuration = baseline["configuration"]
    assert configuration["dataset"] == "KuaiRand-1K"
    assert configuration["split"] == "4+12"
    assert configuration["training_seed"] == 0
    assert configuration["batch_size"] == 4
    assert configuration["records"] == 682
    provenance = baseline["input_provenance"]
    checkpoints = provenance["checkpoints"]
    windows = provenance["windows"]
    pairs = provenance["compiler_pairs"]
    assert [value["version"] for value in checkpoints] == [
        f"theta{version}" for version in range(12)
    ]
    assert [value["version"] for value in windows] == list(range(12))
    assert len(pairs) == 11
    assert provenance["manifest"]["records"] == 682
    assert provenance["manifest"]["target_dates"] == [
        value["target_date"] for value in windows
    ]
    checkpoint_by_version = {
        value["version"]: value for value in checkpoints
    }
    for source_version, pair in enumerate(pairs):
        source = f"theta{source_version}"
        target = f"theta{source_version + 1}"
        assert pair["source_version"] == source
        assert pair["target_version"] == target
        assert pair["history_target_date"] == windows[
            source_version + 1
        ]["target_date"]
        assert pair["source_checkpoint_sha256"] == checkpoint_by_version[
            source
        ]["sha256"]
        assert pair["target_checkpoint_sha256"] == checkpoint_by_version[
            target
        ]["sha256"]
        assert pair["labels_used"] is False
        assert pair["future_history_used"] is False
    edges = baseline["edge_exact_gpu_denominators"]
    assert len(edges) == 11
    for source_version, edge in enumerate(edges):
        assert edge["source_version"] == source_version
        assert edge["target_version"] == source_version + 1
        assert edge["source_model"] == f"theta{source_version}"
        assert edge["target_model"] == f"theta{source_version + 1}"
        assert edge["target_date"] == windows[source_version + 1][
            "target_date"
        ]
        assert 0 < edge["exact_reference_records"] <= 682
        assert math.isfinite(edge["all_exact_reference_ms"])
        assert edge["all_exact_reference_ms"] > 0
    assert math.isclose(
        sum(value["all_exact_reference_ms"] for value in edges),
        baseline["cumulative_exact_gpu_denominator_ms"],
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    endpoints = baseline["endpoint_exact_task"]
    assert len(endpoints) == 12
    for version, endpoint in enumerate(endpoints):
        assert endpoint["version"] == version
        assert endpoint["target_date"] == windows[version]["target_date"]
        assert endpoint["records"] > 0
        for metric in ("catalog_auc", "ndcg_at_100", "hit_at_100"):
            assert math.isfinite(endpoint[metric])
            assert 0 <= endpoint[metric] <= 1
    weighted_values = baseline["record_weighted_exact_task"]
    for name, rows in (
        ("all_twelve_endpoints", endpoints),
        ("eleven_update_endpoints", endpoints[1:]),
    ):
        expected = weighted_values[name]
        assert expected["records"] == sum(row["records"] for row in rows)
        for metric in ("catalog_auc", "ndcg_at_100", "hit_at_100"):
            assert math.isclose(
                expected[metric],
                weighted(rows, metric),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
    assert all(
        0 < value < 1
        for value in baseline["incumbent_stage4_7"].values()
    )


def test_stage48_exact_baseline_matches_frozen_sources() -> None:
    baseline = load_json(BASELINE_PATH)
    artifacts = baseline["source_artifacts"]
    provenance = baseline["input_provenance"]
    chain_path = ROOT / artifacts["stage4_7_chain"]["path"]
    if chain_path.exists():
        chain = load_json(chain_path)
        assert sha256(chain_path) == artifacts["stage4_7_chain"]["sha256"]
        assert chain["protocol"] == artifacts["stage4_7_chain"]["protocol"]
        assert (
            chain["experiment_protocol"]
            == artifacts["stage4_7_chain"]["experiment_protocol"]
        )
        assert chain["status"] == "complete"
        assert (
            chain["repository_commit"]
            == artifacts["stage4_7_chain"]["repository_commit"]
        )
        assert all(
            passed
            for family in chain["checks"].values()
            for passed in family.values()
        )
        assert chain["configuration"]["model"] == baseline[
            "configuration"
        ]["model"]
        assert chain["configuration"]["batch_size"] == baseline[
            "configuration"
        ]["batch_size"]
        assert chain["configuration"]["records"] == baseline[
            "configuration"
        ]["records"]
        assert chain["inputs"]["prepared_data"] == provenance[
            "prepared_data"
        ]
        assert chain["inputs"]["training_result"] == provenance[
            "training_result"
        ]
        assert chain["inputs"]["checkpoints"] == provenance["checkpoints"]
        assert chain["inputs"]["windows"] == provenance["windows"]
        manifest = chain["inputs"]["manifest"]
        expected_manifest = provenance["manifest"]
        assert manifest["protocol"] == expected_manifest["protocol"]
        assert manifest["content_sha256"] == expected_manifest[
            "content_sha256"
        ]
        assert len(manifest["records"]) == expected_manifest["records"]
        assert manifest["timeline"]["base_dates"] == expected_manifest[
            "base_dates"
        ]
        assert manifest["timeline"]["target_dates"] == expected_manifest[
            "target_dates"
        ]
        assert manifest["timeline"]["versions"] == expected_manifest[
            "versions"
        ]
        assert manifest["timeline"]["rule"] == expected_manifest[
            "timeline_rule"
        ]
        expected_edges = []
        for step in chain["steps"]:
            actions = step["actions"]
            expected_edges.append(
                {
                    "source_version": step["source_version"],
                    "target_version": step["target_version"],
                    "source_model": f"theta{step['source_version']}",
                    "target_model": f"theta{step['target_version']}",
                    "target_date": step["prediction_target_date"],
                    "exact_reference_records": (
                        actions["migrate"]
                        + actions["scheduled_selector_exact"]
                        + actions["natural_no_reuse_target_exact"]
                    ),
                    "all_exact_reference_ms": step["cost"][
                        "all_exact_reference_ms"
                    ],
                }
            )
        assert expected_edges == baseline["edge_exact_gpu_denominators"]
        assert (
            chain["cumulative_gpu_cost"]["all_exact_reference_ms"]
            == baseline["cumulative_exact_gpu_denominator_ms"]
        )
        expected_endpoints = []
        for endpoint in chain["endpoints"]:
            task = endpoint["task_metrics"]
            exact = task["all_exact"]
            expected_endpoints.append(
                {
                    "version": endpoint["version"],
                    "target_date": endpoint["target_date"],
                    "records": task["records"],
                    "catalog_auc": exact["catalog_auc"],
                    "ndcg_at_100": exact["ndcg@100"],
                    "hit_at_100": exact["hit@100"],
                }
            )
        assert expected_endpoints == baseline["endpoint_exact_task"]
    compiler_path = ROOT / artifacts["stage4_7_compiler"]["path"]
    if compiler_path.exists():
        compiler = load_json(compiler_path)
        assert sha256(compiler_path) == artifacts[
            "stage4_7_compiler"
        ]["sha256"]
        assert compiler["protocol"] == artifacts[
            "stage4_7_compiler"
        ]["protocol"]
        assert compiler["experiment_protocol"] == artifacts[
            "stage4_7_compiler"
        ]["experiment_protocol"]
        assert compiler["status"] == "complete"
        assert compiler["repository_commit"] == artifacts[
            "stage4_7_compiler"
        ]["repository_commit"]
        assert compiler["inputs"]["prepared_data"] == provenance[
            "prepared_data"
        ]
        assert compiler["inputs"]["training_result"] == provenance[
            "training_result"
        ]
        assert compiler["inputs"]["checkpoints"] == provenance[
            "checkpoints"
        ]
        assert compiler["manifest"]["content_sha256"] == provenance[
            "manifest"
        ]["content_sha256"]
        assert [
            compiler_pair(pair) for pair in compiler["pairs"]
        ] == provenance["compiler_pairs"]
    summary_path = ROOT / artifacts["stage4_7_summary"]["path"]
    if summary_path.exists():
        summary = load_json(summary_path)
        assert sha256(summary_path) == artifacts[
            "stage4_7_summary"
        ]["sha256"]
        assert summary["protocol"] == artifacts[
            "stage4_7_summary"
        ]["protocol"]
        assert summary["status"] == artifacts[
            "stage4_7_summary"
        ]["status"]
        assert summary["result_artifact"]["sha256"] == artifacts[
            "stage4_7_chain"
        ]["sha256"]
        assert summary["implementation_snapshot"]["compiler_result"][
            "sha256"
        ] == artifacts["stage4_7_compiler"]["sha256"]
        incumbent = baseline["incumbent_stage4_7"]
        assert summary["gpu_cost"]["cumulative_update_only_ratio"] == (
            incumbent["primary_update_only_ratio"]
        )
        assert summary["gpu_cost"]["symmetric_lifecycle_ratio"] == (
            incumbent["symmetric_lifecycle_ratio"]
        )
        assert summary["gpu_cost"][
            "common_inclusive_lifecycle_ratio"
        ] == incumbent["common_inclusive_lifecycle_ratio"]
