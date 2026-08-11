from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import verify_evokv_kuairand_large_baseline as baseline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs/evokv_d1/development/kuairand_recursive_chain_design_v0.json"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def repo_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def require_hash(record: dict[str, Any]) -> Path:
    path = repo_path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"bound artifact hash differs: {record['path']}")
    return path


def validate_contract(document: dict[str, Any]) -> None:
    if (
        document.get("schema") != "evokv_kuairand_recursive_d1_design_contract_v0"
        or document.get("status") != "ready_for_mechanism_design"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
    ):
        raise ValueError("recursive D1 design contract differs")
    execution = document["execution"]
    if (
        execution.get("allowed_devices") != [0, 1]
        or execution.get("world_size") != 2
        or execution.get("maximum_concurrent_jobs") != 1
        or execution.get("retain_full_kv_payload_between_stages") is not False
    ):
        raise ValueError("recursive D1 execution contract differs")
    sequence = document["model_sequence"]
    if (
        sequence.get("large_model_versions") != list(range(1, 9))
        or sequence.get("exact_reset_policy") != "forbidden_after_initial_theta0"
        or sequence.get("initial_cache", {}).get("model_version") != 0
        or sequence.get("initial_cache", {}).get("state") != "exact"
    ):
        raise ValueError("recursive D1 model sequence differs")
    edges = [sequence["bootstrap_edge"], *sequence["primary_edges"]]
    if len(edges) != 8:
        raise ValueError("recursive D1 edge count differs")
    for index, edge in enumerate(edges):
        if edge["source_version"] != index or edge["target_version"] != index + 1:
            raise ValueError("recursive D1 edge order differs")
    comparisons = document["comparisons"]
    if (
        comparisons.get("required_endpoints")
        != ["full_recompute", "recursive_reuse", "recursive_method"]
        or comparisons.get("primary_metric") != "ndcg_at_5"
        or comparisons.get("result_shape")
        != "one ordered edge table and one cumulative summary, never another selected 8x8 method matrix"
    ):
        raise ValueError("recursive D1 comparison contract differs")


def validate_bindings(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    registry_path = require_hash(document["bindings"]["baseline_registry"])
    matrix_path = require_hash(document["bindings"]["reference_matrix"])
    registry = load_json(registry_path)
    matrix = load_json(matrix_path)
    if registry["reference_matrix"]["path"] != str(matrix_path.relative_to(ROOT)):
        raise ValueError("registry matrix path differs from D1 binding")
    if registry["reference_matrix"]["sha256"] != file_sha256(matrix_path):
        raise ValueError("registry matrix hash differs from D1 binding")
    if matrix.get("first_version") != 1 or matrix.get("final_version") != 8:
        raise ValueError("D1 reference matrix version range differs")
    return registry, matrix


def validate_geometry(document: dict[str, Any], registry: dict[str, Any]) -> None:
    base_config = load_json(repo_path(registry["bootstrap"]["base_config"]))
    large_config = load_json(repo_path(registry["large_chain"]["config"]))
    expected = document["model_sequence"]["cache_geometry"]
    base_model = base_config["model"]
    large_model = large_config["model"]
    observed_base = {
        "layers": base_model["num_layers"],
        "hidden_size": base_model["hidden_size"],
        "heads": base_model["num_heads"],
        "maximum_valid_tokens": base_model["max_seq_len"],
    }
    observed_large = {
        "layers": large_model["num_layers"],
        "hidden_size": large_model["hidden_size"],
        "heads": large_model["num_heads"],
        "maximum_valid_tokens": 512,
    }
    if observed_base != expected or observed_large != expected:
        raise ValueError("theta0 and large-chain cache geometry differs")
    if (
        registry["large_chain"]["global_model_parameter_bytes"]
        != document["model_sequence"]["large_model_parameter_bytes"]
    ):
        raise ValueError("large-model parameter bytes differ")
    expected_transitions = [
        document["model_sequence"]["bootstrap_edge"],
        *document["model_sequence"]["primary_edges"],
    ]
    for expected_edge, observed_edge in zip(
        expected_transitions, large_config["transitions"], strict=True
    ):
        for field in ("source_version", "target_version", "update_date", "evaluation_date"):
            if expected_edge[field] != observed_edge[field]:
                raise ValueError(f"transition field differs: {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--full-payload-hashes", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    document = load_json(config_path)
    validate_contract(document)
    registry, matrix = validate_bindings(document)
    validate_geometry(document, registry)
    baseline_static = baseline.verify_static(registry)
    baseline_current = baseline.verify_current(registry, args.full_payload_hashes)
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < 64 * (1 << 30):
        raise RuntimeError("D1 workspace has less than 64 GiB free")
    summary = {
        "schema": "evokv_kuairand_recursive_d1_preflight_v0",
        "status": "ready",
        "scientific_result": False,
        "formal_result": False,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": file_sha256(config_path),
        "baseline_registry": document["bindings"]["baseline_registry"],
        "reference_matrix": {
            **document["bindings"]["reference_matrix"],
            "ndcg_at_5_summary": matrix["summaries"]["ndcg_at_5"],
        },
        "rollout": {
            "initial_version": 0,
            "final_version": 8,
            "edge_count": 8,
            "primary_edge_count": 7,
            "exact_resets_after_initialization": 0,
            "result_shape": document["comparisons"]["result_shape"],
        },
        "baseline": {
            "static_artifacts": len(baseline_static),
            "large_checkpoints": len(baseline_current["large_checkpoints"]),
            "large_total_checkpoint_bytes": baseline_current[
                "large_total_checkpoint_bytes"
            ],
            "matrix": baseline_current["matrix"],
        },
        "resources": {
            "allowed_devices": document["execution"]["allowed_devices"],
            "world_size": document["execution"]["world_size"],
            "free_disk_bytes": free_bytes,
        },
        "next_missing_artifacts": [
            "frozen disjoint fit/development record manifests",
            "method implementation and candidate configuration",
            "recursive single-chain runner",
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
