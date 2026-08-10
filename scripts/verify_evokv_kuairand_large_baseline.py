from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = (
    ROOT
    / "configs/evokv_root_cause/kuairand_large_baseline_registry_20260811_v0.json"
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


def verify_static(registry: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for artifact in registry["static_artifacts"]:
        path = repo_path(artifact["path"])
        if not path.is_file():
            raise FileNotFoundError(f"static artifact is absent: {artifact['path']}")
        observed = file_sha256(path)
        if observed != artifact["sha256"]:
            raise ValueError(f"static artifact hash differs: {artifact['path']}")
        if "bytes" in artifact and path.stat().st_size != int(artifact["bytes"]):
            raise ValueError(f"static artifact size differs: {artifact['path']}")
        records.append(
            {
                "path": artifact["path"],
                "bytes": path.stat().st_size,
                "sha256": observed,
            }
        )
    return records


def artifact_path(directory: Path, record: dict[str, Any]) -> Path:
    path = (directory / str(record["path"])).resolve()
    path.relative_to(directory.resolve())
    return path


def verify_manifest(
    root: Path,
    version: int,
    world_size: int,
    parameter_bytes: int,
    storage: str,
    full_payload_hashes: bool,
    reference_sha256: str | None,
) -> dict[str, Any]:
    directory = root / f"theta_{version}"
    path = directory / "manifest.json"
    document = load_json(path)
    if (
        document.get("schema") != "evokv_kuairand_projected_checkpoint_v0"
        or document.get("status") != "complete"
        or document.get("version") != version
        or document.get("world_size") != world_size
        or document.get("embedding_storage", "full") != storage
        or document.get("geometry", {}).get("global_model_parameter_bytes")
        != parameter_bytes
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
    ):
        raise ValueError(f"checkpoint manifest semantics differ: {path}")
    embeddings = document.get("embedding_shards", [])
    trackers = document.get("tracker_shards", [])
    if len(embeddings) != world_size or len(trackers) != world_size:
        raise ValueError(f"checkpoint shard count differs: {path}")
    artifacts = [document["dense"], document["projection"], *embeddings, *trackers]
    observed_bytes = 0
    for artifact in artifacts:
        payload = artifact_path(directory, artifact)
        if not payload.is_file():
            raise FileNotFoundError(f"checkpoint payload is absent: {payload}")
        size = payload.stat().st_size
        if size != int(artifact["bytes"]):
            raise ValueError(f"checkpoint payload size differs: {payload}")
        if full_payload_hashes and file_sha256(payload) != artifact["sha256"]:
            raise ValueError(f"checkpoint payload hash differs: {payload}")
        observed_bytes += size
    if observed_bytes != int(document["checkpoint_bytes"]):
        raise ValueError(f"checkpoint byte total differs: {path}")
    manifest_sha256 = file_sha256(path)
    return {
        "version": version,
        "checkpoint_bytes": observed_bytes,
        "manifest_sha256": manifest_sha256,
        "capacity_lift_source_manifest_sha256": document.get("provenance", {})
        .get("capacity_lift", {})
        .get("source_manifest", {})
        .get("sha256"),
        "reference_manifest_match": (
            None if reference_sha256 is None else manifest_sha256 == reference_sha256
        ),
        "payload_hashes_verified": full_payload_hashes,
    }


def verify_accepted(root: Path, versions: int) -> None:
    for version in range(1, versions + 1):
        path = root / "edges" / f"theta_{version}" / "accepted.json"
        document = load_json(path)
        if document.get("status") != "accepted":
            raise ValueError(f"accepted edge differs: {path}")


def compare_nested(observed: Any, expected: Any, path: str = "matrix") -> None:
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"{path} shape differs")
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            compare_nested(left, right, f"{path}[{index}]")
        return
    if expected is None:
        if observed is not None:
            raise ValueError(f"{path} null pattern differs")
        return
    if abs(float(observed) - float(expected)) > 1e-6:
        raise ValueError(f"{path} value differs: {observed} != {expected}")


def verify_current(
    registry: dict[str, Any], full_payload_hashes: bool
) -> dict[str, Any]:
    bootstrap = registry["bootstrap"]
    theta0 = repo_path(bootstrap["theta0"]["path"])
    if theta0.stat().st_size != int(bootstrap["theta0"]["bytes"]):
        raise ValueError("theta0 byte count differs")
    if file_sha256(theta0) != bootstrap["theta0"]["sha256"]:
        raise ValueError("theta0 hash differs")

    medium = registry["medium_chain"]
    medium_root = repo_path(medium["checkpoint_root"])
    medium_results = repo_path(medium["result_root"])
    medium_records = []
    for version in range(1, int(medium["versions"]) + 1):
        storage = "full" if version <= int(medium["full_prefix_versions"]) else "sparse_delta"
        medium_records.append(
            verify_manifest(
                medium_root,
                version,
                int(medium["world_size"]),
                int(medium["global_model_parameter_bytes"]),
                storage,
                full_payload_hashes,
                medium["reference_manifest_sha256"].get(str(version)),
            )
        )
    verify_accepted(medium_results, int(medium["versions"]))
    medium_result_path = medium_results / "result.json"
    medium_result = load_json(medium_result_path)
    if (
        medium_result.get("status") != "complete"
        or len(medium_result.get("targets", [])) != int(medium["versions"])
        or medium_result.get("geometry", {}).get("global_model_parameter_bytes")
        != int(medium["global_model_parameter_bytes"])
    ):
        raise ValueError("medium-chain result differs")

    large = registry["large_chain"]
    large_root = repo_path(large["checkpoint_root"])
    large_results = repo_path(large["result_root"])
    large_records = []
    for version in range(1, int(large["versions"]) + 1):
        large_records.append(
            verify_manifest(
                large_root,
                version,
                int(large["world_size"]),
                int(large["global_model_parameter_bytes"]),
                "full",
                full_payload_hashes,
                large["reference_manifest_sha256"].get(str(version)),
            )
        )
    verify_accepted(large_results, int(large["versions"]))
    for medium_record, large_record in zip(
        medium_records[: int(large["versions"])], large_records, strict=True
    ):
        if (
            large_record["capacity_lift_source_manifest_sha256"]
            != medium_record["manifest_sha256"]
        ):
            raise ValueError("large checkpoint source-manifest binding differs")
    lift_path = repo_path(large["lift_result"])
    lift = load_json(lift_path)
    if (
        lift.get("schema") != "evokv_kuairand_capacity_lift_v0"
        or lift.get("status") != "complete"
        or lift.get("function_preserving") is not True
        or lift.get("mapping") != "strided_hash_v0"
        or lift.get("final_version") != int(large["versions"])
        or len(lift.get("versions", [])) != int(large["versions"])
        or lift.get("geometry", {}).get("global_model_parameter_bytes")
        != int(large["global_model_parameter_bytes"])
        or any(
            float(value.get("maximum_active_embedding_absolute_error", -1)) != 0.0
            for value in lift.get("versions", [])
        )
    ):
        raise ValueError("capacity-lift result differs")
    lineage_path = repo_path(large["lineage_result"])
    lineage = load_json(lineage_path)
    if (
        lineage.get("status") != "complete"
        or len(lineage.get("targets", [])) != int(large["versions"])
        or lineage.get("geometry", {}).get("global_model_parameter_bytes")
        != int(large["global_model_parameter_bytes"])
        or lineage.get("scientific_result") is not False
        or lineage.get("formal_result") is not False
    ):
        raise ValueError("large-chain lineage result differs")
    for expected_version, target in enumerate(lineage["targets"], start=1):
        target_path = repo_path(target["path"])
        target_document = load_json(target_path)
        if (
            target.get("target_version") != expected_version
            or target_document.get("target_version") != expected_version
            or file_sha256(target_path) != target["sha256"]
        ):
            raise ValueError("large-chain target lineage binding differs")

    reference = registry["reference_matrix"]
    matrix_path = repo_path(reference["path"])
    matrix = load_json(matrix_path)
    if (
        matrix.get("protocol") != "evokv_kuairand_capacity_lift_matrix_v0"
        or matrix.get("status") != "complete_development_evidence"
        or matrix.get("definition") != reference["definition"]
        or matrix.get("first_version") != 1
        or matrix.get("final_version") != int(large["versions"])
        or matrix.get("scientific_result") is not False
        or matrix.get("formal_result") is not False
    ):
        raise ValueError("reference matrix semantics differ")
    for metric, expected in reference["matrices_relative_percent"].items():
        compare_nested(matrix["matrices_relative_percent"][metric], expected, metric)
    if matrix.get("source", {}).get("sha256") != file_sha256(lineage_path):
        raise ValueError("matrix-to-lineage hash binding differs")
    markdown = repo_path(matrix["markdown"]["path"])
    if file_sha256(markdown) != matrix["markdown"]["sha256"]:
        raise ValueError("matrix Markdown binding differs")

    return {
        "theta0": {
            "path": bootstrap["theta0"]["path"],
            "sha256": bootstrap["theta0"]["sha256"],
        },
        "medium_checkpoints": medium_records,
        "large_checkpoints": large_records,
        "large_total_checkpoint_bytes": sum(
            value["checkpoint_bytes"] for value in large_records
        ),
        "matrix": {
            "path": reference["path"],
            "reference_file_sha256_match": file_sha256(matrix_path)
            == reference["sha256"],
            "ndcg_at_5": matrix["summaries"]["ndcg_at_5"],
        },
        "compact_result_reference_hashes": {
            "medium_result": file_sha256(medium_result_path)
            == medium["reference_result_sha256"],
            "lift_result": file_sha256(lift_path) == large["reference_lift_sha256"],
            "lineage_result": file_sha256(lineage_path)
            == large["reference_lineage_sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--scope", choices=("static", "current", "full"), default="current")
    args = parser.parse_args()
    registry_path = Path(args.registry).resolve()
    registry = load_json(registry_path)
    if (
        registry.get("schema") != "evokv_kuairand_large_baseline_registry_v0"
        or registry.get("status") != "selected_development_baseline"
        or registry.get("scientific_result") is not False
        or registry.get("formal_result") is not False
    ):
        raise ValueError("KuaiRand baseline registry differs")
    output: dict[str, Any] = {
        "status": "valid",
        "scope": args.scope,
        "registry": str(registry_path.relative_to(ROOT)),
        "static_artifacts": verify_static(registry),
    }
    if args.scope != "static":
        output["current_baseline"] = verify_current(registry, args.scope == "full")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
