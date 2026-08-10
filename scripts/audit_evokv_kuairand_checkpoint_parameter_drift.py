from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from hstu_kvcache.streaming.kuairand_query_transition import _atomic_json, file_sha256

PROTOCOL = "evokv_kuairand_checkpoint_parameter_drift_v0"


def _relative_l2(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    keys: list[str],
) -> float:
    numerator = sum(
        torch.sum((target[key].float() - source[key].float()) ** 2).item()
        for key in keys
    )
    denominator = sum(torch.sum(source[key].float() ** 2).item() for key in keys)
    if denominator <= 0:
        raise RuntimeError("KuaiRand parameter drift denominator differs")
    return float((numerator / denominator) ** 0.5)


def _projection_relative_l2(source: torch.Tensor, target: torch.Tensor) -> float:
    source_float = source.float().reshape(-1)
    target_float = target.float().reshape(-1)
    denominator = torch.linalg.vector_norm(source_float)
    if float(denominator.item()) <= 0:
        raise RuntimeError("KuaiRand projection drift denominator differs")
    return float(
        (
            torch.linalg.vector_norm(target_float - source_float) / denominator
        ).item()
    )


def _load_version(root: Path, version: int) -> dict[str, Any]:
    directory = root / f"theta_{version}"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dense_path = directory / manifest["dense"]["path"]
    projection_path = directory / manifest["projection"]["path"]
    dense = torch.load(dense_path, map_location="cpu", weights_only=True)["state_dict"]
    projection = torch.load(
        projection_path, map_location="cpu", weights_only=True
    )["projection_weight"]
    return {
        "manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
        },
        "candidate": manifest["provenance"]["accepted_candidate"]["name"],
        "dense": dense,
        "projection": projection,
    }


def run(checkpoint_root: Path, chain_result: Path, output: Path) -> dict[str, Any]:
    if output.is_file():
        result = json.loads(output.read_text())
        if result.get("protocol") != PROTOCOL or result.get("status") != "complete":
            raise RuntimeError("KuaiRand cached parameter drift audit differs")
        return result
    chain = json.loads(chain_result.read_text())
    versions = int(chain["checkpoint_count"])
    if chain.get("status") != "complete" or versions < 2:
        raise ValueError("KuaiRand parameter drift parent differs")
    states = {
        version: _load_version(checkpoint_root, version)
        for version in range(1, versions + 1)
    }
    keys = list(states[1]["dense"])
    kv_keys = [key for key in keys if "k_proj" in key or "v_proj" in key]
    non_kv_keys = [key for key in keys if key not in kv_keys]

    def comparison(source_version: int, target_version: int) -> dict[str, Any]:
        source = states[source_version]
        target = states[target_version]
        return {
            "source_version": source_version,
            "target_version": target_version,
            "source_candidate": source["candidate"],
            "target_candidate": target["candidate"],
            "dense_non_kv_relative_l2": _relative_l2(
                source["dense"], target["dense"], non_kv_keys
            ),
            "kv_projection_relative_l2": _relative_l2(
                source["dense"], target["dense"], kv_keys
            ),
            "embedding_projection_relative_l2": _projection_relative_l2(
                source["projection"], target["projection"]
            ),
        }

    adjacent = [comparison(version - 1, version) for version in range(2, versions + 1)]
    theta1_to_current = [comparison(1, version) for version in range(2, versions + 1)]
    late_edges = [
        row
        for row in adjacent
        if row["source_version"] >= 5
        and row["kv_projection_relative_l2"] < 0.01
    ]
    result = {
        "protocol": PROTOCOL,
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "chain_result": {
            "path": str(chain_result),
            "sha256": file_sha256(chain_result),
        },
        "checkpoint_root": str(checkpoint_root),
        "versions": versions,
        "manifest_records": {
            f"theta{version}": states[version]["manifest"]
            for version in range(1, versions + 1)
        },
        "groups": {
            "kv_projection_keys": kv_keys,
            "dense_non_kv_keys": non_kv_keys,
        },
        "adjacent": adjacent,
        "theta1_to_current": theta1_to_current,
        "diagnosis": {
            "early_theta1_theta2_kv_jump": adjacent[0][
                "kv_projection_relative_l2"
            ],
            "early_theta1_theta2_embedding_projection_jump": adjacent[0][
                "embedding_projection_relative_l2"
            ],
            "late_near_frozen_kv_edges": [
                [row["source_version"], row["target_version"]] for row in late_edges
            ],
            "theta1_column_is_ordinary_age_effect": False,
            "theta1_column_interpretation": "early_cache_coordinate_discontinuity",
            "small_adjacent_interpretation": "late_cache_producing_path_nearly_frozen",
        },
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--chain-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(Path(args.checkpoint_root), Path(args.chain_result), Path(args.output))
    print(json.dumps(result["diagnosis"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
