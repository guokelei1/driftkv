from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from hstu_kvcache.migration.foundation_workload import (
    array_sha256,
    file_sha256,
    stable_user_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theta01-users", type=int, required=True)
    parser.add_argument("--minimum-events", type=int, default=104)
    parser.add_argument("--hash-salt", required=True)
    parser.add_argument(
        "--length-cache",
        type=Path,
        default=Path(
            "data/processed/evokv_foundation/qk_full_user_lengths.npz"
        ),
    )
    parser.add_argument(
        "--source-roles",
        type=Path,
        default=Path("configs/evokv_foundation/qk_post_base_roles.json"),
    )
    parser.add_argument(
        "--upstream-prepared",
        type=Path,
        default=Path("data/processed/evokv_d3_m1_qk_entity_2560.npz"),
    )
    return parser.parse_args()


def load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {
            name: source[name].copy()
            for name in source.files
            if name != "metadata_json"
        }
        metadata = json.loads(str(source["metadata_json"].item()))
    return arrays, metadata


def role_values(document: dict[str, object], name: str) -> np.ndarray:
    roles = document.get("roles")
    if not isinstance(roles, dict) or not isinstance(roles.get(name), dict):
        raise ValueError(f"role is absent: {name}")
    record = roles[name]
    values = np.asarray(record.get("user_ids"), dtype=np.int64)
    if (
        len(values) != int(record.get("count", -1))
        or len(np.unique(values)) != len(values)
        or array_sha256(values) != record.get("user_ids_sha256")
    ):
        raise ValueError(f"role differs: {name}")
    return values


def atomic_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"nested role output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if (
        args.theta01_users < 1
        or args.minimum_events < 96
        or not args.hash_salt
    ):
        raise ValueError("nested role dimensions differ")
    lengths, length_metadata = load_npz(args.length_cache)
    user_ids = np.asarray(lengths.get("user_ids"), dtype=np.int64)
    raw_lengths = np.asarray(lengths.get("raw_lengths"), dtype=np.int64)
    if (
        len(user_ids) != len(raw_lengths)
        or len(user_ids) == 0
        or len(np.unique(user_ids)) != len(user_ids)
    ):
        raise ValueError("QK length cache differs")
    base_document = json.loads(args.base_roles.read_text())
    source_document = json.loads(args.source_roles.read_text())
    base_names = tuple(base_document.get("roles", {}))
    source_names = tuple(source_document.get("roles", {}))
    if not base_names or not source_names:
        raise ValueError("nested role source is empty")
    base = {name: role_values(base_document, name) for name in base_names}
    source_all = np.concatenate(
        [role_values(source_document, name) for name in source_names]
    )
    upstream, upstream_metadata = load_npz(args.upstream_prepared)
    upstream_users = np.asarray(upstream.get("original_user_ids"), dtype=np.int64)
    if len(upstream_users) == 0 or len(np.unique(upstream_users)) != len(
        upstream_users
    ):
        raise ValueError("upstream user exclusion differs")
    existing_theta01 = base.get("theta01")
    if existing_theta01 is None or args.theta01_users <= len(existing_theta01):
        raise ValueError("nested theta01 size must extend the base")
    base_all = np.concatenate(list(base.values()))
    dense_size = max(
        int(user_ids.max()),
        int(base_all.max()),
        int(source_all.max()),
        int(upstream_users.max()),
    ) + 1
    excluded = np.zeros(dense_size, dtype=np.bool_)
    excluded[base_all] = True
    excluded[source_all] = True
    excluded[upstream_users] = True
    eligible = user_ids[
        (raw_lengths >= args.minimum_events) & ~excluded[user_ids]
    ]
    additional_count = args.theta01_users - len(existing_theta01)
    ordered = stable_user_order(eligible, args.hash_salt)
    if len(ordered) < additional_count:
        raise ValueError("insufficient unused QK users for nested roles")
    extra = ordered[:additional_count]
    roles = dict(base)
    roles["theta01"] = np.concatenate((existing_theta01, extra))
    combined = np.concatenate(list(roles.values()))
    if len(np.unique(combined)) != len(combined):
        raise ValueError("nested roles are not pairwise disjoint")
    length_by_user = np.zeros(int(user_ids.max()) + 1, dtype=np.int32)
    length_by_user[user_ids] = raw_lengths.astype(np.int32, copy=False)
    document = {
        "protocol": "evokv_qk_quality_nested_roles_development_v0",
        "scientific_result": False,
        "formal_result": False,
        "hash_salt": args.hash_salt,
        "source": base_document.get("source"),
        "selection": {
            "purpose": "nested independent-user update-data scale diagnostic",
            "minimum_events": args.minimum_events,
            "base_theta01_users": len(existing_theta01),
            "additional_theta01_users": len(extra),
            "total_theta01_users": len(roles["theta01"]),
            "qualification_reused_unchanged": True,
            "base_theta01_prefix_preserved": True,
            "eligible_unused_users": len(eligible),
            "eligible_unused_user_ids_sha256": array_sha256(eligible),
        },
        "bindings": {
            "base_roles": {
                "path": str(args.base_roles),
                "sha256": file_sha256(args.base_roles),
                "protocol": base_document.get("protocol"),
            },
            "length_cache": {
                "path": str(args.length_cache),
                "sha256": file_sha256(args.length_cache),
                "protocol": length_metadata.get("protocol"),
            },
            "source_roles": {
                "path": str(args.source_roles),
                "sha256": file_sha256(args.source_roles),
                "protocol": source_document.get("protocol"),
            },
            "upstream_prepared": {
                "path": str(args.upstream_prepared),
                "sha256": file_sha256(args.upstream_prepared),
                "protocol": upstream_metadata.get("protocol"),
            },
        },
        "roles": {
            name: {
                "count": len(values),
                "user_ids_sha256": array_sha256(values),
                "minimum_raw_length": int(length_by_user[values].min()),
                "maximum_raw_length": int(length_by_user[values].max()),
                "user_ids": [int(value) for value in values],
            }
            for name, values in roles.items()
        },
    }
    atomic_json(args.output, document)
    print(
        json.dumps(
            {
                "additional_theta01_users": len(extra),
                "output": str(args.output),
                "qualification_users": len(roles["qualification"]),
                "status": "complete",
                "theta01_users": len(roles["theta01"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
