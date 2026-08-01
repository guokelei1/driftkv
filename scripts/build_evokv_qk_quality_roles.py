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

PROTOCOL = "evokv_qk_quality_roles_development_v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        default=Path(
            "configs/evokv_foundation/qk_post_base_roles.json"
        ),
    )
    parser.add_argument(
        "--upstream-prepared",
        type=Path,
        default=Path(
            "data/processed/evokv_d3_m1_qk_entity_2560.npz"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hash-salt",
        default="evokv-qk-quality-train8192-qual4096-v0",
    )
    parser.add_argument("--theta01-users", type=int, default=8_192)
    parser.add_argument("--qualification-users", type=int, default=4_096)
    parser.add_argument("--minimum-events", type=int, default=96)
    parser.add_argument(
        "--additional-exclusion-roles",
        type=Path,
        action="append",
        default=[],
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
        raise ValueError(f"source role is absent: {name}")
    record = roles[name]
    values = np.asarray(record.get("user_ids"), dtype=np.int64)
    if (
        len(values) != int(record.get("count", -1))
        or array_sha256(values) != record.get("user_ids_sha256")
        or len(np.unique(values)) != len(values)
        or np.any(values < 0)
    ):
        raise ValueError(f"source role differs: {name}")
    return values


def atomic_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"quality role output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if (
        args.theta01_users < 1
        or args.qualification_users < 1
        or args.minimum_events < 96
        or not args.hash_salt
    ):
        raise ValueError("quality role dimensions differ")
    lengths, length_metadata = load_npz(args.length_cache)
    user_ids = np.asarray(lengths.get("user_ids"), dtype=np.int64)
    raw_lengths = np.asarray(lengths.get("raw_lengths"), dtype=np.int64)
    if (
        len(user_ids) != len(raw_lengths)
        or len(user_ids) == 0
        or len(np.unique(user_ids)) != len(user_ids)
        or np.any(user_ids < 0)
    ):
        raise ValueError("QK length cache differs")
    source_document = json.loads(args.source_roles.read_text())
    inherited_names = ("theta12", "fit", "profile", "final")
    inherited = {
        name: role_values(source_document, name)
        for name in inherited_names
    }
    source_names = tuple(source_document["roles"])
    source_all = np.concatenate(
        [role_values(source_document, name) for name in source_names]
    )
    additional_bindings = []
    additional_values = []
    for path in args.additional_exclusion_roles:
        document = json.loads(path.read_text())
        names = tuple(document.get("roles", {}))
        if not names:
            raise ValueError(f"additional role exclusion is empty: {path}")
        values = np.concatenate(
            [role_values(document, name) for name in names]
        )
        additional_values.append(values)
        additional_bindings.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "protocol": document.get("protocol"),
                "declared_users": len(values),
                "user_ids_sha256": array_sha256(values),
            }
        )
    upstream, upstream_metadata = load_npz(args.upstream_prepared)
    upstream_users = np.asarray(
        upstream.get("original_user_ids"), dtype=np.int64
    )
    if (
        len(upstream_users) == 0
        or len(np.unique(upstream_users)) != len(upstream_users)
        or np.any(upstream_users < 0)
    ):
        raise ValueError("upstream user exclusion differs")
    dense_size = max(
        int(user_ids.max()),
        int(source_all.max()),
        int(upstream_users.max()),
        *(
            [int(values.max()) for values in additional_values]
            if additional_values
            else [0]
        ),
    ) + 1
    excluded = np.zeros(dense_size, dtype=np.bool_)
    excluded[source_all] = True
    excluded[upstream_users] = True
    for values in additional_values:
        excluded[values] = True
    eligible = user_ids[
        (raw_lengths >= args.minimum_events) & ~excluded[user_ids]
    ]
    ordered = stable_user_order(eligible, args.hash_salt)
    required = args.theta01_users + args.qualification_users
    if len(ordered) < required:
        raise ValueError("insufficient unused QK users for quality roles")
    theta01 = ordered[: args.theta01_users]
    qualification = ordered[
        args.theta01_users : args.theta01_users
        + args.qualification_users
    ]
    roles = {
        "theta01": theta01,
        "theta12": inherited["theta12"],
        "fit": inherited["fit"],
        "profile": inherited["profile"],
        "qualification": qualification,
        "final": inherited["final"],
    }
    combined = np.concatenate(list(roles.values()))
    if len(np.unique(combined)) != len(combined):
        raise ValueError("quality roles are not pairwise disjoint")
    length_by_user = np.zeros(int(user_ids.max()) + 1, dtype=np.int32)
    length_by_user[user_ids] = raw_lengths.astype(np.int32, copy=False)
    document = {
        "protocol": PROTOCOL,
        "scientific_result": False,
        "formal_result": False,
        "hash_salt": args.hash_salt,
        "source": source_document.get("source"),
        "selection": {
            "purpose": (
                "replace repeated small-window updates with one pass over "
                "more independent users"
            ),
            "minimum_events": args.minimum_events,
            "eligible_unused_users": len(eligible),
            "eligible_unused_user_ids_sha256": array_sha256(eligible),
            "excluded_source_role_users": len(source_all),
            "excluded_source_role_user_ids_sha256": array_sha256(
                source_all
            ),
            "excluded_upstream_users": len(upstream_users),
            "excluded_upstream_user_ids_sha256": array_sha256(
                upstream_users
            ),
            "additional_exclusion_role_files": len(
                additional_bindings
            ),
            "additional_excluded_users": int(
                np.count_nonzero(
                    np.logical_or.reduce(
                        [
                            np.isin(user_ids, values)
                            for values in additional_values
                        ]
                    )
                )
            )
            if additional_values
            else 0,
            "new_theta01_disjoint_from_all_source_roles": True,
            "new_qualification_disjoint_from_all_source_roles": True,
            "new_roles_disjoint_from_upstream": True,
        },
        "bindings": {
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
            "additional_exclusion_roles": additional_bindings,
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
                "output": str(args.output),
                "theta01_users": len(theta01),
                "qualification_users": len(qualification),
                "eligible_unused_users": len(eligible),
                "status": "complete",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
