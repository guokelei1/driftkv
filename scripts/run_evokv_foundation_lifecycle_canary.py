from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from hstu_kvcache.migration.foundation_lifecycle import (
    FoundationGroupSpec,
    FoundationRecordSpec,
    FoundationRollingLifecycle,
    deterministic_extent_payload,
)

DEFAULT_WORKLOAD = Path(
    "data/processed/evokv_foundation/x_qk_het_foundation.npz"
)
DEFAULT_OUTPUT = Path(
    "results/system/evokv_foundation/lifecycle_canary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layout", choices=("het", "hom"), default="het")
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--workspace-bytes", type=int, default=0)
    parser.add_argument("--source-version", default="theta0")
    parser.add_argument("--target-version", default="theta1")
    parser.add_argument("--skip-fault-injection", action="store_true")
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _metadata(source: np.lib.npyio.NpzFile) -> dict[str, object]:
    if "metadata_json" not in source.files:
        return {}
    value = source["metadata_json"]
    parsed = json.loads(str(value.item()))
    if not isinstance(parsed, dict):
        raise ValueError("foundation workload metadata is invalid")
    return parsed


def _select_indices(
    record_ids: np.ndarray,
    target_lengths: np.ndarray,
) -> tuple[tuple[str, int], ...]:
    if len(record_ids) < 4:
        raise ValueError("foundation lifecycle canary requires four records")
    stable = sorted(
        range(len(record_ids)),
        key=lambda index: (
            int(target_lengths[index]),
            int(record_ids[index]),
        ),
    )
    selected: list[tuple[str, int]] = [
        ("short", stable[0]),
        ("saturated", stable[-1]),
    ]
    used = {stable[0], stable[-1]}
    for label, target in (("mid", 256), ("long", 384)):
        candidate = min(
            (index for index in stable if index not in used),
            key=lambda index: (
                abs(int(target_lengths[index]) - target),
                int(target_lengths[index]),
                int(record_ids[index]),
            ),
        )
        selected.append((label, candidate))
        used.add(candidate)
    by_label = {label: index for label, index in selected}
    return tuple(
        (label, by_label[label])
        for label in ("short", "mid", "long", "saturated")
    )


def _synthetic_arrays() -> dict[str, np.ndarray]:
    return {
        "record_user_ids": np.array(
            [101, 202, 303, 404],
            dtype=np.int64,
        ),
        "old_length": np.array([64, 224, 352, 512], dtype=np.int16),
        "target_length": np.array(
            [96, 256, 384, 512],
            dtype=np.int16,
        ),
        "hom_old_allocated_length": np.full(4, 512, dtype=np.int16),
        "hom_target_allocated_length": np.full(
            4,
            512,
            dtype=np.int16,
        ),
    }


def _load_arrays(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object], str]:
    if not path.is_file():
        return _synthetic_arrays(), {}, "deterministic_synthetic_smoke"
    required = {
        "record_user_ids",
        "old_length",
        "target_length",
        "hom_old_allocated_length",
        "hom_target_allocated_length",
    }
    with np.load(path, allow_pickle=False) as source:
        missing = required - set(source.files)
        if missing:
            raise ValueError(
                f"foundation workload keys are missing: {sorted(missing)}"
            )
        arrays = {
            name: np.asarray(source[name])
            for name in sorted(required)
        }
        metadata = _metadata(source)
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("foundation workload arrays have different lengths")
    return arrays, metadata, "foundation_npz"


def build_records(
    arrays: dict[str, np.ndarray],
    *,
    layout: str,
    source_version: str,
    target_version: str,
) -> tuple[tuple[str, FoundationRecordSpec], ...]:
    record_ids = arrays["record_user_ids"]
    old_lengths = arrays["old_length"]
    target_lengths = arrays["target_length"]
    selected = _select_indices(record_ids, target_lengths)
    output = []
    for label, index in selected:
        old_valid = int(old_lengths[index])
        target_valid = int(target_lengths[index])
        old_allocated = (
            old_valid
            if layout == "het"
            else int(arrays["hom_old_allocated_length"][index])
        )
        target_allocated = (
            target_valid
            if layout == "het"
            else int(arrays["hom_target_allocated_length"][index])
        )
        output.append(
            (
                label,
                FoundationRecordSpec(
                    record_id=int(record_ids[index]),
                    route="action_plan_overlay_pending",
                    source_version=source_version,
                    target_version=target_version,
                    old_valid_tokens=old_valid,
                    old_allocated_tokens=old_allocated,
                    target_valid_tokens=target_valid,
                    target_allocated_tokens=target_allocated,
                ),
            )
        )
    return tuple(output)


def run(args: argparse.Namespace) -> dict[str, object]:
    if (
        args.layers < 1
        or args.width < 1
        or args.dtype_bytes < 1
        or args.workspace_bytes < 0
    ):
        raise ValueError("foundation lifecycle geometry is invalid")
    arrays, metadata, workload_kind = _load_arrays(args.workload)
    labeled_records = build_records(
        arrays,
        layout=args.layout,
        source_version=args.source_version,
        target_version=args.target_version,
    )
    records = tuple(value for _, value in labeled_records)
    bytes_per_token = (
        2 * args.layers * args.width * args.dtype_bytes
    )
    source_payloads = {
        record.record_id: deterministic_extent_payload(
            record.record_id,
            record.source_version,
            record.old_allocated_tokens * bytes_per_token,
        )
        for record in records
    }
    shadow_capacity = max(
        record.target_allocated_tokens * bytes_per_token
        for record in records
    )
    staging_capacity = max(
        max(
            record.old_allocated_tokens,
            record.target_allocated_tokens,
        )
        * bytes_per_token
        for record in records
    )
    lifecycle = FoundationRollingLifecycle(
        records,
        source_payloads,
        bytes_per_token=bytes_per_token,
        shadow_capacity_bytes=shadow_capacity,
        staging_capacity_bytes=staging_capacity,
        workspace_capacity_bytes=args.workspace_bytes,
    )
    receipts = []
    groups = []
    fault_injection_passed = None
    last_group = None
    last_target = None
    for ordinal, (label, record) in enumerate(labeled_records):
        group = FoundationGroupSpec(
            group_id=f"{ordinal:02d}-{label}-{record.record_id}",
            records=(record,),
            staging_bytes=max(
                record.old_allocated_tokens,
                record.target_allocated_tokens,
            )
            * bytes_per_token,
            workspace_bytes=args.workspace_bytes,
        )
        payload = deterministic_extent_payload(
            record.record_id,
            record.target_version,
            record.target_allocated_tokens * bytes_per_token,
        )
        target = lifecycle.prepare_target(record.record_id, payload)
        if ordinal == 0 and not args.skip_fault_injection:
            before = lifecycle.state(record.record_id)
            corrupted = replace(
                target,
                payload_sha256="0" * 64,
            )
            try:
                lifecycle.execute_group(group, (corrupted,))
            except ValueError:
                after = lifecycle.state(record.record_id)
                fault_injection_passed = before == after
            else:
                fault_injection_passed = False
            if not fault_injection_passed:
                raise RuntimeError("failed group changed published state")
        receipt = lifecycle.execute_group(group, (target,))
        receipts.append(receipt.to_dict())
        groups.append(
            {
                "label": label,
                "group_id": group.group_id,
                "record": record.to_dict(),
                "staging_bytes": group.staging_bytes,
                "workspace_bytes": group.workspace_bytes,
            }
        )
        last_group = group
        last_target = target
    assert last_group is not None
    assert last_target is not None
    replay = lifecycle.execute_group(last_group, (last_target,))
    if replay.status != "already_committed":
        raise RuntimeError("repeated group was not recognized idempotently")
    ledger = lifecycle.ledger()
    canary_passed = bool(
        ledger["coverage"]["complete"]
        and ledger["coverage"]["exactly_once"]
        and ledger["capacity"]["shadow_capacity_bound_respected"]
        and ledger["groups_committed"] == 4
        and ledger["idempotent_group_replays"] == 1
        and (
            args.skip_fault_injection
            or (
                fault_injection_passed
                and ledger["groups_failed"] == 1
            )
        )
    )
    if not canary_passed:
        raise RuntimeError("foundation lifecycle canary did not close")
    return {
        "protocol": ledger["protocol"],
        "scientific_result": False,
        "formal_design3": False,
        "purpose": "rolling_capacity_and_transaction_semantics_canary",
        "executes_d1_d2_numeric": False,
        "payload_source": "deterministic_full_allocated_extent_fixture",
        "workload": {
            "kind": workload_kind,
            "path": (
                str(args.workload)
                if args.workload.is_file()
                else None
            ),
            "sha256": (
                _file_sha256(args.workload)
                if args.workload.is_file()
                else None
            ),
            "metadata_protocol": metadata.get("protocol"),
            "layout": args.layout,
        },
        "geometry": {
            "layers": args.layers,
            "width": args.width,
            "dtype_bytes": args.dtype_bytes,
            "bytes_per_token": bytes_per_token,
        },
        "groups": groups,
        "committed_receipts": receipts,
        "fault_injection": {
            "enabled": not args.skip_fault_injection,
            "failed_group_left_live_state_unchanged": (
                fault_injection_passed
            ),
        },
        "idempotent_replay": replay.to_dict(),
        "ledger": ledger,
        "canary_passed": canary_passed,
    }


def main() -> None:
    args = parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
