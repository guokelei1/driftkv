from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(
            "configs/evokv_foundation/selected_checkpoint_registry_development_v0.json"
        ),
    )
    parser.add_argument("--full-payload", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_bound_file(path: Path, expected: dict[str, object], full: bool) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_bytes = path.stat().st_size
    if "bytes" in expected and observed_bytes != int(expected["bytes"]):
        raise ValueError(f"size differs for {path}")
    if full and sha256(path) != expected.get("sha256"):
        raise ValueError(f"SHA-256 differs for {path}")
    return observed_bytes


def manifest_payloads(
    manifest_path: Path,
    manifest: dict[str, object],
) -> list[dict[str, object]]:
    values = [manifest["dense"], manifest["projection"]]
    values.extend(manifest["embedding_shards"])
    active = manifest.get("optimizer_active_rows", {})
    values.extend(active.get("bitmap_shards", []))
    return [
        {**value, "resolved_path": manifest_path.parent / value["path"]}
        for value in values
    ]


def verify_manifest(entry: dict[str, object], full: bool) -> tuple[int, int]:
    path = Path(entry["manifest_path"])
    if not path.is_file() or sha256(path) != entry["manifest_sha256"]:
        raise ValueError(f"manifest binding differs for {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("version") != entry["version"]:
        raise ValueError(f"manifest version differs for {path}")
    payload_bytes = 0
    payload_files = manifest_payloads(path, manifest)
    for payload in payload_files:
        payload_bytes += verify_bound_file(payload["resolved_path"], payload, full)
    return len(payload_files), payload_bytes + path.stat().st_size


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    registry = json.loads(args.registry.read_text())
    if registry.get("schema") != "evokv_selected_checkpoint_registry_development_v0":
        raise ValueError("selected checkpoint registry schema differs")
    chain_reports = {}
    for name, chain in registry["selected_chains"].items():
        if chain.get("status") == "retired_rebuild_required":
            chain_reports[name] = {
                "status": "retired_rebuild_required",
                "retirement_ledger": chain.get("retirement_ledger"),
            }
            continue
        checkpoint_files = 0
        durable_bytes = 0
        for entry in chain["checkpoint_manifests"]:
            files, observed_bytes = verify_manifest(entry, args.full_payload)
            checkpoint_files += files + 1
            durable_bytes += observed_bytes
        for entry in chain.get("optimizer_resume_points", []):
            durable_bytes += verify_bound_file(
                Path(entry["path"]),
                entry,
                args.full_payload,
            )
            checkpoint_files += 1
        for entry in chain["source_results"]:
            path = Path(entry["path"])
            if not path.is_file() or sha256(path) != entry["sha256"]:
                raise ValueError(f"source-result binding differs for {path}")
        chain_reports[name] = {
            "checkpoint_files": checkpoint_files,
            "durable_bytes": durable_bytes,
            "versions": [
                entry["version"] for entry in chain["checkpoint_manifests"]
            ],
        }
    for entry in registry["retained_auxiliary_checkpoint_roots"]:
        if not Path(entry["path"]).is_dir():
            raise FileNotFoundError(entry["path"])
    retirement = registry.get("legacy_small_experiment_retirement")
    if retirement is not None:
        retirement_path = Path(retirement["path"])
        if (
            retirement.get("status") != "complete"
            or not retirement_path.is_file()
            or retirement_path.stat().st_size != int(retirement["bytes"])
            or sha256(retirement_path) != retirement["sha256"]
            or not Path(retirement["recovery_path"]).is_dir()
        ):
            raise ValueError("legacy small experiment retirement differs")
        retirement_ledger = json.loads(retirement_path.read_text())
        present = [
            entry["path"]
            for entry in retirement_ledger["retired_paths"]
            if Path(entry["path"]).exists()
        ]
        if present:
            raise ValueError(f"legacy small experiment paths remain: {present}")
    if registry["cleanup"]["status"] == "complete":
        present = [
            entry["path"]
            for entry in registry["cleanup"]["retired_paths"]
            if Path(entry["path"]).exists()
        ]
        if present:
            raise ValueError(f"retired checkpoint paths remain: {present}")
    report = {
        "cleanup_status": registry["cleanup"]["status"],
        "full_payload_verified": args.full_payload,
        "registry": str(args.registry),
        "registry_sha256": sha256(args.registry),
        "schema": "evokv_selected_checkpoint_verification_v0",
        "selected_chains": chain_reports,
        "status": "pass",
    }
    if args.output is not None:
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
