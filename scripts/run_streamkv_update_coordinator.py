from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from hstu_kvcache.migration import (
    CompiledCacheAdapter,
    DRAMKVUpdateDestination,
    FilesystemKVUpdateDestination,
    HBMKVUpdateDestination,
    InMemoryRemoteObjectStore,
    JaggedMigrationCapsuleBatch,
    MigrationProgram,
    OutOfCoreKVUpdateEngine,
    RemoteKVUpdateDestination,
)

COORDINATOR_PROTOCOL = "streamkv_update_coordinator_concept_v1"
CAPSULE_SHARD_PROTOCOL = "streamkv_jagged_capsule_shard_v1"


@dataclass(frozen=True)
class ProgramArtifact:
    source_version: str
    path: Path


@dataclass(frozen=True)
class CapsuleShard:
    source_version: str
    path: Path


@dataclass(frozen=True)
class DestinationSpec:
    kind: str
    destination_id: str
    root: Path | None
    durable: bool
    capacity_bytes: int | None
    prefix: str


@dataclass(frozen=True)
class RuntimeSpec:
    devices: tuple[str, ...]
    wave_batch_limit: int
    max_inflight_batches: int
    publication_queue_depth: int
    partition_strategy: str


@dataclass(frozen=True)
class KVUpdateJobSpec:
    job_id: str
    target_version: str
    programs: tuple[ProgramArtifact, ...]
    capsule_shards: tuple[CapsuleShard, ...]
    destination: DestinationSpec
    runtime: RuntimeSpec
    protocol: str = COORDINATOR_PROTOCOL

    @classmethod
    def from_dict(cls, value: dict[str, Any], base_dir: Path) -> KVUpdateJobSpec:
        if value.get("protocol") != COORDINATOR_PROTOCOL:
            raise ValueError("coordinator protocol mismatch")
        destination_value = value["destination"]
        runtime_value = value["runtime"]
        programs = tuple(
            ProgramArtifact(
                source_version=str(item["source_version"]),
                path=_resolve_path(base_dir, item["path"]),
            )
            for item in value["programs"]
        )
        capsule_shards = tuple(
            CapsuleShard(
                source_version=str(item["source_version"]),
                path=_resolve_path(base_dir, item["path"]),
            )
            for item in value["capsule_shards"]
        )
        root_value = destination_value.get("root")
        spec = cls(
            job_id=str(value["job_id"]),
            target_version=str(value["target_version"]),
            programs=programs,
            capsule_shards=capsule_shards,
            destination=DestinationSpec(
                kind=str(destination_value["kind"]),
                destination_id=str(
                    destination_value.get(
                        "destination_id",
                        destination_value["kind"],
                    )
                ),
                root=(
                    None
                    if root_value is None
                    else _resolve_path(base_dir, root_value)
                ),
                durable=bool(destination_value.get("durable", True)),
                capacity_bytes=(
                    None
                    if destination_value.get("capacity_bytes") is None
                    else int(destination_value["capacity_bytes"])
                ),
                prefix=str(destination_value.get("prefix", "streamkv")),
            ),
            runtime=RuntimeSpec(
                devices=tuple(str(item) for item in runtime_value["devices"]),
                wave_batch_limit=int(
                    runtime_value.get("wave_batch_limit", 8)
                ),
                max_inflight_batches=int(
                    runtime_value.get("max_inflight_batches", 3)
                ),
                publication_queue_depth=int(
                    runtime_value.get("publication_queue_depth", 2)
                ),
                partition_strategy=str(
                    runtime_value.get("partition_strategy", "greedy_lpt")
                ),
            ),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.job_id or not self.target_version:
            raise ValueError("job_id and target_version must be nonempty")
        if not self.programs or not self.capsule_shards:
            raise ValueError("at least one program and capsule shard are required")
        program_sources = [item.source_version for item in self.programs]
        if len(set(program_sources)) != len(program_sources):
            raise ValueError("program source versions must be unique")
        shard_sources = {item.source_version for item in self.capsule_shards}
        unknown_sources = shard_sources.difference(program_sources)
        if unknown_sources:
            raise ValueError(
                f"capsule shards have no program: {sorted(unknown_sources)}"
            )
        if not self.runtime.devices:
            raise ValueError("at least one execution device is required")
        if self.destination.kind not in {
            "dram",
            "filesystem",
            "hbm",
            "remote_reference",
        }:
            raise ValueError("unsupported destination kind")
        if self.destination.kind == "filesystem" and self.destination.root is None:
            raise ValueError("filesystem destination requires root")
        if self.runtime.wave_batch_limit < 1:
            raise ValueError("wave_batch_limit must be positive")
        if self.runtime.max_inflight_batches < 1:
            raise ValueError("max_inflight_batches must be positive")
        if self.runtime.publication_queue_depth < 1:
            raise ValueError("publication_queue_depth must be positive")


class StreamKVUpdateCoordinator:
    def __init__(self, spec: KVUpdateJobSpec) -> None:
        self.spec = spec

    def plan(self) -> dict[str, Any]:
        shards_by_source = {
            program.source_version: [
                str(shard.path)
                for shard in self.spec.capsule_shards
                if shard.source_version == program.source_version
            ]
            for program in self.spec.programs
        }
        artifacts = [
            {
                "kind": "migration_program",
                "source_version": item.source_version,
                "path": str(item.path),
                "exists": item.path.is_file(),
            }
            for item in self.spec.programs
        ]
        artifacts.extend(
            {
                "kind": "capsule_shard",
                "source_version": item.source_version,
                "path": str(item.path),
                "exists": item.path.is_file(),
            }
            for item in self.spec.capsule_shards
        )
        return {
            "protocol": self.spec.protocol,
            "status": "planned",
            "job_id": self.spec.job_id,
            "target_version": self.spec.target_version,
            "control_flow": [
                "resolve_program_and_capsule_artifacts",
                "group_capsules_by_source_version_cohort",
                "load_verified_program_per_cohort",
                "execute_bounded_multi_device_waves",
                "stage_destination_extents",
                "atomically_publish_target_manifest",
            ],
            "cohorts": [
                {
                    "source_version": item.source_version,
                    "target_version": self.spec.target_version,
                    "program": str(item.path),
                    "capsule_shards": shards_by_source[item.source_version],
                    "action": "compiled_affine_migration",
                }
                for item in self.spec.programs
            ],
            "runtime": asdict(self.spec.runtime),
            "destination": _json_ready(asdict(self.spec.destination)),
            "artifacts": artifacts,
            "ready_to_execute": all(item["exists"] for item in artifacts),
            "scope": {
                "reuse_safety_prediction": False,
                "request_or_serving_scheduler": False,
                "training_orchestration": False,
                "program_compilation": False,
                "capsule_materialization": False,
            },
        }

    def execute(self) -> dict[str, Any]:
        plan = self.plan()
        if not plan["ready_to_execute"]:
            missing = [
                item["path"]
                for item in plan["artifacts"]
                if not item["exists"]
            ]
            raise FileNotFoundError(f"missing update artifacts: {missing}")
        programs = tuple(
            _load_program(item, self.spec.target_version)
            for item in self.spec.programs
        )
        batches = tuple(
            batch
            for shard in self.spec.capsule_shards
            for batch in _load_capsule_shard(shard)
        )
        destination = _make_destination(self.spec)
        engine = OutOfCoreKVUpdateEngine(
            programs,
            devices=self.spec.runtime.devices,
            destination=destination,
            wave_batch_limit=self.spec.runtime.wave_batch_limit,
            max_inflight_batches=self.spec.runtime.max_inflight_batches,
            publication_queue_depth=self.spec.runtime.publication_queue_depth,
            partition_strategy=self.spec.runtime.partition_strategy,
        )
        report = engine.run(self.spec.job_id, batches)
        return {
            **plan,
            "status": "committed",
            "manifest": report.manifest.to_dict(),
            "metrics": _json_ready(asdict(report.metrics)),
            "runtime_protocol": report.protocol,
        }


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _load_program(
    artifact: ProgramArtifact,
    target_version: str,
) -> MigrationProgram:
    payload = torch.load(
        artifact.path,
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(payload, MigrationProgram):
        program = payload
    elif isinstance(payload, dict):
        program = MigrationProgram(
            source_version=str(payload["source_version"]),
            target_version=str(payload["target_version"]),
            adapter=CompiledCacheAdapter(
                weights=payload["weights"].detach().cpu(),
                biases=payload["biases"].detach().cpu(),
                source_rank=int(
                    payload.get("source_rank", payload["weights"].shape[1])
                ),
                ridge=float(payload.get("ridge", 0.0)),
            ),
        )
    else:
        raise TypeError(f"unsupported program artifact: {artifact.path}")
    if program.source_version != artifact.source_version:
        raise ValueError("declared and stored program source versions differ")
    if program.target_version != target_version:
        raise ValueError("program target differs from update job target")
    return program


def _load_capsule_shard(
    shard: CapsuleShard,
) -> tuple[JaggedMigrationCapsuleBatch, ...]:
    payload = torch.load(
        shard.path,
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(payload, dict) and "batches" in payload:
        if payload.get("protocol") not in {None, CAPSULE_SHARD_PROTOCOL}:
            raise ValueError("capsule shard protocol mismatch")
        values = payload["batches"]
    elif isinstance(payload, (list, tuple)):
        values = payload
    else:
        values = (payload,)
    batches = tuple(_decode_capsule_batch(value) for value in values)
    if not batches:
        raise ValueError("capsule shard contains no batches")
    if any(
        batch.migration_anchor_version != shard.source_version
        for batch in batches
    ):
        raise ValueError("declared and stored capsule source versions differ")
    return batches


def _decode_capsule_batch(value: Any) -> JaggedMigrationCapsuleBatch:
    if isinstance(value, JaggedMigrationCapsuleBatch):
        return value.to("cpu")
    if not isinstance(value, dict):
        raise TypeError("capsule batch must be an object or tensor dictionary")
    return JaggedMigrationCapsuleBatch(
        record_ids=tuple(int(item) for item in value["record_ids"]),
        migration_anchor_version=str(value["migration_anchor_version"]),
        normed=value["normed"].detach().cpu(),
        lengths=value["lengths"].detach().cpu(),
        offsets=value["offsets"].detach().cpu(),
    )


def _make_destination(spec: KVUpdateJobSpec):
    value = spec.destination
    if value.kind == "dram":
        return DRAMKVUpdateDestination(value.destination_id)
    if value.kind == "filesystem":
        return FilesystemKVUpdateDestination(
            value.root,
            destination_id=value.destination_id,
            durable=value.durable,
        )
    if value.kind == "hbm":
        return HBMKVUpdateDestination(
            spec.runtime.devices,
            destination_id=value.destination_id,
            capacity_bytes=value.capacity_bytes,
        )
    return RemoteKVUpdateDestination(
        InMemoryRemoteObjectStore(),
        destination_id=value.destination_id,
        prefix=value.prefix,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _template() -> dict[str, Any]:
    return {
        "protocol": COORDINATOR_PROTOCOL,
        "job_id": "theta11-kv-refresh",
        "target_version": "theta11",
        "programs": [
            {
                "source_version": "theta0",
                "path": "artifacts/program_theta0_to_theta11.pt",
            },
            {
                "source_version": "theta4",
                "path": "artifacts/program_theta4_to_theta11.pt",
            },
        ],
        "capsule_shards": [
            {
                "source_version": "theta0",
                "path": "artifacts/capsules_theta0_shard0.pt",
            },
            {
                "source_version": "theta4",
                "path": "artifacts/capsules_theta4_shard0.pt",
            },
        ],
        "runtime": {
            "devices": ["cuda:0", "cuda:1"],
            "wave_batch_limit": 8,
            "max_inflight_batches": 3,
            "publication_queue_depth": 2,
            "partition_strategy": "greedy_lpt",
        },
        "destination": {
            "kind": "filesystem",
            "destination_id": "local-ssd",
            "root": "artifacts/published_kv",
            "durable": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-spec", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-template", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_template:
        print(json.dumps(_template(), indent=2))
        return
    if args.job_spec is None:
        raise ValueError("--job-spec is required unless --print-template is used")
    job_spec_path = args.job_spec.expanduser().resolve()
    spec = KVUpdateJobSpec.from_dict(
        json.loads(job_spec_path.read_text()),
        job_spec_path.parent,
    )
    coordinator = StreamKVUpdateCoordinator(spec)
    result = coordinator.execute() if args.execute else coordinator.plan()
    rendered = json.dumps(_json_ready(result), indent=2)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
