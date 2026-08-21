"""Guarded readers for sealed P7 compact manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

FIDELITY_FORBIDDEN_COLUMNS = frozenset(
    {
        "label",
        "target_index",
        "rankable",
        "target_stratum",
        "is_organic",
        "prior_30m_same_item",
        "latest_item",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class QualificationUnlock:
    """Evidence required to read a sealed qualification manifest."""

    qualification_index_hash: str
    frozen_base_hashes: tuple[str, ...]
    frozen_checkpoint_hashes: tuple[str, ...]
    checkpoint_selection_complete: bool

    def validate(self, expected_index_hash: str) -> None:
        if self.qualification_index_hash != expected_index_hash:
            raise PermissionError("qualification unlock refers to a different sealed index")
        if not self.frozen_base_hashes or not self.frozen_checkpoint_hashes:
            raise PermissionError("base and checkpoint hashes must be frozen before qualification")
        if not self.checkpoint_selection_complete:
            raise PermissionError("checkpoint selection must be complete before qualification")


def load_compact_index(
    path: str | Path,
    *,
    qualification_unlock: QualificationUnlock | None = None,
) -> dict:
    """Load an index while keeping qualification sealed by default."""
    path = Path(path)
    index = json.loads(path.read_text())
    if index.get("split") == "qualification":
        if qualification_unlock is None:
            raise PermissionError("qualification manifest is sealed")
        qualification_unlock.validate(sha256_file(path))
    return index


def read_request_table(
    index_path: str | Path,
    *,
    qualification_unlock: QualificationUnlock | None = None,
):
    """Read and concatenate request shards after enforcing the guard."""
    import pyarrow.parquet as pq

    index_path = Path(index_path)
    index = load_compact_index(index_path, qualification_unlock=qualification_unlock)
    root = index_path.parent
    paths = [root / shard["path"] for shard in index["request_shards"]]
    return pq.read_table(paths)


def read_request_view(
    index_path: str | Path,
    workload: str,
    manifest_kind: str,
    *,
    qualification_unlock: QualificationUnlock | None = None,
):
    """Read one request view and remove target-bearing fields from fidelity views."""
    import pyarrow.compute as pc

    table = read_request_table(index_path, qualification_unlock=qualification_unlock)
    mask = pc.and_(
        pc.equal(table["workload"], workload),
        pc.equal(table["manifest_kind"], manifest_kind),
    )
    table = table.filter(mask)
    if "fidelity" in manifest_kind:
        keep = [name for name in table.column_names if name not in FIDELITY_FORBIDDEN_COLUMNS]
        table = table.select(keep)
    return table
