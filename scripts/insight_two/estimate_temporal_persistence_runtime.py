#!/usr/bin/env python3
"""Seal a conservative 512-user runtime estimate from the persistence canary."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
CANARY = (
    ROOT
    / "results/yambda500m_medium_seed17/insight2_functional_boundary_v1/diagnostic_temporal_persistence_v1/canary/summary.json"
)
OUTPUT = CANARY.parents[1] / "resource_estimate.json"
CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_temporal_persistence_v1.yaml"
)
POPULATION = ROOT / "data/manifests/yambda500m_medium_insight1_locality_v1/population.npz"
PRIMARY = (
    ROOT
    / "data/manifests/yambda500m_medium_hstu_native_d7_d14_v1/requests_fidelity.parquet"
)
V5 = (
    ROOT
    / "data/manifests/yambda500m_medium_hstu_native_d14_v5_extension_v1/requests_fidelity.parquet"
)
DAY = 86_400


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def count_groups(uids: np.ndarray) -> tuple[int, list[int]]:
    uid_to_index = {int(uid): index for index, uid in enumerate(uids)}
    by_rank = [0, 0, 0, 0]
    total = 0
    for edge, cutover_day in enumerate((231, 245, 259, 273, 287)):
        path = V5 if edge == 4 else PRIMARY
        table = pq.read_table(
            path,
            filters=[
                ("time_block", "=", "matrix_horizon"),
                ("target_known", "=", True),
                ("query_timestamp", ">=", cutover_day * DAY),
                ("query_timestamp", "<", (cutover_day + 14) * DAY),
                ("uid", "in", [int(uid) for uid in uids]),
            ],
            columns=["uid", "query_timestamp"],
        )
        groups = Counter(zip(table["uid"].to_pylist(), table["query_timestamp"].to_pylist()))
        total += len(groups)
        for uid, _ in groups:
            by_rank[uid_to_index[int(uid)] % 4] += 1
    return total, by_rank


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    canary = json.loads(CANARY.read_text(encoding="utf-8"))
    population = np.load(POPULATION, allow_pickle=False)["uids"]
    canary_groups, canary_ranks = count_groups(population[:32])
    discovery_groups, discovery_ranks = count_groups(population[:512])
    user_ratio = 512 / 32
    global_request_ratio = discovery_groups / max(1, canary_groups)
    rank_request_ratio = max(discovery_ranks) / max(1, max(canary_ranks))
    conservative_ratio = max(user_ratio, global_request_ratio, rank_request_ratio)
    estimate_seconds = float(canary["elapsed_seconds"]) * conservative_ratio
    payload = {
        "status": "temporal_persistence_discovery_resource_estimate",
        "contract_sha256": sha256(CONTRACT),
        "canary_elapsed_seconds": canary["elapsed_seconds"],
        "canary_request_groups": canary_groups,
        "discovery_request_groups": discovery_groups,
        "canary_request_groups_by_rank": canary_ranks,
        "discovery_request_groups_by_rank": discovery_ranks,
        "user_ratio": user_ratio,
        "global_request_ratio": global_request_ratio,
        "maximum_rank_request_ratio": rank_request_ratio,
        "conservative_scale_ratio": conservative_ratio,
        "estimated_512_user_seconds": estimate_seconds,
        "estimated_512_user_minutes": estimate_seconds / 60,
        "canary_peak_allocated_mib": canary["peak_allocated_mib"],
        "canary_peak_reserved_mib": canary["peak_reserved_mib"],
        "under_30_minutes": estimate_seconds <= 30 * 60,
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
