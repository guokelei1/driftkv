#!/usr/bin/env python3
"""Seal a conservative 512-user resource estimate from both focused canaries."""

from __future__ import annotations

import json

from insight_two.common import CONTRACT, RESULT_ROOT, sha256_file


def main() -> None:
    output = RESULT_ROOT / "resource_estimate.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rank0 = json.loads((RESULT_ROOT / "canary_rank0/summary.json").read_text())
    low = json.loads((RESULT_ROOT / "canary_low_rank/summary.json").read_text())
    if not rank0["passed"] or not low["passed"]:
        raise RuntimeError("both focused canaries must pass")
    if rank0["contract_sha256"] != sha256_file(CONTRACT):
        raise RuntimeError("rank-0 canary contract differs")
    if low["contract_sha256"] != sha256_file(CONTRACT):
        raise RuntimeError("low-rank canary contract differs")
    # The low-rank canary ran four positive ranks. Discovery adds rank zero;
    # scale linearly by users and conservatively by 5/4 configurations.
    estimated = float(low["elapsed_seconds"]) * (512 / 32) * (5 / 4)
    payload = {
        "status": "insight2_discovery_resource_estimate_complete",
        "contract_sha256": sha256_file(CONTRACT),
        "source_canaries": [
            "canary_rank0/summary.json",
            "canary_low_rank/summary.json",
        ],
        "discovery_users": 512,
        "edges": 5,
        "stages": 4,
        "ranks": [0, 1, 2, 4, 8],
        "estimated_wall_seconds": estimated,
        "estimated_wall_minutes": estimated / 60,
        "estimated_under_detached_threshold_30_minutes": estimated < 1800,
        "peak_reserved_mib_per_rank_upper_bound": max(
            float(rank0["peak_reserved_mib"]), float(low["peak_reserved_mib"])
        ),
        "physical_gpus": [0, 1, 2, 3],
        "parallelism": "independent_UID_shards",
        "batch_size_per_rank": 1,
        "labels_read": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

