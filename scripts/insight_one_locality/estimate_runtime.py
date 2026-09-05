#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from insight_one_locality.common import CONTRACT, EDGES, POPULATION, RESULT_ROOT, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_outputs", nargs="+", type=Path)
    parser.add_argument("--canary", type=Path, default=RESULT_ROOT / "canary")
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "resource_estimate.json")
    parser.add_argument("--memory-fraction-limit", type=float, default=0.90)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite resource estimate: {args.output}")
    canary = json.loads((args.canary / "summary.json").read_text(encoding="utf-8"))
    if not canary.get("passed") or canary["contract_sha256"] != sha256_file(CONTRACT):
        raise RuntimeError("resource estimate requires a passing current-contract canary")
    gpu_rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    total_mib = min(int(row.rsplit(",", 1)[1].strip()) for row in gpu_rows)
    trials = []
    for output in args.benchmark_outputs:
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["contract_sha256"] != sha256_file(CONTRACT) or not summary.get("passed"):
            raise RuntimeError(f"benchmark is incomplete or from another contract: {output}")
        if summary["scope"] != "benchmark":
            raise RuntimeError(f"not a benchmark output: {output}")
        edges = len(summary["edges"])
        fixed_overhead = max(
            0.0,
            float(summary["elapsed_seconds"])
            - float(summary["processing_wall_seconds"])
            - float(summary["model_load_wall_seconds"]),
        )
        load_per_edge = float(summary["model_load_wall_seconds"]) / edges
        processing = POPULATION * len(EDGES) / float(
            summary["global_user_edge_throughput_per_second"]
        )
        projected = fixed_overhead + len(EDGES) * load_per_edge + processing
        trials.append(
            {
                "output": str(output),
                "summary_sha256": sha256_file(summary_path),
                "batch_size_per_rank": int(summary["batch_size_per_rank"]),
                "candidate_chunk": int(summary["candidate_chunk"]),
                "global_user_edge_throughput_per_second": float(
                    summary["global_user_edge_throughput_per_second"]
                ),
                "peak_allocated_mib": float(summary["peak_allocated_mib"]),
                "peak_reserved_mib": float(summary["peak_reserved_mib"]),
                "memory_fraction": float(summary["peak_reserved_mib"]) / total_mib,
                "projected_formal_seconds": projected,
                "eligible": (
                    float(summary["peak_reserved_mib"]) / total_mib
                    <= args.memory_fraction_limit
                ),
            }
        )
    eligible = [trial for trial in trials if trial["eligible"]]
    if not eligible:
        raise RuntimeError("no benchmark setting satisfies the memory safety limit")
    winner = max(
        eligible,
        key=lambda trial: (
            trial["global_user_edge_throughput_per_second"],
            -trial["peak_reserved_mib"],
            trial["batch_size_per_rank"],
        ),
    )
    payload = {
        "status": "medium_insight1_locality_resource_estimate_complete",
        "contract_sha256": sha256_file(CONTRACT),
        "canary_summary_sha256": sha256_file(args.canary / "summary.json"),
        "physical_gpus": gpu_rows,
        "minimum_total_memory_mib": total_mib,
        "memory_fraction_limit": args.memory_fraction_limit,
        "formal_users": POPULATION,
        "formal_edges": list(EDGES),
        "formal_locality_configs": 34,
        "trials": trials,
        "recommended": winner,
        "projected_formal_seconds": winner["projected_formal_seconds"],
        "projected_formal_seconds_with_20_percent_margin": 1.2
        * winner["projected_formal_seconds"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
