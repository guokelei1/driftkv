#!/usr/bin/env python3
"""Materialize the frozen 8L all-state cutover population without labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

import train_p7_theta0 as p7

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/scale_8l_method_v1.yaml"
SOURCE = ROOT / "data/manifests/p9_full_population_v1"
OUTPUT = ROOT / "data/manifests/scale_8l_population_v1"


def validate() -> dict:
    value = yaml.safe_load(CONTRACT.read_text())
    checks = {
        "scale_contract_sha256": ROOT / "configs/contracts/scale_8l_v1.yaml",
        "scale_HS_result_sha256": ROOT / "configs/contracts/scale_8l_hs_result_v1.yaml",
        "frozen_full_stack_sha256": ROOT / "configs/contracts/p10_6_full_stack_freeze_v1.yaml",
        "p9_population_sha256": SOURCE / "materialization_summary.json",
        "state_transition_source_sha256": ROOT / "src/hstu_kvcache/models/state_transition.py",
    }
    for key, path in checks.items():
        if p7.sha256_file(path) != value["inputs"][key]:
            raise RuntimeError(f"8L method input mismatch: {key}")
    return value


def main() -> None:
    contract = validate()
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    edges = []
    for edge in ("edge1", "edge2"):
        source_states = pq.read_table(SOURCE / edge / "states.parquet")
        lengths = pc.min_element_wise(
            source_states["raw_prefix_length"], pa.scalar(int(contract["scope"]["context"]))
        ).cast(pa.int64())
        index = source_states.schema.get_field_index("effective_prefix_length")
        states = source_states.set_column(index, "effective_prefix_length", lengths)
        probes = pq.read_table(SOURCE / edge / "cutover_probes.parquet")
        root = OUTPUT / edge
        root.mkdir(parents=True)
        state_path, probe_path = root / "states.parquet", root / "cutover_probes.parquet"
        pq.write_table(states, state_path, compression="zstd")
        pq.write_table(probes, probe_path, compression="zstd")
        values = lengths.to_numpy(zero_copy_only=False).astype(np.int64)
        meta = {
            "edge": edge, "states": len(states), "probes": len(probes),
            "future_labels_or_requests_materialized": False,
            "effective_prefix_length": {
                "p50": int(np.quantile(values, .5)), "p90": int(np.quantile(values, .9)),
                "p99": int(np.quantile(values, .99)), "max": int(values.max()),
                "saturated_1024": int(np.sum(values == 1024)),
            },
            "states_path": str(state_path.relative_to(ROOT)), "states_sha256": p7.sha256_file(state_path),
            "probes_path": str(probe_path.relative_to(ROOT)), "probes_sha256": p7.sha256_file(probe_path),
        }
        (root / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
        edges.append(meta)
    payload = {
        "status": "scale_8l_all_state_target_free_population_materialized",
        "contract_sha256": p7.sha256_file(CONTRACT), "edges": edges,
    }
    (OUTPUT / "materialization_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
