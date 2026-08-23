#!/usr/bin/env python3
"""Run one cell of the frozen expanded rolling-lineage validation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

import eval_p8_release_raw as p8raw
import eval_p9_materialized_lineage_canary as rolling
import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_5_rolling_validation_matrix_v1.yaml"
OUTPUT_ROOT = ROOT / "results/p9/rolling_validation_raw"


def validate_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "canary_result_contract_sha256": ROOT / "configs/contracts/p9_4_materialized_lineage_canary_result_v1.yaml",
        "canary_result_json_sha256": ROOT / "results/p9/p9_4_materialized_lineage_canary_v1/result.json",
        "state_transition_source_sha256": ROOT / "src/hstu_kvcache/models/state_transition.py",
        "request_local_executor_contract_sha256": ROOT / "configs/contracts/p9_4_executor_contract_v1.yaml",
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P9.5 input hash mismatch: {key}")
    return contract


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for view in ("fidelity", "quality"):
        selected_view = [row for row in rows if row["view"] == view]
        noop_mean = float(np.mean([row["online_JS"] for row in selected_view if row["action"] == "noop"]))
        for action in rolling.ACTIONS:
            selected = [row for row in selected_view if row["action"] == action]
            js = np.asarray([row["online_JS"] for row in selected], dtype=np.float64)
            old_js = np.asarray([row["request_local_JS"] for row in selected], dtype=np.float64)
            entry = {
                "view": view,
                "action": action,
                "requests": len(selected),
                "online_JS_mean": float(js.mean()),
                "online_JS_p95": float(np.quantile(js, 0.95)),
                "online_JS_max": float(js.max()),
                "request_local_JS_mean": float(old_js.mean()),
                "online_to_request_local_ratio": float(js.mean() / old_js.mean()) if old_js.mean() > 0 else None,
                "signed_JS_recovery_fraction_vs_noop": (
                    (noop_mean - float(js.mean())) / noop_mean if noop_mean > 1e-15 else None
                ),
                "current_online_vs_request_local_full_JS_mean": float(np.mean([
                    row["current_online_vs_request_local_full_JS"] for row in selected
                ])),
            }
            if view == "quality":
                entry["action_minus_current_logloss"] = float(np.mean([
                    row["action_logloss"] - row["current_logloss"] for row in selected
                ]))
            output.append(entry)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=("r0", "r1_edge1", "r1_edge2", "r2"), required=True)
    parser.add_argument("--model", choices=("m0_f", "m1"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 37, 71), required=True)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), required=True)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = validate_contract()
    output = args.output or OUTPUT_ROOT / args.release / f"{args.model}_seed{args.seed}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    # Reuse the sealed canary implementation while binding its P8 comparison
    # source to this cell. The rolling-state construction itself is unchanged.
    rolling.request_source = lambda release, view: (
        ROOT / "results/p8/staleness_raw" / release
        / f"{args.model}_seed{args.seed}" / f"F_{view}.parquet"
    )
    device = torch.device(args.device)
    checkpoint = p8raw.TRAIN_ROOT / args.release / f"{args.model}_seed{args.seed}" / "selected.pt"
    current, child = p8raw.load_model(checkpoint, device)
    if not child["admitted"]:
        raise RuntimeError("rolling validation refuses non-admitted edge")
    parent_path = ROOT / child["parent_checkpoint"]
    parent, _ = p8raw.load_model(parent_path, device)
    rows = []
    audits = []
    for view in contract["scope"]["views"]:
        values, audit = rolling.evaluate_view(
            args.release, view, current, parent, device, args.threads, contract
        )
        rows.extend(values)
        audits.append(audit)
    summaries = summarize(rows)
    exact_max = max(row["max_exact_all_vs_current_online_abs_logit"] for row in audits)
    r0_noop = max(
        [row["online_JS_max"] for row in summaries if row["action"] == "noop"], default=0.0
    ) if args.release == "r0" else None
    r0_all = max([row["online_JS_max"] for row in summaries], default=0.0) if args.release == "r0" else None
    gates = contract["per_cell_gates"]
    passed = (
        exact_max <= float(gates["exact_all_vs_current_online_max_abs_logit"])
        and all(row["count_mismatches"] == 0 for row in audits)
        and (r0_noop is None or r0_noop <= float(gates["r0_noop_JS_max"]))
        and (r0_all is None or r0_all <= float(gates["r0_all_action_JS_max"]))
    )
    output.mkdir(parents=True)
    raw_path = output / "rolling_actions.parquet"
    pq.write_table(pa.Table.from_pylist(rows), raw_path, compression="zstd")
    payload = {
        "status": "passed" if passed else "failed",
        "release": args.release,
        "model": args.model,
        "seed": args.seed,
        "contract_hash": p7.sha256_file(CONTRACT),
        "checkpoint_hash": p7.sha256_file(checkpoint),
        "parent_checkpoint_hash": p7.sha256_file(parent_path),
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": p7.sha256_file(raw_path),
        "audits": audits,
        "gate_observations": {
            "exact_max_abs_logit": exact_max,
            "r0_noop_JS_max": r0_noop,
            "r0_all_action_JS_max": r0_all,
        },
        "summaries": summaries,
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"], "release": args.release,
        "model": args.model, "seed": args.seed,
        "gate_observations": payload["gate_observations"],
    }, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
