#!/usr/bin/env python3
"""Run the fixed UID-1930/five-edge K/V response-coupling falsifier.

This is a diagnostic-only exact-state intervention.  It reads the first
frozen discovery UID and held-out odd-32 panel, never reads labels or the
confirmation population, and exposes no rank, threshold, probe, or layer
sweep.  Exact Current K/V are used to isolate reader mechanics, so none of the
reported paths are executable migration actions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from insight_one_locality.common import histories_at_cutover  # noqa: E402

from insight_two.common import (  # noqa: E402
    CONTRACT,
    CUTOVER_DAYS,
    DATASET,
    DAY,
    EDGES,
    HELDOUT_INDICES,
    KNOWN_ITEMS,
    OOV_BUCKETS,
    checkpoint,
    load_frozen_inputs,
    metrics_row,
    score_metrics,
    sha256_file,
    verify_model_payload,
)
from insight_two.kv_response_coupling import (  # noqa: E402
    intervene_kv_response_coupling,
)

UID = 1930
MODES = (
    "reuse",
    "current",
    "key_only",
    "value_only",
    "additive_no_interaction",
    "interaction_only",
)


def verify_preflight_inputs() -> dict[str, str]:
    """Verify every immutable input while allowing only live-plan hash drift."""

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if contract["scope"]["edges"] != list(EDGES):
        raise RuntimeError("contract edge order differs")
    frozen = contract["frozen_inputs"]
    plan = frozen["research_plan"]
    plan_path = ROOT / plan["path"]
    if not plan_path.is_file():
        raise RuntimeError("living research plan is missing")
    plan_actual = sha256_file(plan_path)
    for name, record in frozen.items():
        if name in {"research_plan", "checkpoints"}:
            continue
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen preflight input differs: {name}")
    for version in range(6):
        record = frozen["checkpoints"][f"v{version}"]
        path = ROOT / record["path"]
        if path != checkpoint(version):
            raise RuntimeError(f"checkpoint path differs for v{version}")
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint v{version} differs from contract")
    return {
        "expected_research_plan_sha256": str(plan["sha256"]),
        "actual_research_plan_sha256": plan_actual,
        "research_plan_hash_matches_old_contract": str(
            plan_actual == plan["sha256"]
        ).lower(),
        "immutable_frozen_inputs_verified": "true",
    }


def _method_row(
    edge: str,
    method: str,
    exact: torch.Tensor,
    reuse: torch.Tensor,
    observed: torch.Tensor,
) -> dict[str, Any]:
    return {
        "edge": edge,
        "uid": UID,
        "method": method,
        **metrics_row(score_metrics(exact, reuse, observed)),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    if args.device != "cuda:3":
        raise ValueError("this one-UID preflight is fixed to cuda:3")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)

    verification = verify_preflight_inputs()
    all_uids, all_candidates, _ = load_frozen_inputs()
    if int(all_uids[0]) != UID:
        raise RuntimeError("the first frozen discovery UID is no longer 1930")
    history = load_histories(
        [UID],
        oov_buckets=OOV_BUCKETS,
        dataset_path=DATASET,
        known_vocab_size=KNOWN_ITEMS,
        end_timestamp=(CUTOVER_DAYS[-1] + 1) * DAY,
        threads=4,
    )
    rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for edge_index, edge in enumerate(EDGES):
        parent, parent_payload = load_model(checkpoint(edge_index), device)
        current, current_payload = load_model(checkpoint(edge_index + 1), device)
        verify_model_payload(parent_payload)
        verify_model_payload(current_payload)
        _, items_np, actions_np, deltas_np, query_deltas_np = histories_at_cutover(
            history,
            np.asarray([UID], dtype=np.int64),
            CUTOVER_DAYS[edge_index] * DAY,
        )
        items = torch.as_tensor(items_np, dtype=torch.long, device=device)
        actions = torch.as_tensor(actions_np, dtype=torch.long, device=device)
        deltas = torch.as_tensor(deltas_np, dtype=torch.float32, device=device)
        query_delta = torch.as_tensor(
            query_deltas_np, dtype=torch.float32, device=device
        )
        heldout = torch.as_tensor(
            all_candidates[edge_index, 0:1][:, HELDOUT_INDICES],
            dtype=torch.long,
            device=device,
        )

        parent_cache = parent.compute_kv(items, actions, deltas)
        current_cache = current.compute_kv(items, actions, deltas)
        exact_scores = current.score_cc_reuse(current_cache, heldout, query_delta)
        reuse_scores = current.score_cc_reuse(parent_cache, heldout, query_delta)
        edge_rows: list[dict[str, Any]] = []
        for mode in MODES:
            result = intervene_kv_response_coupling(
                current,
                current_cache,
                parent_cache,
                heldout,
                query_delta,
                mode=mode,
            )
            row = _method_row(
                edge,
                mode,
                exact_scores,
                reuse_scores,
                result.scores,
            )
            rows.append(row)
            edge_rows.append(row)
            if mode == "current":
                layer_rows.extend(
                    {"edge": edge, "uid": UID, **layer}
                    for layer in result.layer_metrics
                )
        print(
            json.dumps(
                {
                    "edge": edge,
                    "probability_recovery": {
                        str(row["method"]): row["probability_gap_recovery"]
                        for row in edge_rows
                    },
                }
            ),
            flush=True,
        )
        del parent, current, parent_cache, current_cache
        torch.cuda.empty_cache()

    by_method: dict[str, list[float]] = {mode: [] for mode in MODES}
    for row in rows:
        by_method[str(row["method"])].append(float(row["probability_gap_recovery"]))
    layer_means: list[dict[str, float | int]] = []
    for layer in range(6):
        selected = [row for row in layer_rows if int(row["layer"]) == layer]
        numeric = {
            name: float(np.mean([float(row[name]) for row in selected]))
            for name in (
                "response_joint_over_component_norm_sum",
                "response_key_value_cosine",
                "response_interaction_over_joint",
                "gated_joint_over_component_norm_sum",
                "gated_key_value_cosine",
                "finite_decomposition_relative_l2_error",
            )
        }
        layer_means.append({"layer": layer, **numeric})
    summary = {
        "status": "nonformal_single_uid_mechanism_falsifier_complete",
        "uid": UID,
        "device": args.device,
        "candidate_source": "heldout_odd32_only",
        "labels_read": False,
        "confirmation_read": False,
        "current_exact_used_by_diagnostic": True,
        "executable_migration_action": False,
        "input_verification": verification,
        "elapsed_seconds": time.perf_counter() - started,
        "probability_recovery_by_method": by_method,
        "probability_recovery_edge_mean": {
            method: float(np.mean(values)) for method, values in by_method.items()
        },
        "current_path_edge_mean_by_layer": layer_means,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
