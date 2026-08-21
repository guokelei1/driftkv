#!/usr/bin/env python3
"""Compute target-independent oracle no-op/exact fidelity-cost frontiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EPSILONS = (0.0, 0.1, 0.2, 0.5, 0.9)
USER_BUDGETS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
RANDOM_REPEATS = 100


def arrays(records: list[dict]) -> dict[str, np.ndarray]:
    return {
        "loss": np.asarray([row["top10_overlap_loss"] for row in records], dtype=np.float64),
        "nrms": np.asarray([row["normalized_score_rms"] for row in records], dtype=np.float64),
        "js": np.asarray([row["js_divergence"] for row in records], dtype=np.float64),
        "inversion": np.asarray([row["pairwise_inversion_rate"] for row in records], dtype=np.float64),
        "full_ms": np.asarray([row["full_recompute_latency_ms"] for row in records], dtype=np.float64),
        "reuse_ms": np.asarray([row["reuse_append_latency_ms"] for row in records], dtype=np.float64),
        "full_tokens": np.asarray([row["recomputed_tokens"] for row in records], dtype=np.float64),
        "reuse_tokens": np.asarray([row["reuse_append_tokens"] for row in records], dtype=np.float64),
        "kv_bytes": np.asarray([row["prefix_kv_read_bytes"] for row in records], dtype=np.float64),
        "prefix_length": np.asarray([row["effective_prefix_length"] for row in records], dtype=np.float64),
        "activity": np.asarray([row["raw_prefix_length"] for row in records], dtype=np.float64),
    }


def policy_metrics(values: dict[str, np.ndarray], exact: np.ndarray, epsilon: float) -> dict:
    reuse = ~exact
    n = len(exact)
    full_ms = values["full_ms"].sum()
    full_tokens = values["full_tokens"].sum()
    return {
        "exact_user_fraction": float(exact.mean()),
        "exact_recomputed_token_fraction": float(values["full_tokens"][exact].sum() / max(full_tokens, 1.0)),
        "measured_gpu_time_fraction_of_exact_all": float(
            (values["full_ms"][exact].sum() + values["reuse_ms"][reuse].sum()) / max(full_ms, 1e-12)
        ),
        "prefix_kv_read_bytes": float(values["kv_bytes"][reuse].sum()),
        "reuse_append_tokens": float(values["reuse_tokens"][reuse].sum()),
        "residual_top10_overlap_loss_mean": float((values["loss"] * reuse).mean()),
        "residual_top10_overlap_loss_p95": float(np.percentile(values["loss"][reuse], 95)) if reuse.any() else 0.0,
        "residual_top10_changed_fraction": float(((values["loss"] > 0) & reuse).mean()),
        "residual_slo_violation_fraction": float(((values["loss"] > epsilon) & reuse).mean()),
        "residual_normalized_score_rms_mean": float((values["nrms"] * reuse).mean()),
        "residual_js_divergence_mean": float((values["js"] * reuse).mean()),
        "residual_pairwise_inversion_mean": float((values["inversion"] * reuse).mean()),
        "users": n,
    }


def mask_by_count(order: np.ndarray, count: int) -> np.ndarray:
    exact = np.zeros(len(order), dtype=bool)
    exact[order[:count]] = True
    return exact


def mean_metrics(metrics: list[dict]) -> dict:
    keys = metrics[0].keys()
    return {key: float(np.mean([metric[key] for metric in metrics])) for key in keys}


def frontier_for_edge(records: list[dict], seed: int) -> dict:
    values = arrays(records)
    n = len(records)
    loss_order = np.argsort(-values["loss"], kind="stable")
    prefix_order = np.argsort(-values["prefix_length"], kind="stable")
    activity_order = np.argsort(-values["activity"], kind="stable")
    rng = np.random.default_rng(seed)

    slo = []
    for epsilon in EPSILONS:
        oracle = values["loss"] > epsilon
        count = int(oracle.sum())
        version_gate = np.full(n, bool(np.any(values["loss"] > epsilon)))
        random_metrics = [
            policy_metrics(values, mask_by_count(rng.permutation(n), count), epsilon)
            for _ in range(RANDOM_REPEATS)
        ]
        slo.append({
            "epsilon_top10_overlap_loss": epsilon,
            "reuse_all": policy_metrics(values, np.zeros(n, dtype=bool), epsilon),
            "exact_all": policy_metrics(values, np.ones(n, dtype=bool), epsilon),
            "version_level_uniform_gate": policy_metrics(values, version_gate, epsilon),
            "oracle_user_level_gate": policy_metrics(values, oracle, epsilon),
            "random_exact_same_user_count_mean": mean_metrics(random_metrics),
            "longest_prefix_first_same_user_count": policy_metrics(
                values, mask_by_count(prefix_order, count), epsilon
            ),
            "most_active_same_user_count": policy_metrics(
                values, mask_by_count(activity_order, count), epsilon
            ),
        })

    budget = []
    for user_fraction in USER_BUDGETS:
        count = int(round(user_fraction * n))
        random_metrics = [
            policy_metrics(values, mask_by_count(rng.permutation(n), count), epsilon=0.0)
            for _ in range(RANDOM_REPEATS)
        ]
        budget.append({
            "requested_exact_user_fraction": user_fraction,
            "oracle_highest_top10_loss": policy_metrics(values, mask_by_count(loss_order, count), epsilon=0.0),
            "random_exact_mean": mean_metrics(random_metrics),
            "longest_prefix_first": policy_metrics(values, mask_by_count(prefix_order, count), epsilon=0.0),
            "most_active": policy_metrics(values, mask_by_count(activity_order, count), epsilon=0.0),
        })
    return {
        "fidelity_endpoint": "top10_overlap_loss = 1 - overlap(CurrentFull, Reuse)/10",
        "target_independent": True,
        "slo_sweep": slo,
        "budget_frontier": budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("results/data_audit/yambda50m_v2/compatibility_profile_batchfix_v3_screen.json"),
    )
    parser.add_argument(
        "--numeric-floor", type=Path,
        default=Path("results/data_audit/yambda50m_v2/cache_identity_numeric_floor.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/data_audit/yambda50m_v2/noop_exact_oracle_frontier_v1.json"),
    )
    args = parser.parse_args()
    profile = json.loads(args.input.read_text())
    numeric = json.loads(args.numeric_floor.read_text())
    result = {
        "status": "oracle_no_op_exact_frontier_development_only",
        "source_profile": str(args.input),
        "fidelity_endpoint": "target-independent Top-10 overlap loss",
        "target_injected": False,
        "controller_or_feature_training": False,
        "numeric_floor": {
            "identity_score_rms": numeric["identity_transition"]["score_rms"],
            "identity_hidden_rms": numeric["identity_transition"]["hidden_rms"],
            "identity_prefix_k_rms": numeric["identity_transition"]["prefix"]["k_rms"],
        },
        "edges": {
            "theta0_to_theta1": frontier_for_edge(profile["edge_theta0_theta1"]["compatibility_records"], seed=37),
            "theta1_to_theta2": frontier_for_edge(profile["edge_theta1_theta2"]["compatibility_records"], seed=71),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "edge1_users": len(profile["edge_theta0_theta1"]["compatibility_records"]),
        "edge2_users": len(profile["edge_theta1_theta2"]["compatibility_records"]),
    }, indent=2))


if __name__ == "__main__":
    main()
