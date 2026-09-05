#!/usr/bin/env python3
"""Run the fixed UID1930/five-edge producer-reader commutator diagnostic.

The run uses four exact endpoints and is therefore an oracle structure test,
not a migration action.  It reads only the first frozen discovery user and
the frozen held-out odd-32 candidate panel.  No labels, confirmation users,
rank, probe, threshold, or fitted map are available to the run.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from insight_one_locality.common import histories_at_cutover  # noqa: E402

from insight_two.common import (  # noqa: E402
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
    verify_model_payload,
)
from insight_two.producer_reader_commutator import (  # noqa: E402
    ReaderProducerTrace,
    commuted_endpoint,
    finite_difference_diagnostic,
    trace_reader_producer,
)
from insight_two.run_kv_response_coupling_preflight import (  # noqa: E402
    verify_preflight_inputs,
)

UID = 1930
PATH_NAMES = ("CC", "CP", "PC", "PP")


def _diagnostic_row(
    *,
    edge: str,
    stage: str,
    layer: int | None,
    current_current: torch.Tensor,
    current_parent: torch.Tensor,
    parent_current: torch.Tensor,
    parent_parent: torch.Tensor,
    coordinate_semantics: str,
) -> dict[str, Any]:
    diagnostic = finite_difference_diagnostic(
        current_current,
        current_parent,
        parent_current,
        parent_parent,
    )
    centered = tuple(
        values.float() - values.float().mean(dim=1, keepdim=True)
        for values in (
            current_current,
            current_parent,
            parent_current,
            parent_parent,
        )
    )
    centered_diagnostic = finite_difference_diagnostic(*centered)
    return {
        "edge": edge,
        "uid": UID,
        "stage": stage,
        "layer": layer,
        "coordinate_semantics": coordinate_semantics,
        "mixed_over_current_state_l2": diagnostic.mixed_over_current_state_l2,
        "l2_recovery": diagnostic.l2_recovery,
        "parent_over_current_state_l2": diagnostic.parent_over_current_state_l2,
        "state_effect_cosine": diagnostic.state_effect_cosine,
        "mixed_l2": float(torch.linalg.vector_norm(diagnostic.mixed.float())),
        "current_state_effect_l2": float(
            torch.linalg.vector_norm(diagnostic.current_state_effect.float())
        ),
        "parent_state_effect_l2": float(
            torch.linalg.vector_norm(diagnostic.parent_state_effect.float())
        ),
        "candidate_centered_mixed_over_current_state_l2": (
            centered_diagnostic.mixed_over_current_state_l2
        ),
        "candidate_centered_l2_recovery": centered_diagnostic.l2_recovery,
        "candidate_centered_state_effect_cosine": (
            centered_diagnostic.state_effect_cosine
        ),
    }


def _score_rows(
    edge: str,
    traces: dict[str, ReaderProducerTrace],
) -> list[dict[str, Any]]:
    exact = traces["CC"].scores
    reuse = traces["CP"].scores
    commuted = commuted_endpoint(
        traces["CP"].scores,
        traces["PC"].scores,
        traces["PP"].scores,
    )
    methods = {
        "Current_reader_Parent_cache_reuse": reuse,
        "commuted_exact_cross_endpoint": commuted,
        "Parent_reader_Current_cache_reverse_cross": traces["PC"].scores,
        "Parent_reader_Parent_cache": traces["PP"].scores,
    }
    return [
        {
            "edge": edge,
            "uid": UID,
            "method": method,
            **metrics_row(score_metrics(exact, reuse, values)),
        }
        for method, values in methods.items()
    ]


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
    histories = load_histories(
        [UID],
        oov_buckets=OOV_BUCKETS,
        dataset_path=DATASET,
        known_vocab_size=KNOWN_ITEMS,
        end_timestamp=(CUTOVER_DAYS[-1] + 1) * DAY,
        threads=4,
    )
    score_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for edge_index, edge in enumerate(EDGES):
        parent, parent_payload = load_model(checkpoint(edge_index), device)
        current, current_payload = load_model(checkpoint(edge_index + 1), device)
        verify_model_payload(parent_payload)
        verify_model_payload(current_payload)
        _, items_np, actions_np, deltas_np, query_deltas_np = histories_at_cutover(
            histories,
            np.asarray([UID], dtype=np.int64),
            CUTOVER_DAYS[edge_index] * DAY,
        )
        items = torch.as_tensor(items_np, dtype=torch.long, device=device)
        actions = torch.as_tensor(actions_np, dtype=torch.long, device=device)
        deltas = torch.as_tensor(deltas_np, dtype=torch.float32, device=device)
        query_delta = torch.as_tensor(
            query_deltas_np,
            dtype=torch.float32,
            device=device,
        )
        heldout = torch.as_tensor(
            all_candidates[edge_index, 0:1][:, HELDOUT_INDICES],
            dtype=torch.long,
            device=device,
        )

        parent_cache = parent.compute_kv(items, actions, deltas)
        current_cache = current.compute_kv(items, actions, deltas)
        traces = {
            "CC": trace_reader_producer(
                current,
                current_cache,
                heldout,
                query_delta,
            ),
            "CP": trace_reader_producer(
                current,
                parent_cache,
                heldout,
                query_delta,
            ),
            "PC": trace_reader_producer(
                parent,
                current_cache,
                heldout,
                query_delta,
            ),
            "PP": trace_reader_producer(
                parent,
                parent_cache,
                heldout,
                query_delta,
            ),
        }
        edge_score_rows = _score_rows(edge, traces)
        score_rows.extend(edge_score_rows)
        stage_rows.append(
            _diagnostic_row(
                edge=edge,
                stage="S7_final_score",
                layer=None,
                current_current=traces["CC"].scores,
                current_parent=traces["CP"].scores,
                parent_current=traces["PC"].scores,
                parent_parent=traces["PP"].scores,
                coordinate_semantics=(
                    "shared task scalar; within-reader finite differences are "
                    "comparable, though release calibration may differ"
                ),
            )
        )
        stage_rows.append(
            _diagnostic_row(
                edge=edge,
                stage="S7_final_readout",
                layer=None,
                current_current=traces["CC"].readout,
                current_parent=traces["CP"].readout,
                parent_current=traces["PC"].readout,
                parent_parent=traces["PP"].readout,
                coordinate_semantics=(
                    "shape-compatible but cross-reader hidden bases are not "
                    "identified; raw commutator is descriptive only"
                ),
            )
        )
        for layer in range(len(current.blocks)):
            stage_rows.append(
                _diagnostic_row(
                    edge=edge,
                    stage="S4_aggregated_context",
                    layer=layer,
                    current_current=traces["CC"].layer_s4[layer],
                    current_parent=traces["CP"].layer_s4[layer],
                    parent_current=traces["PC"].layer_s4[layer],
                    parent_parent=traces["PP"].layer_s4[layer],
                    coordinate_semantics=(
                        "shape-compatible head tensor but Parent and Current "
                        "head bases are not identified; descriptive only"
                    ),
                )
            )

        commuted_row = next(
            row
            for row in edge_score_rows
            if row["method"] == "commuted_exact_cross_endpoint"
        )
        score_diagnostic = stage_rows[-(len(current.blocks) + 2)]
        print(
            json.dumps(
                {
                    "edge": edge,
                    "commuted_probability_recovery": commuted_row[
                        "probability_gap_recovery"
                    ],
                    "score_mixed_over_state_l2": score_diagnostic[
                        "mixed_over_current_state_l2"
                    ],
                    "score_state_effect_cosine": score_diagnostic[
                        "state_effect_cosine"
                    ],
                }
            ),
            flush=True,
        )
        del parent, current, parent_cache, current_cache, traces
        torch.cuda.empty_cache()

    recoveries = [
        float(row["probability_gap_recovery"])
        for row in score_rows
        if row["method"] == "commuted_exact_cross_endpoint"
    ]
    score_diagnostics = [
        row for row in stage_rows if row["stage"] == "S7_final_score"
    ]
    readout_diagnostics = [
        row for row in stage_rows if row["stage"] == "S7_final_readout"
    ]
    s4_diagnostics = [
        row for row in stage_rows if row["stage"] == "S4_aggregated_context"
    ]
    supports_approximate_commutation = (
        float(np.mean(recoveries)) >= 0.80
        and sum(value >= 0.80 for value in recoveries) >= 4
    )
    summary = {
        "status": "nonformal_single_uid_oracle_structure_complete",
        "uid": UID,
        "device": args.device,
        "candidate_source": "heldout_odd32_only",
        "labels_read": False,
        "confirmation_read": False,
        "current_exact_cache_read": True,
        "executable_migration_action": False,
        "per_candidate_score_mixing": True,
        "each_reader_uses_own_query_embedding": True,
        "input_verification": verification,
        "configuration": {
            "paths": list(PATH_NAMES),
            "no_parameter_sweep": True,
            "no_probe": True,
            "no_fit_or_mapping": True,
        },
        "commuted_probability_recovery_by_edge": recoveries,
        "commuted_probability_recovery_edge_mean": float(np.mean(recoveries)),
        "commuted_edges_at_or_above_point8": sum(
            value >= 0.80 for value in recoveries
        ),
        "score_mixed_over_state_l2_by_edge": [
            float(row["mixed_over_current_state_l2"])
            for row in score_diagnostics
        ],
        "score_state_effect_cosine_by_edge": [
            float(row["state_effect_cosine"]) for row in score_diagnostics
        ],
        "score_candidate_centered_l2_recovery_by_edge": [
            float(row["candidate_centered_l2_recovery"])
            for row in score_diagnostics
        ],
        "readout_l2_recovery_by_edge": [
            float(row["l2_recovery"]) for row in readout_diagnostics
        ],
        "readout_candidate_centered_l2_recovery_by_edge": [
            float(row["candidate_centered_l2_recovery"])
            for row in readout_diagnostics
        ],
        "s4_l2_recovery_by_layer_then_edge": {
            str(layer): [
                float(row["l2_recovery"])
                for row in s4_diagnostics
                if int(row["layer"]) == layer
            ]
            for layer in range(6)
        },
        "s4_candidate_centered_l2_recovery_by_layer_then_edge": {
            str(layer): [
                float(row["candidate_centered_l2_recovery"])
                for row in s4_diagnostics
                if int(row["layer"]) == layer
            ]
            for layer in range(6)
        },
        "oracle_hypothesis_gate": {
            "rule": "mean recovery >=.80 and at least 4/5 edges >=.80",
            "passed": supports_approximate_commutation,
        },
        "design_adjudication": {
            "decision": "reject_as_Design_1",
            "reasons": [
                "F(P,C) requires exact Current K/V, the unavailable target state",
                "the correction requires a Parent reader call per candidate and score mixing",
                "persisting probe evaluations would collapse to fitted functional mapping",
                "commutation, even if accurate, supplies no no-target sub-20-percent constructor",
            ],
        },
        "score_rows": score_rows,
        "stage_rows": stage_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
