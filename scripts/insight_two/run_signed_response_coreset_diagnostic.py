#!/usr/bin/env python3
"""Run the fit-free oracle signed-response coreset diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

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
    RESULT_ROOT,
    checkpoint,
    load_frozen_inputs,
    metrics_row,
    score_metrics,
    sha256_file,
    verify_contract as verify_boundary_contract,
    verify_model_payload,
)
from insight_two.signed_response_memory import (  # noqa: E402
    SUPPORTED_SAMPLE_COUNTS,
    build_oracle_signed_response_memory,
    intervene_oracle_signed_response_memory,
)


CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_signed_response_coreset_v1.yaml"
)
OUTPUT_ROOT = RESULT_ROOT / "diagnostic_signed_response_coreset_v1"
CANARY_USERS = 32
DISCOVERY_USERS = 512
FULL_COUNT = 1024


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_contract() -> tuple[dict[str, Any], str]:
    verify_boundary_contract()
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if contract["scope"]["edges"] != list(EDGES):
        raise RuntimeError("signed-response edge order differs")
    if tuple(contract["signed_coreset"]["sample_counts"]) != SUPPORTED_SAMPLE_COUNTS:
        raise RuntimeError("signed-response sample-count grid differs")
    if int(contract["signed_coreset"]["instrumentation_full_count"]) != FULL_COUNT:
        raise RuntimeError("signed-response full instrumentation count differs")
    if tuple(contract["candidate_evaluation"]["heldout_indices"]) != HELDOUT_INDICES:
        raise RuntimeError("signed-response heldout split differs")
    for record in contract["frozen_inputs"].values():
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen signed-response input differs: {path}")
    return contract, sha256_file(CONTRACT)


def require_discovery_gate(contract_hash: str) -> None:
    analysis = OUTPUT_ROOT / "canary/analysis/summary.json"
    estimate = OUTPUT_ROOT / "resource_estimate.json"
    if not analysis.is_file() or not estimate.is_file():
        raise RuntimeError("signed-response discovery requires canary adjudication and resources")
    gate = json.loads(analysis.read_text(encoding="utf-8"))
    resource = json.loads(estimate.read_text(encoding="utf-8"))
    if gate.get("contract_sha256") != contract_hash or not gate.get(
        "discovery_launch_gate_passed"
    ):
        raise RuntimeError("signed-response canary did not pass the launch gate")
    if resource.get("contract_sha256") != contract_hash:
        raise RuntimeError("signed-response resource estimate differs")
    if float(resource.get("estimated_512_user_minutes", 1e9)) > 30:
        raise RuntimeError("signed-response discovery exceeds 30 minutes")


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        raise RuntimeError("signed-response diagnostic requires torchrun")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise RuntimeError(f"signed-response diagnostic requires four ranks, got {world}")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


@torch.inference_mode()
def evaluate_user(
    *,
    uid: int,
    edge: str,
    parent,
    current,
    items: torch.Tensor,
    actions: torch.Tensor,
    deltas: torch.Tensor,
    query_delta: torch.Tensor,
    candidates: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exact_cache = current.compute_kv(items, actions, deltas)
    reuse_cache = parent.compute_kv(items, actions, deltas)
    exact_k_before = exact_cache.k.clone()
    exact_v_before = exact_cache.v.clone()
    reuse_k_before = reuse_cache.k.clone()
    reuse_v_before = reuse_cache.v.clone()
    exact_scores, exact_readout = current.observe_cc_reuse(
        exact_cache, candidates, query_delta
    )
    reuse_scores, _ = current.observe_cc_reuse(reuse_cache, candidates, query_delta)
    records: list[dict[str, Any]] = []
    baseline = score_metrics(exact_scores, reuse_scores, reuse_scores)
    common = {"edge": edge, "uid": uid, "source": "heldout_odd32"}
    records.append(
        {
            **common,
            "method": "Current_Reuse",
            "sample_count": 0,
            "atom_count": 0,
            "stored_scalars": 0,
            "stored_to_full_KV_ratio": 0.0,
            **metrics_row(baseline),
        }
    )

    for sample_count in SUPPORTED_SAMPLE_COUNTS:
        memory = build_oracle_signed_response_memory(
            exact_cache, reuse_cache, sample_count=sample_count
        )
        observed = intervene_oracle_signed_response_memory(
            current, reuse_cache, memory, candidates, query_delta
        )
        metrics = score_metrics(exact_scores, reuse_scores, observed.scores)
        stored_scalars = (
            memory.keys[:, 0].numel() + memory.signed_values[:, 0].numel()
        )
        full_scalars = exact_cache.k[:, 0].numel() + exact_cache.v[:, 0].numel()
        records.append(
            {
                **common,
                "method": f"signed_R{sample_count}",
                "sample_count": sample_count,
                "atom_count": memory.atom_count,
                "stored_scalars": stored_scalars,
                "stored_to_full_KV_ratio": stored_scalars / full_scalars,
                **metrics_row(metrics),
            }
        )

    full_memory = build_oracle_signed_response_memory(
        exact_cache, reuse_cache, sample_count=FULL_COUNT
    )
    full = intervene_oracle_signed_response_memory(
        current, reuse_cache, full_memory, candidates, query_delta
    )
    correctness = {
        "edge": edge,
        "uid": uid,
        "full_R1024_max_abs_logit_error": float(
            torch.max(torch.abs(full.scores.float() - exact_scores.float()))
        ),
        "full_R1024_max_abs_readout_error": float(
            torch.max(torch.abs(full.readout.float() - exact_readout.float()))
        ),
        "finite_all_scores": bool(
            torch.isfinite(exact_scores).all()
            and torch.isfinite(reuse_scores).all()
            and torch.isfinite(full.scores).all()
            and all(
                np.isfinite(record["probability_gap_recovery"])
                for record in records
            )
        ),
        "exact_cache_unchanged": bool(
            torch.equal(exact_cache.k, exact_k_before)
            and torch.equal(exact_cache.v, exact_v_before)
        ),
        "reuse_cache_unchanged": bool(
            torch.equal(reuse_cache.k, reuse_k_before)
            and torch.equal(reuse_cache.v, reuse_v_before)
        ),
    }
    return records, correctness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("canary", "discovery"), required=True)
    args = parser.parse_args()
    rank, local_rank, world = distributed_context()
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(4)
    started = time.perf_counter()

    verification: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            _, contract_hash = verify_contract()
            if args.scope == "discovery":
                require_discovery_gate(contract_hash)
            verification[0] = {"ok": True, "contract_sha256": contract_hash}
        except BaseException as error:
            verification[0] = {"ok": False, "error": repr(error)}
    dist.broadcast_object_list(verification, src=0)
    assert verification[0] is not None
    if not verification[0]["ok"]:
        raise RuntimeError(f"contract verification failed: {verification[0]['error']}")
    contract_hash = str(verification[0]["contract_sha256"])

    all_uids, all_candidates, _ = load_frozen_inputs()
    user_count = CANARY_USERS if args.scope == "canary" else DISCOVERY_USERS
    selected_indices = np.arange(user_count, dtype=np.int64)
    local_indices = selected_indices[rank::world]
    local_uids = all_uids[local_indices]
    output = OUTPUT_ROOT / args.scope
    partial = output.with_name(output.name + ".partial")
    if rank == 0:
        if output.exists() or partial.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        partial.mkdir(parents=True)
        atomic_json(
            partial / "configuration.json",
            {
                "contract_sha256": contract_hash,
                "scope": args.scope,
                "users": user_count,
                "edges": list(EDGES),
                "sample_counts": list(SUPPORTED_SAMPLE_COUNTS),
                "full_instrumentation_count": FULL_COUNT,
                "labels_read": False,
                "construction_candidates": None,
                "oracle_exact_cache_used": True,
            },
        )
    dist.barrier()
    rank_output = partial / f"rank{rank}"
    rank_output.mkdir()

    history = load_histories(
        local_uids.tolist(),
        oov_buckets=OOV_BUCKETS,
        dataset_path=DATASET,
        known_vocab_size=KNOWN_ITEMS,
        end_timestamp=(CUTOVER_DAYS[-1] + 1) * DAY,
        threads=8,
    )
    metric_records: list[dict[str, Any]] = []
    correctness_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    peak_allocated_mib = 0.0
    peak_reserved_mib = 0.0

    for edge_index, edge in enumerate(EDGES):
        print(json.dumps({"phase": "edge_start", "rank": rank, "edge": edge}), flush=True)
        edge_started = time.perf_counter()
        parent, parent_payload = load_model(checkpoint(edge_index), device)
        current, current_payload = load_model(checkpoint(edge_index + 1), device)
        verify_model_payload(parent_payload)
        verify_model_payload(current_payload)
        _, items_np, actions_np, deltas_np, query_np = histories_at_cutover(
            history, local_uids, CUTOVER_DAYS[edge_index] * DAY
        )
        candidate_np = all_candidates[edge_index, local_indices]
        torch.cuda.reset_peak_memory_stats(device)

        for local_offset, uid in enumerate(local_uids):
            items = torch.as_tensor(
                items_np[local_offset : local_offset + 1], dtype=torch.long, device=device
            )
            actions = torch.as_tensor(
                actions_np[local_offset : local_offset + 1], dtype=torch.long, device=device
            )
            deltas = torch.as_tensor(
                deltas_np[local_offset : local_offset + 1], dtype=torch.float32, device=device
            )
            query_delta = torch.as_tensor(
                query_np[local_offset : local_offset + 1], dtype=torch.float32, device=device
            )
            candidates = torch.as_tensor(
                candidate_np[local_offset : local_offset + 1, HELDOUT_INDICES],
                dtype=torch.long,
                device=device,
            )
            records, correctness = evaluate_user(
                uid=int(uid),
                edge=edge,
                parent=parent,
                current=current,
                items=items,
                actions=actions,
                deltas=deltas,
                query_delta=query_delta,
                candidates=candidates,
            )
            metric_records.extend(records)
            correctness_records.append(correctness)
            peak_allocated_mib = max(
                peak_allocated_mib, torch.cuda.max_memory_allocated(device) / (1 << 20)
            )
            peak_reserved_mib = max(
                peak_reserved_mib, torch.cuda.max_memory_reserved(device) / (1 << 20)
            )

        edge_seconds = time.perf_counter() - edge_started
        edge_records.append({"edge": edge, "users": len(local_uids), "seconds": edge_seconds})
        print(
            json.dumps(
                {
                    "phase": "edge_complete",
                    "rank": rank,
                    "edge": edge,
                    "users": len(local_uids),
                    "seconds": edge_seconds,
                }
            ),
            flush=True,
        )
        del parent, current, parent_payload, current_payload
        torch.cuda.empty_cache()

    metric_path = rank_output / "metrics.parquet"
    correctness_path = rank_output / "correctness.parquet"
    pd.DataFrame(metric_records).to_parquet(metric_path, index=False)
    pd.DataFrame(correctness_records).to_parquet(correctness_path, index=False)
    atomic_json(
        rank_output / "summary.json",
        {
            "rank": rank,
            "uids": local_uids.tolist(),
            "edge_records": edge_records,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_mib": peak_allocated_mib,
            "peak_reserved_mib": peak_reserved_mib,
            "labels_read": False,
        },
    )
    dist.barrier()

    if rank == 0:
        metrics = pd.concat(
            [pd.read_parquet(partial / f"rank{i}/metrics.parquet") for i in range(world)],
            ignore_index=True,
        )
        correctness = pd.concat(
            [
                pd.read_parquet(partial / f"rank{i}/correctness.parquet")
                for i in range(world)
            ],
            ignore_index=True,
        )
        expected_metrics = user_count * len(EDGES) * (1 + len(SUPPORTED_SAMPLE_COUNTS))
        expected_correctness = user_count * len(EDGES)
        if len(metrics) != expected_metrics or len(correctness) != expected_correctness:
            raise RuntimeError("signed-response raw row count differs")
        if metrics[["edge", "uid", "method"]].duplicated().any():
            raise RuntimeError("duplicate signed-response metric key")
        maximum_logit_error = float(correctness.full_R1024_max_abs_logit_error.max())
        maximum_readout_error = float(correctness.full_R1024_max_abs_readout_error.max())
        passed = bool(
            maximum_logit_error <= 2e-5
            and maximum_readout_error <= 2e-5
            and correctness.finite_all_scores.all()
            and correctness.exact_cache_unchanged.all()
            and correctness.reuse_cache_unchanged.all()
        )
        metric_combined = partial / "metrics.parquet"
        correctness_combined = partial / "correctness.parquet"
        metrics.to_parquet(metric_combined, index=False)
        correctness.to_parquet(correctness_combined, index=False)
        rank_summaries = [
            json.loads((partial / f"rank{i}/summary.json").read_text(encoding="utf-8"))
            for i in range(world)
        ]
        artifacts = {
            path.name: {
                "rows": len(metrics) if path == metric_combined else len(correctness),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (metric_combined, correctness_combined)
        }
        summary = {
            "status": "signed_response_coreset_instrumentation_passed" if passed else "signed_response_coreset_instrumentation_failed",
            "passed": passed,
            "contract_sha256": contract_hash,
            "scope": args.scope,
            "users": user_count,
            "edges": list(EDGES),
            "labels_read": False,
            "oracle_exact_cache_used": True,
            "all_numeric_rows_finite": bool(correctness.finite_all_scores.all()),
            "maximum_full_R1024_logit_error": maximum_logit_error,
            "maximum_full_R1024_readout_error": maximum_readout_error,
            "elapsed_seconds": max(float(row["elapsed_seconds"]) for row in rank_summaries),
            "peak_allocated_mib": max(float(row["peak_allocated_mib"]) for row in rank_summaries),
            "peak_reserved_mib": max(float(row["peak_reserved_mib"]) for row in rank_summaries),
            "artifacts": artifacts,
        }
        atomic_json(partial / "summary.json", summary)
        partial.replace(output)
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
