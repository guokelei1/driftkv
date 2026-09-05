#!/usr/bin/env python3
"""Evaluate cutover-time-aligned, candidate-independent S4 probes."""

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
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from insight_one_locality.common import histories_at_cutover  # noqa: E402
from insight_two.common import (  # noqa: E402
    ANCHOR_INDICES,
    CUTOVER_DAYS,
    DATASET,
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
from insight_two.functional_probe_estimator import functional_probe_cost  # noqa: E402
from insight_two.time_aligned_functional_probe import (  # noqa: E402
    generate_time_aligned_probe,
)
from one_release_refinement import parameter_cast_maps  # noqa: E402
from pro_lazy_reader import build_parent_conditioned_carriers  # noqa: E402
from reader_compatibility_correction import _stage_path, intervene_reader_correction  # noqa: E402


CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_time_aligned_probe_v1.yaml"
)
OUTPUT_ROOT = RESULT_ROOT / "estimator_time_aligned_probe_v1"
CARRIERS = (32, 64)
CANARY_USERS = 32
DISCOVERY_USERS = 512
REPLAY_TOLERANCE = 2e-5


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def cost(carriers: int) -> dict[str, int | float | list[int] | str]:
    return functional_probe_cost(
        layers=6,
        hidden=192,
        heads=6,
        context=1024,
        repair_evidence=128,
        carriers=carriers,
        probes=1,
    )


def verify_contract() -> tuple[dict[str, Any], str]:
    verify_boundary_contract()
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if contract["mechanism"]["carrier_axis"] != list(CARRIERS):
        raise RuntimeError("time-aligned carrier axis differs")
    for record in contract["frozen_inputs"].values():
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen time-aligned input differs: {path}")
    for carriers in CARRIERS:
        observed = float(cost(carriers)["over_full_fraction"])
        expected = float(contract["cost"]["fraction_by_carriers"][carriers])
        if observed != expected or observed > 0.20:
            raise RuntimeError("time-aligned cost differs")
    return contract, sha256_file(CONTRACT)


def require_discovery_gate(contract_hash: str) -> None:
    for relative in ("canary/summary.json", "canary/analysis/summary.json", "resource_estimate.json"):
        if not (OUTPUT_ROOT / relative).is_file():
            raise RuntimeError(f"discovery gate missing {relative}")
    run = json.loads((OUTPUT_ROOT / "canary/summary.json").read_text(encoding="utf-8"))
    analysis = json.loads(
        (OUTPUT_ROOT / "canary/analysis/summary.json").read_text(encoding="utf-8")
    )
    resource = json.loads(
        (OUTPUT_ROOT / "resource_estimate.json").read_text(encoding="utf-8")
    )
    if not run.get("passed") or run.get("contract_sha256") != contract_hash:
        raise RuntimeError("time-aligned canary did not pass")
    if analysis.get("estimator_gate") != "pass":
        raise RuntimeError("time-aligned estimator gate did not pass")
    if resource.get("contract_sha256") != contract_hash:
        raise RuntimeError("time-aligned resource estimate differs")
    if resource.get("estimated_512_user_minutes", 1e9) > 30:
        raise RuntimeError("time-aligned discovery exceeds 30 minutes")


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        raise RuntimeError("time-aligned probe requires torchrun")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise RuntimeError(f"time-aligned probe requires four ranks, got {world}")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def flatten(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([value.float().flatten(1) for value in values], dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("canary", "discovery"), default="canary")
    args = parser.parse_args()
    users = CANARY_USERS if args.scope == "canary" else DISCOVERY_USERS
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
        raise RuntimeError(f"contract gate failed: {verification[0]['error']}")
    contract_hash = str(verification[0]["contract_sha256"])

    all_uids, all_candidates, _ = load_frozen_inputs()
    selected_indices = np.arange(users, dtype=np.int64)
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
                "users": users,
                "edges": list(EDGES),
                "carrier_axis": list(CARRIERS),
                "probes": 1,
                "probe_item": "latest_strictly_pre_cutover_history_item",
                "probe_time": "known_cutover_minus_latest_history_timestamp",
                "labels_read": False,
                "Current_Exact_used_by_estimator": False,
            },
        )
        atomic_json(
            partial / "theoretical_compute.json",
            {str(carriers): cost(carriers) for carriers in CARRIERS},
        )
    dist.barrier()
    rank_output = partial / f"rank{rank}"
    rank_output.mkdir()

    history_started = time.perf_counter()
    history = load_histories(
        local_uids.tolist(),
        oov_buckets=OOV_BUCKETS,
        dataset_path=DATASET,
        known_vocab_size=KNOWN_ITEMS,
        end_timestamp=(CUTOVER_DAYS[-1] + 1) * 86_400,
        threads=8,
    )
    history_seconds = time.perf_counter() - history_started
    rows: list[dict[str, Any]] = []
    edge_records = []
    maximum_replay = 0.0
    peak_allocated = 0.0
    peak_reserved = 0.0

    for edge_index, edge in enumerate(EDGES):
        print(json.dumps({"phase": "edge_start", "rank": rank, "edge": edge}), flush=True)
        edge_started = time.perf_counter()
        parent, parent_payload = load_model(checkpoint(edge_index), device)
        current, current_payload = load_model(checkpoint(edge_index + 1), device)
        verify_model_payload(parent_payload)
        verify_model_payload(current_payload)
        maps = parameter_cast_maps(parent, current)
        _, items_np, actions_np, deltas_np, query_deltas_np = histories_at_cutover(
            history, local_uids, CUTOVER_DAYS[edge_index] * 86_400
        )
        candidates_np = all_candidates[edge_index, local_indices]
        torch.cuda.reset_peak_memory_stats(device)
        for offset, uid in enumerate(local_uids):
            items = torch.as_tensor(items_np[offset : offset + 1], dtype=torch.long, device=device)
            actions = torch.as_tensor(actions_np[offset : offset + 1], dtype=torch.long, device=device)
            deltas = torch.as_tensor(deltas_np[offset : offset + 1], dtype=torch.float32, device=device)
            query_delta = torch.as_tensor(
                query_deltas_np[offset : offset + 1], dtype=torch.float32, device=device
            )
            anchors = torch.as_tensor(
                candidates_np[offset : offset + 1, ANCHOR_INDICES],
                dtype=torch.long,
                device=device,
            )
            heldout = torch.as_tensor(
                candidates_np[offset : offset + 1, HELDOUT_INDICES],
                dtype=torch.long,
                device=device,
            )
            parent_cache = parent.compute_kv(items, actions, deltas)
            current_cache = current.compute_kv(items, actions, deltas)
            exact_scores = current.score_cc_reuse(current_cache, heldout, query_delta)
            reuse_scores = current.score_cc_reuse(parent_cache, heldout, query_delta)
            _, oracle, _ = _stage_path(
                current,
                current_cache,
                parent_cache,
                anchors,
                query_delta,
                stage="av_aggregation",
                mode="shared",
            )
            oracle_flat = flatten(oracle)
            for carrier_count in CARRIERS:
                carriers, layout = build_parent_conditioned_carriers(
                    parent_cache=parent_cache,
                    current=current,
                    item_ids=items,
                    behaviors=actions,
                    time_deltas=deltas,
                    repair_width=128,
                    carrier_count=carrier_count,
                )
                estimate = generate_time_aligned_probe(
                    current,
                    parent_cache,
                    carriers,
                    maps,
                    items[:, -1],
                    query_delta,
                    old_positions=layout.old_positions,
                )
                maximum_replay = max(maximum_replay, estimate.replay_max_abs_error)
                observed, _ = intervene_reader_correction(
                    current,
                    parent_cache,
                    heldout,
                    query_delta,
                    stage="av_aggregation",
                    corrections=estimate.corrections,
                )
                estimated_flat = flatten(estimate.corrections)
                oracle_norm = oracle_flat.norm(dim=1).clamp_min(1e-20)
                rows.append(
                    {
                        "edge": edge,
                        "uid": int(uid),
                        "carriers": carrier_count,
                        "probes": 1,
                        "query_time_delta": float(query_delta[0]),
                        "theoretical_compute_fraction": float(
                            cost(carrier_count)["over_full_fraction"]
                        ),
                        "labels_read": False,
                        "Current_Exact_used_by_estimator": False,
                        **metrics_row(score_metrics(exact_scores, reuse_scores, observed)),
                        "correction_cosine_to_oracle": float(
                            F.cosine_similarity(estimated_flat, oracle_flat, dim=1)[0]
                        ),
                        "correction_norm_ratio_to_oracle": float(
                            (estimated_flat.norm(dim=1) / oracle_norm)[0]
                        ),
                        "correction_relative_l2_to_oracle": float(
                            ((estimated_flat - oracle_flat).norm(dim=1) / oracle_norm)[0]
                        ),
                    }
                )
            peak_allocated = max(
                peak_allocated, torch.cuda.max_memory_allocated(device) / (1 << 20)
            )
            peak_reserved = max(
                peak_reserved, torch.cuda.max_memory_reserved(device) / (1 << 20)
            )
        edge_records.append(
            {"edge": edge, "users": len(local_uids), "seconds": time.perf_counter() - edge_started}
        )
        print(json.dumps({"phase": "edge_complete", "rank": rank, **edge_records[-1]}), flush=True)
        del parent, current, maps
        torch.cuda.empty_cache()

    path = rank_output / "score_records.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    rank_summary = {
        "status": "time_aligned_probe_rank_complete",
        "rank": rank,
        "uids": local_uids.tolist(),
        "labels_read": False,
        "elapsed_seconds": time.perf_counter() - started,
        "history_seconds": history_seconds,
        "maximum_probe_replay_abs_error": maximum_replay,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "edges": edge_records,
        "artifact": {"sha256": sha256_file(path), "bytes": path.stat().st_size},
    }
    atomic_json(rank_output / "summary.json", rank_summary)
    dist.barrier()
    if rank == 0:
        summaries = [
            json.loads((partial / f"rank{value}/summary.json").read_text(encoding="utf-8"))
            for value in range(world)
        ]
        frame = pd.concat(
            [pd.read_parquet(partial / f"rank{value}/score_records.parquet") for value in range(4)],
            ignore_index=True,
        )
        uids = [uid for summary in summaries for uid in summary["uids"]]
        maximum = max(float(summary["maximum_probe_replay_abs_error"]) for summary in summaries)
        passed = (
            len(uids) == len(set(uids)) == users
            and len(frame) == users * len(EDGES) * len(CARRIERS)
            and bool(np.isfinite(frame.select_dtypes(include=[np.number])).all().all())
            and maximum <= REPLAY_TOLERANCE
            and not bool(frame.labels_read.any())
            and not bool(frame.Current_Exact_used_by_estimator.any())
        )
        elapsed = max(float(summary["elapsed_seconds"]) for summary in summaries)
        summary = {
            "status": f"time_aligned_probe_{args.scope}_{'passed' if passed else 'invalid'}",
            "passed": passed,
            "contract_sha256": contract_hash,
            "scope": args.scope,
            "users": users,
            "edges": list(EDGES),
            "carrier_axis": list(CARRIERS),
            "rows": len(frame),
            "labels_read": False,
            "Current_Exact_used_by_estimator": False,
            "maximum_probe_replay_abs_error": maximum,
            "elapsed_seconds": elapsed,
            "peak_allocated_mib": max(float(row["peak_allocated_mib"]) for row in summaries),
            "peak_reserved_mib": max(float(row["peak_reserved_mib"]) for row in summaries),
            "rough_512_user_seconds": elapsed * 512 / users,
            "rank_summaries": [f"rank{value}/summary.json" for value in range(world)],
        }
        atomic_json(partial / "summary.json", summary)
        partial.rename(output)
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
