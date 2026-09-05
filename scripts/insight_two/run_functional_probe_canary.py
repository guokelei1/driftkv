#!/usr/bin/env python3
"""Run the executable S4 functional-probe estimator on Medium checkpoints."""

from __future__ import annotations

import argparse
import hashlib
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
from hstu_kvcache.models import HSTUKVCache  # noqa: E402
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
from insight_two.functional_probe_estimator import (  # noqa: E402
    PROBE_COUNTS,
    estimate_functional_probe_means,
    fixed_history_probe_items,
    medium_cost_grid,
)
from one_release_refinement import parameter_cast_maps  # noqa: E402
from pro_lazy_reader import build_parent_conditioned_carriers  # noqa: E402
from reader_compatibility_correction import (  # noqa: E402
    _stage_path,
    intervene_reader_correction,
)


CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_functional_probe_v1.yaml"
)
OUTPUT_ROOT = RESULT_ROOT / "estimator_functional_probe_v1"
CARRIERS = (8, 16, 32, 64)
CANARY_USERS = 32
DISCOVERY_USERS = 512
REPLAY_TOLERANCE = 2e-5


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def repeat_cache(cache: HSTUKVCache, copies: int) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k.repeat_interleave(copies, dim=1),
        v=cache.v.repeat_interleave(copies, dim=1),
        seq_len=cache.seq_len,
    )


def flatten_correction(correction: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([value.float().flatten(1) for value in correction], dim=1)


def correction_metrics(
    estimated: tuple[torch.Tensor, ...], oracle: tuple[torch.Tensor, ...]
) -> dict[str, float]:
    estimate = flatten_correction(estimated)
    target = flatten_correction(oracle)
    target_norm = target.norm(dim=1).clamp_min(1e-20)
    return {
        "correction_cosine_to_oracle": float(F.cosine_similarity(estimate, target, dim=1)[0]),
        "correction_norm_ratio_to_oracle": float((estimate.norm(dim=1) / target_norm)[0]),
        "correction_relative_l2_to_oracle": float(
            ((estimate - target).norm(dim=1) / target_norm)[0]
        ),
    }


def verify_estimator_contract() -> tuple[dict[str, Any], str]:
    verify_boundary_contract()
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if contract["mechanism"]["carrier_axis"] != list(CARRIERS):
        raise RuntimeError("contract carrier axis differs")
    if contract["mechanism"]["probe_axis"] != list(PROBE_COUNTS):
        raise RuntimeError("contract probe axis differs")
    for record in contract["frozen_inputs"].values():
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen estimator input differs: {path}")
    observed_cost = {
        (int(row["carriers"]), int(row["probes"])): float(row["over_full_fraction"])
        for row in medium_cost_grid()
    }
    expected_cost = {
        (int(row["carriers"]), int(row["probes"])): float(row["over_full_fraction"])
        for row in contract["cost"]["grid"]
    }
    if observed_cost != expected_cost or max(observed_cost.values()) > 0.20:
        raise RuntimeError("estimator cost grid differs from contract")
    return contract, sha256_file(CONTRACT)


def require_discovery_gate(contract_hash: str) -> None:
    path = OUTPUT_ROOT / "canary/summary.json"
    estimate = OUTPUT_ROOT / "resource_estimate.json"
    if not path.is_file() or not estimate.is_file():
        raise RuntimeError("estimator discovery requires canary and resource estimate")
    summary = json.loads(path.read_text(encoding="utf-8"))
    resource = json.loads(estimate.read_text(encoding="utf-8"))
    if not summary.get("passed") or summary.get("contract_sha256") != contract_hash:
        raise RuntimeError("estimator canary gate did not pass")
    if resource.get("contract_sha256") != contract_hash:
        raise RuntimeError("estimator resource estimate differs")
    if resource.get("estimated_512_user_minutes", 1e9) > 30:
        raise RuntimeError("interactive discovery exceeds the frozen 30-minute limit")


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        raise RuntimeError("functional-probe run requires torchrun")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise RuntimeError(f"functional-probe run requires four ranks, got {world}")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


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
            _, contract_hash = verify_estimator_contract()
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
                "probe_axis": list(PROBE_COUNTS),
                "stage": "S4_aggregated_context",
                "estimator_inputs": [
                    "Parent_persistent_KV",
                    "strictly_pre_cutover_raw_history",
                    "Parent_and_Current_parameters",
                ],
                "estimator_forbidden_inputs": [
                    "Current_Exact_KV",
                    "request_candidates",
                    "request_or_future_labels",
                ],
                "labels_read": False,
            },
        )
        atomic_json(partial / "theoretical_compute.json", {"grid": medium_cost_grid()})
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
    cost_by_config = {
        (int(row["carriers"]), int(row["probes"])): row for row in medium_cost_grid()
    }
    score_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    map_seconds: dict[str, float] = {}
    maximum_replay_error = 0.0
    peak_allocated_mib = 0.0
    peak_reserved_mib = 0.0

    for edge_index, edge in enumerate(EDGES):
        print(json.dumps({"phase": "edge_start", "rank": rank, "edge": edge}), flush=True)
        edge_started = time.perf_counter()
        parent, parent_payload = load_model(checkpoint(edge_index), device)
        current, current_payload = load_model(checkpoint(edge_index + 1), device)
        verify_model_payload(parent_payload)
        verify_model_payload(current_payload)
        map_started = time.perf_counter()
        joint_maps = parameter_cast_maps(parent, current)
        torch.cuda.synchronize(device)
        map_seconds[edge] = time.perf_counter() - map_started
        _, items_np, actions_np, deltas_np, query_deltas_np = histories_at_cutover(
            history, local_uids, CUTOVER_DAYS[edge_index] * 86_400
        )
        candidates_np = all_candidates[edge_index, local_indices]
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
                query_deltas_np[local_offset : local_offset + 1],
                dtype=torch.float32,
                device=device,
            )
            anchors = torch.as_tensor(
                candidates_np[local_offset : local_offset + 1, ANCHOR_INDICES],
                dtype=torch.long,
                device=device,
            )
            heldout = torch.as_tensor(
                candidates_np[local_offset : local_offset + 1, HELDOUT_INDICES],
                dtype=torch.long,
                device=device,
            )
            parent_cache = parent.compute_kv(items, actions, deltas)
            current_cache = current.compute_kv(items, actions, deltas)
            exact_scores = current.score_cc_reuse(current_cache, heldout, query_delta)
            reuse_scores = current.score_cc_reuse(parent_cache, heldout, query_delta)
            # Current Exact is evaluation-only. It never enters the estimator API.
            _, oracle_correction, _ = _stage_path(
                current,
                current_cache,
                parent_cache,
                anchors,
                query_delta,
                stage="av_aggregation",
                mode="shared",
            )
            probe_items = fixed_history_probe_items(items)

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
                estimate = estimate_functional_probe_means(
                    current,
                    parent_cache,
                    carriers,
                    joint_maps,
                    probe_items,
                    old_positions=layout.old_positions,
                )
                maximum_replay_error = max(
                    maximum_replay_error, estimate.replay_max_abs_error
                )
                for probe_count in PROBE_COUNTS:
                    correction = estimate.corrections_by_probe_count[probe_count]
                    observed_scores, _ = intervene_reader_correction(
                        current,
                        parent_cache,
                        heldout,
                        query_delta,
                        stage="av_aggregation",
                        corrections=correction,
                    )
                    cost = cost_by_config[(carrier_count, probe_count)]
                    score_records.append(
                        {
                            "edge": edge,
                            "uid": int(uid),
                            "carriers": carrier_count,
                            "probes": probe_count,
                            "theoretical_compute_fraction": float(
                                cost["over_full_fraction"]
                            ),
                            "sidecar_values_fp32_per_user": int(
                                cost["sidecar_write_scalars"]
                            ),
                            "labels_read": False,
                            "current_exact_used_by_estimator": False,
                            **metrics_row(
                                score_metrics(exact_scores, reuse_scores, observed_scores)
                            ),
                            **correction_metrics(correction, oracle_correction),
                        }
                    )
                del carriers, estimate
            peak_allocated_mib = max(
                peak_allocated_mib, torch.cuda.max_memory_allocated(device) / (1 << 20)
            )
            peak_reserved_mib = max(
                peak_reserved_mib, torch.cuda.max_memory_reserved(device) / (1 << 20)
            )
            del (
                items,
                actions,
                deltas,
                query_delta,
                anchors,
                heldout,
                parent_cache,
                current_cache,
                exact_scores,
                reuse_scores,
                oracle_correction,
                probe_items,
            )

        edge_seconds = time.perf_counter() - edge_started
        edge_records.append({"edge": edge, "users": len(local_uids), "seconds": edge_seconds})
        print(
            json.dumps(
                {"phase": "edge_complete", "rank": rank, "edge": edge, "seconds": edge_seconds}
            ),
            flush=True,
        )
        del parent, current, parent_payload, current_payload, joint_maps
        torch.cuda.empty_cache()

    score_path = rank_output / "score_records.parquet"
    pd.DataFrame(score_records).to_parquet(score_path, index=False)
    rank_summary = {
        "status": "functional_probe_rank_complete",
        "rank": rank,
        "uids": local_uids.tolist(),
        "labels_read": False,
        "history_seconds": history_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "map_seconds": map_seconds,
        "maximum_probe_replay_abs_error": maximum_replay_error,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "edges": edge_records,
        "artifact": {
            "path": score_path.name,
            "sha256": sha256_file(score_path),
            "bytes": score_path.stat().st_size,
        },
    }
    atomic_json(rank_output / "summary.json", rank_summary)
    dist.barrier()

    if rank == 0:
        summaries = [
            json.loads((partial / f"rank{value}/summary.json").read_text(encoding="utf-8"))
            for value in range(world)
        ]
        scores = pd.concat(
            [pd.read_parquet(partial / f"rank{value}/score_records.parquet") for value in range(4)],
            ignore_index=True,
        )
        observed_uids = [uid for summary in summaries for uid in summary["uids"]]
        expected_rows = users * len(EDGES) * len(CARRIERS) * len(PROBE_COUNTS)
        maximum_replay = max(
            float(summary["maximum_probe_replay_abs_error"]) for summary in summaries
        )
        passed = (
            len(observed_uids) == users
            and len(set(observed_uids)) == users
            and len(scores) == expected_rows
            and bool(np.isfinite(scores.select_dtypes(include=[np.number])).all().all())
            and maximum_replay <= REPLAY_TOLERANCE
            and not bool(scores.labels_read.any())
            and not bool(scores.current_exact_used_by_estimator.any())
        )
        elapsed = max(float(summary["elapsed_seconds"]) for summary in summaries)
        final_summary = {
            "status": f"functional_probe_{args.scope}_{'passed' if passed else 'invalid'}",
            "passed": passed,
            "contract_sha256": contract_hash,
            "scope": args.scope,
            "users": users,
            "edges": list(EDGES),
            "carrier_axis": list(CARRIERS),
            "probe_axis": list(PROBE_COUNTS),
            "rows": len(scores),
            "labels_read": False,
            "current_exact_used_by_estimator": False,
            "maximum_probe_replay_abs_error": maximum_replay,
            "elapsed_seconds": elapsed,
            "peak_allocated_mib": max(float(row["peak_allocated_mib"]) for row in summaries),
            "peak_reserved_mib": max(float(row["peak_reserved_mib"]) for row in summaries),
            "rough_512_user_seconds": elapsed * DISCOVERY_USERS / users,
            "rank_summaries": [f"rank{value}/summary.json" for value in range(world)],
        }
        atomic_json(partial / "summary.json", final_summary)
        partial.rename(output)
        print(json.dumps(final_summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
