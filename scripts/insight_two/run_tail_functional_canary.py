#!/usr/bin/env python3
"""Run the focused dependency-closed Tail-128 to S4 estimator canary."""

from __future__ import annotations

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
from insight_two.functional_probe_estimator import fixed_history_probe_items  # noqa: E402
from insight_two.tail_functional_estimator import (  # noqa: E402
    PROBE_COUNTS,
    estimate_tail_functional_sidecars,
    medium_tail_functional_costs,
)
from reader_compatibility_correction import (  # noqa: E402
    _stage_path,
    intervene_reader_correction,
)


CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_tail_functional_v1.yaml"
)
OUTPUT_ROOT = RESULT_ROOT / "estimator_tail_functional_v1"
USERS = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_contract() -> tuple[dict[str, Any], str]:
    verify_boundary_contract()
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if contract["mechanism"]["probe_counts"] != list(PROBE_COUNTS):
        raise RuntimeError("tail-functional probe axis differs")
    for record in contract["frozen_inputs"].values():
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen tail-functional input differs: {path}")
    actual = {
        int(row["probes"]): (
            int(row["total_generation_flops_per_user"]),
            float(row["over_full_fraction"]),
        )
        for row in medium_tail_functional_costs()
    }
    expected = {
        int(row["probes"]): (
            int(row["total_generation_flops_per_user"]),
            float(row["over_full_fraction"]),
        )
        for row in contract["cost"]["grid"]
    }
    if actual != expected or max(value[1] for value in actual.values()) >= 0.20:
        raise RuntimeError("tail-functional cost grid differs")
    return contract, sha256(CONTRACT)


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        raise RuntimeError("tail-functional canary requires torchrun")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise RuntimeError(f"tail-functional canary requires four ranks, got {world}")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def flatten(correction: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([value.float().reshape(value.shape[0], -1) for value in correction], dim=1)


def correction_metrics(
    estimated: tuple[torch.Tensor, ...], oracle: tuple[torch.Tensor, ...]
) -> dict[str, float]:
    estimate = flatten(estimated)
    target = flatten(oracle)
    target_norm = target.norm(dim=1).clamp_min(1e-20)
    return {
        "correction_cosine_to_oracle": float(
            F.cosine_similarity(estimate, target, dim=1)[0]
        ),
        "correction_norm_ratio_to_oracle": float(
            (estimate.norm(dim=1) / target_norm)[0]
        ),
        "correction_relative_l2_to_oracle": float(
            ((estimate - target).norm(dim=1) / target_norm)[0]
        ),
    }


@torch.inference_mode()
def main() -> None:
    rank, local_rank, world = distributed_context()
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(4)
    started = time.perf_counter()
    verification: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            _, contract_hash = verify_contract()
            verification[0] = {"ok": True, "contract_sha256": contract_hash}
        except BaseException as error:
            verification[0] = {"ok": False, "error": repr(error)}
    dist.broadcast_object_list(verification, src=0)
    assert verification[0] is not None
    if not verification[0]["ok"]:
        raise RuntimeError(f"contract gate failed: {verification[0]['error']}")
    contract_hash = str(verification[0]["contract_sha256"])

    all_uids, all_candidates, _ = load_frozen_inputs()
    selected_indices = np.arange(USERS, dtype=np.int64)
    local_indices = selected_indices[rank::world]
    local_uids = all_uids[local_indices]
    output = OUTPUT_ROOT / "canary"
    partial = output.with_name("canary.partial")
    if rank == 0:
        if output.exists() or partial.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        partial.mkdir(parents=True)
        atomic_json(
            partial / "configuration.json",
            {
                "contract_sha256": contract_hash,
                "users": USERS,
                "edges": list(EDGES),
                "tail_width": 128,
                "probe_axis": list(PROBE_COUNTS),
                "labels_read": False,
                "Current_Exact_in_estimator": False,
                "persistent_Current_KV_positions": 0,
            },
        )
        atomic_json(
            partial / "theoretical_compute.json",
            {"grid": medium_tail_functional_costs()},
        )
    dist.barrier()
    rank_output = partial / f"rank{rank}"
    rank_output.mkdir()

    history = load_histories(
        local_uids.tolist(),
        oov_buckets=OOV_BUCKETS,
        dataset_path=DATASET,
        known_vocab_size=KNOWN_ITEMS,
        end_timestamp=(CUTOVER_DAYS[-1] + 1) * 86_400,
        threads=8,
    )
    cost = {int(row["probes"]): row for row in medium_tail_functional_costs()}
    records: list[dict[str, Any]] = []
    edges = []
    maximum_replay_error = 0.0
    maximum_prefix_error = 0.0

    for edge_index, edge in enumerate(EDGES):
        edge_started = time.perf_counter()
        print(json.dumps({"phase": "edge_start", "rank": rank, "edge": edge}), flush=True)
        parent, parent_payload = load_model(checkpoint(edge_index), device)
        current, current_payload = load_model(checkpoint(edge_index + 1), device)
        verify_model_payload(parent_payload)
        verify_model_payload(current_payload)
        _, items_np, behaviors_np, deltas_np, query_deltas_np = histories_at_cutover(
            history, local_uids, CUTOVER_DAYS[edge_index] * 86_400
        )
        candidates_np = all_candidates[edge_index, local_indices]

        for offset, uid in enumerate(local_uids):
            items = torch.as_tensor(
                items_np[offset : offset + 1], dtype=torch.long, device=device
            )
            behaviors = torch.as_tensor(
                behaviors_np[offset : offset + 1], dtype=torch.long, device=device
            )
            deltas = torch.as_tensor(
                deltas_np[offset : offset + 1], dtype=torch.float32, device=device
            )
            query_delta = torch.as_tensor(
                query_deltas_np[offset : offset + 1],
                dtype=torch.float32,
                device=device,
            )
            parent_cache = parent.compute_kv(items, behaviors, deltas)
            estimate = estimate_tail_functional_sidecars(
                current,
                parent_cache,
                items,
                behaviors,
                deltas,
                fixed_history_probe_items(items),
                query_delta,
            )
            maximum_replay_error = max(
                maximum_replay_error, estimate.single_probe_replay_max_abs_error
            )
            maximum_prefix_error = max(
                maximum_prefix_error, estimate.parent_prefix_max_abs_change
            )

            # Current Exact is constructed only after the estimator has returned.
            current_cache = current.compute_kv(items, behaviors, deltas)
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
            for probes in PROBE_COUNTS:
                correction = estimate.corrections_by_probe_count[probes]
                scores, _ = intervene_reader_correction(
                    current,
                    parent_cache,
                    heldout,
                    query_delta,
                    stage="av_aggregation",
                    corrections=correction,
                )
                records.append(
                    {
                        "edge": edge,
                        "uid": int(uid),
                        "user_index": int(local_indices[offset]),
                        "probes": probes,
                        "theoretical_compute_fraction": float(
                            cost[probes]["over_full_fraction"]
                        ),
                        "sidecar_scalars_fp32": int(cost[probes]["sidecar_scalars"]),
                        "labels_read": False,
                        "Current_Exact_in_estimator": False,
                        **metrics_row(score_metrics(exact_scores, reuse_scores, scores)),
                        **correction_metrics(correction, oracle),
                    }
                )
        edges.append(
            {
                "edge": edge,
                "users": len(local_uids),
                "seconds": time.perf_counter() - edge_started,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "edge_complete",
                    "rank": rank,
                    "edge": edge,
                    "users": len(local_uids),
                    "seconds": edges[-1]["seconds"],
                }
            ),
            flush=True,
        )
        del parent, current
        torch.cuda.empty_cache()

    frame = pd.DataFrame(records)
    frame.to_parquet(rank_output / "score_records.parquet", index=False)
    atomic_json(
        rank_output / "summary.json",
        {
            "rank": rank,
            "rows": len(frame),
            "edges": edges,
            "maximum_single_probe_replay_error": maximum_replay_error,
            "maximum_Parent_prefix_or_cache_change": maximum_prefix_error,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        },
    )
    dist.barrier()

    if rank == 0:
        rank_summaries = [
            json.loads((partial / f"rank{shard}/summary.json").read_text(encoding="utf-8"))
            for shard in range(world)
        ]
        combined = pd.concat(
            [
                pd.read_parquet(partial / f"rank{shard}/score_records.parquet")
                for shard in range(world)
            ],
            ignore_index=True,
        ).sort_values(["edge", "user_index", "probes"])
        raw = partial / "score_records.parquet"
        combined.to_parquet(raw, index=False)
        replay_error = max(row["maximum_single_probe_replay_error"] for row in rank_summaries)
        prefix_error = max(row["maximum_Parent_prefix_or_cache_change"] for row in rank_summaries)
        finite = bool(
            np.isfinite(combined.select_dtypes(include=[np.number]).to_numpy()).all()
        )
        passed = bool(
            len(combined) == USERS * len(EDGES) * len(PROBE_COUNTS)
            and replay_error <= 2e-5
            and prefix_error == 0.0
            and finite
        )
        summary = {
            "status": "tail_functional_instrumentation_passed" if passed else "tail_functional_instrumentation_failed",
            "passed": passed,
            "contract_sha256": contract_hash,
            "users": USERS,
            "edges": list(EDGES),
            "rows": len(combined),
            "labels_read": False,
            "Current_Exact_in_estimator": False,
            "maximum_single_probe_replay_error": replay_error,
            "maximum_Parent_prefix_or_cache_change": prefix_error,
            "all_numeric_rows_finite": finite,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_mib": max(row["peak_allocated_mib"] for row in rank_summaries),
            "peak_reserved_mib": max(row["peak_reserved_mib"] for row in rank_summaries),
            "raw": {
                "path": raw.name,
                "rows": len(combined),
                "bytes": raw.stat().st_size,
                "sha256": sha256(raw),
            },
        }
        atomic_json(partial / "summary.json", summary)
        partial.replace(output)
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
