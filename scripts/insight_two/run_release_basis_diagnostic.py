#!/usr/bin/env python3
"""Measure release-wide S4 directions and fixed-history-probe Exact ceiling."""

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
from insight_two.functional_probe_estimator import fixed_history_probe_items  # noqa: E402
from insight_two.release_functional_basis import (  # noqa: E402
    fit_oracle_release_basis,
    rank_at_energy,
)
from reader_compatibility_correction import _stage_path, intervene_reader_correction  # noqa: E402


CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_release_basis_v1.yaml"
)
OUTPUT_ROOT = RESULT_ROOT / "diagnostic_release_basis_v1"
RANKS = (0, 1, 2, 4, 8, 16, 32)
CANARY_USERS = 32
CANARY_BASIS_USERS = 8
DISCOVERY_USERS = 512
DISCOVERY_BASIS_USERS = 64


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_contract() -> tuple[dict[str, Any], str]:
    verify_boundary_contract()
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if contract["release_basis"]["rank_axis"] != list(RANKS):
        raise RuntimeError("release-basis rank axis differs")
    for record in contract["frozen_inputs"].values():
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen release-basis input differs: {path}")
    return contract, sha256_file(CONTRACT)


def require_discovery_gate(contract_hash: str) -> None:
    canary = OUTPUT_ROOT / "canary_v2/summary.json"
    resource = OUTPUT_ROOT / "resource_estimate.json"
    if not canary.is_file() or not resource.is_file():
        raise RuntimeError("release-basis discovery requires canary and resource estimate")
    run = json.loads(canary.read_text(encoding="utf-8"))
    estimate = json.loads(resource.read_text(encoding="utf-8"))
    if not run.get("passed") or run.get("contract_sha256") != contract_hash:
        raise RuntimeError("release-basis canary did not pass")
    if estimate.get("contract_sha256") != contract_hash:
        raise RuntimeError("release-basis resource estimate differs")
    if estimate.get("estimated_512_user_minutes", 1e9) > 30:
        raise RuntimeError("release-basis discovery exceeds 30 minutes")


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        raise RuntimeError("release-basis diagnostic requires torchrun")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise RuntimeError(f"release-basis diagnostic requires four ranks, got {world}")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def correction_array(values: tuple[torch.Tensor, ...]) -> np.ndarray:
    return np.stack(
        [value.float().reshape(value.shape[0], -1)[0].detach().cpu().numpy() for value in values]
    ).astype(np.float32)


def correction_tuple(
    values: np.ndarray, *, heads: int, head_dim: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    if values.ndim != 2 or values.shape[1] != heads * head_dim:
        raise ValueError("projected S4 correction has the wrong shape")
    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    return tuple(tensor[layer].reshape(1, heads, head_dim) for layer in range(len(values)))


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("canary", "discovery"), default="canary")
    args = parser.parse_args()
    users = CANARY_USERS if args.scope == "canary" else DISCOVERY_USERS
    basis_users = CANARY_BASIS_USERS if args.scope == "canary" else DISCOVERY_BASIS_USERS
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
    output = OUTPUT_ROOT / ("canary_v2" if args.scope == "canary" else "discovery")
    partial = output.with_name(output.name + ".partial")
    if rank == 0:
        if output.exists() or partial.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        partial.mkdir(parents=True)
        (partial / "targets").mkdir()
        atomic_json(
            partial / "configuration.json",
            {
                "contract_sha256": contract_hash,
                "scope": args.scope,
                "users": users,
                "basis_indices_half_open": [0, basis_users],
                "evaluation_indices_half_open": [basis_users, users],
                "rank_axis": list(RANKS),
                "history_probes": 8,
                "labels_read": False,
                "oracle_diagnostic_only": True,
            },
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
    score_records: list[dict[str, Any]] = []
    structure_records: list[dict[str, Any]] = []
    edge_records = []
    peak_allocated = 0.0
    peak_reserved = 0.0

    for edge_index, edge in enumerate(EDGES):
        print(json.dumps({"phase": "edge_start", "rank": rank, "edge": edge}), flush=True)
        edge_started = time.perf_counter()
        parent, parent_payload = load_model(checkpoint(edge_index), device)
        current, current_payload = load_model(checkpoint(edge_index + 1), device)
        verify_model_payload(parent_payload)
        verify_model_payload(current_payload)
        _, items_np, actions_np, deltas_np, query_deltas_np = histories_at_cutover(
            history, local_uids, CUTOVER_DAYS[edge_index] * 86_400
        )
        candidates_np = all_candidates[edge_index, local_indices]
        local_records = []
        target_payload = []

        for offset, (global_index, uid) in enumerate(zip(local_indices, local_uids, strict=True)):
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
            _, anchor_correction, _ = _stage_path(
                current,
                current_cache,
                parent_cache,
                anchors,
                query_delta,
                stage="av_aggregation",
                mode="shared",
            )
            history_probes = fixed_history_probe_items(items)
            _, history_correction, _ = _stage_path(
                current,
                current_cache,
                parent_cache,
                history_probes,
                query_delta,
                stage="av_aggregation",
                mode="shared",
            )
            history_target = correction_array(history_correction)
            anchor_target = correction_array(anchor_correction)
            target_payload.append(
                {
                    "index": int(global_index),
                    "uid": int(uid),
                    "history_target": history_target,
                    "anchor_target": anchor_target,
                }
            )
            local_records.append(
                {
                    "index": int(global_index),
                    "uid": int(uid),
                    "parent_cache": parent_cache,
                    "heldout": heldout,
                    "query_delta": query_delta,
                    "exact_scores": exact_scores,
                    "reuse_scores": reuse_scores,
                    "history_target": history_target,
                    "anchor_target": anchor_target,
                }
            )
            del items, actions, deltas, anchors, current_cache

        gathered: list[list[dict[str, Any]] | None] = [None] * world
        dist.all_gather_object(gathered, target_payload)
        all_targets = sorted(
            [record for shard in gathered for record in (shard or [])],
            key=lambda record: int(record["index"]),
        )
        if len(all_targets) != users or [row["index"] for row in all_targets] != list(range(users)):
            raise RuntimeError("release target panel is incomplete")
        history_matrix = np.stack([row["history_target"] for row in all_targets])
        anchor_matrix = np.stack([row["anchor_target"] for row in all_targets])
        basis = fit_oracle_release_basis(history_matrix[:basis_users])
        for layer, singular in enumerate(basis.layer_singular_values):
            energy = np.square(singular)
            total = max(float(energy.sum()), 1e-20)
            structure_records.append(
                {
                    "edge": edge,
                    "layer": layer,
                    "basis_users": basis_users,
                    "rank90": rank_at_energy(singular, 0.90),
                    "rank95": rank_at_energy(singular, 0.95),
                    "rank1_energy": float(energy[:1].sum() / total),
                    "rank2_energy": float(energy[:2].sum() / total),
                    "rank4_energy": float(energy[:4].sum() / total),
                }
            )
        if rank == 0:
            np.savez_compressed(
                partial / f"targets/{edge}.npz",
                uids=np.asarray([row["uid"] for row in all_targets], dtype=np.int64),
                history_targets=history_matrix,
                anchor_targets=anchor_matrix,
            )

        for record in local_records:
            if record["index"] < basis_users:
                continue
            history_values = record["history_target"]
            anchor_values = record["anchor_target"]
            history_flat = torch.as_tensor(history_values.reshape(1, -1), device=device)
            anchor_flat = torch.as_tensor(anchor_values.reshape(1, -1), device=device)
            anchor_norm = anchor_flat.norm(dim=1).clamp_min(1e-20)
            direct = correction_tuple(
                history_values,
                heads=current.cfg.num_heads,
                head_dim=current.blocks[0].attn.head_dim,
                device=device,
                dtype=record["parent_cache"].k.dtype,
            )
            direct_scores, _ = intervene_reader_correction(
                current,
                record["parent_cache"],
                record["heldout"],
                record["query_delta"],
                stage="av_aggregation",
                corrections=direct,
            )
            score_records.append(
                {
                    "edge": edge,
                    "uid": record["uid"],
                    "method": "exact_fixed_history_probe_mean",
                    "rank": -1,
                    "basis_users": basis_users,
                    "oracle_evaluation_coefficients": False,
                    "Current_Exact_used": True,
                    "labels_read": False,
                    **metrics_row(
                        score_metrics(
                            record["exact_scores"], record["reuse_scores"], direct_scores
                        )
                    ),
                    "correction_cosine_to_anchor_target": float(
                        F.cosine_similarity(history_flat, anchor_flat, dim=1)[0]
                    ),
                    "correction_norm_ratio_to_anchor_target": float(
                        (history_flat.norm(dim=1) / anchor_norm)[0]
                    ),
                    "correction_relative_l2_to_anchor_target": float(
                        ((history_flat - anchor_flat).norm(dim=1) / anchor_norm)[0]
                    ),
                }
            )
            for low_rank in RANKS:
                projected = basis.project(history_values[None], low_rank)[0]
                correction = correction_tuple(
                    projected,
                    heads=current.cfg.num_heads,
                    head_dim=current.blocks[0].attn.head_dim,
                    device=device,
                    dtype=record["parent_cache"].k.dtype,
                )
                observed, _ = intervene_reader_correction(
                    current,
                    record["parent_cache"],
                    record["heldout"],
                    record["query_delta"],
                    stage="av_aggregation",
                    corrections=correction,
                )
                projected_tensor = torch.as_tensor(projected.reshape(1, -1), device=device)
                score_records.append(
                    {
                        "edge": edge,
                        "uid": record["uid"],
                        "method": "oracle_release_basis",
                        "rank": low_rank,
                        "basis_users": basis_users,
                        "oracle_evaluation_coefficients": low_rank > 0,
                        "Current_Exact_used": True,
                        "labels_read": False,
                        **metrics_row(
                            score_metrics(
                                record["exact_scores"], record["reuse_scores"], observed
                            )
                        ),
                        "correction_cosine_to_anchor_target": float(
                            F.cosine_similarity(projected_tensor, anchor_flat, dim=1)[0]
                        ),
                        "correction_norm_ratio_to_anchor_target": float(
                            (projected_tensor.norm(dim=1) / anchor_norm)[0]
                        ),
                        "correction_relative_l2_to_anchor_target": float(
                            ((projected_tensor - anchor_flat).norm(dim=1) / anchor_norm)[0]
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
            {
                "edge": edge,
                "basis_users": basis_users,
                "evaluation_users": users - basis_users,
                "seconds": time.perf_counter() - edge_started,
            }
        )
        print(json.dumps({"phase": "edge_complete", "rank": rank, **edge_records[-1]}), flush=True)
        del parent, current, local_records, basis
        torch.cuda.empty_cache()

    score_path = rank_output / "score_records.parquet"
    structure_path = rank_output / "structure_records.parquet"
    pd.DataFrame(score_records).to_parquet(score_path, index=False)
    pd.DataFrame(structure_records).to_parquet(structure_path, index=False)
    rank_summary = {
        "status": "release_basis_rank_complete",
        "rank": rank,
        "uids": local_uids.tolist(),
        "labels_read": False,
        "oracle_diagnostic_only": True,
        "elapsed_seconds": time.perf_counter() - started,
        "history_seconds": history_seconds,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "edges": edge_records,
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (score_path, structure_path)
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
        uids = [uid for summary in summaries for uid in summary["uids"]]
        expected_rows = (users - basis_users) * len(EDGES) * (1 + len(RANKS))
        passed = (
            len(uids) == len(set(uids)) == users
            and len(scores) == expected_rows
            and bool(np.isfinite(scores.select_dtypes(include=[np.number])).all().all())
            and not bool(scores.labels_read.any())
        )
        elapsed = max(float(summary["elapsed_seconds"]) for summary in summaries)
        target_artifacts = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted((partial / "targets").glob("*.npz"))
        }
        final = {
            "status": f"release_basis_{args.scope}_{'passed' if passed else 'invalid'}",
            "passed": bool(passed),
            "contract_sha256": contract_hash,
            "scope": args.scope,
            "users": users,
            "basis_users": basis_users,
            "evaluation_users": users - basis_users,
            "edges": list(EDGES),
            "rank_axis": list(RANKS),
            "rows": len(scores),
            "labels_read": False,
            "oracle_diagnostic_only": True,
            "elapsed_seconds": elapsed,
            "rough_512_user_seconds": elapsed * 512 / users,
            "peak_allocated_mib": max(float(row["peak_allocated_mib"]) for row in summaries),
            "peak_reserved_mib": max(float(row["peak_reserved_mib"]) for row in summaries),
            "target_artifacts": target_artifacts,
            "rank_summaries": [f"rank{value}/summary.json" for value in range(world)],
        }
        atomic_json(partial / "summary.json", final)
        partial.rename(output)
        print(json.dumps(final, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
